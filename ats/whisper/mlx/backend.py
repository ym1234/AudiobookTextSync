import os
from dataclasses import dataclass
import numpy as np
import mlx.core as mx
from tqdm.auto import tqdm
from ats.np import np as cnp, no_cuda
from ats.whisper.tokenizer import Tokenizer
from ats.whisper.mlx.model import load_model
from ats.whisper.mlx import search
from ats.whisper.models import CT2_MODELS, MLX_MODELS, download

@dataclass
class GenerationResult:
    sequences_ids: list
    scores: list
    no_speech_prob: float

class MlxBackend:
    def __init__(self, model, device='auto', device_index=0, quantize=True, download_root=None, local_files_only=False):
        if device == 'cpu':
            mx.set_default_device(mx.cpu)
        elif device in ('auto', 'gpu', 'metal', 'cuda'):
            mx.set_default_device(mx.gpu)
        else:
            raise ValueError(f"unknown device {device!r}, mlx only has 'cpu' and 'gpu'")
        if device_index: tqdm.write("--device-index does nothing with mlx, ignoring it")

        if model in CT2_MODELS and model not in MLX_MODELS:
            raise ValueError(f"there's no mlx conversion of {model}, use --backend ct2 or one of {', '.join(MLX_MODELS)}")
        self.model_name = model
        self.dtype = mx.float16 if quantize else mx.float32
        model_path = model if os.path.isdir(model) else download(MLX_MODELS.get(model, model), download_root, local_files_only)
        self.model = load_model(model_path, self.dtype)
        self.tokenizer = Tokenizer(self._tokenizer_path(model, model_path, download_root, local_files_only))

    def _tokenizer_path(self, model, model_path, download_root, local_files_only):
        # the mlx conversions only ship the weights, so grab a tokenizer.json from the faster-whisper repo instead
        if os.path.exists(os.path.join(model_path, 'tokenizer.json')): return model_path
        repos = [CT2_MODELS[model]] if model in CT2_MODELS else ([] if os.path.isdir(model) else [model])
        n_vocab = self.model.dims.n_vocab
        repos.append('Systran/faster-whisper-large-v3' if n_vocab > 51865 else
                     'Systran/faster-whisper-large-v2' if n_vocab == 51865 else 'Systran/faster-whisper-tiny.en')
        for repo in repos:
            try: return download(repo, download_root, local_files_only, allow_patterns=['tokenizer.json'])
            except Exception as e: tqdm.write(f"couldn't get a tokenizer from {repo}: {e}")
        raise RuntimeError(f"couldn't find a tokenizer for {model}")

    @property
    def device(self): return str(mx.default_device())
    @property
    def compute_type(self): return str(self.dtype).rsplit('.', 1)[-1]
    @property
    def n_mels(self): return self.model.dims.n_mels
    @property
    def is_multilingual(self): return self.model.is_multilingual

    def encode(self, features):
        if not no_cuda: features = cnp.asnumpy(features)
        features = mx.array(np.ascontiguousarray(features), self.dtype)
        encoded = self.model.encoder(features.transpose(0, 2, 1)) # mlx convolutions are channels last
        mx.eval(encoded)
        return encoded

    def detect_language(self, encoded):
        encoded = mx.stack(encoded)
        tokens = mx.full((encoded.shape[0], 1), self.tokenizer.sot, dtype=mx.int32)
        logits = np.array(self.model.logits(tokens, encoded)[:, 0].astype(mx.float32))
        mask = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
        mask[self.tokenizer.language_tokens] = 0
        return (logits + mask).argmax(-1).tolist()

    def _suppressed(self, suppress_tokens):
        tok = self.tokenizer
        if isinstance(suppress_tokens, str):
            suppress_tokens = [t for t in suppress_tokens.split(',') if t.strip()]
        out = set()
        for t in (suppress_tokens or []):
            t = int(t)
            if t == -1: out.update(tok.non_speech_tokens)
            elif t >= 0: out.add(t)
        out.update(t for t in [tok.sot, tok.transcribe, tok.translate, tok.no_timestamps, tok.no_speech,
                               tok.token_to_id("<|startofprev|>"), tok.token_to_id("<|startoflm|>")] if t is not None)
        return np.array(sorted(out), dtype=np.int64)

    def generate(self, encoded, prompts, temperatures, num_hypotheses, sampling_topk=0,
                 suppress_blank=True, suppress_tokens=(-1,), max_initial_timestamp_index=None,
                 repetition_penalty=1, no_repeat_ngram_size=0):
        """temperatures is one per window, greedy and sampled windows share a call"""
        if int(no_repeat_ngram_size) > 1:
            raise ValueError("--no-repeat-ngram-size isn't supported by the mlx backend")
        temperatures = np.asarray(temperatures, dtype=np.float32)

        tok, dims = self.tokenizer, self.model.dims
        groups = len(prompts)
        blank = [tok.encode(" ")[0], tok.eot] if suppress_blank else None
        rules = search.LogitRules(tok, dims.n_vocab, self._suppressed(suppress_tokens), blank,
                                  max_initial_timestamp_index, repetition_penalty)
        # a greedy window needs one row, a sampled one its best-of-n
        counts = np.where(temperatures == 0, 1, max(1, num_hypotheses))
        rows = np.repeat(np.arange(groups), counts)

        prompt = mx.array(np.array(prompts, dtype=np.int32)[rows])
        encoded = mx.stack(encoded)
        # one encoding per row, a sampled window's hypotheses each get a copy of its audio
        xa = encoded if (counts == 1).all() else mx.take(encoded, mx.array(rows), axis=0)
        logits, kv_cache, _ = self.model.decoder(prompt, xa)
        mx.eval(logits, kv_cache.state)

        no_speech = [0.] * groups
        if tok.no_speech is not None: # the <|nospeech|> probability is the one predicted at the <|startoftranscript|>
            firsts = mx.array(np.cumsum(counts) - counts) # the first row of each window, they all share an encoding
            probs = mx.softmax(mx.take(logits, firsts, axis=0)[:, 0].astype(mx.float32), axis=-1)
            no_speech = np.array(probs[:, tok.no_speech]).tolist()

        seqs, sum_logprobs = search.sample(self.model.decoder, rules, logits, kv_cache, xa, np.repeat(temperatures, counts),
                                           sampling_topk, dims.n_text_ctx // 2)
        hypotheses, start = [], 0
        for c in counts:
            h = sorted(((seqs[start+j].tolist(), float(sum_logprobs[start+j])) for j in range(c)), key=lambda x: x[1], reverse=True)
            hypotheses.append([(seq[:seq.index(tok.eot)] if tok.eot in seq else seq, score) for seq, score in h])
            start += c

        return [GenerationResult(sequences_ids=[seq for seq, _ in h], scores=[score for _, score in h],
                                 no_speech_prob=no_speech[i]) for i, h in enumerate(hypotheses)]
