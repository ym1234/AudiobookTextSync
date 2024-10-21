import os
import multiprocessing
import huggingface_hub
import tokenizers
import time
import numpy as np
from functools import cached_property
from pprint import pprint, pformat
from ats.text import SubLine
from tqdm import tqdm

# Stupid hack because python doesn't have lazy imports (torch)
def _import_c2():
    import sys
    import importlib

    finder = importlib.machinery.PathFinder()
    c2spec = finder.find_spec('ctranslate2')

    if sys.platform == "win32":
        import ctypes
        from importlib import resources

        c2resources = resources.files(importlib.util.module_from_spec(c2spec))

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            add_dll_directory(str(c2resources)) # Need str here?

        for library in c2resources.glob('*.dll'):
            ctypes.CDLL(library)

    extspec = finder.find_spec('_ext', c2spec.submodule_search_locations)
    c2ext = importlib.util.module_from_spec(extspec)
    try:
        extspec.loader.exec_module(c2ext) # idk what this does, seems to work without it
    except:
        pass
    return c2ext.Whisper, c2ext.WhisperGenerationResult, c2ext.get_cuda_device_count, c2ext.get_supported_compute_types, c2ext.StorageView
Whisper, WhisperGenerationResult, get_cuda_device_count, get_supported_compute_types, StorageView, = _import_c2()
# import ctranslate2
# Whisper, WhisperGenerationResult, get_cuda_device_count, get_supported_compute_types, StorageView = ctranslate2.models.Whisper, ctranslate2.models.WhisperGenerationResult, ctranslate2.get_cuda_device_count, ctranslate2.get_supported_compute_types, ctranslate2.StorageView

_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

def available_models(): return list(_MODELS.keys())
def download_model(model_name, local_dir=None, local_files_only=False):
    repo_id = _MODELS.get(model_name, model_name)
    return huggingface_hub.snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False,
                                             local_files_only=local_files_only, max_workers=multiprocessing.cpu_count())

class Tokenizer:
    def __init__(self, path):
        self.tokenizer = tokenizers.Tokenizer.from_file(path=os.path.join(path, "tokenizer.json"))

    @cached_property
    def transcribe(self) -> int: return self.tokenizer.token_to_id("<|transcribe|>")
    @cached_property
    def sot(self) -> int: return self.tokenizer.token_to_id("<|startoftranscript|>")
    @cached_property
    def eot(self) -> int: return self.tokenizer.token_to_id("<|endoftext|>")
    @cached_property
    def no_timestamps(self) -> int: return self.tokenizer.token_to_id("<|notimestamps|>")
    @property
    def timestamp_begin(self) -> int: return self.no_timestamps + 1

    def token_to_id(self, token): return self.tokenizer.token_to_id(token)
    def encode(self, text: str): return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens): return self.tokenizer.decode(tokens)
    def decode_with_timestamps(self, tokens) -> str:
        outputs = []

        start, end = 0, 1
        while start < len(tokens):
            while end < len(tokens) and tokens[end] < self.timestamp_begin:
                end += 1
            outputs.append(tokens[start:end+1])
            start, end = end+1, end+2
        return outputs

    def split_tokens_on_unicode(self, tokens):
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            try:
                replacement_char_index = decoded.index(replacement_char)
                replacement_char_index += unicode_offset
            except ValueError:
                replacement_char_index = None

            if replacement_char_index is None or (
                replacement_char_index < len(decoded_full)
                and decoded_full[replacement_char_index] == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(self, tokens):
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens

class Model:
    def __init__(self, model_size_or_path, device='auto', device_index=0, quantize=True, download_root=None, local_files_only=False):
        model_path = model_size_or_path if os.path.isdir(model_size_or_path) else download_model(model_size_or_path, download_root, local_files_only)
        self.model = Whisper(model_path,
                             device='cpu' if get_cuda_device_count() == 0 else device,
                             device_index=device_index,
                             compute_type='auto' if quantize else 'default')
                             # intra_threads=multiprocessing.cpu_count()) # I have no idea why this makes it **slower**
        self.tokenizer = Tokenizer(path=model_path)

    @property
    def device(self): return self.model.device
    @property
    def compute_type(self): return self.model.compute_type
    @property
    def n_mels(self): return self.model.n_mels

    def encode(self, features):
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        features = np.ascontiguousarray(features)
        features = StorageView.from_array(features.astype(np.float32))
        return self.model.encode(features, to_cpu=to_cpu)

    def generate_with_fallback(self, encoded, languages, temperatures, beam_size, patience, num_hypotheses, length_penalty,
                               logprob_threshold, nospeech_threshold, **model_args):
        batch_size = encoded.shape[0]

        no_speech = [0]*batch_size
        cands = [[] for _ in range(batch_size)]

        prompts = [[self.tokenizer.sot, l, self.tokenizer.transcribe] for l in languages]
        needs_fallback = [True]*batch_size
        for i, t in enumerate(temperatures):
            # explore more combinations?
            # beam size alone is trash
            decode_args = {"beam_size": beam_size, "patience": patience, "sampling_temperature": t}
            if i != 0:
                decode_args['num_hypotheses'] = num_hypotheses
            # decode_args = {"beam_size": beam_size, "patience": patience} if i == 0 else {"beam_size": beam_size, "num_hypotheses": num_hypotheses}
            # decode_args = {"beam_size": beam_size, "patience": patience, "num_hypotheses": num_hypotheses, "sampling_temperature": t}
            rs = self.model.generate(encoded, prompts, return_scores=True, return_no_speech_prob=True,
                                     length_penalty=0, **decode_args, **model_args)
            for i, r in enumerate(rs):
                no_speech[i] = r.no_speech_prob
                for j, k in enumerate(r.sequences_ids):
                    if k[-1] >= self.tokenizer.timestamp_begin and k[-1] < (self.tokenizer.timestamp_begin + 1500): # only well formed sequences
                        avg_logprob = r.scores[j] / (len(k)+1)
                        needs_fallback[i] &= avg_logprob < logprob_threshold and no_speech[i] < nospeech_threshold
                        if avg_logprob > logprob_threshold:
                            cands[i].append((r.scores[j], k))
                    else:
                        tqdm.write(f"FAILED {k[-1] >= self.tokenizer.timestamp_begin} {k[-1] < (self.tokenizer.timestamp_begin + 1500)}")

            tqdm.write(str(needs_fallback))
            if all(not k for k in needs_fallback):
                break

        results = []
        for c in cands:
            if c == []:
                tqdm.write("Decoding failed")
                results.append(None)
                continue
            # tqdm.write(pformat(c))
            penalty = (lambda x: x[0]/len(x[1])) if length_penalty is None else (lambda x: x[0]/(((5 + len(x[1]))/6)**length_penalty))
            results.append(sorted(c, key=penalty, reverse=True)[0])

        tqdm.write('')

        return results

    def transcribe(self, streams, bars, batch_size, language=None, **model_args):
        results = [[] for i in range(len(streams))]

        languages = language if isinstance(language, list) else [language] * len(streams)
        if language is not None:
            languages = [self.tokenizer.token_to_id("<|"+l+"|>") for l in languages]

        if len(streams) != len(languages):
            raise Exception("Idk")

        batch_size = min(len(streams), batch_size)

        active = list(range(batch_size)) # TODO sort by duration
        t = [next(streams[i]) for i in range(batch_size)]
        buffers = [k[0] for k in t]
        ends = [k[1] for k in t]
        seeks = [0] * batch_size
        pending = batch_size
        for k in active:
            bars[k].unpause()

        while len(active):
            padded = [np.pad(b[:, :3000], [(0, 0), (0, max(0, 3000 - b.shape[-1]))])
                      for b in buffers]
            encoded = self.encode(np.stack(padded))

            if any(languages[k] is None for k in active):
                r = self.model.detect_language(encoded)
                for i, k in enumerate(active):
                    languages[k] = self.tokenizer.token_to_id(r[i][0][0])

            rs = self.generate_with_fallback(encoded, [languages[i] for i in active], **model_args)
            discard = []
            for i, r in enumerate(rs):
                bar = bars[active[i]]
                if r is not None:
                    segments = self.tokenizer.decode_with_timestamps(r[1])
                    if len(segments[-1]) == 1:
                        seek = segments[-1][-1] - self.tokenizer.timestamp_begin
                        segments = segments[:-1]
                    else:
                        seek = 1500
                else: # Silence
                    segments = []
                    seek = 1500

                # tqdm.write(pformat(segments))
                pseek = seeks[i]

                results[active[i]].append((pseek, pseek+seek, segments))
                seeks[i]  = seek + pseek

                if ends[i] and seek >= buffers[i].shape[-1]:
                    bar.update(bar.total - bar.n)
                    bar.close()
                    if pending < len(streams):
                        active[i] = pending
                        n, end = next(streams[pending])
                        buffers[i] = n
                        ends[i] = end
                        seeks[i] = 0
                        pending += 1
                    else:
                        discard.append(i)
                else:
                    bar.update(seek*0.02)
                    buffers[i] = buffers[i][:, 2*seek:]
                    if not ends[i] and buffers[i].shape[-1] < 3000:
                        n, nend = next(streams[active[i]])
                        ends[i] = nend
                        buffers[i] = np.concatenate((buffers[i], n), axis=-1)
                        bars[i].unpause()
                bar.refresh()
            for k in discard:
                batch_size -= 1
                active.pop(k)
                ends.pop(k)
                buffers.pop(k)
                seeks.pop(k)

        # TODO trash
        out = []
        for k in results:
            out.append([])
            for j in k:
                segments = j[-1]
                if len(segments):
                    out[-1].extend([SubLine(idx=-1, content=self.tokenizer.decode(s[1:-1]),
                                            start=(j[0] + s[0] - self.tokenizer.timestamp_begin)*0.02,
                                            end=(j[0] + s[-1] - self.tokenizer.timestamp_begin)*0.02)
                                    for s in segments])
        return out

