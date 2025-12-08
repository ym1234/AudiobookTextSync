import os
from queue import Queue
import multiprocessing
import huggingface_hub
import tokenizers
import numpy as np
from typing import Generator
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from ats.text import SubLine
from ats.audio import Stream
from ats import mel
from tqdm.auto import tqdm
from datetime import datetime
from ats.mel import MelWorker
from ats.np import np as cnp, no_cuda
import numpy as np

# Stupid hack because python doesn't have lazy imports (torch)
def _import_c2():
    import sys
    import importlib

    finder = importlib.machinery.PathFinder()
    c2spec = finder.find_spec('ctranslate2')
    if c2spec is None:
        raise ModuleNotFoundError('c2translate2')

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
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
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



@dataclass
class _TranscriptionState:
    idx: int
    buffer: any
    lines: list
    chunks: list
    seek: int
    bar: tqdm
    language: int
    done: bool

@dataclass
class Chunk:
    start: int # in mel frames
    end: int
    segment_start: int
    segment_end: int
    temperature: float
    logprob: float
    nospeech_prob: float
    tokens: list # idk if i care about this

@dataclass
class ChapterTranscript:
    title: str
    start: float
    end: float
    language: str
    segments: list
    chunks: [Chunk]

@dataclass
class Transcript:
    model: str
    at: datetime
    confidence: float
    chapters: [ChapterTranscript]

class Model:
    def __init__(self, model, device='auto', device_index=0, quantize=True, download_root=None, local_files_only=False):
        model_path = model if os.path.isdir(model) else download_model(model, download_root, local_files_only)
        self.model_name = model
        num_cuda = get_cuda_device_count()
        device = 'cpu' if num_cuda == 0 else device
        self.model = Whisper(model_path, max_queued_batches=-1, device=device, device_index=device_index, compute_type='auto' if quantize else 'default')
        self.tokenizer = Tokenizer(path=model_path)

    @property
    def device(self): return self.model.device
    @property
    def compute_type(self): return self.model.compute_type
    @property
    def n_mels(self): return self.model.n_mels

    def encode(self, features):
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        features = cnp.ascontiguousarray(features)
        features = StorageView.from_array(features.astype(cnp.float32))
        return self.model.encode(features, to_cpu=to_cpu)

    def generate_with_fallback(self, encoded, languages, temperatures, beam_size, patience, num_hypotheses, length_penalty,
                               logprob_threshold, nospeech_threshold, **model_args):
        batch_size = encoded.shape[0]

        no_speech = [0]*batch_size
        cands = [[] for _ in range(batch_size)]

        prompts = [[self.tokenizer.sot, l, self.tokenizer.transcribe] for l in languages]
        needs_fallback = [True]*batch_size
        for i, t in enumerate(temperatures):
            # beam size alone is trash
            if t > 0:
                decode_args = dict(beam_size=1, sampling_temperature=t, num_hypotheses=num_hypotheses)
            else:
                decode_args = dict(beam_size=beam_size, patience=patience)
            if i != 0:
                tqdm.write(f"DECODING FAILED!! {i}")
            rs = self.model.generate(encoded, prompts, return_scores=True, return_no_speech_prob=True,
                                     length_penalty=0, **decode_args, **model_args)
            for i, r in enumerate(rs):
                no_speech[i] = r.no_speech_prob
                for j, k in enumerate(r.sequences_ids):
                    if k[-1] < self.tokenizer.timestamp_begin or k[-1] >= self.tokenizer.timestamp_begin + 1500: # only well formed sequences
                        continue
                    avg_logprob = r.scores[j] / (len(k)+1)
                    needs_fallback[i] &= avg_logprob < logprob_threshold and no_speech[i] < nospeech_threshold
                    if avg_logprob > logprob_threshold:
                        cands[i].append((k, r.scores[j], no_speech[i], t))

            if all(not k for k in needs_fallback):
                break

        def gnmt(x): return x[1] / ((5 + len(x[0])) / 6)**length_penalty
        def norm(x): return x[1] / len(x[0])
        penalty = norm if length_penalty is None else gnmt
        return [sorted(c, key=penalty)[-1] if c else ([], -1, no_speech[i], -1) for c in cands]

    def transcribe(self, requests, num_chunks, batch_size, use_stream_language=False, **model_args):
        jobs = []
        for r in requests:
            file = r['file']
            stream = file.streams[r['stream']] if 'stream' in r else file.streams[file.default_stream]
            language = r['language'] if 'language' in r else stream.language if use_stream_language else None
            language = self.tokenizer.token_to_id("<|" + language + "|>") if language is not None else None  # gets filled in later by whisper
            for c in file.chapters:
                if c in r.get('ignore', {}):
                    continue
                jobs.append(dict(path=file.path, stream=stream.idx, language=language, chapter=c, queue=Queue(maxsize=1)))

        results = self._transcribe(jobs, num_chunks, batch_size, **model_args)
        idx = [0] + np.cumsum([len(r['file'].chapters) for r in requests], dtype=int).tolist()
        grouped = [results[s:e] for s, e in zip(idx, idx[1:])]

        def confidence(group):
            vals = np.array([(c.logprob, len(c.tokens)) for ch in group for c in ch.chunks])
            return np.exp(vals[:, 0].sum()/(vals[:, 1].sum()+1))

        return [Transcript(model=self.model_name, at=datetime.now(), confidence=confidence(grouped[i]), chapters=grouped[i])
                for i in range(len(requests))]

    def _transcribe(self, jobs, num_chunks, batch_size, **model_args):
        main_bar = tqdm(total=len(jobs), desc="Transcribing", position=0,
                        bar_format="{l_bar}{bar}{n:3.2f}/{total:3.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                        leave=True)

        total_length = sum(j['chapter'].end - j['chapter'].start for j in jobs)
        jobs_sorted = sorted(range(len(jobs)), key=lambda x: jobs[x]['chapter'].end - jobs[x]['chapter'].start, reverse=True)

        mel_queue = Queue()
        for k in jobs_sorted: mel_queue.put(jobs[k])
        mel_workers = [MelWorker(mel_queue, n_mels=self.n_mels, num_chunks=num_chunks) for _ in range(multiprocessing.cpu_count())]
        # mel_queue.shutdown() # This is pretty new
        for _ in mel_workers: mel_queue.put(None) # poison value
        for w in mel_workers: w.start()

        results = [None for _ in range(len(jobs))]
        def finalize_transcript(active):
            idx = active.idx
            chapter = jobs[idx]['chapter']
            results[idx] = ChapterTranscript(title=chapter.title, start=chapter.start, end=chapter.end, language=jobs[idx]['language'],
                                             chunks=active.chunks, segments=active.lines)
            active.bar.close()
            main_bar.update((chapter.end - chapter.start)/total_length * len(jobs))

        def new_active(pending):
            idx = jobs_sorted[pending]
            chapter = jobs[idx]['chapter']
            bar = tqdm(total=chapter.end-chapter.start, unit_scale=True, unit=" seconds", desc=chapter.title)
            return _TranscriptionState(idx=idx, buffer=cnp.zeros((self.n_mels, 0)), lines=[], chunks=[],
                                       seek=0, bar=bar, language=jobs[idx]['language'], done=False)
        active, pending = [], 0
        for i in range(min(len(jobs), batch_size)):
            active.append(new_active(i))
            pending += 1

        while True:
            i = 0
            while i < len(active):
                a = active[i]
                if not a.done and a.buffer.shape[-1] < 3000:
                    buf, end = jobs[a.idx]['queue'].get()
                    a.buffer = cnp.concatenate((a.buffer, buf), axis=-1)
                    if end: a.done = True
                elif a.buffer.shape[-1] == 0:
                    finalize_transcript(a)
                    if pending < len(jobs):
                        active[i] = new_active(pending)
                        pending += 1
                    else:
                        active.pop(i)
                    i -= 1 # lol hacky
                i += 1

            if not len(active):
                break

            padded = [cnp.pad(a.buffer[:, :3000], [(0, 0), (0, max(0, int(3000 - a.buffer.shape[-1])))])
                    for a in active]
            encoded = self.encode(cnp.stack(padded))

            if any(a.language is None for a in active):
                r = self.model.detect_language(encoded)
                for i, a in enumerate(active):
                    a.language = self.tokenizer.token_to_id(r[i][0][0])

            rs = self.generate_with_fallback(encoded, [a.language for a in active], **model_args)
            for i, a in enumerate(active):
                tokens, score, no_speech, temperature = rs[i]

                segments, seek = [], 1500
                if tokens:
                    segments = self.tokenizer.decode_with_timestamps(tokens)
                    if len(segments[-1]) == 1:
                        seek = segments[-1][-1] - self.tokenizer.timestamp_begin
                        segments = segments[:-1]

                lines = [SubLine(content=self.tokenizer.decode(s[1:-1]),
                                 start=jobs[a.idx]['chapter'].start + (a.seek+s[0]-self.tokenizer.timestamp_begin)*0.02,
                                 end=jobs[a.idx]['chapter'].start + (a.seek+s[-1]-self.tokenizer.timestamp_begin)*0.02)
                         for s in segments]

                nc = Chunk(start=a.seek*2, end=a.seek*2+seek*2,
                           segment_start=len(a.lines), segment_end=len(a.lines)+len(lines),
                           temperature=temperature, nospeech_prob=no_speech,
                           logprob=score, tokens=tokens)
                a.lines.extend(lines)
                a.chunks.append(nc)
                a.seek += seek

                a.buffer = a.buffer[:, 2*seek:]
                a.bar.update(min(a.bar.total - a.bar.n, seek*0.02))

        for w in mel_workers: w.join()
        main_bar.close()
        return results

