import os
import multiprocessing
import huggingface_hub
import tokenizers
import time
import numpy as np
from functools import cached_property
from pprint import pprint
from ats.text import SubLine

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
        while end < len(tokens):
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
                             compute_type='auto' if quantize else 'default',
                             intra_threads=multiprocessing.cpu_count())
        self.tokenizer = Tokenizer(path=model_path)

    @property
    def device(self): return self.model.device
    @property
    def compute_type(self): return self.model.compute_type
    @property
    def n_mels(self): return self.model.n_mels

    def encode(self, features):
        features = np.ascontiguousarray(features)
        features = StorageView.from_array(features.astype(np.float32))
        return self.model.encode(features)#, to_cpu=to_cpu)

    def transcribe(self, streams, batch_size, language=None): # todo more params
        results = [[] for i in range(len(streams))]
        languages = language if isinstance(language, list) else [language] * len(streams)

        batch_size = min(len(streams), batch_size)

        active = list(range(batch_size))
        buffers = [next(streams[i]) for i in range(batch_size)]
        seeks = [0] * batch_size # offsets?
        pending = batch_size

        while len(active):
            pads = [max(0, 3000 - b.shape[-1]) for b in buffers]
            padded = [np.pad(b[:, :3000], [(0, 0), (0, pads[i])]) for i, b in enumerate(buffers)]

            batch = np.stack(padded)
            encoded = self.encode(batch)

            if any(languages[k] is None for k in active):
                r = self.model.detect_language(encoded)
                for i, k in enumerate(active):
                    languages[k] = self.tokenizer.token_to_id(r[i][0][0])

            rs = self.model.generate(encoded,
                                     [[self.tokenizer.sot, languages[i], self.tokenizer.transcribe] for i in active],
                                     return_scores=True, suppress_blank=False, return_no_speech_prob=True, repetition_penalty=5,
                                     beam_size=5)
            discard = []
            for i, r in enumerate(rs):
                segments = self.tokenizer.decode_with_timestamps(r.sequences_ids[0])

                if segments[-1][-1]  < self.tokenizer.timestamp_begin:
                    print(r.sequences_ids[0])
                    print(i, "DECODING FAILED")
                    # exit(0)

                seek = (segments[-1][-1] - self.tokenizer.timestamp_begin) * 2
                pseek = seeks[i]

                results[active[i]].append((pseek, pseek+seek, segments))
                seeks[i]  = seek + pseek

                if seek >= buffers[i].shape[-1]:
                    if pending < len(streams):
                        active[i] = pending
                        buffers[i] = next(streams[pending])
                        seeks[i] = 0
                        pending += 1
                    else:
                        discard.append(i)
                else:
                    buffers[i] = buffers[i][:, seek:]
                    if buffers[i].shape[-1] < 3000:
                        try:
                            n = next(streams[active[i]])
                            buffers[i] = np.concatenate((buffers[i], n), axis=-1)
                        except StopIteration:
                            pass
                print(i, buffers[i].shape, seek, r.scores, r.no_speech_prob, self.tokenizer.decode(r.sequences_ids[0]))
            for k in discard:
                batch_size -= 1
                active.pop(k)
                buffers.pop(k)
                seeks.pop(k)

        out = []
        for k in results:
            out.append([])
            for j in k:
                segments = j[-1]
                out[-1].extend([SubLine(idx=-1, content=self.tokenizer.decode(s[1:-1]), start=j[0]*0.01 + (s[0] - self.tokenizer.timestamp_begin)*0.02, end=j[0]*0.01 + (s[-1] - self.tokenizer.timestamp_begin)*0.02) for s in segments])
        # pprint(out)
        return out

