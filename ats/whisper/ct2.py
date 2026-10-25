import os
import multiprocessing
import huggingface_hub
import numpy as np
from ats.np import np as cnp
from ats.whisper.tokenizer import Tokenizer

# Stupid hack because python doesn't have lazy imports (torch)
def _import_c2():
    import sys
    import importlib

    finder = importlib.machinery.PathFinder()
    c2spec = finder.find_spec('ctranslate2')
    if c2spec is None:
        raise ModuleNotFoundError('ctranslate2')

    if sys.platform == "win32":
        import ctypes
        from importlib import resources

        c2resources = resources.files(importlib.util.module_from_spec(c2spec))

        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            add_dll_directory(str(c2resources)) # Need str here?

        for library in c2resources.glob('*.dll'):
            ctypes.CDLL(library)

    if sys.platform == "linux":
        # ct2 loads cublas by soname, but the pip wheels (nvidia-cublas-cu12, which cupy-cuda12x[ctk] pulls in) put it
        # in site-packages where the loader never looks. without this ct2 only found it if cupy happened to be imported
        # first and loaded it. a system install still gets found the usual way when there's no wheel
        import ctypes
        nvidia = finder.find_spec('nvidia')
        for d in (nvidia.submodule_search_locations if nvidia else []):
            for lib in ('cublas/lib/libcublasLt.so.12', 'cublas/lib/libcublas.so.12'):
                if os.path.exists(path := os.path.join(d, lib)):
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

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

class Ct2Backend:
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
    @property
    def is_multilingual(self): return self.model.is_multilingual

    def encode(self, features):
        features = StorageView.from_array(cnp.ascontiguousarray(features).astype(cnp.float32))
        return np.asarray(self.model.encode(features, to_cpu=True)) # TODO, make it always live on the gpu. For now we can't do that with cupy

    def detect_language(self, encoded):
        return [self.tokenizer.token_to_id(r[0][0]) for r in self.model.detect_language(StorageView.from_array(encoded))]

    def generate(self, encoded, prompts, temperature, beam_size, patience, num_hypotheses, **model_args):
        # beam size alone is trash, and it only makes sense at temperature zero. the temperature always has to be
        # passed because ct2 only uses its greedy sampler at temperature 0 or topk 1, and otherwise samples, beam search
        # included, at its default temperature of 1
        args = dict(beam_size=beam_size, patience=patience) if temperature == 0 else \
               dict(beam_size=1, num_hypotheses=num_hypotheses)
        stacked = np.stack(encoded) # from_array doesn't take ownership, so this has to outlive the call
        return self.model.generate(StorageView.from_array(stacked), prompts, return_scores=True,
                                   return_no_speech_prob=True, length_penalty=0, sampling_temperature=temperature,
                                   **args, **model_args)
