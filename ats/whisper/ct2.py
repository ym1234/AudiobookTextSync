import os
import numpy as np
from ats.np import np as cnp
from ats.whisper.tokenizer import Tokenizer
from ats.whisper.models import CT2_MODELS, download

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

class Ct2Backend:
    def __init__(self, model, device='auto', device_index=0, quantize=True, download_root=None, local_files_only=False):
        model_path = model if os.path.isdir(model) else download(CT2_MODELS.get(model, model), download_root, local_files_only)
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
        stacked = np.stack(encoded) # from_array doesn't take ownership, so this has to outlive the call
        return [self.tokenizer.token_to_id(r[0][0]) for r in self.model.detect_language(StorageView.from_array(stacked))]

    def generate(self, encoded, prompts, temperatures, opts):
        """one ct2 call per distinct temperature, ct2 only takes a single temperature per call"""
        results = [None] * len(prompts)
        for temperature in set(temperatures):
            rows = [i for i, t in enumerate(temperatures) if t == temperature]
            args = dict(beam_size=1, num_hypotheses=1 if temperature == 0 else opts.num_hypotheses)
            stacked = np.stack([encoded[i] for i in rows]) # from_array doesn't take ownership, this has to outlive the call
            rs = self.model.generate(StorageView.from_array(stacked), [prompts[i] for i in rows], return_scores=True,
                                     return_no_speech_prob=True, length_penalty=0, sampling_temperature=temperature,
                                     sampling_topk=opts.sampling_topk, suppress_blank=opts.suppress_blank,
                                     suppress_tokens=opts.suppress_tokens, repetition_penalty=opts.repetition_penalty,
                                     no_repeat_ngram_size=opts.no_repeat_ngram_size,
                                     max_initial_timestamp_index=opts.max_initial_timestamp_index, **args)
            for i, r in zip(rows, rs): results[i] = r
        return results
