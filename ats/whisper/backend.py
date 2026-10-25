import sys
import platform
import importlib.util

BACKENDS = ('auto', 'mlx', 'ct2')

def load_backend(backend='auto', **model_args):
    """the backends are imported here and not at the top, mlx only exists on apple silicon and ct2 is optional there"""
    if backend == 'auto':
        apple_silicon = sys.platform == 'darwin' and platform.machine() == 'arm64'
        backend = 'mlx' if apple_silicon and importlib.util.find_spec('mlx') else 'ct2'
    if backend == 'mlx':
        from ats.whisper.mlx.backend import MlxBackend
        return MlxBackend(**model_args)
    if backend == 'ct2':
        from ats.whisper.ct2 import Ct2Backend
        return Ct2Backend(**model_args)
    raise ValueError(f"unknown backend {backend!r}, one of {', '.join(BACKENDS)}")
