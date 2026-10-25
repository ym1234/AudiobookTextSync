#!/usr/bin/env python3
import os

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

import platform

# ats/calign/simd.h picks avx2 or neon from what the compiler targets. neon is part of every arm64 cpu, avx2 has to
# be asked for: -mavx2 rather than -march=native, so a cpu that runs avx2 without advertising it (rosetta) still works
extra_compile_args = ['-O3']
extra_link_args = []
if platform.machine().lower() in ('x86_64', 'amd64'):
    extra_compile_args += ['-mavx2', '-mfma', '-mtune=native']

if os.environ.get("ATS_SANITIZE"):
    extra_compile_args += ['-O0', '-fsanitize=address', '-fsanitize=undefined']
    extra_link_args += ['-fsanitize=address', '-fsanitize=undefined']

calign = Extension(name='ats.calign.calign',
                   sources=['ats/calign/calign.pyx'],
                   include_dirs=[np.get_include()],
                   define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                   extra_compile_args=extra_compile_args,
                   extra_link_args=extra_link_args)

setup(ext_modules=cythonize([calign], language_level="3"))
