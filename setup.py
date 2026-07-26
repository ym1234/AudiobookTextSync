#!/usr/bin/env python3
import os

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extra_compile_args = ['-O3', '-march=native', '-fopenmp']
extra_link_args = ['-fopenmp']

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
