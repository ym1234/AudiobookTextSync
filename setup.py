#!/usr/bin/env python3
from setuptools import setup, find_packages
from pathlib import Path
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

calign = Extension(name='ats.calign.calign',
                   sources=['ats/calign/calign.pyx'],
                   include_dirs=[np.get_include()],
                   define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                   extra_compile_args=['-O0', '-fsanitize=undefined', '-fsanitize=address', '-march=native', '-fopenmp'],  # -g, -O2
                   extra_link_args=['-fsanitize=undefined', '-fopenmp'])

setup(
    name='ats',
    version='1.2.0',
    description='Sync your ebooks & audiobooks',
    author='ym <>, KanjiEater <kanjieat3r@gmail.com>',
    license='MIT',
    license_files=('LICENSE',),
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=cythonize([calign], language_level="3"),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        "numpy",
        "wcwidth",
        "ctranslate2",
        "beautifulsoup4", # TODO
        "tqdm",
        "regex",
        "EbookLib", # TODO?
        "chinese_converter",
        "pycountry"
    ],
    python_requires='>=3.10',
    include_package_data=True
)
