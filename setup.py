#!/usr/bin/env python3

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
                   extra_compile_args=['-march=native', '-fopenmp'],  # -g, -O2
                   extra_link_args=['-fopenmp'])

hirschberg = Extension(name='ats.extra.hirschberg',
                       sources=['ats/extra/hirschberg.pyx'],
                       include_dirs=[np.get_include()],
                       define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                       extra_compile_args=['-march=native', "-O2"])


setup(name='ats',
      version='1.2.0',
      description='Sync your ebooks & audiobooks',
      author='ym <>, KanjiEater <kanjieat3r@gmail.com>',
      license='MIT',
      license_files=('LICENSE',),
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['ats', 'ats.calign'],
      ext_modules=cythonize([calign, hirschberg], language_level="3"),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=["numpy",
                        "wcwidth",
                        "faster_whisper",
                        "biopython", # TODO
                        "beautiful_soup", # TODO
                        "tqdm",
                        "rapidfuzz", # TODO?
                        "ffmpeg_python",
                        "regex",
                        "EbookLib", # TODO?
                        "chinese_converter",
                        "pycountry"],
      python_requires='>=3.10',
      include_package_data=True)
