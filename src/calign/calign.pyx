# cython: language_level=3, autogen_pxd=True
from libc.stdint cimport int64_t, uint16_t, int16_t
from libc.stddef cimport size_t
# import pyximport
# py_importer, pyx_importer = pyximport.install()
# from .calign import *


import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "calign_api.c":
    cdef struct AlignmentParams:
        pass

    cdef struct Result:
        size_t *traceback;
        size_t ltrace;
        int64_t score;

    # size_t SIMD_WIDTH
    size_t SIMD_ELEM
    Result hirschberg(
        uint16_t *x, uint16_t *y,
        size_t lx, size_t ly, size_t alx, size_t aly,
        int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
        size_t *traceback);
    int align(size_t, size_t);


cdef array_align(a: cnp.ndarray[cnp.uint16_t], n: int):
    zeros = np.zeros(align(len(a), n) - len(a), dtype=a.dtype)
    return np.concatenate([a, zeros], axis=0, dtype=a.dtype)

def pyhirschberg(query: str, database: str, match: int = 1, mismatch: int = -1, gap_open: int = -1, gap_extend: int = -1):
    cdef cnp.ndarray[cnp.uint32_t] query_np = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32)
    cdef cnp.ndarray[cnp.uint32_t] database_np = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32)

    cdef cnp.ndarray[cnp.uint32_t] alphabet = np.union1d(query_np, database_np)
    if len(alphabet) > 2**16:
        raise Exception("Doesn't support alphabet sizes more than 2**16")

    cdef cnp.ndarray[cnp.uint16_t] q = array_align(np.searchsorted(alphabet, query_np).astype(np.uint16), SIMD_ELEM*2)
    cdef cnp.ndarray[cnp.uint16_t] d = array_align(np.searchsorted(alphabet, database_np).astype(np.uint16), SIMD_ELEM*2)


    requirements = ['A', 'C', 'W', 'O', 'E']
    cdef cnp.ndarray[cnp.uint16_t] q16 = np.require(q, requirements=requirements)
    cdef cnp.ndarray[cnp.uint16_t] d16 = np.require(d, requirements=requirements)

    cdef cnp.ndarray[cnp.uintp_t] traceback = np.zeros(len(q16) + len(d16), dtype=np.uintp)

    print(len(q16))
    print(len(d16))
    hirschberg(<cnp.uint16_t *> q16.data, <cnp.uint16_t *> d16.data,
               len(query), len(database), len(q16), len(d16),
               match, mismatch, gap_open, gap_extend,
               <cnp.uintp_t *> traceback.data)
