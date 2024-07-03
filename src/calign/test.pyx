# cython: language_level=3, autogen_pxd=True
from libc.stdint cimport uint16_t, int16_t
from libc.stddef cimport size_t

import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "calign.c":
    cdef struct AlignmentState:
        uint16_t *traceback;
        size_t idx

    size_t SIMD_ELEM
    int hirschberg(
        uint16_t *, uint16_t *,
        size_t, size_t,
        int16_t, int16_t, int16_t, int16_t,
        AlignmentState *);
    int semiglobal(AlignmentState *);
    int align(size_t, size_t);


def array_align(a: cnp.ndarray[cnp.uint16_t], n: int):
    zeros = np.zeros(align(len(a), n) - len(a), dtype=a.dtype)
    print(zeros)
    return np.concatenate([a, zeros], axis=0)

def pyhirschberg(query: str, database: str, match: int = 1, mismatch: int = -1, gap_open: int = -1, gap_extend: int = -1):
    cdef cnp.ndarray[cnp.uint32_t] query_np = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32)
    cdef cnp.ndarray[cnp.uint32_t] database_np = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32)

    cdef cnp.ndarray[cnp.uint32_t] alphabet = np.union1d(query_np, database_np)
    if len(alphabet) > 2**16:
        raise Exception("Doesn't support alphabet sizes more than 2**16")

    cdef cnp.ndarray[cnp.uint16_t] q = array_align(np.searchsorted(alphabet, query_np).astype(np.uint16), SIMD_ELEM)
    cdef cnp.ndarray[cnp.uint16_t] d = array_align(np.searchsorted(alphabet, database_np).astype(np.uint16), SIMD_ELEM)
    q = np.ascontiguousarray(q)
    d = np.ascontiguousarray(d)

    hirschberg(<cnp.uint16_t *> q.data, <cnp.uint16_t *> d.data, len(q), len(d), match, mismatch, gap_open, gap_extend, NULL)
