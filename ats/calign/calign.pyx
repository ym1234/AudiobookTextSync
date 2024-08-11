from libc.stdint cimport int64_t, uint16_t, int16_t, uint32_t
from libc.stddef cimport size_t
from libc.stdio cimport printf

from cython.parallel import threadid, parallel, prange
import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "hirschberg.c":
    cdef struct AlignmentParams:
        pass

    cdef struct Result:
        size_t *traceback
        size_t ltrace
        int64_t score

    size_t SIMD_ELEM
    void stride_seq(uint16_t *, uint16_t *, size_t, size_t)
    void reverse16(uint16_t *, uint16_t *, size_t)
    Result hirschberg(uint16_t *x, uint16_t *y,
                      size_t lx, size_t ly, size_t alx, size_t aly,
                      int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
                      # size_t *traceback)
                      int64_t *traceback)
    int align(size_t, size_t)


cdef array_align(a: cnp.ndarray[cnp.uint16_t], n: int):
    ret = np.zeros(align(len(a), n)+1, dtype=a.dtype) # +1 because I'm dumb and use i32gather
    ret[:len(a)] = a
    return np.require(ret, requirements=['A', 'C', 'W', 'O', 'E'])[:-1]

def test_stride(x: cnp.ndarray[cnp.uint16_t]):
    cdef cnp.ndarray[cnp.uint16_t] xa = array_align(x, SIMD_ELEM)
    cdef cnp.ndarray[cnp.uint16_t] out = np.zeros(len(xa), dtype=xa.dtype)
    stride_seq(<cnp.uint16_t *>out.data, <cnp.uint16_t *>xa.data, len(x), len(xa))

    return np.all(out[:-1] == xa[:-1].reshape(SIMD_ELEM, -1).T.flatten())

def test_reverse16(x: cnp.ndarray[cnp.uint16_t]):
    cdef cnp.ndarray[cnp.uint16_t] xa = array_align(x, SIMD_ELEM)
    cdef cnp.ndarray[cnp.uint16_t] out = np.zeros(len(xa), dtype=xa.dtype)
    reverse16(<cnp.uint16_t *>out.data, <cnp.uint16_t *>xa.data, len(xa))
    print(out)
    print(xa)
    return np.all(out == xa[::-1])

def test_reverse16_2(x: cnp.ndarray[cnp.uint16_t]):
    cdef cnp.ndarray[cnp.uint16_t] xa = array_align(x, SIMD_ELEM)
    cdef cnp.ndarray[cnp.uint16_t] rxa = array_align(x[::-1], SIMD_ELEM)
    cdef cnp.ndarray[cnp.uint16_t] out = np.zeros(len(xa), dtype=xa.dtype)
    reverse16(<cnp.uint16_t *>out.data, <cnp.uint16_t *>xa.data, len(x))
    print(out)
    print(rxa)
    return np.all(out == rxa)

def pyhirschberg(query: str, database: str, match: int = 1, mismatch: int = -1, gap_open: int = -1, gap_extend: int = -1):
    # le very important otherwise you get the BOM or w/e
    cdef cnp.ndarray[cnp.uint32_t] query_np = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32)
    cdef cnp.ndarray[cnp.uint32_t] database_np = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32)

    cdef cnp.ndarray[cnp.uint32_t] alphabet = np.union1d(query_np, database_np)
    if len(alphabet) > 2**16:
        raise Exception("Doesn't support alphabet sizes more than 2**16")

    cdef cnp.ndarray[cnp.uint16_t] q = array_align(np.searchsorted(alphabet, query_np).astype(np.uint16)   , SIMD_ELEM) + 1 # use 0 as padding
    cdef cnp.ndarray[cnp.uint16_t] d = array_align(np.searchsorted(alphabet, database_np).astype(np.uint16), SIMD_ELEM) + 1

    cdef cnp.ndarray[cnp.intp_t] traceback = np.zeros(len(q) + len(d), dtype=np.intp)

    result = hirschberg(<cnp.uint16_t *> q.data, <cnp.uint16_t *> d.data,
               len(query), len(database), len(q), len(d),
               match, mismatch, gap_open, gap_extend,
               <int64_t *> traceback.data)
    return result.score, traceback[:result.ltrace].reshape(-1, 16).T.flatten()[:len(query)] # For debugging for now

