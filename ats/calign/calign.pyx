# cython: language_level=3
from libc.stdint cimport int64_t, uint16_t, int16_t, uint32_t
from libc.stddef cimport size_t

import numpy as np
cimport numpy as cnp
cnp.import_array()

import parasail

cdef extern from "calign_api.c":
    cdef struct AlignmentParams:
        pass

    cdef struct Result:
        size_t *traceback
        size_t ltrace
        int64_t score

    size_t SIMD_ELEM
    int16_t *semiglobal(uint16_t *x, uint32_t lx, uint16_t *y, uint32_t ly,
                      int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend)
    void stride_seq(uint16_t *ret, uint16_t *seq, size_t len)

    void reverse16(uint16_t * r, uint16_t * s, size_t len, size_t alen)
    # Result hirschberg(uint16_t *x, uint16_t *y,
    #                   size_t lx, size_t ly, size_t alx, size_t aly,
    #                   int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
    #                   # size_t *traceback)
    #                   int64_t *traceback)
    int align(size_t, size_t)


cdef array_align(a: cnp.ndarray[cnp.uint16_t], n: int):
    # print(len(a), align(len(a), n))
    ret = np.zeros(align(len(a), n), dtype=a.dtype) # +1 because I'm dumb and use i32gather
    ret[:len(a)] = a
    return np.require(ret, requirements=['A', 'C', 'W', 'O', 'E'])

cdef test_stride(x: cnp.ndarray[cnp.uint16_t]):
    cdef cnp.ndarray[cnp.uint16_t] xa = array_align(x, SIMD_ELEM)
    cdef cnp.ndarray[cnp.uint16_t] out = np.zeros(len(xa), dtype=xa.dtype)
    stride_seq(<cnp.uint16_t *>out.data, <cnp.uint16_t *>xa.data, len(xa))

    return np.all(out[:-1] == xa[:-1].reshape(SIMD_ELEM, -1).T.flatten())

cdef test_reverse16(x: cnp.ndarray[cnp.uint16_t]):
    cdef cnp.ndarray[cnp.uint16_t] xa = array_align(x, SIMD_ELEM)
    cdef cnp.ndarray[cnp.uint16_t] out = np.zeros(len(xa), dtype=xa.dtype)
    reverse16(<cnp.uint16_t *>out.data, <cnp.uint16_t *>xa.data, len(xa), len(xa)-1)
    print(out)
    print(xa)
    return np.all(out == xa[::-1])

cdef test_reverse16_2(x: cnp.ndarray[cnp.uint16_t]):
    cdef cnp.ndarray[cnp.uint16_t] xa = array_align(x, SIMD_ELEM)
    cdef cnp.ndarray[cnp.uint16_t] rxa = array_align(x[::-1], SIMD_ELEM)
    cdef cnp.ndarray[cnp.uint16_t] out = np.zeros(len(xa), dtype=xa.dtype)
    reverse16(<cnp.uint16_t *>out.data, <cnp.uint16_t *>xa.data, len(x), len(xa)-1)
    print(out)
    print(rxa)
    return np.all(out == rxa)

cdef clastcol(x: cnp.ndarray[cnp.uint16_t], y: cnp.ndarray[cnp.uint16_t], match: int, mismatch: int, gap_open: int, gap_extend:int, reverse=False):
    cdef cnp.ndarray[cnp.uint16_t] idk = array_align(x[::-1] if reverse else x , SIMD_ELEM).reshape(SIMD_ELEM, -1).T.flatten()
    cdef cnp.ndarray[cnp.uint16_t] xa = np.require(idk, requirements=['A', 'C', 'W', 'O', 'E'])
    cdef cnp.ndarray[cnp.uint16_t] ya = np.require(y[::-1] if reverse else y, requirements=['A', 'C', 'W', 'O', 'E'])
    cdef cnp.int16_t[:] ret = <cnp.int16_t[:len(xa)]> semiglobal(<cnp.uint16_t *>xa.data,
                                                                   len(xa),
                                                                   <cnp.uint16_t *>ya.data,
                                                                   len(ya),
                                                                   match, mismatch, gap_open, gap_extend)
    idk2 = np.asarray(ret).reshape(-1, SIMD_ELEM).T.flatten()
    return idk2[:len(x)]
    # return idk2[len(idk)-len(x):] if reverse else idk2[:len(x)]


def plastcol(x: cnp.ndarray[cnp.uint16_t], y: cnp.ndarray[cnp.uint16_t], match: int, mismatch: int, gap_open: int, gap_extend:int, reverse=False):
    if reverse:
        x = x[::-1]
        y = y[::-1]
    x = ''.join([chr(i) for i in x])
    y = ''.join([chr(i) for i in y])

    alphabet = ''.join(np.union1d(list(x), list(y)))
    matrix = parasail.matrix_create(alphabet, match=match, mismatch=mismatch)
    r = parasail.sg_qx_rowcol_striped_16(x, y, open=abs(gap_open), extend=abs(gap_extend), matrix=matrix)
    return np.copy(r.score_col) # Need the copy for weird ctypes/gc reasons

def lastcol(x, y, match, mismatch, gap_open, gap_extend, reverse=False):
    if reverse:
        x = x[::-1]
        y = y[::-1]
    lx, ly = len(x), len(y)
    h = np.zeros((lx+1), dtype=np.int64)
    e = np.zeros((lx+1), dtype=np.int64)
    f = np.zeros((lx+1), dtype=np.int64)
    h_prev = 0

    for j in range(1, ly+1):
        f[0] = gap_open + (j-1) * gap_extend
        h_prev, h[0] = h[0], f[0]
        for i in range(1, lx+1):
            score = match if x[i-1] == y[j-1] else mismatch
            e[i] = max(e[i]+gap_extend, h[i]+gap_open)
            f[i] = max(f[i-1]+gap_extend, h[i-1]+gap_open)
            h_prev, h[i] = h[i], max(e[i], f[i], h_prev + score)
    return h[1:]

cdef hirschberg_inner(x: cnp.ndarray[cnp.uint16_t], y: cnp.ndarray[cnp.uint16_t], match:int, mismatch:int, gap_open:int, gap_extend:int, use_c:bool, use_p:bool):
    lx, ly = len(x), len(y)
    if lx == 0:
        return np.vstack((np.arange(len(y)), np.zeros(len(y)))).T
    if ly == 0:
        return np.vstack((np.arange(len(x)), np.zeros(len(x)))).T

    if lx == 1:
        return np.array([(0, lastcol(x, y, match, mismatch, gap_open, gap_extend).argmax())])
    if ly == 1:
        return np.array([(lx - lastcol(y, x, match, mismatch, gap_open, gap_extend).argmax()-1, 0)])

    if use_c:
        f = clastcol(x, y[:ly//2], match, mismatch, gap_open, gap_extend)
        s = clastcol(x, y[ly//2:], match, mismatch, gap_open, gap_extend, reverse=True)
    elif use_p:
        f = plastcol(x, y[:ly//2], match, mismatch, gap_open, gap_extend)
        s = plastcol(x, y[ly//2:], match, mismatch, gap_open, gap_extend, reverse=True)
    else:
        f = lastcol(x, y[:ly//2], match, mismatch, gap_open, gap_extend)
        s = lastcol(x, y[ly//2:], match, mismatch, gap_open, gap_extend, reverse=True)

    # Better way to deal with this?
    idk = (f[:-1] + s[:-1][::-1])
    mid = idk.argmax()
    print(f)
    print(s)

    ns = np.array([idk[mid], s[-1] + gap_open + (ly//2 - 1) * gap_extend, f[-1] + gap_open + (len(y) - ly//2 - 1) * gap_extend]).argmax()
    if ns == 1:
        mid = -1
    elif ns == 2:
        mid = len(x)-1

    mid += 1

    return np.concatenate((hirschberg_inner(x[:mid], y[:ly//2], match, mismatch, gap_open, gap_extend, use_c, use_p),
                           [(mid, ly//2)],
                           np.array([(mid+1, ly//2+1)]) + hirschberg_inner(x[mid+1:], y[ly//2+1:], match, mismatch, gap_open, gap_extend, use_c, use_p)), axis=0)

def hirschberg(query:str, database:str, match:int=1, mismatch:int=-1, gap_open:int=-1, gap_extend:int=-1, use_c=False, use_p=False):
    cdef cnp.ndarray[cnp.uint32_t] query_np = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32)
    cdef cnp.ndarray[cnp.uint32_t] database_np = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32)

    cdef cnp.ndarray[cnp.uint32_t] alphabet = np.union1d(query_np, database_np)
    if len(alphabet) > 2**16:
        raise Exception("Doesn't support alphabet sizes more than 2**16")

    cdef cnp.ndarray[cnp.uint16_t] q = np.searchsorted(alphabet, query_np).astype(np.uint16) + 1 # +1 For parasail, which uses \0 terminated strings
    cdef cnp.ndarray[cnp.uint16_t] d = np.searchsorted(alphabet, database_np).astype(np.uint16) + 1

    # return test_stride(q), test_reverse16(q), test_reverse16_2(q)
    return hirschberg_inner(q, d, match, mismatch, gap_open, gap_extend, use_c, use_p).T


# def pyhirschberg(query: str, database: str, match: int = 1, mismatch: int = -1, gap_open: int = -1, gap_extend: int = -1):
#     # le very important otherwise you get the BOM or w/e
#     # We don't support big endian lul
#     # np.dtype(int).np.newbyteorder('>')
#     cdef cnp.ndarray[cnp.uint32_t] query_np = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32)
#     cdef cnp.ndarray[cnp.uint32_t] database_np = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32)

#     cdef cnp.ndarray[cnp.uint32_t] alphabet = np.union1d(query_np, database_np)
#     if len(alphabet) > 2**16:
#         raise Exception("Doesn't support alphabet sizes more than 2**16")

#     cdef cnp.ndarray[cnp.uint16_t] q = array_align(np.searchsorted(alphabet, query_np).astype(np.uint16)   , SIMD_ELEM)
#     cdef cnp.ndarray[cnp.uint16_t] d = array_align(np.searchsorted(alphabet, database_np).astype(np.uint16), SIMD_ELEM)

#     cdef cnp.ndarray[cnp.intp_t] traceback = np.zeros(len(q) + len(d), dtype=np.intp)

#     print(len(q), len(d))
#     result = hirschberg(<cnp.uint16_t *> q.data, <cnp.uint16_t *> d.data,
#                len(query), len(database), len(q)-1, len(d)-1,
#                match, mismatch, gap_open, gap_extend,
#                <int64_t *> traceback.data)
#     print(result.ltrace)
#     return result.score, traceback[:result.ltrace].reshape(-1, 16).T.flatten() # For debugging for now

