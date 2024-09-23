from libc.stdint cimport int64_t, uint16_t, int16_t, uint32_t
from libc.stddef cimport size_t

from posix.mman cimport munmap
from cython cimport view
import ctypes

import numpy as np
cimport numpy as cnp
cnp.import_array()

import parasail

cdef extern from "semiglobal.c":
    size_t SIMD_ELEM
    int16_t *semiglobal(uint16_t *x, uint32_t lx, uint16_t *y, uint32_t ly,
                      int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend)
    size_t align(size_t, size_t)

cdef extern from "needleman.c":
    int16_t *needleman(uint16_t *x, uint32_t lx, uint16_t *y, uint32_t ly,
                      int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
                      int16_t **rH, int16_t **rE)

cdef array_align(a: cnp.ndarray[cnp.uint16_t], n: int):
    # print(len(a), align(len(a), n))
    ret = np.zeros(align(len(a), n), dtype=a.dtype) # +1 because I'm dumb and use i32gather
    ret[:len(a)] = a
    return np.require(ret, requirements=['A', 'C', 'W', 'O', 'E'])

def sclastcol(x: cnp.ndarray[cnp.uint16_t], y: cnp.ndarray[cnp.uint16_t], match: int, mismatch: int, gap_open: int, gap_extend:int, reverse=False):
    cdef cnp.ndarray[cnp.uint16_t] idk = array_align(x[::-1] if reverse else x , SIMD_ELEM).reshape(SIMD_ELEM, -1).T.flatten()
    cdef cnp.ndarray[cnp.uint16_t] xa = np.require(idk, requirements=['A', 'C', 'W', 'O', 'E'])
    cdef int z = len(xa)
    cdef cnp.ndarray[cnp.uint16_t] ya = np.require(y[::-1] if reverse else y, requirements=['A', 'C', 'W', 'O', 'E'])
    cdef cnp.int16_t[:] ret = <cnp.int16_t[:len(xa)]> semiglobal(<cnp.uint16_t *>xa.data,
                                                                   len(xa),
                                                                   <cnp.uint16_t *>ya.data,
                                                                   len(ya),
                                                                   match, mismatch, gap_open, gap_extend)
    # ret.callback_free_data = lambda x: munmap(<void *>x, align(z * sizeof(uint16_t), 4096))
    idk2 = np.asarray(ret).reshape(-1, SIMD_ELEM).T.flatten()
    return idk2[:len(x)]

def nw(x: cnp.ndarray[cnp.uint16_t], y: cnp.ndarray[cnp.uint16_t], match: int, mismatch: int, gap_open: int, gap_extend:int, reverse=False):
    cdef int16_t *H = NULL
    cdef int16_t *E = NULL

    cdef cnp.ndarray[cnp.uint16_t] idk = array_align(x[::-1] if reverse else x , SIMD_ELEM).reshape(SIMD_ELEM, -1).T.flatten()
    cdef cnp.ndarray[cnp.uint16_t] xa = np.require(idk, requirements=['A', 'C', 'W', 'O', 'E'])
    cdef int z = len(xa)
    cdef cnp.ndarray[cnp.uint16_t] ya = np.require(y[::-1] if reverse else y, requirements=['A', 'C', 'W', 'O', 'E'])
    needleman(<cnp.uint16_t *>xa.data, len(xa), <cnp.uint16_t *>ya.data, len(ya), match, mismatch, gap_open, gap_extend, &H, &E)

    cdef view.array mH = <cnp.int16_t[:len(xa)]> H
    cdef view.array mE = <cnp.int16_t[:len(xa)]> E
    # Idk why this isn't working, tried to tie it to a module-level object so that the gc doesn't kill it but...
    # https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#cython-arrays
    # def free(x):
    #     munmap(<void *>x, align(z * sizeof(uint16_t), 4096))
    # f = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(free)
    # cdef (void (*)(void *) noexcept) ptr = (<void (*)(void *) noexcept><size_t>ctypes.addressof(refs[-1]))
    # mH.callback_free_data = ptr
    # mE.callback_free_data = ptr
    vH, vE = np.asarray(mH).copy().reshape(-1, SIMD_ELEM).T.flatten()[:len(x)].astype(np.int64), np.asarray(mE).copy().reshape(-1, SIMD_ELEM).T.flatten()[:len(x)].astype(np.int64)
    munmap(H,  align(z * sizeof(int16_t), 4096))
    munmap(E,  align(z * sizeof(int16_t), 4096))
    return vH, vE

def lastcol(x, y, match, mismatch, gap_open, gap_extend, reverse=False):
    if reverse:
        x, y = x[::-1], y[::-1]
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

def traceback(x, y, H, E, F, cx, cy, match, mismatch, gap_open, gap_extend, start=False):
    cur, traceback = 0, []
    while cx > 0 and cy > 0:
        if cur == 0:
            score = match if x[cx-1] == y[cy-1] else mismatch
            if H[cx, cy] == E[cx, cy]:
                cur = 1
            elif H[cx, cy] == F[cx, cy]:
                cur = 2
            elif (H[cx-1, cy-1] + score) == H[cx, cy]:
                traceback.append((cx, cy))
                cx, cy = cx-1, cy-1
        elif cur == 1:
            traceback.append((cx, cy))
            if (H[cx, cy-1] + gap_open) == E[cx, cy] and (E[cx, cy-1] + gap_extend) != E[cx, cy]:
                cur = 0
            cy -= 1
        elif cur == 2:
            traceback.append((cx, cy))
            if (H[cx-1, cy] + gap_open) == F[cx, cy] and (F[cx-1, cy] + gap_extend) != F[cx, cy]:
                cur = 0
            cx -= 1

    if start:
        while cx > 0:
            traceback.append((cx, cy))
            cx -= 1

        while cy > 0:
            traceback.append((cx, cy))
            cy -= 1

    return np.array(traceback)[::-1].swapaxes(-1, -2)-1

def nw_full(x, y, match=1, mismatch=-1, gap_open=-1, gap_extend=-1):
    lx, ly = len(x), len(y)

    h = np.zeros((lx+1, ly+1))
    e = np.full((lx+1, ly+1), fill_value=-np.inf)
    f = np.full((lx+1, ly+1), fill_value=-np.inf)

    for i in range(1, ly+1):
        e[0, i] = max(e[0, i-1]+gap_extend, h[0, i-1]+gap_open)
        h[0, i] = e[0, i]

    for i in range(1, lx+1):
        f[i, 0] = max(f[i-1, 0]+gap_extend, h[i-1, 0]+gap_open)
        h[i, 0] = f[i, 0]

    for i in range(1, lx+1):
        for j in range(1, ly+1):
            score = match if x[i-1] == y[j-1] else mismatch
            e[i, j] = max(e[i, j-1]+gap_extend, h[i, j-1]+gap_open)
            f[i, j] = max(f[i-1, j]+gap_extend, h[i-1, j]+gap_open)
            h[i, j] = max(e[i, j], f[i, j], h[i-1, j-1]+score)

    return traceback(x, y, h, e, f, lx, ly, match, mismatch, gap_open, gap_extend, start=True)

# Debug
# Have actual tests?
# def do_parasail(x, y, match, mismatch, gap_open, gap_extend):
#     import parasail
#     matrix = parasail.matrix_create(''.join(list(np.union1d(x, y))), match=match, mismatch=mismatch, case_sensitive=True)
#     r = parasail.nw_rowcol_striped_16(x, y, open=abs(gap_open), extend=abs(gap_extend), matrix=matrix)
#     return np.copy(r.score_col)

# def do_nw(x, y, match, mismatch, gap_open, gap_extend):
#     query_np = np.frombuffer(x.encode('utf-32le'), dtype=np.uint32)
#     database_np = np.frombuffer(y.encode('utf-32le'), dtype=np.uint32)
#     alphabet = np.union1d(query_np, database_np)
#     q = np.searchsorted(alphabet, query_np).astype(np.uint16) + 1
#     d = np.searchsorted(alphabet, database_np).astype(np.uint16) + 1
#     return extra.nw(q, d, match, mismatch, gap_open, gap_extend)[0]

def hirschberg_inner(x, y, match, mismatch, gap_open, gap_extend):
    lx, ly = len(x), len(y)
    if lx < 2 or ly < 2:
        return nw_full(x, y, match, mismatch, gap_open, gap_extend)

    f, fe = nw(x, y[:ly//2], match, mismatch, gap_open, gap_extend)
    s, se = nw(x, y[ly//2:], match, mismatch, gap_open, gap_extend, reverse=True)

    # return f, s
    j =  f[:-1] + s[:-1][::-1]
    k =  fe[:-1] + se[:-1][::-1] - gap_open
    # print(j)
    # print(k)
    # mid, mid2 = len(j) - j[::-1].argmax() - 1, len(k) - k[::-1].argmax() - 1
    mid, mid2 = j.argmax()+1, k.argmax()+1
    ns = np.array([j[mid-1], k[mid2-1], s[-1] + gap_open + (ly//2 - 1) * gap_extend, f[-1] + gap_open + (len(y) - ly//2 - 1) * gap_extend]).argmax()
    # print(mid, j[mid-1], mid2, k[mid2-1], ns)

    if ns == 1:
        split1 = hirschberg_inner(x[:mid2], y[:ly//2-1], match, mismatch, gap_open, gap_extend)
        split2 = hirschberg_inner(x[mid2:], y[ly//2+1:], match, mismatch, gap_open, gap_extend)
        return np.concatenate([split1, np.array([[mid2-1], [ly//2-1]]), np.array([[mid2-1], [ly//2]]), np.array([[mid2], [ly//2+1]]) + split2], axis=1)

    if ns == 2:
        mid = 0
    elif ns == 3:
        mid = len(x)

    split1 = hirschberg_inner(x[:mid], y[:ly//2], match, mismatch, gap_open, gap_extend)
    split2 = hirschberg_inner(x[mid:], y[ly//2:], match, mismatch, gap_open, gap_extend)
    return np.concatenate([split1, np.array([[mid], [ly//2]]) + split2], axis=1)

def chirschberg(query:str, database:str, match:int=1, mismatch:int=-1, gap_open:int=-1, gap_extend:int=-1, lastcol=lastcol):
    cdef cnp.ndarray[cnp.uint32_t] query_np = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32)
    cdef cnp.ndarray[cnp.uint32_t] database_np = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32)

    cdef cnp.ndarray[cnp.uint32_t] alphabet = np.union1d(query_np, database_np)
    if len(alphabet) > 2**16:
        raise Exception("Doesn't support alphabet sizes more than 2**16")

    cdef cnp.ndarray[cnp.uint16_t] q = np.searchsorted(alphabet, query_np).astype(np.uint16) + 1 # +1 For parasail, which uses \0 terminated strings
    cdef cnp.ndarray[cnp.uint16_t] d = np.searchsorted(alphabet, database_np).astype(np.uint16) + 1

    # return nw(q, d, match, mismatch, gap_open, gap_extend)
    return hirschberg_inner(q, d, match, mismatch, gap_open, gap_extend) #.T



