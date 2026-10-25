cimport cython
from libc.stdint cimport int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdio cimport printf
from libc.string cimport memcpy
import mmap

import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "simd.h":
    ctypedef int vec  # opaque here, only ever handled through pointers
    enum: LANES

cdef extern from "striped.h":
    void reset32(
        vec *, vec *, vec *, vec *,
        vec *,
        uint32_t *,
        int32_t, int32_t,
        int64_t, int64_t
    )
    void sgcol32(
        vec *, uint32_t *,
        int64_t, int64_t,
        vec *, vec *,
        int32_t, int32_t, int32_t, int32_t, int32_t
    )
    void sgtable32(
        vec *, uint32_t *,
        int64_t, int64_t,
        vec *, vec *, vec *,
        int32_t, int32_t, int32_t, int32_t, int32_t
    )
    int32_t trace32(
        uint32_t *, uint32_t *,
        int64_t, int64_t, int64_t,
        int32_t *, int32_t *, int32_t *,
        int32_t, int32_t, int32_t, int32_t, int32_t, int,
        int64_t *, int64_t *, int64_t, int64_t
    )
    void reverse_query32(const vec *, vec *, int64_t, int64_t)
    int pairmax32(const vec *, const vec *, vec *, int64_t, int64_t, int32_t)


cdef extern from "diffrec.h":
    int64_t do_the_thing(
        uint8_t *qseq, int64_t unaligned, int64_t qlen,
        uint8_t *tseq, int64_t tlen,
        int8_t match, int8_t mismatch,
        int8_t gapo, int8_t gape
    )
    enum: HAVE_DIFFREC

cdef inline void _push(int64_t *t, int64_t *n, int64_t x, int64_t y) noexcept:
    # append a path point unless it repeats the last one (junctions between nodes are shared)
    if n[0] >= 2 and t[n[0] - 2] == x and t[n[0] - 1] == y:
        return
    t[n[0]] = x
    t[n[0] + 1] = y
    n[0] += 2

cdef inline int64_t _align(int64_t n, int64_t alignment) noexcept:
    # assert alignment > 1, "alignment < 0" # remove noexcept if you enable this
    cdef int64_t a = alignment - 1
    return (n + a) & ~a

# the striped layout: position p lives in vector p % stride, lane p / stride
cdef inline int64_t _striped(int64_t p, int64_t stride) noexcept:
    return (p % stride) * LANES + p // stride

cdef inline int64_t _unstriped(int64_t i, int64_t stride) noexcept:
    return (i % LANES) * stride + i // LANES

cdef class BumpAllocator:
    cdef char *mem
    file: mmap.mmap
    cdef int64_t allocsize
    cdef int64_t cursor

    cdef int64_t[32] checkpoints
    cdef int checkpoint

    def __init__(self, size: int):
        cdef int64_t allocsize = _align(size,  2*1024*1024)
        self.file = mmap.mmap(-1, allocsize)
        cdef char[:] buf = self.file
        self.mem = &buf[0]

        self.cursor = 0
        self.allocsize = allocsize

    cdef inline bint check(self, int64_t size) noexcept:
        return self.cursor + size <= self.allocsize

    cdef inline int64_t size(self) noexcept:
        return self.cursor - self.checkpoints[self.checkpoint-1]

    cdef inline bint isoverflow(self) noexcept:
        return self.cursor > self.allocsize

    cdef inline void save(self) noexcept:
        self.checkpoints[self.checkpoint] = self.cursor
        self.checkpoint += 1

    cdef inline void restore(self) noexcept:
        self.checkpoint -= 1
        self.cursor = self.checkpoints[self.checkpoint]

    cdef inline char *alloc(self, int64_t size) noexcept:
        ret = &self.mem[self.cursor]
        self.cursor += size
        return ret

    cdef inline void clear(self) noexcept:
        self.cursor = 0
        self.checkpoint = 0

    # def size(self):
    #     return self.cursor

    def __enter__(self):
        self.save()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

    def close(self):
        self.mem = NULL
        self.file.close()

cdef class Aligner:
    cdef int match, mismatch, gap_open, gap_extend
    cdef public long long max_cells
    allocator: BumpAllocator

    def __init__(self, memsize: int = 1024*1024, match: int = 1, mismatch: int = -1, gap_open: int = -1, gap_extend: int = -1):
        self.max_cells = 1 << 62   # test hook: split any node with more cells than this, even if it fits in memory
        self.match = match
        self.mismatch = mismatch
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.allocator = BumpAllocator(memsize)


    def score_difference_recurrence(self, query, database):
        if not HAVE_DIFFREC:
            raise NotImplementedError("score_difference_recurrence is only written for avx2")
        # XXX: Don't ferget you're using utf-8 here you idiot
        cdef const cnp.uint8_t[::1] q = np.frombuffer(query.encode('utf-8'), dtype=np.uint8) if isinstance(query, str) else  query.astype(np.uint8)
        cdef const cnp.uint8_t[::1] d = np.frombuffer(database.encode('utf-8'), dtype=np.uint8) if isinstance(database, str) else database.astype(np.uint8)
        cdef int64_t lq = len(q)
        cdef int64_t ld = len(d)

        cdef int64_t aligned = _align(lq, 32)
        cdef int64_t stride = aligned // 32

        cdef int64_t bufsize = _align(aligned * sizeof(int8_t), 64)
        with self.allocator as allocator:
            buf: cython.pointer(uint8_t) = <uint8_t *> self.allocator.alloc(bufsize)
            memcpy(buf, &q[0], sizeof(int8_t) * lq)
            ret: int64_t = do_the_thing(buf, lq, aligned, &d[0], ld,  self.match, self.mismatch, self.gap_open, self.gap_extend)
            return ret

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def similarity(self, query: Union[str, np.ndarray], database: Union[str, np.ndarray]):
        cdef const cnp.uint32_t[::1] q = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32) if isinstance(query, str) else  query.astype(np.uint32)
        cdef const cnp.uint32_t[::1] d = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32) if isinstance(database, str) else database.astype(np.uint32)
        cdef int64_t lq = len(q)
        cdef int64_t ld = len(d)

        cdef int64_t aligned = _align(lq, LANES)
        cdef int64_t stride = aligned // LANES

        cdef int64_t bufsize = _align(aligned * sizeof(int32_t), 64)

        with self.allocator as allocator:
            strided : cython.pointer(vec) = <vec *> self.allocator.alloc(bufsize)
            pvH     : cython.pointer(vec) = <vec *> self.allocator.alloc(bufsize)
            pvE     : cython.pointer(vec) = <vec *> self.allocator.alloc(bufsize)

            if self.allocator.isoverflow(): raise MemoryError(f"need at least {float(allocator.size())/1024**3:.2f} GiB(s)")

            reset32(pvH, NULL, pvE, NULL, strided, &q[0], self.gap_open, self.gap_extend, lq, stride)
            sgcol32(strided, &d[0], stride, ld, pvH, pvE,
                    self.match, self.mismatch, self.gap_open, self.gap_extend, self.gap_open)

            ra: cython.pointer(int32_t) = <int32_t *> pvH
            r: int32_t = ra[_striped(lq - 1, stride)]
            return r

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def align(self, query: Union[str, np.ndarray], database: Union[str, np.ndarray]):
        cdef const cnp.uint32_t[::1] q = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32) if isinstance(query, str) else  query.astype(np.uint32)
        cdef const cnp.uint32_t[::1] d = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32) if isinstance(database, str) else database.astype(np.uint32)
        cdef int64_t lq = len(q)
        cdef int64_t ld = len(d)

        cdef int64_t tracelen = 0
        cdef cnp.ndarray[cnp.int64_t] traceback = np.zeros(2*(lq + ld + 1), dtype=np.int64)

        cdef int64_t aligned = _align(lq, LANES)
        cdef int64_t stride = aligned // LANES

        cdef int64_t bufsize = _align(aligned*sizeof(int32_t), 64)
        cdef int64_t tablesize = _align((ld + 1)*aligned*sizeof(int32_t), 64)

        with self.allocator as allocator:
            strided: cython.pointer(vec)  = <vec *> self.allocator.alloc(bufsize)

            pvH: cython.pointer(vec) = <vec *> self.allocator.alloc(tablesize)
            pvE: cython.pointer(vec) = <vec *> self.allocator.alloc(tablesize)
            pvF: cython.pointer(vec) = <vec *> self.allocator.alloc(tablesize)

            if self.allocator.isoverflow(): raise MemoryError(f"need {float(self.allocator.size())/1024**3:.2f} GiB(s)")

            reset32(pvH, NULL, pvE, NULL, strided, &q[0], self.gap_open, self.gap_extend, lq, stride)
            sgtable32(strided, &d[0], stride, ld, pvH, pvE, pvF, self.match, self.mismatch, self.gap_open, self.gap_extend, self.gap_open)

            score: int32_t = trace32(&q[0], &d[0], lq, aligned, ld,
                                        <int32_t *> pvH, <int32_t *> pvE, <int32_t *> pvF,
                                        self.match, self.mismatch, self.gap_open, self.gap_extend, self.gap_open, 0,
                                        <int64_t *> traceback.data, &tracelen, 0, 0)
            return score, traceback[:tracelen].reshape(-1, 2)[::-1].T

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def hirschberg(self, query: Union[str, np.ndarray], database: Union[str, np.ndarray]):
        cdef const cnp.uint32_t[::1] q = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32) if isinstance(query, str) else  query.astype(np.uint32)
        cdef const cnp.uint32_t[::1] d = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32) if isinstance(database, str) else database.astype(np.uint32)
        cdef int64_t lq = len(q)
        cdef int64_t ld = len(d)

        cdef int64_t tracelen = 0
        cdef cnp.ndarray[cnp.int64_t] traceback = np.zeros(2*(lq + ld + 1), dtype=np.int64)   # a path has at most lq + ld + 1 points

        cdef int64_t aligned = _align(lq, LANES)
        cdef int64_t stride  = aligned // LANES

        cdef int64_t  bufsize = _align(aligned*sizeof(int32_t), 64)
        cdef int64_t dbufsize = _align(ld*sizeof(int32_t), 64)

        if not self.allocator.check(k := 7*bufsize + dbufsize): # More understandable error messages
            raise MemoryError(f"need at least {k/1024**3:.2f} GiB(s)")

        cdef uint32_t[::1] dr = <uint32_t[:ld:1]> <uint32_t *> self.allocator.alloc(dbufsize)
        cdef vec *strided = <vec *> self.allocator.alloc(bufsize)

        dr[...] = d[::-1]

        cdef list nodes = [(0, lq, 0, ld, 0, 0)]

        cdef int64_t score = 0 if lq == ld == 0 else self.gap_open + (max(lq, ld) - 1) * self.gap_extend if lq == 0 or ld == 0 else 0
        root = True
        # Can't use cdefs here thats the dumbest shit ever
        while len(nodes):
            self.allocator.save()

            k = nodes.pop()
            sx: int64_t = k[0]
            lx: int64_t = k[1]
            sy: int64_t = k[2]
            ly: int64_t = k[3]
            tb, te = k[4], k[5]
            open_tb = self.gap_extend if tb else self.gap_open
            open_te = self.gap_extend if te else self.gap_open

            if lx <= 0 or ly <= 0:
                while ly > 0:
                    _push(<int64_t *> traceback.data, &tracelen, sx + lx, sy + ly)
                    ly -= 1
                while lx > 0:
                    _push(<int64_t *> traceback.data, &tracelen, sx + lx, sy + ly)
                    lx -= 1
                _push(<int64_t *> traceback.data, &tracelen, sx, sy)
                if tb:                            # the two carved arrows of a gap-crossing split
                    _push(<int64_t *> traceback.data, &tracelen, sx, sy - 1)
                    _push(<int64_t *> traceback.data, &tracelen, sx, sy - 2)
                self.allocator.restore()
                continue

            aligned = _align(lx, LANES)
            stride = aligned // LANES

            bufsize = _align(aligned*sizeof(int32_t), 64)

            tablesize: int64_t = _align((ly + 1)*aligned*sizeof(int32_t), 64)
            if self.allocator.check(3*tablesize) and (lx * ly <= self.max_cells or ly < 2):
                pvH: cython.pointer(vec) = <vec *> self.allocator.alloc(tablesize)
                pvE: cython.pointer(vec) = <vec *> self.allocator.alloc(tablesize)
                pvF: cython.pointer(vec) = <vec *> self.allocator.alloc(tablesize)
                reset32(pvH, NULL, pvE, NULL, strided, &q[sx], self.gap_open, self.gap_extend, lx, stride)
                sgtable32(strided, &d[sy], stride, ly, pvH, pvE, pvF, self.match, self.mismatch, self.gap_open, self.gap_extend, open_tb)
                s: int32_t = trace32(&q[sx], &d[sy], lx, aligned, ly,
                                     <int32_t *> pvH, <int32_t *> pvE, <int32_t *> pvF,
                                     self.match, self.mismatch, self.gap_open, self.gap_extend, open_tb, te,
                                     <int64_t *> traceback.data, &tracelen, sx, sy)
                if tb:                            # the two carved arrows of a gap-crossing split
                    _push(<int64_t *> traceback.data, &tracelen, sx, sy - 1)
                    _push(<int64_t *> traceback.data, &tracelen, sx, sy - 2)
                if root: score = s
                root = False
                self.allocator.restore()
                continue

            rstrided:  cython.pointer(vec) = <vec *> self.allocator.alloc(bufsize)

            # the forward pass's rows and the reverse pass's, each worked on in place
            vH1: cython.pointer(vec) = <vec *> self.allocator.alloc(bufsize)
            vE1: cython.pointer(vec) = <vec *> self.allocator.alloc(bufsize)
            vH2: cython.pointer(vec) = <vec *> self.allocator.alloc(bufsize)
            vE2: cython.pointer(vec) = <vec *> self.allocator.alloc(bufsize)

            reset32(vH1, vH2, vE1, vE2, strided, &q[sx], self.gap_open, self.gap_extend, lx, stride)
            reverse_query32(strided, rstrided, stride, aligned - lx)

            ymid: int64_t = ly // 2

            # vH1/vE1 end up holding row ymid: its scores, and the vertical gaps that end in it
            sgcol32(strided, &d[sy], stride, ymid, vH1, vE1,
                    self.match, self.mismatch, self.gap_open, self.gap_extend, open_tb)

            sgcol32(rstrided, &dr[ld - sy - ly], stride, ly - ymid, vH2, vE2,
                    self.match, self.mismatch, self.gap_open, self.gap_extend, open_te)

            H1: cython.pointer(int32_t) = <int32_t *> vH1
            E1: cython.pointer(int32_t) = <int32_t *> vE1
            H2: cython.pointer(int32_t) = <int32_t *> vH2
            E2: cython.pointer(int32_t) = <int32_t *> vE2
            vsums: cython.pointer(vec) = <vec *> self.allocator.alloc(bufsize)
            sums: cython.pointer(int32_t) = <int32_t *> vsums
            below = ly - ymid
            down_left = 0 if ymid == 0 else open_tb + (ymid - 1) * self.gap_extend
            down_right = 0 if below == 0 else open_te + (below - 1) * self.gap_extend
            NEGV = -(1 << 40)
            last = _striped(lx - 1, stride)

            best_cross = pairmax32(vH1, vH2, vsums, stride, aligned - lx, 0)
            cross = sums[best_cross] if lx >= 2 else NEGV; x_cross = _unstriped(best_cross, stride) + 1
            for cand, cx in ((down_left + H2[last], 0), (H1[last] + down_right, lx)):  # x = 0 and x = lx
                if cand > cross: cross = cand; x_cross = cx

            best_gap = pairmax32(vE1, vE2, vsums, stride, aligned - lx, self.gap_extend - self.gap_open)
            gap = sums[best_gap] if lx >= 2 else NEGV; x_gap = _unstriped(best_gap, stride) + 1
            dd0 = down_left if ymid >= 1 else NEGV
            ssl = down_right if below >= 1 else NEGV
            for cand, cx in ((dd0 + E2[last] - self.gap_open + self.gap_extend, 0),
                             (E1[last] + ssl - self.gap_open + self.gap_extend, lx)):
                if cand > gap: gap = cand; x_gap = cx

            in_gap = gap > cross
            xmid, H = (x_gap, gap) if in_gap else (x_cross, cross)
            if root: score = H
            root = False
            xmid = min(xmid, lx)

            if not in_gap:
                nodes.append((sx, xmid, sy, ymid, tb, 0))
                nodes.append((sx+xmid, lx-xmid, sy+ymid, ly-ymid, 0, te))
            else:
                nodes.append((sx, xmid, sy, ymid-1, tb, 1))
                nodes.append((sx+xmid, lx-xmid, sy+ymid+1, ly-ymid-1, 1, te))

            self.allocator.restore()

        self.allocator.clear()
        return score, traceback[:tracelen].reshape(-1, 2)[::-1].T
