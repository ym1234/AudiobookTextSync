cimport cython
from libc.stdint cimport int64_t, uint32_t, int32_t
from posix cimport mman

import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "x86intrin.h":
    ctypedef int  __m256i

cdef extern from "impl32.h":
    void reset32(
        __m256i *, __m256i *, __m256i *, __m256i *,
        __m256i *,  __m256i *,
        uint32_t *,
        int32_t, int32_t,
        int64_t, int64_t
    )
    void sgcol32(
        __m256i *, uint32_t *,
        int64_t, int64_t,
        __m256i *, __m256i *, __m256i *, __m256i *,
        int32_t, int32_t, int32_t, int32_t
    )
    void sgtable32(
        __m256i *, uint32_t *,
        int64_t, int64_t,
        __m256i *, __m256i *, __m256i *,
        int32_t, int32_t, int32_t, int32_t
    )
    int32_t trace32(
        uint32_t *, uint32_t *,
        int64_t, int64_t, int64_t,
        int32_t *, int32_t *, int32_t *,
        int32_t, int32_t, int32_t, int32_t,
        int64_t *, int64_t *, int64_t, int64_t
    )
    int maxsum32(__m256i *, const __m256i *, int)

cdef inline int64_t _align(int64_t n, int64_t alignment) noexcept:
    # assert alignment > 1, "alignment < 0" # remove noexcept if you enable this
    cdef int64_t a = alignment - 1
    return (n + a) & ~a

cdef class BumpAllocator:
    cdef char *mem
    cdef int64_t size
    cdef int64_t cursor

    cdef int64_t[32] checkpoints
    cdef int checkpoint

    def __init__(self, size: int):
        cdef int64_t allocsize = _align(size,  2*1024*1024)
        cdef void *chunk = mman.mmap(NULL, allocsize, mman.PROT_READ | mman.PROT_WRITE, mman.MAP_ANONYMOUS | mman.MAP_PRIVATE | mman.MAP_POPULATE, -1, 0)
        if chunk == <void*> -1:
            raise MemoryError("mmap -1")

        self.mem = <char *>chunk
        self.cursor = 0
        self.size = allocsize

    cdef inline bint check(self, int64_t size) noexcept:
        return self.cursor + size <= self.size

    cdef inline bint isoverflow(self) noexcept:
        return self.cursor > self.size

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

    def __dealloc__(self):
        if mman.munmap(self.mem, self.size) < 0:
            raise MemoryError("munmap -1")

cdef class Aligner:
    cdef int match, mismatch, gap_open, gap_extend
    allocator: BumpAllocator

    def __init__(self, memsize: int = 1024*1024, match: int = 1, mismatch: int = -1, gap_open: int = -1, gap_extend: int = -1):
        self.match = match
        self.mismatch = mismatch
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.allocator = BumpAllocator(memsize)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def similarity(self, query: Union[str, np.ndarray], database: Union[str, np.ndarray]):
        cdef const cnp.uint32_t[::1] q = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32) if isinstance(query, str) else  query.astype(np.uint32)
        cdef const cnp.uint32_t[::1] d = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32) if isinstance(database, str) else database.astype(np.uint32)
        cdef int64_t lq = len(q)
        cdef int64_t ld = len(d)

        cdef int64_t aligned = _align(lq, 8)
        cdef int64_t stride = aligned // 8

        cdef int64_t bufsize = _align(aligned * sizeof(int32_t), 64)

        cdef __m256i *strided  = <__m256i *> self.allocator.alloc(bufsize)
        cdef __m256i *pvHLoad  = <__m256i *> self.allocator.alloc(bufsize)
        cdef __m256i *pvHStore = <__m256i *> self.allocator.alloc(bufsize)
        cdef __m256i *pvELoad  = <__m256i *> self.allocator.alloc(bufsize)
        cdef __m256i *pvEStore = <__m256i *> self.allocator.alloc(bufsize)

        if self.allocator.isoverflow():
            mem = float(self.allocator.cursor)
            self.allocator.clear()
            raise MemoryError(f"need at least {mem/1024**3:.2f} GiB(s)")

        reset32(pvHLoad, NULL, pvELoad, NULL, strided, NULL,  &q[0], self.gap_open, self.gap_extend, lq, stride)
        sgcol32(strided, &d[0],
                stride, ld,
                pvHLoad, pvHStore, pvELoad, pvEStore,
                self.match, self.mismatch, self.gap_open, self.gap_extend)

        cdef int32_t *ra = <int32_t *> (pvHStore if ld & 1 else pvHLoad)
        cdef int32_t r = ra[(lq-1) // stride + ((lq-1) % stride) * 8]
        self.allocator.clear()
        return r

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def align(self, query: Union[str, np.ndarray], database: Union[str, np.ndarray]):
        cdef const cnp.uint32_t[::1] q = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32) if isinstance(query, str) else  query.astype(np.uint32)
        cdef const cnp.uint32_t[::1] d = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32) if isinstance(database, str) else database.astype(np.uint32)
        cdef int64_t lq = len(q)
        cdef int64_t ld = len(d)

        cdef int64_t tracelen = 0
        cdef cnp.ndarray[cnp.int64_t] traceback = np.zeros(lq*2+ld*2, dtype=np.int64)

        cdef int64_t aligned = _align(lq, 8)
        cdef int64_t stride = aligned // 8

        cdef int64_t bufsize = _align(aligned*sizeof(int32_t), 64)
        cdef int64_t tablesize = _align((ld + 1)*aligned*sizeof(int32_t), 64)

        cdef __m256i *strided  = <__m256i *> self.allocator.alloc(bufsize)

        cdef __m256i *pvH = <__m256i *> self.allocator.alloc(tablesize)
        cdef __m256i *pvE = <__m256i *> self.allocator.alloc(tablesize)
        cdef __m256i *pvF = <__m256i *> self.allocator.alloc(tablesize)

        if self.allocator.isoverflow():
            mem = float(self.allocator.cursor)
            self.allocator.clear()
            raise MemoryError(f"need {mem/1024**3:.2f} GiB(s)")

        reset32(pvH, NULL, pvE, NULL, strided, NULL, &q[0], self.gap_open, self.gap_extend, lq, stride)
        sgtable32(strided, &d[0], stride, ld, pvH, pvE, pvF, self.match, self.mismatch, self.gap_open, self.gap_extend)

        cdef int32_t score = trace32(&q[0], &d[0], lq, aligned, ld,
                                     <int32_t *> pvH, <int32_t *> pvE, <int32_t *> pvF,
                                     self.match, self.mismatch, self.gap_open, self.gap_extend,
                                     <int64_t *> traceback.data, &tracelen, 0, 0)
        self.allocator.clear()
        return score, traceback[:tracelen].reshape(-1, 2)[::-1].T

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def hirschberg(self, query: Union[str, np.ndarray], database: Union[str, np.ndarray]):
        # print('\nHIRSCHBERG')
        cdef const cnp.uint32_t[::1] q = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32) if isinstance(query, str) else  query.astype(np.uint32)
        cdef const cnp.uint32_t[::1] d = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32) if isinstance(database, str) else database.astype(np.uint32)
        cdef int64_t lq = len(q)
        cdef int64_t ld = len(d)

        cdef int64_t tracelen = 0
        cdef cnp.ndarray[cnp.int64_t] traceback = np.zeros(lq*2 + ld*2, dtype=np.int64)

        cdef int64_t aligned = _align(lq, 8)
        cdef int64_t stride  = aligned // 8

        cdef int64_t  bufsize = _align(aligned*sizeof(int32_t), 64)
        cdef int64_t dbufsize = _align(ld*sizeof(int32_t), 64)

        if not self.allocator.check(k := 10*bufsize + dbufsize): # More understandable error messages
            raise MemoryError(f"need at least {k/1024**3:.2f} GiB(s)")

        cdef uint32_t[::1] dr = <uint32_t[:ld:1]> <uint32_t *> self.allocator.alloc(dbufsize)
        cdef __m256i *strided = <__m256i *> self.allocator.alloc(bufsize)

        dr[...] = d[::-1]

        # Tried to type this, didn't change anything
        cdef list nodes = [(0, lq, 0, ld, 0, 0)]

        cdef int32_t score = 0
        # Can't use cdefs here thats the dumbest shit ever
        while len(nodes):
            self.allocator.save()

            k = nodes.pop()
            sx: int64_t = k[0]
            lx: int64_t = k[1]
            sy: int64_t = k[2]
            ly: int64_t = k[3]
            # print(len(nodes), sx, lx, sy, ly)
            kind = k[4]

            if lx <= 0 or ly <= 0:
                while ly > 0:
                    traceback[tracelen] = sx + lx - 1
                    traceback[tracelen+1] = sy + ly - 1
                    tracelen += 2
                    ly -= 1
                while lx > 0:
                    traceback[tracelen] = sx + lx - 1
                    traceback[tracelen+1] = sy + ly - 1
                    tracelen += 2
                    lx -= 1
                if kind != 0:
                    # print("KIND1", tracelen)
                    traceback[tracelen] = sx - 1
                    traceback[tracelen+1] = sy - 1
                    traceback[tracelen+2] = sx - 1
                    traceback[tracelen+3] = sy - 2
                    tracelen += 4
                self.allocator.restore()
                continue

            aligned: int64_t = _align(lx, 8)
            stride: int64_t = aligned // 8

            bufsize: int64_t = _align(aligned*sizeof(int32_t), 64)

            tablesize: int64_t = _align((ly + 1)*aligned*sizeof(int32_t), 64)
            if self.allocator.check(3*tablesize):
                pvH: cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(tablesize)
                pvE: cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(tablesize)
                pvF: cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(tablesize)
                reset32(pvH, NULL, pvE, NULL, strided, NULL, &q[sx], self.gap_open, self.gap_extend, lx, stride)
                sgtable32(strided, &d[sy], stride, ly, pvH, pvE, pvF, self.match, self.mismatch, self.gap_open, self.gap_extend)
                s: int32_t = trace32(&q[sx], &d[sy], lx, aligned, ly,
                                     <int32_t *> pvH, <int32_t *> pvE, <int32_t *> pvF,
                                     self.match, self.mismatch, self.gap_open, self.gap_extend,
                                     <int64_t *> traceback.data, &tracelen, sx, sy)
                if kind != 0:
                    traceback[tracelen] = sx - 1
                    traceback[tracelen+1] = sy - 1
                    traceback[tracelen+2] = sx - 1
                    traceback[tracelen+3] = sy - 2
                    tracelen += 4
                score += s
                self.allocator.restore()
                continue

            rstrided:  cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(bufsize)

            pvHLoad:   cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(bufsize)
            pvHStore:  cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(bufsize)
            pvELoad:   cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(bufsize)
            pvEStore:  cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(bufsize)

            pvHLoad2:  cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(bufsize)
            pvHStore2: cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(bufsize)
            pvELoad2:  cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(bufsize)
            pvEStore2: cython.pointer(__m256i) = <__m256i *> self.allocator.alloc(bufsize)

            reset32(pvHLoad, pvHLoad2, pvELoad, pvELoad2, strided, rstrided, &q[sx], self.gap_open, self.gap_extend, lx, stride)

            ymid: int64_t = ly // 2

            sgcol32(strided, &d[sy],
                    stride, ymid,
                    pvHLoad, pvHStore, pvELoad, pvEStore,
                    self.match, self.mismatch, self.gap_open, self.gap_extend)

            vH1: cython.pointer(__m256i) = pvHStore if ymid & 1 else pvHLoad
            vE1: cython.pointer(__m256i) = pvELoad  if ymid & 1 else pvEStore

            sgcol32(rstrided, &dr[ld - sy - ly],
                    stride, ly - ymid,
                    pvHLoad2, pvHStore2, pvELoad2, pvEStore2,
                    self.match, self.mismatch, self.gap_open, self.gap_extend)

            vH2: cython.pointer(__m256i) = pvHStore2 if (ly - ymid) & 1 else pvHLoad2
            vE2: cython.pointer(__m256i) = pvELoad2  if (ly - ymid) & 1 else pvEStore2

            idx: int32_t = maxsum32(vH1, vH2, stride)
            idx2: int32_t = maxsum32(vE1, vE2, stride)

            H1: cython.pointer(int32_t) = <int32_t *> vH1
            H2: cython.pointer(int32_t) = <int32_t *> vH2
            E1: cython.pointer(int32_t) = <int32_t *> vE1

            # stupid boundary conditions
            H: int32_t = H1[idx]
            xmid: int64_t = idx / 8 + idx % 8 * stride + 1
            nkind = 0

            if H < (t := H2[aligned-1] + self.gap_open + (ymid-1)*self.gap_extend):
                # print("START")
                H = t
                xmid = 0
            if H < (t := H1[aligned-1] + self.gap_open + (ly-ymid-1)*self.gap_extend):
                # print("END")
                H = t
                xmid = lx
            # this code path is annoying to debug
            if H < E1[idx2] - self.gap_open:
                xmid = idx2 / 8 + idx2 % 8 * stride + 1
                nkind = 1

            xmid = min(xmid, lx)

            if nkind == 0:
                nodes.append((sx, xmid, sy, ymid, kind))
                nodes.append((sx+xmid, lx-xmid, sy+ymid, ly-ymid, 0))
            else:
                nodes.append((sx, xmid, sy, ymid-1, kind))
                nodes.append((sx+xmid, lx-xmid, sy+ymid+1, ly-ymid-1, 1))

            self.allocator.restore()

        self.allocator.clear()
        return score, traceback[:tracelen].reshape(-1, 2)[::-1].T
