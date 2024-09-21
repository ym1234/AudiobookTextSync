from libc.stdint cimport int64_t, uint16_t, int16_t, uint32_t, int32_t
from libc.stddef cimport size_t
from libc.stdio cimport printf

from cython.parallel import threadid, parallel, prange
import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "hirschberg.c":
        int32_t hirschberg(
            uint32_t * x, uint32_t * y,
            int64_t lx, int64_t ly,
            int32_t match, int32_t mismatch,
            int32_t gapopen, int32_t gapextend,
            int64_t *traceback, int64_t *tracelen,
            int64_t memsize
        )

def pyhirschberg(query: str, database: str, match: int = 1, mismatch: int = -1, gap_open: int = -1, gap_extend: int = -1):
    # le very important otherwise you get the BOM or w/e
    cdef cnp.ndarray[cnp.uint32_t] q = np.frombuffer(query.encode('utf-32le'), dtype=np.uint32)
    cdef cnp.ndarray[cnp.uint32_t] d = np.frombuffer(database.encode('utf-32le'), dtype=np.uint32)

    cdef cnp.ndarray[cnp.int64_t] traceback = np.zeros(len(q) + len(d), dtype=np.int64)
    tracelen: int64_t = 0
    score = hirschberg(<cnp.uint32_t *> q.data, <cnp.uint32_t *> d.data,
                        len(q), len(d), match, mismatch, gap_open, gap_extend,
                        <int64_t *> traceback.data, &tracelen, 1*(1024**3))
    return traceback[:tracelen], score


