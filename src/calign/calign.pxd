from libc.stdint cimport uint16_t, int16_t #, size_t
from libc.stdlib import size_T
cdef extern from "calign.c":
    cdef struct AlignmentState:
        pass
    int hirschberg(
        uint16_t *x, uint16_t *y,
        size_t lx, size_t ly,
        int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
        AlignmentState *state
    );
    int semiglobal(AlignmentState *state);
