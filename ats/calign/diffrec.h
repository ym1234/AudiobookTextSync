#ifndef __DIFFREC_H__
#define __DIFFREC_H__
/* the uint8 difference recurrence (score_difference_recurrence) only exists for avx2 so far */
#if defined(__AVX2__)
#include "avx2.c"
#define HAVE_DIFFREC 1
#else
#include <stdint.h>
#define HAVE_DIFFREC 0
static int64_t do_the_thing(uint8_t *qseq, int64_t unaligned, int64_t qlen, uint8_t *tseq, int64_t tlen,
                            int8_t match, int8_t mismatch, int8_t gapo, int8_t gape) { return 0; }
#endif
#endif
