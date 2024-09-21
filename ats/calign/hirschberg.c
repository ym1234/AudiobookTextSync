#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <x86intrin.h>

#include "hirschberg.h"
#include "impl32.h"

// TODO: https://arxiv.org/pdf/1909.00899
// _mm256_cvtepu16_epi32//_mm256_cvtepi16_epi32

/* int sum(int16_t *n1, int16_t *n2, int lx, int alx) { */
/*   int stride = alx / SIMD_ELEM; */
/*   int max = INT32_MIN; */
/*   int midx = 0; */
/*   for (int i = 0; i < lx-1; i++) { */
/*     int j = lx - i - 2; */
/*     int f =  n1[i / stride + i % stride * SIMD_ELEM]; */
/*     int s =  n2[j / stride + j % stride * SIMD_ELEM]; */
/*     int r = f + s; */
/*     if (r > max) { */
/*       max = r; */
/*       midx = i; */
/*     } */
/*   } */
/*   return midx; */
/* } */

// X and Y are both aligned
int32_t hirschberg_internal(
    uint32_t * restrict x, uint32_t * restrict y, uint32_t * restrict ry,
    size_t lx, size_t ly, Buffers *bufs,
    int32_t match, int32_t mismatch,
    int32_t gapopen, int32_t gapextend,
    int64_t *traceback, int64_t *tracelen
) {
  int64_t alx = align(lx, SIMD_ELEM32);
  *tracelen = alx;
  reset32(bufs, x, gapopen, gapextend, lx, alx);

  __m256i *n1 = sgcol32(bufs->vQuery, y, alx, ly/2, bufs->vHLoad, bufs->vHStore, bufs->vELoad, bufs->vEStore, match, mismatch, gapopen, gapextend);
  __m256i *n2 = sgcol32(bufs->vQueryR, ry, alx, ly - ly/2, bufs->vHLoad2, bufs->vHStore2, bufs->vELoad2, bufs->vEStore2, match, mismatch, gapopen, gapextend);

  int idx = maxsum32(n1, n2, alx);
  printf("%d\n", idx);

  int32_t *n1_32 = (int32_t *) n1;
  for (int i = 0; i < alx; i++) {
    traceback[i] = n1_32[i];
  }

}

int32_t hirschberg(
    uint32_t * restrict x, uint32_t * restrict y,

    int64_t lx, int64_t ly,
    int32_t match, int32_t mismatch,
    int32_t gapopen, int32_t gapextend,
    int64_t *traceback, int64_t *tracelen,
    int64_t memsize
) {
  int64_t alx = align(lx, 64);
  int64_t aly = align(ly, 64);
  int64_t minmem = alx * 10 + aly;

  int64_t allocsize = max(memsize, minmem) * sizeof(int32_t);
  void *buffers = mmap(0, allocsize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

  Buffers bufs = {};
  bufs.DatabaseR = buffers;
  bufs.Query = bufs.DatabaseR + aly;
  bufs.QueryR = bufs.Query + alx;

  bufs.HLoad = (int32_t *) bufs.QueryR + alx;
  bufs.allocsize = allocsize - (int64_t)((char *) bufs.HLoad - (char *) buffers);
  bufs.HLoad2 = bufs.HLoad + alx;
  bufs.HStore = bufs.HLoad2 + alx;
  bufs.HStore2 = bufs.HStore + alx;
  bufs.ELoad = bufs.HStore2 + alx;
  bufs.EStore = bufs.ELoad + alx;
  bufs.ELoad2 = bufs.EStore + alx;
  bufs.EStore2 = bufs.ELoad2 + alx;


  for (int i = 0; i < ly; i++) {
    bufs.DatabaseR[ly - i - 1] = y[i];
  }

  return hirschberg_internal(x, y, bufs.DatabaseR, lx, ly, &bufs, match, mismatch, gapopen, gapextend, traceback, tracelen);
}

