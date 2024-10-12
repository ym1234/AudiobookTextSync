#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <x86intrin.h>
#include <errno.h>

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

/* // TODO turn this into while (for?) loop */
/* int32_t hirschberg_internal( */
/*     uint32_t * restrict x, uint32_t * restrict y, uint32_t * restrict ry, */
/*     int64_t lx, int64_t ly, Buffers *bufs, */
/*     int32_t match, int32_t mismatch, */
/*     int32_t gapopen, int32_t gapextend, */
/*     int64_t *traceback, int64_t *tracelen, */
/*     int64_t offsetx, int64_t offsety */
/* ) { */
/*   if (!lx) { */
/*     while (ly--) { */
/*       traceback[(*tracelen)++] = offsetx + lx - 1; */
/*       traceback[(*tracelen)++] = offsety + ly - 1; */
/*     } */
/*   } */
/*   if (!ly) { */
/*     while (lx--) { */
/*       traceback[(*tracelen)++] = offsetx + lx - 1; */
/*       traceback[(*tracelen)++] = offsety + ly - 1; */
/*     } */
/*   } */

/*   int64_t alx = align(lx, SIMD_ELEM32); */
/*   int stride = alx / SIMD_ELEM32; */
/*   int64_t fullmem = 3*(ly + 1)*alx*sizeof(int32_t); */
/*   reset32(bufs, x, gapopen, gapextend, lx, stride); */
/*   if (fullmem < bufs->allocsize) { */
/*     __m256i *pvH = bufs->vHLoad; */
/*     __m256i *pvE = pvH + (ly + 1) * stride; */
/*     __m256i *pvF = pvE + (ly + 1) * stride; */
/*     __m256i vGapOpen = _mm256_set1_epi32(gapopen); */
/*     for (int i = 0; i < alx/SIMD_ELEM32; i++) { */
/*       pvE[i] = _mm256_add_epi32(pvH[i], vGapOpen); */
/*     } */
/*     sgtable32(bufs->vQuery, y, stride, ly, pvH, pvE, pvF, match, mismatch, gapopen, gapextend); */
/*     return trace32(x, y, lx, alx, ly, */
/*         (int32_t *) pvH, (int32_t *) pvE, (int32_t *) pvF, */
/*         match, mismatch, gapopen, gapextend, */
/*         traceback, tracelen, offsetx, offsety); */
/*   } */

/*   Tuple n1 = sgcol32(bufs->vQuery, y, stride, ly/2, bufs->vHLoad, bufs->vHStore, bufs->vELoad, bufs->vEStore, match, mismatch, gapopen, gapextend); */
/*   Tuple n2 = sgcol32(bufs->vQueryR, ry, stride, ly - ly/2, bufs->vHLoad2, bufs->vHStore2, bufs->vELoad2, bufs->vEStore2, match, mismatch, gapopen, gapextend); */

/*   n1.H[alx-1] += gapopen + (ly - ly/2 - 2) * gapextend; */
/*   int second = n2.H[alx-1] + gapopen + (ly/2 - 2) * gapextend; */

/*   int idx = maxsum32(n1.vH, n2.vH, stride); */
/*   int idx2 = maxsum32(n1.vE, n2.vE, stride); */

/*   printf("%d, %d, %d, %d\n", n1.E[idx2] + gapopen, n1.H[idx], second, n2.H[alx-1]); */
/*   // This shit is so confusing idk if it's right */
/*   if ((n1.E[idx2] + gapopen) >= n1.H[idx] && (n1.E[idx2] + gapopen) >= second) { */
/*     int real2 = idx2 / SIMD_ELEM32 + idx2 % SIMD_ELEM32 * stride + 1; */
/*     if (real2 > lx) { */
/*       real2 = lx; */
/*     } */
/*     printf("\n\n\nreal2, %d\n\n\n\n", real2); */
/*     hirschberg_internal(x+real2, y+ly/2+1, ry, lx-real2, ly-ly/2-1, bufs, match, mismatch, gapopen, gapextend, traceback, tracelen, offsetx+real2, offsety+ly/2+1); */
/*     traceback[(*tracelen)++] = offsetx + real2 - 1; */
/*     traceback[(*tracelen)++] = offsety + ly/2; */
/*     traceback[(*tracelen)++] = offsetx + real2 - 1; */
/*     traceback[(*tracelen)++] = offsety + ly/2 - 1; */
/*     hirschberg_internal(x, y, ry + ly - ly/2 + 1, real2, ly/2 - 1, bufs, match, mismatch, gapopen, gapextend, traceback, tracelen, offsetx, offsety); */
/*     return 0; */
/*   } */

/*   int real = second >= n1.H[idx] ? 0 : (idx / SIMD_ELEM32 + idx % SIMD_ELEM32 * stride + 1); */
/*   if (real > lx) { // How should padding be accounted for? */
/*     real = lx; */
/*   } */
/*   printf("\n\n\nreal, %d\n\n\n\n", real); */
/*   hirschberg_internal(x+real, y+ly/2, ry, lx-real, ly-ly/2, bufs, match, mismatch, gapopen, gapextend, traceback, tracelen, offsetx+real, offsety+ly/2); */
/*   hirschberg_internal(x, y, ry + ly - ly/2, real, ly/2, bufs, match, mismatch, gapopen, gapextend, traceback, tracelen, offsetx, offsety); */
/*   return 0; */
/* } */

/* int32_t hirschberg( */
/*     uint32_t * restrict x, uint32_t * restrict y, */
/*     int64_t lx, int64_t ly, */
/*     int32_t match, int32_t mismatch, */
/*     int32_t gapopen, int32_t gapextend, */
/*     int64_t *traceback, int64_t *tracelen, */
/*     int64_t memsize */
/* ) { */
/*   int64_t alx = align(lx, 64); */
/*   int64_t aly = align(ly, 64); */
/*   int64_t minmem = alx * 10 + aly; */

/*   int64_t allocsize = align(max(memsize, (int64_t)(minmem * sizeof(int32_t))), 2*1024*1024); */
/*   char *buffers = (char *)mmap(0, allocsize, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_POPULATE, -1, 0); */
/*   if (buffers == ((void*) -1)) { */
/*     printf("\n\nerrno str %s, %ld\n\n", strerror(errno), allocsize); */
/*   } */

/*   Buffers bufs = {}; */
/*   bufs.DatabaseR = (uint32_t *) buffers; */
/*   bufs.Query = bufs.DatabaseR + aly; */
/*   bufs.QueryR = bufs.Query + alx; */

/*   bufs.HLoad = (int32_t *) bufs.QueryR + alx; */
/*   bufs.allocsize = allocsize - (int64_t)((char *) bufs.HLoad -  buffers); */
/*   bufs.HLoad2 = bufs.HLoad + alx; */
/*   bufs.HStore = bufs.HLoad2 + alx; */
/*   bufs.HStore2 = bufs.HStore + alx; */
/*   bufs.ELoad = bufs.HStore2 + alx; */
/*   bufs.EStore = bufs.ELoad + alx; */
/*   bufs.ELoad2 = bufs.EStore + alx; */
/*   bufs.EStore2 = bufs.ELoad2 + alx; */


/*   for (int i = 0; i < ly; i++) { */
/*     bufs.DatabaseR[ly - i - 1] = y[i]; */
/*   } */
/*   int32_t score = hirschberg_internal(x, y, bufs.DatabaseR, lx, ly, &bufs, match, mismatch, gapopen, gapextend, traceback, tracelen, 0, 0); */
/*   munmap(buffers, allocsize); */
/*   return score; */
/* } */

