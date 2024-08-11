#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <x86intrin.h>

#include "hirschberg.h"
#include "utils.h"

#include "semiglobal.c"
/* #include "nw.c" */

// TODO: https://arxiv.org/pdf/1909.00899
// _mm256_cvtepu16_epi32//_mm256_cvtepi16_epi32

// This was the easiest one to modify, checkout the other algorithms later?
// https://en.algorithmica.org/hpc/algorithms/argmin
int argmax(__m256i *a, int n) {
    int16_t max = INT16_MIN, idx = 0;
    __m256i p = _mm256_set1_epi32(max);
    for (int i = 0; i < n; i += 16) {
        __m256i mask = _mm256_cmpgt_epi16(a[i/16], p);
        if (unlikely(!_mm256_testz_si256(mask, mask))) {
            for (int j = i; j < i + 16; j++) {
                int16_t *b = (int16_t *) a;
                if (b[j] > max) {
                    max = b[j];
                    idx = j;
                }
            }
            p = _mm256_set1_epi16(max);
        }
    }

    return idx;
}

int maxsum(__m256i *n1, __m256i *n2, int lx, int alx) {
  int stride = alx / SIMD_ELEM;
  for (int i = 0; i < stride-1; i++) {
    int j = stride - i - 2;
    n1[i] = _mm256_adds_epi16(n1[i], rev256(n2[j]));
  }
  n1[stride-1] = _mm256_adds_epi16(n1[stride-1], _mm256_srli_si256_rpl(rev256(n2[stride-1]), 2));
  /* for (int i = 0; i < stride; i++) { */
  /*   print256_num("Maxes: ", n1[i]); */
  /* } */
  int idx = argmax(n1, alx);
  /* int16_t *w = (int16_t *) n1; */
  /* printf("%d\n", w[idx]); */
  return idx / SIMD_ELEM + idx % SIMD_ELEM * stride;
}

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
void hirschberg_internal(
    uint16_t * restrict x, uint16_t * restrict y, uint16_t * restrict ry,
    size_t lx, size_t ly, size_t alx, size_t aly,
    int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
    AlignmentState *state, Result *result, size_t alloc_size
) {

  stride_seq((uint16_t *) state->query, x, lx, alx);
  reverse16((uint16_t *) state->query_reversed, (uint16_t *) state->query, alx);

  AlignmentParams params = {
    .vHMin = state->vHMin,
    .pvHLoad = state->pvHLoad,
    .pvHStore = state->pvHStore,

    .vEMin = state->vEMin,
    .pvE = state->pvE,

    .query = state->query,
    .query_len = alx,

    .database = y,
    .database_len = ly/2,
  };

  sgcol(&params, match, mismatch, gapopen, gapextend);
  /* int16_t *n1 = (int16_t *) params.pvHLoad; */
  __m256i *n1 =  params.pvHLoad;

  if (n1 == state->pvHStore) {
    params.pvHStore = state->pvHLoad2;
    params.pvHLoad = state->pvHLoad;
  } else {
    params.pvHLoad = state->pvHLoad2;
  }
  params.query = state->query_reversed;
  params.database = ry;
  params.database_len = ly - params.database_len;

  sgcol(&params, match, mismatch, gapopen, gapextend);
  /* int16_t *n2 = (int16_t *) params.pvHLoad; */
  __m256i *n2 = params.pvHLoad;

  printf("%d\n", maxsum(n1, n2, lx, alx));
}

Result hirschberg(
    uint16_t * restrict x, uint16_t * restrict y,
    size_t lx, size_t ly, size_t alx, size_t aly,
    int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
    size_t *traceback
) {
  Result r = {
    .traceback = traceback,
    .ltrace = 0,
    .score = 0
  };

  int bsize = alx * sizeof(uint32_t);
  int bstride = bsize/SIMD_WIDTH_BYTES;
  /* printf("Size %lx, stride %lx\n", bsize, bstride); */

  size_t constants_size = align(3*bsize + aly * sizeof(uint16_t), 4096);
  void *constants = mmap(0, constants_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  size_t buffers_size = align(7*bsize, 4096);
  void *buffers = mmap(0, buffers_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

  AlignmentState state = {0};
  state.vHMin = constants;
  state.vEMin = state.vHMin + bstride;
  state.database_reversed = (uint16_t *) (state.vEMin + bstride);

  state.pvHLoad = buffers;
  state.pvHStore = state.pvHLoad + bstride;
  state.pvHLoad2 = state.pvHStore + bstride;

  state.pvE = state.pvHLoad2 + bstride;
  state.pvE2 = state.pvE + bstride;

  state.query = state.pvE2 + bstride;
  state.query_reversed = state.query + bstride;


  // Global
  {
    int16_t J[16];
    for (int i = 0; i < 16; i++) {
      J[i] = (int16_t) (i * bstride * gapextend < INT16_MIN ? INT16_MIN : i * bstride * gapextend);
    }
    __m256i T = _mm256_load_si256((__m256i *) J);
    __m256i vGapO = _mm256_set1_epi16(gapopen);
    for (int i = 0; i < bstride; i++) {
      __m256i J =_mm256_adds_epi16(_mm256_set1_epi16(i * gapextend < INT16_MIN ? INT16_MIN : i * gapextend), T);
      state.vHMin[i] = _mm256_adds_epi16(J, vGapO);
      state.vEMin[i] = _mm256_adds_epi16(state.vHMin[i], vGapO);
    }
  }

  // Semiglobal
  // __m256i Min = _mm256_set1_epi16(gapopen);
  // for (int i = 0; i < bstride; i++) {
  //   state.vEMin[i] = Min;
  // }

  reverse16(state.database_reversed, y, ly);
  hirschberg_internal(x, y, state.database_reversed,
      lx, ly, alx, aly,
      match, mismatch, gapopen, gapextend,
      &state, &r, buffers_size);

  munmap(constants, constants_size);
  munmap(buffers, buffers_size);
  return r;
}

int main() {
  (void) hirschberg;
  (void) sgcol;
}
