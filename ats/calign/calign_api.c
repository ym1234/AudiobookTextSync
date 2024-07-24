#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <x86intrin.h>

#include "calign_api.h"
#include "utils.h"

// TODO: https://arxiv.org/pdf/1909.00899
// _mm256_cvtepu16_epi32//_mm256_cvtepi16_epi32

void semiglobal(AlignmentParams *state, int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend) {
  __m256i vMatch = _mm256_set1_epi16(match);
  __m256i vMismatch = _mm256_set1_epi16(mismatch);
  __m256i vGapO = _mm256_set1_epi16(gapopen);
  __m256i vGapE = _mm256_set1_epi16(gapextend);

  __m256i * pvELoad = state->vEMin;
  __m256i * pvEStore = state->pvE;

  __m256i * restrict pvHLoad = state->vHMin;
  __m256i * restrict pvHStore = state->pvHStore;

  size_t stride = state->query_len / SIMD_ELEM;
  for (int i = 0; i < state->database_len; i++) {
    __m256i vY = _mm256_set1_epi16(state->database[i]);
    __m256i vF = _mm256_set1_epi16(INT16_MIN);
    vF = _mm256_insert_epi16(vF, 2*gapopen + i*gapextend, 0); // Does this get optimized out?
    /* __m256i vF = _mm256_set_epi16( */
    /*     INT16_MIN, INT16_MIN, INT16_MIN, INT16_MIN, */
    /*     INT16_MIN, INT16_MIN, INT16_MIN, INT16_MIN, */
    /*     INT16_MIN, INT16_MIN, INT16_MIN, INT16_MIN, */
    /*     INT16_MIN, INT16_MIN, INT16_MIN, 2*gapopen + i*gapextend); */

    __m256i vH = _mm256_slli_si256_rpl(pvHLoad[stride - 1], 2);
    vH = _mm256_insert_epi16(vH, gapopen + (i-1)*gapextend, 0);
    if (unlikely(!i)) {
      vH = _mm256_insert_epi16(vH, 0, 0);
    }

    for (int j = 0; j < stride; j++) {
      __m256i vX = _mm256_load_si256(state->query + j);
      __m256i vCmp = _mm256_cmpeq_epi16(vY, vX);
      __m256i MatchScore = _mm256_and_si256(vCmp, vMatch);
      __m256i MismatchScore =_mm256_andnot_si256(vCmp, vMismatch);
      __m256i vScore = _mm256_or_si256(MatchScore, MismatchScore);

      vH = _mm256_adds_epi16(vH, vScore);

      __m256i vE = pvELoad[j];

      vH = _mm256_max_epi16(vH, vE);
      vH = _mm256_max_epi16(vH, vF);
      pvHStore[j] = vH;

      vH = _mm256_adds_epi16(vH, vGapO);

      vE = _mm256_adds_epi16(vE, vGapE);
      vE = _mm256_max_epi16(vE, vH);
      pvEStore[j] = vE;

      vF = _mm256_adds_epi16(vF, vGapE);
      vF = _mm256_max_epi16(vF, vH);

      vH = pvHLoad[j];
    }

    for (int k = 0; k < SIMD_ELEM; ++k) {
      vF = _mm256_slli_si256_rpl(vF, 2);
      vF = _mm256_insert_epi16(vF, i*gapextend+2*gapopen, 0);
      for (int j = 0; j < stride; ++j) {
        vH = _mm256_max_epi16(pvHStore[j], vF);
        pvHStore[j] = vH;
        vH = _mm256_adds_epi16(vH, vGapO);
        vF = _mm256_adds_epi16(vF, vGapE);
        if (!_mm256_movemask_epi8(_mm256_cmpgt_epi16(vF, vH))) goto end;
      }
    }
end:

    __m256i *vp = state->pvHLoad;
    pvHLoad = state->pvHStore;
    pvHStore = vp;

    state->pvHLoad = pvHLoad;
    state->pvHStore = pvHStore;

    pvELoad = pvEStore;
  }
}

// X and Y are both aligned
void hirschberg_internal(
    uint16_t * restrict x, uint16_t * restrict y, uint16_t * restrict ry,
    size_t lx, size_t ly, size_t alx, size_t aly,
    int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
    AlignmentState *state, Result *result, size_t alloc_size
) {

  stride_seq((uint16_t *) state->query, x, alx);
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

  semiglobal(&params, match, mismatch, gapopen, gapextend);
  int16_t *n1 = (int16_t *) params.pvHLoad;

  if (n1 == state->pvHStore) {
    params.pvHStore = state->pvHLoad2;
    params.pvHLoad = state->pvHLoad;
  } else {
    params.pvHLoad = state->pvHLoad2;
  }
  params.query = state->query_reversed;
  params.database = ry;
  params.database_len += 1;
  semiglobal(&params, match, mismatch, gapopen, gapextend);
  for (int i = 0; i < alx; i++) {
      result->traceback[i] = ((int16_t *) params.pvHLoad)[i];
  }
  result->ltrace = alx;

  int16_t *n2 = (int16_t *) params.pvHLoad;
  int16_t m = INT16_MIN;
  int idx = 0;
  for (int i = 0; i < alx-1; i++) {
    int16_t k = n1[i] + n2[alx - i - 2];
    /* printf("(%d, %d), ", n1[i], n2[alx - i - 2]); */
    printf("%d, ", n1[i]);
    /* printf("%d, ", k); */
    if (k > m) {
      m = k;
      idx = i;
    }
  }
  int col = idx/SIMD_ELEM;
  int row = idx%SIMD_ELEM;
  printf("\n%d\n",  col * (alx/SIMD_ELEM) + row);
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

  size_t bsize = alx * sizeof(uint32_t);
  size_t bstride = bsize/SIMD_WIDTH_BYTES;
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

  __m256i Min = _mm256_set1_epi16(gapopen);
  for (int i = 0; i < bstride; i++) {
    state.vEMin[i] = Min;
  }

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
  (void) semiglobal;
}
