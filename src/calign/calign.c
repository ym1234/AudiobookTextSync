#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
// #include <immintrin.h>
#include <errno.h>
#include <stdbool.h>
#include <x86intrin.h>

/* void queue_get(queue q, void **val_r) { */
/*     pthread_mutex_lock(&q->mtx); */

/*     /1* Wait for element to become available. *1/ */
/*     while (empty(q)) */
/*         rc = pthread_cond_wait(&q->cond, &q->mtx); */

/*     /1* We have an element. Pop it normally and return it in val_r. *1/ */

/*     pthread_mutex_unlock(&q->mtx); */
/* } */

/* void queue_add(queue q, void *value) { */
/*     pthread_mutex_lock(&q->mtx); */

/*     /1* Add element normally. *1/ */

/*     pthread_mutex_unlock(&q->mtx); */

/*     /1* Signal waiting threads. *1/ */
/*     pthread_cond_signal(&q->cond); */
/* } */


// Parasail
// 3 or 2 doesn't really matter
// permute 2x128 checks the highest bit in the nibble and if it's set then it writes zeros
// _MM_SHUFFLE = x << 6 | y << 4 | z << 2 | f
#define _mm256_slli_si256_rpl(a,imm)  _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0,0,2,0)), 16-imm)

#define MATCH 1
#define MISMATCH  -1
#define GAP_OPEN  -1
#define GAP_EXTEND -1

#define SIMD_WIDTH ((int) 256)
#define SIMD_WIDTH_BYTES (SIMD_WIDTH/8)
#define ELEM_WIDTH ((int) 16)
#define ELEM_WIDTH_BYTES (ELEM_WIDTH / 8)
#define SIMD_ELEM (SIMD_WIDTH/ELEM_WIDTH)

static inline int32_t max(int32_t a, int32_t b) {
  return (a > b ? a : b);
}

static inline int32_t min(int32_t a, int32_t b) {
  return (a < b ? a : b);
}

static inline size_t align(size_t n, size_t alignment) {
  size_t a = alignment - 1;
  return (n+a) & ~a;
}

void stride_seq(uint16_t * restrict seq, uint16_t * restrict ret, size_t len, size_t lanes) {
  size_t stride = len / lanes;
  for (int i = 0; i < len; i++) {
    int vi = i/lanes, vj = i%lanes;
    ret[i] = seq[vj * stride + vi];
  }
}

void unstride_seq(uint16_t * restrict seq, uint16_t * restrict ret, size_t len, size_t lanes) {
  size_t stride = len / lanes;
  for (int i = 0; i < len; i++) {
    int vi = i/lanes, vj = i%lanes;
    ret[vj * stride + vi] = seq[i];
  }
}

typedef struct AlignmentState {
  size_t alloc_size;

  // State
  // query_len
  __m256i *vHMin;

  __m256i *pvHLoad;
  __m256i *pvHStore;

  __m256i *vEMin;
  __m256i *pvE;

  // Input
  uint16_t * restrict query;
  size_t query_len;
  uint16_t * restrict database;
  size_t database_len;

  // Hirschberg state
  uint16_t *traceback; // database_len + query_len
  size_t idx;

  // Scores
  __m256i vMatch;
  __m256i vMisMatch;
  __m256i vGapOpen;
  __m256i vGapExtend;
} AlignmentState;

/* // TODO: https://arxiv.org/pdf/1909.00899 */
/* int semiglobal_scan(uint16_t * restrict x, size_t lx, uint16_t * restrict y, size_t ly) { */
/* } */

// Return the last column
// TODO change h and e sizes depending on lx and ly
// lx and ly are assumed to be aligned
/* int semiglobal(uint16_t * restrict x, size_t lx, uint16_t * restrict y, size_t ly) { */
static int semiglobal(AlignmentState *state) {
  size_t stride = (state->query_len + SIMD_ELEM - 1) / SIMD_ELEM;
  __m256i *vQuery = (__m256i *) state->query;

  __m256i *pvELoad = state->vEMin;
  __m256i *pvEStore = state->pvE;

  __m256i *pvHLoad = state->vHMin;
  __m256i *pvHStore = state->pvHStore;

  for (int i = 0; i < state->database_len; i++) {
    __m256i vY = _mm256_set1_epi16(state->database[i]);
    __m256i vF = _mm256_set1_epi16(INT16_MIN);

    __m256i vH = _mm256_loadu_si256(state->pvHLoad + stride - 1);
    vH = _mm256_slli_si256_rpl(vH, 2);

    for (int j = 0; j < stride; j++) {
      __m256i vX = _mm256_loadu_si256(vQuery + j);
      __m256i vCmp = _mm256_cmpeq_epi16(vY, vX);
      __m256i MatchScore = _mm256_and_si256(vCmp, state->vMatch);
      __m256i MismatchScore =_mm256_andnot_si256(vCmp, state->vMisMatch);
      __m256i vScore = _mm256_or_si256(MatchScore, MismatchScore);

      vH = _mm256_adds_epi16(vH, vScore);
      __m256i vE = _mm256_loadu_si256(pvELoad + j);

      vH = _mm256_max_epi16(vH, vE);
      vH = _mm256_max_epi16(vH, vF);
      _mm256_storeu_si256(pvHStore + j, vH);

      vH = _mm256_adds_epi16(vH, state->vGapOpen);
      vE = _mm256_adds_epi16(vE, state->vGapExtend);
      vE = _mm256_max_epi16(vE, vH);
      _mm256_storeu_si256(pvEStore + j, vE);

      vF = _mm256_adds_epi16(vF, state->vGapExtend);
      vF = _mm256_max_epi16(vF, vH);

      vH = _mm256_loadu_si256(pvHLoad + j);
    }

    //  Copied from parasail
    int counter = 0;
    for (int k = 0; k < SIMD_ELEM; ++k) {
        vF = _mm256_slli_si256_rpl(vF, 2);
        vF = _mm256_insert_epi16(vF, INT16_MIN, 0);
        for (int i = 0; i < stride; ++i) {
            counter += 1;
            vH = _mm256_loadu_si256(pvHStore + i);
            vH = _mm256_max_epi16(vH, vF);
            _mm256_storeu_si256(pvHStore + i, vH);
            vH = _mm256_adds_epi16(vH, state->vGapOpen);
            vF = _mm256_adds_epi16(vF, state->vGapExtend);
            if (! _mm256_movemask_epi8(_mm256_cmpgt_epi16(vF, vH))) goto end;
        }
    }
end:
    printf("%d\n", counter);

    pvELoad = pvEStore;

    __m256i *vp = state->pvHLoad;
    pvHLoad = state->pvHStore;
    pvHStore = vp;
  }
}


static int hirschberg(
    uint16_t * restrict x, uint16_t * restrict y,
    size_t lx, size_t ly,
    int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
    AlignmentState *state
) {
  printf("Hello world\n");
  size_t squery = align(lx, SIMD_ELEM);
  size_t sdatabase = align(ly, SIMD_ELEM);
  size_t stride = squery / SIMD_ELEM;
  /* printf("%d\n", stride); */

  if (state == NULL) {
    size_t sstruct = align(sizeof(AlignmentState), SIMD_WIDTH_BYTES);
    size_t buffer = stride * sizeof(__m256i);
    size_t sbuffers = 5 * buffer;
    size_t straceback = lx + ly;

    size_t sall = align(sstruct + sbuffers + (straceback + squery + sdatabase)*sizeof(uint16_t), 4096);
    void *chunk = mmap(0, sall, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    state = (AlignmentState *) chunk;
    state->alloc_size = sall;

    state->vHMin = chunk + sstruct;
    state->pvHLoad = state->vHMin + stride;
    /* printf("%u, %u\n", buffer, ((void *) state->pvHLoad) - ((void *) state->vHMin)); */
    state->pvHStore = state->pvHLoad + stride;

    state->vEMin = state->pvHStore + stride;
    state->pvE = state->vEMin + stride;

    state->traceback = (uint16_t *) (state->pvE + stride);
    state->idx = 0;

    state->query = state->traceback + straceback;
    state->database = state->query + squery;
    // How should this be specified?
    state->query_len = squery;
    state->database_len = ly; // sdatabase;

    state->vMatch = _mm256_set1_epi16(match);
    state->vMisMatch = _mm256_set1_epi16(mismatch);
    state->vGapOpen = _mm256_set1_epi16(gapopen);
    state->vGapExtend = _mm256_set1_epi16(gapextend);

    __m256i Min = _mm256_set1_epi16(INT16_MIN);;
    for (int i = 0; i < stride; i++) {
      _mm256_storeu_si256(state->vEMin+i, Min);
    }
  }
}
