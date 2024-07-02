#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
// #include <immintrin.h>
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


#define _mm256_slli_si256_rpl(a,imm)  _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0,0,3,0)), 16-imm)

#define MATCH 1
#define MISMATCH  ((int16_t) -1)
#define GAP_OPEN  ((int16_t) -1)
#define GAP_EXTEND  ((int16_t) -1)

#define SIMD_WIDTH ((int) 256)
/* #define SIMD_WIDTH ((int) 64) */

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

/* // TODO: https://arxiv.org/pdf/1909.00899 */
/* int semiglobal_scan(uint16_t * restrict x, size_t lx, uint16_t * restrict y, size_t ly) { */
/* } */


// Return the last column
// TODO change h and e sizes depending on lx and ly
// lx and ly are assumed to be aligned
int semiglobal(uint16_t * restrict x, size_t lx, uint16_t * restrict y, size_t ly) {
  uint16_t *x_striped = calloc(lx, sizeof(*x_striped));
  size_t stride = (lx / SIMD_ELEM);
  for (int i = 0; i < lx; i++) {
    int vi = i/SIMD_ELEM, vj = i%SIMD_ELEM;
    x_striped[i] = x[vj * stride + vi];
  }

  for (int i = 0; i < lx; i++) {
    printf("%c ", x_striped[i]);
  }
  printf("\n");

  __m256i *vX = (__m256i *) x_striped;

  __m256i *pvHLoad = calloc(stride, sizeof(*pvHLoad));;
  __m256i *pvHStore = calloc(stride, sizeof(*pvHStore));;
  __m256i *pvE = calloc(stride, sizeof(*pvE));;

  __m256i Min = _mm256_set1_epi16(INT16_MIN);;
  for (int i = 0; i < stride; i++) {
    _mm256_storeu_si256(pvE+i, Min);
  }

  __m256i vMatch = _mm256_set1_epi16(MATCH);
  __m256i vMismatch = _mm256_set1_epi16(MISMATCH);
  __m256i vGapOpen = _mm256_set1_epi16(GAP_OPEN);
  __m256i vGapExtend = _mm256_set1_epi16(GAP_EXTEND);

  for (int i = 0; i < ly; i++) {
    __m256i vY = _mm256_set1_epi16(y[i]);

    __m256i vH = _mm256_loadu_si256(pvHStore + stride - 1);
    vH = _mm256_slli_si256_rpl(vH, 2);

    __m256i vF = _mm256_set1_epi16(INT16_MIN);

    __m256i *vp = pvHLoad;
    pvHLoad = pvHStore;
    pvHStore = vp;

    __m256i vE;
    for (int j = 0; j < stride; j++) {
      __m256i vX2 = _mm256_loadu_si256(vX + j);
      __m256i vCmp = _mm256_cmpeq_epi16(vY, vX2);
      __m256i MatchScore = _mm256_and_si256(vCmp, vMatch);
      __m256i MismatchScore =_mm256_andnot_si256(vCmp, vMismatch);
      __m256i vScore = _mm256_or_si256(MatchScore, MismatchScore);

      vH = _mm256_adds_epi16(vH, vScore);
      vE = _mm256_loadu_si256(pvE + j);

      vH = _mm256_max_epi16(vH, vE);
      vH = _mm256_max_epi16(vH, vF);
      _mm256_storeu_si256(pvHStore + j, vH);

      vH = _mm256_adds_epi16(vH, vGapOpen);
      vE = _mm256_adds_epi16(vE, vGapExtend);
      vE = _mm256_max_epi16(vE, vH);
      _mm256_storeu_si256(pvE + j, vE);

      vF = _mm256_adds_epi16(vF, vGapExtend);
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
            vH = _mm256_adds_epi16(vH, vGapOpen);
            vF = _mm256_adds_epi16(vF, vGapExtend);
            if (! _mm256_movemask_epi8(_mm256_cmpgt_epi16(vF, vH))) goto end;
        }
    }
end:
    printf("%d\n", counter);
  }

  int16_t *result_striped = (int16_t*) pvHStore;
  int16_t *result = malloc(lx*sizeof(*result));
  for (int i = 0; i < lx; i++) {
    int vi = i/SIMD_ELEM, vj = i%SIMD_ELEM;
    /* vj = (SIMD_ELEM - vj - 1); */
    result[vi + vj * stride] = result_striped[i];
  }
  for (int i = 0; i < lx; i++) {
    printf("%d ", result[i]);
  }
  printf("\n");
  /* for (int i = 0; i < lx; i++) { */
  /*   printf("%d ", result_striped[i]); */
  /* } */
  /* printf("\n"); */
}

/* int hirschberg() { */
/* } */

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("%s seq1 seq2\n", argv[0]);
    return 1;
  }
  size_t l[2] = {strlen(argv[1]), strlen(argv[2])};
  size_t nx = 1, ny = 2;
  /* if (l[1] > l[0]) { */
  /*   size_t t = nx; */
  /*   nx = ny; */
  /*   ny = t; */
  /* } */

  size_t lx = l[nx-1], ly = l[ny-1];
  size_t alx = align(lx, SIMD_ELEM), aly = align(ly, SIMD_ELEM);
  printf("%u, %u\n", alx, aly);
  uint16_t *n = calloc((alx+aly), sizeof(*n));
  uint16_t *r = calloc((alx+aly), sizeof(*r));
  for (int i = 0; i < lx; i++) {
    n[i] = (uint16_t)argv[nx][i];
    r[lx-i-1] = (uint16_t)argv[nx][i];
  }
  for (int i = 0; i < ly; i++) {
    n[alx+i] = (uint16_t)argv[ny][i];
    r[alx+ly-i-1] = (uint16_t)argv[ny][i];
  }
  printf("%u, %u, %u\n", alx, aly, SIMD_ELEM);
  /* diagdtw(n, r, lx, n+lx, r+lx, ly); */
  semiglobal(n, alx, n+alx, ly);
  free(n);
  free(r);
}
