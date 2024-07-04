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

// Casting is for utility functions is really annoying, switch to cpp or something?
// lik ethe underyilng code doesn't really change for most utility functions (ex reverse16)
// are cpp compilers smart enough to realize that?
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdbool.h>
#include <x86intrin.h>
// #include <immintrin.h>

// Parasail
// 3 or 2 doesn't really matter
// permute 2x128 checks the highest bit in the nibble and if it's set then it writes zeros
// _MM_SHUFFLE = x << 6 | y << 4 | z << 2 | f
#define _mm256_slli_si256_rpl(a,imm)  _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0,0,2,0)), 16-imm)


// _mm256_cvtepu16_epi32//_mm256_cvtepi16_epi32

#define SIMD_WIDTH ((int) 256)
#define SIMD_WIDTH_BYTES (SIMD_WIDTH/8)
#define ELEM_WIDTH ((int) 16)
#define ELEM_WIDTH_BYTES (ELEM_WIDTH / 8)
#define SIMD_ELEM (SIMD_WIDTH/ELEM_WIDTH)

static inline int32_t max(int32_t a, int32_t b) {
  return a > b ? a : b;
}

static inline int32_t min(int32_t a, int32_t b) {
  return a < b ? a : b;
}

static inline size_t align(size_t n, size_t alignment) {
  size_t a = alignment - 1;
  return (n+a) & ~a;
}

// Can this be even faster than the serial version? It has all the no-nos that simd people don't recommend
// This is probably memory bandwidth limited anyway
static void stride_seq(uint16_t * restrict seq, uint16_t * restrict ret, size_t len) {
  size_t stride = len / SIMD_ELEM;
  __m256i vStride = _mm256_mullo_epi32(_mm256_set1_epi32(stride), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
  __m256i Mask = _mm256_set_epi16(0x00, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff,0x00, 0xff, 0x00, 0xff,0x00, 0xff, 0x00, 0xff);

  for (int i = 0; i < stride; i++) {
    __m256i First = _mm256_i32gather_epi32((int *) (seq+i), vStride, 2);
    __m256i Second = _mm256_i32gather_epi32((int *) (seq+i+8*stride), vStride, 2);

    // __m128i _mm256_cvtepi32_epi16(__m256i a) (VPMOVDW) is AVX512 only
    // https://stackoverflow.com/questions/49721807/what-is-the-inverse-of-mm256-cvtepi16-epi32
    __m256i F = _mm256_and_si256(First, Mask);
    __m256i S = _mm256_and_si256(Second, Mask);

    __m256i packed = _mm256_packus_epi32(F, S);
    __m256i corrected = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
    _mm256_store_si256((__m256i *) (ret + i * SIMD_ELEM), corrected);
  }
}

// s may be not aligned, r must be aligned
// TODO https://stackoverflow.com/questions/40919766/unaligned-load-versus-unaligned-store
// https://lwn.net/Articles/255364/
// Although the biggest problem here is permute2x128 probably
static void reverse16(uint16_t * restrict s,  uint16_t * restrict r, size_t len, size_t alen) {
  __m256i ShuffleMask = _mm256_set_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
    );

  int i;
  for (i = 0; (i + SIMD_ELEM - 1) < len; i += SIMD_ELEM) {
    __m256i original = _mm256_loadu_si256((__m256i *) (s + len - i - SIMD_ELEM));
    __m256i reversed = _mm256_shuffle_epi8(original, ShuffleMask);
    __m256i lanes = _mm256_permute2x128_si256(reversed, reversed, 1);
    _mm256_store_si256((__m256i *) (r + i), lanes);
  }
  for (; i < len; i++) {
    r[i] = s[len - i - 1];
  }
}

// All the restricts because I don't understand how this shit works wtf
// UB hellscape
// https://www.dii.uchile.cl/~daespino/files/Iso_C_1999_definition.pdf 6.7.3.1
// https://davmac.wordpress.com/2013/08/07/what-restrict-really-means/
// Looking at code gen it doesn't seem to make much of a difference
typedef struct AlignmentState {
  __m256i * restrict vHMin;
  __m256i * restrict vEMin;
  uint16_t * restrict database_reversed;

  __m256i * restrict pvHLoad;
  __m256i * restrict pvHStore;
  __m256i * restrict pvHLoad2;

  __m256i * restrict pvE;
  __m256i * restrict pvE2;

  __m256i * restrict query;
  __m256i * restrict query_reversed;
} AlignmentState;


typedef struct AlignmentParams {
  __m256i * restrict vHMin;
  __m256i * restrict pvHLoad;
  __m256i * restrict pvHStore;

  __m256i * restrict vEMin;
  __m256i * restrict pvE;

  __m256i * restrict query;
  uint16_t * restrict database;
  size_t query_len;
  size_t database_len;
} AlignmentParams;

typedef struct Result {
  size_t *traceback;
  size_t ltrace;
  int64_t score;
} Result;

// TODO: https://arxiv.org/pdf/1909.00899
/* int semiglobal_scan(uint16_t * restrict x, size_t lx, uint16_t * restrict y, size_t ly) { */
/* } */

// Return the last column
// TODO change h and e sizes depending on lx and ly
// lx and ly are assumed to be aligned
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

    __m256i vH = _mm256_slli_si256_rpl(pvHLoad[stride - 1], 2);

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
      _mm256_store_si256(pvHStore + j, vH);

      vH = _mm256_adds_epi16(vH, vGapO);
      vE = _mm256_adds_epi16(vE, vGapE);
      vE = _mm256_max_epi16(vE, vH);
      _mm256_store_si256(pvEStore + j, vE);

      vF = _mm256_adds_epi16(vF, vGapE);
      vF = _mm256_max_epi16(vF, vH);

      vH = pvHLoad[j];
    }

    int j = 0;
    vH = pvHStore[j];
    vF = _mm256_slli_si256_rpl(vF, 2);
    vF = _mm256_insert_epi16(vF, INT16_MIN, 0);
    __m256i vTemp = _mm256_adds_epi16(vH, vGapO);
    vTemp = _mm256_cmpgt_epi16(vF, vTemp);
    int cmp = _mm256_movemask_epi8(vTemp);
    while (cmp != 0x0000) {
      __m256i vE = pvEStore[j];
      vH = _mm256_max_epi16(vH, vF);
      pvHStore[j] = vH;

      vH = _mm256_adds_epi16(vH, vGapO);
      pvEStore[j] = _mm256_max_epi16(vE, vH);

      vF = _mm256_adds_epi16(vF, vGapE);

      j++;
      if (j >= stride) {
        j = 0;
        vF = _mm256_slli_si256_rpl(vF, 2);
        vF = _mm256_insert_epi16(vF, INT16_MIN, 0);
      }

      vH = pvHStore[j];
      vTemp = _mm256_adds_epi16(vH, vGapO);
      vTemp = _mm256_cmpgt_epi16(vF, vTemp);
      cmp = _mm256_movemask_epi8(vTemp);
    }

    //  Copied from parasail
    //  This seems to produce "Wrong" results in a few places, and i'm not whether that's intentional or not (or a bug?)
    //  ie wether it doesn't affect the traceback score
    //  It probably biases the traceback to take a certain shape (without changing the score) but i'm not sure
    /* for (int k = 0; k < SIMD_ELEM; ++k) { */
    /*     vF = _mm256_slli_si256_rpl(vF, 2); */
    /*     vF = _mm256_insert_epi16(vF, INT16_MIN, 0); */
    /*     for (int i = 0; i < stride; ++i) { */
    /*         vH = _mm256_max_epi16(pvHStore[i], vF); */
    /*         pvHStore[i] = vH; */
    /*         vH = _mm256_adds_epi16(vH, vGapO); */
    /*         vF = _mm256_adds_epi16(vF, vGapE); */
    /*         if (! _mm256_movemask_epi8(_mm256_cmpgt_epi16(vF, vH))) goto end; // Breaks shit? */
    /*     } */
    /* } */
/* end: */

    __m256i *vp = state->pvHLoad;
    pvHLoad = state->pvHStore;
    pvHStore = vp;

    state->pvHLoad = pvHLoad;
    state->pvHStore = pvHStore;

    pvELoad = pvEStore;
  }
}

size_t add_max(__m256i *n1, __m256i *n2, size_t len) {
  __m256i Max = n1[0];
  size_t max_idx = 0;
  for (int i = 1; i < len; i++) {
    __m256i r = _mm256_adds_epi16(n1[i], n2[i]);

    printf("%x\n",_mm256_movemask_epi8(_mm256_cmpgt_epi16(r, Max)));
    if (_mm256_movemask_epi8(_mm256_cmpgt_epi16(r, Max)) == 0x0000ffff) {
      printf("HERE\n");
      Max = _mm256_max_epi16(Max, r);
      max_idx = i;
    }
  }
  return max_idx;
}

// https://stackoverflow.com/questions/23590610/find-index-of-maximum-element-in-x86-simd-vector
uint32_t hmax_index(const __m256i v)
{
    __m256i vmax = v;

    vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 2));
    vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
    vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
    vmax = _mm256_max_epi16(vmax, _mm256_permute2x128_si256(vmax, vmax, 0x01));

    __m256i vcmp = _mm256_cmpeq_epi16(v, vmax);
    int16_t *l = (int16_t *) &vmax;
    printf("MAX %d\n", l[0]);
    /* for (int i 0; i < 16; i++) { */
    /* } */

    uint32_t mask = _mm256_movemask_epi8(vcmp);
    uint32_t idx  = __builtin_ctz(mask) >> 1;
    printf("%x\n", mask);
    return idx;
}

// X and Y are both aligned
void hirschberg_internal(
    uint16_t * restrict x, uint16_t * restrict y, uint16_t * restrict ry,
    size_t lx, size_t ly, size_t alx, size_t aly,
    int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
    AlignmentState *state, Result *result, size_t alloc_size
) {

  printf("QOriginal: ");
  for (int i = 0; i < alx; i++) {
    printf("%d, ", ((int16_t *) x)[i]);
  }
  printf("\nQStrided: ");
  stride_seq(x, (uint16_t *) state->query, alx);
  for (int i = 0; i < alx; i++) {
    printf("%d, ", ((int16_t *) state->query)[i]);
  }
  printf("\nQReversed: ");
  reverse16((uint16_t *) state->query, (uint16_t *) state->query_reversed, alx, alx);
  for (int i = 0; i < alx; i++) {
    printf("%d, ", ((int16_t *) state->query_reversed)[i]);
  }
  printf("\n");

  AlignmentParams params = {
    .vHMin = state->vHMin,
    .pvHLoad = state->pvHLoad,
    .pvHStore = state->pvHStore,

    .vEMin = state->vEMin,
    .pvE = state->pvE,
    .query = state->query,
    .database = y,
    .query_len = alx,
    .database_len = ly/2,
  };
  printf("len db %d\n", params.database_len);
  semiglobal(&params, match, mismatch, gapopen, gapextend);
  __m256i *n1 = params.pvHLoad;

  printf("First Split: ");
  for (int i = 0; i < SIMD_ELEM; i++) {
    for (int j = 0; j < alx/SIMD_ELEM; j += 1) {
      int idx = i + j*SIMD_ELEM;
      printf("%d, ", ((int16_t *) n1)[idx]);
    }
  }
  printf("\n");

  params.pvHLoad = state->pvHLoad2;
  params.pvE = state->pvE2;

  params.query = state->query_reversed;
  params.database = ry;
  params.database_len = ly - ly/2 - 1;
  printf("reverse db len %d\n", params.database_len);
  semiglobal(&params, match, mismatch, gapopen, gapextend);

  /* printf("Reverse Split"); */
  reverse16((uint16_t *) state->pvHLoad2, (uint16_t *) state->pvHStore, alx, alx);
  /* for (int i = 0; i < lx; i++) { */
  /*   printf("%d, ", ((int16_t *) state->pvHStore)[i]); */
  /* } */
  /* printf("\n"); */


  printf("Second Split: ");
  for (int i = 0; i < SIMD_ELEM; i++) {
    for (int j = 0; j < alx/SIMD_ELEM; j += 1) {
      /* int idx = -i + (alx/SIMD_ELEM - j - 1) * SIMD_ELEM; */
      /* printf("j %d\n", j); */
      /* int idx = i + j*SIMD_ELEM; */
      int idx = i + j*SIMD_ELEM;
      printf("%d, ", ((int16_t *) state->pvHStore)[idx]);
    }
  }
  printf("\n");

  /* size_t idx = add_max(n1, state->pvHLoad2, alx/SIMD_ELEM); */
  /* printf("%d\n", idx); */
  /* size_t tidx = hmax_index(_mm256_adds_epi16(n1[idx], state->pvHLoad2[idx])); */
  /* printf("%d\n", tidx); */
  /* n1[] */
}


Result hirschberg(
    uint16_t * restrict x, uint16_t * restrict y,
    size_t lx, size_t ly, size_t alx, size_t aly,
    int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend,
    size_t *traceback
) {
  Result r = (struct Result) {
    .traceback = traceback,
    .ltrace = 0,
    .score = 0
  };

  size_t bsize = alx * sizeof(uint32_t);
  size_t bstride = bsize/SIMD_WIDTH_BYTES;
  printf("Size %lx, stride %lx\n", bsize, bstride);

  size_t constants_size = align(2*bsize + aly * sizeof(uint16_t), 4096);
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

  __m256i Min = _mm256_set1_epi16(INT16_MIN);;
  for (int i = 0; i < bstride; i++) {
    _mm256_store_si256(state.vEMin+i, Min);
  }

  printf("DBOriginal: ");
  for (int i = 0; i < aly; i++) {
    printf("%d, ", y[i]);
  }
  reverse16(y, state.database_reversed, ly, aly);
  printf("\nDBReversed: ");
  for (int i = 0; i < aly; i++) {
    printf("%d, ", state.database_reversed[i]);
  }
  printf("\n");

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
