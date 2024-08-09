#include <stdlib.h>
#include <sys/mman.h>
#include <x86intrin.h>

#define _mm256_slli_si256_rpl(a,imm)  _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0,0,2,0)), 16-imm)

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#define SIMD_WIDTH ((int) 256)
#define SIMD_WIDTH_BYTES (SIMD_WIDTH/8)
#define ELEM_WIDTH ((int) 16)
#define ELEM_WIDTH_BYTES (ELEM_WIDTH / 8)
#define SIMD_ELEM (SIMD_WIDTH/ELEM_WIDTH)

static inline size_t align(size_t n, size_t alignment) {
  size_t a = alignment - 1;
  return (n+a) & ~a;
}

int16_t *semiglobal(
    uint16_t *query, uint32_t query_len,
    uint16_t *database, uint32_t database_len,
    int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend
) {
  size_t stride = query_len / SIMD_ELEM;
  __m256i vMatch = _mm256_set1_epi16(match);
  __m256i vMismatch = _mm256_set1_epi16(mismatch);
  __m256i vGapO = _mm256_set1_epi16(gapopen);
  __m256i vGapE = _mm256_set1_epi16(gapextend);

  size_t bufsz = align(query_len * sizeof(int16_t), 4096);

  __m256i * pvE = (__m256i *) mmap(0, bufsz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  for (int i = 0; i < stride; i++) {
    pvE[i] = _mm256_set1_epi16(gapopen);
  }

  __m256i *pvHLoad  = (__m256i *) mmap(0, bufsz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  __m256i *pvHStore = (__m256i *) mmap(0, bufsz, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

  __m256i *q = (__m256i *) query;

  /* outer loop over database sequence */
  for (int j = 0; j < database_len; j++) {
    __m256i vY = _mm256_set1_epi16(database[j]);
    __m256i vE;
    /* Initialize F value to -inf.  Any errors to vH values will be
     * corrected in the Lazy_F loop.  */
    __m256i vF = _mm256_set1_epi16(INT16_MIN);
    vF = _mm256_slli_si256_rpl(vF, 2);
    vF = _mm256_insert_epi16(vF, 2*gapopen + j*gapextend, 0);

    /* load final segment of pvHStore and shift left by 2 bytes */
    __m256i vH = _mm256_slli_si256_rpl(pvHStore[stride - 1], 2);

    /* Swap the 2 H buffers. */
    __m256i* pv = pvHLoad;
    pvHLoad = pvHStore;
    pvHStore = pv;

    /* insert upper boundary condition */
    vH = _mm256_insert_epi16(vH, gapopen+(j-1)*gapextend, 0);
    if (unlikely(!j)) {
      vH = _mm256_insert_epi16(vH, 0, 0);
    }

    /* inner loop to process the query sequence */
    for (int i = 0; i < stride; i++) {
      __m256i vX = _mm256_load_si256(q + i);
      __m256i vCmp = _mm256_cmpeq_epi16(vY, vX);
      __m256i MatchScore = _mm256_and_si256(vCmp, vMatch);
      __m256i MismatchScore =_mm256_andnot_si256(vCmp, vMismatch);
      __m256i vScore = _mm256_or_si256(MatchScore, MismatchScore);

      vH = _mm256_adds_epi16(vH, vScore);
      vE = _mm256_load_si256(pvE + i);

      /* Get max from vH, vE and vF. */
      vH = _mm256_max_epi16(vH, vE);
      vH = _mm256_max_epi16(vH, vF);
      /* Save vH values. */
      _mm256_store_si256(pvHStore + i, vH);

      vH = _mm256_adds_epi16(vH, vGapO);

      /* Update vE value. */
      vE = _mm256_adds_epi16(vE, vGapE);
      vE = _mm256_max_epi16(vE, vH);
      _mm256_store_si256(pvE + i, vE);

      /* Update vF value. */
      vF = _mm256_adds_epi16(vF, vGapE);
      vF = _mm256_max_epi16(vF, vH);

      /* Load the next vH. */
      vH = _mm256_load_si256(pvHLoad + i);
    }

    /* Lazy_F loop: has been revised to disallow adjecent insertion and
     * then deletion, so don't update E(i, i), learn from SWPS3 */
    int count = 0;
    for (int k = 0; k < 16; ++k) {
      /* int64_t tmp = s2_beg ? -open : (boundary[j+1]-open); */
      int64_t tmp = 2*gapopen + j*gapextend;
      int16_t tmp2 = tmp < INT16_MIN ? INT16_MIN : tmp;
      vF = _mm256_slli_si256_rpl(vF, 2);
      vF = _mm256_insert_epi16(vF, tmp2, 0);
      for (int i = 0; i < stride; ++i) {
        count += 1;
        vH = _mm256_load_si256(pvHStore + i);
        vH = _mm256_max_epi16(vH,vF);
        _mm256_store_si256(pvHStore + i, vH);
        vH = _mm256_adds_epi16(vH, vGapO);
        vF = _mm256_adds_epi16(vF, vGapE);
        if (! _mm256_movemask_epi8(_mm256_cmpgt_epi16(vF, vH))) goto end;
        /* vF = _mm256_max_epi16(vF, vH); */
      }
    }
end:
    /* printf("count %d\n", count); */
  }

  munmap(pvHLoad, bufsz);
  munmap(pvE, bufsz);
  return (int16_t *) pvHStore;
}
