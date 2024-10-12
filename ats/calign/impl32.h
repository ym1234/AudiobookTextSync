#ifndef __IMPL32_H__
#define __IMPL32_H__

#define SIMD_WIDTH ((int) 256)
#define SIMD_WIDTH_BYTES (SIMD_WIDTH/8)

#define SIMD_ELEM32 (SIMD_WIDTH_BYTES / (int)sizeof(int32_t))

#define _mm256_slli_si256_rpl(a,imm)  _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0,0,2,0)), 16-imm)
#define _mm256_srli_si256_rpl(a,imm)  _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, _MM_SHUFFLE(2,0,0,0)), a, imm)

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

void print32_hex(char *str, __m256i var) {
    uint32_t val[8];
    memcpy(val, &var, sizeof(val));
    printf("%s: %04X %04X %04X %04X %04X %04X %04X %04X\n", str,
           val[0], val[1], val[2], val[3],
           val[4], val[5], val[6], val[7]);
}

void print32_num(char *str, __m256i var) {
    int32_t val[8];
    memcpy(val, &var, sizeof(val));
    printf("%s: %d %d %d %d %d %d %d %d\n", str,
           val[0], val[1], val[2], val[3],
           val[4], val[5], val[6], val[7]);
}

__attribute__ ((always_inline))
static inline __m256i rev32(__m256i v)  {
    __m256i reversed = _mm256_shuffle_epi32(v, _MM_SHUFFLE(0, 1, 2, 3));
    return _mm256_permute2x128_si256(reversed, reversed, 1);
}

static void reset32(
    __m256i *restrict vHLoad, __m256i *restrict vHLoad2, __m256i *restrict vELoad, __m256i *restrict vELoad2,
    __m256i *restrict vQuery,  __m256i *restrict vQueryR,
    const uint32_t *restrict seq,
    int32_t gapopen, int32_t gapextend,
    int64_t len, int64_t stride
) {
  __m256i Max = _mm256_set1_epi32((int32_t) len);

  __m256i vStride = _mm256_mullo_epi32(_mm256_set1_epi32(stride), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

  __m256i vGapExtend = _mm256_set1_epi32(gapextend);
  __m256i vGapOpen = _mm256_set1_epi32(gapopen);

  __m256i vZero = _mm256_set1_epi32(0);

  for (int i = 0; i < stride; i++) {
    __m256i Idx = _mm256_add_epi32(vStride, _mm256_set1_epi32(i));
    __m256i vR = _mm256_mask_i32gather_epi32(vZero,  (int32_t *) seq, Idx, _mm256_cmpgt_epi32(Max, Idx), 4);

    vQuery[i] = vR;
    if (vQueryR) {
      vQueryR[stride - i - 1] = rev32(vR);
    }

    vHLoad[i] = _mm256_add_epi32(_mm256_mullo_epi32(Idx, vGapExtend), vGapOpen);
    /* bufs->vELoad[i] = _mm256_set1_epi32(INT32_MIN); */
    vELoad[i] = _mm256_add_epi32(vHLoad[i], vGapOpen);

    if (vHLoad2) {
      vHLoad2[i] = vHLoad[i];
      vELoad2[i] = vELoad[i];
    }
  }
}

// This was the easiest one to modify, checkout the other algorithms later?
// https://en.algorithmica.org/hpc/algorithms/argmin
int argmax32(__m256i *restrict a, int stride) {
  int32_t max = INT32_MIN, idx = 0;
  __m256i p = _mm256_set1_epi32(max);
  for (int i = 0; i < stride; i++) {
    __m256i mask = _mm256_cmpgt_epi32(a[i], p);
    if (unlikely(!_mm256_testz_si256(mask, mask))) {
      int32_t *b = (int32_t *) a;
      int k = i * SIMD_ELEM32;
      for (int j = k; j < k + SIMD_ELEM32; j++) {
        if (b[j] >= max) {
          max = b[j];
          idx = j;
        }
      }
      p = _mm256_set1_epi32(max);
    }
  }

  return idx;
}

int maxsum32(__m256i *restrict V1, const __m256i *restrict V2, int stride) {
  for (int i = 0; i < stride-1; i++) {
    int j = stride - i - 2;
    V1[i] = _mm256_add_epi32(V1[i], rev32(V2[j]));
  }
  __m256i K = _mm256_srli_si256_rpl(rev32(V2[stride-1]), 4);
  K = _mm256_insert_epi32(K, 0, 7);
  V1[stride-1] = _mm256_add_epi32(V1[stride-1], K);
  return argmax32(V1, stride);
}

void sgcol32(
    const __m256i *restrict vQuery, const uint32_t *restrict database,
    int64_t stride, int64_t ly,
    __m256i *restrict pvHLoad, __m256i *restrict pvHStore, __m256i *restrict pvELoad, __m256i *restrict pvEStore,
    int32_t match, int32_t mismatch, int32_t gapopen, int32_t gapextend
) {
  __m256i vMatch = _mm256_set1_epi32(match);
  __m256i vMismatch = _mm256_set1_epi32(mismatch);
  __m256i vGapO = _mm256_set1_epi32(gapopen);
  __m256i vGapE = _mm256_set1_epi32(gapextend);

  __m256i vInf = _mm256_set1_epi32(INT32_MIN);
  __m256i vStrideExtend = _mm256_mullo_epi32(_mm256_set1_epi32(gapextend*stride), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
  for (int i = 0; i < ly; i++) {
    __m256i vY = _mm256_set1_epi32(database[i]);
    __m256i vF = vInf;
    __m256i vH = _mm256_slli_si256_rpl(pvHLoad[stride - 1], 4);
    vH = _mm256_insert_epi32(vH, gapopen + (i-1)*gapextend, 0);
    if (unlikely(!i)) {
      vH = _mm256_insert_epi32(vH, 0, 0);
    }

    __m256i vGapper = _mm256_set1_epi32(gapextend*stride + gapopen);
    for (int j = 0; j < stride; j++) {
      __builtin_prefetch(vQuery + j);
      __builtin_prefetch(pvHLoad + j, 0);
      __builtin_prefetch(pvELoad + j, 0);
      vGapper = _mm256_sub_epi32(vGapper, vGapE);
      __m256i vCmp = _mm256_cmpeq_epi32(vY, vQuery[j]);
      __m256i MatchScore = _mm256_and_si256(vCmp, vMatch);
      __m256i MismatchScore =_mm256_andnot_si256(vCmp, vMismatch);
      __m256i vScore = _mm256_or_si256(MatchScore, MismatchScore);
      vH = _mm256_add_epi32(vH, vScore);
      vH = _mm256_max_epi32(vH, pvELoad[j]);
      vF = _mm256_max_epi32(vF, _mm256_add_epi32(vH, vGapper));
      pvHStore[j] = vH;
      vH = pvHLoad[j];
    }

    __m256i vW = _mm256_sub_epi32(vF, vStrideExtend);

    /* prefix sum */
    vW = _mm256_max_epi32(vW, _mm256_alignr_epi8(vW, vInf, 12));
    vW = _mm256_max_epi32(vW, _mm256_alignr_epi8(vW, vInf, 8));
    __m256i vK = _mm256_blend_epi32(_mm256_set1_epi32(_mm256_extract_epi32(vW, 3)), vInf, 0x0F);;
    vW = _mm256_max_epi32(vW, vK);

    vW = _mm256_add_epi32(vW, vStrideExtend);
    vW = _mm256_slli_si256_rpl(vW, 4);
    vW = _mm256_insert_epi32(vW, i*gapextend+2*gapopen, 0);

    for (int j = 0; j < stride; j++) {
      __builtin_prefetch(pvHStore + j + 1, 0);
      __builtin_prefetch(pvELoad + j + 1, 0);
      __builtin_prefetch(pvEStore + j + 1, 0);
      __m256i vH = _mm256_max_epi32(vW, pvHStore[j]);
      pvHStore[j] = vH;

      vH = _mm256_add_epi32(vH, vGapO);

      __m256i vE = _mm256_add_epi32(pvELoad[j], vGapE);
      pvEStore[j] = _mm256_max_epi32(vH, vE);

      vW = _mm256_add_epi32(vW, vGapE);
      vW = _mm256_max_epi32(vH, vW);
    }

    __m256i *tmp = pvHLoad;
    pvHLoad = pvHStore;
    pvHStore = tmp;

    tmp = pvELoad;
    pvELoad = pvEStore;
    pvEStore = tmp;
  }
}


void sgtable32(
    const __m256i *restrict vQuery, const uint32_t *restrict database,
    int64_t stride, int64_t ly,
    __m256i *restrict pvH, __m256i *restrict pvE, __m256i *restrict pvF,
    int32_t match, int32_t mismatch, int32_t gapopen, int32_t gapextend
) {
  __m256i vMatch = _mm256_set1_epi32(match);
  __m256i vMismatch = _mm256_set1_epi32(mismatch);
  __m256i vGapO = _mm256_set1_epi32(gapopen);
  __m256i vGapE = _mm256_set1_epi32(gapextend);

  __m256i vInf = _mm256_set1_epi32(INT32_MIN);
  __m256i vStrideExtend = _mm256_mullo_epi32(_mm256_set1_epi32(gapextend*stride), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
  for (int i = 0; i < ly; i++) {
    __m256i *pvHLoad = pvH + i * stride;
    __m256i *pvHStore = pvHLoad + stride;

    __m256i *pvELoad = pvE + i * stride;
    __m256i *pvEStore = pvELoad + stride;

    __m256i *pvFStore = pvF + i * stride;

    __m256i vY = _mm256_set1_epi32(database[i]);
    __m256i vF = vInf;
    __m256i vH = _mm256_slli_si256_rpl(pvHLoad[stride - 1], 4);
    vH = _mm256_insert_epi32(vH, gapopen + (i-1)*gapextend, 0);
    if (unlikely(!i)) {
      vH = _mm256_insert_epi32(vH, 0, 0);
    }

    __m256i vGapper = _mm256_set1_epi32(gapextend*stride + gapopen);
    for (int j = 0; j < stride; j++) {
      __builtin_prefetch(vQuery + j + 1);
      __builtin_prefetch(pvHLoad + j, 0);
      __builtin_prefetch(pvELoad + j, 0);
      vGapper = _mm256_sub_epi32(vGapper, vGapE);
      __m256i vCmp = _mm256_cmpeq_epi32(vY, vQuery[j]);
      __m256i MatchScore = _mm256_and_si256(vCmp, vMatch);
      __m256i MismatchScore =_mm256_andnot_si256(vCmp, vMismatch);
      __m256i vScore = _mm256_or_si256(MatchScore, MismatchScore);
      vH = _mm256_add_epi32(vH, vScore);
      vH = _mm256_max_epi32(vH, pvELoad[j]);
      vF = _mm256_max_epi32(vF, _mm256_add_epi32(vH, vGapper));
      pvHStore[j] = vH;
      vH = pvHLoad[j];
    }

    __m256i vW = _mm256_sub_epi32(vF, vStrideExtend);

    /* prefix sum */
    vW = _mm256_max_epi32(vW, _mm256_alignr_epi8(vW, vInf, 12));
    vW = _mm256_max_epi32(vW, _mm256_alignr_epi8(vW, vInf, 8));
    __m256i vK = _mm256_blend_epi32(_mm256_set1_epi32(_mm256_extract_epi32(vW, 3)), vInf, 0x0F);;
    vW = _mm256_max_epi32(vW, vK);

    vW = _mm256_add_epi32(vW, vStrideExtend);
    vW = _mm256_slli_si256_rpl(vW, 4);
    vW = _mm256_insert_epi32(vW, i*gapextend+2*gapopen, 0);

    for (int j = 0; j < stride; j++) {
      __builtin_prefetch(pvHStore + j + 1, 0);
      __builtin_prefetch(pvELoad + j + 1, 0);
      __builtin_prefetch(pvEStore + j + 1, 0);
      __m256i vH = _mm256_max_epi32(vW, pvHStore[j]);
      pvHStore[j] = vH;
      _mm256_stream_si256(pvFStore + j, vW);

      vH = _mm256_add_epi32(vH, vGapO);

      __m256i vE = _mm256_add_epi32(pvELoad[j], vGapE);
      pvEStore[j] = _mm256_max_epi32(vH, vE);

      vW = _mm256_add_epi32(vW, vGapE);
      vW = _mm256_max_epi32(vH, vW);
    }
  }
}

int32_t trace32(
    const uint32_t *restrict query, const uint32_t *restrict database,
    int64_t lx, int64_t alx, int64_t ly,
    const int32_t *restrict vH, const int32_t *restrict vE, const int32_t *restrict vF,
    int32_t match, int32_t mismatch, int32_t gapopen, int32_t gapextend,
    int64_t *traceback, int64_t *tracelen, int64_t offsetx, int64_t offsety
) {
  int stride = alx / SIMD_ELEM32;
  int32_t score = vH[(ly-1) * alx + (lx-1) / stride + (lx-1) % stride * SIMD_ELEM32];

  int curarr = 0;
  int64_t tracepos = *tracelen;

#define TRACEAPPEND do { \
    traceback[tracepos++] = offsetx + lx - 1; \
    traceback[tracepos++] = offsety + ly - 1; \
  } while(0);

  #define max(a, b) ((b) > (a) ? (b) : (a))
  int32_t minimum = INT32_MIN - max(gapopen, gapextend);
  while (lx > 0 && ly > 0) {
    int cx = (lx-1) / stride + (lx-1) % stride * SIMD_ELEM32;
    int px = (lx-2) / stride + (lx-2) % stride * SIMD_ELEM32;

    int32_t cHcx = vH[ly * alx + cx];

    int32_t cHpx = lx < 2 ? gapopen + (ly-1)*gapextend : vH[ly * alx + px];
    int32_t top =  ly >= 2 ? gapopen + (ly-2)*gapextend : 0;
    int32_t pHpx = lx < 2 ? top : vH[(ly-1) * alx + px];
    int32_t pHcx = vH[(ly-1) * alx + cx];

    int32_t cEcx = vE[(ly-1) * alx + cx];
    int32_t pEcx = ly < 2 ? minimum : vE[(ly-2) * alx + cx];

    int32_t cFcx = vF[(ly-1) * alx + cx];
    int32_t cFpx = lx < 2 ? minimum : vF[(ly-1) * alx + px];

    if (curarr == 0) {
      int32_t score = query[lx-1] == database[ly-1] ? match : mismatch;
      if (cHcx == cEcx) {
        curarr = 1;
      } else if (cHcx == cFcx) {
        curarr = 2;
      } else if (cHcx == (pHpx + score)) {
        TRACEAPPEND;
        lx -= 1;
        ly -= 1;
      } else {
        printf("WTF cx %d, ly %ld, chcx %d, chpx %d, score %d\n", cx, ly, cHcx, cHpx, score);
        exit(-1);
      }
    } else if (curarr == 1) {
      TRACEAPPEND;
      if (cEcx == (pHcx + gapopen) && (cEcx != (pEcx + gapextend))) {
        curarr = 0;
      }
      ly -= 1;
    } else if (curarr == 2) {
      TRACEAPPEND;
      if (cFcx == (cHpx + gapopen) && cFcx != (cFpx + gapextend)) {
        curarr = 0;
      }
      lx -= 1;
    }
  }

  while (lx != 0) {
      TRACEAPPEND;
      lx -= 1;
  }
  while (ly != 0) {
      TRACEAPPEND;
      ly -= 1;
  }

  *tracelen = tracepos;
  return score;
}

#endif

