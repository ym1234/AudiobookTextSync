#ifndef __IMPL32_H__
#define __IMPL32_H__

#define SIMD_ELEM32 (SIMD_WIDTH_BYTES / 4)

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
  __m256i ShuffleMask = _mm256_set_epi8(
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12
      );
    __m256i reversed = _mm256_shuffle_epi8(v, ShuffleMask);
    return _mm256_permute2x128_si256(reversed, reversed, 1);
}

static void reset32(
    Buffers *bufs, uint32_t * restrict seq,
    int32_t gapopen, int32_t gapextend,
    int64_t len, int64_t alen
) {
  __m256i Max = _mm256_set1_epi32((int32_t) len);

  int stride = alen / SIMD_ELEM32;
  __m256i vStride = _mm256_mullo_epi32(_mm256_set1_epi32(stride), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

  __m256i vGapExtend = _mm256_set1_epi32(gapextend);
  __m256i vGapOpen = _mm256_set1_epi32(gapopen);

  __m256i vZero = _mm256_set1_epi32(0);
  __m256i vNeg = _mm256_set1_epi32(INT32_MIN);

  for (int i = 0; i < stride; i++) {
    __m256i Idx = _mm256_add_epi32(vStride, _mm256_set1_epi32(i));
    __m256i vR = _mm256_mask_i32gather_epi32(vZero, seq, Idx, _mm256_cmpgt_epi32(Max, Idx), 4);

    bufs->vQuery[i] = vR;
    bufs->vQueryR[stride - i - 1] = rev32(vR);

    bufs->vHLoad[i] = _mm256_add_epi32(_mm256_mullo_epi32(Idx, vGapExtend), vGapOpen);
    /* bufs->vELoad[i] = vNeg; */
    bufs->vELoad[i] = _mm256_add_epi32(bufs->vHLoad[i], vGapOpen);

    bufs->vHLoad2[i] = bufs->vHLoad[i];
    bufs->vELoad2[i] = bufs->vELoad[i];
  }
}

// This was the easiest one to modify, checkout the other algorithms later?
// https://en.algorithmica.org/hpc/algorithms/argmin
int argmax32(__m256i *a, int n) {
  int32_t max = INT32_MIN, idx = 0;
  __m256i p = _mm256_set1_epi32(max);
  for (int i = 0; i < n; i += SIMD_ELEM32) {
    __m256i mask = _mm256_cmpgt_epi32(a[i/SIMD_ELEM32], p);
    if (unlikely(!_mm256_testz_si256(mask, mask))) {
      int32_t *b = (int32_t *) a;
      for (int j = i; j < i + SIMD_ELEM32; j++) {
        if (b[j] > max) {
          max = b[j];
          idx = j;
        }
      }
      p = _mm256_set1_epi32(max);
    }
  }

  return idx;
}

int maxsum32(__m256i *n1, __m256i *n2, int alx) {
  int stride = alx / SIMD_ELEM32;
  for (int i = 0; i < stride-1; i++) {
    int j = stride - i - 2;
    n1[i] = _mm256_add_epi32(n1[i], rev32(n2[j]));
  }
  n1[stride-1] = _mm256_add_epi32(n1[stride-1], _mm256_srli_si256_rpl(rev32(n2[stride-1]), 4));
  int idx = argmax32(n1, alx);
  return idx / SIMD_ELEM32 + idx % SIMD_ELEM32 * stride;
}

__m256i *sgcol32(
    __m256i *restrict vQuery, uint32_t *restrict database,
    int64_t lx, int64_t ly,
    __m256i *restrict pvHLoad, __m256i *restrict pvHStore, __m256i *restrict pvELoad, __m256i *restrict pvEStore,
    int32_t match, int32_t mismatch, int32_t gapopen, int32_t gapextend
) {
  __m256i vMatch = _mm256_set1_epi32(match);
  __m256i vMismatch = _mm256_set1_epi32(mismatch);
  __m256i vGapO = _mm256_set1_epi32(gapopen);
  __m256i vGapE = _mm256_set1_epi32(gapextend);

  __m256i vInf = _mm256_set1_epi32(INT32_MIN);

  int stride = lx / SIMD_ELEM32;
  __m256i vStrideExtend = _mm256_mullo_epi32(_mm256_set1_epi32(gapextend*stride), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
  /* print32_num("vStrideExtend", vStrideExtend); */

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
      /* _mm256_stream_si256(pvHStore + j, vH); */
      pvHStore[j] = vH;
      vH = pvHLoad[j];
    }

    /* print32_num("vF", vF); */
    /* __m256i vW = vF; */
    __m256i vW = _mm256_sub_epi32(vF, vStrideExtend);

    /* prefix sum */
    vW = _mm256_max_epi32(vW, _mm256_alignr_epi8(vW, vInf, 12));
    vW = _mm256_max_epi32(vW, _mm256_alignr_epi8(vW, vInf, 8));
    __m256i vK = _mm256_blend_epi32(_mm256_set1_epi32(_mm256_extract_epi32(vW, 3)), vInf, 0x0F);;
    vW = _mm256_max_epi32(vW, vK);

    /* __builtin_prefetch(pvHStore + j); */
    vW = _mm256_add_epi32(vW, vStrideExtend);
    vW = _mm256_slli_si256_rpl(vW, 4);
    vW = _mm256_insert_epi32(vW, i*gapextend+2*gapopen, 0);
    /* print32_num("vW", vW); */

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

  return pvHLoad;
}


__m256i *sgtable32(
    __m256i *restrict vQuery, uint32_t *restrict database,
    int64_t lx, int64_t ly,
    __m256i *restrict pvH, __m256i *restrict pvE, __m256i *restrict pvF,
    int32_t match, int32_t mismatch, int32_t gapopen, int32_t gapextend
) {
  __m256i vMatch = _mm256_set1_epi32(match);
  __m256i vMismatch = _mm256_set1_epi32(mismatch);
  __m256i vGapO = _mm256_set1_epi32(gapopen);
  __m256i vGapE = _mm256_set1_epi32(gapextend);

  __m256i vInf = _mm256_set1_epi32(INT32_MIN);

  int stride = lx / SIMD_ELEM32;
  __m256i vStrideExtend = _mm256_mullo_epi32(_mm256_set1_epi32(gapextend*stride), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

  for (int i = 0; i < ly; i++) {
    __m256i *pvHLoad = pvH + i * stride;
    __m256i *pvHStore = pvHLoad + stride;

    __m256i *pvELoad = pvE + i * stride;
    __m256i *pvEStore = pvELoad + stride;

    __m256i *pvFStore = pvF + i * stride;

    __m256i vY = _mm256_set1_epi32(database[i]);
    __m256i vF = _mm256_set1_epi32(INT32_MIN);
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

    /* print32_num("vF", vF); */
    /* __m256i vW = vF; */
    __m256i vW = _mm256_sub_epi32(vF, vStrideExtend);

    /* prefix sum */
    vW = _mm256_max_epi32(vW, _mm256_alignr_epi8(vW, vInf, 12));
    vW = _mm256_max_epi32(vW, _mm256_alignr_epi8(vW, vInf, 8));
    __m256i vK = _mm256_blend_epi32(_mm256_set1_epi32(_mm256_extract_epi32(vW, 3)), vInf, 0x0F);;
    vW = _mm256_max_epi32(vW, vK);

    /* __builtin_prefetch(pvHStore + j); */
    vW = _mm256_add_epi32(vW, vStrideExtend);
    vW = _mm256_slli_si256_rpl(vW, 4);
    vW = _mm256_insert_epi32(vW, i*gapextend+2*gapopen, 0);
    /* print32_num("vW", vW); */

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

// Utter trash
void trace32(
    uint32_t *restrict query, uint32_t *restrict database,
    int64_t lx, int64_t alx, int64_t ly,
    int32_t *restrict vH, int32_t *restrict vE, int32_t *restrict vF,
    int32_t match, int32_t mismatch, int32_t gapopen, int32_t gapextend,
    int64_t *traceback, int64_t *tracelen, int64_t offset
) {
  int stride = alx / SIMD_ELEM32;
  int curarr = 0;
  int64_t tracepos = *tracelen;
  while (lx > 1 && ly > 0) {
    int cx = (lx-1) / stride + (lx-1) % stride * SIMD_ELEM32;
    int px = (lx-2) / stride + (lx-2) % stride * SIMD_ELEM32;
    int32_t *cH = vH + ly * alx;
    int32_t *pH = vH + (ly-1) * alx;
    int32_t *cE = vE + ly * alx;
    int32_t *pE = vE + (ly-1) * alx;
    int32_t *cF = vF + (ly-1) * alx;
    /* printf("%d, %d, %d, %d, %c, %c\n", lx, ly, cx, ly, query[lx-1], database[ly-1]); */
    if (curarr == 0) {
      int32_t score = query[lx-1] == database[ly-1] ? match : mismatch;
      if (cH[cx] == cE[cx]) {
        curarr = 1;
      } else if (cH[cx] == cF[cx]) {
        curarr = 2;
      } else if (cH[cx] == (pH[px] + score)) {
        traceback[tracepos] = lx-1;
        traceback[tracepos+1] = ly-1;
        ly -= 1;
        lx -= 1;
        tracepos += 2;
      } else {
        printf("WTF, %d, %d\n", lx, ly);
        exit(-1);
      }
    } else if (curarr == 1) {
      if (cE[cx] == (pH[cx] + gapopen) && cE[cx] != (pE[cx] + gapextend)) {
        curarr = 0;
      }
      traceback[tracepos] = lx-1;
      traceback[tracepos+1] = ly-1;
      ly -= 1;
      tracepos += 2;
    } else if (curarr = 2) {
      if (cF[cx] == (cH[px] + gapopen) && cF[cx] != (cF[px] + gapextend)) {
        curarr = 0;
      }
      traceback[tracepos] = lx-1;
      traceback[tracepos+1] = ly-1;
      lx -= 1;
      tracepos += 2;
    }
  }

  printf("%d, %d\n", lx, ly);

  if (ly == 0) {
    while (lx > 0) {
      traceback[tracepos] = lx-1;
      traceback[tracepos+1] = ly-1;
      lx -= 1;
      tracepos += 2;
    }
  }

  while (lx == 1) {
    int32_t pV = ly == 1 ? 0 : (ly-2)*gapextend + gapopen;
    int cx = (lx-1) / stride + (lx-1) % stride * SIMD_ELEM32;
    int32_t *cH = vH + ly * alx;
    int32_t *pH = vH + (ly-1) * alx;
    int32_t *cE = vE + ly * alx;
    int32_t *pE = vE + (ly-1) * alx;
    int32_t *cF = vF + (ly-1) * alx;
    if (curarr == 0) {
      int32_t score = query[lx-1] == database[ly-1] ? match : mismatch;
      if (cH[cx] == cE[cx]) {
        curarr = 1;
      } else if (cH[cx] == cF[cx]) {
        curarr = 2;
      } else if (cH[cx] == (pV + score)) {
        traceback[tracepos] = lx-1;
        traceback[tracepos+1] = ly-1;
        ly -= 1;
        lx -= 1;
        tracepos += 2;
      } else {
        printf("WTF2, %d, %d\n", lx, ly);
        exit(-1);
      }
    } else if (curarr == 1) {
      if (cE[cx] == (pH[cx] + gapopen) && cE[cx] != (pE[cx] + gapextend)) {
        curarr = 0;
      }
      traceback[tracepos] = lx-1;
      traceback[tracepos+1] = ly-1;
      ly -= 1;
      tracepos += 2;
    } else {
      traceback[tracepos] = lx-1;
      traceback[tracepos+1] = ly-1;
      lx -= 1;
      tracepos += 2;
      break;
    }
  }

  while (ly > 0) {
    traceback[tracepos] = lx-1;
    traceback[tracepos+1] = ly-1;
    ly -= 1;
    tracepos += 2;
  }

  *tracelen = tracepos;
}

#endif

