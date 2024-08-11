void sgcol(AlignmentParams *state, int16_t match, int16_t mismatch, int16_t gapopen, int16_t gapextend) {
  __m256i vMatch = _mm256_set1_epi16(match);
  __m256i vMismatch = _mm256_set1_epi16(mismatch);
  __m256i vGapO = _mm256_set1_epi16(gapopen);
  __m256i vGapE = _mm256_set1_epi16(gapextend);

  __m256i * pvELoad = state->vEMin;
  __m256i * pvEStore = state->pvE;

  __m256i * restrict pvHLoad = state->vHMin;
  __m256i * restrict pvHStore = state->pvHStore;

  int stride = state->query_len / SIMD_ELEM;
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
