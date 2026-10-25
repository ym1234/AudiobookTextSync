#ifndef __SIMD_AVX2_H__
#define __SIMD_AVX2_H__
/* the vector operations striped.h is written in, for avx2: 8 int32 lanes */
#include <immintrin.h>

#define LANES 8
typedef __m256i vec;
typedef __m256i vmask;

#define ALWAYS_INLINE static inline __attribute__((always_inline))

ALWAYS_INLINE vec vset1(int32_t x) { return _mm256_set1_epi32(x); }
ALWAYS_INLINE vec viota(void) { return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); }
ALWAYS_INLINE vec vload(const int32_t *p) { return _mm256_loadu_si256((const __m256i *) p); }

ALWAYS_INLINE vec vadd(vec a, vec b) { return _mm256_add_epi32(a, b); }
ALWAYS_INLINE vec vsub(vec a, vec b) { return _mm256_sub_epi32(a, b); }
ALWAYS_INLINE vec vmul(vec a, vec b) { return _mm256_mullo_epi32(a, b); }
ALWAYS_INLINE vec vmax(vec a, vec b) { return _mm256_max_epi32(a, b); }
ALWAYS_INLINE vec vand(vec a, vec b) { return _mm256_and_si256(a, b); }

ALWAYS_INLINE vmask vgt(vec a, vec b) { return _mm256_cmpgt_epi32(a, b); }
ALWAYS_INLINE vec vblend(vmask m, vec yes, vec no) { return _mm256_blendv_epi8(no, yes, m); }
ALWAYS_INLINE vec vselect_eq(vec a, vec b, vec yes, vec no) {
  const vmask m = _mm256_cmpeq_epi32(a, b);
  return _mm256_or_si256(_mm256_and_si256(m, yes), _mm256_andnot_si256(m, no));
}
ALWAYS_INLINE int vany_gtv(vec a, vec b) {
  const vmask m = _mm256_cmpgt_epi32(a, b);
  return !_mm256_testz_si256(m, m);
}
ALWAYS_INLINE int vany_gt(vec a, int32_t x) {
  const vmask m = _mm256_cmpgt_epi32(a, _mm256_set1_epi32(x));
  return !_mm256_testz_si256(m, m);
}

/* out lane l = v lane idx[l] */
ALWAYS_INLINE vec vperm(vec v, vec idx) { return _mm256_permutevar8x32_epi32(v, idx); }

/* every lane moves up one, x comes in at lane 0. avx2 shifts bytes within each 128 bit half only, so the low half's
   top lane is carried over to the high half by hand */
ALWAYS_INLINE vec vshift_in(vec v, int32_t x) {
  const vec shifted = _mm256_alignr_epi8(v, _mm256_permute2x128_si256(v, v, _MM_SHUFFLE(0, 0, 2, 0)), 12);
  return _mm256_insert_epi32(shifted, x, 0);
}

/* lane l = the max of lanes 0..l, fill coming in from below */
ALWAYS_INLINE vec vscan_max(vec v, vec fill) {
  v = _mm256_max_epi32(v, _mm256_alignr_epi8(v, fill, 12)); /* within each half: max with the lane below */
  v = _mm256_max_epi32(v, _mm256_alignr_epi8(v, fill, 8));  /* ... and two below, the halves are done */
  return _mm256_max_epi32(v, _mm256_blend_epi32(_mm256_set1_epi32(_mm256_extract_epi32(v, 3)), fill, 0x0F));
}

#endif
