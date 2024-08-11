#ifndef __UTILS_H__
#define __UTILS_H__

// Parasail
// 3 or 2 doesn't really matter
// permute 2x128 checks the highest bit in the nibble and if it's set then it writes zeros
// _MM_SHUFFLE = x << 6 | y << 4 | z << 2 | f
#define _mm256_slli_si256_rpl(a,imm)  _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0,0,2,0)), 16-imm)
#define _mm256_srli_si256_rpl(a,imm)  _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, _MM_SHUFFLE(2,0,0,0)), a, imm)

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

__attribute__ ((always_inline))
static inline size_t align(size_t n, size_t alignment) {
  size_t a = alignment - 1;
  return (n+a) & ~a;
}

// Can this be even faster than the serial version? It has all the no-nos that simd people don't recommend
// This is probably memory bandwidth limited anyway
// ALSO ALSO You need an extra +1 because you are using *i32*gather (you can use a different mask too ig)
static void stride_seq(uint16_t * restrict ret, uint16_t * restrict seq, size_t len, size_t alen) {
  int stride = alen / SIMD_ELEM;
  __m256i Max = _mm256_set1_epi32(len);
  __m256i vStride = _mm256_mullo_epi32(_mm256_set1_epi32(stride), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
  __m256i Mask = _mm256_set_epi16(
      0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff,
      0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff, 0x0000, 0xffff
    );

  for (int i = 0; i < stride; i++) {
    __m256i First = _mm256_i32gather_epi32((int *) (seq+i), vStride, 2);
    __m256i Second = _mm256_i32gather_epi32((int *) (seq+i+8*stride), vStride, 2);

    // Zero out padding
    __m256i vI = _mm256_set1_epi32(i);

    __m256i FA = _mm256_add_epi32(vStride, vI);
    __m256i FM = _mm256_cmpgt_epi32(FA, Max);

    __m256i vE = _mm256_set1_epi32(8*stride);
    __m256i SA = _mm256_add_epi32(vStride, _mm256_add_epi32(vI, vE));
    __m256i SM = _mm256_cmpgt_epi32(SA, Max);

    // __m128i _mm256_cvtepi32_epi16(__m256i a) (VPMOVDW) is AVX512 only
    // https://stackoverflow.com/questions/49721807/what-is-the-inverse-of-mm256-cvtepi16-epi32
    __m256i F = _mm256_and_si256(_mm256_andnot_si256(FM, Mask), First);
    __m256i S = _mm256_and_si256(_mm256_andnot_si256(SM, Mask), Second);

    __m256i packed = _mm256_packus_epi32(F, S);
    __m256i corrected = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
    _mm256_store_si256((__m256i *) (ret + i * SIMD_ELEM), corrected);
  }
}

__attribute__ ((always_inline))
static inline __m256i rev256(__m256i v)  {
  __m256i ShuffleMask = _mm256_set_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
    );
    __m256i reversed = _mm256_shuffle_epi8(v, ShuffleMask);
    return _mm256_permute2x128_si256(reversed, reversed, 1);
}

// s may be not aligned, r must be aligned
// TODO https://stackoverflow.com/questions/40919766/unaligned-load-versus-unaligned-store
// https://lwn.net/Articles/255364/
// Although the biggest problem here is permute2x128 (and memory) probably
static void reverse16(uint16_t * restrict r, uint16_t * restrict s, size_t len) {
  int i;
  for (i = 0; (i + SIMD_ELEM - 1) < len; i += SIMD_ELEM) {
    __m256i original = _mm256_loadu_si256((__m256i *) (s + len - i - SIMD_ELEM));
    _mm256_store_si256((__m256i *) (r + i), rev256(original));
  }
  for (; i < len; i++) {
    r[i] = s[len - i - 1];
  }
}

// https://stackoverflow.com/questions/23590610/find-index-of-maximum-element-in-x86-simd-vector
__attribute__ ((always_inline))
uint32_t inline hmax(const __m256i v) {
    __m256i vmax = v;

    // TODO Replace with shifts instead of alignrs
    vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 2));
    vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
    vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
    vmax = _mm256_max_epi16(vmax, _mm256_permute2x128_si256(vmax, vmax, 0x01));

    uint32_t mask = _mm256_movemask_epi8(_mm256_cmpeq_epi16(v, vmax));
    return __builtin_ctz(mask) >> 1;
}

void print256_hex(char *str, __m256i var) {
    uint16_t val[16];
    memcpy(val, &var, sizeof(val));
    printf("%s: %04hX %04hX %04hX %04hX %04hX %04hX %04hX %04hX %04hX %04hX %04hX %04hX %04hX %04hX %04hX %04hX\n", str,
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7],
           val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
}

void print256_num(char *str, __m256i var) {
    int16_t val[16];
    memcpy(val, &var, sizeof(val));
    printf("%s: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", str,
           val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7],
           val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]);
}

#endif
