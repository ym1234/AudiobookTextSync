#ifndef __HIRSCHBERG_H__
#define __HIRSCHBERG_H__

#define SIMD_WIDTH ((int) 256)
#define SIMD_WIDTH_BYTES (SIMD_WIDTH/8)

// Parasail
// 3 or 2 doesn't really matter
// permute 2x128 checks the highest bit in the nibble and if it's set then it writes zeros
// _MM_SHUFFLE = x << 6 | y << 4 | z << 2 | f
#define _mm256_slli_si256_rpl(a,imm)  _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0,0,2,0)), 16-imm)
#define _mm256_srli_si256_rpl(a,imm)  _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, _MM_SHUFFLE(2,0,0,0)), a, imm)

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#define max(a, b) ((b) > (a) ? (b) : (a))
#define min(a, b) ((a) > (b) ? (b) : (a))

#include <assert.h>

/* __attribute__ ((always_inline)) */
/* static inline size_t align(int64_t n, int64_t alignment) { */
/*   assert(alignment > 0); */
/*   size_t a = alignment - 1; */
/*   return (n+a) & ~a; */
/* } */

// https://www.dii.uchile.cl/~daespino/files/Iso_C_1999_definition.pdf 6.7.3.1
// https://davmac.wordpress.com/2013/08/07/what-restrict-really-means/
typedef struct Buffers {
  uint32_t *restrict DatabaseR;

  union {
    struct {
      uint32_t *restrict Query;
      uint32_t *restrict QueryR;
    };
    struct {
      __m256i *restrict vQuery;
      __m256i *restrict vQueryR;
    };
  };

  int64_t allocsize;
  union {
    struct {
      __m256i *restrict vHLoad;
      __m256i *restrict vHLoad2;
      __m256i *restrict vHStore;
      __m256i *restrict vHStore2;

      __m256i *restrict vEStore;
      __m256i *restrict vELoad;
      __m256i *restrict vEStore2;
      __m256i *restrict vELoad2;
    };
    struct {
      int32_t *restrict HLoad;
      int32_t *restrict HLoad2;
      int32_t *restrict HStore;
      int32_t *restrict HStore2;

      int32_t *restrict ELoad;
      int32_t *restrict EStore;
      int32_t *restrict ELoad2;
      int32_t *restrict EStore2;
    };
  };
} Buffers;

typedef struct Tuple {
    union {
      struct {
        __m256i *restrict vH;
        __m256i *restrict vE;
      };
      struct {
        int32_t *restrict H;
        int32_t *restrict E;
      };
    };
} Tuple;

#endif
