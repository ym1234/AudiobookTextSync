#ifndef __HIRSCHBERG_H__
#define __HIRSCHBERG_H__

#define SIMD_WIDTH ((int) 256)
#define SIMD_WIDTH_BYTES (SIMD_WIDTH/8)
#define ELEM_WIDTH ((int) 16)
#define ELEM_WIDTH_BYTES (ELEM_WIDTH / 8)
#define SIMD_ELEM (SIMD_WIDTH/ELEM_WIDTH)

// All the restricts because I don't understand how this shit works wtf
// UB hellscape
// https://www.dii.uchile.cl/~daespino/files/Iso_C_1999_definition.pdf 6.7.3.1
// https://davmac.wordpress.com/2013/08/07/what-restrict-really-means/
// Looking at code gen it doesn't seem to make much of a difference
typedef struct AlignmentState {
  __m256i * restrict vHMin;
  __m256i * restrict vEMin;
  uint16_t * database_reversed;

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

  __m256i * query;
  uint16_t * database;
  size_t query_len;
  size_t database_len;
} AlignmentParams;

typedef struct Result {
  size_t *traceback;
  size_t ltrace;
  int64_t score;
} Result;

#endif
