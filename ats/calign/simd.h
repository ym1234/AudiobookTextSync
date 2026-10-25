#ifndef __SIMD_H__
#define __SIMD_H__
#include <stdint.h>

#if defined(__AVX2__)
#include "simd_avx2.h"
#elif defined(__ARM_NEON)
#include "simd_neon.h"
#else
#error "the aligner needs avx2 (x86, build with -mavx2) or neon (arm64)"
#endif

#endif
