#ifndef __SIMD_NEON_H__
#define __SIMD_NEON_H__
/* the vector operations striped.h is written in, for neon: 4 int32 lanes */
#include <arm_neon.h>

#define LANES 4
typedef int32x4_t vec;
typedef uint32x4_t vmask;

#define ALWAYS_INLINE static inline __attribute__((always_inline))

ALWAYS_INLINE vec vset1(int32_t x) { return vdupq_n_s32(x); }
ALWAYS_INLINE vec viota(void) { static const int32_t iota[4] = {0, 1, 2, 3}; return vld1q_s32(iota); }
ALWAYS_INLINE vec vload(const int32_t *p) { return vld1q_s32(p); }

ALWAYS_INLINE vec vadd(vec a, vec b) { return vaddq_s32(a, b); }
ALWAYS_INLINE vec vsub(vec a, vec b) { return vsubq_s32(a, b); }
ALWAYS_INLINE vec vmul(vec a, vec b) { return vmulq_s32(a, b); }
ALWAYS_INLINE vec vmax(vec a, vec b) { return vmaxq_s32(a, b); }
ALWAYS_INLINE vec vand(vec a, vec b) { return vandq_s32(a, b); }

ALWAYS_INLINE vmask vgt(vec a, vec b) { return vcgtq_s32(a, b); }
ALWAYS_INLINE vec vblend(vmask m, vec yes, vec no) { return vbslq_s32(m, yes, no); }
ALWAYS_INLINE vec vselect_eq(vec a, vec b, vec yes, vec no) { return vbslq_s32(vceqq_s32(a, b), yes, no); }
ALWAYS_INLINE int vany_gtv(vec a, vec b) { return vmaxvq_u32(vcgtq_s32(a, b)) != 0; }
ALWAYS_INLINE int vany_gt(vec a, int32_t x) { return vmaxvq_s32(a) > x; }

/* out lane l = v lane idx[l]: a byte table lookup, lane k is bytes 4k .. 4k+3 */
ALWAYS_INLINE vec vperm(vec v, vec idx) {
  const uint32x4_t bytes = vmlaq_n_u32(vdupq_n_u32(0x03020100), vreinterpretq_u32_s32(idx), 0x04040404);
  return vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(v), vreinterpretq_u8_u32(bytes)));
}

/* every lane moves up one, x comes in at lane 0 */
ALWAYS_INLINE vec vshift_in(vec v, int32_t x) { return vextq_s32(vdupq_n_s32(x), v, 3); }

/* lane l = the max of lanes 0..l, fill coming in from below */
ALWAYS_INLINE vec vscan_max(vec v, vec fill) {
  v = vmaxq_s32(v, vextq_s32(fill, v, 3)); /* max with the lane below */
  return vmaxq_s32(v, vextq_s32(fill, v, 2)); /* ... and two below */
}

#endif
