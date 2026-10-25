#ifndef __STRIPED_H__
#define __STRIPED_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "simd.h"

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

static inline int64_t striped(int64_t p, int64_t stride) { return (p % stride) * LANES + p / stride; }

static void reset32(
    vec *restrict vHLoad, vec *restrict vHLoad2, vec *restrict vELoad, vec *restrict vELoad2,
    vec *restrict vQuery,
    const uint32_t *restrict seq,
    int32_t gapopen, int32_t gapextend,
    int64_t len, int64_t stride
) {
  for (int64_t i = 0; i < stride; i++) {
    int32_t q[LANES], h[LANES];
    for (int l = 0; l < LANES; l++) {
      const int64_t p = l * stride + i;
      q[l] = p < len ? (int32_t) seq[p] : 0;
      h[l] = (int32_t) (p * gapextend + gapopen);
    }
    vQuery[i] = vload(q);
    vHLoad[i] = vload(h);
    vELoad[i] = vadd(vHLoad[i], vset1(gapopen));

    if (vHLoad2) {
      vHLoad2[i] = vHLoad[i];
      vELoad2[i] = vELoad[i];
    }
  }
}

// This was the easiest one to modify, checkout the other algorithms later?
// https://en.algorithmica.org/hpc/algorithms/argmin
int argmax32(const vec *restrict a, int64_t stride) {
  int32_t max = INT32_MIN;
  int idx = 0;
  for (int64_t i = 0; i < stride; i++) {
    if (unlikely(vany_gt(a[i], max))) {
      const int32_t *b = (const int32_t *) (a + i);
      for (int j = 0; j < LANES; j++) {
        if (b[j] >= max) {
          max = b[j];
          idx = (int) (i * LANES + j);
        }
      }
    }
  }
  return idx;
}

/*
 * rotrev32 writes out[] so that its position p holds in[] position (aligned - 1 - p - shift):
 * the whole array reversed, then moved down by `shift` positions, with `fill` wherever that
 * source position would be negative. A shift of s positions moves vectors by s % stride and lanes
 * by s / stride, and the vectors that wrap past vector 0 come back from the end one lane lower,
 * so each output vector is a single lane permute (reverse + lane shift) plus a blend with `fill`.
 */
static inline void rotrev_pattern(int64_t s, vec *idx, vmask *keep) {
  const vec src = vsub(vset1(LANES - 1 - (int32_t) s), viota());
  *keep = vgt(src, vset1(-1));               /* source lane LANES-1 - l - s exists */
  *idx = vand(src, vset1(LANES - 1));
}

static inline void rotrev32(const vec *restrict in, vec *restrict out, int64_t stride, int64_t shift, vec fill) {
  const int64_t k = shift / stride, r = shift % stride;
  vec idx0, idx1;
  vmask keep0, keep1;
  rotrev_pattern(k, &idx0, &keep0);          /* vectors that don't wrap */
  rotrev_pattern(k + 1, &idx1, &keep1);      /* vectors that wrap past vector 0: one lane more */
  for (int64_t v = 0; v < stride; v++) {
    const int64_t u = stride - 1 - r - v;    /* walks back from the last stripe before the shift */
    out[v] = u >= 0 ? vblend(keep0, vperm(in[u], idx0), fill)
                    : vblend(keep1, vperm(in[u + stride], idx1), fill);
  }
}

void reverse_query32(const vec *restrict query, vec *restrict rquery, int64_t stride, int64_t pad) {
  rotrev32(query, rquery, stride, pad, vset1(0));
}

int pairmax32(const vec *restrict a1, const vec *restrict a2, vec *restrict tmp,
              int64_t stride, int64_t pad, int32_t bias) {
  rotrev32(a2, tmp, stride, pad + 1, vset1(-(1 << 30)));
  const vec b = vset1(bias);
  for (int64_t v = 0; v < stride; v++)
    tmp[v] = vadd(vadd(a1[v], tmp[v]), b);
  return argmax32(tmp, stride);
}

ALWAYS_INLINE void sg_row(
    int64_t i, uint32_t y, const vec *restrict vQuery, int64_t stride,
    const vec *pvHPrev, vec *pvH, const vec *pvEIn, vec *pvEOut, int update_e,
    vec *restrict pvF, vec vMatch, vec vMismatch, vec vInf, vec vStrideExtend,
    int32_t gapopen, int32_t gapextend, int32_t open0
) {
  const vec vY = vset1(y), vGapO = vset1(gapopen), vGapE = vset1(gapextend);
  vec vH = vshift_in(pvHPrev[stride - 1], i ? open0 + (int32_t) (i-1)*gapextend : 0);
  vec vF = vset1(INT32_MIN / 2);

  for (int64_t j = 0; j < stride; j++) {
    __builtin_prefetch(vQuery + j + 1);
    __builtin_prefetch(pvHPrev + j + 1, 0);
    __builtin_prefetch(pvEIn + j + 1, 0);
    const vec vHPrev = pvHPrev[j];
    const vec vE = pvEIn[j];
    vH = vadd(vH, vselect_eq(vY, vQuery[j], vMatch, vMismatch));
    vH = vmax(vmax(vH, vE), vF);
    pvH[j] = vH;
    if (pvF) pvF[j] = vF;

    const vec vHOpen = vadd(vH, vGapO);
    if (update_e) pvEOut[j] = vmax(vHOpen, vadd(vE, vGapE));
    vF = vmax(vHOpen, vadd(vF, vGapE));
    vH = vHPrev;
  }

  vec vW = vshift_in(vF, open0 + (int32_t) i*gapextend + gapopen);
  vW = vadd(vscan_max(vsub(vW, vStrideExtend), vInf), vStrideExtend);

  const vec vSlack = vset1(gapopen - gapextend);
  for (int64_t j = 0; j < stride; j++) {
    const vec vHFirst = pvH[j];
    const vec vHj = vmax(vHFirst, vW);
    pvH[j] = vHj;
    if (pvF) pvF[j] = vmax(pvF[j], vW);
    if (update_e) pvEOut[j] = vmax(pvEOut[j], vadd(vHj, vGapO));
    if (!vany_gtv(vW, vadd(vHFirst, vSlack))) break;
    vW = vadd(vW, vGapE);
  }
}

void sgcol32(
    const vec *restrict vQuery, const uint32_t *restrict database,
    int64_t stride, int64_t ly,
    vec *restrict pvH, vec *restrict pvE,
    int32_t match, int32_t mismatch, int32_t gapopen, int32_t gapextend, int32_t open0
) {
  const vec vMatch = vset1(match), vMismatch = vset1(mismatch), vInf = vset1(INT32_MIN);
  const vec vStrideExtend = vmul(vset1(gapextend*stride), viota());
  for (int64_t i = 0; i + 1 < ly; i++)
    sg_row(i, database[i], vQuery, stride, pvH, pvH, pvE, pvE, 1, NULL,
           vMatch, vMismatch, vInf, vStrideExtend, gapopen, gapextend, open0);
  if (ly > 0)
    sg_row(ly - 1, database[ly - 1], vQuery, stride, pvH, pvH, pvE, pvE, 0, NULL,
           vMatch, vMismatch, vInf, vStrideExtend, gapopen, gapextend, open0);
}

void sgtable32(
    const vec *restrict vQuery, const uint32_t *restrict database,
    int64_t stride, int64_t ly,
    vec *restrict pvH, vec *restrict pvE, vec *restrict pvF,
    int32_t match, int32_t mismatch, int32_t gapopen, int32_t gapextend, int32_t open0
) {
  const vec vMatch = vset1(match), vMismatch = vset1(mismatch), vInf = vset1(INT32_MIN);
  const vec vStrideExtend = vmul(vset1(gapextend*stride), viota());
  for (int64_t i = 0; i < ly; i++) {
    sg_row(i, database[i], vQuery, stride, pvH + i * stride, pvH + (i+1) * stride, pvE + i * stride, pvE + (i+1) * stride, 1,
           pvF + i * stride, vMatch, vMismatch, vInf, vStrideExtend, gapopen, gapextend, open0);
  }
}

int32_t trace32(
    const uint32_t *restrict query, const uint32_t *restrict database,
    int64_t lx, int64_t alx, int64_t ly,
    const int32_t *restrict vH, const int32_t *restrict vE, const int32_t *restrict vF,
    int32_t match, int32_t mismatch, int32_t gapopen, int32_t gapextend, int32_t open0, int te,
    int64_t *traceback, int64_t *tracelen, int64_t offsetx, int64_t offsety
) {
  const int64_t stride = alx / LANES;
  int32_t score = vH[ly * alx + striped(lx - 1, stride)];

  int curarr = 0;
  if (te) {
    int32_t e_end = vE[(ly-1) * alx + striped(lx - 1, stride)] - gapopen + gapextend;
    if (e_end > score) { score = e_end; curarr = 1; }
  }
  int64_t tracepos = *tracelen;

#define TRACEAPPEND do { \
    int64_t px_ = offsetx + lx, py_ = offsety + ly; \
    /* don't write dups */ \
    if (tracepos < 2 || traceback[tracepos-2] != px_ || traceback[tracepos-1] != py_) { \
      traceback[tracepos++] = px_; \
      traceback[tracepos++] = py_; \
    } \
  } while(0);

  #define max(a, b) ((b) > (a) ? (b) : (a))
  int32_t minimum = INT32_MIN - max(gapopen, gapextend);
  while (lx > 0 && ly > 0) {
    const int64_t cx = striped(lx - 1, stride);
    const int64_t px = lx < 2 ? 0 : striped(lx - 2, stride);

    int32_t cHcx = vH[ly * alx + cx];

    int32_t cHpx = lx < 2 ? open0 + (ly-1)*gapextend : vH[ly * alx + px];
    int32_t top =  ly >= 2 ? open0 + (ly-2)*gapextend : 0;
    int32_t pHpx = lx < 2 ? top : vH[(ly-1) * alx + px];
    int32_t pHcx = vH[(ly-1) * alx + cx];

    int32_t cEcx = vE[(ly-1) * alx + cx];
    int32_t pEcx = ly < 2 ? minimum : vE[(ly-2) * alx + cx];

    int32_t cFcx = vF[(ly-1) * alx + cx];
    int32_t cFpx = lx < 2 ? minimum : vF[(ly-1) * alx + px];

    if (curarr == 0) {
      int32_t score = query[lx-1] == database[ly-1] ? match : mismatch;
      if (cHcx == (pHpx + score)) {
        TRACEAPPEND;
        lx -= 1;
        ly -= 1;
      } else if (cHcx == cEcx) {
        curarr = 1;
      } else if (cHcx == cFcx) {
        curarr = 2;
      } else {
        printf("WTF cx %lld, ly %lld, chcx %d, chpx %d, score %d\n", (long long) cx, (long long) ly, cHcx, cHpx, score);
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
  TRACEAPPEND;

#undef TRACEAPPEND

  *tracelen = tracepos;
  return score;
}

#endif
