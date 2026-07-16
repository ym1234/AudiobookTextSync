// Modified from bsalign/bsalign.h GPL3

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <x86intrin.h>

#define _mm256_slli_si256_rpl(a, imm)                                          \
  _mm256_alignr_epi8(                                                          \
      a, _mm256_permute2x128_si256(a, a, _MM_SHUFFLE(0, 0, 2, 0)), 16 - imm)

#define WORDSIZE 32
#define MIN(n1, n2) (((n1) < (n2)) ? (n1) : (n2))
#define MAX(n1, n2) (((n1) > (n2)) ? (n1) : (n2))

#define I8_MIN (-128)
#define I8_MAX (127)

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

static inline i64 banded_striped_epi8_seqalign_getscore(i8 *us, i32 *ubegs,
                                                        i32 W, i32 pos) {
  int x = pos % W;
  int y = pos / W;
  int s = ubegs[y];
  for (int i = 0; i < x; i++) {
    s += us[i * WORDSIZE + y];
  }
  return s;
}

#if 0
static inline void
banded_striped_epi8_seqalign_piecex_row_check_ubegs(i8 *us, int *ubegs, int W) {
  int i, j, sc;
  sc = ubegs[0];
  for (j = 0; j < WORDSIZE; j++) {
    for (i = 0; i < W; i++)
      sc += us[i * WORDSIZE + j];
    if (sc != ubegs[j + 1]) {
      // if(true){
      fflush(stdout);
      fprintf(stderr, " -- something wrong in %s %d %d -- %s:%d --\n", __FUNCTION__, sc, ubegs[j + 1], __FILE__, __LINE__);
      fflush(stderr);
      // abort();
    }
  }
}
#else
#define banded_striped_epi8_seqalign_piecex_row_check_ubegs(...)
#endif

static inline int banded_striped_epi8_seqalign_piecex_row_cal_tail_codes(
    __m256i h, __m256i u, __m256i v, i8 *us[2], int *ubegs[2], i64 W) {
  i8 __attribute__((aligned(WORDSIZE))) vs[WORDSIZE];
  // revise the first striped block of u
  v = _mm256_subs_epi8(h, u); // v(x - 1, y) = h(x - 1, y) - u(x - 1, y - 1)
  _mm256_store_si256((__m256i *)vs, v);
  for (int i = 1; i <= WORDSIZE; i++) {
    ubegs[1][i] = ubegs[0][i] + vs[i - 1];
  }
  v = _mm256_slli_si256_rpl(v, 1);
  u = _mm256_loadu_si256((__m256i *)us[1]);
  u = _mm256_subs_epi8(u, v); // because I previously set v to zero, now update it,
                              // u(x, y) = h(x, y) - v(x - 1, y)
  _mm256_storeu_si256((__m256i *)us[1], u);
  // shift score to fit EPI8
  ubegs[1][0] = ubegs[0][0] + us[1][0];
  us[1][0] = 0;
  banded_striped_epi8_seqalign_piecex_row_check_ubegs(us[1], ubegs[1], W);
  return ubegs[1][0];
}

//__attribute__((always_inline))
static inline __m256i
banded_striped_epi8_seqalign_piecex_row_cal_FPenetration_codes(i32 W, __m256i f,
                                                               int *ubegs[2],
                                                               i8 gape) {
  i8 __attribute__((aligned(WORDSIZE))) fs[WORDSIZE];
  f = _mm256_slli_si256_rpl(f, 1);
  _mm256_store_si256((__m256i *)fs, f);
  fs[0] = I8_MIN;
  int s = W * gape + fs[0] - (ubegs[0][1] - ubegs[0][0]);
  for (int i = 2; i < WORDSIZE; i++) {
    if (fs[i] < s)
      fs[i] = s;
    s = W * gape + fs[i] - (ubegs[0][i + 1] - ubegs[0][i]);
  }
  f = _mm256_load_si256((__m256i *)fs);
  return f;
}

static void banded_striped_epi8_seqalign_piecex_row_init(i8 *us, i8 *es,
                                                         int *ubegs, i32 W,
                                                         i8 max_nt, i8 min_nt,
                                                         i8 gapo1, i8 gape1) {
  __m256i MIN = _mm256_set1_epi8(I8_MIN);
  __m256i GAP = _mm256_set1_epi8(gape1);

  for (int k = 0; k < W; k++) {
    _mm256_storeu_si256((__m256i *)es + k, MIN);
    _mm256_storeu_si256((__m256i *)us + k, GAP);
  }

  us[0] = gapo1 + min_nt - max_nt;
  for (int k = 0; k < WORDSIZE; k++)
    ubegs[k] = gape1 * W;
  ubegs[0] += us[0] - gape1;

  u32 s = max_nt - min_nt;
  for (int k = 0; k < WORDSIZE; k++) {
    u32 t = ubegs[k];
    ubegs[k] = s;
    s += t;
  }
  ubegs[WORDSIZE] = s;
}

static inline void print_value(__m256i val) {
  char buf[WORDSIZE];
  _mm256_store_si256((__m256i *)buf, val);
  for (int i = 0; i < WORDSIZE; i++) {
    printf("%+d, ", buf[i]);
  }
  printf("\n");
}

static inline int
banded_striped_epi8_seqalign_piece1_row_cal(u8 base, i8 *us[2], i8 *es[2],
                                            int *ubegs[2], i8 *qprof, i8 gapo1,
                                            i8 gape1, i32 W, int rh) {
  // __m256i GapOE = _mm256_set1_epi8(gapo1 + gape1);
  __m256i GapOE = _mm256_set1_epi8(gapo1);
  __m256i GapE = _mm256_set1_epi8(gape1);

  // ::: max(h, e, f)

  i32 h0 = (rh - ubegs[0][0]) + qprof[0]; // h
  if (h0 >= us[0][0] + es[0][0]) {
    if (h0 > I8_MAX) h0 = I8_MAX; // score will loss, please never set rh - ph >= I8_MAX - base_match_score
  } else {
    h0 = I8_MIN;
  }

  __m256i f = _mm256_set1_epi8(I8_MIN);
  for (int i = 0; i < W; i++) {
    __m256i h = _mm256_loadu_si256((__m256i *)qprof + i); // * 4);// + base);
    if (i == 0) {
      h = _mm256_insert_epi8(h, h0, 0);
    }
    // print_value(h);
    __m256i u = _mm256_loadu_si256((__m256i *)us[0] + i);
    __m256i e = _mm256_loadu_si256((__m256i *)es[0] + i);

    // max h, e, f
    e = _mm256_adds_epi8(e, u);
    h = _mm256_max_epi8(e, h);
    h = _mm256_max_epi8(f, h);

    // preparing next f
    f = _mm256_adds_epi8(f, GapE);
    h = _mm256_adds_epi8(h, GapOE);
    f = _mm256_max_epi8(f, h);
    f = _mm256_subs_epi8(f, u);
  }

  f = banded_striped_epi8_seqalign_piecex_row_cal_FPenetration_codes(
      W, f, ubegs, gape1);

  // main loop
  // don't use the h from last W - 1 to calculate v = h - u, because this h
  // maybe updated when F-penetration will revise v after this loop
  __m256i h = _mm256_loadu_si256((__m256i *)qprof); // + base);
  __m256i v = _mm256_set1_epi8(0);
  for (int i = 0; i < W; i++) {
    __m256i z = _mm256_loadu_si256((__m256i *)qprof + i); // * 4);// + base);
    if (i == 0) {
      z = _mm256_insert_epi8(z, h0, 0);
    }
    __m256i u = _mm256_loadu_si256((__m256i *)us[0] + i);
    __m256i e = _mm256_loadu_si256((__m256i *)es[0] + i);

    // max(e, h)
    e = _mm256_adds_epi8(e, u);
    h = _mm256_max_epi8(e, z);

    // max(f, h)
    h = _mm256_max_epi8(f, h);

    // calculate u(x, y)
    v = _mm256_subs_epi8(h, v);
    _mm256_storeu_si256((__m256i *)us[1] + i, v);
    v = _mm256_subs_epi8(h, u);

    // calculate e(x, y)
    e = _mm256_adds_epi8(e, GapE);
    e = _mm256_subs_epi8(e, h);
    e = _mm256_max_epi8(e, GapOE);
    _mm256_storeu_si256((__m256i *)es[1] + i, e);

    // calculate f(x, y)
    f = _mm256_adds_epi8(f, GapE);
    h = _mm256_adds_epi8(h, GapOE);
    f = _mm256_max_epi8(f, h);
    f = _mm256_subs_epi8(f, u);
  }
  h = _mm256_subs_epi8(h, GapOE);
  __m256i last = _mm256_loadu_si256((__m256i *)us[0] + W - 1);
  return banded_striped_epi8_seqalign_piecex_row_cal_tail_codes(h, last, v, us,
                                                                ubegs, W);
}

static void calc_query_profile(i8 *qprof, u8 tbase, u8 *qseq, i64 qlen,
                               i64 unaligned, i8 match, i8 mismatch) {
  i64 W = qlen / WORDSIZE;
  for (int i = 0; i < qlen; i++) {
    int x = i % W;
    int y = i / W;
    if (i >= unaligned) {
      qprof[x * WORDSIZE + y] = I8_MIN;
      continue;
    }
    i8 r = qseq[i] == tbase;
    qprof[x * WORDSIZE + y] = r * match + (1 - r) * mismatch;
  }
}

#define swap(a, b)                                                             \
  do {                                                                         \
    __typeof__(a) _tmp = (a);                                                  \
    (a) = (b);                                                                 \
    (b) = _tmp;                                                                \
  } while (0)

static i64
do_the_thing(u8 *qseq, i64 unaligned,
             i64 qlen, // qlen must be a multiple of WORDSIZE (for now)
             u8 *tseq, i64 tlen, i8 match, i8 mismatch, i8 gapo, i8 gape) {
  i32 W = qlen / WORDSIZE;

  i8 *u1 = calloc(qlen, sizeof *u1);
  i8 *u2 = calloc(qlen, sizeof *u2);

  i8 *e1 = calloc(qlen, sizeof *e1);
  i8 *e2 = calloc(qlen, sizeof *e2);

  i32 *ubegs1 = calloc(WORDSIZE + 1, sizeof *ubegs1);
  i32 *ubegs2 = calloc(WORDSIZE + 1, sizeof *ubegs2);

  banded_striped_epi8_seqalign_piecex_row_init(u1, e1, ubegs1, W, match,
                                               mismatch, gapo, gape);
  i8 *qprof = calloc(qlen, sizeof *qprof);
  for (int i = 0; i < tlen; i++) {
    u8 tbase = tseq[i];
    calc_query_profile(qprof, tbase, qseq, qlen, unaligned, match, mismatch);
    i8 *us[2] = {u1, u2};
    i8 *es[2] = {e1, e2};
    i32 *ubegs[2] = {ubegs1, ubegs2};
    i32 rh = gapo + gape * (i - 1);
    if (i == 0) {
      rh = 0;
    }
    banded_striped_epi8_seqalign_piece1_row_cal(
        tseq[i], us, es, ubegs, qprof, gapo, gape, W, rh); // last arg is rh
    swap(u1, u2);
    swap(e1, e2);
    swap(ubegs1, ubegs2);
  }
  free(qprof);

  i64 score =
      banded_striped_epi8_seqalign_getscore(u1, ubegs1, W, unaligned);
  return score;
}
