/*
 * arMLEt - f3d_vtx_fx_neon.h: the F3D vertex pipeline in NEON (aarch64); included by f3d_vtx_fx.c
 *
 * The same arithmetic as the scalar reference (f3d_vtx_fx.c) with the 48-bit accumulator sums done in 64-bit lanes, two vertices per call (the 8 lanes of a vector register). Bit-exact with the
 * scalar form; checked against it (tools/f3d/test_fx -N under qemu-aarch64, and on the devices).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3D_VTX_FX_NEON_H
#define ARMLET_F3D_VTX_FX_NEON_H

#include <arm_neon.h>

/* VCH + VCL of 8 lanes against the W lane of each half: masks of the le and ge flags */
static inline void neon_clip_flags(uint16x8_t hi, uint16x8_t lo, uint16x8_t whi, uint16x8_t wlo, uint16x8_t *le_out, uint16x8_t *ge_out)
{
    int16x8_t vs = vreinterpretq_s16_u16(hi), vt = vreinterpretq_s16_u16(whi);
    uint16x8_t sign = vcltzq_s16(veorq_s16(vs, vt));
    int16x8_t negvt = vbslq_s16(sign, vnegq_s16(vt), vt);
    int16x8_t diff = vsubq_s16(vs, negvt);
    uint16x8_t diff_zero = vceqzq_s16(diff), diff_gt = vcgtzq_s16(diff), vt_neg = vcltzq_s16(vt);
    uint16x8_t ge = vbslq_u16(sign, vt_neg, vorrq_u16(diff_gt, diff_zero));
    uint16x8_t le = vbslq_u16(sign, vmvnq_u16(diff_gt), vt_neg);
    uint16x8_t vce = vandq_u16(sign, vceqq_s16(diff, vdupq_n_s16(-1)));
    uint16x8_t ne = vmvnq_u16(vorrq_u16(diff_zero, vce));
    uint16x8_t sneg = vbslq_u16(sign, vreinterpretq_u16_s16(vnegq_s16(vreinterpretq_s16_u16(wlo))), wlo);
    uint16x8_t ldiff = vsubq_u16(lo, sneg);
    uint16x8_t ncarry = vceqq_u16(ldiff, vqaddq_u16(lo, wlo));
    uint16x8_t ldz = vceqzq_u16(ldiff);
    uint16x8_t le_eq = vorrq_u16(vandq_u16(vandq_u16(ldz, ncarry), vmvnq_u16(vce)), vandq_u16(vce, vorrq_u16(ldz, ncarry)));
    uint16x8_t ge_eq = vcleq_u16(wlo, lo);
    le = vbslq_u16(vandq_u16(vmvnq_u16(ne), sign), le_eq, le);
    ge = vbslq_u16(vorrq_u16(sign, ne), ge, ge_eq);
    *le_out = le;
    *ge_out = ge;
}

static inline int64x2_t neon_wrap48(int64x2_t v) { return vshrq_n_s64(vshlq_n_s64(v, 16), 16); }
static inline int64x2_t neon_not64(int64x2_t v) { return veorq_s64(v, vdupq_n_s64(-1)); }

/* constants of one G_VTX call */
struct vtx_neon_ctx {
    int32x4_t m[3];            /* rows 0..2 of the matrix as 16.16 values, columns 0..3 */
    int64x2_t cpart[2];        /* row 3 fractions (columns 0,1 / 2,3) */
    int64x2_t cint[2];         /* row 3 integer parts << 16 */
    int16x8_t vscale, vtrans;  /* viewport, per lane */
    int16x8_t lim_lo, lim_hi;  /* screen clamps, per lane */
    uint16_t persp;
    int wrap;                  /* a matrix entry is large enough for the sums to leave 48 bits */
    int ok32;                  /* the matrix is small enough for the 32-bit transform below (|Mi + carry| <= 16384) */
    int16x4_t mip[3];          /* Mi + (Mf >= 0x8000): the fraction as a signed value, its borrow moved into the integer part */
    int16x4_t mfs[3];          /* Mf as a signed 16-bit value */
    int32x4_t h3, l3, mi3;     /* row 3: the borrow of its fraction, the signed fraction, the integer part */
};

struct vtx_neon_out {
    uint16x8_t hi, lo;         /* clip coordinates; lanes 0..3 the first vertex, 4..7 the second */
    uint16x8_t shi, slo;       /* clip coordinates over W, times the perspective normalisation */
    int16x8_t scr;             /* screen x, y, z, w per lane */
    uint16x8_t vlo;            /* the fractions of the screen coordinates, all lanes */
    unsigned flags;            /* VCC as CFC2 reads it */
};

static inline void neon_wlanes(uint16x8_t v, uint16x8_t *w)
{
    *w = vcombine_u16(vdup_lane_u16(vget_low_u16(v), 3), vdup_lane_u16(vget_high_u16(v), 3));
}

/* hi:lo (8 lanes) times a 16-bit unsigned factor p / 65536 -> int32 per lane: the 32-bit result acc = hi * p + (lo * p >> 16) */
static inline void neon_mulp(uint16x8_t hi, uint16x8_t lo, uint16_t p, int32x4_t *acc_lo4, int32x4_t *acc_hi4)
{
    int32x4_t p32 = vdupq_n_s32((int32_t)p);
    uint16x4_t p16 = vdup_n_u16(p);
    int32x4_t a0 = vmulq_s32(vmovl_s16(vreinterpret_s16_u16(vget_low_u16(hi))), p32);
    int32x4_t a1 = vmulq_s32(vmovl_s16(vreinterpret_s16_u16(vget_high_u16(hi))), p32);
    a0 = vaddq_s32(a0, vreinterpretq_s32_u32(vshrq_n_u32(vmull_u16(vget_low_u16(lo), p16), 16)));
    a1 = vaddq_s32(a1, vreinterpretq_s32_u32(vshrq_n_u32(vmull_u16(vget_high_u16(lo), p16), 16)));
    *acc_lo4 = a0;
    *acc_hi4 = a1;
}

/* the clip coordinates of two vertices (x, y, z each) and their clip flags (VCC as CFC2 reads it) */
static inline void neon_transform(const struct vtx_neon_ctx *c, const int16_t px[2][3], uint16x8_t *hi_out, uint16x8_t *lo_out,
                                  unsigned *vcc_out)
{
    uint16x4_t his[2], los[2];
    for (int g = 0; g < 2; g++) {
        int32_t x = px[g][0], y = px[g][1], z = px[g][2];
        int32x2_t hs[2];
        uint32x2_t ls[2];
        for (int h = 0; h < 2; h++) {
            int32x2_t m0 = h ? vget_high_s32(c->m[0]) : vget_low_s32(c->m[0]);
            int32x2_t m1 = h ? vget_high_s32(c->m[1]) : vget_low_s32(c->m[1]);
            int32x2_t m2 = h ? vget_high_s32(c->m[2]) : vget_low_s32(c->m[2]);
            int64x2_t acc = vmlal_n_s32(vmlal_n_s32(vmlal_n_s32(c->cpart[h], m0, x), m1, y), m2, z);
            int64x2_t part = acc, full = vaddq_s64(acc, c->cint[h]);
            if (c->wrap) {
                part = neon_wrap48(part);
                full = neon_wrap48(full);
            }
            hs[h] = vshrn_n_s64(full, 16);
            /* the clamp of the partial sum (before the integer part of row 3 is added): its low 16 bits when it fits in 32 bits, else
             * 0 / 0xffff by its sign */
            int64x2_t s = vshrq_n_s64(part, 31);
            uint64x2_t fits = vorrq_u64(vceqzq_s64(s), vceqq_s64(s, vdupq_n_s64(-1)));
            uint64x2_t sat = vreinterpretq_u64_s64(neon_not64(vshrq_n_s64(part, 63)));
            ls[h] = vmovn_u64(vbslq_u64(fits, vreinterpretq_u64_s64(part), sat));
        }
        his[g] = vreinterpret_u16_s16(vqmovn_s32(vcombine_s32(hs[0], hs[1])));
        los[g] = vmovn_u32(vcombine_u32(ls[0], ls[1]));
    }
    uint16x8_t hi = vcombine_u16(his[0], his[1]), lo = vcombine_u16(los[0], los[1]);
    *hi_out = hi;
    *lo_out = lo;
    uint16x8_t whi, wlo, le, ge;
    neon_wlanes(hi, &whi);
    neon_wlanes(lo, &wlo);
    neon_clip_flags(hi, lo, whi, wlo, &le, &ge);
    static const uint16_t wt[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    uint16x8_t w = vld1q_u16(wt);
    *vcc_out = (unsigned)vaddvq_u16(vandq_u16(le, w)) | ((unsigned)vaddvq_u16(vandq_u16(ge, w)) << 8);
}

/* The same clip coordinates in 32-bit lanes, for |x|, |y|, |z| <= 16384 and a matrix with |Mi + borrow| <= 16384 (ok32): with Mf_s the fraction as a signed
 * 16-bit value, x * Mf = x * Mf_s + (x << 16) * borrow, so the 48-bit sum is H * 65536 + L with
 *   H = sum x * (Mi + borrow)   (+ the borrow of row 3),   L = sum x * Mf_s   (+ the signed fraction of row 3),
 * both inside 32 bits at these bounds; S = H + (L >> 16) is the sum shifted right by 16, the high result is S + Mi3 (saturated) and the low result
 * is the low half of L when S fits 16 bits (the sum before Mi3, as the microcode clamps it), else 0 / 0xffff by the sign of S. */
static inline void neon_transform32(const struct vtx_neon_ctx *c, const int16_t px[2][3], uint16x8_t *hi_out, uint16x8_t *lo_out, unsigned *vcc_out)
{
    uint16x4_t his[2], los[2];
    for (int g = 0; g < 2; g++) {
        int16_t x = px[g][0], y = px[g][1], z = px[g][2];
        int32x4_t H = vmlal_n_s16(vmlal_n_s16(vmull_n_s16(c->mip[0], x), c->mip[1], y), c->mip[2], z);
        int32x4_t L = vmlal_n_s16(vmlal_n_s16(vmull_n_s16(c->mfs[0], x), c->mfs[1], y), c->mfs[2], z);
        H = vaddq_s32(H, c->h3);
        L = vaddq_s32(L, c->l3);
        int32x4_t S = vaddq_s32(H, vshrq_n_s32(L, 16));
        int16x4_t s16 = vqmovn_s32(S);
        uint32x4_t fits = vceqq_s32(vmovl_s16(s16), S);
        uint16x4_t lo16 = vmovn_u32(vreinterpretq_u32_s32(L));
        uint16x4_t sat = vmovn_u32(vmvnq_u32(vreinterpretq_u32_s32(vshrq_n_s32(S, 31))));   /* S < 0: 0, else 0xffff */
        los[g] = vbsl_u16(vmovn_u32(fits), lo16, sat);
        his[g] = vreinterpret_u16_s16(vqmovn_s32(vaddq_s32(S, c->mi3)));
    }
    uint16x8_t hi = vcombine_u16(his[0], his[1]), lo = vcombine_u16(los[0], los[1]);
    *hi_out = hi;
    *lo_out = lo;
    uint16x8_t whi, wlo, le, ge;
    neon_wlanes(hi, &whi);
    neon_wlanes(lo, &wlo);
    neon_clip_flags(hi, lo, whi, wlo, &le, &ge);
    static const uint16_t wt[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    uint16x8_t w = vld1q_u16(wt);
    *vcc_out = (unsigned)vaddvq_u16(vandq_u16(le, w)) | ((unsigned)vaddvq_u16(vandq_u16(ge, w)) << 8);
}

/* clip coordinates over W, times the perspective normalisation, then the viewport: the screen coordinates of the pair.
 * rhc / rlo: the reciprocals of W (hi clamped to be non-negative, lo) of the two vertices. */
static inline void neon_screen(const struct vtx_neon_ctx *c, uint16x8_t hi, uint16x8_t lo, const uint16_t rhc[2], const uint16_t rlo[2],
                               struct vtx_neon_out *o)
{
    /* P = hi:lo as a signed 32-bit value (4 lanes at a time), R = rhc:rlo; full = P * R >> 16, partial = full - (hi * rhc << 16) */
    int32x4_t P[2];
    P[0] = vreinterpretq_s32_u32(vorrq_u32(vshll_n_u16(vget_low_u16(hi), 16), vmovl_u16(vget_low_u16(lo))));
    P[1] = vreinterpretq_s32_u32(vorrq_u32(vshll_n_u16(vget_high_u16(hi), 16), vmovl_u16(vget_high_u16(lo))));
    int16x8_t hs = vreinterpretq_s16_u16(hi);
    int32x2_t hs32[4];
    uint16x8_t shi_v, slo_v;
    int32x4_t sh_part[2], sl_part[2];
    (void)hs32;
    uint16x4_t shi4[2], slo4[2];
    for (int g = 0; g < 2; g++) {
        int32_t R = (int32_t)(((uint32_t)rhc[g] << 16) | rlo[g]);
        int16x4_t hg = g ? vget_high_s16(hs) : vget_low_s16(hs);
        int32x4_t hr = vmull_s16(hg, vdup_n_s16((int16_t)rhc[g]));  /* hi * rhc */
        int32x2_t fh[2];
        uint32x2_t fl[2];
        for (int h = 0; h < 2; h++) {
            int32x2_t Ph = h ? vget_high_s32(P[g]) : vget_low_s32(P[g]);
            int64x2_t full = vshrq_n_s64(vmull_n_s32(Ph, R), 16);
            int32x2_t hrh = h ? vget_high_s32(hr) : vget_low_s32(hr);
            int64x2_t part = vsubq_s64(full, vshll_n_s32(hrh, 16));
            fh[h] = vshrn_n_s64(full, 16);
            int64x2_t s = vshrq_n_s64(part, 31);
            uint64x2_t fits = vorrq_u64(vceqzq_s64(s), vceqq_s64(s, vdupq_n_s64(-1)));
            uint64x2_t sat = vreinterpretq_u64_s64(neon_not64(vshrq_n_s64(part, 63)));
            fl[h] = vmovn_u64(vbslq_u64(fits, vreinterpretq_u64_s64(part), sat));
        }
        shi4[g] = vreinterpret_u16_s16(vqmovn_s32(vcombine_s32(fh[0], fh[1])));
        slo4[g] = vmovn_u32(vcombine_u32(fl[0], fl[1]));
    }
    (void)sh_part; (void)sl_part;
    shi_v = vcombine_u16(shi4[0], shi4[1]);
    slo_v = vcombine_u16(slo4[0], slo4[1]);
    /* times the perspective normalisation again */
    int32x4_t a0, a1;
    neon_mulp(shi_v, slo_v, c->persp, &a0, &a1);
    uint16x8_t shi2 = vreinterpretq_u16_s16(vcombine_s16(vshrn_n_s32(a0, 16), vshrn_n_s32(a1, 16)));
    uint16x8_t slo2 = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(a0)), vmovn_u32(vreinterpretq_u32_s32(a1)));
    o->shi = shi2;
    o->slo = slo2;
    /* viewport: trans << 16 + scale * (shi2:slo2) */
    int32x4_t Q[2];
    Q[0] = vreinterpretq_s32_u32(vorrq_u32(vshll_n_u16(vget_low_u16(shi2), 16), vmovl_u16(vget_low_u16(slo2))));
    Q[1] = vreinterpretq_s32_u32(vorrq_u32(vshll_n_u16(vget_high_u16(shi2), 16), vmovl_u16(vget_high_u16(slo2))));
    int16x8_t vsc = c->vscale, vtr = c->vtrans;
    int16x4_t sc4[2] = {vget_low_s16(vsc), vget_high_s16(vsc)}, tr4[2] = {vget_low_s16(vtr), vget_high_s16(vtr)};
    int32x2_t vh[4];
    uint32x2_t vl[4];
    for (int q = 0; q < 2; q++) {
        int32x4_t sc32 = vmovl_s16(sc4[q]), tr32 = vmovl_s16(tr4[q]);
        for (int h = 0; h < 2; h++) {
            int32x2_t Qh = h ? vget_high_s32(Q[q]) : vget_low_s32(Q[q]);
            int32x2_t sch = h ? vget_high_s32(sc32) : vget_low_s32(sc32);
            int32x2_t trh = h ? vget_high_s32(tr32) : vget_low_s32(tr32);
            int64x2_t acc = vaddq_s64(vshll_n_s32(trh, 16), vmull_s32(Qh, sch));
            vh[2 * q + h] = vshrn_n_s64(acc, 16);
            int64x2_t s = vshrq_n_s64(acc, 31);
            uint64x2_t fits = vorrq_u64(vceqzq_s64(s), vceqq_s64(s, vdupq_n_s64(-1)));
            uint64x2_t sat = vreinterpretq_u64_s64(neon_not64(vshrq_n_s64(acc, 63)));
            vl[2 * q + h] = vmovn_u64(vbslq_u64(fits, vreinterpretq_u64_s64(acc), sat));
        }
    }
    int16x8_t scr = vcombine_s16(vqmovn_s32(vcombine_s32(vh[0], vh[1])), vqmovn_s32(vcombine_s32(vh[2], vh[3])));
    if (F3D_FX_MUTATE == 18) scr = vaddq_s16(scr, vdupq_n_s16(1));   /* mutation 18: screen coordinates one off */
    scr = vmaxq_s16(scr, c->lim_lo);
    scr = vminq_s16(scr, c->lim_hi);
    o->scr = scr;
    o->vlo = vcombine_u16(vmovn_u32(vcombine_u32(vl[0], vl[1])), vmovn_u32(vcombine_u32(vl[2], vl[3])));
}


#endif
