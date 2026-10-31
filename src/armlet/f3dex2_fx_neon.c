/*
 * arMLEt - f3dex2_fx_neon.c: the NEON forms of the F3DEX2 geometry functions (f3dex2_fx.h)
 *
 * The same arithmetic as f3dex2_fx.c, four lanes at a time: the 16-bit halves of the (integer, fraction) pairs are multiplied with the widening multiplies, the sums of the 48-bit accumulator are
 * 64-bit lanes (two int64x2 for four lanes), the saturating readouts are the saturating narrows. tools/hle/test_fx2.c checks every function bit for bit against the scalar form.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifdef __aarch64__
#include <arm_neon.h>
#include <string.h>

#include "f3dex2_fx.h"
#include "f3dex2_fx_int.h"
#include "hle_fx.h"
#include "rej_fx.h"

/* test builds only (-DFX2_NEON_MUTATE=n): a deliberate error in the NEON form only, to show that tools/hle/test_fx2.c catches it */
#ifdef FX2_NEON_MUTATE
#define NMUT(n, ok, bad) (FX2_NEON_MUTATE == (n) ? (bad) : (ok))
#else
#define NMUT(n, ok, bad) (ok)
#endif

#define NI static inline __attribute__((always_inline))

/* HLE_FX2_SCALAR=1: the scalar forms (A/B timing on a device) */
#include <stdlib.h>
static int g_scalar = -1;
NI int use_scalar(void)
{
    if (__builtin_expect(g_scalar < 0, 0))
        g_scalar = getenv("HLE_FX2_SCALAR") != NULL;
    return g_scalar;
}

/* four lanes of 48-bit accumulator in two int64x2 */
typedef struct {
    int64x2_t lo, hi;
} v64;

NI v64 v64_add(v64 a, v64 b)
{
    v64 r = { vaddq_s64(a.lo, b.lo), vaddq_s64(a.hi, b.hi) };
    return r;
}
NI v64 v64_wrap48(v64 a)
{
    v64 r = { vshrq_n_s64(vshlq_n_s64(a.lo, 16), 16), vshrq_n_s64(vshlq_n_s64(a.hi, 16), 16) };
    return NMUT(3, r, a);
}
NI v64 v64_from_s32(int32x4_t v)
{
    v64 r = { vmovl_s32(vget_low_s32(v)), vmovl_high_s32(v) };
    return r;
}
NI v64 v64_addw_s32(v64 a, int32x4_t v)
{
    v64 r = { vaddw_s32(a.lo, vget_low_s32(v)), vaddw_high_s32(a.hi, v) };
    return r;
}
NI v64 v64_shl16_s32(int32x4_t v)
{
    v64 r = { vshll_n_s32(vget_low_s32(v), 16), vshll_high_n_s32(v, 16) };
    return r;
}
NI v64 v64_zero(void)
{
    v64 r = { vdupq_n_s64(0), vdupq_n_s64(0) };
    return r;
}
/* VMUDH...: bits 47..16, saturated to s16 */
NI int16x4_t sclamp4(v64 a)
{
    return vqmovn_s32(vcombine_s32(vqmovn_s64(vshrq_n_s64(a.lo, 16)), vqmovn_s64(vshrq_n_s64(a.hi, 16))));
}
/* low 16 bits, 0xffff / 0 when the accumulator does not fit in 32 bits (the saturation of the narrow gives exactly that) */
NI uint16x4_t uclamp4(v64 a)
{
    return NMUT(4, vmovn_u32(vreinterpretq_u32_s32(vcombine_s32(vqmovn_s64(a.lo), vqmovn_s64(a.hi)))), vmovn_u32(vreinterpretq_u32_s32(vcombine_s32(vmovn_s64(a.lo), vmovn_s64(a.hi)))));
}

/* t = base + (af*bf >> 16) + ai*bf + af*bi; full = t + (ai*bi << 16); each wrapped to 48 bits. b is a scalar pair. */
NI void mm4s(v64 base, int16x4_t ai, uint16x4_t af, int16_t bi, uint16_t bf, v64 *t, v64 *full)
{
    uint32x4_t p1 = NMUT(2, vshrq_n_u32(vmull_n_u16(af, bf), 16), vshrq_n_u32(vmull_n_u16(af, bf), 15));
    int32x4_t p2 = vmulq_n_s32(vmovl_s16(ai), (int32_t)bf);
    int32x4_t p3 = vmulq_n_s32(vreinterpretq_s32_u32(vmovl_u16(af)), (int32_t)bi);
    int32x4_t p4 = vmull_n_s16(ai, bi);
    v64 tt = v64_addw_s32(v64_addw_s32(v64_addw_s32(base, vreinterpretq_s32_u32(p1)), p2), p3);
    tt = v64_wrap48(tt);
    *t = tt;
    *full = v64_wrap48(v64_add(tt, v64_shl16_s32(p4)));
}
/* the same with a vector b */
NI void mm4v(v64 base, int16x4_t ai, uint16x4_t af, int16x4_t bi, uint16x4_t bf, v64 *t, v64 *full)
{
    uint32x4_t p1 = vshrq_n_u32(vmull_u16(af, bf), 16);
    int32x4_t p2 = vmulq_s32(vmovl_s16(ai), vreinterpretq_s32_u32(vmovl_u16(bf)));
    int32x4_t p3 = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(af)), vmovl_s16(bi));
    int32x4_t p4 = vmull_s16(ai, bi);
    v64 tt = v64_addw_s32(v64_addw_s32(v64_addw_s32(base, vreinterpretq_s32_u32(p1)), p2), p3);
    tt = v64_wrap48(tt);
    *t = tt;
    *full = v64_wrap48(v64_add(tt, v64_shl16_s32(p4)));
}

/* ---- triangle attributes ------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* four lanes of the lane loop: att0/1/2 = the pairs of H, M, L for the lanes; returns st, dX, dY, dE as (int, frac) vectors */
struct pr4 {
    int16x4_t i;
    uint16x4_t f;
};
NI int32x4_t pack32(int16x4_t i, uint16x4_t f)
{
    uint16x4x2_t z = vzip_u16(f, vreinterpret_u16_s16(i));
    return vreinterpretq_s32_u16(vcombine_u16(z.val[0], z.val[1]));
}
NI void unpack32(int32x4_t v, int16x4_t *i, uint16x4_t *f)
{
    uint16x8_t u = vreinterpretq_u16_s32(v);
    uint16x4_t lo = vget_low_u16(u), hi = vget_high_u16(u);
    uint16x4x2_t z = vuzp_u16(lo, hi);
    *f = z.val[0];
    *i = vreinterpret_s16_u16(z.val[1]);
}
/* VSUBC / VSUB on pairs: the integer half saturates */
NI void fsub4(int16x4_t ai, uint16x4_t af, int16x4_t bi, uint16x4_t bf, int16x4_t *ri, uint16x4_t *rf)
{
    uint16x4_t borrow = vclt_u16(af, bf);                                   /* 0xffff where af < bf */
    int32x4_t d = vsubl_s16(ai, bi);
    d = vaddq_s32(d, NMUT(1, vmovl_s16(vreinterpret_s16_u16(borrow)), vdupq_n_s32(0)));      /* - borrow */
    *ri = vqmovn_s32(d);
    *rf = vsub_u16(af, bf);
}

/* The lane loop for NH halves of four lanes (lanes lo0 .. lo0 + 4 NH - 1), the stages of the halves interleaved so that the chains of dependent multiplies of the two overlap (the in-order cores are
 * latency bound). att0/1/2 = the pairs of H, M, L for the lanes; returns st, dX, dY, dE as (int, frac) vectors. */
NI void attrs_n(const int NH, const struct fx2_pair att[3][8], unsigned lo0, struct fx2_pair R1, struct fx2_pair sl, struct fx2_pair C0, int16_t dyMH, int16_t nyLH, int16_t dxLH, int16_t nxMH,
                struct pr4 *stv, struct pr4 *dX, struct pr4 *dY, struct pr4 *dE)
{
    int16x4_t a0i[2], a1i[2], a2i[2], dLHi[2], dMHi[2], nxi[2], nyi[2];
    uint16x4_t a0f[2], a1f[2], a2f[2], dLHf[2], dMHf[2], nxf[2], nyf[2];
    int32x4_t pvLH[2], pvMH[2], nx32[2], ny32[2];
    v64 a1[2], a2[2], t[2], full[2], t2[2], f2[2], z = v64_zero();
    int h;

    for (h = 0; h < NH; h++) {                          /* the (int, frac) pairs of 4 lanes of each vertex: vld2 de-interleaves them */
        int16x4x2_t v0 = vld2_s16((const int16_t *)&att[0][lo0 + 4 * h]), v1 = vld2_s16((const int16_t *)&att[1][lo0 + 4 * h]), v2 = vld2_s16((const int16_t *)&att[2][lo0 + 4 * h]);
        a0i[h] = v0.val[0];
        a0f[h] = vreinterpret_u16_s16(v0.val[1]);
        a1i[h] = v1.val[0];
        a1f[h] = vreinterpret_u16_s16(v1.val[1]);
        a2i[h] = v2.val[0];
        a2f[h] = vreinterpret_u16_s16(v2.val[1]);
    }
    for (h = 0; h < NH; h++) {
        fsub4(a2i[h], a2f[h], a0i[h], a0f[h], &dLHi[h], &dLHf[h]);
        fsub4(a1i[h], a1f[h], a0i[h], a0f[h], &dMHi[h], &dMHf[h]);
    }
    for (h = 0; h < NH; h++) {
        pvLH[h] = pack32(dLHi[h], dLHf[h]);
        pvMH[h] = pack32(dMHi[h], dMHf[h]);
    }
    for (h = 0; h < NH; h++) {                          /* a1 = pv(dLH) * dyMH + pv(dMH) * nyLH; a2 = pv(dMH) * dxLH + pv(dLH) * nxMH, 48-bit */
        a1[h].lo = vmlal_n_s32(vmull_n_s32(vget_low_s32(pvLH[h]), dyMH), vget_low_s32(pvMH[h]), nyLH);
        a1[h].hi = vmlal_n_s32(vmull_n_s32(vget_high_s32(pvLH[h]), dyMH), vget_high_s32(pvMH[h]), nyLH);
        a2[h].lo = vmlal_n_s32(vmull_n_s32(vget_low_s32(pvMH[h]), dxLH), vget_low_s32(pvLH[h]), nxMH);
        a2[h].hi = vmlal_n_s32(vmull_n_s32(vget_high_s32(pvMH[h]), dxLH), vget_high_s32(pvLH[h]), nxMH);
    }
    for (h = 0; h < NH; h++) {
        a1[h] = v64_wrap48(a1[h]);
        a2[h] = v64_wrap48(a2[h]);
    }
    for (h = 0; h < NH; h++) {                          /* nx = (a1 >> 32, a1 >> 16): the low 32 bits of a1 >> 16 as a pair */
        nx32[h] = vcombine_s32(vmovn_s64(vshrq_n_s64(a1[h].lo, 16)), vmovn_s64(vshrq_n_s64(a1[h].hi, 16)));
        ny32[h] = vcombine_s32(vmovn_s64(vshrq_n_s64(a2[h].lo, 16)), vmovn_s64(vshrq_n_s64(a2[h].hi, 16)));
    }
    for (h = 0; h < NH; h++) {
        unpack32(nx32[h], &nxi[h], &nxf[h]);
        unpack32(ny32[h], &nyi[h], &nyf[h]);
    }
    for (h = 0; h < NH; h++) {
        mm4s(z, nxi[h], nxf[h], R1.i, R1.f, &t[h], &full[h]);
        mm4s(z, nyi[h], nyf[h], R1.i, R1.f, &t2[h], &f2[h]);
    }
    for (h = 0; h < NH; h++) {
        dX[h].i = sclamp4(full[h]);
        dX[h].f = uclamp4(t[h]);
        dY[h].i = sclamp4(f2[h]);
        dY[h].f = uclamp4(t2[h]);
    }
    for (h = 0; h < NH; h++)                            /* dE = dY + dX * DxHDy, on top of the accumulator the dY chain left */
        mm4s(f2[h], dX[h].i, dX[h].f, sl.i, sl.f, &t[h], &full[h]);
    for (h = 0; h < NH; h++) {
        dE[h].i = sclamp4(full[h]);
        dE[h].f = uclamp4(t[h]);
    }
    for (h = 0; h < NH; h++)                            /* st = att0 + dE * C */
        mm4s(v64_from_s32(pack32(a0i[h], a0f[h])), dE[h].i, dE[h].f, C0.i, C0.f, &t[h], &full[h]);
    for (h = 0; h < NH; h++) {
        stv[h].i = sclamp4(full[h]);
        stv[h].f = uclamp4(t[h]);
    }
}

NI void put_be16x4(uint8_t *p, int16x4_t v)
{
    vst1_u16((uint16_t *)p, vrev32_u16(vreinterpret_u16_s16(v)));          /* the host words of DMEM: the halves of each word swapped */
}

unsigned fx2_tri_attrs_neon(const struct fx2_tri_in *in, const struct fx2_tri_state *st, uint32_t *cmd)
{
    uint32_t g = in->g;
    unsigned shade = g & 4, zb = g & 1, tex = in->tex, o = 32, k;
    struct fx2_pair att[3][8], R1;
    struct pr4 s4[2], x4[2], y4[2], e4[2];
    unsigned need = fx2_tri_attr_prep(in, st, att, &R1);
    unsigned nh = (need & 0x0f ? 1 : 0) | (need & 0xf0 ? 2 : 0);

    if (nh == 3)
        attrs_n(2, att, 0, R1, st->slope[2], st->C0, st->dyMH, st->nyLH, st->dxLH, st->nxMH, s4, x4, y4, e4);
    else if (nh == 1)
        attrs_n(1, att, 0, R1, st->slope[2], st->C0, st->dyMH, st->nyLH, st->dxLH, st->nxMH, &s4[0], &x4[0], &y4[0], &e4[0]);
    else if (nh == 2)
        attrs_n(1, att, 4, R1, st->slope[2], st->C0, st->dyMH, st->nyLH, st->dxLH, st->nxMH, &s4[1], &x4[1], &y4[1], &e4[1]);
    for (k = 0; k < 2; k++) {
        unsigned base;
        if (k == 0) {
            if (!shade)
                continue;
            base = o;
        } else {
            if (!tex)
                continue;
            base = o + (shade ? 64 : 0);
        }
        put_be16x4((uint8_t *)cmd + base, s4[k].i);
        put_be16x4((uint8_t *)cmd + base + 8, x4[k].i);
        put_be16x4((uint8_t *)cmd + base + 16, vreinterpret_s16_u16(s4[k].f));
        put_be16x4((uint8_t *)cmd + base + 24, vreinterpret_s16_u16(x4[k].f));
        put_be16x4((uint8_t *)cmd + base + 32, e4[k].i);
        put_be16x4((uint8_t *)cmd + base + 40, y4[k].i);
        put_be16x4((uint8_t *)cmd + base + 48, vreinterpret_s16_u16(e4[k].f));
        put_be16x4((uint8_t *)cmd + base + 56, vreinterpret_s16_u16(y4[k].f));
    }
    o += (shade ? 64 : 0) + (tex ? 64 : 0);
    if (zb) {                                           /* lane 7 of the second half: z */
        struct fx2_pair dX7 = { x4[1].i[3], x4[1].f[3] }, dY7 = { y4[1].i[3], y4[1].f[3] }, dE7 = { e4[1].i[3], e4[1].f[3] }, st7 = { s4[1].i[3], s4[1].f[3] };
        struct fx2_pair zst, zx, zy, ze;
        hle_acc a;
        uint16_t v10 = (uint16_t)((uint32_t)dE7.f * (uint32_t)(int32_t)(int16_t)st->C0.f);
        a = (hle_acc)dX7.f * 32 + (((hle_acc)dX7.i * 32) << 16);
        zx.i = hle_sclamp(a);
        zx.f = (uint16_t)(dX7.f * 32);
        a = (hle_acc)dY7.f * 32 + (((hle_acc)dY7.i * 32) << 16);
        zy.i = hle_sclamp(a);
        zy.f = (uint16_t)(dY7.f * 32);
        a = (hle_acc)dE7.f * 32 + (((hle_acc)dE7.i * 32) << 16);
        ze.i = hle_sclamp(a);
        ze.f = (uint16_t)(dE7.f * 32);
        a = (hle_acc)(((uint32_t)v10 * 32) >> 16) + (hle_acc)st7.f * 32 + (((hle_acc)st7.i * 32) << 16);
        zst.i = hle_sclamp(a);
        zst.f = hle_uclamp(a - (((hle_acc)st7.i * 32) << 16));      /* before the integer product (see f3dex2_fx.c) */
        {
            uint16_t zv[8] = { (uint16_t)zst.i, zst.f, (uint16_t)zx.i, zx.f, (uint16_t)ze.i, ze.f, (uint16_t)zy.i, zy.f };
            vst1q_u16((uint16_t *)((uint8_t *)cmd + o), vrev32q_u16(vld1q_u16(zv)));
        }
        o += 16;
    }
    return o;
}

int fx2_tri_setup(const struct fx2_tri_in *in, uint32_t *cmd, unsigned *len)
{
    struct fx2_tri_state st;
    int r;
    if (use_scalar())
        return fx2_tri_setup_scalar(in, cmd, len);
    r = fx2_tri_prepare(in, cmd, &st);
    if (r != FX2_TRI_DRAW)
        return r;
    *len = (in->g & 4) || in->tex || (in->g & 1) ? fx2_tri_attrs_neon(in, &st, cmd) : 32;
    return FX2_TRI_DRAW;
}


/* ---- lighting ------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* The lit colour is the scalar form: a NEON form (the dot products with the horizontal adds, the colours in lanes) was measured on the A55 at 86 ns a vertex against the scalar form's 62. */
NI void lit_vertex(const struct fx2_vctx *cx, unsigned g, const int8_t nb[3], uint8_t alpha, uint8_t out[4], int texgen, int16_t sgen[2])
{
    fx2_lit_vertex_scalar(cx, g, nb, alpha, out, texgen, sgen);
}

/* ---- vertices ------------------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* a = ((f * k) >> 16) + i * k per lane, k an unsigned 16-bit scalar (the perspective normalisation) */
NI v64 mulk4(int16x4_t i, uint16x4_t f, uint16_t k)
{
    uint32x4_t p1 = vshrq_n_u32(vmull_n_u16(f, k), 16);
    int32x4_t p2 = vmulq_n_s32(vmovl_s16(i), (int32_t)k);
    return v64_addw_s32(v64_from_s32(p2), vreinterpretq_s32_u32(p1));
}

/* the clip-flag bits of the four lanes of X against W (lane 3 of W4): bit k "le" of lane k, bit 8 + k "ge" */
NI unsigned clipbits4(int32x4_t X, int32x4_t W)
{
    static const uint32_t wl[4] = { 1, 2, 4, 8 }, wg[4] = { 0x100, 0x200, 0x400, 0x800 };
    uint32x4_t xneg = vcltzq_s32(X), wneg = vcltzq_s32(W);
    uint32x4_t opp = veorq_u32(xneg, wneg);
    uint32x4_t le_o = vcleq_s32(vaddq_s32(X, W), vdupq_n_s32(NMUT(5, 0, -1)));
    uint32x4_t ge_o = vcgeq_s32(vsubq_s32(X, W), vdupq_n_s32(0));
    uint32x4_t le = vbslq_u32(opp, le_o, wneg), ge = vbslq_u32(opp, wneg, ge_o);
    return vaddvq_u32(vorrq_u32(vandq_u32(le, vld1q_u32(wl)), vandq_u32(ge, vld1q_u32(wg))));
}

/* P = the clip coordinates: (Mi[3] << 16) + Mf[3] + sum over j < 3 of Mf[j] v[j] + (Mi[j] v[j]) << 16, per column k (the four lanes) */
NI void vtx_P4(const struct fx2_mtx *mp, const int16_t v[3], int16x4_t *Pi, uint16x4_t *Pf)
{
    v64 acc = v64_from_s32(pack32(vld1_s16(mp->i[3]), vld1_u16(mp->f[3])));
    v64 last = v64_zero();
    unsigned j;
    for (j = 0; j < 3; j++) {
        int16x4_t mi = vld1_s16(mp->i[j]);
        uint16x4_t mf = vld1_u16(mp->f[j]);
        int32x4_t pf = vmulq_n_s32(vreinterpretq_s32_u32(vmovl_u16(mf)), (int32_t)v[j]);
        int32x4_t pi = vmull_n_s16(mi, v[j]);
        last = v64_shl16_s32(pi);
        acc = v64_wrap48(v64_add(v64_addw_s32(acc, pf), last));
    }
    *Pi = sclamp4(acc);
    {   /* uclamp(wrap48(acc - last)) */
        v64 d = { vsubq_s64(acc.lo, last.lo), vsubq_s64(acc.hi, last.hi) };
        *Pf = uclamp4(v64_wrap48(d));
    }
}

/* ---- one pair of vertices, the four coordinates in lanes ------------------------------------------------------------------------------------------------------------------------
 * The form for the short commands (Gauntlet Legends: 58 percent of its G_VTX commands load one vertex, 19 percent two): the matrix product, the clip flags and Q = P * pn in 64-bit lanes per vertex, the
 * reciprocal and its refinement scalar, the projection and the viewport in lanes again. The results are those of vtx_project in f3dex2_fx.c. */

static void vtx_xform_pair(const struct fx2_vctx *c, const struct fx2_vin *in, unsigned n, struct fx2_vtx *out)
{
    unsigned vcc_a[2], vcc_b[2], i, nn = n + (n & 1);
    int16x4_t Pi[2];
    uint16x4_t Pf[2];
    int16_t Qi3[2], T_i[2], R1i[2];
    uint16_t Qf3[2], T_f[2], R1f[2];
    uint32_t R0[2];
    uint16x4_t Pf_dummy;
    (void)Pf_dummy;

    {   /* stage 1 */
        const int16_t sc[4] = { c->scale[0], c->scale[1], c->scale[2], c->scale[3] };
        (void)sc;
    }
    for (i = 0; i < nn; i++) {
        const struct fx2_vin *vi = &in[i];
        int16_t v[3] = { vi->x, vi->y, vi->z };
        int32x4_t X, W;
        int16_t P3i;
        uint16_t P3f, P2f;
        hle_acc prod;
        int32_t W2;
        int32x4_t X2;
        vtx_P4(&c->mp, v, &Pi[i], &Pf[i]);
        X = pack32(Pi[i], Pf[i]);
        W = vdupq_laneq_s32(X, 3);
        P3i = vget_lane_s16(Pi[i], 3);
        P3f = vget_lane_u16(Pf[i], 3);
        P2f = vget_lane_u16(Pf[i], 2);
        prod = (hle_acc)P3f * c->ratio + (((hle_acc)P3i * c->ratio) << 16);
        W2 = (int32_t)((uint32_t)(uint16_t)hle_sclamp(prod) << 16 | (uint16_t)prod);
        X2 = vsetq_lane_s32((int32_t)((uint32_t)(uint16_t)P3i << 16 | P2f), X, 3);
        vcc_a[i] = clipbits4(X, W);
        vcc_b[i] = clipbits4(X2, vdupq_n_s32(W2));
        if (i < n) {
            v64 Q = mulk4(Pi[i], Pf[i], c->pn);
            int64_t q3 = vgetq_lane_s64(Q.hi, 1);             /* lane 3 only */
            Qi3[i] = hle_sclamp(q3);
            Qf3[i] = hle_uclamp(q3);
            R0[i] = hle_rcp32((int32_t)((uint32_t)(uint16_t)Qi3[i] << 16 | Qf3[i]));
        }
    }
#pragma GCC unroll 4
    for (i = 0; i < n; i++) {                           /* T = Q3 * R0; E = 4 - 4 T */
        int16_t R0i = (int16_t)(R0[i] >> 16);
        uint16_t R0f = (uint16_t)R0[i];
        hle_acc tt = hle_wrap48(((hle_acc)Qf3[i] * R0f >> 16) + (hle_acc)Qi3[i] * R0f + (hle_acc)Qf3[i] * R0i);
        hle_acc fu = hle_wrap48(tt + (((hle_acc)Qi3[i] * R0i) << 16));
        T_i[i] = hle_sclamp(fu);
        T_f[i] = hle_uclamp(tt);
    }
#pragma GCC unroll 4
    for (i = 0; i < n; i++) {                           /* R1 = E * R0 */
        int16_t R0i = (int16_t)(R0[i] >> 16), Ei;
        uint16_t R0f = (uint16_t)R0[i], Ef;
        hle_acc e1 = ((hle_acc)4 << 16) - (hle_acc)4 * T_f[i];
        hle_acc e2 = e1 - (((hle_acc)4 * T_i[i]) << 16);
        hle_acc t2, f2;
        Ei = hle_sclamp(e2);
        Ef = hle_uclamp(e1);
        t2 = hle_wrap48(((hle_acc)Ef * R0f >> 16) + (hle_acc)Ei * R0f + (hle_acc)Ef * R0i);
        f2 = hle_wrap48(t2 + (((hle_acc)Ei * R0i) << 16));
        R1i[i] = hle_sclamp(f2);
        R1f[i] = hle_uclamp(t2);
    }
    {
        int16_t sc[4] = { c->scale[0], c->scale[1], c->scale[2], c->scale[3] };
        int32_t base32[4] = { c->trans[0], c->trans[1], c->trans[2], c->trans[3] + (int32_t)c->c31_3 };
        v64 b = v64_shl16_s32(vld1q_s32(base32)), z = v64_zero();
        int16x4_t scale = vld1_s16(sc);
        for (i = 0; i < n; i++) {                       /* stage 5 and the rest of the record */
            const struct fx2_vin *vi = &in[i];
            struct fx2_vtx *o = &out[i];
            int16_t s = vi->s, tt = vi->t, fog;
            uint8_t col[4] = { vi->c[0], vi->c[1], vi->c[2], vi->c[3] };
            int16_t rint = vget_lane_s16(Pi[i], 3) >= 0 ? R1i[i] : 0x7fff;
            v64 t, fl, n2, t1, f1;
            int16x4_t ndci, nni, Si;
            uint16x4_t ndcf, nni_f, Sf;
            int32x4_t pf;
            if ((c->mode & FX2_LIGHTING)) {
                int8_t nb[3] = { (int8_t)vi->c[0], (int8_t)vi->c[1], (int8_t)vi->c[2] };
                uint8_t lo[4];
                int16_t sg[2] = { 0, 0 };
                lit_vertex(c, (i & 1) ? 0 : 1, nb, vi->c[3], lo, (c->mode & FX2_TEXGEN) != 0, sg);
                memcpy(col, lo, 4);
                if (c->mode & FX2_TEXGEN) {
                    s = sg[0];
                    tt = sg[1];
                }
            }
            mm4s(z, Pi[i], Pf[i], rint, R1f[i], &t, &fl);
            ndci = sclamp4(fl);
            ndcf = uclamp4(t);
            n2 = mulk4(ndci, ndcf, c->pn);
            {
                int16x4_t ni = sclamp4(n2);
                uint16x4_t nf = uclamp4(n2);
                nni = vcopy_lane_s16(ni, 3, ni, 2);
                nni_f = vcopy_lane_u16(nf, 3, nf, 2);
            }
            pf = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(nni_f)), vmovl_s16(scale));
            t1 = v64_wrap48(v64_addw_s32(b, pf));
            f1 = v64_wrap48(v64_add(t1, v64_shl16_s32(vmull_s16(nni, scale))));
            Si = sclamp4(f1);
            Sf = uclamp4(t1);
            vst1_s16(o->cx, Pi[i]);
            vst1_u16(o->cf, Pf[i]);
            o->sx = vget_lane_s16(Si, 0);
            o->sy = vget_lane_s16(Si, 1);
            {
                int16_t z2 = vget_lane_s16(Si, 2);
                o->szi = z2 < 0 && !NMUT(6, c->gen204, 0) ? 0 : z2;
            }
            o->szf = vget_lane_u16(Sf, 2);
            o->rwi = R1i[i];
            o->rwf = R1f[i];
            fog = vget_lane_s16(Si, 3);
            if (c->mode & FX2_FOG)
                col[3] = (uint8_t)(fog < 0x7f00 ? 0x7f00 : fog);
            memcpy(o->col, col, 4);
            o->s = (int16_t)(((int32_t)s * c->sc_s) >> 16);
            o->t = (int16_t)(((int32_t)tt * c->sc_t) >> 16);
            (void)Sf;
        }
    }
    for (i = 0; i < n; i++) {
        unsigned pa = i & ~1u, pb = pa + 1;
        unsigned ca = (vcc_a[pa] & 0x0f0f) | ((vcc_a[pb] & 0x0f0f) << 4);
        unsigned cb = (vcc_b[pa] & 0x0f0f) | ((vcc_b[pb] & 0x0f0f) << 4);
        out[i].flags = (i & 1) ? (uint32_t)(cb << 16 | ca) : (uint32_t)(((cb << 4) & 0xffff) << 16 | ((ca << 4) & 0xffff));
    }
}

/* ---- the vertex pipeline, four vertices per pass ---------------------------------------------------------------------------------------------------------------------------------
 * One vertex per 32-bit lane. The 48-bit accumulator sums are never formed: every readout the scalar form takes of them (hle_sclamp: bits 47..16 saturated to s16; hle_uclamp: bits 15..0, or
 * 0 / 0xffff when the sum does not fit in s32) is computed from the high and the low 16 bits of the 32-bit terms. With a sum of 32-bit terms t_j plus a base b << 16:
 *   floor(sum / 2^16) = b + sum_j (t_j >> 16) + ((sum_j (t_j & 0xffff)) >> 16)       (exact; the carry of the low halves is at most 3)
 *   sum & 0xffff = (sum_j (t_j & 0xffff)) & 0xffff
 * and the 48-bit wrap is the 32-bit wrap of floor(sum / 2^16), so sclamp = vqmovn of that 32-bit value and uclamp = the low halfword when floor(sum / 2^16) fits in s16, else the sign's constant.
 * The same bit-exact results as vtx_one / vtx_project in f3dex2_fx.c (tools/hle/test_fx2.c). The lit colour stays scalar per vertex. */
NI uint32x4_t sel_sat16(int32x4_t hi, uint32x4_t lo)
{   /* hle_uclamp of the sum whose floor(/2^16) is hi (32-bit) and whose low halfword is lo: lo when hi fits in s16, else 0 (negative) / 0xffff */
    uint32x4_t fits = vceqq_s32(hi, vmovl_s16(vqmovn_s32(hi)));
    uint32x4_t sat = vbicq_u32(vdupq_n_u32(0xffff), vreinterpretq_u32_s32(vshrq_n_s32(hi, 31)));
    return vbslq_u32(fits, lo, sat);
}
/* the pair product chain of f3dex2_fx.c (mm + rd) on four lanes: tt = (af bf >> 16) + ai bf + af bi, full = tt + (ai bi << 16); ri = sclamp(full), rf = uclamp(tt). b as 16-bit vectors (bi16, bf16)
 * and widened to 32-bit lanes (bi32 = bi16 sign-extended, bf32 = bf16 zero-extended) */
NI void chain4(int16x4_t ai, uint16x4_t af, int16x4_t bi16, uint16x4_t bf16, int32x4_t bi32, int32x4_t bf32, int16x4_t *ri, uint16x4_t *rf)
{
    uint32x4_t m16 = vdupq_n_u32(0xffff);
    uint32x4_t t1 = vshrq_n_u32(vmull_u16(af, bf16), 16);
    int32x4_t t2 = vmulq_s32(vmovl_s16(ai), bf32);
    int32x4_t t3 = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(af)), bi32);
    int32x4_t p4 = vmull_s16(ai, bi16);
    uint32x4_t lo = vaddq_u32(vaddq_u32(t1, vandq_u32(vreinterpretq_u32_s32(t2), m16)), vandq_u32(vreinterpretq_u32_s32(t3), m16));
    int32x4_t hi = vaddq_s32(vaddq_s32(vshrq_n_s32(t2, 16), vshrq_n_s32(t3, 16)), vreinterpretq_s32_u32(vshrq_n_u32(lo, 16)));
    *rf = vmovn_u32(sel_sat16(hi, vandq_u32(lo, m16)));
    *ri = vqmovn_s32(vaddq_s32(p4, hi));
}
/* a = (f k >> 16) + i k with the 16-bit scalar k (fits in s32): sclamp and uclamp of it */
NI void mulk4v(int16x4_t i, uint16x4_t f, uint16_t k, int16x4_t *ri, uint16x4_t *rf)
{
    int32x4_t a = vaddq_s32(vreinterpretq_s32_u32(vshrq_n_u32(vmull_n_u16(f, k), 16)), vmulq_n_s32(vmovl_s16(i), (int32_t)k));
    *ri = vqshrn_n_s32(a, 16);
    *rf = vmovn_u32(vreinterpretq_u32_s32(a));
}
NI int32x4_t pack32v(int16x4_t i, uint16x4_t f)
{
    return vreinterpretq_s32_u32(vorrq_u32(vshlq_n_u32(vmovl_u16(vreinterpret_u16_s16(i)), 16), vmovl_u16(f)));
}
/* the clip-flag masks of four lanes of X against W (hle_clip_le / hle_clip_ge) */
NI void clipmask4(int32x4_t X, int32x4_t W, uint32x4_t *le, uint32x4_t *ge)
{
    uint32x4_t xneg = vcltzq_s32(X), wneg = vcltzq_s32(W), opp = veorq_u32(xneg, wneg);
    uint32x4_t le_o = vclezq_s32(vaddq_s32(X, W)), ge_o = vcgezq_s32(vsubq_s32(X, W));
    *le = vbslq_u32(opp, le_o, wneg);
    *ge = vbslq_u32(opp, wneg, ge_o);
}

_Static_assert(sizeof(struct fx2_vtx) == 40 && sizeof(struct fx2_vin) == 16, "the records are copied as 8-byte groups / loaded as one vector");

void fx2_vtx_xform(const struct fx2_vctx *c, const struct fx2_vin *in, unsigned n, struct fx2_vtx *out)
{
    unsigned nn = n + (n & 1), nb = (nn + 3) & ~3u, b, i, k;
    const uint32x4_t m16 = vdupq_n_u32(0xffff);

    if (use_scalar()) {
        fx2_vtx_xform_scalar(c, in, n, out);
        return;
    }
    if (nn == 2) {                                      /* one or two vertices: the pair form (the block form costs its four lanes whatever n is) */
        vtx_xform_pair(c, in, n, out);
        return;
    }
    for (b = 0; b < nb; b += 4) {                      /* (in[] has 34 entries and n <= 32: the lanes past nn read what follows and are dropped) */
        int16x4_t V16[3], S16, T16, Pi[4], Qi3, R0i, R1i, rint, T_i, Ei, Si[4];
        uint16x4_t Pf[4], Qf3, R0f, R1f, T_f, Ef, Szf, C01, C23;
        int32x4_t V[3], X[4], W, W2, R0i32, R0f32, rint32, R1f32;
        uint32x4_t vcc_a, vcc_b, flags;
        uint32_t R0_a[4];
        uint8_t grp[5][4][8];
        unsigned m = n > b + 4 ? 4 : n - b;

        {   /* the four input records transposed: x y z (s t c0c1 c2c3) in lanes */
            int16x8_t r0 = vld1q_s16((const int16_t *)&in[b]), r1 = vld1q_s16((const int16_t *)&in[b + 1]), r2 = vld1q_s16((const int16_t *)&in[b + 2]), r3 = vld1q_s16((const int16_t *)&in[b + 3]);
            int16x4x2_t t01 = vtrn_s16(vget_low_s16(r0), vget_low_s16(r1)), t23 = vtrn_s16(vget_low_s16(r2), vget_low_s16(r3));
            int16x4x2_t h01 = vtrn_s16(vget_high_s16(r0), vget_high_s16(r1)), h23 = vtrn_s16(vget_high_s16(r2), vget_high_s16(r3));
            int32x2x2_t u0 = vtrn_s32(vreinterpret_s32_s16(t01.val[0]), vreinterpret_s32_s16(t23.val[0])), u1 = vtrn_s32(vreinterpret_s32_s16(t01.val[1]), vreinterpret_s32_s16(t23.val[1]));
            int32x2x2_t v0 = vtrn_s32(vreinterpret_s32_s16(h01.val[0]), vreinterpret_s32_s16(h23.val[0])), v1 = vtrn_s32(vreinterpret_s32_s16(h01.val[1]), vreinterpret_s32_s16(h23.val[1]));
            V16[0] = vreinterpret_s16_s32(u0.val[0]);
            V16[1] = vreinterpret_s16_s32(u1.val[0]);
            V16[2] = vreinterpret_s16_s32(u0.val[1]);
            S16 = vreinterpret_s16_s32(v0.val[0]);
            T16 = vreinterpret_s16_s32(v1.val[0]);
            C01 = vreinterpret_u16_s32(v0.val[1]);
            C23 = vreinterpret_u16_s32(v1.val[1]);
        }
        for (k = 0; k < 3; k++)
            V[k] = vmovl_s16(V16[k]);
        /* stage 1: the clip coordinates, per column */
        for (k = 0; k < 4; k++) {
            int32x4_t pf0 = vmulq_n_s32(V[0], (int32_t)c->mp.f[0][k]), pf1 = vmulq_n_s32(V[1], (int32_t)c->mp.f[1][k]), pf2 = vmulq_n_s32(V[2], (int32_t)c->mp.f[2][k]);
            uint32x4_t lo = vaddq_u32(vaddq_u32(vandq_u32(vreinterpretq_u32_s32(pf0), m16), vandq_u32(vreinterpretq_u32_s32(pf1), m16)),
                                      vaddq_u32(vandq_u32(vreinterpretq_u32_s32(pf2), m16), vdupq_n_u32(c->mp.f[3][k])));
            int32x4_t Ahi = vaddq_s32(vaddq_s32(vshrq_n_s32(pf0, 16), vshrq_n_s32(pf1, 16)), vaddq_s32(vshrq_n_s32(pf2, 16), vreinterpretq_s32_u32(vshrq_n_u32(lo, 16))));
            int32x4_t B2 = vmlaq_n_s32(vmlaq_n_s32(vdupq_n_s32(c->mp.i[3][k]), V[0], c->mp.i[0][k]), V[1], c->mp.i[1][k]);
            int32x4_t B = vmlaq_n_s32(B2, V[2], c->mp.i[2][k]);
            Pi[k] = vqmovn_s32(vaddq_s32(B, Ahi));
            Pf[k] = vmovn_u32(sel_sat16(vaddq_s32(B2, Ahi), vandq_u32(lo, m16)));
            X[k] = pack32v(Pi[k], Pf[k]);
        }
        /* the clip flags: against w, and against the guard-band w (lane 3 of the second compare is (w integer, z fraction)) */
        W = X[3];
        {
            uint32x4_t pf = vmull_n_u16(Pf[3], c->ratio);
            int32x4_t hi = vaddq_s32(vmulq_n_s32(vmovl_s16(Pi[3]), (int32_t)c->ratio), vreinterpretq_s32_u32(vshrq_n_u32(pf, 16)));
            W2 = vreinterpretq_s32_u32(vorrq_u32(vshlq_n_u32(vmovl_u16(vreinterpret_u16_s16(vqmovn_s32(hi))), 16), vandq_u32(pf, m16)));
        }
        vcc_a = vdupq_n_u32(0);
        vcc_b = vdupq_n_u32(0);
        for (k = 0; k < 4; k++) {
            uint32x4_t le, ge;
            clipmask4(X[k], W, &le, &ge);
            vcc_a = vorrq_u32(vcc_a, vorrq_u32(vandq_u32(le, vdupq_n_u32(1u << k)), vandq_u32(ge, vdupq_n_u32(0x100u << k))));
            clipmask4(k == 3 ? pack32v(Pi[3], Pf[2]) : X[k], W2, &le, &ge);
            vcc_b = vorrq_u32(vcc_b, vorrq_u32(vandq_u32(le, vdupq_n_u32(1u << k)), vandq_u32(ge, vdupq_n_u32(0x100u << k))));
        }
        {   /* pair-merged, as the record holds them (fx2_vtx_xform_scalar) */
            static const uint32_t oddl[4] = { 0, 0xffffffffu, 0, 0xffffffffu };
            uint32x4_t odd = vld1q_u32(oddl), m0f = vdupq_n_u32(0x0f0f);
            uint32x4_t a = vandq_u32(vcc_a, m0f), bb = vandq_u32(vcc_b, m0f), as = vrev64q_u32(a), bs = vrev64q_u32(bb);
            uint32x4_t ca = vbslq_u32(odd, vorrq_u32(as, vshlq_n_u32(a, 4)), vorrq_u32(a, vshlq_n_u32(as, 4)));
            uint32x4_t cb = vbslq_u32(odd, vorrq_u32(bs, vshlq_n_u32(bb, 4)), vorrq_u32(bb, vshlq_n_u32(bs, 4)));
            uint32x4_t fo = vorrq_u32(vshlq_n_u32(cb, 16), ca);
            uint32x4_t fe = vorrq_u32(vshlq_n_u32(vandq_u32(vshlq_n_u32(cb, 4), m16), 16), vandq_u32(vshlq_n_u32(ca, 4), m16));
            flags = vbslq_u32(odd, fo, fe);
        }
        /* stage 2: Q3 = P3 * pn and its reciprocal (scalar, the table) */
        mulk4v(Pi[3], Pf[3], c->pn, &Qi3, &Qf3);
        vst1q_s32((int32_t *)R0_a, pack32v(Qi3, Qf3));
        for (i = 0; i < 4; i++)
            R0_a[i] = hle_rcp32((int32_t)R0_a[i]);
        {
            uint32x4_t R0 = vld1q_u32(R0_a);
            R0i = vreinterpret_s16_u16(vshrn_n_u32(R0, 16));
            R0f = vmovn_u32(R0);
            R0i32 = vshrq_n_s32(vreinterpretq_s32_u32(R0), 16);
            R0f32 = vreinterpretq_s32_u32(vandq_u32(R0, m16));
        }
        /* stages 3 and 4: T = Q3 R0, E = 4 - 4 T, R1 = E R0 */
        chain4(Qi3, Qf3, R0i, R0f, R0i32, R0f32, &T_i, &T_f);
        {
            int32x4_t e1 = vsubq_s32(vdupq_n_s32(4 << 16), vshlq_n_s32(vreinterpretq_s32_u32(vmovl_u16(T_f)), 2));
            Ei = vqmovn_s32(vsubq_s32(vshrq_n_s32(e1, 16), vshlq_n_s32(vmovl_s16(T_i), 2)));
            Ef = vmovn_u32(vreinterpretq_u32_s32(e1));
        }
        chain4(Ei, Ef, R0i, R0f, R0i32, R0f32, &R1i, &R1f);
        rint = vbsl_s16(vcgez_s16(Pi[3]), R1i, vdup_n_s16(0x7fff));
        rint32 = vmovl_s16(rint);
        R1f32 = vreinterpretq_s32_u32(vmovl_u16(R1f));
        /* stage 5: the projection (P * R1, then * pn) and the viewport; the fourth lane takes z (the fog) */
        for (k = 0; k < 4; k++) {
            int16x4_t ni, nni;
            uint16x4_t nf, nnf;
            int32x4_t p, t1hi;
            chain4(Pi[k == 3 ? 2 : k], Pf[k == 3 ? 2 : k], rint, R1f, rint32, R1f32, &ni, &nf);
            mulk4v(ni, nf, c->pn, &nni, &nnf);
            p = vmulq_n_s32(vreinterpretq_s32_u32(vmovl_u16(nnf)), (int32_t)c->scale[k]);
            t1hi = vaddq_s32(vdupq_n_s32((int32_t)c->trans[k] + (k == 3 ? (int32_t)c->c31_3 : 0)), vshrq_n_s32(p, 16));
            Si[k] = vqmovn_s32(vaddq_s32(vmull_n_s16(nni, c->scale[k]), t1hi));
            if (k == 2)
                Szf = vmovn_u32(sel_sat16(t1hi, vandq_u32(vreinterpretq_u32_s32(p), m16)));
        }
        if (c->mode & FX2_LIGHTING) {                   /* the lit colours (and generated texture coordinates), scalar per vertex */
            uint8_t col[4][4];
            int16_t sg[4][2];
            for (i = 0; i < 4; i++) {
                int8_t nb8[3] = { (int8_t)in[b + i].c[0], (int8_t)in[b + i].c[1], (int8_t)in[b + i].c[2] };
                sg[i][0] = sg[i][1] = 0;
                lit_vertex(c, ((b + i) & 1) ? 0 : 1, nb8, in[b + i].c[3], col[i], (c->mode & FX2_TEXGEN) != 0, sg[i]);
            }
            {
                uint16x4x2_t z = vuzp_u16(vreinterpret_u16_u8(vld1_u8(col[0])), vreinterpret_u16_u8(vld1_u8(col[2])));
                C01 = z.val[0];
                C23 = z.val[1];
            }
            if (c->mode & FX2_TEXGEN) {
                int16x4x2_t z = vld2_s16(&sg[0][0]);
                S16 = z.val[0];
                T16 = z.val[1];
            }
        }
        if (c->mode & FX2_FOG)                          /* the alpha byte: the low byte of max(fog, 0x7f00) */
            C23 = vorr_u16(vand_u16(C23, vdup_n_u16(0xff)), vshl_n_u16(vreinterpret_u16_s16(vmax_s16(Si[3], vdup_n_s16(0x7f00))), 8));
        {   /* the records, as five 8-byte groups of four vertices each (vst4: vertex-major) */
            int16x4x4_t g0 = { { Pi[0], Pi[1], Pi[2], Pi[3] } };
            uint16x4x4_t g1 = { { Pf[0], Pf[1], Pf[2], Pf[3] } };
            int16x4x4_t g2 = { { vreinterpret_s16_u16(C01), vreinterpret_s16_u16(C23), vshrn_n_s32(vmulq_n_s32(vmovl_s16(S16), (int32_t)c->sc_s), 16), vshrn_n_s32(vmulq_n_s32(vmovl_s16(T16), (int32_t)c->sc_t), 16) } };
            int16x4x4_t g3 = { { Si[0], Si[1], NMUT(6, c->gen204, 0) ? Si[2] : vmax_s16(Si[2], vdup_n_s16(0)), vreinterpret_s16_u16(Szf) } };
            uint16x4x4_t g4 = { { vreinterpret_u16_s16(R1i), R1f, vmovn_u32(flags), vshrn_n_u32(flags, 16) } };
            vst4_s16((int16_t *)grp[0], g0);
            vst4_u16((uint16_t *)grp[1], g1);
            vst4_s16((int16_t *)grp[2], g2);
            vst4_s16((int16_t *)grp[3], g3);
            vst4_u16((uint16_t *)grp[4], g4);
            for (i = 0; i < m; i++) {
                uint8_t *o = (uint8_t *)&out[b + i];
                memcpy(o, grp[0][i], 8);
                memcpy(o + 8, grp[1][i], 8);
                memcpy(o + 16, grp[2][i], 8);
                memcpy(o + 24, grp[3][i], 8);
                memcpy(o + 32, grp[4][i], 8);
            }
        }
    }
}

/* ---- matrices ------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
void fx2_mtx_mul(const struct fx2_mtx *a, const struct fx2_mtx *b, struct fx2_mtx *out)
{
    struct fx2_mtx r;
    unsigned i, k;
    if (use_scalar()) {
        fx2_mtx_mul_scalar(a, b, out);
        return;
    }
    for (i = 0; i < 4; i++) {                           /* row i of the result: the four columns are the lanes */
        v64 acc = v64_zero(), last = v64_zero();
        for (k = 0; k < 4; k++) {
            int16x4_t bi = vld1_s16(b->i[k]);
            uint16x4_t bf = vld1_u16(b->f[k]);
            uint16_t af = a->f[i][k];
            int16_t ai = a->i[i][k];
            uint32x4_t p1 = vshrq_n_u32(vmull_n_u16(bf, af), 16);
            int32x4_t p2 = vmulq_n_s32(vmovl_s16(bi), (int32_t)af);
            int32x4_t p3 = vmulq_n_s32(vreinterpretq_s32_u32(vmovl_u16(bf)), (int32_t)ai);
            last = v64_shl16_s32(vmull_n_s16(bi, ai));
            acc = v64_addw_s32(v64_addw_s32(v64_addw_s32(acc, vreinterpretq_s32_u32(p1)), p2), p3);
            acc = v64_wrap48(v64_add(acc, last));
        }
        vst1_s16(r.i[i], sclamp4(acc));
        if (i < 2) {                                    /* rows 0 and 1: the fraction is read after the last term, rows 2 and 3 before its integer product */
            vst1_u16(r.f[i], uclamp4(acc));
        } else {
            v64 d = { vsubq_s64(acc.lo, last.lo), vsubq_s64(acc.hi, last.hi) };
            vst1_u16(r.f[i], uclamp4(v64_wrap48(d)));
        }
    }
    *out = r;
}


/* ---- the Rej vertex pipeline (rej_fx.h) --------------------------------------------------------------------------------------------------------------------------------------------- */
/* The stages of fx2_vtx_xform with the differences of rej_project (rej_fx.c): the clip bits only of x y z against the guard band (a vertex outside it is marked: 0x7fff * 0x7fff goes into the
 * viewport accumulator), no limit on the integer part of 1/w, the optional rounding of y (REJ_YQ), the record of 20 bytes without clip coordinates. The results are those of rej_vtx_xform_scalar. */
void rej_vtx_xform(const struct fx2_vctx *c, unsigned flags, const struct rej_flx_lit *flx, const struct fx2_vin *in, unsigned n, struct rej_vtx *out)
{
    unsigned i;
    int16x4_t Pi[REJ_NSLOTS];
    uint16x4_t Pf[REJ_NSLOTS];
    int16_t Qi3[REJ_NSLOTS], T_i[REJ_NSLOTS], R1i[REJ_NSLOTS];
    uint16_t Qf3[REJ_NSLOTS], T_f[REJ_NSLOTS], R1f[REJ_NSLOTS];
    uint32_t R0[REJ_NSLOTS];
    unsigned marked[REJ_NSLOTS];

    if (use_scalar()) {
        rej_vtx_xform_scalar(c, flags, flx, in, n, out);
        return;
    }
    for (i = 0; i < n; i++) {
        const struct fx2_vin *vi = &in[i];
        int16_t v[3] = { vi->x, vi->y, vi->z };
        int32x4_t X2;
        int16_t P3i;
        uint16_t P3f, P2f;
        hle_acc prod;
        int32_t W2;
        vtx_P4(&c->mp, v, &Pi[i], &Pf[i]);
        P3i = vget_lane_s16(Pi[i], 3);
        P3f = vget_lane_u16(Pf[i], 3);
        P2f = vget_lane_u16(Pf[i], 2);
        prod = (hle_acc)P3f * c->ratio + (((hle_acc)P3i * c->ratio) << 16);
        W2 = (int32_t)((uint32_t)(uint16_t)hle_sclamp(prod) << 16 | (uint16_t)prod);
        X2 = vsetq_lane_s32((int32_t)((uint32_t)(uint16_t)P3i << 16 | P2f), pack32(Pi[i], Pf[i]), 3);
        marked[i] = (clipbits4(X2, vdupq_n_s32(W2)) & NMUT(368, 0x0707, 0x0303)) != 0;
        {
            v64 Q = mulk4(Pi[i], Pf[i], c->pn);
            int64_t q3 = vgetq_lane_s64(Q.hi, 1);             /* lane 3 only */
            Qi3[i] = hle_sclamp(q3);
            Qf3[i] = hle_uclamp(q3);
            R0[i] = hle_rcp32((int32_t)((uint32_t)(uint16_t)Qi3[i] << 16 | Qf3[i]));
        }
    }
#pragma GCC unroll 4
    for (i = 0; i < n; i++) {                           /* T = Q3 * R0; E = 4 - 4 T */
        int16_t R0i = (int16_t)(R0[i] >> 16);
        uint16_t R0f = (uint16_t)R0[i];
        hle_acc tt = hle_wrap48(((hle_acc)Qf3[i] * R0f >> 16) + (hle_acc)Qi3[i] * R0f + (hle_acc)Qf3[i] * R0i);
        hle_acc fu = hle_wrap48(tt + (((hle_acc)Qi3[i] * R0i) << 16));
        T_i[i] = hle_sclamp(fu);
        T_f[i] = hle_uclamp(tt);
    }
#pragma GCC unroll 4
    for (i = 0; i < n; i++) {                           /* R1 = E * R0 */
        int16_t R0i = (int16_t)(R0[i] >> 16), Ei;
        uint16_t R0f = (uint16_t)R0[i], Ef;
        hle_acc e1 = ((hle_acc)4 << 16) - (hle_acc)4 * T_f[i];
        hle_acc e2 = e1 - (((hle_acc)4 * T_i[i]) << 16);
        hle_acc t2, f2;
        Ei = hle_sclamp(e2);
        Ef = hle_uclamp(e1);
        t2 = hle_wrap48(((hle_acc)Ef * R0f >> 16) + (hle_acc)Ei * R0f + (hle_acc)Ef * R0i);
        f2 = hle_wrap48(t2 + (((hle_acc)Ei * R0i) << 16));
        R1i[i] = hle_sclamp(f2);
        R1f[i] = hle_uclamp(t2);
    }
    {
        int16_t sc[4] = { c->scale[0], c->scale[1], c->scale[2], c->scale[3] };
        int32_t base32[4] = { c->trans[0], (flags & REJ_YQ) ? hle_clamp16((int32_t)c->trans[1] + NMUT(370, 2, 1)) : c->trans[1], c->trans[2], c->trans[3] + (int32_t)c->c31_3 };
        v64 b = v64_shl16_s32(vld1q_s32(base32)), bm = b;
        int16x4_t scale = vld1_s16(sc);
        {
            int64_t K = ((int64_t)NMUT(369, 0x7fff, 0x7ffe) * c->c31_7) << 16;       /* the marked vertices' term */
            bm.lo = vaddq_s64(b.lo, vdupq_n_s64(K));
            bm.hi = vaddq_s64(b.hi, vdupq_n_s64(K));
        }
        for (i = 0; i < n; i++) {                       /* stage 5 and the rest of the record */
            const struct fx2_vin *vi = &in[i];
            struct rej_vtx *o = &out[i];
            int16_t s = vi->s, tt = vi->t, fog;
            uint8_t col[4] = { vi->c[0], vi->c[1], vi->c[2], vi->c[3] };
            v64 t, fl, n2, t1, f1;
            int16x4_t ndci, nni, Si;
            uint16x4_t ndcf, nni_f, Sf;
            int32x4_t pf;
            if ((flags & REJ_LIT_LX) && (c->mode & FX2_LIGHTING)) {
                int8_t nb[3] = { (int8_t)vi->c[0], (int8_t)vi->c[1], (int8_t)vi->c[2] };
                uint8_t lo[4];
                int16_t sg[2] = { 0, 0 };
                rej_lx_lit_vertex(c, (i & 1) ? 0 : 1, nb, vi->c[3], lo, (c->mode & FX2_TEXGEN) != 0, sg);
                memcpy(col, lo, 4);
                if (c->mode & FX2_TEXGEN) {
                    s = sg[0];
                    tt = sg[1];
                }
            } else if (flx && (c->mode & FX2_LIGHTING)) {
                int8_t nb[3] = { (int8_t)vi->c[0], (int8_t)vi->c[1], (int8_t)vi->c[2] };
                rej_flx_lit_vertex(flx, (i & 1) ? 0 : 1, nb, col);
            } else if ((c->mode & FX2_LIGHTING)) {
                int8_t nb[3] = { (int8_t)vi->c[0], (int8_t)vi->c[1], (int8_t)vi->c[2] };
                uint8_t lo[4];
                int16_t sg[2] = { 0, 0 };
                lit_vertex(c, (i & 1) ? 0 : 1, nb, vi->c[3], lo, (c->mode & FX2_TEXGEN) != 0, sg);
                memcpy(col, lo, 4);
                if (c->mode & FX2_TEXGEN) {
                    s = sg[0];
                    tt = sg[1];
                }
            }
            mm4s(v64_zero(), Pi[i], Pf[i], R1i[i], R1f[i], &t, &fl);
            ndci = sclamp4(fl);
            ndcf = uclamp4(t);
            n2 = mulk4(ndci, ndcf, c->pn);
            {
                int16x4_t ni = sclamp4(n2);
                uint16x4_t nf = uclamp4(n2);
                nni = vcopy_lane_s16(ni, 3, ni, 2);
                nni_f = vcopy_lane_u16(nf, 3, nf, 2);
            }
            pf = vmulq_s32(vreinterpretq_s32_u32(vmovl_u16(nni_f)), vmovl_s16(scale));
            t1 = v64_wrap48(v64_addw_s32(marked[i] ? bm : b, pf));
            f1 = v64_wrap48(v64_add(t1, v64_shl16_s32(vmull_s16(nni, scale))));
            Si = sclamp4(f1);
            Sf = uclamp4(t1);
            o->sx = vget_lane_s16(Si, 0);
            o->sy = (flags & REJ_YQ) ? (int16_t)(vget_lane_s16(Si, 1) & NMUT(371, ~3, ~1)) : vget_lane_s16(Si, 1);
            o->szi = vget_lane_s16(Si, 2);
            o->szf = vget_lane_u16(Sf, 2);
            o->rwi = R1i[i];
            o->rwf = R1f[i];
            fog = vget_lane_s16(Si, 3);
            if (c->mode & FX2_FOG)
                col[3] = (uint8_t)(fog < 0x7f00 ? 0x7f00 : fog);
            memcpy(o->col, col, 4);
            o->s = (int16_t)(((int32_t)s * c->sc_s) >> 16);
            o->t = (int16_t)(((int32_t)tt * c->sc_t) >> 16);
        }
    }
}

int rej_tri_setup(const struct rej_tri_in *in, uint32_t *cmd, unsigned *len)
{
    return rej_tri_setup_core(in, cmd, len, !use_scalar());
}

#endif
