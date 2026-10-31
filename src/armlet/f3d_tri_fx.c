/*
 * arMLEt - f3d_tri_fx.c: the F3D triangle command as a pure function; docs/ucode-f3d.md 8
 *
 * Decision (flags, area, facing) and the RDP triangle bytes (header, shade, texture, z blocks) from three vertex records. The arithmetic is the microcode's chain forms in hle_fx.h's closed
 * form; the places where the low word is read before the last product ("lo3") use hle_mac_32x32_before_h. The microcode also computes junk lanes (stale vector lanes) that only reach dead
 * DMEM scratch; they are not computed here.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <string.h>

#include "f3d_fx.h"
#include "hle_fx.h"
#include "f3d_fx_int.h"

#ifndef F3D_FX_MUTATE
#define F3D_FX_MUTATE 0
#endif
#define FMUT(n, ok, bad) (F3D_FX_MUTATE == (n) ? (bad) : (ok))

typedef uint16_t u16;
typedef int16_t s16;

/* VSUB with the borrow of a preceding VSUBC: a - b - borrow, the borrow applied to b first, saturating */
static inline u16 vsub16(u16 a, u16 b, int borrow)
{
    s16 unsat = (s16)(b + borrow);
    s16 sat = hle_clamp16((int32_t)(s16)b + borrow);
    s16 d = hle_clamp16((int32_t)(s16)a - sat);
    if (sat > unsat)
        d = hle_clamp16((int32_t)d - 1);
    return (u16)d;
}

/* a 32-bit difference of two (hi : lo) values as VSUBC + VSUB: the low words wrap, the high words saturate; returned as the signed 32-bit value hi * 65536 + lo */
static inline int32_t sub32v(int32_t a, int32_t b)
{
    u16 al = (u16)a, bl = (u16)b;
    u16 dh = vsub16((u16)((uint32_t)a >> 16), (u16)((uint32_t)b >> 16), al < bl);
    return (int32_t)((uint32_t)dh << 16 | (u16)(al - bl));
}

static inline int32_t pair_of(hle_acc acc) { return (int32_t)((uint32_t)(u16)hle_sclamp(acc) << 16 | hle_uclamp(acc)); }
static inline int32_t mulc32(int32_t a, int32_t b)    /* mulc with lo3 */
{
    return (int32_t)((uint32_t)(u16)hle_sclamp(hle_mac_32x32(a, b)) << 16 | (F3D_FX_MUTATE == 4 ? hle_uclamp(hle_mac_32x32(a, b)) : hle_uclamp(hle_mac_32x32_before_h(a, b))));   /* mutation 4: the full low word */
}
static inline int32_t mul16_32(int16_t v, int32_t f)  /* v * f for a 16-bit v (VMUDM, VMADH) */
{
    return pair_of(hle_wrap48((hle_acc)v * (uint16_t)f + (((hle_acc)v * (int16_t)(f >> 16)) << 16)));
}
static inline u16 abs16s(u16 v) { return (s16)v < 0 ? (u16)hle_clamp16(-(int32_t)(s16)v) : v; }
static inline int ge32(int32_t a, int32_t b) { return a >= b; }

/* big-endian store */
static inline void put16(uint8_t *p, u16 v) { p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v; }


/* VCH + VCL of one lane against the 16-bit value vt (low word 0): the result vector of the pair. With vt = -c (c > 0) a value of magnitude below c is replaced by +-c (the sign of the value; 0 is
 * +c), anything else passes: the minimum magnitude of the z gradient (docs/ucode-f3d.md 8.5). */
static void vch_vcl_result(u16 *vhi, u16 *vlo, u16 vt)
{
    s16 a = (s16)*vhi, b = (s16)vt;
    int sign = (a ^ b) < 0;
    s16 negvt = sign ? (s16)(-b) : b;
    s16 diff = (s16)(a - negvt);
    int diff_zero = diff == 0, vt_neg = b < 0, diff_gt = diff > 0;
    int ge = sign ? vt_neg : (diff_gt || diff_zero);
    int le = sign ? !diff_gt : vt_neg;
    int vce = sign && diff == -1;
    int ne = !(diff_zero || vce);
    int mask = sign ? le : ge;
    u16 hi = (u16)(mask ? negvt : a);
    u16 x = *vlo, t = 0;
    u16 nvt = sign ? (u16)(0 - t) : t;
    u16 ldiff = (u16)(x - nvt);
    uint32_t sum = (uint32_t)x + t;
    int ncarry = ldiff == (sum > 0xffff ? 0xffff : sum);
    int le_eq = (ldiff == 0 && ncarry && !vce) || ((ldiff == 0 || ncarry) && vce);
    int ge_eq = !(t > x);
    if (!ne && sign) le = le_eq;
    if (!(sign || ne)) ge = ge_eq;
    int m2 = sign ? le : ge;
    *vlo = m2 ? nvt : x;
    *vhi = hi;
}

/* ---- the attribute gradients of one lane --------------------------------------------------------------------------------------------------------------------------------------- */
struct lane_out {
    int32_t init, gx, gy, ge;       /* 16.16 values (hi : lo) */
};

struct tri_geom {
    int16_t v4s[6];                 /* 4 * (dx21 dy21 dx10 dy10 dx20 dy20), saturated */
    int32_t R;                      /* the refined reciprocal of the area */
    int32_t slope20;                /* (c_20 : s1_20) */
    u16 y2f;                        /* the fractional y of the top vertex << 14 */
};

static void lane_grad(const struct tri_geom *G, const int32_t V[3], struct lane_out *o)
{
    int32_t d10 = sub32v(V[1], V[0]), d02 = sub32v(V[0], V[2]), d20 = sub32v(V[2], V[0]), d01 = sub32v(V[0], V[1]);
    hle_acc ax = hle_wrap48((hle_acc)d20 * G->v4s[3] + (hle_acc)d01 * G->v4s[5]);
    hle_acc ay = hle_wrap48((hle_acc)d10 * G->v4s[4] + (hle_acc)d02 * G->v4s[2]);
    int32_t nx = (int32_t)(uint32_t)(ax >> 16), ny = (int32_t)(uint32_t)(ay >> 16);
    int32_t gx = mulc32(nx, G->R), gy = mulc32(ny, G->R);
    /* along the major edge: gy + gx * slope(H-L); the low word is read before the high product is added */
    hle_acc acc = hle_wrap48((hle_acc)gy + hle_mac_32x32(gx, G->slope20));
    u16 gel = hle_uclamp(hle_wrap48(acc - (((hle_acc)(s16)(gx >> 16) * (s16)(G->slope20 >> 16)) << 16)));
    u16 geh = hle_sclamp(acc);
    int32_t ge = (int32_t)((uint32_t)geh << 16 | gel);
    int32_t t = pair_of(hle_mac_32xu16(ge, G->y2f));
    o->init = sub32v(V[0], t);
    o->gx = gx;
    o->gy = gy;
    o->ge = ge;
}


#ifdef __ARM_NEON
#include <arm_neon.h>
/* ---- NEON: four lanes of 32-bit (hi : lo) values per vector ------------------------------------------------------------------------------------------------------------------- */
static inline uint32x2_t n_uclamp64(int64x2_t t)       /* the low-word clamp of 2 accumulators: the low 16 bits (in u32 lanes) while they fit in 32 bits, else 0xffff / 0 */
{
    int64x2_t s = vshrq_n_s64(t, 31);
    uint64x2_t fits = vorrq_u64(vceqzq_s64(s), vceqq_s64(s, vdupq_n_s64(-1)));
    uint64x2_t sat = vreinterpretq_u64_s64(veorq_s64(vshrq_n_s64(t, 63), vdupq_n_s64(-1)));
    return vmovn_u64(vbslq_u64(fits, vreinterpretq_u64_s64(t), sat));
}
/* (sclamp(acc), uclamp(acc - sub)) of 4 lanes as 32-bit values hi << 16 | lo */
static inline int32x4_t n4_pack(int64x2_t acc0, int64x2_t acc1, int64x2_t sub0, int64x2_t sub1)
{
    int16x4_t hi = vqmovn_s32(vcombine_s32(vshrn_n_s64(acc0, 16), vshrn_n_s64(acc1, 16)));
    uint16x4_t lo = vmovn_u32(vcombine_u32(n_uclamp64(vsubq_s64(acc0, sub0)), n_uclamp64(vsubq_s64(acc1, sub1))));
    uint16x4x2_t z = vzip_u16(lo, vreinterpret_u16_s16(hi));
    return vreinterpretq_s32_u16(vcombine_u16(z.val[0], z.val[1]));
}
static inline int32x4_t n4_sub(int32x4_t a, int32x4_t b)           /* VSUBC + VSUB */
{
    int16x4_t ah = vshrn_n_s32(a, 16), bh = vshrn_n_s32(b, 16);
    uint16x4_t al = vmovn_u32(vreinterpretq_u32_s32(a)), bl = vmovn_u32(vreinterpretq_u32_s32(b));
    int16x4_t b1 = vreinterpret_s16_u16(vshr_n_u16(vclt_u16(al, bl), 15));
    int16x4_t unsat = vadd_s16(bh, b1), sat = vqadd_s16(bh, b1);
    int16x4_t d = vqsub_s16(ah, sat);
    if (F3D_FX_MUTATE != 13) d = vbsl_s16(vcgt_s16(sat, unsat), vqsub_s16(d, vdup_n_s16(1)), d);   /* mutation 13: no VSUB saturation quirk */
    uint16x4x2_t z = vzip_u16(vsub_u16(al, bl), vreinterpret_u16_s16(d));
    return vreinterpretq_s32_u16(vcombine_u16(z.val[0], z.val[1]));
}
static inline int32x4_t n4_mulc(int32x4_t x, int32_t R)            /* mulc32 */
{
    int64x2_t a0 = vshrq_n_s64(vmull_n_s32(vget_low_s32(x), R), 16), a1 = vshrq_n_s64(vmull_n_s32(vget_high_s32(x), R), 16);
    int32x4_t hr = vmull_n_s16(vshrn_n_s32(x, 16), (int16_t)(R >> 16));
    if (F3D_FX_MUTATE == 14) return n4_pack(a0, a1, vdupq_n_s64(0), vdupq_n_s64(0));   /* mutation 14: the full low word */
    return n4_pack(a0, a1, vshll_n_s32(vget_low_s32(hr), 16), vshll_n_s32(vget_high_s32(hr), 16));
}
static inline int32x4_t n4_dot(int32x4_t d1, int32_t c1, int32x4_t d2, int32_t c2)   /* (d1 c1 + d2 c2) >> 16 as a 32-bit value */
{
    int64x2_t a0 = vmlal_n_s32(vmull_n_s32(vget_low_s32(d1), c1), vget_low_s32(d2), c2);
    int64x2_t a1 = vmlal_n_s32(vmull_n_s32(vget_high_s32(d1), c1), vget_high_s32(d2), c2);
    return vcombine_s32(vshrn_n_s64(a0, 16), vshrn_n_s64(a1, 16));
}
static inline int32x4_t n4_ge(int32x4_t gy, int32x4_t gx, int32_t S)
{
    int64x2_t a0 = vaddq_s64(vmovl_s32(vget_low_s32(gy)), vshrq_n_s64(vmull_n_s32(vget_low_s32(gx), S), 16));
    int64x2_t a1 = vaddq_s64(vmovl_s32(vget_high_s32(gy)), vshrq_n_s64(vmull_n_s32(vget_high_s32(gx), S), 16));
    int32x4_t hr = vmull_n_s16(vshrn_n_s32(gx, 16), (int16_t)(S >> 16));
    if (F3D_FX_MUTATE == 15) hr = vdupq_n_s32(0);                    /* mutation 15: the low word read after the high product */
    return n4_pack(a0, a1, vshll_n_s32(vget_low_s32(hr), 16), vshll_n_s32(vget_high_s32(hr), 16));
}
static inline int32x4_t n4_mul_u16(int32x4_t x, uint16_t y)       /* pair_of(floor(x y / 65536)) */
{
    int64x2_t a0 = vshrq_n_s64(vmull_n_s32(vget_low_s32(x), (int32_t)y), 16), a1 = vshrq_n_s64(vmull_n_s32(vget_high_s32(x), (int32_t)y), 16);
    return n4_pack(a0, a1, vdupq_n_s64(0), vdupq_n_s64(0));
}
/* the gradients of lanes 4h .. 4h+3, as vectors of 32-bit (hi : lo) values */
struct grad4 { int32x4_t init, gx, gy, ge; };
static inline struct grad4 lane_grad4(const struct tri_geom *G, const int32_t V[3][8], int h)
{
    int32x4_t v0 = vld1q_s32(&V[0][4 * h]), v1 = vld1q_s32(&V[1][4 * h]), v2 = vld1q_s32(&V[2][4 * h]);
    int32x4_t d10 = n4_sub(v1, v0), d02 = n4_sub(v0, v2), d20 = n4_sub(v2, v0), d01 = n4_sub(v0, v1);
    struct grad4 g;
    g.gx = n4_mulc(n4_dot(d20, G->v4s[3], d01, G->v4s[5]), G->R);
    g.gy = n4_mulc(n4_dot(d10, G->v4s[4], d02, G->v4s[2]), G->R);
    g.ge = n4_ge(g.gy, g.gx, G->slope20);
    g.init = n4_sub(v0, n4_mul_u16(g.ge, G->y2f));
    return g;
}
/* the 64-byte block: init.hi gx.hi init.lo gx.lo ge.hi gy.hi ge.lo gy.lo, four big-endian halfwords per row */
static inline void n4_store_row(uint8_t *p, uint16x4_t v) { vst1_u8(p, vrev16_u8(vreinterpret_u8_u16(v))); }
static inline uint16x4_t n4_hi(int32x4_t v) { return vreinterpret_u16_s16(vshrn_n_s32(v, 16)); }
static inline uint16x4_t n4_lo(int32x4_t v) { return vmovn_u32(vreinterpretq_u32_s32(v)); }
static inline void n4_store_block(uint8_t *o, const struct grad4 *g)
{
    n4_store_row(o + 0, n4_hi(g->init)); n4_store_row(o + 8, n4_hi(g->gx));
    n4_store_row(o + 16, n4_lo(g->init)); n4_store_row(o + 24, n4_lo(g->gx));
    n4_store_row(o + 32, n4_hi(g->ge)); n4_store_row(o + 40, n4_hi(g->gy));
    n4_store_row(o + 48, n4_lo(g->ge)); n4_store_row(o + 56, n4_lo(g->gy));
}
#endif

/* ---- the command ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */
static int tri_core(const struct f3d_tri_in *in, int test_flags, int neon, uint8_t *cmd, unsigned *len)
{
    const struct f3d_vtx *A = in->v[0], *B = in->v[1], *C = in->v[2];
    if (test_flags) {
        int32_t fa = (int16_t)A->flags, fb = (int16_t)B->flags, fc = (int16_t)C->flags;
        if ((fa & fb & fc & in->reject_mask) != 0)
            return F3D_TRI_REJECT;
        if (((fa | fb | fc) & in->clip_mask) != 0)
            return F3D_TRI_CLIP;
    }
    /* area and sort */
    s16 dyBA = (s16)vsub16((u16)B->sy, (u16)A->sy, in->vco_carry != 0);
    s16 dxCA = (s16)vsub16((u16)C->sx, (u16)A->sx, 0), dyCA = (s16)vsub16((u16)C->sy, (u16)A->sy, 0);
    s16 dxAB = (s16)vsub16((u16)A->sx, (u16)B->sx, 0);
    int32_t m1 = (int32_t)dxCA * dyBA, m2 = (int32_t)dxAB * dyCA;
    u16 alo = (u16)((u16)m1 + (u16)m2);
    u16 ahi = (u16)hle_clamp16((int32_t)(s16)(m1 >> 16) + (s16)(m2 >> 16) + ((uint32_t)(u16)m1 + (u16)m2 > 0xffff));
    const struct f3d_vtx *p[3] = { A, B, C };
    int par = 0;
    if (in->sort_given && F3D_FX_MUTATE != 35) {                  /* mutation 35: Indy sorts by Y as F3D does */
        p[0] = in->v[in->ord[0]]; p[1] = in->v[in->ord[1]]; p[2] = in->v[in->ord[2]];
        par = F3D_FX_MUTATE == 36 ? 0 : in->par != 0;                 /* mutation 36: the table's parity ignored */
    } else
    for (;;) {
        if (FMUT(7, p[1]->sy < p[0]->sy, p[1]->sy <= p[0]->sy)) { const struct f3d_vtx *q = p[0]; p[0] = p[1]; p[1] = q; par ^= 1; }
        if (p[2]->sy < p[1]->sy) { const struct f3d_vtx *q = p[1]; p[1] = p[2]; p[2] = q; par ^= 1; continue; }
        break;
    }
    {
        int neg = (s16)ahi < 0;
        if (((ahi | alo) & 0xffff) == 0)
            return in->variant == F3D_TRIV_INDY && F3D_FX_MUTATE != 33 ? F3D_TRI_REJECT : F3D_TRI_CULL;       /* mutation 33: culled */       /* (Indy: a different exit of the microcode) */
        uint32_t cf = in->geom & F3D_G_CULL_FRONT, cb = in->geom & F3D_G_CULL_BACK;
        if (in->variant == F3D_TRIV_INDY && F3D_FX_MUTATE != 32)       /* mutation 32: the F3D culling. Indy: bit 13 culls the faces of negative area, bit 12 does nothing */
            cf = 0;
        if (in->variant == F3D_TRIV_RARE && F3D_FX_MUTATE != 27) {     /* Rare: bit 13 turns culling on, bit 12 selects the front faces; bit 12 alone does nothing; mutation 27: the F3D rule */
            if (cf) {
                cf = cb >> 1;
                cb = 0;
            }
        }
        if ((!neg && cf && !(in->variant == F3D_TRIV_WR && F3D_FX_MUTATE != 24)) || (neg && cb))   /* mutation 24: Wave Race culls front faces */
            return F3D_TRI_CULL;
    }
    if (par && F3D_FX_MUTATE != 2) {                 /* mutation 2: the area is not negated after an odd number of swaps */
        hle_acc a = hle_wrap48(-(hle_acc)(int32_t)((uint32_t)ahi << 16 | alo));
        ahi = hle_sclamp(a);
        alo = hle_uclamp(a);
    }
    struct hle_pair Rp = f3d_recip_refine((int32_t)((uint32_t)ahi << 16 | alo));
    const int flip = (s16)ahi < 0 && F3D_FX_MUTATE != 3;    /* mutation 3: no flip bit */
    const uint32_t a1 = (uint32_t)(in->geom & 0xff) | 0xc8;
    const int has_shade = in->variant == F3D_TRIV_WR && F3D_FX_MUTATE != 25 ? (a1 & 7) != 0 : (a1 & 4) != 0, has_tex = (a1 & 2) != 0, has_z = (a1 & 1) != 0;
    const u16 a2 = (u16)(((flip ? 128u : 0u) | in->tile_byte) & 0xff);
    const struct f3d_vtx *H = p[0], *M = p[1], *L = p[2];

    /* edges */
    s16 dx[3], dy[3];                                /* 21 (L-M), 10 (M-H), 20 (L-H) */
    dx[0] = (s16)vsub16((u16)L->sx, (u16)M->sx, 0); dy[0] = (s16)vsub16((u16)L->sy, (u16)M->sy, 0);
    dx[1] = (s16)vsub16((u16)M->sx, (u16)H->sx, 0); dy[1] = (s16)vsub16((u16)M->sy, (u16)H->sy, 0);
    dx[2] = (s16)vsub16((u16)L->sx, (u16)H->sx, 0); dy[2] = (s16)vsub16((u16)L->sy, (u16)H->sy, 0);
    struct tri_geom G;
    G.v4s[0] = hle_clamp16(4 * (int32_t)dx[0]); G.v4s[1] = hle_clamp16(4 * (int32_t)dy[0]);
    G.v4s[2] = hle_clamp16(4 * (int32_t)dx[1]); G.v4s[3] = hle_clamp16(4 * (int32_t)dy[1]);
    G.v4s[4] = hle_clamp16(4 * (int32_t)dx[2]); G.v4s[5] = hle_clamp16(4 * (int32_t)dy[2]);
    G.R = hle_pair32(Rp);
    G.y2f = (u16)((uint32_t)(u16)H->sy << 14);
    int32_t slope[3];                                /* (c : s1) per edge, and the masked low words for the start-x correction */
    u16 s1m[3];
    for (int e = 0; e < 3; e++) {
        int32_t inv = (int32_t)hle_rcp32((int32_t)dy[e]);
        int32_t d = pair_of(hle_wrap48(((hle_acc)inv * 0x20) >> 16));                  /* L(lo, 0x20) + M(hi, 0x20) */
        int32_t q = (int32_t)((uint32_t)(u16)(dx[e] >> 2) << 16 | (u16)(dx[e] << 14)); /* dx / 4 as 16.16 */
        hle_acc u_acc = hle_mac_32x32(d, q);
        int32_t u = (int32_t)((uint32_t)(u16)hle_sclamp(u_acc) << 16 | hle_uclamp(u_acc));   /* full: not lo3 */
        int32_t s = pair_of(hle_wrap48(((hle_acc)u * 0x4000) >> 16));                  /* u / 4 */
        s16 c = (s16)(s >> 16);
        c = c > 0x1cc ? 0x1cc : c < -0x1cd ? -0x1cd : c;
        slope[e] = (int32_t)((uint32_t)(u16)c << 16 | (u16)s);
        s1m[e] = (u16)s & FMUT(5, 0xfff8, 0xffff);                                      /* mutation 5: the low word not masked */
    }
    G.slope20 = slope[2];
    /* header */
    uint8_t *o = cmd;
    o[0] = (uint8_t)a1;
    o[1] = (uint8_t)a2;
    put16(o + 2, (u16)L->sy);
    put16(o + 4, (u16)M->sy);
    put16(o + 6, (u16)H->sy);
    {
        /* start x: x / 4 as 16.16, the edge's slope times the fractional y subtracted */
        int32_t xl = (int32_t)((uint32_t)(u16)(M->sx >> 2) << 16 | (u16)(M->sx << 14));
        put16(o + 8, (u16)(xl >> 16)); put16(o + 10, (u16)xl);
        put16(o + 12, (u16)(slope[0] >> 16)); put16(o + 14, (u16)slope[0]);
        int32_t xh = (int32_t)((uint32_t)(u16)(H->sx >> 2) << 16 | (u16)(H->sx << 14));
        for (int w = 0; w < 2; w++) {                       /* w = 0: XH with the H-L edge, w = 1: XM with the H-M edge */
            int e = w ? 1 : 2;
            int32_t sm = (int32_t)((uint32_t)(u16)(slope[e] >> 16) << 16 | s1m[e]);
            int32_t t = pair_of(hle_mac_32xu16(sm, G.y2f));
            int32_t x = sub32v(xh, t);
            put16(o + (w ? 24 : 16), (u16)(x >> 16)); put16(o + (w ? 26 : 18), (u16)x);
            put16(o + (w ? 28 : 20), (u16)(slope[e] >> 16)); put16(o + (w ? 30 : 22), (u16)slope[e]);
        }
    }
    o += 32;

    /* attribute vectors per sorted vertex: lanes 0..3 colour (byte : 0x8000), 4..6 s' t' w', 7 z */
    int32_t V[3][8];
    for (int k = 0; k < 3; k++) {
        const struct f3d_vtx *P = p[k];
        const struct f3d_vtx *Pc = ((in->geom & F3D_G_SMOOTH) || (in->variant == F3D_TRIV_INDY && F3D_FX_MUTATE != 34) || (in->variant == F3D_TRIV_WR && F3D_FX_MUTATE != 26)) ? P : in->flat;   /* mutations 25, 26: the SM64 shade rule, flat shading in Wave Race */
        for (int i = 0; i < 4; i++)
            V[k][i] = (int32_t)((uint32_t)Pc->col[i] << 16 | 0x8000);
        V[k][7] = (int32_t)((uint32_t)(u16)P->sz << 16 | P->szf);
    }
    int32_t mcmp[3] = { 0, 0, 0 };                      /* the magnitude measure of s, t (hi : lo) and the w term */
    if (has_tex) {
        /* the nearest scaled w of the three vertices */
        int32_t mw = 0;
        for (int k = 0; k < 3; k++) {
            const struct f3d_vtx *P = k == 0 ? A : k == 1 ? B : C;
            int32_t w32 = (int32_t)((uint32_t)(u16)P->cx[3] << 16 | P->cf[3]);
            int32_t w = pair_of(hle_mac_32xu16(w32, in->persp));
            if (k == 0 || (F3D_FX_MUTATE == 9 ? ge32(w, mw) : !ge32(w, mw))) mw = w;   /* mutation 9: the farthest vertex's w */
        }
        int32_t m2v = mw >> 1;
        mcmp[2] = m2v;
        int32_t sv[3][3];
        for (int k = 0; k < 3; k++) {
            const struct f3d_vtx *P = p[k];
            int32_t f = mulc32((int32_t)((uint32_t)(u16)P->rwi << 16 | P->rwf), m2v);
            sv[k][0] = mul16_32(P->s, f);
            sv[k][1] = mul16_32(P->t, f);
            sv[k][2] = mul16_32(0x7fff, f);
            for (int i = 0; i < 3; i++) V[k][4 + i] = sv[k][i];
        }
        /* the magnitude measure: of |s'| (and |t'|) of the three vertices the largest high word (vertex order H, L, M; the earlier one wins ties), with its low word */
        for (int l = 0; l < 2 && (in->variant != F3D_TRIV_RARE || F3D_FX_MUTATE == 28); l++) {
            u16 a = abs16s((u16)(V[0][4 + l] >> 16)), b = abs16s((u16)(V[2][4 + l] >> 16));
            u16 x17 = (s16)a >= (s16)b ? a : b;
            u16 x18 = (s16)a >= (s16)b ? (u16)V[0][4 + l] : (u16)V[2][4 + l];
            u16 cc = abs16s((u16)(V[1][4 + l] >> 16));
            if ((s16)x17 >= (s16)cc) mcmp[l] = (int32_t)((uint32_t)x17 << 16 | x18);
            else mcmp[l] = (int32_t)((uint32_t)cc << 16 | (u16)V[1][4 + l]);
        }
    }
    struct lane_out lo[8];
    memset(lo, 0, sizeof lo);
    int need[8] = { has_shade, has_shade, has_shade, has_shade, has_tex, has_tex, has_tex, has_tex || has_z };
#ifdef __ARM_NEON
    struct grad4 g4[2];
    if (neon) {
        if (need[0]) g4[0] = lane_grad4(&G, V, 0);
        if (need[4] || (need[7] && F3D_FX_MUTATE != 12)) g4[1] = lane_grad4(&G, V, 1);   /* mutation 12: the z lane skipped when only z is on (the bug the real-data check found) */
        for (int h = 0; h < 2; h++) {
            if (!(h ? (need[4] || need[7]) : need[0])) continue;
            int32_t a[4], b[4], c[4], d[4];
            vst1q_s32(a, g4[h].init); vst1q_s32(b, g4[h].gx); vst1q_s32(c, g4[h].gy); vst1q_s32(d, g4[h].ge);
            for (int i = 0; i < 4; i++) { lo[4 * h + i].init = a[i]; lo[4 * h + i].gx = b[i]; lo[4 * h + i].gy = c[i]; lo[4 * h + i].ge = d[i]; }
        }
    } else
#endif
    for (int l = 0; l < 8; l++) {
        if (!need[l]) continue;
        int32_t Vl[3] = { V[0][l], V[1][l], V[2][l] };
        lane_grad(&G, Vl, &lo[l]);
    }
    if (has_shade) {
#ifdef __ARM_NEON
        if (neon) n4_store_block(o, &g4[0]); else
#endif
        for (int i = 0; i < 4; i++) {
            put16(o + 0 + 2 * i, (u16)(lo[i].init >> 16)); put16(o + 8 + 2 * i, (u16)(lo[i].gx >> 16));
            put16(o + 16 + 2 * i, (u16)lo[i].init);       put16(o + 24 + 2 * i, (u16)lo[i].gx);
            put16(o + 32 + 2 * i, (u16)(lo[i].ge >> 16)); put16(o + 40 + 2 * i, (u16)(lo[i].gy >> 16));
            put16(o + 48 + 2 * i, (u16)lo[i].ge);         put16(o + 56 + 2 * i, (u16)lo[i].gy);
        }
        o += 64;
    }
    if (has_tex && in->variant == F3D_TRIV_RARE && F3D_FX_MUTATE != 28) {      /* mutation 28: the texture block scaled by K as F3D's is */
        /* Rare: the gradients of lanes 4 .. 7 as they are (no scale K) */
#ifdef __ARM_NEON
        if (neon) n4_store_block(o, &g4[1]); else
#endif
        for (int j = 0; j < 4; j++) {
            int i = 4 + j;
            put16(o + 0 + 2 * j, (u16)(lo[i].init >> 16)); put16(o + 8 + 2 * j, (u16)(lo[i].gx >> 16));
            put16(o + 16 + 2 * j, (u16)lo[i].init);       put16(o + 24 + 2 * j, (u16)lo[i].gx);
            put16(o + 32 + 2 * j, (u16)(lo[i].ge >> 16)); put16(o + 40 + 2 * j, (u16)(lo[i].gy >> 16));
            put16(o + 48 + 2 * j, (u16)lo[i].ge);         put16(o + 56 + 2 * j, (u16)lo[i].gy);
        }
        o += 64;
    } else if (has_tex) {
        /* the scale from the largest gradient measure of s, t, w */
        int32_t X[3];
        for (int j = 0; j < 3; j++) {
            int i = 4 + j;
            int32_t A_ = pair_of(hle_wrap48((hle_acc)(s16)abs16s((u16)(lo[i].gx >> 16)) * 2048));
            int32_t B_ = pair_of(hle_wrap48((hle_acc)(s16)abs16s((u16)(lo[i].gy >> 16)) * 2048));
            int32_t C_ = pair_of(hle_mac_32xu16(mcmp[j], 2048));
            /* X = 2 A + B + C, the low word read before the high word of C is added */
            hle_acc acc = hle_wrap48(2 * (hle_acc)A_ + B_ + (hle_acc)(u16)C_);
            u16 xl = hle_uclamp(acc);
            acc = hle_wrap48(acc + (((hle_acc)(s16)(C_ >> 16)) << 16));
            X[j] = (int32_t)((uint32_t)(u16)hle_sclamp(acc) << 16 | xl);
        }
        int32_t Mx = X[0];
        if (!ge32(Mx, X[1])) Mx = X[1];
        int32_t X2 = X[2];
        if (!ge32(X2, X[1])) X2 = X[1];
        if (!ge32(Mx, X2)) Mx = X2;
        int32_t S = pair_of(hle_wrap48(((hle_acc)Mx * 64) >> 16));
        int32_t F = (int32_t)hle_rcp32(S);
        u16 Kl = (u16)(hle_acc)((hle_acc)(u16)F * 2);
        u16 Kh = hle_sclamp(hle_wrap48((hle_acc)(u16)F * 2 + (((hle_acc)(s16)(F >> 16) * 2) << 16)));
        if (!((s16)Kh < 1)) { Kh = 1; Kl = 0; }
        int32_t K = (int32_t)((uint32_t)Kh << 16 | Kl);
#ifdef __ARM_NEON
        if (neon) {
            struct grad4 k4 = { n4_mulc(g4[1].init, K), n4_mulc(g4[1].gx, K), n4_mulc(g4[1].gy, K), n4_mulc(g4[1].ge, K) };
            n4_store_block(o, &k4);
        } else
#endif
        for (int j = 0; j < 4; j++) {
            int i = 4 + j;
            int32_t vi = mulc32(lo[i].init, K), vx = mulc32(lo[i].gx, K), vy = mulc32(lo[i].gy, K), ve = mulc32(lo[i].ge, K);
            put16(o + 0 + 2 * j, (u16)(vi >> 16)); put16(o + 8 + 2 * j, (u16)(vx >> 16));
            put16(o + 16 + 2 * j, (u16)vi);        put16(o + 24 + 2 * j, (u16)vx);
            put16(o + 32 + 2 * j, (u16)(ve >> 16)); put16(o + 40 + 2 * j, (u16)(vy >> 16));
            put16(o + 48 + 2 * j, (u16)ve);         put16(o + 56 + 2 * j, (u16)vy);
        }
        o += 64;
    }
    if (has_z) {
        const struct lane_out *z = &lo[7];
        #define SHL5(v) pair_of(hle_wrap48((hle_acc)(u16)(v) * FMUT(6, 0x20, 0x10) + (((hle_acc)(s16)((v) >> 16) * FMUT(6, 0x20, 0x10)) << 16)))   /* mutation 6: shifted by 4 */
        int32_t zv = SHL5(V[0][7]), zx = SHL5(z->gx), zy = SHL5(z->gy), ze = SHL5(z->ge);
        #undef SHL5
        if (in->zclamp && F3D_FX_MUTATE != 8) {                                       /* mutation 8: no z gradient clamp */
            u16 yh = (u16)(zy >> 16), yl = (u16)zy;
            vch_vcl_result(&yh, &yl, (u16)(0 - (unsigned)in->zclamp));
            zy = (int32_t)((uint32_t)yh << 16 | yl);
        }
        int32_t t = pair_of(hle_mac_32xu16(ze, G.y2f));
        int32_t zinit = sub32v(zv, t);
        put16(o + 0, (u16)(zinit >> 16)); put16(o + 2, (u16)zinit);
        put16(o + 4, (u16)(zx >> 16));    put16(o + 6, (u16)zx);
        put16(o + 8, (u16)(ze >> 16));    put16(o + 10, (u16)ze);
        put16(o + 12, (u16)(zy >> 16));   put16(o + 14, (u16)zy);
        o += 16;
    }
    *len = (unsigned)(o - cmd);
    return F3D_TRI_OK;
}

int f3d_tri_setup_scalar(const struct f3d_tri_in *in, uint8_t *cmd, unsigned *len) { return tri_core(in, 1, 0, cmd, len); }
int f3d_tri_setup_noclip_scalar(const struct f3d_tri_in *in, uint8_t *cmd, unsigned *len) { return tri_core(in, 0, 0, cmd, len); }
int f3d_tri_setup(const struct f3d_tri_in *in, uint8_t *cmd, unsigned *len) { return tri_core(in, 1, 1, cmd, len); }
int f3d_tri_setup_noclip(const struct f3d_tri_in *in, uint8_t *cmd, unsigned *len) { return tri_core(in, 0, 1, cmd, len); }
