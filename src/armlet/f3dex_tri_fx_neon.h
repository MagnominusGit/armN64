/*
 * arMLEt - f3dex_tri_fx_neon.h: the NEON form of f3dex_tri_setup / f3dex_tri_setup_noclip (aarch64, generic armv8-a); #included at the end of f3dex_tri_fx.c, so it may use that file's helpers
 *
 * Bit-exact with tri_setup (the scalar reference, checked against the real microcode); tools/f3dex/fxn_tri.c compares the two. Layout of the work:
 *  - Early part, scalar: flag tests, area, culls (the culls need only the sign of the area), the vertex sort (a 7-entry table).
 *  - Edge part, one vector over the three edges (lanes L-M, M-H, L-H): saturating coordinate differences (vqsub), the three rcp32 lookups (clz / shift in vector code, three table loads),
 *    the slope products in 64-bit lanes, the limit, the start-x correction.
 *  - Attribute part, four lanes per vector (shade: r g b a; texture: s t w z): the differences are int32 lanes (the saturating integer half and the wrapping fraction of VSUBC / VSUB are
 *    vqsub / sub merged by a mask), the 48-bit chains are int64 lanes (smull / smlal), the accumulator read-outs are saturating narrows. Big-endian output by vuzp + vrev16 / vrev32.
 *
 * Why the closed forms are exact (|x| bounds in the comments of the helpers):
 *  - the accumulator never needs hle_wrap48: the largest sums are below 2^47 except the nx / ny dot product (2^47 exactly at most), and only its bits 47..16 are used, which wrapping does not change;
 *  - hle_pair_acc(a) = (sclamp(a), uclamp(a)) is the saturated 32-bit value of a (the integer half of the saturated value is the clamped integer half, the fraction is 0xffff / 0 on saturation),
 *    which is what vqmovn_s64 / vqshrn_n_s64 produce; mul_c, mul_s, mul_x and mul_fx are single saturating narrows;
 *  - mul_mid = (hi half of sat32(full), low half of sat32(full - ai*bi*2^32 >> 16));
 *  - the area cross = e1*e2 + e3*e4 with e in [-32768, 32767] fits an int32 and is never INT32_MIN (a sum of +-2^31 would need all four differences to be -32768, which needs x0 >= 0 and x0 < 0), so the
 *    pair negation of the odd-parity case is a plain negation.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEX_TRI_FX_NEON_H
#define ARMLET_F3DEX_TRI_FX_NEON_H

#include <arm_neon.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "f3dex_tri_fx_neon.h assumes a little-endian host"
#endif

#define TN_I static inline __attribute__((always_inline))
#define TN_HI vdupq_n_u32(0xffff0000u)

TN_I uint32_t tn_ld32(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* sort table (the one at DMEM 0x360): per idx H | M << 2 | L << 4 | parity << 6; idx 7 cannot occur */
static const uint8_t tn_perm[8] = { 0x24, 0x24, 0x09, 0x61, 0x12, 0x58, 0x46, 0x24 };

/* ---- 4-lane primitives on pair32 lanes (int part in the high half, fraction in the low half) ------------------------------------------------------------------------------------- */
/* sub_p: VSUBC + VSUB: the integer half saturates (and takes the borrow), the fraction wraps */
TN_I int32x4_t tn_sub(int32x4_t a, int32x4_t b)
{
    int32x4_t s = vqsubq_s32(a, b);
    return FMUT(128, vbslq_s32(TN_HI, s, vsubq_s32(a, b)), s);
}
/* mul_mid(a, b), b a scalar pair: |a b| <= 2^62, so full = (a b) >> 16 is below 2^46 and wrap48 is the identity */
TN_I int32x4_t tn_mm(int32x4_t a, int32_t b)
{
    int64x2_t p0 = vmull_n_s32(vget_low_s32(a), b), p1 = vmull_high_n_s32(a, b);
    int32x4_t h = vmull_n_s16(vshrn_n_s32(a, 16), (int16_t)(b >> 16));                       /* ai * bi */
    int32x4_t hi = vcombine_s32(vqshrn_n_s64(p0, 16), vqshrn_n_s64(p1, 16));
    int32x4_t lo = vcombine_s32(vqshrn_n_s64(vsubq_s64(p0, vshll_n_s32(vget_low_s32(h), 32)), 16),
                                vqshrn_n_s64(vsubq_s64(p1, vshll_high_n_s32(h, 32)), 16));
    return vbslq_s32(TN_HI, hi, FMUT(124, lo, hi));
}
/* (a b) >> 16 as a plain int32 (a 32-bit pair times an unsigned 16-bit c: |a c| < 2^47, the result fits) */
TN_I int32x4_t tn_mulc(int32x4_t a, int32_t c)
{
    return vcombine_s32(vshrn_n_s64(vmull_n_s32(vget_low_s32(a), c), 16), vshrn_n_s64(vmull_high_n_s32(a, c), 16));
}
/* dX * slope + dY: DaDe. tfull = dY + floor(dX sl / 65536) (the five terms of the scalar sum are the expansion of that product), ta = tfull without the integer x integer term */
TN_I int32x4_t tn_de(int32x4_t dY, int32x4_t dX, int32_t sl)
{
    int64x2_t p0 = vmull_n_s32(vget_low_s32(dX), sl), p1 = vmull_high_n_s32(dX, sl);
    int64x2_t q0 = vsraq_n_s64(vmovl_s32(vget_low_s32(dY)), p0, 16), q1 = vsraq_n_s64(vmovl_high_s32(dY), p1, 16);       /* < 2^41: no wrap48 */
    int32x4_t h = vmull_n_s16(vshrn_n_s32(dX, 16), (int16_t)(sl >> 16));
    int32x4_t hi = vcombine_s32(vqmovn_s64(q0), vqmovn_s64(q1));
    int32x4_t lo = vcombine_s32(vqmovn_s64(vsubq_s64(q0, vshll_n_s32(vget_low_s32(h), 16))), vqmovn_s64(vsubq_s64(q1, vshll_high_n_s32(h, 16))));
    return vbslq_s32(TN_HI, hi, FMUT(132, lo, hi));
}
/* the raw bits 47..16 of d1 * K[p] + d2 * K[q] (K lanes sign-extended s16): no clamp, no wrap48 needed (bits 47..0 of a sum below 2^48 in magnitude are those of its wrap) */
#define TN_DOT2(d1, d2, K, p, q) __extension__ ({ \
    int64x2_t a0_ = vmlal_laneq_s32(vmull_laneq_s32(vget_low_s32(d1), K, p), vget_low_s32(d2), K, q); \
    int64x2_t a1_ = vmlal_high_laneq_s32(vmull_high_laneq_s32(d1, K, p), d2, K, q); \
    vcombine_s32(vshrn_n_s64(a0_, 16), vshrn_n_s64(a1_, 16)); })

struct tn_grad { int32x4_t st, dX, dY, dE; };
/* from nx, ny and the start value: the four coefficient vectors of 4 lanes */
TN_I struct tn_grad tn_grad4(int32x4_t nx, int32x4_t ny, int32x4_t aH, int32_t R, int32_t sl, uint16_t csub, int dlx)
{
    struct tn_grad g;
    g.dX = tn_mm(nx, R);
    g.dY = tn_mm(ny, R);
    g.dE = tn_de(g.dY, g.dX, sl);
    g.st = dlx ? aH : tn_sub(aH, tn_mulc(g.dE, FMUT(139, csub, csub ^ 1)));
    return g;
}
/* the 64-byte block: start.i dX.i | start.f dX.f | dE.i dY.i | dE.f dY.f, big-endian halfwords */
TN_I void tn_store_block(uint8_t *o, int32x4_t st, int32x4_t dX, int32x4_t dE, int32x4_t dY)
{
    uint16x8x2_t a = vuzpq_u16(vreinterpretq_u16_s32(st), vreinterpretq_u16_s32(dX));
    uint16x8x2_t b = vuzpq_u16(vreinterpretq_u16_s32(dE), vreinterpretq_u16_s32(dY));
    uint8x16x4_t r = { { vrev16q_u8(vreinterpretq_u8_u16(a.val[1])), vrev16q_u8(vreinterpretq_u8_u16(a.val[0])),
                         vrev16q_u8(vreinterpretq_u8_u16(b.val[1])), vrev16q_u8(vreinterpretq_u8_u16(b.val[0])) } };
    vst1q_u8_x4(o, r);
}
/* (abs16 of the integer half, the fraction as is) */
TN_I int32x4_t tn_absc(int32x4_t a)
{
    return vbslq_s32(TN_HI, vqabsq_s32(vandq_s32(a, vreinterpretq_s32_u32(TN_HI))), a);
}
/* the texture-lane vector of a vertex: s', t', w' = x * K (x = s, t, 0x7fff; K the vertex's 1/w times the scale), lane 3 the screen z. x K is below 2^46: sat32 of it is mul_x */
#define TN_ATT(v, K, Zv, k) __extension__ ({ \
    int32x2_t st_ = vget_low_s32(vmovl_s16(vreinterpret_s16_u32(vdup_n_u32(tn_ld32(&(v)->s))))); \
    int32x4_t r_ = vcombine_s32(vqmovn_s64(vmull_laneq_s32(st_, K, k)), vqmovn_s64(vmull_laneq_s32(vcreate_s32(0x7fffull), K, k))); \
    vcopyq_laneq_s32(r_, 3, Zv, k); })
#define TN_ATT0(Zv, k) vcopyq_laneq_s32(vdupq_n_s32(0), 3, Zv, k)       /* no texture: only the z lane matters */

/* ---- the triangle ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
static int tn_draw(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len, const unsigned pe, const int32_t area);

TN_I int tn_tri(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len, const int clip_tests)
{
    const struct f3dex_build *B = in->b;
    const struct f3dex_vtx *const V0 = in->v[0], *const V1 = in->v[1], *const V2 = in->v[2];
    const uint32_t g = in->geom;
    const int dlx = B->dlx;

    /* early rejection (5.2) */
    if (clip_tests && (!dlx || (g & F3DEX_G_CLIPPING))) {
        unsigned f0 = V0->flags, f1 = V1->flags, f2 = V2->flags;
        if (f0 & f1 & f2 & B->reject_mask)
            return F3DEX_TRI_REJECT;
        if ((f0 | f1 | f2) & 0x4343)
            return F3DEX_TRI_CLIP;
    }
    /* area: saturating differences (the unsaturated ones fit s16 unless a coordinate pair is more than 32767 apart: then the slow path clamps) */
    const int32_t x0 = V0->sx, y0 = V0->sy, x1 = V1->sx, y1 = V1->sy, x2 = V2->sx, y2 = V2->sy;
    int32_t e1 = x2 - x0, e2 = y1 - y0, e3 = x0 - x1, e4 = y2 - y0;
    if (FMUT(130, __builtin_expect(((uint32_t)(e1 + 0x8000) | (uint32_t)(e2 + 0x8000) | (uint32_t)(e3 + 0x8000) | (uint32_t)(e4 + 0x8000)) > 0xffffu, 0), 0)) {
        e1 = hle_clamp16(e1); e2 = hle_clamp16(e2); e3 = hle_clamp16(e3); e4 = hle_clamp16(e4);
    }
    int32_t cross = e1 * e2 + e3 * e4;                      /* exact in int32 (see the header comment); its sign is the sign of r17 */
    if (cross <= 0) {
        if (cross == 0)
            return F3DEX_TRI_DEGEN;
        if (g & F3DEX_G_CULL_BACK)
            return F3DEX_TRI_CULL;
    } else if (g & F3DEX_G_CULL_FRONT)
        return F3DEX_TRI_CULL;

    /* vertex order (5.3) */
    const unsigned pe = tn_perm[(y2 < y1) * 4 + (y1 < y0) * 2 + (y0 < y2)];
    return tn_draw(in, cmd, len, pe, FMUT(133, (pe >> 6) ? -cross : cross, cross));
}

/* a triangle that is neither rejected, clipped nor culled: pe = the sort table entry, area = the (signed, parity corrected) area pair */
static __attribute__((noinline)) int tn_draw(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len, const unsigned pe, const int32_t area)
{
    const struct f3dex_vtx *const V0 = in->v[0], *const V1 = in->v[1], *const V2 = in->v[2];
    const uint32_t g = in->geom;
    const int dlx = in->b->dlx;
    const struct f3dex_vtx *const vH = in->v[pe & 3], *const vM = in->v[(pe >> 2) & 3], *const vL = in->v[(pe >> 4) & 3];
    /* header and edges (5.4) */
    const unsigned flags = g & 7;
    const uint32_t pL = tn_ld32(&vL->sx), pM = tn_ld32(&vM->sx), pH = tn_ld32(&vH->sx);        /* x | y << 16 */
    const int32_t xM = (int16_t)pM, xH = (int16_t)pH;
    const uint16_t csub = (uint16_t)(pH >> 16 << 14);                                           /* the fractional row of the top vertex, as a multiplier */
    const int32_t xHf = xH * 0x4000, xMf = xM * 0x4000;
    const int16x8_t D = vqsubq_s16(vreinterpretq_s16_u32(vsetq_lane_u32(pM, vdupq_n_u32(pL), 1)), vreinterpretq_s16_u32(vsetq_lane_u32(pM, vdupq_n_u32(pH), 0)));
    /* D = [xL-xM yL-yM | xM-xH yM-yH | xL-xH yL-yH | ..]: lanes of dx / dy: L-M, M-H, L-H */
    const int16x4_t dx = vget_low_s16(vuzp1q_s16(D, D)), dy = vget_low_s16(vuzp2q_s16(D, D));
    /* rcp32 of the three dy (5.4 / 1.5) */
    int32x4_t rin = vmovl_s16(dy);
    int32x4_t rmask = vshrq_n_s32(rin, 31);
    int32x4_t rdata = vabsq_s32(rin);                                                           /* data of hle_rcp32 (the input -32768 is overridden below) */
    int32x4_t rsh = vclzq_s32(rdata);
    uint32x4_t ridx = vshrq_n_u32(vandq_u32(vshlq_u32(vreinterpretq_u32_s32(rdata), rsh), vdupq_n_u32(0x7FC00000u)), 22);
    uint16x4_t rr16 = vld1_lane_u16(&rsp_reciprocal_rom[vgetq_lane_u32(ridx, 0)], vdup_n_u16(0), 0);
    rr16 = vld1_lane_u16(&rsp_reciprocal_rom[vgetq_lane_u32(ridx, 1)], rr16, 1);
    rr16 = vld1_lane_u16(&rsp_reciprocal_rom[vgetq_lane_u32(ridx, 2)], rr16, 2);
    uint32x4_t rc = vshlq_u32(vshlq_n_u32(vorrq_u32(vmovl_u16(rr16), vdupq_n_u32(0x10000)), 14), vsubq_s32(rsh, vdupq_n_s32(FMUT(134, 31, 30))));
    rc = veorq_u32(rc, vreinterpretq_u32_s32(rmask));
    rc = vbslq_u32(vceqzq_s32(rdata), vdupq_n_u32(0x7fffffffu), rc);
    rc = vbslq_u32(vceqq_s32(rin, vdupq_n_s32(-32768)), vdupq_n_u32(0xffff0000u), rc);
    /* A = mul_c(rc, 0x20) = rc >> 11; B = dx * 0x4000; S = A x B (F3DLX: mul_mid), / 4, the integer half limited */
    int32x4_t Av = vshrq_n_s32(vreinterpretq_s32_u32(rc), 11);
    int32x4_t Bv = vshll_n_s16(dx, 14);
    int64x2_t sp0 = vmull_s32(vget_low_s32(Av), vget_low_s32(Bv)), sp1 = vmull_high_s32(Av, Bv);
    int32x4_t S = vcombine_s32(vqshrn_n_s64(sp0, 16), vqshrn_n_s64(sp1, 16));
    if (dlx) {
        int32x4_t h = vmull_s16(vshrn_n_s32(Av, 16), vshrn_n_s32(Bv, 16));
        int32x4_t lo = vcombine_s32(vqshrn_n_s64(vsubq_s64(sp0, vshll_n_s32(vget_low_s32(h), 32)), 16), vqshrn_n_s64(vsubq_s64(sp1, vshll_high_n_s32(h, 32)), 16));
        S = vbslq_s32(TN_HI, S, FMUT(122, lo, S));
    }
    S = vshrq_n_s32(S, 2);
    int32x4_t slope = vsliq_n_s32(S, vmaxq_s32(vminq_s32(vshrq_n_s32(S, 16), vdupq_n_s32(FMUT(120, 0x1cc, 0x1cb))), vdupq_n_s32(-0x1cd)), 16);
    /* start x: XL = xM, XH / XM = xH - slope (S.f & 0xfff8) * csub (F3DLX: xH) */
    int32x4_t X;
    if (!dlx) {
        int32x4_t ms = vandq_s32(slope, vdupq_n_s32(FMUT(125, (int32_t)0xfffffff8, -1)));
        X = vsubq_s32(vdupq_n_s32(xHf), tn_mulc(ms, csub));
    } else
        X = vdupq_n_s32(xHf);
    X = vsetq_lane_s32(xMf, X, 0);                                                              /* [xMf XM XH ..] */
    uint64_t y3 = (uint64_t)(pL >> 16) | (uint64_t)(pM >> 16) << 16 | (uint64_t)(pH >> 16) << 32;
    y3 = ((y3 & 0x00ff00ff00ff00ffull) << 8) | ((y3 >> 8) & 0x00ff00ff00ff00ffull);             /* big-endian halfwords */
    const uint64_t w0 = (uint64_t)((g & 0xff) | 0xc8) | (uint64_t)(in->tile_byte | (area < 0 ? 0x80 : 0)) << 8 | y3 << 16;
    memcpy(cmd, &w0, 8);
    {
        int32x4_t z1 = vzip1q_s32(X, slope), z2 = vzip2q_s32(X, slope);                         /* [XL sLM XM sMH] [XH sLH ..] */
        uint8x16_t q1 = vreinterpretq_u8_s32(vcombine_s32(vget_low_s32(z1), vget_low_s32(z2))); /* xMf sLM XH sLH */
        vst1q_u8(cmd + 8, vrev32q_u8(q1));
        vst1_u8(cmd + 24, vrev32_u8(vreinterpret_u8_s32(vget_high_s32(z1))));                   /* XM sMH */
    }
    if (!flags) {
        *len = 32;
        return F3DEX_TRI_OK;
    }

    /* attributes (5.5) */
    const struct f3dex_pair_ { int32_t R; } Rp = { (int32_t)pv(f3dex_recip_refine(area)) };
    const int32_t R = Rp.R;
    const int32_t sl = vgetq_lane_s32(slope, 2);                                                /* the limited slope of the edge L-H */
    /* A4 B4 | D4 C4 (clamp16 of 4 * the deltas), as s16 lanes for the 32-bit chains */
    const int16x4_t dy4 = vqshl_n_s16(dy, 2), dx4 = vqshl_n_s16(dx, 2);
    const int16x4_t Kv = vreinterpret_s16_s32(vzip1_s32(vreinterpret_s32_s16(vext_s16(dy4, dy4, 1)), vreinterpret_s32_s16(vext_s16(dx4, dx4, 1))));     /* [A4 B4 D4 C4] */
    unsigned o = 32;

    if (flags & 4) {
        int32x4_t aHs, dX, dY, dE, st;
        if (g & F3DEX_G_SMOOTH) {
            int16x4_t cH = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(tn_ld32(vH->col))))));
            int16x4_t cM = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(tn_ld32(vM->col))))));
            int16x4_t cL = vget_low_s16(vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(tn_ld32(vL->col))))));
            const int16x4_t dLH = vsub_s16(cL, cH), dHM = vsub_s16(cH, cM);
            /* exact: the colour bytes differ by at most 255 (fraction 0x8000 on both), nx = dLH A4 + dHM B4 (the accumulator is that << 16) */
            int32x4_t nx = vmlal_lane_s16(vmull_lane_s16(dLH, Kv, 0), dHM, Kv, 1);
            int32x4_t ny = vnegq_s32(vmlal_lane_s16(vmull_lane_s16(dHM, Kv, FMUT(123, 3, 2)), dLH, Kv, FMUT(123, 2, 3)));
            aHs = vorrq_s32(vshll_n_s16(cH, 16), vdupq_n_s32(0x8000));
            struct tn_grad gr = tn_grad4(nx, ny, aHs, R, sl, csub, dlx);
            st = gr.st; dX = gr.dX; dY = gr.dY; dE = gr.dE;
            tn_store_block(cmd + o, st, dX, dE, dY);
        } else {
            /* flat: the three colours are the same, nx = ny = 0, dX = dY = dE = 0, start = colour : 0x8000 */
            uint16x4_t c16 = vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(tn_ld32(in->flat->col)))));
            uint8x16_t z = vdupq_n_u8(0);
            uint8x16x4_t r = { { vcombine_u8(vrev16_u8(vreinterpret_u8_u16(c16)), vdup_n_u8(0)),
                                 vcombine_u8(vreinterpret_u8_u16(vdup_n_u16(FMUT(135, 0x0080, 0x0180))), vdup_n_u8(0)), z, z } };
            vst1q_u8_x4(cmd + o, r);
        }
        o += 64;
    }
    if (flags & 3) {
        /* the texture-lane vectors of H, M, L */
        int32x4_t aH, aM, aL;
        int32x4_t texmax = vdupq_n_s32(0);
        int32_t texB = 0;
        uint32x4_t zq = vdupq_n_u32(tn_ld32(&vH->sz));
        zq = vsetq_lane_u32(tn_ld32(&vM->sz), zq, 1);
        zq = vsetq_lane_u32(tn_ld32(&vL->sz), zq, 2);
        const int32x4_t Zv = vreinterpretq_s32_u16(vrev32q_u16(vreinterpretq_u16_u32(zq)));       /* sz : szf pairs */
        if (flags & 2) {
            const int32_t w0_ = (int32_t)((uint32_t)(uint16_t)V0->cx[3] << 16 | V0->cf[3]), w1_ = (int32_t)((uint32_t)(uint16_t)V1->cx[3] << 16 | V1->cf[3]),
                          w2_ = (int32_t)((uint32_t)(uint16_t)V2->cx[3] << 16 | V2->cf[3]);
            int32_t wm = w0_ < w1_ ? w0_ : w1_;
            wm = wm < w2_ ? wm : w2_;
            /* min of the scaled w, halved: floor(floor(w p / 65536) / 2) = floor(w p / 2^17) (monotone in w: the minimum can be taken first) */
            texB = (int32_t)(((int64_t)wm * FMUT(121, in->persp, 0xffff)) >> FMUT(137, 17, 16));
            uint32x4_t rwq = vdupq_n_u32(tn_ld32(&vH->rwi));
            rwq = vsetq_lane_u32(tn_ld32(&vM->rwi), rwq, 1);
            rwq = vsetq_lane_u32(tn_ld32(&vL->rwi), rwq, 2);
            const int32x4_t K = tn_mm(vreinterpretq_s32_u16(vrev32q_u16(vreinterpretq_u16_u32(rwq))), texB);      /* 1/w of the vertex times the scale */
            aH = TN_ATT(vH, K, Zv, 0);
            aM = TN_ATT(vM, K, Zv, 1);
            aL = TN_ATT(vL, K, Zv, 2);
            /* the largest |s|, |t| (integer half; the fraction of the chosen one): H against L first, then against M */
            const int32x4_t cH = tn_absc(aH), cM = tn_absc(aM), cL = tn_absc(aL);
            const int32x4_t kH = vandq_s32(cH, vreinterpretq_s32_u32(TN_HI)), kM = vandq_s32(cM, vreinterpretq_s32_u32(TN_HI)), kL = vandq_s32(cL, vreinterpretq_s32_u32(TN_HI));
            const uint32x4_t m1 = FMUT(129, vcgeq_s32(kH, kL), vcgtq_s32(kH, kL));
            const int32x4_t t1 = vbslq_s32(m1, cH, cL), k1 = vbslq_s32(m1, kH, kL);
            const uint32x4_t m2 = vcgeq_s32(k1, kM);
            texmax = vsetq_lane_s32(texB, vbslq_s32(m2, t1, cM), 2);
        } else {
            aH = TN_ATT0(Zv, 0);
            aM = TN_ATT0(Zv, 1);
            aL = TN_ATT0(Zv, 2);
        }
        const int32x4_t K4 = vmovl_s16(Kv);
        int32x4_t nx = TN_DOT2(tn_sub(aL, aH), tn_sub(aH, aM), K4, 0, 1);
        int32x4_t ny = TN_DOT2(tn_sub(aM, aH), tn_sub(aH, aL), K4, FMUT(131, 3, 2), FMUT(131, 2, 3));
        const struct tn_grad t = tn_grad4(nx, ny, aH, R, sl, csub, dlx);
        if (flags & 2) {
            /* the scale: P = (2 |DaDx.i| + |DaDy.i|) * 0x800 + texmax / 32 (an int32: below 2^28), M = max of lanes 0..2, S = min(1, 2 / (M / 1024)) */
            const int32x4_t E = vshrq_n_s32(texmax, FMUT(136, 5, 6));
            const int32x4_t ax = vshll_n_s16(vqabs_s16(vshrn_n_s32(t.dX, 16)), 12), ay = vshll_n_s16(vqabs_s16(vshrn_n_s32(t.dY, 16)), FMUT(138, 11, 10));
            const int32x4_t P = vaddq_s32(vaddq_s32(ax, ay), E);
            const int32_t M4 = vmaxvq_s32(vsetq_lane_s32(INT32_MIN, P, 3));
            const int32_t rr = (int32_t)hle_rcp32(M4 >> 10);
            const int32_t di = hle_clamp16(rr >> 15);
            const int32_t S = di < FMUT(126, 1, 2) ? (int32_t)((uint32_t)di << 16 | (uint16_t)(rr * 2)) : 0x10000;
            if (S == 0x10000) {
                tn_store_block(cmd + o, t.st, t.dX, t.dE, t.dY);                                    /* mul_mid by 1.0 is the identity */
            } else
                tn_store_block(cmd + o, tn_mm(t.st, S), tn_mm(t.dX, S), tn_mm(t.dE, S), tn_mm(t.dY, S));
            o += 64;
        }
        if (flags & 1) {
            /* z: lane 3, times 32 (saturating), start corrected like the others (F3DLX: not) */
            int32x4_t zq4 = vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(vcopyq_laneq_s32(aH, 0, aH, 3), 1, t.dX, 3), 2, t.dE, 3), 3, t.dY, 3);       /* [zH dX dE dY] */
            zq4 = vqshlq_n_s32(zq4, FMUT(127, 5, 4));
            if (!dlx) {
                const int32x2_t tt = vshrn_n_s64(vmull_n_s32(vdup_n_s32(vgetq_lane_s32(zq4, 2)), (int32_t)csub), 16);
                zq4 = vcopyq_laneq_s32(zq4, 0, tn_sub(zq4, vcombine_s32(tt, tt)), 0);
            }
            vst1q_u8(cmd + o, vrev32q_u8(vreinterpretq_u8_s32(zq4)));                           /* start, DzDx, DzDe, DzDy */
            o += 16;
        }
    }
    *len = o;
    return F3DEX_TRI_OK;
}

int f3dex_tri_setup(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len) { return tn_tri(in, cmd, len, 1); }
int f3dex_tri_setup_noclip(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len) { return tn_tri(in, cmd, len, 0); }

#endif
