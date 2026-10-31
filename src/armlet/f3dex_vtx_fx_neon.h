/*
 * arMLEt - f3dex_vtx_fx_neon.h: the NEON form of f3dex_vtx_xform (aarch64, generic armv8-a); #included at the end of f3dex_vtx_fx.c, so it may use that file's helpers
 *
 * Bit-exact with f3dex_vtx_xform_scalar (the reference, checked against the real microcode on the JIT: spec.f3dex.light.vtx / spec.f3dex.fxvtx.*); tools/f3dex/fxn_vtx.c compares the two on random and
 * extreme inputs, tools/f3dex/bench_vtx.c times them. Layout of the work:
 *  - Position half, per pair of vertices (the microcode's lane halves A / B): the 4x3 matrix transform in 64-bit lanes (smlal by element, 6 per vertex), the clip flags in 32-bit lanes (the VCH / VCL
 *    pair reduced to `le = sat32(X + W) <= 0 [&& (X | W) != 0]`, `ge = X >= W`), the reciprocal in scalar code (table look-up, vn_recip32), the projection and the viewport in 64-bit lanes (smull;
 *    the saturating narrows vqmovn / vqshl replace hle_uclamp / hle_sclamp), fog on the two z values.
 *  - Lighting and texgen, per block of 8 vertices with one vertex per 16-bit lane (vqrdmulh == vmulf); the light-record patterns (lane parity = vertex parity = A / B half) come from table look-ups.
 *
 * The bounds (every step is exact for all inputs; nothing falls back to the scalar function):
 *  - transform: the sums are exact in 64-bit lanes (|x|, |y|, |z| <= 32768, |M| < 2^31 raw: |sum| < 3 * 2^46 + 2^32). The integer half is bits 16..47 of the sum, and the narrowing (shrn #16 to 32 bits)
 *    takes exactly those bits, so it is the 48-bit wrap. Only the low word needs the explicit wrap (it saturates the whole sum to s32 first): vn_init proves sum_j |M_jk| * 2^15 + 2^16 <= 2^47 for
 *    every column (the part sum is then below 2^47 in magnitude: no wrap possible) and takes the wrapping variant (vn_run_wrap) otherwise. The bound is tight (tools/f3dex/fxn_vtx generates matrices at it).
 *  - clip flags: `ge` and `le` are the case analysis of hle_clip_ge / hle_clip_le (comment at vn_cmp); the X == W == 0 exception of `le` is only possible when w or the K-scaled w is 0, i.e. w == 0 or
 *    the ratio == 0: those pairs use the exact variant. K-scaled w: |w * k| < 2^46, its integer half saturates (vqshl), the low word is the product's.
 *  - reciprocal: vn_recip32 is the reference for |w'| <= 32768 and a leaner identical form otherwise; checked against f3dex_recip_refine for all 2^32 inputs (fxn_vtx -recip).
 *  - projection: P * Rc < 2^62; q0 >> 16 < 2^30 fits s32 (the integer half is a vqshl#16 saturation); the low word of (q0 - Pi * Rci << 16) is a vqmovn saturation; (q0 * persp) >> 16 fits s32;
 *    trans << 16 + Q * scale < 2^47 + 2^31 (no 48-bit wrap, the narrowing takes bits 16..47 anyway).
 *  - lighting: 16-bit saturating operations as the microcode's; the texgen sum 2 c0 D + 0x8000 + K Dprev (each term < 2^31 in magnitude) is formed with a halving add (exact) and shifted by 15;
 *    the linear polynomial in 64-bit lanes.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEX_VTX_FX_NEON_H
#define ARMLET_F3DEX_VTX_FX_NEON_H

#define VN_I static inline __attribute__((always_inline))

/* ---- the reciprocal (portable C; tools/f3dex/fxn_vtx -recip checks it against f3dex_recip_refine for every 32-bit input) --------------------------------------------------------------------- */
/* f3dex_recip_refine (hle_rcp32 + one Newton step) for |W| > 32768, in a leaner but identical form. With data = |W| (one's complement for negative W: W <= -32768 takes no +1), shift = clz(data) <= 16,
 * R = ((0x10000 | rom[idx]) << 14 >> (31 - shift)) ^ mask is the half of the doubled reciprocal the code works with (the doubling is folded into the shifts below: R2 = 2 R is below 2^31 in magnitude
 * here, so it never saturates). Then T = R2 * W >> 16 is about 1.0 (65536), N = 2.0 - T = 0x20000 - (s32)T in S15.16 (the pair formed by the "borrow" rule of hle_sub32 is exactly that difference), and the
 * result R2 * N >> 16 is the s15.16 reciprocal: the clamps of the reference never trigger for |W| > 32768 (all the intermediate sums stay far inside s32 / s16). Smaller |W| (and -32768, 0, +-1) take the
 * reference function. */
static uint32_t vn_rtab[512];                   /* (0x10000 | rom[i]) << 14 */
__attribute__((constructor)) static void vn_rtab_init(void)
{
    for (unsigned i = 0; i < 512; i++)
        vn_rtab[i] = (0x10000u | rsp_reciprocal_rom[i]) << 14;
}
VN_I int32_t vn_recip_fast(int32_t W)
{
    int32_t m = W >> 31;
    uint32_t d = (uint32_t)(W ^ m);
    unsigned sh = (unsigned)__builtin_clz(d);
    int32_t R = (int32_t)((vn_rtab[((d << sh) >> 22) & 0x1ff] >> (sh ^ 31)) ^ (uint32_t)m);
    int32_t T = (int32_t)(((int64_t)R * W) >> 15);
    int32_t N = FMUT(80, 0x20000, 0x20001) - T;
    return (int32_t)(((int64_t)R * N) >> 15);
}
VN_I int vn_recip_slow(int32_t W) { return (uint32_t)W + 32768u <= 65536u; }
/* returns the reciprocal as the s32 (rwi in the high half, rwf in the low half), i.e. hle_pair32(f3dex_recip_refine(W)) */
VN_I int32_t vn_recip32(int32_t W)
{
    if (__builtin_expect(vn_recip_slow(W), 0))
        return hle_pair32(f3dex_recip_refine(W));
    return vn_recip_fast(W);
}
/* two reciprocals (the fast bodies in one basic block, for the scheduler to interleave) */
VN_I void vn_recip2(int32_t Wa, int32_t Wb, int32_t *ra, int32_t *rb)
{
    if (__builtin_expect(vn_recip_slow(Wa) | vn_recip_slow(Wb), 0)) {
        *ra = vn_recip32(Wa);
        *rb = vn_recip32(Wb);
        return;
    }
    *ra = vn_recip_fast(Wa);
    *rb = vn_recip_fast(Wb);
}

#if defined(__aarch64__) && !defined(__ARM_BIG_ENDIAN) && !defined(VN_RECIP_ONLY)      /* (the intrinsics below are AArch64's; a 32-bit NEON or big-endian target gets the scalar form) */
#include <arm_neon.h>

/* unaligned 16 / 32-bit stores and loads of record fields (the records are 38 bytes: odd ones are only 2-aligned) */
VN_I void vn_st32(void *p, uint32_t v) { memcpy(p, &v, 4); }
VN_I uint32_t vn_ld32(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* ---- the call constants ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */
struct vn {
    const struct f3dex_vctx *c;
    int clip, lit, texgen, linear, fog, non, dlx, wrap;
    int32x2_t mlo[3], mhi[3];           /* rows 0..2 of the matrix as raw s15.16, columns (0,1) and (2,3) */
    int64x2_t cplo, cphi;               /* row 3 fractions */
    int64x2_t cilo, cihi;               /* row 3 integers << 16 */
    int32x4_t sc4;                      /* viewport scale (sx, sy, sz, 0) */
    int32x4_t c64k;
    int64x2_t trlo, trhi;               /* viewport translation << 16 */
    int16x8_t limlo, limhi;             /* screen clamps for the lanes of two vertices (lane 3 / 7: none) */
    uint16x8_t wE, wA;                  /* flag bit weights of the le compares [le2 x y z w | le1 x y z w] and of both (le and ge, the ge weights being 0x100 << k, 0x1000 << k) */
    int32_t persp, clipk, fog0, fog1, fog2;
    int16x4_t tsv, tsfix;               /* the texture scale (s t s t) as signed halfwords and the correction mask for scales >= 0x8000 */
};

VN_I void vn_init(struct vn *k, const struct f3dex_vctx *c)
{
    k->c = c;
    k->clip = !c->b->dlx || (c->geom & F3DEX_G_CLIPPING) != 0;
    k->lit = (c->geom & F3DEX_G_LIGHTING) != 0;
    k->texgen = (c->geom & F3DEX_G_TEXGEN) != 0;
    k->linear = (c->geom & F3DEX_G_TEXGEN_LINEAR) != 0;
    k->fog = (c->geom & F3DEX_G_FOG) != 0;
    k->non = c->b->non;
    k->dlx = c->b->dlx;
    k->persp = c->persp;
    k->clipk = c->clip_k;
    k->fog0 = c->fog[0];
    k->fog1 = c->fog[1];
    k->fog2 = c->fog[2];
    /* matrix: rows 0..2 raw (i << 16 | f); the wrap guard: with |x|, |y|, |z| <= 32768 the low-word sum of a column stays below 2^47 in magnitude when sum |M| * 2^15 + 2^16 <= 2^47 */
    uint64x2_t sl = vdupq_n_u64(0), sh = vdupq_n_u64(0);
    for (int j = 0; j < 3; j++) {
        int32x4_t m = vorrq_s32(vshll_n_s16(vld1_s16(c->mp.i[j]), 16), vreinterpretq_s32_u32(vmovl_u16(vld1_u16(c->mp.f[j]))));
        k->mlo[j] = vget_low_s32(m);
        k->mhi[j] = vget_high_s32(m);
        uint32x4_t a = vreinterpretq_u32_s32(vabsq_s32(m));              /* INT_MIN stays 2^31 as unsigned */
        sl = vaddw_u32(sl, vget_low_u32(a));
        sh = vaddw_u32(sh, vget_high_u32(a));
    }
    int16x4_t i3 = vld1_s16(c->mp.i[3]);
    uint32x4_t f3 = vmovl_u16(vld1_u16(c->mp.f[3]));
    k->cplo = vreinterpretq_s64_u64(vmovl_u32(vget_low_u32(f3)));
    k->cphi = vreinterpretq_s64_u64(vmovl_u32(vget_high_u32(f3)));
    int32x4_t i3w = vmovl_s16(i3);
    k->cilo = vshll_n_s32(vget_low_s32(i3w), 16);
    k->cihi = vshll_n_s32(vget_high_s32(i3w), 16);
    /* + 2 (units of 2^15): the row-3 fraction (< 2^16); the integer part of row 3 is added after the low word is read and cannot matter (the integer half is the sum's bits 16..47: the narrowing drops the rest) */
    sl = vaddq_u64(sl, vdupq_n_u64(2));
    sh = vaddq_u64(sh, vdupq_n_u64(2));
    const uint64_t thr = FMUT(78, FMUT(79, 1ull << 32, (1ull << 32) + 65536), 1ull << 34);
    uint64x2_t big = vorrq_u64(vcgtq_u64(sl, vdupq_n_u64(thr)), vcgtq_u64(sh, vdupq_n_u64(thr)));
    k->wrap = FMUT(60, (vgetq_lane_u64(big, 0) | vgetq_lane_u64(big, 1)) != 0, 0);
    /* viewport */
    int32_t sc[3];
    for (int j = 0; j < 3; j++)
        sc[j] = hle_clamp16((int32_t)c->vscale[j] * c->vsign[j]);
    k->sc4 = vcombine_s32(vcreate_s32((uint64_t)(uint32_t)sc[0] | (uint64_t)(uint32_t)sc[1] << 32), vcreate_s32((uint64_t)(uint32_t)sc[2]));
    k->c64k = vdupq_n_s32(65536);
    k->trlo = vcombine_s64(vcreate_s64((uint64_t)((int64_t)c->vtrans[0] << 16)), vcreate_s64((uint64_t)((int64_t)c->vtrans[1] << 16)));
    k->trhi = vcombine_s64(vcreate_s64((uint64_t)((int64_t)c->vtrans[2] << 16)), vcreate_s64(0));
    {
        const uint64_t l1 = (uint16_t)c->lim[1], l3 = (uint16_t)c->lim[3], h0 = (uint16_t)c->lim[0], h2 = (uint16_t)c->lim[2];
        uint64_t lo4 = l1 | l1 << 16 | l3 << 32 | 0x8000ull << 48, hi4 = h0 | h0 << 16 | h2 << 32 | 0x7fffull << 48;      /* lane 3 / 7 (the z fraction): no clamp */
        k->limlo = vreinterpretq_s16_u64(vcombine_u64(vcreate_u64(lo4), vcreate_u64(lo4)));
        k->limhi = vreinterpretq_s16_u64(vcombine_u64(vcreate_u64(hi4), vcreate_u64(hi4)));
    }
    {
        static const uint16_t wl[2][8] = { { 1, 2, 4, 0, 16, 32, 64, 0 }, { 1, 2, 4, 0, 16, 32, FMUT(61, 0, 64), 0 } };      /* F3DEX.NoN: no "z <= -w" bit in the first compare */
        static const uint16_t wa[2][8] = { { 0x101, 0x202, 0x404, 0, 0x1010, 0x2020, 0x4040, 0 }, { 0x101, 0x202, 0x404, 0, 0x1010, 0x2020, FMUT(61, 0x4000, 0x4040), 0 } };
        k->wE = vld1q_u16(wl[k->non]);
        k->wA = vld1q_u16(wa[k->non]);
    }
    /* texture scale */
    {
        const uint64_t t0 = c->tscale[0], t1 = c->tscale[1];
        const uint64_t m = (uint16_t)FMUT(66, -1, 0), f0 = t0 >= 0x8000 ? m : 0, f1 = t1 >= 0x8000 ? m : 0;
        k->tsv = vreinterpret_s16_u64(vcreate_u64(t0 | t1 << 16 | t0 << 32 | t1 << 48));
        k->tsfix = vreinterpret_s16_u64(vcreate_u64(f0 | f1 << 16 | f0 << 32 | f1 << 48));
    }
}

/* ---- the position half --------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* 48-bit wrap of 64-bit lanes */
VN_I int64x2_t vn_wrap48(int64x2_t v) { return vshrq_n_s64(vshlq_n_s64(v, 16), 16); }

/* The clip coordinates of one vertex: hi = the integer halves (sat16 of the sum shifted right by 16, after the row-3 integer part), lo = the fractions (the low word of the sum before the row-3
 * integer part, 0 / 0xffff when that sum leaves s32: the low word of its saturation to 32 bits). The sums are exact in 64-bit lanes; without `wrap` they cannot leave 48 bits (vn_init). */
VN_I void vn_xform(const struct vn *k, const struct f3dex_vin *in, int16x4_t *hi, uint16x4_t *lo, const int wrap)
{
    int32x4_t x = vmovl_s16(vld1_s16(&in->x));                      /* x y z s */
    int64x2_t a0 = vmlal_laneq_s32(k->cplo, k->mlo[0], x, 0);
    int64x2_t a1 = vmlal_laneq_s32(k->cphi, k->mhi[0], x, 0);
    a0 = vmlal_laneq_s32(a0, k->mlo[1], x, 1);
    a1 = vmlal_laneq_s32(a1, k->mhi[1], x, 1);
    a0 = vmlal_laneq_s32(a0, k->mlo[2], x, 2);
    a1 = vmlal_laneq_s32(a1, k->mhi[2], x, 2);
    int64x2_t f0 = vaddq_s64(a0, k->cilo), f1 = vaddq_s64(a1, k->cihi);
    if (wrap) {                         /* only the low word needs it: the narrowing below to bits 16..47 of the full sum is the wrap */
        a0 = vn_wrap48(a0);
        a1 = vn_wrap48(a1);
    }
    *hi = vqmovn_s32(vshrn_high_n_s64(vshrn_n_s64(f0, 16), f1, 16));
    *lo = vreinterpret_u16_s16(vmovn_s32(FMUT(70, vqmovn_high_s64(vqmovn_s64(a0), a1), vcombine_s32(vmovn_s64(a0), vmovn_s64(a1)))));
}

/* X <= -W and X >= W as the VCH / VCL pair gives them (hle_clip_le / hle_clip_ge; f3dex_vtx_flagparts), for 4 coordinates against a broadcast w, in 32-bit lanes:
 *   ge = X >= W   (opposite signs: W < 0 is X >= 0 and then X >= W; same sign: X - W >= 0; X < 0 <= W: false and X < W)
 *   le = sat(X + W) <= 0, except X == W == 0 (opposite signs: X + W <= 0 exactly; both negative: true, the sum saturates below 0; both >= 0: false, the sum is > 0 unless both are 0)
 * The exception needs W == 0: `exact` adds it (used only for a pair that has a zero w or K-scaled w). */
VN_I void vn_cmp(int32x4_t X, int32x4_t W, uint32x4_t *le, uint32x4_t *ge, const int exact)
{
    uint32x4_t l = FMUT(62, vclezq_s32(vqaddq_s32(X, W)), vcltzq_s32(vqaddq_s32(X, W)));
    if (exact) {
        int32x4_t o = vorrq_s32(X, W);
        l = vandq_u32(l, vtstq_s32(o, o));
    }
    *le = l;
    *ge = FMUT(72, vcgeq_s32(X, W), vcgtq_s32(X, W));
}

VN_I uint16x8_t vn_flags_sets(const struct vn *k, int32x4_t P0, int32x4_t P1, int32x2_t W2, const int exact)
{
    uint32x4_t le1[2], ge1[2], le2[2], ge2[2];
    vn_cmp(P0, vdupq_laneq_s32(P0, 3), &le1[0], &ge1[0], exact);
    vn_cmp(P1, vdupq_laneq_s32(P1, 3), &le1[1], &ge1[1], exact);
    vn_cmp(P0, vdupq_lane_s32(W2, 0), &le2[0], &ge2[0], exact);
    vn_cmp(P1, vdupq_lane_s32(W2, 1), &le2[1], &ge2[1], exact);
    uint16x8_t F[2];
    for (int v = 0; v < 2; v++) {
        uint16x8_t E = vuzp1q_u16(vreinterpretq_u16_u32(le2[v]), vreinterpretq_u16_u32(le1[v]));
        uint16x8_t G = vuzp1q_u16(vreinterpretq_u16_u32(ge2[v]), vreinterpretq_u16_u32(ge1[v]));
        F[v] = vandq_u16(vbslq_u16(k->wE, E, G), k->wA);          /* (E & wE) | (G & wG), the weights being disjoint bit sets */
    }
    uint16x8_t s = vpaddq_u16(F[0], F[1]);
    s = vpaddq_u16(s, s);
    return vpaddq_u16(s, s);
}

/* the clip flags of two vertices (P0, P1: x y z w as s32 lanes, w0 / w1 their w): u16 lanes 0 and 1 of the result (f2 | f1 << 4, see f3dex_vtx_xform_scalar). K-scaled w is zero exactly when w or the
 * ratio is. */
VN_I uint16x8_t vn_flags(const struct vn *k, int32x4_t P0, int32x4_t P1, int32_t w0, int32_t w1)
{
    /* W2 = ratio * W: the 48-bit product's integer half saturates to s16, the fraction is its low 16 bits */
    int32x4_t Wp = vtrn2q_s32(P0, P1);                                  /* (.., .., w0, w1) */
    int64x2_t a = vmull_high_n_s32(Wp, k->clipk);
    int32x2_t t = vqshl_n_s32(vshrn_n_s64(a, 16), 16);
    int32x2_t W2 = vreinterpret_s32_u32(vbsl_u32(vdup_n_u32(0xffff0000u), vreinterpret_u32_s32(FMUT(69, t, vshl_n_s32(vshrn_n_s64(a, 16), 16))), vreinterpret_u32_s32(vmovn_s64(a))));
    if (__builtin_expect(FMUT(81, w0 == 0 || w1 == 0 || k->clipk == 0, 0), 0))
        return vn_flags_sets(k, P0, P1, W2, 1);
    return vn_flags_sets(k, P0, P1, W2, 0);
}

/* the reciprocals of w (scalar: table look-up) of two vertices: Rc = the clamped reciprocal as s15.16 for the projection, Rci its integer half, rw the record's rwi / rwf word */
VN_I void vn_recip(const struct vn *k, uint32_t Wa, uint32_t Wb, int32_t *Rca, int32_t *Rcia, uint32_t *rwa, int32_t *Rcb, int32_t *Rcib, uint32_t *rwb)
{
    /* hle_mac_32xu16 (|.| < 2^31: the pair (sclamp, uclamp) is the s32 itself) */
    int32_t wa = (int32_t)(((int64_t)(int32_t)Wa * k->persp) >> 16), wb = (int32_t)(((int64_t)(int32_t)Wb * k->persp) >> 16);
    int32_t ra, rb;
    vn_recip2(wa, wb, &ra, &rb);
    int32_t ia = ra >> 16, ib = rb >> 16;
    *Rca = ia >= 0 ? ra : (int32_t)(0x7fff0000u | (uint32_t)(ra & 0xffff));
    *Rcb = ib >= 0 ? rb : (int32_t)(0x7fff0000u | (uint32_t)(rb & 0xffff));
    *Rcia = ia >= 0 ? ia : 0x7fff;
    *Rcib = ib >= 0 ? ib : 0x7fff;
    *rwa = (uint32_t)ra >> 16 | (uint32_t)ra << 16;
    *rwb = (uint32_t)rb >> 16 | (uint32_t)rb << 16;
}

/* Projection and viewport of one vertex. P = x y z w as s32 (w unused), returns Q (the projected position, s32 lanes) and the screen halfwords x y z and the low word of the z sum (lane 3).
 *   q0 = P * Rc >> 16 (64 bits): integer half sat16(q0 >> 16) = sat16(P * Rc >> 32); fraction = low word of sat32(q0 - (Pi * Rci << 16)) (the word read before the last term, "lo3");
 *   Q = (q0 * persp) >> 16 (fits s32); v = trans << 16 + Q * scale: screen = sat16(v >> 16), z fraction = low word of sat32(v). */
VN_I void vn_proj(const struct vn *k, int32x4_t P, int16x4_t Pi, int32_t Rc, int32_t Rci, int32x4_t *Qout, int16x4_t *scr)
{
    int64x2_t pl = vmull_n_s32(vget_low_s32(P), Rc), ph = vmull_high_n_s32(P, Rc);
    int32x4_t qir = vshrn_high_n_s64(vshrn_n_s64(pl, 32), ph, 32);
    int32x4_t q0h = vqshlq_n_s32(qir, 16);
    int32x4_t pr = vmull_n_s16(Pi, (int16_t)FMUT(65, Rci, 0));          /* Pi * Rci */
    int64x2_t tl = vmlsl_s32(vshrq_n_s64(pl, 16), vget_low_s32(pr), vget_low_s32(k->c64k));
    int64x2_t th = vmlsl_high_s32(vshrq_n_s64(ph, 16), pr, k->c64k);
    int32x4_t ts = FMUT(77, vqmovn_high_s64(vqmovn_s64(tl), th), vcombine_s32(vmovn_s64(tl), vmovn_s64(th)));
    int32x4_t Q0 = vreinterpretq_s32_u32(vbslq_u32(vdupq_n_u32(0xffff0000u), vreinterpretq_u32_s32(q0h), vreinterpretq_u32_s32(ts)));
    int64x2_t ql = vmull_n_s32(vget_low_s32(Q0), (int32_t)k->persp), qh = vmull_high_n_s32(Q0, (int32_t)k->persp);
    int32x4_t Q = vshrn_high_n_s64(vshrn_n_s64(ql, 16), qh, 16);
    int64x2_t vl = vmlal_s32(k->trlo, vget_low_s32(Q), vget_low_s32(k->sc4));
    int64x2_t vh = vmlal_high_s32(k->trhi, Q, k->sc4);
    *Qout = Q;
    int16x4_t s4 = vqmovn_s32(vshrn_high_n_s64(vshrn_n_s64(vl, 16), vh, 16));
    *scr = vcopy_lane_s16(s4, 3, vreinterpret_s16_s32(vqmovn_s64(vh)), 0);        /* lane 3: the low word of sat32(v z) */
}

/* ---- one pair (or the last single vertex) -------------------------------------------------------------------------------------------------------------------------------------- */
/* oa / ob: the records (ob is a scratch record when two == 0); adj: F3DLX without G_CLIPPING, unlit, first pair of the call: the VCO carry goes into the integer halves; fogb: where the fog bytes
 * go when the lit block writes the colours (NULL: the colour word is written here). */
VN_I void vn_pair(const struct vn *k, const struct f3dex_vin *ia, const struct f3dex_vin *ib, struct f3dex_vtx *oa, struct f3dex_vtx *ob, const int two, const int adj, uint8_t *fogb, const int wrap)
{
    int16x4_t ha, hb;
    uint16x4_t la, lb;
    vn_xform(k, ia, &ha, &la, wrap);
    vn_xform(k, ib, &hb, &lb, wrap);
    int16x8_t hi = vcombine_s16(ha, hb);
    uint16x8_t lo = vcombine_u16(la, lb);
    int16x8_t hiA = hi;
    if (adj) {
        static const uint16_t bit_w[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
        uint16x8_t bits = vld1q_u16(bit_w);
        uint16x8_t cr = FMUT(68, vshrq_n_u16(vtstq_u16(vdupq_n_u16(k->c->vco_carry), bits), 15), vdupq_n_u16(0));
        hiA = vqaddq_s16(hi, vreinterpretq_s16_u16(cr));
    }
    /* s32 lanes: vertex a (x y z w), vertex b */
    uint16x8x2_t z = vzipq_u16(lo, vreinterpretq_u16_s16(hiA));
    int32x4_t PA = vreinterpretq_s32_u16(z.val[0]), PB = vreinterpretq_s32_u16(z.val[1]);
    /* reciprocals of w (the unadjusted integer half: with adj the carry does not reach the reciprocal) */
    uint32_t Wa = adj ? (uint32_t)(uint16_t)vget_lane_s16(ha, 3) << 16 | vget_lane_u16(la, 3) : (uint32_t)vgetq_lane_s32(PA, 3);
    uint32_t Wb = adj ? (uint32_t)(uint16_t)vget_lane_s16(hb, 3) << 16 | vget_lane_u16(lb, 3) : (uint32_t)vgetq_lane_s32(PB, 3);
    if (k->clip) {
        uint16x8_t fl = vn_flags(k, PA, PB, (int32_t)Wa, (int32_t)Wb);
        uint32_t fa = vgetq_lane_u16(fl, 0), fb = vgetq_lane_u16(fl, 1);
        if (k->non && two) {
            uint32_t na = (uint32_t)(uint16_t)vgetq_lane_s16(hi, 3) >> 15, nb = (uint32_t)(uint16_t)vgetq_lane_s16(hi, 7) >> 15;
            fb |= FMUT(63, na << 2 | nb << 6, na << 6 | nb << 2);
        }
        oa->flags = (uint16_t)fa;
        ob->flags = (uint16_t)fb;
    } else {
        oa->flags = 0;
        ob->flags = 0;
    }
    int32_t Rca, Rcia, Rcb, Rcib;
    uint32_t rwa, rwb;
    vn_recip(k, Wa, Wb, &Rca, &Rcia, &rwa, &Rcb, &Rcib, &rwb);
    int32x4_t Qa, Qb;
    int16x4_t sa, sb;
    vn_proj(k, PA, vget_low_s16(hiA), Rca, Rcia, &Qa, &sa);
    vn_proj(k, PB, vget_high_s16(hiA), Rcb, Rcib, &Qb, &sb);
    int16x8_t scr = vminq_s16(vmaxq_s16(vcombine_s16(sa, sb), k->limlo), k->limhi);
    if (k->dlx) {                       /* F3DLX: (y + 2) & ~3 */
        static const int16_t add2[8] = { 0, 2, 0, 0, 0, 2, 0, 0 }, mask[8] = { -1, FMUT(71, -4, -8), -1, -1, -1, FMUT(71, -4, -8), -1, -1 };
        scr = vandq_s16(vaddq_s16(scr, vld1q_s16(add2)), vld1q_s16(mask));
    }
    vst1_s16(oa->cx, vget_low_s16(hiA));
    vst1_s16(ob->cx, vget_high_s16(hiA));
    vst1_u16(oa->cf, vget_low_u16(lo));
    vst1_u16(ob->cf, vget_high_u16(lo));
    vst1_s16(&oa->sx, vget_low_s16(scr));
    vst1_s16(&ob->sx, vget_high_s16(scr));
    vn_st32(&oa->rwi, rwa);
    vn_st32(&ob->rwi, rwb);
    /* fog byte of the two vertices from the projected z */
    uint32_t fga = 0, fgb = 0;
    if (k->fog) {
        int64x2_t r = vzip1q_s64(vmull_n_s32(vget_high_s32(Qa), k->fog0), vmull_n_s32(vget_high_s32(Qb), k->fog0));
        int32x2_t s = vshrn_n_s64(r, 16);
        int16x4_t f = vqadd_s16(vqmovn_s32(vcombine_s32(s, s)), vdup_n_s16((int16_t)k->fog1));
        f = FMUT(64, vmax_s16(f, vdup_n_s16(0)), f);
        f = vmin_s16(f, vdup_n_s16((int16_t)k->fog2));
        fga = vget_lane_u16(vreinterpret_u16_s16(f), 0) & 0xff;
        fgb = vget_lane_u16(vreinterpret_u16_s16(f), 1) & 0xff;
        if (fogb) {
            fogb[0] = (uint8_t)fga;
            fogb[1] = (uint8_t)fgb;
        }
    }
    /* colour word and texture coordinates (unlit; lit: the lit block writes the colours, and the texture coordinates when texgen is on) */
    if (!k->lit) {
        uint32_t ca = vn_ld32(ia->c), cb = vn_ld32(ib->c);
        if (k->fog) {
            ca = (ca & 0x00ffffffu) | fga << 24;
            cb = (cb & 0x00ffffffu) | fgb << 24;
        }
        vn_st32(oa->col, ca);
        vn_st32(ob->col, cb);
    }
    if (!(k->lit && k->texgen)) {
        uint32x2_t w = vset_lane_u32(vn_ld32(&ib->s), vdup_n_u32(vn_ld32(&ia->s)), 1);       /* s t of a, s t of b */
        int16x4_t st = vreinterpret_s16_u32(w);
        int16x4_t r = vadd_s16(vshrn_n_s32(vmull_s16(st, k->tsv), 16), vand_s16(st, k->tsfix));
        uint32x2_t rr = vreinterpret_u32_s16(r);
        vn_st32(&oa->s, vget_lane_u32(rr, 0));
        vn_st32(&ob->s, vget_lane_u32(rr, 1));
    }
}

/* ---- lighting and texgen: 8 vertices, one per 16-bit lane --------------------------------------------------------------------------------------------------------------------------- */
static const uint8_t vn_idx_nrm[4][16] = {                  /* byte (3 + ...) of word i of the 8 gathered words */
    { 255, 0, 255, 4, 255, 8, 255, 12, 255, 16, 255, 20, 255, 24, 255, 28 },
    { 255, 1, 255, 5, 255, 9, 255, 13, 255, 17, 255, 21, 255, 25, 255, 29 },
    { 255, 2, 255, 6, 255, 10, 255, 14, 255, 18, 255, 22, 255, 26, 255, 30 },
    { 3, 7, 11, 15, 19, 23, 27, 31, 255, 255, 255, 255, 255, 255, 255, 255 } };
static const uint8_t vn_idx_dir[3][16] = {                  /* lane parity = A / B half: byte j of the A or B part of a record, in the high byte of the halfword */
    { 255, 0, 255, 4, 255, 0, 255, 4, 255, 0, 255, 4, 255, 0, 255, 4 },
    { 255, 1, 255, 5, 255, 1, 255, 5, 255, 1, 255, 5, 255, 1, 255, 5 },
    { 255, 2, 255, 6, 255, 2, 255, 6, 255, 2, 255, 6, 255, 2, 255, 6 } };
static const uint8_t vn_idx_col[3][16] = {                  /* halfword j or 4 + j of an 8 x u16 table */
    { 0, 1, 8, 9, 0, 1, 8, 9, 0, 1, 8, 9, 0, 1, 8, 9 },
    { 2, 3, 10, 11, 2, 3, 10, 11, 2, 3, 10, 11, 2, 3, 10, 11 },
    { 4, 5, 12, 13, 4, 5, 12, 13, 4, 5, 12, 13, 4, 5, 12, 13 } };
static const uint8_t vn_idx_lk[2][16] = {                   /* byte j or 4 + j in the low byte */
    { 0, 255, 4, 255, 0, 255, 4, 255, 0, 255, 4, 255, 0, 255, 4, 255 },
    { 1, 255, 5, 255, 1, 255, 5, 255, 1, 255, 5, 255, 1, 255, 5, 255 } };

/* the colours of record k as 3 vectors (u8 << 7 per lane, lane parity picks the A / B half) */
VN_I void vn_colpat(const uint8_t *row, int16x8_t out[3])
{
    uint8x16_t t = vreinterpretq_u8_u16(vshll_n_u8(vld1_u8(row), 7));
    for (int j = 0; j < 3; j++)
        out[j] = vreinterpretq_s16_u8(vqtbl1q_u8(t, vld1q_u8(vn_idx_col[j])));
}

VN_I int16x8_t vn_dot(const int16x8_t n[3], const int8_t *row)
{
    uint8x16_t r = vld1q_u8((const uint8_t *)row);                  /* 16 bytes: the following record (or the bytes after the table) are not looked at */
    int16x8_t d0 = vreinterpretq_s16_u8(vqtbl1q_u8(r, vld1q_u8(vn_idx_dir[0])));
    int16x8_t p0 = vqrdmulhq_s16(n[0], d0);          /* (the rounding term never matters here: both factors have a zero low byte) */
    int16x8_t p1 = vqrdmulhq_s16(n[1], vreinterpretq_s16_u8(vqtbl1q_u8(r, vld1q_u8(vn_idx_dir[1]))));
    int16x8_t p2 = vqrdmulhq_s16(n[2], vreinterpretq_s16_u8(vqtbl1q_u8(r, vld1q_u8(vn_idx_dir[2]))));
    return vqaddq_s16(vqaddq_s16(p0, p1), p2);
}

VN_I void vn_lit8(const struct vn *k, const struct f3dex_vin *in, struct f3dex_vtx *out, unsigned m, const uint8_t *fogb)
{
    const struct f3dex_vctx *c = k->c;
    uint32x4_t qa = vdupq_n_u32(0), qb = vdupq_n_u32(0);
    qa = vsetq_lane_u32(vn_ld32(in[0].c), qa, 0);
    qa = vsetq_lane_u32(vn_ld32(in[1].c), qa, 1);
    qa = vsetq_lane_u32(vn_ld32(in[2].c), qa, 2);
    qa = vsetq_lane_u32(vn_ld32(in[3].c), qa, 3);
    qb = vsetq_lane_u32(vn_ld32(in[4].c), qb, 0);
    qb = vsetq_lane_u32(vn_ld32(in[5].c), qb, 1);
    qb = vsetq_lane_u32(vn_ld32(in[6].c), qb, 2);
    qb = vsetq_lane_u32(vn_ld32(in[7].c), qb, 3);
    uint8x16x2_t tb = { { vreinterpretq_u8_u32(qa), vreinterpretq_u8_u32(qb) } };
    int16x8_t n[3];
    for (int j = 0; j < 3; j++)
        n[j] = vreinterpretq_s16_u8(vqtbl2q_u8(tb, vld1q_u8(vn_idx_nrm[j])));      /* (signed byte) << 8 */
    /* ambient + the lights (records nrec .. 1): acc += vmulf(colour << 7, max(D, 0)); the contribution of record 0 is not added */
    int16x8_t acc[3];
    vn_colpat(c->lcol[c->nrec + 1], acc);
    int16x8_t D = vdupq_n_s16(0), Dprev = D;
    for (int r = (int)c->nrec; r >= 1; r--) {
        D = vn_dot(n, c->ldir[r]);
        int16x8_t Dp = FMUT(75, vmaxq_s16(D, vdupq_n_s16(0)), D);
        int16x8_t cl[3];
        vn_colpat(c->lcol[r], cl);
        for (int j = 0; j < 3; j++)
            acc[j] = vqaddq_s16(acc[j], FMUT(73, vqrdmulhq_s16(cl[j], Dp), vqdmulhq_s16(cl[j], Dp)));
    }
    Dprev = D;
    int16x8_t sT = vdupq_n_s16(0), tT = sT;
    if (k->texgen) {
        /* texgen: record 0's dot product and colour, the previous dot product (record 1) weighted by the lk bytes */
        D = vn_dot(n, c->ldir[0]);
        int16x8_t c0[3];
        vn_colpat(c->lcol[0], c0);
        uint8x16_t lkb = vcombine_u8(vld1_u8(c->lk), vdup_n_u8(0));
        int16x8_t v[2];
        for (int j = 0; j < 2; j++) {
            int16x8_t kw = vreinterpretq_s16_u8(vqtbl1q_u8(lkb, vld1q_u8(vn_idx_lk[j])));
            /* a = 2 c0 D + 0x8000 + (lk << 8) * Dprev: sat16(a >> 16) as the halving sum shifted by 15 */
            int32x4_t A0 = vaddq_s32(vqdmull_s16(vget_low_s16(c0[j]), vget_low_s16(D)), vdupq_n_s32(FMUT(74, 0x8000, 0)));
            int32x4_t A1 = vaddq_s32(vqdmull_high_s16(c0[j], D), vdupq_n_s32(FMUT(74, 0x8000, 0)));
            int32x4_t B0 = vshlq_n_s32(vmull_s16(vget_low_s16(kw), vget_low_s16(Dprev)), 8);
            int32x4_t B1 = vshlq_n_s32(vmull_high_s16(kw, Dprev), 8);
            v[j] = vcombine_s16(vqshrn_n_s32(vhaddq_s32(A0, B0), 15), vqshrn_n_s32(vhaddq_s32(A1, B1), 15));
            if (k->linear) {
                /* v = sat16((2 cu tg0 + 0x8000 + 2 cu tg2 + 2 v tg1) >> 16), sq = vmulf(v, v), cu = vmulf(sq, v) */
                int16x8_t sq = vqrdmulhq_s16(v[j], v[j]);
                int16x8_t cu = vqrdmulhq_s16(sq, v[j]);
                int16x4_t t0 = vdup_n_s16(c->tg[0]), t1 = vdup_n_s16(FMUT(76, c->tg[1], c->tg[0])), t2 = vdup_n_s16(c->tg[2]);
                int32x4_t p0 = vmull_s16(vget_low_s16(cu), t0), p1 = vmull_s16(vget_low_s16(cu), t2), p2 = vmull_s16(vget_low_s16(v[j]), t1);
                int32x4_t q0 = vmull_s16(vget_high_s16(cu), t0), q1 = vmull_s16(vget_high_s16(cu), t2), q2 = vmull_s16(vget_high_s16(v[j]), t1);
                int64x2_t s0 = vaddw_s32(vaddl_s32(vget_low_s32(p0), vget_low_s32(p1)), vget_low_s32(p2));
                int64x2_t s1 = vaddw_high_s32(vaddl_high_s32(p0, p1), p2);
                int64x2_t s2 = vaddw_s32(vaddl_s32(vget_low_s32(q0), vget_low_s32(q1)), vget_low_s32(q2));
                int64x2_t s3 = vaddw_high_s32(vaddl_high_s32(q0, q1), q2);
                int64x2_t h = vdupq_n_s64(0x4000);
                int32x4_t r0 = vshrn_high_n_s64(vshrn_n_s64(vaddq_s64(s0, h), 15), vaddq_s64(s1, h), 15);
                int32x4_t r1 = vshrn_high_n_s64(vshrn_n_s64(vaddq_s64(s2, h), 15), vaddq_s64(s3, h), 15);
                v[j] = vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1));
            }
        }
        int16x8_t bias = vdupq_n_s16(c->tgbias);
        int16x8_t s8 = FMUT(67, vqaddq_s16(v[0], bias), vaddq_s16(v[0], bias)), t8 = FMUT(67, vqaddq_s16(v[1], bias), vaddq_s16(v[1], bias));
        /* texture scale: (st * scale) >> 16 */
        int16_t ss = (int16_t)c->tscale[0], ts = (int16_t)c->tscale[1];
        sT = vcombine_s16(vshrn_n_s32(vmull_n_s16(vget_low_s16(s8), ss), 16), vshrn_n_s32(vmull_n_s16(vget_high_s16(s8), ss), 16));
        tT = vcombine_s16(vshrn_n_s32(vmull_n_s16(vget_low_s16(t8), ts), 16), vshrn_n_s32(vmull_n_s16(vget_high_s16(t8), ts), 16));
        if (c->tscale[0] >= 0x8000 && FMUT(66, 1, 0))
            sT = vaddq_s16(sT, s8);
        if (c->tscale[1] >= 0x8000 && FMUT(66, 1, 0))
            tT = vaddq_s16(tT, t8);
    }
    /* colour words: rgb = acc >> 7, a = the vertex's own alpha or the fog byte */
    uint8x8_t r8 = vshrn_n_u16(vreinterpretq_u16_s16(acc[0]), 7), g8 = vshrn_n_u16(vreinterpretq_u16_s16(acc[1]), 7), b8 = vshrn_n_u16(vreinterpretq_u16_s16(acc[2]), 7);
    uint8x8_t a8 = k->fog ? vld1_u8(fogb) : vget_low_u8(vqtbl2q_u8(tb, vld1q_u8(vn_idx_nrm[3])));
    uint8x8x2_t rg = vzip_u8(r8, g8), ba = vzip_u8(b8, a8);
    uint16x4x2_t w01 = vzip_u16(vreinterpret_u16_u8(rg.val[0]), vreinterpret_u16_u8(ba.val[0]));
    uint16x4x2_t w45 = vzip_u16(vreinterpret_u16_u8(rg.val[1]), vreinterpret_u16_u8(ba.val[1]));
    uint32x2_t cw[4] = { vreinterpret_u32_u16(w01.val[0]), vreinterpret_u32_u16(w01.val[1]), vreinterpret_u32_u16(w45.val[0]), vreinterpret_u32_u16(w45.val[1]) };
    int16x8x2_t stz;
    if (k->texgen)
        stz = vzipq_s16(sT, tT);
    #define VN_PUT(i)  do { if ((i) < m) { \
            vn_st32(out[i].col, vget_lane_u32(cw[(i) >> 1], (i) & 1)); \
            if (k->texgen) vn_st32(&out[i].s, vgetq_lane_u32(vreinterpretq_u32_s16(stz.val[(i) >> 2]), (i) & 3)); } } while (0)
    VN_PUT(0); VN_PUT(1); VN_PUT(2); VN_PUT(3); VN_PUT(4); VN_PUT(5); VN_PUT(6); VN_PUT(7);
    #undef VN_PUT
}

VN_I void vn_run(const struct vn *kp, const struct f3dex_vin *in, unsigned n, struct f3dex_vtx *out, const int wrap)
{
    const struct vn *k = kp;
    const int adj0 = !k->clip && !k->lit;
    struct f3dex_vtx dummy;
    for (unsigned base = 0; base < n; base += 8) {
        unsigned m = n - base < 8 ? n - base : 8;
        uint8_t fogb[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        for (unsigned p = 0; p < m; p += 2) {
            const struct f3dex_vin *a = in + base + p;
            uint8_t *fb = k->lit ? fogb + p : NULL;
            if (p + 1 < m) {                    /* (the pair variants are separate inlined copies: two / adj / wrap are constants in each) */
                if (__builtin_expect(adj0 && base + p == 0, 0))
                    vn_pair(k, a, a + 1, out + base + p, out + base + p + 1, 1, 1, fb, wrap);
                else
                    vn_pair(k, a, a + 1, out + base + p, out + base + p + 1, 1, 0, fb, wrap);
            } else if (adj0 && base + p == 0) {
                vn_pair(k, a, a, out + base + p, &dummy, 0, 1, fb, wrap);
            } else {
                vn_pair(k, a, a, out + base + p, &dummy, 0, 0, fb, wrap);
            }
        }
        if (k->lit) {
            if (m == 8) {
                vn_lit8(k, in + base, out + base, 8, fogb);
            } else {
                struct f3dex_vin tmp[8];
                memset(tmp, 0, sizeof tmp);
                memcpy(tmp, in + base, m * sizeof tmp[0]);
                vn_lit8(k, tmp, out + base, m, fogb);
            }
        }
    }
}

/* the matrices that can overflow 48 bits: the same code with the wrapping low-word sums (cold) */
static __attribute__((noinline)) void vn_run_wrap(const struct vn *k, const struct f3dex_vin *in, unsigned n, struct f3dex_vtx *out) { vn_run(k, in, n, out, 1); }

int f3dex_vtx_xform(const struct f3dex_vctx *c, const struct f3dex_vin *in, unsigned n, struct f3dex_vtx *out)
{
    if (n == 0 || n > 32)
        return 1;
    if (c->geom & F3DEX_G_LIGHTING) {
        if (c->nrec > F3DEX_MAX_REC || ((c->geom & F3DEX_G_TEXGEN) && c->nrec == 0))
            return 1;
    }
    struct vn k;
    vn_init(&k, c);
    if (__builtin_expect(k.wrap, 0))
        vn_run_wrap(&k, in, n, out);
    else
        vn_run(&k, in, n, out, 0);
    return 0;
}

#elif !defined(VN_RECIP_ONLY)
int f3dex_vtx_xform(const struct f3dex_vctx *c, const struct f3dex_vin *in, unsigned n, struct f3dex_vtx *out) { return f3dex_vtx_xform_scalar(c, in, n, out); }
#endif      /* AArch64 */

#endif
