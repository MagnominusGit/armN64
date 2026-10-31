/*
 * arMLEt - f3dexbg_fx_neon.h: the NEON form of fbg_lit_pass (aarch64, generic armv8-a); #included in f3dexbg_fx.c after the scalar pass, whose helpers (MUT, dotn, fbg_texgen) it uses
 *
 * Bit-exact with fbg_lit_pass_scalar (tools/hle/test_f3dexbg.c compares them on random and extreme inputs). One vertex per 32-bit lane, four vertices at a time, the light slots shared (the
 * lanes alternate between the two lane groups of the slots, as the vertex index parity does); the lights are the outer loop, the blocks of a chunk of up to 16 vertices the inner one. Every step
 * is exact for all inputs; nothing falls back to the scalar function:
 *  - view-space position: (clip integer : fraction) x (khi : klo) >> 32, a 64-bit product (vmull_s32), clamped to s16 by the narrow.
 *  - distance: d = V - pos saturates (vqsub); the squares are < 2^30, so the split-and-carry sum of the scalar dist2 is P = sq0 + sq1 (and Q = sq2 + sq3) as u32 (<= 2^31) with the integer half
 *    min(P >> 16, 0x7fff), the fraction P & 0xffff; D2 = ((hxy + hzw + carry of the fractions) clamped to 0x7fff) << 16 | (lxy + lzw) & 0xffff.
 *  - reciprocal: D2 < 2^31 (never negative), so hle_rcp32 is clz, the 9 bits below the top one, the ROM (4 halfword lane loads) and a variable shift; D2 == 0 gives 0x7fffffff.
 *  - falloff: K = atten << 12 (< 2^20) and R < 2^31: R * K is a 64-bit product (vmull_u32); F_hi = min(P >> 32, 0x7fff); the fraction half read before the integer term,
 *    (R * K >> 16) - (Rh * Kh << 16), is below 2^31 for every R and K, so hle_uclamp never clamps and F_lo = (R * K >> 16) & 0xffff. col * F >> 16 = col * F_hi + (col * F_lo >> 16)
 *    (both products fit s32: |col| <= 2^15, F_lo < 2^16, F_hi <= 0x7fff), saturated by the narrow.
 *  - n . d: the normals are signed bytes << 8, so (2 n d summed + 0x8000) >> 16 is (sum nb d + 64) >> 7 with |sum| < 3 * 2^22: no overflow for any 16-bit d.
 *  - macf chains: 2 (a b + c d + e f) + 0x8000 >> 16 would overflow 32 bits (3 products of up to 2^30); fbg_mac3 halves the first pair with a halving add (exact) and carries its dropped bit.
 *  - vertex colour: vqrdmulh is VMULF.
 * Texgen (a fraction of a per cent of the pass) stays scalar, per vertex.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEXBG_FX_NEON_H
#define ARMLET_F3DEXBG_FX_NEON_H

#include <arm_neon.h>

/* test builds only (-DFBG_NEON_MUTATE=n): a deliberate error in the NEON form only, to show that tools/hle/test_f3dexbg.c catches it */
#ifdef FBG_NEON_MUTATE
#define NMUT(n, ok, bad) (FBG_NEON_MUTATE == (n) ? (bad) : (ok))
#else
#define NMUT(n, ok, bad) (ok)
#endif

#define FN_I static inline __attribute__((always_inline))

#define FBG_CHUNK 16                                /* vertices per chunk: 4 blocks of 4 */
#define FBG_GE MUT(111, 1u, 0u)                     /* the lane group of the even / odd lanes (vertex index parity) */
#define FBG_GO MUT(111, 0u, 1u)

/* a 4-lane vector of a per-group value: the even lanes take group GE's, the odd ones GO's */
FN_I int16x4_t fn_p16(int16_t e, int16_t o) { int16_t a[4] = { e, o, e, o }; return vld1_s16(a); }
FN_I int32x4_t fn_p32(int32_t e, int32_t o) { int32_t a[4] = { e, o, e, o }; return vld1q_s32(a); }

/* ((p0 + p1 + p2 + 0x4000) >> 15) saturated to s16 = the VMULF / VMACF chain 2 p0 + 0x8000 + 2 p1 + 2 p2 read as bits 31..16 */
FN_I int16x4_t fn_mac3(int32x4_t p0, int32x4_t p1, int32x4_t p2)
{
    int32x4_t t = vhaddq_s32(p0, p1);                                               /* floor((p0 + p1) / 2) */
    int32x4_t r = vandq_s32(veorq_s32(p0, p1), vdupq_n_s32(NMUT(6, 1, 0)));         /* the dropped bit */
    int32x4_t w = vshrq_n_s32(vaddq_s32(vaddq_s32(p2, vdupq_n_s32(0x4000)), r), 1);
    return vqmovn_s32(vshrq_n_s32(vaddq_s32(t, w), 14));
}

/* the constants of one light: direction, position, colour per lane (lane group), the attenuation numerator */
struct fn_light {
    int16x8_t pxy, pzw;
    int16x4_t dir[3];
    int16x4_t col[3];
    int32x4_t col32[3];
    uint32_t K;
};

/* slot a: direction, position, attenuation; slot b: colour */
FN_I void fn_light(struct fn_light *l, const struct fbg_slot *a, const struct fbg_slot *b)
{
    unsigned k;
    l->pxy = vcombine_s16(NMUT(12, fn_p16(a->pos[FBG_GE][0], a->pos[FBG_GO][0]), fn_p16(a->pos[FBG_GO][0], a->pos[FBG_GE][0])), fn_p16(a->pos[FBG_GE][1], a->pos[FBG_GO][1]));
    l->pzw = vcombine_s16(fn_p16(a->pos[FBG_GE][2], a->pos[FBG_GO][2]), fn_p16(a->pos[FBG_GE][3], a->pos[FBG_GO][3]));
    for (k = 0; k < 3; k++) {
        l->dir[k] = fn_p16(a->d[FBG_GE][k], a->d[FBG_GO][k]);
        l->col[k] = fn_p16(b->col[FBG_GE][k], b->col[FBG_GO][k]);
        l->col32[k] = vmovl_s16(l->col[k]);
    }
    l->K = hle_fx((int16_t)(a->atten >> 4), (uint16_t)((a->atten & 15u) << MUT(100, 12, 11)));
}

/* the lambert term: n . d read as the unsigned VMULU chain (clamped to 0 .. 0xffff) before the mask */
FN_I int32x4_t fn_lam(int16x8_t nxy, int16x4_t nz, const int16x4_t dir[3])
{
    int32x4_t s = vmull_s16(vget_low_s16(nxy), dir[0]);
    s = vmlal_s16(s, vget_high_s16(nxy), dir[1]);
    s = vmlal_s16(s, nz, dir[2]);
    int32x4_t v = vshrq_n_s32(vaddq_s32(s, vdupq_n_s32(NMUT(5, 64, 63))), 7);
    return vbslq_s32(vcgtq_s32(v, vdupq_n_s32(0x7fff)), vdupq_n_s32(NMUT(8, 0xffff, 0x7fff)), vmaxq_s32(v, vdupq_n_s32(0)));
}

/* min(colour, colour x (atten / 16) / d^2) per component, for four vertices */
FN_I void fn_atten(const struct fn_light *l, int16x8_t vxy, int16x8_t vzw, int16x4_t out[3])
{
    int16x8_t dxy = vqsubq_s16(vxy, l->pxy), dzw = vqsubq_s16(vzw, l->pzw);
    uint32x4_t s0 = vreinterpretq_u32_s32(vmull_s16(vget_low_s16(dxy), vget_low_s16(dxy)));
    uint32x4_t s1 = vreinterpretq_u32_s32(vmull_high_s16(dxy, dxy));
    uint32x4_t s2 = vreinterpretq_u32_s32(vmull_s16(vget_low_s16(dzw), vget_low_s16(dzw)));
    uint32x4_t s3 = vreinterpretq_u32_s32(vmull_high_s16(dzw, dzw));
    uint32x4_t M = vdupq_n_u32(0xffff), H = vdupq_n_u32(NMUT(4, 0x7fff, 0xffff));
    uint32x4_t P, Q, hxy, hzw, ls, h, D2;
    uint32x4_t sh, nrm, idx, rom, res, zero, R;
    uint32_t ix[4];
    uint16x4_t rv = vdup_n_u16(0);
    uint64x2_t plo, phi;
    uint32x4_t Fh, Fl;
    unsigned k;

    s3 = MUT(114, s3, vdupq_n_u32(0));
    P = vaddq_u32(s0, s1);
    Q = vaddq_u32(s2, s3);
    hxy = vshrq_n_u32(P, 16);                                                       /* (the scalar form saturates the halves first: the clamp of the sum covers it) */
    hzw = vshrq_n_u32(Q, 16);
    ls = vaddq_u32(vandq_u32(P, M), vandq_u32(Q, M));
    h = vminq_u32(vaddq_u32(vaddq_u32(hxy, hzw), vshrq_n_u32(ls, NMUT(3, 16, 31))), H);
    D2 = vorrq_u32(vshlq_n_u32(h, 16), vandq_u32(ls, M));

    /* hle_rcp32 of a non-negative input */
    sh = vclzq_u32(D2);
    nrm = vshlq_u32(D2, vreinterpretq_s32_u32(sh));                              /* (a count of 32 gives 0) */
    idx = vandq_u32(vshrq_n_u32(nrm, 22), vdupq_n_u32(0x1ff));
    vst1q_u32(ix, idx);
    rv = vld1_lane_u16(&rsp_reciprocal_rom[ix[0]], rv, 0);
    rv = vld1_lane_u16(&rsp_reciprocal_rom[ix[1]], rv, 1);
    rv = vld1_lane_u16(&rsp_reciprocal_rom[ix[2]], rv, 2);
    rv = vld1_lane_u16(&rsp_reciprocal_rom[ix[3]], rv, 3);
    rom = vshlq_n_u32(vorrq_u32(vmovl_u16(rv), vdupq_n_u32(0x10000)), 14);
    res = vshlq_u32(rom, vreinterpretq_s32_u32(vsubq_u32(sh, vdupq_n_u32(NMUT(1, 31, 30)))));      /* >> (31 - shift) */
    zero = vceqzq_u32(D2);
    R = vbslq_u32(zero, vdupq_n_u32(NMUT(9, 0x7fffffffu, 0u)), res);

    plo = vmull_u32(vget_low_u32(R), vdup_n_u32(l->K));
    phi = vmull_high_u32(R, vdupq_n_u32(l->K));
    Fh = vcombine_u32(vshrn_n_u64(plo, 32), vshrn_n_u64(phi, 32));
    Fh = NMUT(2, vminq_u32(Fh, vdupq_n_u32(0x7fff)), Fh);
    Fl = vandq_u32(vcombine_u32(vshrn_n_u64(plo, 16), vshrn_n_u64(phi, 16)), vdupq_n_u32(NMUT(7, 0xffff, 0x7fff)));
    for (k = 0; k < 3; k++) {
        int32x4_t a = vmulq_s32(l->col32[k], vreinterpretq_s32_u32(Fh));
        int32x4_t b = vmulq_s32(l->col32[k], vreinterpretq_s32_u32(Fl));
        int16x4_t w = vqmovn_s32(vaddq_s32(a, NMUT(10, vshrq_n_s32(b, 16), vdupq_n_s32(0))));
        out[k] = MUT(101, vmin_s16(w, l->col[k]), vmax_s16(w, l->col[k]));
    }
}

/* the vertices of a block of four: view-space position, normal, colour, the running colour */
struct fn_blk {
    int16x8_t vxy, vzw;
    int16x8_t nxy;                                  /* normal bytes x (lanes 0 .. 3) and y (4 .. 7) */
    int16x4_t nz;
    int16x4_t vcol[3];
    int16x4_t run[3];
};

static const uint8_t fn_idx_nxy[16] = { 255, 0, 255, 8, 255, 16, 255, 24, 255, 1, 255, 9, 255, 17, 255, 25 };
static const uint8_t fn_idx_nz[16] = { 255, 2, 255, 10, 255, 18, 255, 26, 255, 2, 255, 10, 255, 18, 255, 26 };
static const uint8_t fn_idx_c01[16] = { 4, 255, 12, 255, 20, 255, 28, 255, 5, 255, 13, 255, 21, 255, 29, 255 };
static const uint8_t fn_idx_c2[16] = { 6, 255, 14, 255, 22, 255, 30, 255, 6, 255, 14, 255, 22, 255, 30, 255 };

static void fn_chunk(const struct fx2_vctx *vc, const struct fbg_ctx *bc, const struct fbg_vin *in, unsigned cnt, struct fx2_vtx *out)
{
    struct fn_blk B[FBG_CHUNK / 4];
    struct fn_light L1, L2;
    int32x4_t kv[4];
    int16x4_t zo[4];
    unsigned nb = NMUT(13, (cnt + 3) / 4, cnt > 4 ? cnt / 4 : 1), b, j, k, m;
    int16x4_t c7fff = vdup_n_s16(0x7fff);
    const struct fbg_slot *top = &bc->s[bc->top];

    for (k = 0; k < 4; k++) {
        kv[k] = fn_p32(hle_fx(bc->khi[FBG_GE][k], MUT(103, bc->klo[FBG_GE][k], bc->klo[FBG_GE][k] ^ 1)), hle_fx(bc->khi[FBG_GO][k], MUT(103, bc->klo[FBG_GO][k], bc->klo[FBG_GO][k] ^ 1)));
        zo[k] = fn_p16(MUT(102, bc->zoff[FBG_GE][k], 0), MUT(102, bc->zoff[FBG_GO][k], 0));
    }
    for (b = 0; b < nb; b++) {
        struct fn_blk *B_ = &B[b];
        const struct fbg_vin *ip = in + 4 * b;
        struct fbg_vin tin[4];
        const struct fx2_vtx *rp[4];
        int16x4_t cx[4], cf[4];
        uint8x16x2_t tb;
        int16x4_t V[4];
        int32x4_t lam;
        int16x4_t lam16;

        if (4 * b + 4 > cnt) {
            memset(tin, 0, sizeof tin);
            memcpy(tin, ip, (cnt - 4 * b) * sizeof tin[0]);
            ip = tin;
        }
        for (j = 0; j < 4; j++)
            rp[j] = 4 * b + j < cnt ? out + 4 * b + j : out;
        {   /* the clip integers / fractions of the four vertices, transposed to one vector per component */
            int16x4_t r0 = vld1_s16(rp[0]->cx), r1 = vld1_s16(rp[1]->cx), r2 = vld1_s16(rp[2]->cx), r3 = vld1_s16(rp[3]->cx);
            int16x4x2_t a = vtrn_s16(r0, r1), c = vtrn_s16(r2, r3);
            int32x2x2_t e = vtrn_s32(vreinterpret_s32_s16(a.val[0]), vreinterpret_s32_s16(c.val[0]));
            int32x2x2_t f = vtrn_s32(vreinterpret_s32_s16(a.val[1]), vreinterpret_s32_s16(c.val[1]));
            cx[0] = vreinterpret_s16_s32(e.val[0]);
            cx[2] = vreinterpret_s16_s32(e.val[1]);
            cx[1] = vreinterpret_s16_s32(f.val[0]);
            cx[3] = vreinterpret_s16_s32(f.val[1]);
            r0 = vld1_s16((const int16_t *)rp[0]->cf);
            r1 = vld1_s16((const int16_t *)rp[1]->cf);
            r2 = vld1_s16((const int16_t *)rp[2]->cf);
            r3 = vld1_s16((const int16_t *)rp[3]->cf);
            a = vtrn_s16(r0, r1);
            c = vtrn_s16(r2, r3);
            e = vtrn_s32(vreinterpret_s32_s16(a.val[0]), vreinterpret_s32_s16(c.val[0]));
            f = vtrn_s32(vreinterpret_s32_s16(a.val[1]), vreinterpret_s32_s16(c.val[1]));
            cf[0] = vreinterpret_s16_s32(e.val[0]);
            cf[2] = vreinterpret_s16_s32(e.val[1]);
            cf[1] = vreinterpret_s16_s32(f.val[0]);
            cf[3] = vreinterpret_s16_s32(f.val[1]);
        }
        for (k = 0; k < 4; k++) {                   /* V = ((sat16(cx + zoff) : cf) x (khi : klo)) >> 32, clamped */
            int32x4_t A = vorrq_s32(vshll_n_s16(vqadd_s16(cx[k], zo[k]), 16), vreinterpretq_s32_u32(vmovl_u16(vreinterpret_u16_s16(cf[k]))));
            int64x2_t lo = vmull_s32(vget_low_s32(A), vget_low_s32(kv[k])), hi = vmull_high_s32(A, kv[k]);
            V[k] = vqmovn_s32(vcombine_s32(vshrn_n_s64(lo, NMUT(11, 32, 31)), vshrn_n_s64(hi, 32)));
        }
        B_->vxy = vcombine_s16(V[0], V[1]);
        B_->vzw = vcombine_s16(V[2], V[3]);
        tb.val[0] = vcombine_u8(vld1_u8((const uint8_t *)ip), vld1_u8((const uint8_t *)(ip + 1)));
        tb.val[1] = vcombine_u8(vld1_u8((const uint8_t *)(ip + 2)), vld1_u8((const uint8_t *)(ip + 3)));
        B_->nxy = vshrq_n_s16(vreinterpretq_s16_u8(vqtbl2q_u8(tb, vld1q_u8(fn_idx_nxy))), 8);
        B_->nz = vshr_n_s16(vget_low_s16(vreinterpretq_s16_u8(vqtbl2q_u8(tb, vld1q_u8(fn_idx_nz)))), 8);
        {
            int16x8_t c01 = vshlq_n_s16(vreinterpretq_s16_u8(vqtbl2q_u8(tb, vld1q_u8(fn_idx_c01))), 7);
            int16x8_t c2 = vshlq_n_s16(vreinterpretq_s16_u8(vqtbl2q_u8(tb, vld1q_u8(fn_idx_c2))), 7);
            B_->vcol[0] = vget_low_s16(c01);
            B_->vcol[1] = vget_high_s16(c01);
            B_->vcol[2] = vget_low_s16(c2);
        }
        /* the ambient colour, then the top light (direction of slot top, colour of slot top - 1) lambert-weighted without attenuation */
        {
            int16x4_t dir[3];
            for (k = 0; k < 3; k++)
                dir[k] = fn_p16(top->d[FBG_GE][k], top->d[FBG_GO][k]);
            lam = vandq_s32(fn_lam(B_->nxy, B_->nz, dir), vdupq_n_s32(MUT(112, 0xffff, 0x7fff)));
            lam16 = vmovn_s32(lam);
            for (k = 0; k < 3; k++) {
                int16x4_t tc = fn_p16(top->col[FBG_GE][k], top->col[FBG_GO][k]), c1 = fn_p16(bc->s[bc->top - 1].col[FBG_GE][k], bc->s[bc->top - 1].col[FBG_GO][k]);
                B_->run[k] = fn_mac3(vmull_s16(tc, vdup_n_s16(MUT(104, 0x7fff, 0x7ffe))), vmull_s16(c1, lam16), vdupq_n_s32(0));
            }
        }
    }
    for (m = bc->top - 1; m >= MUT(113, 3u, 4u); m -= 2) {
        fn_light(&L1, &bc->s[m], &bc->s[m - 1]);
        fn_light(&L2, &bc->s[m - 1], &bc->s[m - 2]);
        for (b = 0; b < nb; b++) {
            struct fn_blk *B_ = &B[b];
            int16x4_t c1[3], c2[3];
            fn_atten(&L1, B_->vxy, B_->vzw, c1);
            fn_atten(&L2, B_->vxy, B_->vzw, c2);
            if (bc->positional) {
                int16x4_t l1 = vmovn_s32(vandq_s32(fn_lam(B_->nxy, B_->nz, L1.dir), vdupq_n_s32(MUT(105, 0x7fff, 0x3fff))));
                int16x4_t l2 = vmovn_s32(vandq_s32(fn_lam(B_->nxy, B_->nz, L2.dir), vdupq_n_s32(NMUT(16, 0x7fff, 0x3fff))));
                for (k = 0; k < 3; k++)
                    B_->run[k] = fn_mac3(vmull_s16(B_->run[k], c7fff), vmull_s16(c1[k], l1), vmull_s16(c2[k], l2));
            } else {
                for (k = 0; k < 3; k++)
                    B_->run[k] = vqadd_s16(NMUT(14, vqadd_s16(B_->run[k], c1[k]), vadd_s16(B_->run[k], c1[k])), MUT(106, c2[k], vdup_n_s16(0)));
            }
        }
    }
    for (b = 0; b < nb; b++) {
        struct fn_blk *B_ = &B[b];
        int16x4_t r[3];
        uint8_t rg[8], bb[8];
        for (k = 0; k < 3; k++)
            r[k] = MUT(107, vqrdmulh_s16(B_->run[k], B_->vcol[k]), B_->run[k]);
        vst1_u8(rg, vmovn_u16(vreinterpretq_u16_s16(vshrq_n_s16(vcombine_s16(r[0], r[1]), 7))));
        vst1_u8(bb, vmovn_u16(vreinterpretq_u16_s16(vshrq_n_s16(vcombine_s16(r[2], r[2]), 7))));
        for (j = 0; j < 4 && 4 * b + j < cnt; j++) {
            unsigned i = 4 * b + j;
            if (MUT(110, in[i].flag >= 0, in[i].flag > 0)) {
                out[i].col[0] = rg[j];
                out[i].col[1] = rg[4 + j];
                out[i].col[2] = bb[j];
            }
            if (vc->mode & FX2_TEXGEN) {
                int16_t n[3] = { (int16_t)(in[i].n[0] * 256), (int16_t)(in[i].n[1] * 256), (int16_t)(in[i].n[2] * 256) }, sg[2];
                fbg_texgen(vc, bc, MUT(111, (i & 1) ? 0 : 1, i & 1), n, sg);
                out[i].s = (int16_t)(((int32_t)sg[0] * vc->sc_s) >> 16);
                out[i].t = (int16_t)(((int32_t)sg[1] * vc->sc_t) >> 16);
            }
        }
    }
}

void fbg_lit_pass(const struct fx2_vctx *vc, const struct fbg_ctx *bc, const struct fbg_vin *in, unsigned n, struct fx2_vtx *out)
{
    unsigned base;
    for (base = 0; base < n; base += FBG_CHUNK)
        fn_chunk(vc, bc, in + base, n - base < FBG_CHUNK ? n - base : FBG_CHUNK, out + base);
}

#endif
