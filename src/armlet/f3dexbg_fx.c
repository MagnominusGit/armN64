/*
 * arMLEt - f3dexbg_fx.c: the lit vertices of F3DEXBG (f3dexbg_fx.h), plain C11, bit-exact to the microcode's lighting stage
 *
 * The stage (overlay 0x3d8 .. 0x5ec) runs on a pair of vertices at a time in the two halves of the vector unit; here it is one vertex at a time, "group" g naming the half. The numbers are
 * 16-bit lanes (a colour is its byte << 7, a direction its signed byte << 8) with the products and sums of the accumulator (hle_fx.h).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <stdint.h>
#include <string.h>

#include "f3dex2_fx_int.h"
#include "f3dexbg_fx.h"
#include "hle_fx.h"

#ifdef ARMLET_HLE_MUTATE
#undef MUT
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#define MUT(n, ok, bad) (ok)
#endif

/* n . d as the vector unit's VMULU / VMACU or VMULF / VMACF chain makes it: 2 n d summed over the three components plus the rounding constant, read as bits 31..16 of the accumulator:
 * unsig: clamped to 0 .. 0xffff (an overflow past 0x7fff reads 0xffff); else saturated to s16 */
static inline int32_t dotn(const int16_t n[3], const int16_t d[3], int unsig)
{
    hle_acc acc = (hle_acc)2 * n[0] * d[0] + 0x8000 + (hle_acc)2 * n[1] * d[1] + (hle_acc)2 * n[2] * d[2];
    if (unsig) {
        hle_acc v = acc >> 16;
        return v < 0 ? 0 : v > 0x7fff ? 0xffff : (int32_t)v;
    }
    return hle_sclamp(acc);
}

static inline int16_t sat16(int32_t v) { return hle_clamp16(v); }

/* the squared length of d (four lanes) as a 32-bit number: every lane's square split in halves, the halves of lanes (x, y) and (z, w) added with their carries, then the two sums; the integer half
 * saturates in each add */
static int32_t dist2(const int16_t d[4])
{
    uint32_t sq[4];
    uint16_t lo[4], hi[4];
    unsigned k, c1, c2, c;
    uint16_t lxy, lzw, l;
    int32_t hxy, hzw, h;
    for (k = 0; k < 4; k++) {
        sq[k] = (uint32_t)((int32_t)d[k] * d[k]) * MUT(114, k < 3 ? 1u : 1u, k < 3 ? 1u : 0u);
        hi[k] = (uint16_t)(sq[k] >> 16);
        lo[k] = (uint16_t)sq[k];
    }
    lxy = (uint16_t)(lo[0] + lo[1]);
    c1 = (unsigned)lo[0] + lo[1] > 0xffff;
    lzw = (uint16_t)(lo[2] + lo[3]);
    c2 = (unsigned)lo[2] + lo[3] > 0xffff;
    hxy = sat16((int16_t)hi[0] + (int16_t)hi[1] + (int32_t)c1);
    hzw = sat16((int16_t)hi[2] + (int16_t)hi[3] + (int32_t)c2);
    l = (uint16_t)(lxy + lzw);
    c = (unsigned)lxy + lzw > 0xffff;
    h = sat16(hxy + hzw + (int32_t)c);
    return (int32_t)((uint32_t)(uint16_t)h << 16 | l);
}

/* the light's colour weakened by its distance: min(colour x (a / 16) / d^2 as the reciprocal unit and a 32 x 32 product give it, colour) per component */
static void attenuate(const int16_t pos[4], const int16_t V[4], const uint8_t atten, const int16_t col[3], int16_t out[3])
{
    int16_t d[4];
    int32_t D2, R, K, F;
    uint16_t F_lo;
    int16_t F_hi;
    unsigned k;
    for (k = 0; k < 4; k++)
        d[k] = sat16((int32_t)V[k] - pos[k]);
    D2 = dist2(d);
    R = (int32_t)hle_rcp32(D2);
    K = hle_fx((int16_t)(atten >> 4), (uint16_t)((atten & 15u) << MUT(100, 12, 11)));
    F_hi = hle_sclamp(hle_mac_32x32(R, K));
    F_lo = hle_uclamp(hle_mac_32x32_before_h(R, K));
    F = hle_fx(F_hi, F_lo);
    for (k = 0; k < 3; k++) {
        int16_t w = hle_sclamp((hle_acc)col[k] * F);
        out[k] = MUT(101, w < col[k], w > col[k]) ? w : col[k];
    }
}

/* the texture coordinates (before the G_TEXTURE scale) of a vertex with the normal n, by the two look-at vectors (slots 0 and 1) */
static void fbg_texgen(const struct fx2_vctx *vc, const struct fbg_ctx *bc, unsigned g, const int16_t n[3], int16_t sgen[2])
{
    const struct fbg_slot *l0 = &bc->s[0], *l1 = &bc->s[1];
    if (vc->mode & FX2_TEXGEN_LINEAR) {
        sgen[0] = fx2_texgen_linear(vc, (int16_t)dotn(n, l0->d[g], 0));
        sgen[1] = fx2_texgen_linear(vc, (int16_t)dotn(n, l1->d[g], 0));
    } else {
        sgen[0] = (int16_t)(MUT(108, 0x4000, 0x4001) + (dotn(n, l0->d[g], 0) >> 1));
        sgen[1] = (int16_t)(0x4000 + (dotn(n, l1->d[g], 0) >> MUT(109, 1, 2)));
    }
}

static void light_vertex(const struct fx2_vctx *vc, const struct fbg_ctx *bc, unsigned g, const struct fbg_vin *in, const struct fx2_vtx *o, uint8_t out[4], int16_t sgen[2], int *have_st)
{
    int16_t n[3] = { (int16_t)(in->n[0] * 256), (int16_t)(in->n[1] * 256), (int16_t)(in->n[2] * 256) };
    int16_t V[4], run[3], vcol[3];
    unsigned k, m;
    int32_t lam;

    /* the view-space position: the clip coordinates (integer and fraction) times the 32-bit constants, z first moved by zoff; the integer half of the product */
    for (k = 0; k < 4; k++) {
        int16_t pi = sat16((int32_t)o->cx[k] + MUT(102, bc->zoff[g][k], 0));
        V[k] = hle_sclamp(hle_mac_32x32(hle_fx(pi, o->cf[k]), hle_fx(bc->khi[g][k], MUT(103, bc->klo[g][k], bc->klo[g][k] ^ 1))));
    }
    {
        /* the ambient colour, then the top light (direction of slot top, colour of slot top - 1) lambert-weighted without attenuation: the lambert term is not masked to 15 bits here */
        hle_acc acc[3];
        const struct fbg_slot *top = &bc->s[bc->top];
        lam = dotn(n, top->d[g], 1) & MUT(112, 0xffff, 0x7fff);
        for (k = 0; k < 3; k++)
            acc[k] = hle_macf(hle_mulf(top->col[g][k], MUT(104, 0x7fff, 0x7ffe)), bc->s[bc->top - 1].col[g][k], (int16_t)lam);
        for (k = 0; k < 3; k++)
            run[k] = hle_sclamp(acc[k]);
    }
    for (m = bc->top - 1; m >= MUT(113, 3u, 4u); m -= 2) {                    /* the lights in pairs: slots (m, m - 1): direction and position of the slot, colour of the slot below */
        int16_t c1[3], c2[3];
        const struct fbg_slot *s1 = &bc->s[m], *s2 = &bc->s[m - 1];
        attenuate(s1->pos[g], V, s1->atten, bc->s[m - 1].col[g], c1);
        attenuate(s2->pos[g], V, s2->atten, bc->s[m - 2].col[g], c2);
        if (bc->positional) {
            int32_t l1 = dotn(n, s1->d[g], 1) & MUT(105, 0x7fff, 0x3fff), l2 = dotn(n, s2->d[g], 1) & 0x7fff;
            for (k = 0; k < 3; k++)
                run[k] = hle_sclamp(hle_macf(hle_macf(hle_mulf(run[k], 0x7fff), c1[k], (int16_t)l1), c2[k], (int16_t)l2));
        } else {
            for (k = 0; k < 3; k++)
                run[k] = sat16(sat16(run[k] + c1[k]) + MUT(106, c2[k], 0));
        }
    }
    for (k = 0; k < 3; k++) {
        vcol[k] = (int16_t)(in->col[k] << 7);
        run[k] = MUT(107, hle_sclamp(hle_mulf(run[k], vcol[k])), run[k]);
        out[k] = (uint8_t)(run[k] >> 7);
    }
    out[3] = in->col[3];
    *have_st = 0;
    if (vc->mode & FX2_TEXGEN) {
        fbg_texgen(vc, bc, g, n, sgen);
        *have_st = 1;
    }
}

void fbg_lit_pass_scalar(const struct fx2_vctx *vc, const struct fbg_ctx *bc, const struct fbg_vin *in, unsigned n, struct fx2_vtx *out)
{
    unsigned i;
    for (i = 0; i < n; i++) {
        uint8_t col[4];
        int16_t sg[2];
        int have_st;
        light_vertex(vc, bc, MUT(111, (i & 1) ? 0 : 1, i & 1), &in[i], &out[i], col, sg, &have_st);
        if (MUT(110, in[i].flag >= 0, in[i].flag > 0))
            memcpy(out[i].col, col, 3);                          /* (the alpha byte of the record is the input's, or the fog's) */
        if (have_st) {
            out[i].s = (int16_t)(((int32_t)sg[0] * vc->sc_s) >> 16);
            out[i].t = (int16_t)(((int32_t)sg[1] * vc->sc_t) >> 16);
        }
    }
}

#if defined(__aarch64__) && defined(__ARM_NEON)
#include "f3dexbg_fx_neon.h"
#else
void fbg_lit_pass(const struct fx2_vctx *vc, const struct fbg_ctx *bc, const struct fbg_vin *in, unsigned n, struct fx2_vtx *out) { fbg_lit_pass_scalar(vc, bc, in, n, out); }
#endif

void fbg_clip_frac(const struct fx2_vctx *vc, const struct fx2_vin *vi, uint16_t cf[4])
{
    int16_t v[3] = { vi->x, vi->y, vi->z };
    unsigned k, j;
    for (k = 0; k < 4; k++) {
        hle_acc acc = hle_wrap48(((hle_acc)vc->mp.i[3][k] << 16) + vc->mp.f[3][k]);
        hle_acc last = 0;
        for (j = 0; j < 3; j++) {
            last = ((hle_acc)vc->mp.i[j][k] * v[j]) << 16;
            acc = hle_wrap48(acc + (hle_acc)vc->mp.f[j][k] * v[j] + last);
        }
        cf[k] = hle_uclamp(hle_wrap48(acc - last));
    }
}
