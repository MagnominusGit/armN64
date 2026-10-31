/*
 * arMLEt - f3dex2_fx.c: the scalar reference of the F3DEX2 geometry functions (f3dex2_fx.h)
 *
 * Plain C11 on structs, bit-exact to what the microcode's vector code computes; tools/hle/test_fx2.c checks it against the interpreter running the real microcode.
 * The arithmetic is hle_fx.h's. Where the code is a transcription of the microcode's chains the comments name the chain, not the instruction (docs/ucode-f3dex2.md).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <stdint.h>
#include <string.h>

#include "f3dex2_fx.h"
#include "f3dex2_fx_int.h"
#include "hle_fx.h"

#ifdef ARMLET_HLE_MUTATE
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#define MUT(n, ok, bad) (ok)
#endif

/* ---- (integer, fraction) pairs ------------------------------------------------------------------------------------------------------------------------------------------------ */

/* [VMUDL VMADM VMADN | VMADH]: the sum of the partial products of a 32 x 32 multiply, shifted right by 16. t is what the accumulator holds before the product of the two integer halves
 * (where the chain reads the fraction), full is the whole sum (where it reads the integer half). */
static inline void mm(struct fx2_pair a, struct fx2_pair b, hle_acc *t, hle_acc *full)
{
    *t = hle_wrap48(((hle_acc)a.f * b.f >> 16) + (hle_acc)a.i * b.f + (hle_acc)a.f * b.i);
    *full = hle_wrap48(*t + (((hle_acc)a.i * b.i) << 16));
}
static inline struct fx2_pair rd(hle_acc t, hle_acc full)
{
    struct fx2_pair r = { hle_sclamp(full), hle_uclamp(t) };
    return r;
}
static inline hle_acc pv(struct fx2_pair a) { return (hle_acc)a.i * 65536 + a.f; }
static inline struct fx2_pair fxs(int32_t v)
{
    struct fx2_pair r = { (int16_t)(v >> 16), (uint16_t)v };
    return r;
}
/* VSUBC / VSUB on pairs: the integer half saturates */
static inline struct fx2_pair fsub(struct fx2_pair a, struct fx2_pair b)
{
    struct fx2_pair r = { hle_clamp16((int32_t)a.i - b.i - (a.f < b.f)), (uint16_t)(a.f - b.f) };
    return r;
}

/* ---- matrices ------------------------------------------------------------------------------------------------------------------------------------------------------------------ */
void fx2_mtx_mul_scalar(const struct fx2_mtx *a, const struct fx2_mtx *b, struct fx2_mtx *out)
{
    struct fx2_mtx r;
    unsigned i, j, k;
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++) {
            hle_acc acc = 0, last_h = 0;
            for (k = 0; k < 4; k++) {
                last_h = ((hle_acc)b->i[k][j] * a->i[i][k]) << 16;
                acc = hle_wrap48(acc + ((hle_acc)b->f[k][j] * a->f[i][k] >> 16) + (hle_acc)b->i[k][j] * a->f[i][k] + (hle_acc)b->f[k][j] * a->i[i][k] + last_h);
            }
            r.i[i][j] = hle_sclamp(acc);
            /* rows 0 and 1 are read with VMADN after the last term (the whole accumulator); rows 2 and 3 by the VMADN of the last term, before its integer product */
            r.f[i][j] = hle_uclamp(i < 2 ? acc : hle_wrap48(acc - last_h));
        }
    *out = r;
}

/* ---- light setup ----------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* Per direction d (signed bytes as 16-bit lanes, byte << 8): T = M d, a 16.16 number per row of M (the 48-bit accumulator read as hi:mid); the sum of the squares of the three components as a
 * 32-bit sum (hle_mac_32x32 with the pair clamps, then 32-bit adds); R = rsq(sum); T' = T R, scaled by the halfword at 0x1c6; the byte is the integer half >> 8. */
void fx2_light_setup_scalar(const struct fx2_mtx *mv, const int8_t raw[][3], unsigned nlights, int16_t scale, int8_t out[][3])
{
    unsigned m, i, j;
    for (m = 0; m <= MUT(17, nlights + 1, nlights); m++) {
        int16_t d[3];
        int32_t T[3];
        struct hle_pair sq[3], sum;
        int32_t R, Tn[3];
        for (j = 0; j < 3; j++)
            d[j] = (int16_t)(raw[m][j] * 256);
        for (i = 0; i < 3; i++) {
            hle_acc acc = 0;
            for (j = 0; j < 3; j++)
                acc = hle_wrap48(acc + (hle_acc)mv->f[i][j] * d[j] + (((hle_acc)mv->i[i][j] * d[j]) << 16));
            T[i] = (int32_t)((uint32_t)hle_acc_hi(acc) << 16 | hle_acc_md(acc));
            sq[i].i = hle_sclamp(hle_mac_32x32(T[i], T[i]));
            sq[i].f = hle_uclamp(hle_mac_32x32_before_h(T[i], T[i]));
        }
        sum = hle_add32(hle_pair32(hle_add32(hle_pair32(sq[0]), hle_pair32(sq[1]))), hle_pair32(sq[2]));
        R = (int32_t)hle_rsq32(hle_pair32(MUT(18, sum, sq[0])));
        for (i = 0; i < 3; i++) {
            hle_acc a = hle_mac_32x32(T[i], R);
            int16_t hi = hle_sclamp(a);
            uint16_t lo = hle_uclamp(hle_mac_32x32_before_h(T[i], R));
            hle_acc b = hle_wrap48((hle_acc)lo * scale + (((hle_acc)hi * scale) << 16));
            Tn[i] = hle_sclamp(b);
            out[m][i] = (int8_t)(Tn[i] >> 8);
        }
    }
}

/* ---- vertices ------------------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* The lit colour of one vertex (the lighting overlay, per lane group of the vector unit: the pair's second vertex is group 0, the first group 1). The loop takes two slots at a time from the top (the
 * ambient slot m = n + 1), pairing the direction of slot m with the colour of slot m - 1: lambert = (normal . direction) read unsigned, & 0x7fff; the running colour is rescaled by 0x7fff (VMULF) at
 * every round and the colours times their lambert terms are added (VMACF), the readout clamped to s16; a lone last slot (m = 2) is the same with one term. The colour of each light is the 8-bit value
 * << 7; the result byte is lane >> 7. The alpha lane is the vertex alpha << 7 rescaled once. With texture generation the texture coordinates are 0x4000 + (normal . direction of slot 0 / 1) / 2
 * (the directions of the look-at vectors). */
static inline int32_t dotn(const int16_t n[3], const int16_t d[3], int unsig)
{
    hle_acc acc = (hle_acc)2 * n[0] * d[0] + 0x8000 + (hle_acc)2 * n[1] * d[1] + (hle_acc)2 * n[2] * d[2];
    if (unsig) {
        hle_acc v = acc >> 16;
        return v < 0 ? 0 : v > 0x7fff ? 0xffff : (int32_t)v;
    }
    return hle_sclamp(acc);
}

/* the linear texture generation (G_TEXTURE_GEN_LINEAR): from the sum 0x4000 + d / 2 the code goes on with a polynomial. x = 0x4000 + c0 + d / 2 (the accumulator adds c0 << 16 to the sum, the
 * integer half saturates); then x2 = x x (VMULF), p = x k + x c2 (VMULF, VMACF; k = v31[7]); the result is 0x4000 + 2 x c1 + 2 x2 p, in the accumulator, saturated. F3DEX2 2.04 / 2.04H: the
 * instruction that restarts the accumulator also overwrites the register x is in, so the linear term is 2 * 0x4000 * c1. */
int16_t fx2_texgen_linear(const struct fx2_vctx *c, int16_t d)
{
    hle_acc acc0 = ((hle_acc)0x4000 << 16) + ((hle_acc)d << 15) + ((hle_acc)c->tg[0] << 16);
    int16_t x = hle_sclamp(acc0), x2, p;
    hle_acc a;
    x2 = hle_sclamp(hle_mulf(x, x));
    p = hle_sclamp(hle_macf(hle_mulf(x, c->c31_7), x, c->tg[2]));
    a = hle_macf(hle_macf(((hle_acc)0x4000 << 16), MUT(215, c->gen204, 0) ? 0x4000 : x, c->tg[1]), x2, p);
    return hle_sclamp(a);
}

void fx2_lit_vertex_scalar(const struct fx2_vctx *cx, unsigned g, const int8_t nb[3], uint8_t alpha, uint8_t out[4], int texgen, int16_t sgen[2])
{
    const struct fx2_lights *L = &cx->L;
    int16_t n[3] = { (int16_t)(nb[0] * 256), (int16_t)(nb[1] * 256), (int16_t)(nb[2] * 256) };
    int16_t cur[3];
    unsigned m = L->top, c;
    for (c = 0; c < 3; c++)
        cur[c] = L->c[g][m][c];
    for (;;) {
        hle_acc a[3];
        if (m == 2) {
            int32_t lam = dotn(n, L->d[g][m], 1) & 0x7fff;
            for (c = 0; c < 3; c++)
                a[c] = hle_macf(hle_mulf(cur[c], MUT(10, 0x7fff, 0x7f00)), L->c[g][m - 1][c], (int16_t)lam);
        } else {
            int32_t l1 = dotn(n, L->d[g][m], 1) & 0x7fff, l2 = dotn(n, L->d[g][m - 1], 1) & 0x7fff;
            for (c = 0; c < 3; c++)
                a[c] = hle_macf(hle_macf(hle_mulf(cur[c], MUT(10, 0x7fff, 0x7f00)), L->c[g][m - 1][c], (int16_t)l1), L->c[g][m - 2][c], (int16_t)l2);
        }
        for (c = 0; c < 3; c++)
            cur[c] = hle_sclamp(a[c]);
        if (m == 2 || m - 1 == 2)
            break;
        m -= 2;
    }
    for (c = 0; c < 3; c++)
        out[c] = (uint8_t)(cur[c] >> 7);
    out[3] = (uint8_t)(hle_sclamp(hle_mulf((int16_t)(alpha << 7), 0x7fff)) >> 7);
    if (texgen) {
        if (cx->mode & FX2_TEXGEN_LINEAR) {
            sgen[0] = fx2_texgen_linear(cx, (int16_t)dotn(n, L->d[g][0], 0));
            sgen[1] = fx2_texgen_linear(cx, (int16_t)dotn(n, L->d[g][1], 0));
        } else {
            sgen[0] = (int16_t)(MUT(11, 0x4000, 0x4001) + (dotn(n, L->d[g][0], 0) >> 1));
            sgen[1] = (int16_t)(0x4000 + (dotn(n, L->d[g][1], 0) >> 1));
        }
    }
}

static inline uint16_t clip_bits_for(int32_t X, int32_t W, unsigned k)
{
    return (uint16_t)((unsigned)hle_clip_le(X, W) << k | (unsigned)hle_clip_ge(X, W) << (8 + k));
}

/* From the clip coordinates P on: the clip flags (each coordinate against w, and against w scaled by the guard-band ratio), the perspective division with the refined reciprocal, the viewport.
 * Sets cx, cf, sx, sy, szi, szf, rwi, rwf; *fog = the fog lane's integer part; returns the two clip-bit words (the compares against w, and against the scaled w) in *vcc_a, *vcc_b. */
static void vtx_project(const struct fx2_vctx *c, const struct fx2_pair P[4], int full, struct fx2_vtx *o, unsigned *vcc_a, unsigned *vcc_b, int16_t *fog)
{
    struct fx2_pair Q[4], R0, T, E, R1, ndc[4], ndc2[4], S[4];
    hle_acc tt, fl;
    unsigned vcc = 0, vb = 0, k;
    int16_t rint;
    int32_t W, W2, Wq;

    W = (int32_t)((uint32_t)(uint16_t)P[3].i << 16 | P[3].f);
    for (k = 0; k < 4; k++)
        vcc |= clip_bits_for((int32_t)((uint32_t)(uint16_t)P[k].i << 16 | P[k].f), W, k);
    {
        hle_acc prod = (hle_acc)P[3].f * c->ratio + (((hle_acc)P[3].i * c->ratio) << 16);
        W2 = (int32_t)((uint32_t)(uint16_t)hle_sclamp(prod) << 16 | (uint16_t)prod);
        for (k = 0; k < 4; k++) {   /* the fourth lane of the compare is mixed: the LSV that puts z into the fraction vector (for the fog's reciprocal product) comes before VCL, the one for the
                                     * integer vector after VCH, so lane 3 is compared as (w integer, z fraction) */
            struct fx2_pair X = P[k];
            if (k == 3)
                X.f = P[2].f;
            vb |= clip_bits_for((int32_t)((uint32_t)(uint16_t)X.i << 16 | X.f), W2, k);
        }
    }
    *vcc_a = vcc;
    *vcc_b = vb;
    if (!full)
        return;
    for (k = 0; k < 4; k++) {
        hle_acc a = ((hle_acc)P[k].f * c->pn >> 16) + (hle_acc)P[k].i * c->pn;
        Q[k].i = hle_sclamp(a);
        Q[k].f = hle_uclamp(a);
    }
    Wq = (int32_t)((uint32_t)(uint16_t)Q[3].i << 16 | Q[3].f);
    R0.i = (int16_t)(hle_rcp32(Wq) >> 16);
    R0.f = (uint16_t)hle_rcp32(Wq);
    mm(Q[3], R0, &tt, &fl);
    T = rd(tt, fl);
    {
        hle_acc e1 = ((hle_acc)4 << 16) - (hle_acc)4 * T.f;
        hle_acc e2 = e1 - (((hle_acc)4 * T.i) << 16);
        E.i = hle_sclamp(e2);
        E.f = hle_uclamp(e1);
    }
    mm(E, R0, &tt, &fl);
    R1 = rd(tt, fl);
    rint = P[3].i >= 0 ? R1.i : 0x7fff;
    for (k = 0; k < 4; k++) {
        struct fx2_pair Rr = { rint, R1.f };
        mm(P[k], Rr, &tt, &fl);
        ndc[k] = rd(tt, fl);
        {
            hle_acc a = ((hle_acc)ndc[k].f * c->pn >> 16) + (hle_acc)ndc[k].i * c->pn;
            ndc2[k].i = hle_sclamp(a);
            ndc2[k].f = hle_uclamp(a);
        }
    }
    for (k = 0; k < 4; k++) {
        /* the fourth lane is the fog: the multiplication by the reciprocal takes z in place of w there (LSV into lane 3 of the clip coordinates), so the fog is z * multiplier + offset (+ 0x7f00,
         * which the readout of the fog byte removes again) */
        struct fx2_pair nn = k == 3 ? ndc2[2] : ndc2[k];
        hle_acc acc = ((hle_acc)c->trans[k] << 16) + (k == 3 ? ((hle_acc)1 * c->c31_3 << 16) : 0);
        hle_acc t1 = hle_wrap48(acc + (hle_acc)nn.f * c->scale[k]);
        hle_acc f1 = hle_wrap48(t1 + (((hle_acc)nn.i * c->scale[k]) << 16));
        S[k] = rd(t1, f1);
    }
    for (k = 0; k < 4; k++) {
        o->cx[k] = P[k].i;
        o->cf[k] = P[k].f;
    }
    o->sx = S[0].i;
    o->sy = S[1].i;
    o->szi = S[2].i < 0 && !MUT(212, c->gen204, 0) ? 0 : S[2].i;
    o->szf = S[2].f;
    o->rwi = R1.i;
    o->rwf = R1.f;
    *fog = S[3].i;
}

/* the vertex pipeline for one vertex (lane); returns its record without the pair-merged flags and the two clip-bit words (against w, against the guard-band w) */
static void vtx_one(const struct fx2_vctx *c, const struct fx2_vin *vi, unsigned idx, int phantom, struct fx2_vtx *o, unsigned *vcc_a, unsigned *vcc_b)
{
    int16_t v[3] = { vi->x, vi->y, vi->z };
    int16_t s = vi->s, tt_ = vi->t, fog = 0;
    uint8_t col[4] = { vi->c[0], vi->c[1], vi->c[2], vi->c[3] };
    struct fx2_pair P[4];
    unsigned k, j;

    if ((c->mode & FX2_LIGHTING) && !phantom) {
        int8_t nb[3] = { (int8_t)vi->c[0], (int8_t)vi->c[1], (int8_t)vi->c[2] };
        uint8_t lo[4];
        int16_t sg[2] = { 0, 0 };
        fx2_lit_vertex_scalar(c, MUT(12, (idx & 1) ? 0 : 1, idx & 1), nb, vi->c[3], lo, (c->mode & FX2_TEXGEN) != 0, sg);
        memcpy(col, lo, 4);
        if (c->mode & FX2_TEXGEN) {
            s = sg[0];
            tt_ = sg[1];
        }
    }
    for (k = 0; k < 4; k++) {
        hle_acc acc = hle_wrap48(((hle_acc)c->mp.i[3][k] << 16) + c->mp.f[3][k]);
        hle_acc last = 0;
        for (j = 0; j < 3; j++) {
            last = ((hle_acc)c->mp.i[j][k] * v[j]) << 16;
            acc = hle_wrap48(acc + (hle_acc)c->mp.f[j][k] * v[j] + last);
        }
        P[k].i = hle_sclamp(acc);
        P[k].f = hle_uclamp(hle_wrap48(acc - last));
    }
    vtx_project(c, P, !phantom, o, vcc_a, vcc_b, &fog);
    if (phantom)
        return;
    if (c->mode & FX2_FOG)
        col[3] = (uint8_t)(fog < MUT(13, 0x7f00, 0x7e00) ? 0x7f00 : fog);   /* the byte is the low byte of max(S3.i, 0x7f00) */
    memcpy(o->col, col, 4);
    o->s = (int16_t)(((int32_t)s * c->sc_s) >> 16);
    o->t = (int16_t)(((int32_t)tt_ * c->sc_t) >> 16);
}

void fx2_vtx_xform_scalar(const struct fx2_vctx *c, const struct fx2_vin *in, unsigned n, struct fx2_vtx *out)
{
    unsigned vcc_a[130], vcc_b[130], i;
    for (i = 0; i < n + (n & 1); i++) {
        struct fx2_vtx tmp;
        vtx_one(c, &in[i], i, i >= n, i < n ? &out[i] : &tmp, &vcc_a[i], &vcc_b[i]);
    }
    for (i = 0; i < n; i++) {
        unsigned pa = i & ~1u, pb = pa + 1;
        unsigned ca = (vcc_a[pa] & 0x0f0f) | ((vcc_a[pb] & 0x0f0f) << 4);
        unsigned cb = (vcc_b[pa] & 0x0f0f) | ((vcc_b[pb] & 0x0f0f) << 4);
        out[i].flags = (i & 1) ? (uint32_t)(cb << 16 | ca) : (uint32_t)(((cb << 4) & 0xffff) << 16 | ((ca << 4) & 0xffff));
    }
}

/* ---- triangles ------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* the command is built as the host words of DMEM hold it: word k = the big-endian value of bytes 4k .. 4k+3 (the halfword at byte offset o is in the half of the word that DMEM's accessors pick) */
typedef uint16_t __attribute__((may_alias)) u16a;
static inline void put16(uint32_t *p, unsigned o, uint16_t v) { ((u16a *)p)[(o >> 1) ^ 1] = v; }
static inline void put32(uint32_t *p, unsigned o, uint32_t v) { p[o >> 2] = v; }

struct vrec {
    int16_t x, y;
    const struct fx2_vtx *v;
};

/* The first part of the triangle setup, shared by the scalar and the NEON forms: the rejection / clip / cull decisions, the sort, the edge slopes, the header of the RDP command. */
int fx2_tri_prepare(const struct fx2_tri_in *in, uint32_t *cmd, struct fx2_tri_state *st)
{
    const struct fx2_vtx *va = in->v[0], *vb = in->v[1], *vc = in->v[2];
    struct vrec A = { va->sx, va->sy, va }, B = { vb->sx, vb->sy, vb }, C = { vc->sx, vc->sy, vc };
    struct vrec minAB, maxAB, mid, H, M, L;
    int16_t cross16;
    int32_t t3;
    uint32_t g = in->g;
    int16_t dxMH, dyMH, dxLH, dyLH, dxLM, dyLM, nxMH, nyLH;
    int16_t dy[3], dx[3];
    struct fx2_pair slope_m[3], XH = { 0, 0 }, XM = { 0, 0 };
    uint32_t cfrac;
    int32_t xHf, Cv;
    unsigned k;

    if (va->flags & vb->flags & vc->flags & in->reject_mask)
        return FX2_TRI_SKIP;
    if ((va->flags | vb->flags | vc->flags) & in->clip_mask)
        return FX2_TRI_CLIP;

    /* the signed area from the unsorted vertices decides the culling: (Ax-Bx)(Ay-Cy) + (Ax-Cx)(By-Ay), saturated 16-bit differences, the sum clamped to 16 bits */
    {
        int32_t sum = (int32_t)hle_clamp16((int32_t)A.x - B.x) * hle_clamp16((int32_t)A.y - C.y) + (int32_t)hle_clamp16((int32_t)A.x - C.x) * hle_clamp16((int32_t)B.y - A.y);
        cross16 = hle_clamp16(sum);
    }
    t3 = (int32_t)in->cull + cross16;
    if (t3 >= 0 || cross16 == 0)
        return FX2_TRI_SKIP;                       /* culled, or degenerate */

    /* the sort: min / max networks of VLT / VGE / VMRG, ties as they fall out of them */
    minAB = A.y < B.y ? A : B;
    maxAB = A.y >= B.y ? A : B;
    {
        int cmp = minAB.y >= C.y, mv;
        mid = cmp ? minAB : C;
        H = cmp ? C : minAB;
        mv = mid.y < maxAB.y;
        M = mv ? mid : maxAB;
        L = mv ? maxAB : mid;
    }
    dxMH = hle_clamp16((int32_t)M.x - H.x);
    dyMH = hle_clamp16((int32_t)M.y - H.y);
    dxLH = hle_clamp16((int32_t)L.x - H.x);
    dyLH = hle_clamp16((int32_t)L.y - H.y);
    dxLM = hle_clamp16((int32_t)L.x - M.x);
    dyLM = hle_clamp16((int32_t)L.y - M.y);
    nxMH = hle_clamp16((int32_t)H.x - M.x);
    nyLH = hle_clamp16((int32_t)H.y - L.y);
    st->area = (int32_t)dyMH * dxLH + (int32_t)dyLH * nxMH;
    dy[0] = dyLM; dy[1] = dyMH; dy[2] = dyLH;      /* the lanes 0, 2, 3: DxLDy, DxMDy, DxHDy */
    dx[0] = dxLM; dx[1] = dxMH; dx[2] = dxLH;
    cfrac = ((uint32_t)(uint16_t)H.y * 0x4000u) & 0xffff;
    xHf = (int32_t)H.x * 0x4000;
    Cv = -(int32_t)cfrac;
    st->C0.i = (int16_t)(Cv >> 16);
    st->C0.f = (uint16_t)Cv;
    for (k = 0; k < 3; k++) {
        int32_t Rr = (int32_t)hle_rcp32((int32_t)dy[k]);
        hle_acc a = hle_mac_32xu16(Rr, in->rcp_k);                       /* VMUDL VMADM VMADN with v30[15] = 0x20 */
        struct fx2_pair A32 = { hle_sclamp(a), hle_uclamp(a) };
        struct fx2_pair dxs = fxs(dx[k] * (int32_t)in->dx_k);                  /* VMUDM / VMADN 0 with v30[10] = 0x1000 */
        hle_acc tt, full;
        struct fx2_pair S;
        int16_t lim = in->slope_lim;                                     /* VCR with v30[11] = 0x100 */
        mm(dxs, A32, &tt, &full);
        S = rd(tt, full);
        if (S.i > lim)
            S.i = lim;
        if (S.i < ~lim)
            S.i = ~lim;
        st->slope[k] = S;
        slope_m[k].i = S.i;
        slope_m[k].f = S.f & MUT(6, in->frac_mask, in->frac_mask ^ 0x10);        /* VAND with v30[13] = 0xfff8 */
    }
    /* XH, XM = H.x * 0x4000 + floor(slope * (-(frac of y/4)) / 65536), the fraction read before the integer product */
    for (k = 1; k < 3; k++) {
        hle_acc tt = (hle_acc)xHf + ((hle_acc)slope_m[k].f * st->C0.f >> 16) + (hle_acc)slope_m[k].i * st->C0.f + (hle_acc)slope_m[k].f * st->C0.i;
        struct fx2_pair r = rd(tt, tt + (((hle_acc)slope_m[k].i * st->C0.i) << 16));
        if (k == 1)
            XM = r;
        else
            XH = r;
    }
    put16(cmd, 0, (uint16_t)(((((g | 0xc8) | in->tex) & 0xff) << 8) | (((st->area < 0 ? 0x80 : 0) | in->tile) & 0xff)));
    put16(cmd, 2, (uint16_t)L.y);
    put16(cmd, 4, (uint16_t)M.y);
    put16(cmd, 6, (uint16_t)H.y);
    put32(cmd, 8, (uint32_t)((int32_t)M.x << 14));
    put16(cmd, 12, (uint16_t)st->slope[0].i);
    put16(cmd, 14, st->slope[0].f);
    put16(cmd, 16, (uint16_t)XH.i);
    put16(cmd, 18, XH.f);
    put16(cmd, 20, (uint16_t)st->slope[2].i);
    put16(cmd, 22, st->slope[2].f);
    put16(cmd, 24, (uint16_t)XM.i);
    put16(cmd, 26, XM.f);
    put16(cmd, 28, (uint16_t)st->slope[1].i);
    put16(cmd, 30, st->slope[1].f);
    st->H = H.v;
    st->M = M.v;
    st->L = L.v;
    st->dyMH = dyMH;
    st->nyLH = nyLH;
    st->dxLH = dxLH;
    st->nxMH = nxMH;
    return FX2_TRI_DRAW;
}

/* The attributes: the vertices H M L (and A for the flat colour) give 8 lanes each, 0..3 colour, 4..6 s t w, 7 z, each an (int, frac) pair; the gradients of each (per x, per y, along the edge) and the
 * value at the start of the edge, as the shade, texture and z blocks of the RDP command. */
unsigned fx2_tri_attr_prep(const struct fx2_tri_in *in, const struct fx2_tri_state *st, struct fx2_pair att[3][8], struct fx2_pair *R1out)
{
    uint32_t g = in->g;
    unsigned shade = g & 4, zb = g & 1, tex = in->tex, k, l;
    const struct fx2_vtx *va = in->flat ? in->flat : in->v[0];
    const struct fx2_vtx *vs3[3] = { st->H, st->M, st->L };
    struct fx2_pair R0, R1, T, E, Rmw = { 0, 0 };
    hle_acc tt, full;
    unsigned smooth = g & 0x200000;
    int32_t area = st->area;

    for (k = 0; k < 3; k++) {
        for (l = 0; l < 4; l++) {
            const struct fx2_vtx *src = smooth || l == 3 ? vs3[k] : va;
            att[k][l].i = src->col[l];
            att[k][l].f = MUT(9, 0x8000, 0x8001);
        }
        att[k][7].i = vs3[k]->szi;
        att[k][7].f = vs3[k]->szf;
        for (l = 4; l < 7; l++) {
            att[k][l].i = 0;
            att[k][l].f = 0;
        }
    }
    if (tex) {
        uint32_t wv[3];
        int32_t mw;
        struct fx2_pair I[3];
        for (k = 0; k < 3; k++) {
            wv[k] = (uint32_t)(uint16_t)vs3[k]->rwi << 16 | vs3[k]->rwf;
            I[k] = fxs((int32_t)wv[k]);
        }
        {   /* the minimum of the three as the microcode finds it: a - ((a - b) & ((a - b) >> 31)) twice, in 32-bit arithmetic */
            uint32_t a1 = wv[0], d = a1 - wv[1], m = (uint32_t)((int32_t)d >> 31);
            a1 -= d & m;
            d = a1 - wv[2];
            m = (uint32_t)((int32_t)d >> 31);
            a1 -= d & m;
            mw = (int32_t)a1;
        }
        Rmw = fxs((int32_t)hle_rcp32(mw));
        for (k = 0; k < 3; k++) {
            struct fx2_pair r;
            int16_t x[3] = { vs3[k]->s, vs3[k]->t, 0x7fff };
            mm(I[k], Rmw, &tt, &full);
            r = rd(tt, full);
            for (l = 0; l < 3; l++) {
                hle_acc acc = (hle_acc)x[l] * r.f + (((hle_acc)x[l] * r.i) << 16);
                att[k][4 + l].i = hle_sclamp(acc);
                att[k][4 + l].f = hle_uclamp(acc);
            }
        }
    }
    /* 1 / area, one Newton step: R0 = rcp32(area), T = R0 * area, E = 16 - 16 T, R1 = R0 * E */
    R0 = fxs((int32_t)hle_rcp32(area));
    mm(R0, fxs(area), &tt, &full);
    T = rd(tt, full);
    {
        hle_acc e1 = ((hle_acc)16 << 16) - (hle_acc)16 * T.f;
        hle_acc e2 = e1 - (((hle_acc)16 * T.i) << 16);
        E.i = hle_sclamp(e2);
        E.f = hle_uclamp(MUT(7, e1, e1 + 1));
    }
    mm(E, R0, &tt, &full);
    R1 = rd(tt, full);
    *R1out = R1;
    return (shade ? 0x0f : 0) | (tex ? 0xf0 : 0) | (zb ? 0x80 : 0);
}

unsigned fx2_tri_attrs_scalar(const struct fx2_tri_in *in, const struct fx2_tri_state *st, uint32_t *cmd)
{
    uint32_t g = in->g;
    unsigned shade = g & 4, zb = g & 1, tex = in->tex, o = 32, k, l;
    struct fx2_pair att[3][8], R1;
    struct fx2_pair dX[8], dY[8], dE[8], stv[8];
    struct fx2_pair sl = st->slope[2], C0 = st->C0;
    int16_t dyMH = st->dyMH, nyLH = st->nyLH, dxLH = st->dxLH, nxMH = st->nxMH;
    unsigned need = fx2_tri_attr_prep(in, st, att, &R1);
    hle_acc tt, full;

    for (l = 0; l < 8; l++) {
        struct fx2_pair dMH, dLH, nx, ny;
        hle_acc a1, a2, t2, f2, tE, fE, tS, fS;
        if (!(need >> l & 1))
            continue;
        dLH = fsub(att[2][l], att[0][l]);
        dMH = fsub(att[1][l], att[0][l]);
        a1 = hle_wrap48(pv(dLH) * dyMH + pv(dMH) * nyLH);
        a2 = hle_wrap48(pv(dMH) * dxLH + pv(dLH) * nxMH);
        nx.i = (int16_t)(a1 >> 32);
        nx.f = (uint16_t)(a1 >> 16);
        ny.i = (int16_t)(a2 >> 32);
        ny.f = (uint16_t)(a2 >> 16);
        mm(nx, R1, &tt, &full);
        dX[l] = rd(tt, full);
        mm(ny, R1, &t2, &f2);
        dY[l] = rd(t2, f2);
        /* dE = dY + dX * DxHDy, on top of the accumulator the dY chain left */
        tE = hle_wrap48(f2 + ((hle_acc)dX[l].f * sl.f >> 16) + (hle_acc)dX[l].i * sl.f + (hle_acc)dX[l].f * sl.i);
        fE = hle_wrap48(tE + (((hle_acc)dX[l].i * sl.i) << 16));
        dE[l] = rd(tE, fE);
        tS = hle_wrap48(pv(att[0][l]) + ((hle_acc)dE[l].f * C0.f >> 16) + (hle_acc)dE[l].i * C0.f + (hle_acc)dE[l].f * C0.i);
        fS = hle_wrap48(tS + (((hle_acc)dE[l].i * C0.i) << 16));
        stv[l] = rd(tS, fS);
    }
    for (k = 0; k < 2; k++) {
        unsigned base, lo;
        if (k == 0) {
            if (!shade)
                continue;
            base = o;
            lo = 0;
        } else {
            if (!tex)
                continue;
            base = o + (shade ? 64 : 0);
            lo = 4;
        }
        for (l = 0; l < 4; l++) {
            put16(cmd, base + 2 * l, (uint16_t)stv[lo + l].i);
            put16(cmd, base + 8 + 2 * l, (uint16_t)dX[lo + l].i);
            put16(cmd, base + 16 + 2 * l, MUT(8, stv[lo + l].f, (uint16_t)(stv[lo + l].f + 1)));
            put16(cmd, base + 24 + 2 * l, dX[lo + l].f);
            put16(cmd, base + 32 + 2 * l, (uint16_t)dE[lo + l].i);
            put16(cmd, base + 40 + 2 * l, (uint16_t)dY[lo + l].i);
            put16(cmd, base + 48 + 2 * l, dE[lo + l].f);
            put16(cmd, base + 56 + 2 * l, dY[lo + l].f);
        }
    }
    o += (shade ? 64 : 0) + (tex ? 64 : 0);
    if (zb) {
        struct fx2_pair zst, zx, zy, ze;
        hle_acc a;
        uint16_t v10 = (uint16_t)((uint32_t)dE[7].f * (uint32_t)(int32_t)(int16_t)C0.f);
        a = (hle_acc)dX[7].f * 32 + (((hle_acc)dX[7].i * 32) << 16);
        zx.i = hle_sclamp(a);
        zx.f = (uint16_t)(dX[7].f * 32);
        a = (hle_acc)dY[7].f * 32 + (((hle_acc)dY[7].i * 32) << 16);
        zy.i = hle_sclamp(a);
        zy.f = (uint16_t)(dY[7].f * 32);
        a = (hle_acc)dE[7].f * 32 + (((hle_acc)dE[7].i * 32) << 16);
        ze.i = hle_sclamp(a);
        ze.f = (uint16_t)(dE[7].f * 32);
        a = (hle_acc)(((uint32_t)v10 * 32) >> 16) + (hle_acc)stv[7].f * 32 + (((hle_acc)stv[7].i * 32) << 16);
        zst.i = hle_sclamp(a);
        zst.f = hle_uclamp(a - (((hle_acc)stv[7].i * 32) << 16));      /* the fraction is read before the integer product is added (a saturated integer part leaves the fraction alone) */
        put16(cmd, o + 0, (uint16_t)zst.i);
        put16(cmd, o + 2, zst.f);
        put16(cmd, o + 4, (uint16_t)zx.i);
        put16(cmd, o + 6, zx.f);
        put16(cmd, o + 8, (uint16_t)ze.i);
        put16(cmd, o + 10, ze.f);
        put16(cmd, o + 12, (uint16_t)zy.i);
        put16(cmd, o + 14, zy.f);
        o += 16;
    }
    return o;
}

int fx2_tri_setup_scalar(const struct fx2_tri_in *in, uint32_t *cmd, unsigned *len)
{
    struct fx2_tri_state st;
    int r = fx2_tri_prepare(in, cmd, &st);
    if (r != FX2_TRI_DRAW)
        return r;
    *len = (in->g & 4) || in->tex || (in->g & 1) ? fx2_tri_attrs_scalar(in, &st, cmd) : 32;
    return FX2_TRI_DRAW;
}

/* ---- the clipper ------------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* The intersection of the edge inside -> outside with a plane: the distances of both ends (the 48-bit products summed over the lanes with the carrying 32-bit adds), the fraction t = d_in /
 * (d_in - d_out) from a reciprocal (rcp32, doubled, sign made positive; its low half scales the difference; the rcp of that product, one Newton step; times d_in; rescaled; clamped to [1, 0xffff]
 * as a 16-bit fraction), and the new clip coordinates out * t + in * (1 - t) in 32-bit pairs. Lane 3 of the vector code holds the sum; the colour lanes and the texture coordinates are
 * interpolated with the same t. */
static struct hle_pair lane_sum(const struct hle_pair l[4])
{
    struct hle_pair a = hle_add32(hle_pair32(l[3]), hle_pair32(l[2])), b = hle_add32(hle_pair32(l[1]), hle_pair32(l[0]));
    return hle_add32(hle_pair32(a), hle_pair32(b));
}

static void clip_intersect(const struct fx2_clip_ctx *c, const int16_t pl[4], const struct fx2_vtx *in, const struct fx2_vtx *out, struct fx2_vtx *nv, struct fx2_pair P[4])
{
    struct hle_pair din[4], dd[4], dsum_in, dsum_d;
    unsigned j;
    uint16_t Rlo, t, nt;
    uint32_t R0;
    int16_t sg;
    int32_t D32, T32, R2;
    struct fx2_pair T, E, R3, mt;
    hle_acc tt, fl, acc;

    for (j = 0; j < 4; j++) {
        hle_acc a = (hle_acc)in->cf[j] * pl[j];
        hle_acc a2;
        din[j].f = hle_uclamp(a);
        a += ((hle_acc)in->cx[j] * pl[j]) << 16;
        din[j].i = hle_sclamp(a);
        a += (hle_acc)out->cf[j] * (-pl[j]);
        dd[j].f = hle_uclamp(a);
        a2 = a + (((hle_acc)out->cx[j] * (-pl[j])) << 16);
        dd[j].i = hle_sclamp(a2);
    }
    dsum_in = lane_sum(din);
    dsum_d = lane_sum(dd);
    D32 = hle_pair32(dsum_d);
    R0 = hle_rcp32(D32);
    if (MUT(210, c->sign0, 0))                          /* VABS of the integer part itself (2.04): 0 for 0 */
        sg = dsum_d.i < 0 ? -2 : dsum_d.i > 0 ? 2 : 0;
    else                                                /* VABS of the integer part with its low bit set (VOR with 1) */
        sg = (int16_t)(dsum_d.i | 1) < 0 ? -2 : 2;
    {
        hle_acc a = (hle_acc)(uint16_t)R0 * sg;
        uint16_t v2 = hle_uclamp(a);
        int16_t v3;
        a += ((hle_acc)(int16_t)(R0 >> 16) * sg) << 16;
        v3 = hle_sclamp(a);
        Rlo = v3 == 0 ? v2 : 0xffff;
    }
    {
        hle_acc a = hle_mac_32xu16(D32, Rlo);
        T.i = hle_sclamp(a);
        T.f = hle_uclamp(a);
    }
    T32 = (int32_t)((uint32_t)(uint16_t)T.i << 16 | T.f);
    R2 = (int32_t)hle_rcp32(T32);
    {
        struct fx2_pair r2 = fxs(R2);
        mm(r2, T, &tt, &fl);
        mt = rd(tt, fl);
        {
            hle_acc e1 = ((hle_acc)4 << 16) - (hle_acc)4 * mt.f;
            hle_acc e2 = e1 - (((hle_acc)4 * mt.i) << 16);
            E.i = hle_sclamp(e2);
            E.f = hle_uclamp(e1);
        }
        mm(r2, E, &tt, &fl);
        R3 = rd(tt, fl);
    }
    {
        struct fx2_pair di = { dsum_in.i, dsum_in.f }, t0;
        mm(di, R3, &tt, &fl);
        t0 = rd(tt, fl);
        acc = hle_mac_32xu16((int32_t)((uint32_t)(uint16_t)t0.i << 16 | t0.f), Rlo);
        {
            int16_t ti = hle_sclamp(acc);
            uint16_t tf = hle_uclamp(acc);
            if (ti >= 1) {                                  /* VLT: min(int, 1); the fraction is 0xffff where the integer part was not below 1 */
                ti = 1;
                tf = 0xffff;
            }
            /* VSUBC: borrow if tf < 1, ne if tf != 1; VGE: int > 0, or int == 0 unless (tf == 0) */
            if (ti < 0 || (ti == 0 && tf == 0))
                tf = 1;
            t = tf;
        }
    }
    nt = (uint16_t)(0u - t);
    for (j = 0; j < 4; j++) {
        hle_acc a = ((hle_acc)out->cf[j] * t >> 16) + (hle_acc)out->cx[j] * t + ((hle_acc)in->cf[j] * nt >> 16) + (hle_acc)in->cx[j] * nt;
        P[j].i = hle_sclamp(a);
        P[j].f = hle_uclamp(a);
    }
    for (j = 0; j < 4; j++) {
        hle_acc a = (hle_acc)((int16_t)(out->col[j] << 7)) * t + (hle_acc)((int16_t)(in->col[j] << 7)) * nt;
        nv->col[j] = (uint8_t)(hle_sclamp(a) >> 7);
    }
    nv->s = hle_sclamp((hle_acc)out->s * t + (hle_acc)in->s * nt);
    nv->t = hle_sclamp((hle_acc)out->t * t + (hle_acc)in->t * nt);
}

int fx2_clip_tri_scalar(const struct fx2_clip_ctx *c, const struct fx2_vtx *const v[3], struct fx2_clip_out *o)
{
    const struct fx2_vtx *list[2][16];
    unsigned n[2], cur = 0, p, i;
    const struct fx2_vtx *last;

    o->ntri = o->nnew = 0;
    list[0][0] = v[0];
    list[0][1] = v[1];
    list[0][2] = v[2];
    n[0] = 3;
    last = v[2];
    for (p = 0; p < 6; p++) {
        uint32_t m = c->mask[p];
        unsigned s0 = (last->flags & m) != 0, nxt = cur ^ 1;
        const struct fx2_vtx *prev = last;
        n[nxt] = 0;
        for (i = 0; i < n[cur]; i++) {
            const struct fx2_vtx *v2 = list[cur][i];
            unsigned s2 = (v2->flags & m) != 0;
            if (s2 != s0) {                             /* the edge crosses the plane */
                const struct fx2_vtx *in = s2 ? prev : v2, *out = s2 ? v2 : prev;
                struct fx2_vtx *nv;
                struct fx2_pair P[4];
                unsigned va, vb;
                int16_t fogv;
                if (o->nnew >= FX2_CLIP_MAX_NEW)
                    return 1;
                nv = &o->nv[o->nnew++];
                clip_intersect(c, c->plane[p], in, out, nv, P);
                vtx_project(c->v, P, 1, nv, &va, &vb, &fogv);
                /* the flags of the t7 lane group: the compares of lanes 0-3, shifted by 4 (the junk of the other lane group is left out) */
                nv->flags = (uint32_t)(((vb & 0x0f0f) << 4) & 0xffff) << 16 | (((va & 0x0f0f) << 4) & 0xffff);
                if (n[nxt] >= 15)
                    return 1;
                list[nxt][n[nxt]++] = nv;
            }
            if (!s2) {
                if (n[nxt] >= 15)
                    return 1;
                list[nxt][n[nxt]++] = v2;
            }
            s0 = s2;
            prev = v2;
        }
        if (n[nxt] < 3)
            return 0;                                   /* nothing is left of the triangle */
        cur = nxt;
        last = list[cur][n[cur] - 1];
    }
    for (i = 0; i + 2 < n[cur]; i++) {
        if (MUT(211, c->fan_l0, 0)) {
            o->tri[o->ntri][0] = list[cur][0];
            o->tri[o->ntri][1] = list[cur][n[cur] - 2 - i];
            o->tri[o->ntri][2] = list[cur][n[cur] - 1 - i];
        } else {
            o->tri[o->ntri][0] = list[cur][i];
            o->tri[o->ntri][1] = list[cur][i + 1];
            o->tri[o->ntri][2] = list[cur][n[cur] - 1];
        }
        o->ntri++;
    }
    return 0;
}

/* ---- the non-NEON targets -------------------------------------------------------------------------------------------------------------------------------------------------- */
#ifndef __aarch64__
void fx2_mtx_mul(const struct fx2_mtx *a, const struct fx2_mtx *b, struct fx2_mtx *out) { fx2_mtx_mul_scalar(a, b, out); }
#endif
void fx2_light_setup(const struct fx2_mtx *mv, const int8_t raw[][3], unsigned nlights, int16_t scale, int8_t out[][3]) { fx2_light_setup_scalar(mv, raw, nlights, scale, out); }
#ifndef __aarch64__
void fx2_vtx_xform(const struct fx2_vctx *c, const struct fx2_vin *in, unsigned n, struct fx2_vtx *out) { fx2_vtx_xform_scalar(c, in, n, out); }
#endif
#ifndef __aarch64__
int fx2_tri_setup(const struct fx2_tri_in *in, uint32_t *cmd, unsigned *len) { return fx2_tri_setup_scalar(in, cmd, len); }
#endif
int fx2_clip_tri(const struct fx2_clip_ctx *c, const struct fx2_vtx *const v[3], struct fx2_clip_out *o) { return fx2_clip_tri_scalar(c, v, o); }
