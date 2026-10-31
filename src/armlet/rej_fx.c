/*
 * arMLEt - rej_fx.c: the scalar reference of the Rej geometry functions (rej_fx.h)
 *
 * Plain C11 on structs, bit-exact to what the microcode's vector code computes (tools/hle/test_rej.c checks it against translations of the microcode). The arithmetic is hle_fx.h's.
 * docs/ucode-rej.md has the microcode's side.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <stdint.h>
#include <string.h>

#include "f3dex2_fx_int.h"
#include "hle_fx.h"
#include "rej_fx.h"

#ifdef ARMLET_HLE_MUTATE
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#define MUT(n, ok, bad) (ok)
#endif

/* ---- (integer, fraction) pairs: the chains of f3dex2_fx.c ------------------------------------------------------------------------------------------------------------------ */
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


/* ---- F3DFLX.Rej's lighting ----------------------------------------------------------------------------------------------------------------------------------------------- */
/* VMULU / VMACU read the accumulator as an unsigned number: below zero 0, from 0x8000 (bit 31 of the 32 bits behind the 16-bit fraction) 0xffff, else bits 31..16 */
static inline uint16_t clampu(hle_acc a)
{
    if (a < 0)
        return 0;
    a >>= 16;
    return a > 0x7fff ? 0xffff : (uint16_t)a;
}

/* The lit colour of one vertex: the lambert term n . d (VMULU, VMACU, VMACU: each product doubled, 0x8000 rounding in the first) clamped as an unsigned 16-bit number but then used as a signed factor
 * (a value from 0x8000 on, a normal beyond the light's cone of 90 degrees cannot be one, is negative), the colour byte c << 7 times 0x7fff plus the light's colour byte << 7 times the lambert
 * term, clamped the same way; the byte is the result >> 7. */
void rej_flx_lit_vertex(const struct rej_flx_lit *fl, unsigned g, const int8_t nb[3], uint8_t out[4])
{
    int16_t n[3] = { (int16_t)(nb[0] * 256), (int16_t)(nb[1] * 256), (int16_t)(nb[2] * 256) }, L;
    hle_acc acc = (hle_acc)2 * n[0] * fl->dir[g][0] + 0x8000;
    unsigned c;
    acc = hle_wrap48(acc + (hle_acc)2 * n[1] * fl->dir[g][1]);
    acc = hle_wrap48(acc + (hle_acc)2 * n[2] * fl->dir[g][2]);
    L = (int16_t)clampu(acc);
    for (c = 0; c < 3; c++) {
        hle_acc a = (hle_acc)2 * (fl->amb[g][c] << 7) * fl->k7 + 0x8000;
        a = hle_wrap48(a + (hle_acc)2 * (fl->col[g][c] << 7) * L);
        out[c] = (uint8_t)(clampu(a) >> 7);
    }
    if (fl->texgen) {                               /* n . dir1 (VMULF, VMACF, VMACF), its high byte (SPV) indexes the table: the alpha */
        hle_acc d = hle_mulf(n[0], fl->dir1[g][0]);
        d = hle_macf(d, n[1], fl->dir1[g][1]);
        d = hle_macf(d, n[2], fl->dir1[g][2]);
        out[3] = fl->tab[(uint8_t)(hle_sclamp(d) >> 8)];
    }
}

/* n . d as the accumulator holds it: 2 n0 d0 + 0x8000 + 2 n1 d1 + 2 n2 d2 (VMULU / VMULF, then two VMACs), and the signed 16-bit readout (the texture coordinates') */
static inline hle_acc lx_dot(const int16_t n[3], const int16_t d[3])
{
    return hle_wrap48((hle_acc)2 * n[0] * d[0] + 0x8000 + (hle_acc)2 * n[1] * d[1] + (hle_acc)2 * n[2] * d[2]);
}
static inline int16_t lx_dots(const int16_t n[3], const int16_t d[3]) { return hle_sclamp(lx_dot(n, d)); }


/* F3DLX.Rej 2.03's lit colour: the loop of fx2_lit_vertex_scalar (pairs of slots from the ambient slot down) with VMULU / VMACU: the lambert term (n . d, clamped as an unsigned 16-bit number but not masked to 15 bits)
 * multiplies as a signed factor; the running colour is the accumulator read as an unsigned number; the byte is its bits 14 .. 7. The texture coordinates are the F3DEX2 ones. */
void rej_lx_lit_vertex(const struct fx2_vctx *cx, unsigned g, const int8_t nb[3], uint8_t alpha, uint8_t out[4], int texgen, int16_t sgen[2])
{
    const struct fx2_lights *L = &cx->L;
    int16_t n[3] = { (int16_t)(nb[0] * 256), (int16_t)(nb[1] * 256), (int16_t)(nb[2] * 256) };
    int16_t cur[3];
    unsigned m = L->top, c;
    for (c = 0; c < 3; c++)
        cur[c] = L->c[g][m][c];
    for (;;) {
        hle_acc a[3];
        int16_t lam1 = (int16_t)clampu(lx_dot(n, L->d[g][m])), lam2 = 0;
        if (m != 2)
            lam2 = (int16_t)clampu(lx_dot(n, L->d[g][m - 1]));
        for (c = 0; c < 3; c++) {
            cur[c] &= MUT(374, 0x7fff, -1);                 /* (VAND with 0x7fff: in the delay slot of the branch to the single slot's code, too) */
            a[c] = hle_wrap48((hle_acc)2 * cur[c] * MUT(373, 0x7fff, 0x7f00) + 0x8000);
            a[c] = hle_wrap48(a[c] + (hle_acc)2 * L->c[g][m - 1][c] * lam1);
            if (m != 2)
                a[c] = hle_wrap48(a[c] + (hle_acc)2 * L->c[g][m - 2][c] * lam2);
        }
        for (c = 0; c < 3; c++)
            cur[c] = (int16_t)clampu(a[c]);
        if (m == 2 || m - 1 == 2)
            break;
        m -= 2;
    }
    for (c = 0; c < 3; c++)
        out[c] = (uint8_t)((uint16_t)cur[c] >> 7);
    out[3] = alpha;                                         /* (SBV: the input alpha) */
    if (texgen) {
        if (cx->mode & FX2_TEXGEN_LINEAR) {
            sgen[0] = fx2_texgen_linear(cx, lx_dots(n, L->d[g][0]));
            sgen[1] = fx2_texgen_linear(cx, lx_dots(n, L->d[g][1]));
        } else {
            sgen[0] = (int16_t)(0x4000 + (lx_dots(n, L->d[g][0]) >> 1));
            sgen[1] = (int16_t)(0x4000 + (lx_dots(n, L->d[g][1]) >> 1));
        }
    }
}

/* One direction of the lights' setup (fx2_light_setup_scalar's body): T = d x M (the three rows of the matrix, each product the 48-bit chain read as high:middle), the squares as 32-bit products, their
 * 32-bit sum, R = the reciprocal square root, T R times the scale */
void rej_light_dir(const struct fx2_mtx *mv, const int16_t d[3], int16_t scale, int16_t out[3])
{
    int32_t T[3], R;
    struct hle_pair sq[3], sum;
    unsigned i, j;
    for (i = 0; i < 3; i++) {
        hle_acc acc = 0;
        for (j = 0; j < 3; j++)
            acc = hle_wrap48(acc + (hle_acc)mv->f[i][j] * d[j] + (((hle_acc)mv->i[i][j] * d[j]) << 16));
        T[i] = (int32_t)((uint32_t)hle_acc_hi(acc) << 16 | hle_acc_md(acc));
        sq[i].i = hle_sclamp(hle_mac_32x32(T[i], T[i]));
        sq[i].f = hle_uclamp(hle_mac_32x32_before_h(T[i], T[i]));
    }
    sum = hle_add32(hle_pair32(hle_add32(hle_pair32(sq[0]), hle_pair32(sq[1]))), hle_pair32(sq[2]));
    R = (int32_t)hle_rsq32(hle_pair32(sum));
    for (i = 0; i < 3; i++) {
        hle_acc a = hle_mac_32x32(T[i], R);
        int16_t hi = hle_sclamp(a);
        uint16_t lo = hle_uclamp(hle_mac_32x32_before_h(T[i], R));
        out[i] = hle_sclamp(hle_wrap48((hle_acc)lo * scale + (((hle_acc)hi * scale) << 16)));
    }
}

/* ---- vertices ------------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* From the clip coordinates P (x y z w, 32-bit) on. A vertex outside the guard band (x, y or z beyond w times the clip ratio: the compare of VCH / VCL against the scaled w) is marked: the
 * viewport transformation's accumulator starts with 0x7fff * 0x7fff, so every coordinate saturates, and the screen y (with its low two bits cleared) is 0x7ffc, which the triangle code rejects. */
static void rej_project(const struct fx2_vctx *c, unsigned flags, const struct fx2_pair P[4], struct rej_vtx *o, int16_t *fog)
{
    struct fx2_pair Q[4], R0, T, E, R1, ndc[4], ndc2[4], S[4];
    hle_acc tt, fl;
    unsigned vb = 0, k;
    int32_t Wq;
    int marked;

    {
        hle_acc prod = (hle_acc)P[3].f * c->ratio + (((hle_acc)P[3].i * c->ratio) << 16);
        int32_t W2 = (int32_t)((uint32_t)(uint16_t)hle_sclamp(prod) << 16 | (uint16_t)prod);
        for (k = 0; k < 3; k++) {
            int32_t X = (int32_t)((uint32_t)(uint16_t)P[k].i << 16 | P[k].f);
            vb |= (unsigned)hle_clip_le(X, W2) << k | (unsigned)hle_clip_ge(X, W2) << (8 + k);
        }
    }
    marked = MUT(377, vb != 0, (vb & 0x0303) != 0);
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
    for (k = 0; k < 4; k++) {
        const struct fx2_pair *p = &P[k == 3 ? 2 : k];                   /* the fourth lane is the fog: z in place of w */
        mm(*p, R1, &tt, &fl);
        ndc[k] = rd(tt, fl);
        {
            hle_acc a = ((hle_acc)ndc[k].f * c->pn >> 16) + (hle_acc)ndc[k].i * c->pn;
            ndc2[k].i = hle_sclamp(a);
            ndc2[k].f = hle_uclamp(a);
        }
    }
    for (k = 0; k < 4; k++) {
        hle_acc acc = ((hle_acc)(k == 1 && (flags & REJ_YQ) ? hle_clamp16((int32_t)c->trans[1] + MUT(379, 2, 1)) : c->trans[k]) << 16) + (k == 3 ? ((hle_acc)c->c31_3 << 16) : 0) + (marked ? (((hle_acc)MUT(378, 0x7fff, 0x7ffe) * c->c31_7) << 16) : 0);
        hle_acc t1 = hle_wrap48(acc + (hle_acc)ndc2[k].f * c->scale[k]);
        hle_acc f1 = hle_wrap48(t1 + (((hle_acc)ndc2[k].i * c->scale[k]) << 16));
        S[k] = rd(t1, f1);
    }
    o->sx = S[0].i;
    o->sy = (flags & REJ_YQ) ? (int16_t)(S[1].i & MUT(380, ~3, ~1)) : S[1].i;
    o->szi = S[2].i;
    o->szf = S[2].f;
    o->rwi = R1.i;
    o->rwf = R1.f;
    *fog = S[3].i;
}

static void rej_vtx_one(const struct fx2_vctx *c, unsigned flags, const struct rej_flx_lit *fl, const struct fx2_vin *vi, unsigned idx, struct rej_vtx *o)
{
    int16_t v[3] = { vi->x, vi->y, vi->z };
    int16_t s = vi->s, tt_ = vi->t, fog = 0;
    uint8_t col[4] = { vi->c[0], vi->c[1], vi->c[2], vi->c[3] };
    struct fx2_pair P[4];
    unsigned k, j;

    if ((flags & REJ_LIT_LX) && (c->mode & FX2_LIGHTING)) {
        int8_t nb[3] = { (int8_t)vi->c[0], (int8_t)vi->c[1], (int8_t)vi->c[2] };
        uint8_t lo[4];
        int16_t sg[2] = { 0, 0 };
        rej_lx_lit_vertex(c, (idx & 1) ? 0 : 1, nb, vi->c[3], lo, (c->mode & FX2_TEXGEN) != 0, sg);
        memcpy(col, lo, 4);
        if (c->mode & FX2_TEXGEN) {
            s = sg[0];
            tt_ = sg[1];
        }
    } else if (fl && (c->mode & FX2_LIGHTING)) {
        int8_t nb[3] = { (int8_t)vi->c[0], (int8_t)vi->c[1], (int8_t)vi->c[2] };
        rej_flx_lit_vertex(fl, (idx & 1) ? 0 : 1, nb, col);        /* (the alpha stays the input's unless the table gives it) */
    } else if (c->mode & FX2_LIGHTING) {
        int8_t nb[3] = { (int8_t)vi->c[0], (int8_t)vi->c[1], (int8_t)vi->c[2] };
        uint8_t lo[4];
        int16_t sg[2] = { 0, 0 };
        fx2_lit_vertex_scalar(c, (idx & 1) ? 0 : 1, nb, vi->c[3], lo, (c->mode & FX2_TEXGEN) != 0, sg);
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
    rej_project(c, flags, P, o, &fog);
    if (c->mode & FX2_FOG)
        col[3] = (uint8_t)(fog < MUT(381, 0x7f00, 0x7e00) ? 0x7f00 : fog);   /* the byte is the low byte of max(S3.i, 0x7f00) */
    memcpy(o->col, col, 4);
    o->s = (int16_t)(((int32_t)s * c->sc_s) >> 16);
    o->t = (int16_t)(((int32_t)tt_ * c->sc_t) >> 16);
}

void rej_vtx_xform_scalar(const struct fx2_vctx *c, unsigned flags, const struct rej_flx_lit *fl, const struct fx2_vin *in, unsigned n, struct rej_vtx *out)
{
    unsigned i;
    for (i = 0; i < n; i++)
        rej_vtx_one(c, flags, fl, &in[i], i, &out[i]);
}

#ifndef __aarch64__
void rej_vtx_xform(const struct fx2_vctx *c, unsigned flags, const struct rej_flx_lit *fl, const struct fx2_vin *in, unsigned n, struct rej_vtx *out)
{
    rej_vtx_xform_scalar(c, flags, fl, in, n, out);
}
#endif

/* ---- triangles ------------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* The triangle code is the F3DEX2 one without the clipper (f3dex2_fx.c): the same slopes, header and shade, texture and z blocks (the attribute part is fx2_tri_attrs_*), with F3DZEX's lanes of the
 * constant vectors and no clamp on the slopes, behind other decisions: a vertex whose y is 0x7ffc or more is a rejected one (rej_project) and takes the triangle with it; the cull table has a bit per
 * sign of the cross product; and the sort of the three vertices breaks the tie of the two larger y values the other way (the larger of the pair that was compared with the third is the middle one). */
typedef uint16_t __attribute__((may_alias)) u16a;
static inline void put16(uint32_t *p, unsigned o, uint16_t v) { ((u16a *)p)[(o >> 1) ^ 1] = v; }
static inline void put32(uint32_t *p, unsigned o, uint32_t v) { p[o >> 2] = v; }
static inline struct fx2_pair fxs(int32_t v)
{
    struct fx2_pair r = { (int16_t)(v >> 16), (uint16_t)v };
    return r;
}

struct vrec {
    int16_t x, y;
    const struct fx2_vtx *v;
};

static int rej_tri_prepare(const struct rej_tri_in *in, const struct fx2_vtx *fv, uint32_t *cmd, struct fx2_tri_state *st)
{
    const struct fx2_vtx *va = &fv[0], *vb = &fv[1], *vc = &fv[2];
    struct vrec A = { va->sx, va->sy, va }, B = { vb->sx, vb->sy, vb }, C = { vc->sx, vc->sy, vc };
    struct vrec minAB, maxAB, mid, H, M, L;
    int16_t cross16;
    uint32_t g = in->g;
    int16_t dxMH, dyMH, dxLH, dyLH, dxLM, dyLM, nxMH, nyLH;
    int16_t dy[3], dx[3];
    struct fx2_pair slope_m[3], XH = { 0, 0 }, XM = { 0, 0 };
    uint32_t cfrac;
    int32_t xHf, Cv;
    unsigned k;

    for (k = 0; k < 3; k++)
        if (fv[k].sy >= MUT(383, REJ_MARK_Y, REJ_MARK_Y + 1))
            return FX2_TRI_SKIP;
    {
        int32_t sum = (int32_t)hle_clamp16((int32_t)A.x - B.x) * hle_clamp16((int32_t)A.y - C.y) + (int32_t)hle_clamp16((int32_t)A.x - C.x) * hle_clamp16((int32_t)B.y - A.y);
        cross16 = hle_clamp16(sum);
    }
    if (cross16 == 0 || (cross16 < 0 ? !MUT(360, in->pass_neg, in->pass_pos) : !MUT(360, in->pass_pos, in->pass_neg)))
        return FX2_TRI_SKIP;

    minAB = A.y < B.y ? A : B;
    maxAB = A.y >= B.y ? A : B;
    {
        int cmp = minAB.y >= C.y, ge;
        mid = cmp ? minAB : C;
        H = cmp ? C : minAB;
        ge = maxAB.y >= mid.y;
        M = MUT(384, ge, !ge) ? mid : maxAB;
        L = ge ? maxAB : mid;
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
    dy[0] = dyLM; dy[1] = dyMH; dy[2] = dyLH;      /* DxLDy, DxMDy, DxHDy */
    dx[0] = dxLM; dx[1] = dxMH; dx[2] = dxLH;
    cfrac = (in->flags & REJ_YQ) ? MUT(376, 0u, ((uint32_t)(uint16_t)H.y * 0x4000u) & 0xffff) : ((uint32_t)(uint16_t)H.y * 0x4000u) & 0xffff;      /* (the builds that round y to whole pixels do not correct for the fraction of the top vertex's y: the texts have no such code, and the records of stale slots can have one) */
    xHf = (int32_t)H.x * 0x4000;
    Cv = -(int32_t)cfrac;
    st->C0.i = (int16_t)(Cv >> 16);
    st->C0.f = (uint16_t)Cv;
    for (k = 0; k < 3; k++) {
        int32_t Rr = (int32_t)hle_rcp32((int32_t)dy[k]);
        hle_acc a = hle_mac_32xu16(Rr, in->rcp_k);                       /* VMUDL VMADM VMADN with v31[10] = 8 */
        struct fx2_pair A32 = { hle_sclamp(a), hle_uclamp(a) };
        struct fx2_pair dxs = fxs(dx[k] * (int32_t)in->dx_k);                  /* VMUDM / VMADN 0 with v31[13] = 0x4000 */
        hle_acc tt, full;
        struct fx2_pair S;
        mm(dxs, A32, &tt, &full);
        S = rd(tt, full);
        st->slope[k] = S;
        slope_m[k] = S;
    }
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

static void rej_tri_in_f2(const struct rej_tri_in *in, struct fx2_vtx fv[3], struct fx2_tri_in *ti)
{
    unsigned k;
    memset(fv, 0, 3 * sizeof fv[0]);
    for (k = 0; k < 3; k++) {
        const struct rej_vtx *r = in->v[k];
        memcpy(fv[k].col, r->col, 4);
        fv[k].s = r->s;
        fv[k].t = r->t;
        fv[k].sx = r->sx;
        fv[k].sy = r->sy;
        fv[k].szi = r->szi;
        fv[k].szf = r->szf;
        fv[k].rwi = r->rwi;
        fv[k].rwf = r->rwf;
    }
    if (MUT(396, (in->flags & REJ_FLAT_A), 0) && !(in->g & 0x200000))     /* (flat shading: the colour of the first vertex, with F3DFLX its alpha too) */
        for (k = 1; k < 3; k++)
            fv[k].col[3] = fv[0].col[3];
    memset(ti, 0, sizeof *ti);
    for (k = 0; k < 3; k++)
        ti->v[k] = &fv[k];
    ti->g = in->g;
    ti->tex = in->tex;
    ti->tile = in->tile;
}

int rej_tri_setup_core(const struct rej_tri_in *in, uint32_t *cmd, unsigned *len, int neon)
{
    struct fx2_vtx fv[3];
    struct fx2_tri_in ti;
    struct fx2_tri_state st;
    int r;
    rej_tri_in_f2(in, fv, &ti);
    r = rej_tri_prepare(in, fv, cmd, &st);
    if (r != FX2_TRI_DRAW)
        return r;
    if (!((in->g & 4) || in->tex || (in->g & 1)))
        *len = 32;
#ifdef __aarch64__
    else if (neon)
        *len = fx2_tri_attrs_neon(&ti, &st, cmd);
#endif
    else
        *len = fx2_tri_attrs_scalar(&ti, &st, cmd);
    (void)neon;
    return FX2_TRI_DRAW;
}

int rej_tri_setup_scalar(const struct rej_tri_in *in, uint32_t *cmd, unsigned *len)
{
    return rej_tri_setup_core(in, cmd, len, 0);
}

#ifndef __aarch64__
int rej_tri_setup(const struct rej_tri_in *in, uint32_t *cmd, unsigned *len)
{
    return rej_tri_setup_core(in, cmd, len, 0);
}
#endif
