/*
 * arMLEt - f3dex_vtx_fx.c: the F3DEX 1.x / F3DLX vertex pipeline (scalar reference)
 *
 * One vertex at a time as lane arithmetic on its four lanes; docs/ucode-f3dex/light.md (the numbers in the comments are its sections). The microcode processes two vertices per iteration in the two
 * halves of its vector registers; the halves never mix except through the F3DEX.NoN clip flags (handled in f3dex_vtx_xform_scalar) and a stale register that this model declines.
 * The sums are hle_fx.h's closed forms of the multiply-accumulate chains; the places where the microcode reads the low word of a chain before the last product is added are marked "lo3".
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <string.h>

#include "f3dex_fx_int.h"
#include "f3dex_dmem.h"

/* the DMEM layout of the F3DEX 1.2x builds (docs/ucode-f3dex.md section 3); the others are in f3dex.c */
const struct f3dex_layout f3dex_lay_std = { .jt0 = 0xc0, .jt2 = 0x76, .half1 = 0x10c, .persp = 0x112, .geom = 0x114, .omh = 0x118, .oml = 0x11c, .texture = 0x120, .rdp_wp = 0x128, .lights = 0x12c,
    .mtx_sp = 0x134, .mtx_lim = 0x15c, .depth = 0x148, .fifo = 0x150, .segs = 0x160, .light = 0x1a0, .mm_tab = 0x270, .mw_tab = 0x30e, .vtab = 0x31e, .viewport = 0x380, .fog = 0x390,
    .retstack = 0x398, .mv = 0x3e0, .proj = 0x420, .mvp = 0x460, .vbuf = 0x4a0, .window = 0x9a0, .operand = 0xae0, .outbuf = 0xce0,
    .lim = 0x28, .tgbias = 0x38, .vsign = 0x50, .clip_k = 0x76, .planes = 0x98, .tg = 0xa2, .plane_masks = 0xb2 };

const struct f3dex_build f3dex_build_dex = { 0x7070, 0, 0, NULL };
const struct f3dex_build f3dex_build_non = { 0x7030, 1, 0, NULL };
const struct f3dex_build f3dex_build_dlx = { 0x7070, 0, 1, NULL };
const struct f3dex_build f3dex_build_dlx_non = { 0x7030, 1, 1, NULL };

const struct f3dex_build *f3dex_build_of_ucode(uint64_t id)
{
    switch (id) {
    case 0x208a567e9aead5eaull:     /* F3DEX 1.21 */
    case 0x047252e810e35d56ull:     /* F3DEX 1.23 (and Pokemon Puzzle League's binary) */
    case 0x34dad67335a54cdaull:     /* F3DEX 0.95 */
        return &f3dex_build_dex;
    case 0xfd5a31ccf6d40b62ull:     /* F3DEX.NoN 1.22 */
        return &f3dex_build_non;
    case 0x7502cfb7098eeffbull:     /* F3DLX 1.23 */
    case 0x5f33f7c2be924de0ull:     /* F3DLX 0.95 */
    case 0x76c763bc18ad3ab3ull:     /* F3DLX 1.21 */
        return &f3dex_build_dlx;
    case 0x0a7cc6c93363c663ull:     /* F3DLX.NoN 1.21 */
    case 0xcc09096cb6a48e7bull:     /* F3DLX.NoN 1.23 */
        return &f3dex_build_dlx_non;
    case 0x5063f108118bf62aull:     /* F3DEX.NoN 1.00 */
        return &f3dex_build_non;
    }
    return 0;
}

static inline int clip_on(const struct f3dex_vctx *c) { return !c->b->dlx || (c->geom & F3DEX_G_CLIPPING); }
static inline int16_t vmulf16(int16_t a, int16_t b) { return hle_sclamp(hle_mulf(a, b)); }

/* ---- clip flags (5.1) ------------------------------------------------------------------------------------------------------------------------------------------------------------ */
void f3dex_vtx_flagparts(const struct f3dex_vctx *c, const int16_t Pi[4], const uint16_t Pf[4], struct f3dex_flagparts *fp)
{
    int32_t W = hle_fx(Pi[3], Pf[3]);
    /* W2 = ratio * W: the S15.16 product with the integer half saturating, the fraction the low 16 bits */
    hle_acc a = hle_wrap48((hle_acc)Pf[3] * FMUT(14, c->clip_k, 1) + (((hle_acc)Pi[3] * FMUT(14, c->clip_k, 1)) << 16));
    int32_t W2 = hle_fx(hle_sclamp(a), (uint16_t)a);
    unsigned f1 = 0, f2 = 0;
    for (int k = 0; k < 3; k++) {
        int32_t X = hle_fx(Pi[k], Pf[k]);
        unsigned la = hle_clip_le(X, W), ha = hle_clip_ge(X, W);
        unsigned lb = hle_clip_le(X, W2), hb = hle_clip_ge(X, W2);
        f1 |= ((c->b->non && k == 2) ? 0 : la) << k | ha << (8 + k);        /* NoN: no "z <= -w" bit in the first compare */
        f2 |= lb << k | hb << (8 + k);
    }
    fp->f1 = f1;
    fp->f2 = f2;
    fp->wneg = Pi[3] < 0;
}

/* ---- the tail: 1/w, projection, viewport (5.1) ---------------------------------------------------------------------------------------------------------------------------------- */
void f3dex_vtx_tail(const struct f3dex_vctx *c, const int16_t Pi[4], int16_t wi, const uint16_t Pf[4], struct f3dex_vtx *o, struct f3dex_tail_out *t)
{
    int32_t W = hle_fx(wi, Pf[3]);
    hle_acc wa = hle_mac_32xu16(W, FMUT(13, c->persp, 0xffff));                              /* VMUDL VMADM */
    struct hle_pair R = f3dex_recip_refine(hle_pair32(hle_pair_acc(wa)));
    o->rwi = R.i;
    o->rwf = R.f;
    int32_t Rc = hle_fx(R.i >= 0 ? R.i : 0x7fff, R.f);
    for (int k = 0; k < 3; k++) {
        int32_t P = hle_fx(Pi[k], Pf[k]);
        hle_acc a = hle_mac_32x32(P, Rc);                                   /* VMUDL VMADM VMADN VMADH */
        uint16_t q0f = FMUT(4, hle_uclamp(hle_mac_32x32_before_h(P, Rc)), hle_uclamp(a));   /* lo3; mutation 4: the full sum */
        int16_t q0i = hle_sclamp(a);
        hle_acc b = hle_mac_32xu16(hle_fx(q0i, q0f), c->persp);            /* times the perspective normalisation */
        int16_t qi = hle_sclamp(b);
        uint16_t qf = hle_uclamp(b);
        int16_t sc = hle_clamp16((int32_t)c->vscale[k] * FMUT(15, c->vsign[k], 1));
        hle_acc v = hle_wrap48(((hle_acc)c->vtrans[k] << 16) + (hle_acc)hle_fx(qi, qf) * sc);
        int16_t s = hle_sclamp(v);
        int16_t lo = k < 2 ? c->lim[1] : c->lim[3], hi = k < 2 ? c->lim[0] : c->lim[2];
        if (s < lo) s = lo;
        if (s > hi) s = hi;
        if (k == 0) o->sx = s;
        else if (k == 1) o->sy = c->b->dlx ? (int16_t)((s + 2) & FMUT(9, 0xfffc, 0xfff8)) : s;
        else {
            o->sz = s;
            o->szf = hle_uclamp(v);
            if (t) { t->qzi = qi; t->qzf = qf; }
        }
    }
}

/* ---- fog (5.5) ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
uint8_t f3dex_fog_byte(const struct f3dex_vctx *c, int16_t qzi, uint16_t qzf)
{
    hle_acc a = hle_wrap48((hle_acc)hle_fx(qzi, qzf) * c->fog[0]);         /* VMUDN VMADH */
    int16_t f = hle_clamp16((int32_t)hle_sclamp(a) + c->fog[1]);
    if (f < 0) f = 0;
    if (f > c->fog[2]) f = c->fog[2];
    return (uint8_t)FMUT(1, f, f + 1);
}

/* ---- one vertex --------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
struct vinfo {
    int wneg;
};

/* h = 0 / 1: the first / second vertex of a pair (bytes 4h .. 4h+3 of the light record rows); first = the vertex belongs to the first pair of the call */
static void vtx_one(const struct f3dex_vctx *c, const struct f3dex_vin *in, unsigned h, int first, struct f3dex_vtx *o, struct vinfo *vi)
{
    const int clip = clip_on(c), lit = (c->geom & F3DEX_G_LIGHTING) != 0;
    int16_t Pi[4], P21[4];
    uint16_t Pf[4];
    const int16_t px[3] = { in->x, in->y, in->z };
    /* clip coordinates: M is s15.16, the rows 0..2 times x y z plus row 3; the low word is read before the integer part of row 3 is added (lo3) */
    for (int k = 0; k < 4; k++) {
        hle_acc acc = 0;
        for (int j = 0; j < 3; j++)
            acc += (hle_acc)hle_fx(c->mp.i[j][k], c->mp.f[j][k]) * px[j];
        acc = hle_wrap48(acc + c->mp.f[3][k]);
        Pf[k] = hle_uclamp(acc);
        Pi[k] = hle_sclamp(hle_wrap48(acc + ((hle_acc)c->mp.i[3][k] << 16)));
    }
    /* F3DLX without G_CLIPPING: v21 = v29 + the VCO carry (VCL clears it in the normal path) */
    for (int k = 0; k < 4; k++)
        P21[k] = (clip || !first || lit) ? Pi[k] : hle_clamp16((int32_t)Pi[k] + FMUT(10, (c->vco_carry >> (4 * h + k)) & 1, 0));
    struct f3dex_flagparts fp;
    f3dex_vtx_flagparts(c, Pi, Pf, &fp);
    vi->wneg = fp.wneg;
    o->flags = clip ? (uint16_t)(fp.f2 | fp.f1 << 4) : 0;
    struct f3dex_tail_out t;
    f3dex_vtx_tail(c, P21, Pi[3], Pf, o, &t);
    for (int k = 0; k < 4; k++) {
        o->cx[k] = P21[k];
        o->cf[k] = Pf[k];
    }
    /* colour and texture coordinates */
    uint8_t col[4];
    memcpy(col, in->c, 4);
    int16_t st[2] = { in->s, in->t };
    if (lit) {
        int16_t n[3] = { (int16_t)((int8_t)in->c[0] << 8), (int16_t)((int8_t)in->c[1] << 8), (int16_t)((int8_t)in->c[2] << 8) };
        int16_t acc[3] = { 0, 0, 0 }, cur[3];
        for (int j = 0; j < 3; j++)
            cur[j] = (int16_t)(c->lcol[c->nrec + 1][4 * h + j] << 7);       /* the ambient colour, added first */
        int16_t D = 0, Dprev = 0, col0[3] = { 0, 0, 0 };
        for (int k = (int)c->nrec; k >= 0; k--) {
            for (int j = 0; j < 3; j++)
                acc[j] = hle_clamp16((int32_t)acc[j] + (FMUT(12, cur[j] > 0, 1) ? cur[j] : 0));
            int16_t p[3];
            for (int j = 0; j < 3; j++)
                p[j] = vmulf16(n[j], (int16_t)(c->ldir[k][4 * h + j] << 8));
            Dprev = D;
            D = hle_clamp16((int32_t)hle_clamp16((int32_t)p[0] + p[1]) + p[2]);
            for (int j = 0; j < 3; j++) {
                col0[j] = (int16_t)(c->lcol[k][4 * h + j] << 7);
                cur[j] = vmulf16(col0[j], D);
            }
        }
        for (int j = 0; j < 3; j++)
            col[j] = (uint8_t)(acc[j] >> 7);
        if (c->geom & F3DEX_G_TEXGEN) {
            for (int j = 0; j < 2; j++) {
                hle_acc a = hle_mulf(col0[j], D);                              /* the accumulator the last VMULF left: 2 col0 D + 0x8000 */
                a = hle_wrap48(a + (hle_acc)(uint16_t)(c->lk[4 * h + j] << 8) * Dprev);    /* VMADN: unsigned weight x signed dot product */
                int16_t v = hle_sclamp(a);
                if (c->geom & F3DEX_G_TEXGEN_LINEAR) {
                    int16_t sq = vmulf16(v, v);
                    int16_t cu = vmulf16(sq, v);
                    hle_acc b = (hle_acc)2 * cu * c->tg[0] + 0x8000;
                    b += (hle_acc)2 * cu * FMUT(11, c->tg[2], c->tg[1]);
                    b += (hle_acc)2 * v * c->tg[1];
                    v = hle_sclamp(hle_wrap48(b));
                }
                st[j] = hle_clamp16((int32_t)v + FMUT(2, c->tgbias, c->tgbias + 1));
            }
        }
    }
    memcpy(o->col, col, 4);
    o->s = hle_sclamp((hle_acc)st[0] * c->tscale[0]);                     /* VMUDM: (s * scale) >> 16 */
    o->t = hle_sclamp((hle_acc)st[1] * c->tscale[1]);
    if (c->geom & F3DEX_G_FOG)
        o->col[3] = f3dex_fog_byte(c, t.qzi, t.qzf);
}

int f3dex_vtx_xform_scalar(const struct f3dex_vctx *c, const struct f3dex_vin *in, unsigned n, struct f3dex_vtx *out)
{
    if (n == 0 || n > 32)
        return 1;
    if (c->geom & F3DEX_G_LIGHTING) {
        if (c->nrec > F3DEX_MAX_REC || ((c->geom & F3DEX_G_TEXGEN) && c->nrec == 0))
            return 1;
    }
    struct vinfo vi[32];
    for (unsigned i = 0; i < n; i++)
        vtx_one(c, &in[i], FMUT(7, i & 1, 0), i < 2, &out[i], &vi[i]);
    /* F3DEX.NoN: the sign of w of both vertices of a pair goes into the second vertex's flags (bits 2 and 6) */
    if (c->b->non && clip_on(c))
        for (unsigned i = 0; i + 1 < n; i += 2)
            out[i + 1].flags |= (uint16_t)(vi[i].wneg << FMUT(8, 2, 3) | vi[i + 1].wneg << 6);
    return 0;
}

#ifdef __ARM_NEON
#include "f3dex_vtx_fx_neon.h"          /* f3dex_vtx_xform: the NEON form */
#else
int f3dex_vtx_xform(const struct f3dex_vctx *c, const struct f3dex_vin *in, unsigned n, struct f3dex_vtx *out) { return f3dex_vtx_xform_scalar(c, in, n, out); }
#endif
