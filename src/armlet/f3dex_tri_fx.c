/*
 * arMLEt - f3dex_tri_fx.c: the F3DEX 1.x / F3DLX triangle setup (scalar reference): culling, vertex order, and the RDP triangle bytes
 *
 * docs/ucode-f3dex.md section 5 (the numbers in the comments are its subsections); the F3DLX variant is marked "F3DLX:". The microcode computes the coefficients for eight lanes at once
 * (lanes 0..3 = r g b a, 4..6 = s t w, 7 = z); only the lanes that are stored matter, so only those are computed.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <string.h>

#include "f3dex_fx_int.h"

static inline int32_t pv(struct hle_pair p) { return hle_pair32(p); }
static inline struct hle_pair mk(int16_t i, uint16_t f) { struct hle_pair p = { i, f }; return p; }
/* floor(A * c / 65536), c unsigned 16-bit: VMUDL VMADM VMADN */
static inline struct hle_pair mul_c(struct hle_pair a, uint16_t c) { return hle_pair_acc(hle_mac_32xu16(pv(a), c)); }
/* A x B with the fraction read after the integer term (a final VMADN 0) */
static inline struct hle_pair mul_fx(struct hle_pair a, struct hle_pair b) { return hle_pair_acc(hle_mac_32x32(pv(a), pv(b))); }
/* A x B with the fraction read before the integer x integer term (the chain has no final VMADN) */
static inline struct hle_pair mul_mid(struct hle_pair a, struct hle_pair b)
{
    hle_acc full = hle_mac_32x32(pv(a), pv(b));
    return mk(hle_sclamp(full), FMUT(100, hle_uclamp(hle_mac_32x32_before_h(pv(a), pv(b))), hle_uclamp(full)));
}
/* x (s16) times a 32-bit pair: VMUDM VMADH VMADN 0 */
static inline struct hle_pair mul_x(int16_t x, struct hle_pair k) { return hle_pair_acc(hle_wrap48((hle_acc)x * pv(k))); }
/* a pair times a signed 16-bit constant: VMUDN VMADH VMADN 0 */
static inline struct hle_pair mul_s(struct hle_pair a, int16_t c) { return hle_pair_acc(hle_wrap48((hle_acc)pv(a) * c)); }
static inline struct hle_pair sub_p(struct hle_pair a, struct hle_pair b) { return hle_sub32(pv(a), pv(b)); }
static inline int16_t sat_sub(int16_t a, int16_t b) { return hle_clamp16((int32_t)a - b); }
static inline struct hle_pair neg_p(struct hle_pair a) { return hle_sub32(0, pv(a)); }
static inline struct hle_pair xpair(int16_t x) { return mk((int16_t)(((int32_t)x * 0x4000) >> 16), (uint16_t)((int32_t)x * 0x4000)); }   /* x (10.2) in 16.16 pixels */
static inline void put16(uint8_t *p, unsigned o, uint16_t v) { p[o] = (uint8_t)(v >> 8); p[o + 1] = (uint8_t)v; }
static inline struct hle_pair maxp(struct hle_pair a, struct hle_pair b) { int32_t r = hle_max32(pv(a), pv(b)); return mk(hle_fx_i(r), hle_fx_f(r)); }
static inline struct hle_pair minp(struct hle_pair a, struct hle_pair b) { int32_t r = hle_min32(pv(a), pv(b)); return mk(hle_fx_i(r), hle_fx_f(r)); }

static int tri_setup(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len, int clip_tests)
{
    const struct f3dex_build *B = in->b;
    const struct f3dex_vtx *V[3] = { in->v[0], in->v[1], in->v[2] };
    const uint32_t g = in->geom;
    const int dlx = B->dlx;

    /* 5.2 early rejection (F3DEX always; F3DLX only with G_CLIPPING), area, culling */
    if (clip_tests && (!dlx || (g & F3DEX_G_CLIPPING))) {
        unsigned f0 = V[0]->flags, f1 = V[1]->flags, f2 = V[2]->flags;
        if (f0 & f1 & f2 & B->reject_mask)
            return F3DEX_TRI_REJECT;
        if ((f0 | f1 | f2) & 0x4343)
            return F3DEX_TRI_CLIP;
    }
    int16_t x[3], y[3];
    for (int i = 0; i < 3; i++) { x[i] = V[i]->sx; y[i] = V[i]->sy; }
    int64_t cross64 = (int64_t)sat_sub(x[2], x[0]) * sat_sub(y[1], y[0]) + (int64_t)sat_sub(x[0], x[1]) * sat_sub(y[2], y[0]);
    int32_t cross = (int32_t)cross64;                                       /* the area pair is the low 32 bits of the accumulator sum */
    int32_t r17 = cross64 > 32767 ? 32767 : cross64 < -32768 ? -32768 : (int32_t)cross64;
    if ((g & FMUT(105, F3DEX_G_CULL_BACK, F3DEX_G_CULL_FRONT)) && r17 < 0)
        return F3DEX_TRI_CULL;
    /* 5.3 vertex order: the table at DMEM 0x360 as logic */
    int idx = 4 * (y[2] < y[1]) + 2 * (y[1] < y[0]) + (y[0] < y[2]);
    static const uint8_t L_[7] = { 2, 2, 0, 2, 1, 1, 0 }, M_[7] = { 1, 1, 2, 0, 0, 2, 1 }, H_[7] = { 0, 0, 1, 1, 2, 0, 2 }, PAR[7] = { 0, 0, 0, 1, 0, 1, 1 };
    if (idx > 6)
        return F3DEX_TRI_CULL;                                              /* cannot occur */
    const struct f3dex_vtx *vH = V[H_[idx]], *vM = V[M_[idx]], *vL = V[L_[idx]];
    struct hle_pair area = mk((int16_t)(cross >> 16), (uint16_t)cross);
    if (FMUT(104, PAR[idx], 0))
        area = neg_p(area);
    if (r17 == 0)
        return F3DEX_TRI_DEGEN;                                             /* degenerate */
    if ((g & F3DEX_G_CULL_FRONT) && r17 >= 0)
        return F3DEX_TRI_CULL;

    /* 5.4 header and edge coefficients */
    const int16_t xH = vH->sx, yH = vH->sy, xM = vM->sx, yM = vM->sy, xL = vL->sx, yL = vL->sy;
    const unsigned flags = g & 7;
    memset(cmd, 0, 32);
    cmd[0] = (uint8_t)((g & 0xff) | 0xc8);
    cmd[1] = (uint8_t)(in->tile_byte | (area.i < 0 ? 0x80 : 0));
    put16(cmd, 2, (uint16_t)yL);
    put16(cmd, 4, (uint16_t)yM);
    put16(cmd, 6, (uint16_t)yH);
    const int16_t dy[3] = { sat_sub(yL, yM), sat_sub(yM, yH), sat_sub(yL, yH) };       /* edges L-M, M-H, L-H */
    const int16_t dx[3] = { sat_sub(xL, xM), sat_sub(xM, xH), sat_sub(xL, xH) };
    const uint16_t c_sub = (uint16_t)(((uint32_t)(uint16_t)yH * 0x4000u) & 0xffff);     /* the fractional row of the top vertex, as a multiplier */
    struct hle_pair slope[3], slope_m[3];
    for (int k = 0; k < 3; k++) {
        uint32_t rc = hle_rcp32(dy[k]);
        struct hle_pair A = mul_c(mk((int16_t)(rc >> 16), (uint16_t)rc), 0x20);
        struct hle_pair Bp = xpair(dx[k]);
        struct hle_pair S = dlx ? mul_mid(A, Bp) : mul_fx(A, Bp);                     /* F3DLX: the chain has no final VMADN 0 */
        S = mul_c(S, 0x4000);                                                          /* divide by 4 */
        int16_t si = S.i;
        if (si > 0x1cc) si = FMUT(106, 0x1cc, 0x1cb);                                                    /* VCR with v30[6] */
        if (si < ~0x1cc) si = ~0x1cc;
        slope[k] = mk(si, S.f);
        slope_m[k] = mul_c(mk(si, (uint16_t)(S.f & FMUT(111, 0xfff8, 0xffff))), c_sub);
    }
    struct hle_pair xMf = xpair(xM), xHf = xpair(xH);
    struct hle_pair XM = sub_p(xHf, slope_m[1]), XH = sub_p(xHf, slope_m[2]);
    if (dlx) { XM = xHf; XH = xHf; }                                                   /* F3DLX: no sub-pixel correction */
    put16(cmd, 8, (uint16_t)xMf.i);    put16(cmd, 10, xMf.f);
    put16(cmd, 12, (uint16_t)slope[0].i); put16(cmd, 14, slope[0].f);
    put16(cmd, 16, (uint16_t)XH.i);    put16(cmd, 18, XH.f);
    put16(cmd, 20, (uint16_t)slope[2].i); put16(cmd, 22, slope[2].f);
    put16(cmd, 24, (uint16_t)XM.i);    put16(cmd, 26, XM.f);
    put16(cmd, 28, (uint16_t)slope[1].i); put16(cmd, 30, slope[1].f);
    unsigned o = 32;
    if (!flags) {
        *len = o;
        return F3DEX_TRI_OK;
    }

    /* 5.5 attribute values for the lanes that are stored: shade lanes 0..3, texture lanes 4..7, z lane 7 */
    const struct f3dex_vtx *vs3[3] = { vH, vM, vL };
    const unsigned need = ((flags & 4) ? 0x0f : 0) | ((flags & 2) ? 0xf0 : 0) | ((flags & 1) ? 0x80 : 0);
    struct hle_pair att[3][8];
    struct hle_pair texmax[3] = { { 0, 0 }, { 0, 0 }, { 0, 0 } };       /* lane 4 / 5 / 6 terms of the texture scale: max |s|, max |t|, half the smallest w */
    if (flags & 2) {
        /* texture preparation: w of the three vertices in command order, scaled by the perspective normalisation; the smallest, halved */
        struct hle_pair w[3];
        for (int i = 0; i < 3; i++)
            w[i] = mul_c(mk(V[i]->cx[3], V[i]->cf[3]), FMUT(109, in->persp, 0xffff));
        struct hle_pair texB = mul_c(minp(minp(w[0], w[1]), w[2]), 0x8000);
        int16_t pi[3][2];
        uint16_t pf[3][2];
        for (int k = 0; k < 3; k++) {
            struct hle_pair kk = mul_mid(mk(vs3[k]->rwi, vs3[k]->rwf), texB);         /* 1/w of the vertex times the scale */
            int16_t xs[3] = { vs3[k]->s, vs3[k]->t, 0x7fff };
            for (int j = 0; j < 3; j++) {
                struct hle_pair r = mul_x(xs[j], kk);
                att[k][4 + j] = r;
                if (j < 2) { pi[k][j] = r.i; pf[k][j] = r.f; }
            }
        }
        /* the largest |s|, |t| (integer half; the fraction of the chosen one, not negated): H against L first, then against M */
        for (int j = 0; j < 2; j++) {
            struct hle_pair cH = mk(hle_abs16(pi[0][j]), pf[0][j]), cM = mk(hle_abs16(pi[1][j]), pf[1][j]), cL = mk(hle_abs16(pi[2][j]), pf[2][j]);
            struct hle_pair t1 = FMUT(110, cH.i >= cL.i, cH.i > cL.i) ? cH : cL;
            texmax[j] = t1.i >= cM.i ? t1 : cM;
        }
        texmax[2] = texB;
    }
    const struct f3dex_vtx *colv[3];
    for (int k = 0; k < 3; k++)
        colv[k] = (g & F3DEX_G_SMOOTH) ? vs3[k] : in->flat;               /* flat shading: all three from the command's first vertex */
    for (int k = 0; k < 3; k++) {
        for (unsigned l = 0; l < 4; l++)
            att[k][l] = mk((int16_t)colv[k]->col[l], 0x8000);
        att[k][7] = mk(vs3[k]->sz, vs3[k]->szf);
    }

    /* deltas */
    const int16_t A4 = hle_clamp16(4 * (int32_t)sat_sub(yM, yH)), B4 = hle_clamp16(4 * (int32_t)sat_sub(yL, yH));
    const int16_t C4 = hle_clamp16(4 * (int32_t)sat_sub(xL, xH)), D4 = hle_clamp16(4 * (int32_t)sat_sub(xM, xH));
    const struct hle_pair Rr = f3dex_recip_refine(pv(area));
    struct hle_pair st[8], dX[8], dY[8], dE[8];
    for (unsigned l = 0; l < 8; l++) {
        if (!(need >> l & 1))
            continue;
        struct hle_pair aH = att[0][l], aM = att[1][l], aL = att[2][l];
        hle_acc acc1 = hle_wrap48((hle_acc)pv(sub_p(aL, aH)) * A4 + (hle_acc)pv(sub_p(aH, aM)) * B4);
        struct hle_pair nx = mk((int16_t)hle_acc_hi(acc1), hle_acc_md(acc1));
        hle_acc acc2 = hle_wrap48((hle_acc)pv(sub_p(aM, aH)) * C4 + (hle_acc)pv(sub_p(aH, aL)) * D4);
        struct hle_pair ny = mk((int16_t)hle_acc_hi(acc2), hle_acc_md(acc2));
        dX[l] = mul_mid(nx, Rr);
        dY[l] = mul_mid(ny, Rr);
        /* DaDe = DaDy + DaDx * DxHDy (the limited integer half, the full fraction) */
        struct hle_pair sl = slope[2];
        hle_acc ta = (hle_acc)pv(dY[l]) + (hle_acc)(((int64_t)dX[l].f * sl.f) >> 16) + (hle_acc)dX[l].i * sl.f + (hle_acc)dX[l].f * sl.i;
        hle_acc tfull = ta + (((hle_acc)dX[l].i * sl.i) << 16);
        dE[l] = mk(hle_sclamp(hle_wrap48(tfull)), hle_uclamp(hle_wrap48(ta)));
        st[l] = dlx ? aH : sub_p(aH, mul_c(dE[l], FMUT(102, c_sub, (uint16_t)(c_sub ^ 1))));                             /* F3DLX: no sub-pixel correction */
    }

    /* shade (flags & 4) */
    if (flags & 4) {
        memset(cmd + o, 0, 64);
        for (int l = 0; l < 4; l++) {
            put16(cmd, o + 2 * l, (uint16_t)st[l].i);        put16(cmd, o + 16 + 2 * l, st[l].f);
            put16(cmd, o + 8 + 2 * l, (uint16_t)dX[l].i);    put16(cmd, o + 24 + 2 * l, dX[l].f);
            put16(cmd, o + 32 + 2 * l, (uint16_t)dE[l].i);   put16(cmd, o + 48 + 2 * l, dE[l].f);
            put16(cmd, o + 40 + 2 * l, (uint16_t)dY[l].i);   put16(cmd, o + 56 + 2 * l, dY[l].f);
        }
        o += 64;
    }
    /* texture (flags & 2): lanes 4..7 times the scale 2 / max(...), limited to 1.0 */
    if (flags & 2) {
        struct hle_pair P[3];
        for (int l = 4; l < 7; l++) {
            struct hle_pair E = mul_c(texmax[l - 4], FMUT(107, 0x800, 0x7ff));
            hle_acc tot = (hle_acc)2 * ((hle_acc)hle_abs16(dX[l].i) * 0x800) + (hle_acc)hle_abs16(dY[l].i) * FMUT(108, 0x800, 0x400) + E.f + (((hle_acc)E.i) << 16);
            hle_acc part = tot - (((hle_acc)E.i) << 16);
            P[l - 4] = mk(hle_sclamp(hle_wrap48(tot)), hle_uclamp(hle_wrap48(part)));
        }
        struct hle_pair M4 = maxp(maxp(P[0], P[1]), maxp(P[2], P[1]));
        struct hle_pair Ms = mul_c(M4, 0x40);
        uint32_t rr = hle_rcp32(pv(Ms));
        hle_acc da = (hle_acc)(uint16_t)rr * 2 + (((hle_acc)(int16_t)(rr >> 16) * 2) << 16);
        int16_t di = hle_sclamp(da);
        uint16_t df = (uint16_t)((hle_acc)(uint16_t)rr * 2);
        struct hle_pair S = di < FMUT(101, 1, 2) ? mk(di, df) : mk(1, 0);
        memset(cmd + o, 0, 64);
        for (int l = 4; l < 8; l++) {
            struct hle_pair a = mul_mid(st[l], S), b = mul_mid(dX[l], S), c = mul_mid(dY[l], S), d = mul_mid(dE[l], S);
            int j = l - 4;
            put16(cmd, o + 2 * j, (uint16_t)a.i);        put16(cmd, o + 16 + 2 * j, a.f);
            put16(cmd, o + 8 + 2 * j, (uint16_t)b.i);    put16(cmd, o + 24 + 2 * j, b.f);
            put16(cmd, o + 40 + 2 * j, (uint16_t)c.i);   put16(cmd, o + 56 + 2 * j, c.f);
            put16(cmd, o + 32 + 2 * j, (uint16_t)d.i);   put16(cmd, o + 48 + 2 * j, d.f);
        }
        o += 64;
    }
    /* z (flags & 1): lane 7 scaled by 32; the start value corrected like the others (F3DLX: not) */
    if (flags & 1) {
        struct hle_pair zx = mul_s(dX[7], FMUT(103, 0x20, 0x21)), zy = mul_s(dY[7], 0x20), ze = mul_s(dE[7], 0x20), zh = mul_s(att[0][7], 0x20);
        struct hle_pair zs = dlx ? zh : sub_p(zh, mul_c(ze, c_sub));
        put16(cmd, o + 0, (uint16_t)zs.i);  put16(cmd, o + 2, zs.f);
        put16(cmd, o + 4, (uint16_t)zx.i);  put16(cmd, o + 6, zx.f);
        put16(cmd, o + 8, (uint16_t)ze.i);  put16(cmd, o + 10, ze.f);
        put16(cmd, o + 12, (uint16_t)zy.i); put16(cmd, o + 14, zy.f);
        o += 16;
    }
    *len = o;
    return F3DEX_TRI_OK;
}

int f3dex_tri_setup_scalar(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len) { return tri_setup(in, cmd, len, 1); }
int f3dex_tri_setup_noclip_scalar(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len) { return tri_setup(in, cmd, len, 0); }

#ifdef __ARM_NEON
#include "f3dex_tri_fx_neon.h"          /* f3dex_tri_setup, f3dex_tri_setup_noclip: the NEON forms */
#else
int f3dex_tri_setup(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len) { return tri_setup(in, cmd, len, 1); }
int f3dex_tri_setup_noclip(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len) { return tri_setup(in, cmd, len, 0); }
#endif
