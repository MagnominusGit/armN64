/*
 * arMLEt - s2dex1_sprite.c: G_OBJ_SPRITE of S2DEX 1.06 / 1.07 (docs/ucode-s2dex1.md, "Sprites") in closed form
 *
 * A sprite (uObjSprite, 24 bytes at DMEM 0x530) through the object matrix (uObjMtx at 0x1d0: A B C D s15.16 as halfword pairs, X Y s10.2 at 0x1e0) becomes SET_TILE + SET_TILE_SIZE
 * and two textured triangles (RDP command 0x0a, 96 bytes each). The microcode (IMEM 0x258 .. 0x64c) keeps a corner in a lane of each vector; here the same numbers come out of
 * integer expressions, one per instruction chain (hle_fx.h's), in the order of the code:
 *
 *   1. the corner rectangle (as G_OBJ_RECTANGLE's width, per axis: the reciprocal of the scale, (size - trim) x it) at the position + bias, four corners in the order of a table
 *      (DMEM 0x2b0 + 16 x the row, a row by the signs of two matrix bytes), the texture coordinates of the corners (the flip flags toggle the table's selectors);
 *   2. the matrix: x' = X + x A + y B, y' = Y + x C + y D (the position is masked by the render mode's table, as the rectangle's);
 *   3. the corners against the scissor: some corner at or past the upper left and some before the lower right on each axis, else nothing is drawn;
 *   4. the corners (0, 1) and (2, 3) ordered by y (swapped when y0 >= y1 / y2 >= y3), then the tile commands (as the rectangle's);
 *   5. the triangles (2, 0, 1) and (0, 1, 3): a vertex is a vector [x x x s t 0x7fff y y], the differences of two vertices give the area N and the gradient numerators (the lanes of the
 *      attributes), the reciprocals of the three edges' dy (with the Newton step) the slopes, the reciprocal of the area the gradients; the sub-pixel start values correct for the
 *      fraction of the top y.
 *
 * Checked against the microcode (the rsp2c translation of 0x258 .. 0x64c, tools/hle/test_s2dex1.c) on random DMEM images, every lane the command stores included.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#define S2_L1 1
#include <stdint.h>
#include <string.h>

#include "hle_fx.h"
#include "s2dex2_fx.h"
#include "s2dex2_int.h"

enum {
    OBJ = 0x530, O_STRIDE = 0x540, O_ADRS = 0x542, O_FMT = 0x544, O_SIZ = 0x545, O_PAL = 0x546, O_FLAGS = 0x547,
    RM_WORDS = L_RM_WORDS, RM_PTR_A = L_RM_PTR_A, RM_PTR_B = L_RM_PTR_B, SCISSOR = L_SCISSOR, TILE_STATE = L_TILE_STATE, TILE_TOGGLE = L_TILE_TOGGLE, SCRATCH = L_SCRATCH,
    MTX = 0x1d0, MTX_XY = 0x1e0, FLIP_TABLE = 0x2b0,
};
enum { CMD_SYNC_TILE = 0x28, CMD_SET_TILE = 0x35, SET_TILE_SIZE_W0 = 0x32000000 };

/* The 1/scale the microcode computes with (as s2dex2_rect.c) */
static inline int32_t refined_recip(int16_t scale)
{
    uint32_t r = hle_rcp32(scale), lo = r & 0xffff, hi = r >> 16;
    uint32_t l = lo + hi;
    int32_t h = sat16((int16_t)hi + (int32_t)(l >> 16));
    return (int32_t)((uint32_t)h << 16 | (l & 0xffff));
}

static inline struct hle_pair mk(int16_t i, uint16_t f) { struct hle_pair p = { i, f }; return p; }
static inline int32_t pv(struct hle_pair p) { return hle_pair32(p); }
/* VMUDL VMADM VMADN: A x c / 65536, c an unsigned 16-bit number */
static inline struct hle_pair mul_c(int32_t a, uint16_t c) { return hle_pair_acc(hle_mac_32xu16(a, c)); }
/* VMUDL VMADM VMADN VMADH (the fraction read before the last term): A x B / 65536 */
static inline struct hle_pair mul_mid(int32_t a, int32_t b)
{
    return mk(hle_sclamp(hle_mac_32x32(a, b)), hle_uclamp(hle_mac_32x32_before_h(a, b)));
}
/* VMUDN (on the fraction) then VMADH (on the integer half): n x A, the fraction half read after the first of the two */
static inline struct hle_pair times_n(struct hle_pair a, int n)
{
    return mk(hle_sclamp(hle_wrap48(((hle_acc)pv(a)) * n)), (uint16_t)hle_uclamp((hle_acc)a.f * n));
}

static void put16(uint8_t *p, unsigned o, unsigned v) { p[o] = (uint8_t)(v >> 8); p[o + 1] = (uint8_t)v; }

/* One triangle: the vertices a, b, c as vectors [x x x s t 0x7fff y y]; 96 bytes */
static void triangle(const int16_t a[8], const int16_t b[8], const int16_t c[8], unsigned tile, uint8_t *o)
{
    int16_t dca[8], dba[8], dab[8], dcb[8];
    int64_t N[8], M[8];
    struct hle_pair Ns, Rn, rf[3], slope[3], sl4[3], xh, xm, gx[8], gy[8], dE[8], st[8];
    int32_t N32, rn32;
    uint16_t K, vcc1;
    unsigned l, k;
    int16_t dy[3], dx[3];
    int r16[3];

    for (l = 0; l < 8; l++) {
        dca[l] = hle_clamp16((int32_t)c[l] - a[l]);
        dba[l] = hle_clamp16((int32_t)b[l] - a[l]);
        dab[l] = hle_clamp16((int32_t)a[l] - b[l]);
        dcb[l] = hle_clamp16((int32_t)c[l] - b[l]);
    }
    /* the numerators: N = (c - a) dy(b - a) + (a - b) dy(c - a); the area is the lane of an x */
    for (l = 0; l < 8; l++)
        N[l] = (int64_t)dca[l] * dba[6] + (int64_t)dab[l] * dca[6];
    N32 = (int32_t)N[2];
    vcc1 = hle_abs16((int16_t)(N[1] >> 16)) < MUT(548, 8, 9);                   /* the scale of the area: small areas are scaled up (lane 1 decides, the x lanes are equal) */
    K = vcc1 ? MUT(549, 0x8000, 0x4000) : 0x0100;
    Ns = mul_c(N32, K);
    rn32 = (int32_t)hle_rcp32(pv(Ns));
    Rn = mk((int16_t)(rn32 >> 16), (uint16_t)rn32);
    (void)Rn;

    /* the three edges: a - c (the long one), a - b, b - c */
    dy[0] = dca[6]; dy[1] = dba[6]; dy[2] = dcb[6];
    dx[0] = dca[2]; dx[1] = dba[2]; dx[2] = dcb[2];
    for (k = 0; k < 3; k++) {
        int32_t R = refined_recip(dy[k]);
        struct hle_pair R2 = mk(hle_sclamp(hle_wrap48((hle_acc)R * 2)), (uint16_t)((uint32_t)(uint16_t)R * 2));       /* 2 R: the fraction before the integer product */
        struct hle_pair T = hle_pair_acc(hle_mac_32xu16(pv(R2), (uint16_t)dy[k]));
        struct hle_pair E = hle_sub32(hle_fx(MUT(550, 2, 3), 0), pv(T));
        rf[k] = mul_mid(pv(R2), pv(E));
        {
            hle_acc acc = hle_wrap48((hle_acc)pv(rf[k]) * dx[k]);                   /* VMUDN VMADH: R x dx, read with VSAR */
            slope[k] = mk((int16_t)hle_acc_hi(acc), hle_acc_md(acc));
        }
        sl4[k] = mul_c(pv(slope[k]), MUT(551, 0x4000, 0x2000));
        r16[k] = slope[k].i >= 0;
    }
    {
        /* the top of the long edge and of the a - b edge at the row of a: x_a - slope x (y_a & 3) / 4 */
        int16_t nfy = (int16_t)-(int)((uint16_t)a[6] & MUT(553, 3, 1));
        struct hle_pair xs[2];
        for (k = 0; k < 2; k++) {
            hle_acc acc = (hle_acc)0x4000 * a[2];
            hle_acc part = hle_wrap48(acc + (hle_acc)sl4[k].f * nfy);
            hle_acc full = hle_wrap48(part + (((hle_acc)sl4[k].i * nfy) << 16));
            xs[k] = mk(hle_sclamp(full), hle_uclamp(part));
        }
        xh = xs[0];
        {
            unsigned f = (unsigned)xs[1].f + (unsigned)MUT(552, r16[0], r16[1]);                           /* +1 in the fraction, by the sign of the long edge's slope */
            xm = mk(hle_clamp16((int32_t)xs[1].i + (f > 0xffff)), (uint16_t)f);
        }
    }

    /* the attributes: s, t, w and the y lane (stored as the fourth halfword) */
    {
        int16_t nfy = (int16_t)-(int)((uint16_t)a[6] & 3);
        int32_t F = (int32_t)nfy * MUT(556, 0x4000, 0x2000);
        for (l = 3; l <= 6; l++) {
            struct hle_pair n = mk((int16_t)(N[l] >> 16), (uint16_t)N[l]), m;
            struct hle_pair gxs, gys;
            int32_t M32;
            M[l] = (int64_t)dba[l] * dca[2] + (int64_t)dca[l] * dab[2];
            M32 = (int32_t)M[l];
            m = mk((int16_t)(M32 >> 16), (uint16_t)M32);
            gxs = mul_mid(pv(n), rn32);
            gys = mul_mid(pv(m), rn32);
            gxs = mul_c(pv(gxs), K);
            gys = mul_c(pv(gys), K);
            gx[l] = times_n(gxs, MUT(554, 8, 4));
            gy[l] = times_n(gys, 8);
            {
                hle_acc p = hle_mac_32x32(pv(gx[l]), pv(slope[0]));
                hle_acc dfr = hle_wrap48(p + MUT(555, gy[l].f, 0));
                hle_acc dfull = hle_wrap48(dfr + (((hle_acc)gy[l].i) << 16));
                dE[l] = mk(hle_sclamp(dfull), hle_uclamp(dfr));
            }
            {
                hle_acc s0 = (hle_acc)a[l] << 16;
                st[l] = mk(hle_sclamp(hle_wrap48(s0 + hle_mac_32x32(pv(dE[l]), F))), hle_uclamp(hle_wrap48(s0 + hle_mac_32x32_before_h(pv(dE[l]), F))));
            }
        }
    }

    /* the command */
    memset(o, 0, 96);
    put16(o, 0, 0x0a00 | tile | (MUT(557, Ns.i < 0, Ns.i <= 0) ? 0x80 : 0));
    put16(o, 2, (uint16_t)c[6]);
    put16(o, 4, (uint16_t)b[6]);
    put16(o, 6, (uint16_t)a[6]);
    {
        uint32_t xl = (uint32_t)(int32_t)b[2] << MUT(558, 14, 13);
        o[8] = (uint8_t)(xl >> 24); o[9] = (uint8_t)(xl >> 16); o[10] = (uint8_t)(xl >> 8); o[11] = (uint8_t)xl;
    }
    put16(o, 12, (uint16_t)slope[2].i); put16(o, 14, slope[2].f);
    put16(o, 16, (uint16_t)xh.i);       put16(o, 18, xh.f);
    put16(o, 20, (uint16_t)slope[0].i); put16(o, 22, slope[0].f);
    put16(o, 24, (uint16_t)xm.i);       put16(o, 26, xm.f);
    put16(o, 28, (uint16_t)slope[1].i); put16(o, 30, slope[1].f);
    for (l = 3; l <= 6; l++) {
        unsigned j = 2 * (l - 3);
        put16(o, 32 + j, (uint16_t)st[l].i); put16(o, 40 + j, (uint16_t)gx[l].i);
        put16(o, 48 + j, st[l].f);           put16(o, 56 + j, gx[l].f);
        put16(o, 64 + j, (uint16_t)dE[l].i); put16(o, 72 + j, (uint16_t)gy[l].i);
        put16(o, 80 + j, dE[l].f);           put16(o, 88 + j, gy[l].f);
    }
}

int s2dex1_obj_sprite(uint32_t *dm, unsigned *s7p)
{
    unsigned s7 = *s7p, l, g;
    unsigned ptr_a = m16(dm, RM_PTR_A), ptr_b = m16(dm, RM_PTR_B);
    const int bt = (int16_t)m16(dm, RM_WORDS), tr = (int16_t)m16(dm, RM_WORDS + 2), b3 = (int16_t)m16(dm, RM_WORDS + 6);
    int mask, bs, bt2, row[8], sel[8], dim[2], width[2], xc[4], yc[4], xo[4], yo[4], tc[8], base[2];
    const unsigned flags = m8(dm, O_FLAGS);
    int16_t va[8], vb[8], vc[8], vd[8];
    int hit;
    unsigned at;
    int bx, by, b3t;
    unsigned tile;

    if (memcmp(dm + L_CONSTS / 4, k_consts, sizeof k_consts) || ((ptr_a | ptr_b) & 7))
        return S2_UNSUPPORTED;
    mask = (int16_t)m16(dm, ptr_a);                              /* (the table A lanes are loaded into a vector register first: the scratch writes below may be over the table) */
    bs = (int16_t)m16(dm, ptr_a + 4);
    b3t = (int16_t)m16(dm, ptr_a + 6);
    bt2 = hle_clamp16(bt + bs);
    /* the table row: the signs of the matrix bytes 0x1d8 and 0x1dc pick it */
    {
        int v0 = (m8(dm, MTX + 8) ^ (unsigned)(int32_t)(int8_t)m8(dm, MTX + 12)) & MUT(543, 0x180, 0x100);
        at = FLIP_TABLE + (unsigned)(v0 >> 3);
    }
    for (l = 0; l < 8; l++)
        row[l] = (int16_t)m16(dm, at + 2 * l);
    for (g = 0; g < 2; g++) {
        const unsigned o = OBJ + 8 * g;
        const int pos = (int16_t)m16(dm, o), scale = (int16_t)m16(dm, o + 2), size = (int16_t)m16(dm, o + 4);
        const int32_t rr = refined_recip((int16_t)scale);
        const uint16_t rl = (uint16_t)rr, rh = (uint16_t)((uint32_t)rr >> 16);
        const int v7 = hle_clamp16(size - tr);
        hle_acc acc = (hle_acc)(((uint32_t)(uint16_t)v7 * rl) >> MUT(540, 16, 15)) + (hle_acc)v7 * rh;
        struct hle_pair q = hle_pair_acc(acc);
        hle_acc acc2 = (hle_acc)q.i * MUT(541, 0x100, 0x80) + (((hle_acc)q.f * 0x100) >> 16);
        dim[g] = v7;
        width[g] = (int16_t)hle_uclamp(acc2);
        base[g] = hle_clamp16(pos + b3);
    }
    for (l = 0; l < 4; l++) {
        xc[l] = hle_clamp16(base[0] + row[l] * width[0]);
        yc[l] = hle_clamp16(base[1] + row[4 + l] * width[1]);
    }
    for (l = 0; l < 8; l++) {
        const int f = (flags >> (l < 4 ? 0 : MUT(542, 4, 0))) & 1;
        sel[l] = (int16_t)(row[l] ^ f);
        tc[l] = hle_clamp16(bt2 + sel[l] * dim[l >> 2]);
    }
    for (l = 0; l < 4; l++) {                                    /* the scratch: SQV of the corner rectangle */
        w16(dm, SCRATCH + 2 * l, (unsigned)xc[l] & MUT(561, 0xffff, 0));
        w16(dm, SCRATCH + 8 + 2 * l, (unsigned)yc[l] & 0xffff);
    }
    bx = hle_clamp16((int16_t)m16(dm, MTX_XY) + b3t) & mask;
    by = hle_clamp16((int16_t)m16(dm, MTX_XY + 2) + b3t) & mask;
    {
        const int32_t A = (int32_t)m32(dm, MTX), B = (int32_t)m32(dm, MTX + 4), C = (int32_t)m32(dm, MTX + 8), D = (int32_t)m32(dm, MTX + 12);
        for (l = 0; l < 4; l++) {
            hle_acc ax = ((hle_acc)(int16_t)bx << 16) + (hle_acc)xc[l] * A + (hle_acc)yc[l] * MUT(544, B, C);
            hle_acc ay = ((hle_acc)(int16_t)by << 16) + (hle_acc)xc[l] * C + (hle_acc)yc[l] * D;
            xo[l] = hle_sclamp(hle_wrap48(ax));
            yo[l] = hle_sclamp(hle_wrap48(ay));
        }
    }
    {
        const int ulx = (int16_t)m16(dm, SCISSOR), lrx = (int16_t)m16(dm, SCISSOR + 2), uly = (int16_t)m16(dm, SCISSOR + 4), lry = (int16_t)m16(dm, SCISSOR + 6);
        hit = 0;
        for (l = 0, g = 0; l < 4; l++) g |= MUT(545, xo[l] >= ulx, xo[l] > ulx);
        hit = g;
        for (l = 0, g = 0; l < 4; l++) g |= yo[l] >= uly;
        hit = hit && g;
        for (l = 0, g = 0; l < 4; l++) g |= xo[l] < lrx;
        hit = hit && g;
        for (l = 0, g = 0; l < 4; l++) g |= yo[l] < lry;
        hit = hit && g;
        if (!hit)
            return S2_CLIPPED;                                   /* (SQV's 16 bytes are written: the caller keeps them) */
    }
    for (l = 0; l < 4; l++)
        w16(dm, SCRATCH + 2 * l, (unsigned)yo[l] & 0xffff);      /* SDV of the y lanes over the first half */

    /* pairs (0, 1) and (2, 3) ordered by y */
    for (g = 0; g < 4; g += 2) {
        if (!MUT(546, yo[g] < yo[g + 1], yo[g] <= yo[g + 1])) {
            int t;
            t = xo[g]; xo[g] = xo[g + 1]; xo[g + 1] = t;
            t = yo[g]; yo[g] = yo[g + 1]; yo[g + 1] = t;
            t = tc[g]; tc[g] = tc[g + 1]; tc[g + 1] = t;
            t = tc[4 + g]; tc[4 + g] = tc[4 + g + 1]; tc[4 + g + 1] = t;
        }
    }

    /* the tile commands (as G_OBJ_RECTANGLE's) */
    {
        const unsigned toggle = m8(dm, TILE_TOGGLE) ^ 2u;
        const uint32_t state = (uint32_t)(int32_t)(int8_t)m8(dm, TILE_STATE);
        const uint32_t tile_attr = m32(dm, ptr_b + 4);
        unsigned imageW = m16(dm, OBJ + 4), imageH = m16(dm, OBJ + 12);
        uint32_t w0, w1;

        tile = toggle;
        w8(dm, TILE_TOGGLE, tile);
        w8(dm, TILE_STATE, tile);
        w8(dm, s7, CMD_SYNC_TILE);
        if (tile == state)
            s7 += 8;
        w0 = (uint32_t)m8(dm, O_FMT) << 21 | (uint32_t)m8(dm, O_SIZ) << 19 | (uint32_t)m16(dm, O_STRIDE) << 9 | m16(dm, O_ADRS);
        w32(dm, s7, CMD_SET_TILE << 24 | (w0 & 0xffffff));
        w1 = (uint32_t)m8(dm, O_PAL) << 20 | tile_attr;
        w32(dm, s7 + 4, tile << 24 | (w1 & 0xffffff));
        w32(dm, s7 + 8, SET_TILE_SIZE_W0);
        w1 = ((imageW & 0x7ff8u) << 9 | imageH >> 3) - 16388u;
        w32(dm, s7 + 12, tile << 24 | (w1 & 0xffffff));
        if (MUT(560, 1, 0))
            w8(dm, s7 + 16, 0x27);                               /* (a SYNC_PIPE: one byte) */
        s7 += 24;
    }

    /* the vertices: corner 2 (a of the first triangle), 0, 1, 3 */
    {
        static const unsigned ord[4] = { 2, MUT(559, 0, 1), MUT(559, 1, 0), 3 };
        int16_t *v[4] = { va, vb, vc, vd };
        unsigned i;
        for (i = 0; i < 4; i++) {
            const unsigned c = ord[i];
            v[i][0] = v[i][1] = v[i][2] = (int16_t)xo[c];
            v[i][3] = (int16_t)tc[c];
            v[i][4] = (int16_t)tc[4 + c];
            v[i][5] = 0x7fff;
            v[i][6] = v[i][7] = (int16_t)yo[c];
        }
    }
    {
        uint8_t cmd[96];
        unsigned i;
        triangle(va, vb, vc, tile, cmd);                         /* (corner 2, 0, 1) */
        for (i = 0; i < 96; i += 4)
            w32(dm, s7 + i, (uint32_t)cmd[i] << 24 | (uint32_t)cmd[i + 1] << 16 | (uint32_t)cmd[i + 2] << 8 | cmd[i + 3]);
        s7 += 96;
        triangle(vb, vc, vd, tile, cmd);                         /* (corner 0, 1, 3) */
        for (i = 0; i < 96; i += 4)
            w32(dm, s7 + i, (uint32_t)cmd[i] << 24 | (uint32_t)cmd[i + 1] << 16 | (uint32_t)cmd[i + 2] << 8 | cmd[i + 3]);
        s7 += 96;
    }
    *s7p = s7;
    return S2_DONE;
}
