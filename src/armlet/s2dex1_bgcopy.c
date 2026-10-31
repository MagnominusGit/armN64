/*
 * arMLEt - s2dex1_bgcopy.c: G_BG_COPY of S2DEX 1.06 (docs/ucode-s2dex1.md, section 4) in closed form; 1.07 has the 2.x routine (s2dex2_bg.c, S2_L1)
 *
 * The routine is the one of s2dex2_bg.c (s2dex2_obj_bgcopy) with four differences, all found by reading the two texts side by side (0x968 .. 0xc80 of the 1.x text against 0x210 .. 0x554 of Worms'):
 *   - the DMEM layout (s2dex2_int.h, S2_L1: the object block 0x170 higher, the scratch word at 0x1b0, the scissor at 0xc0, the format tables at 0x260);
 *   - a mirrored background's y origin is `(v10 << 2) & 0xffff` where 2.x has `(v10 << 2) - 1`, and the 32-bit product that moves the row pointer of the last strip is its low 16 bits (the lane of VMUDN is taken
 *     with MFC2, 2.x stores the product's two halves through DMEM), no `(sp << 16) | lhu` for the mirrored case;
 *   - the strip is written in another order: SET_TEXTURE_IMAGE, SYNC_LOAD, the load, SYNC_PIPE, TEXTURE_RECTANGLE (2.x: SYNC_LOAD first), 48 bytes;
 *   - the second pass (the rows past the image's last row) is 80 bytes: SET_TEXTURE_IMAGE, SYNC_LOAD, the load, SYNC_PIPE, SET_TEXTURE_IMAGE, SET_TILE, the load, SYNC_PIPE, TEXTURE_RECTANGLE; with no SET_TILE in front of the first load.
 * The rest (the vector part, the three setup commands, the image address and the visible rectangle) is the same arithmetic. Checked against the microcode (rsp2c translation, tools/hle/test_s2dex1.c).
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

static inline int32_t max0(int32_t v) { return v < 0 ? 0 : v; }

enum {
    OBJ = 0x520,                            /* the uObjBg: imageX imageW frameX frameW imageY imageH frameY frameH */
    IMAGE_PTR = 0x530,
    B_B4 = 0x534, B_B5 = 0x535, IMG_FMT = 0x536, IMG_SIZ = 0x537, B_PAL = 0x539, FLIP = 0x53b, SCALE = 0x53c, S5 = 0x53e, B_V = 0x544, B_V2 = 0x546, B_O0 = 0x540, B_O2 = 0x542,
    B_H = 0x52a,                            /* imageH of the object's y group */
    B_TAB = 0x260,                          /* per texel size: 4 halfwords */
    SCRATCH = 0x1b0, SCISSOR = 0xc0, TILE_W1 = 0x24c,
};
#define BG_DOMAIN(v) ((v) < -0x2000 || (v) >= 0x2000)

int s2dex1_obj_bgcopy106(uint32_t *dm, uint32_t *s7p, uint32_t *s6p, const struct s2_ctx *c)
{
    int32_t L[8], lo[2], hi[2], cut[2], ex[2], vis[2], F[2], W[2], tpos[2], yp, zx;
    uint32_t s7 = *s7p, s6 = *s6p;
    uint32_t at, v0, v1, a0, a1, a2, a3, t0, t1, t2, t3, t4, t5, t6, t7, s0, s5, sq, gp, t8;
    unsigned k, g, flip, Q, V, T[4], Zu, fmtb;
    uint32_t v10_0, v8_0;
    int64_t acc1;

    if (memcmp(dm + L_CONSTS / 4, k_consts, sizeof k_consts))
        return S2_UNSUPPORTED;
    for (k = 0; k < 8; k++) {
        L[k] = (int16_t)m16(dm, OBJ + 2 * k);
        if (BG_DOMAIN(L[k]) && (k & 3))                 /* (the image positions, u10.5, are only shifted and masked) */
            return S2_UNSUPPORTED;
    }
    for (g = 0; g < 2; g++) {
        lo[g] = (int16_t)m16(dm, SCISSOR + 4 * g);
        hi[g] = (int16_t)m16(dm, SCISSOR + 4 * g + 2);
        if (BG_DOMAIN(lo[g]) || BG_DOMAIN(hi[g]))
            return S2_UNSUPPORTED;
        F[g] = L[4 * g + 2];
        W[g] = L[4 * g + 3];
        cut[g] = max0(lo[g] - F[g] + MUT(500, 0, 1));
        ex[g] = max0(F[g] + W[g] - hi[g] + MUT(501, 0, 1));
        vis[g] = W[g] - cut[g] - ex[g];
    }
    if (MUT(502, vis[0] <= 0 || vis[1] <= 0, vis[0] < 0 || vis[1] < 0))                     /* outside the scissor: nothing is written */
        return S2_DONE;

    /* the image position of the first visible pixel, whole pixels in quarter pixels, wrapped once into the image (x: also the start of the row) */
    for (g = 0; g < 2; g++) {
        tpos[g] = (int16_t)((L[4 * g] >> MUT(503, 3, 2)) & 0xfffc) + cut[g];
        if (tpos[g] > 32767)                            /* (VSUB would saturate) */
            return S2_UNSUPPORTED;
    }
    {
        int32_t pre = tpos[1] - L[5];
        yp = (MUT(504, pre >= 0, pre > 0) ? pre : tpos[1]) >> 2;
    }
    flip = m8(dm, FLIP);
    zx = flip ? sat16(tpos[0] + ex[0]) : tpos[0];
    Zu = (unsigned)zx & 0xffff;
    fmtb = m8(dm, IMG_SIZ);
    if (fmtb > 3)                                       /* (the table of 4 rows at 0x260; past it the lookup reads the microcode's other tables) */
        return S2_UNSUPPORTED;
    for (k = 0; k < 4; k++)
        T[k] = m16(dm, B_TAB + 8 * fmtb + 2 * k);
    V = m16(dm, B_V);
    Q = m16(dm, SCALE);

    /* ---- the three setup commands ---- */
    w32(dm, s7, 0x35100000u | ((uint32_t)(int32_t)(int8_t)m8(dm, B_B4) & (Q << 9)));
    w32(dm, s7 + 4, 0x27000000u);
    w32(dm, s7 + 8, 0x32000000u);
    w32(dm, s7 + 12, 0);
    w32(dm, s7 + 16, (((m8(dm, IMG_FMT) << 2 | m8(dm, IMG_SIZ)) << 19 | (Q << 9)) & 0xffffff) | 0x35000000u);
    w32(dm, s7 + 20, m32(dm, TILE_W1) | m8(dm, B_PAL) << 20);
    s7 += 24;

    /* ---- the image address and the visible rectangle ---- */
    v8_0 = T[0] & Zu;
    v10_0 = (v8_0 + (unsigned)vis[0]) & 0xffff;
    acc1 = (int64_t)((T[1] * Zu) >> 16) + (int64_t)V * yp;
    w16(dm, SCRATCH + 2, hle_uclamp(acc1));
    w16(dm, SCRATCH, (uint16_t)hle_sclamp(acc1));
    t8 = c->seg(c->ctx, m32(dm, IMAGE_PTR));
    v0 = ((m32(dm, SCRATCH) >> 1) << 3) + t8;
    v1 = t8;
    a3 = (uint32_t)(int32_t)(int16_t)((T[1] * Zu) >> 16);
    {
        int32_t v3x = sat16(F[0] + cut[0]), v3y = sat16(F[1] + cut[1]);
        t0 = (uint32_t)v3x;
        t1 = (uint32_t)v3y;
        t2 = (uint32_t)(int32_t)(int16_t)(v3x + vis[0] - 1);
        at = (uint32_t)vis[1];
    }
    a1 = m16(dm, B_H) & 0xfffc;
    a2 = (uint32_t)yp << 2;
    a0 = a1 - a2;
    t0 <<= 12;
    if ((int32_t)a0 <= 0)
        goto done;
    t2 <<= 12;
    t4 = 0xe4000000u;
    if ((int32_t)a3 > 0)
        a0 -= 4;
    if ((int32_t)(a0 - at) > 0)
        a0 = at;
    t2 |= t4;
    t7 = (uint32_t)(int32_t)(int8_t)m8(dm, B_B5);
    t5 = m16(dm, B_V);
    a2 = m16(dm, B_O2);
    a1 = m16(dm, B_O0);
    if ((int32_t)t7 < 0)
        a1 = (((uint32_t)(int32_t)(int16_t)((T[2] * v10_0) >> 16)) << 2) & MUT(510, 0xffffu, 0xffffffffu);
    t6 = ((t5 << 1) - 1) | 0x3d100000u;
    t7 <<= 24;
    a1 = (a1 | 0x7000u) << 12;
    {
        int do_loop = (int32_t)a0 > 0;
        uint32_t v8out;
        at -= a0;
        {
            uint32_t v8f = flip ? (uint32_t)sat16(1 - (int16_t)v10_0) : v8_0;
            v8out = (v8f * 8) & 0xffff;
        }
        for (;;) {
            if (do_loop) {
                sq = m16(dm, B_V2);
                s5 = m16(dm, S5);
                s0 = a1 | a2;
                do {
                    a0 -= s5;
                    if ((int32_t)a0 < 0) {
                        int64_t P = (int64_t)V * (int16_t)a0;
                        t3 = (uint32_t)(int32_t)(int16_t)(uint16_t)P;         /* MFC2 of the VMUDN lane: its low 16 bits */
                        s5 += a0;
                        sq += MUT(511, t3, 0);
                        if ((int32_t)t7 >= 0) {
                            t3 = ((sq - 2) | 0xe000u) << 11;
                            s0 = t3 | a2;
                        } else {
                            t3 = s5 - 1;
                            s0 = t3 | a1;
                        }
                    }
                    gp = s5 + t1 - 1;
                    w32(dm, s7, t6);
                    w32(dm, s7 + 4, v0);
                    w8(dm, s7 + 8, 0x26);
                    w32(dm, s7 + 16, t7);
                    w32(dm, s7 + 20, s0);
                    w8(dm, s7 + 24, 0x27);
                    w32(dm, s7 + 32, t2 | gp);
                    w32(dm, s7 + 36, t0 | t1);
                    w16(dm, s7 + 40, v8out);
                    w16(dm, s7 + 42, 0);
                    w32(dm, s7 + 44, 0x10000400u);
                    s7 += 48;
                    if (c->flush(c->ctx, &s7, &s6))
                        return S2_UNSUPPORTED;
                    t1 = gp + 1;
                    v0 += sq;
                } while ((int32_t)a0 > 0);
            }
            w32(dm, s7, t6);                                /* (the delay slot of the branch out, and the first word of the second pass) */
            if ((int32_t)at <= 0)
                break;
            w32(dm, s7 + 4, v0);                            /* (the delay slot of the test of a3) */
            if ((int32_t)a3 > 0) {
                /* the rows past the image's last: a second pass */
                w8(dm, s7 + 8, 0x26);
                w32(dm, s7 + 16, 0x33000000u);
                a3 >>= 1;
                s5 = (t5 >> 1) - a3;
                w32(dm, s7 + 20, (s5 << 14) - 4096u);
                w8(dm, s7 + 20, 0x27);
                w8(dm, s7 + 24, 0x27);
                w32(dm, s7 + 32, t6);
                w32(dm, s7 + 36, v1);
                w32(dm, s7 + 40, 0x35100000u);
                w16(dm, s7 + 42, s5);
                w32(dm, s7 + 44, 0x06000000u);
                w32(dm, s7 + 48, 0x33000000u);
                w32(dm, s7 + 52, ((a3 << MUT(512, 14, 13)) - 4096u) | 0x06000000u);
                w8(dm, s7 + 56, 0x27);
                w32(dm, s7 + 64, t2 | t1);
                w32(dm, s7 + 68, t0 | t1);
                w16(dm, s7 + 72, v8out);
                w16(dm, s7 + 74, 0);
                w32(dm, s7 + 76, 0x10000400u);
                s7 += 80;
                if (c->flush(c->ctx, &s7, &s6))
                    return S2_UNSUPPORTED;
                t1 += 4;
                at -= 4;
                if ((int32_t)at <= 0)
                    break;
            }
            v0 = v1 + (a3 << 3);
            a0 = at;
            at = 0;
            do_loop = 1;
        }
    }
done:
    *s7p = s7;
    *s6p = s6;
    return S2_DONE;
}
