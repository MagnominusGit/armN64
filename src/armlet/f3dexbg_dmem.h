/*
 * arMLEt - f3dexbg_dmem.h: the lit G_VTX of F3DEXBG on a task's DMEM (f3dexbg_fx.h's structs from and to DMEM, and the whole command)
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEXBG_DMEM_H
#define ARMLET_F3DEXBG_DMEM_H

#include "f3dex2_dmem.h"
#include "f3dexbg_fx.h"

#ifdef ARMLET_HLE_MUTATE
#undef MUT
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#ifndef MUT
#define MUT(n, ok, bad) (ok)
#endif
#endif

/* F3DEXBG's data: the light slots are 48 bytes (the processed direction at +16 of the raw slot, position at +24, colour at +40, attenuation byte at +4), the vertex normals' x and y are the
 * halfword table at 0x510 (G_MOVEMEM 14, indexed by the vertex slot), the view-space reconstruction constants are at 0x1d0 / 0x1e0 / 0x1f0 */
enum { FBG_LIGHT_STRIDE = 48, FBG_NORMALS = 0x510, FBG_K_HI = 0x1d0, FBG_K_LO = 0x1e0, FBG_ZOFF = 0x1f0 };

/* the lights as the lighting loop reads them: the slots at light_base + 48 m, m = 0 .. nlights + 1 */
static inline void fbg_ld_ctx(const struct hle_task *t, unsigned nlights, struct fbg_ctx *c)
{
    unsigned g, m, k;
    c->top = nlights + 1;
    c->positional = (hle_dm16(t, F2_GEOM) >> MUT(120, 6, 5)) & 1;
    for (m = 0; m <= c->top; m++) {
        unsigned S = F2_LIGHT_BASE + FBG_LIGHT_STRIDE * m;
        c->s[m].atten = hle_dm8(t, S - 12);
        for (g = 0; g < 2; g++) {
            for (k = 0; k < 3; k++) {
                c->s[m].d[g][k] = (int16_t)((int8_t)hle_dm8(t, S + 4 * g + k) * 256);
                c->s[m].col[g][k] = (int16_t)(hle_dm8(t, S + 24 + 4 * g + k) << 7);
            }
            for (k = 0; k < 4; k++)
                c->s[m].pos[g][k] = (int16_t)hle_dm16(t, S + 8 + 8 * g + 2 * k);
        }
    }
    for (g = 0; g < 2; g++)
        for (k = 0; k < 4; k++) {
            c->khi[g][k] = (int16_t)hle_dm16(t, FBG_K_HI + 8 * g + 2 * k);
            c->klo[g][k] = (int16_t)hle_dm16(t, FBG_K_LO + 8 * g + 2 * k);
            c->zoff[g][k] = (int16_t)hle_dm16(t, FBG_ZOFF + 8 * g + 2 * k);
        }
}

/* The lit G_VTX (w0, w1 resolved to va: the vertices in RDRAM; lean: they are read from there, else from DMEM where the DMA of the microcode put them; dl: RDRAM address of the display list's current chunk, the input buffer's contents in the lean mode). Returns NULL or the reason the task declines. */
static inline const char *fbg_cmd_vtx(struct hle_task *t, uint32_t w0, uint32_t va, int lean, uint32_t dl)
{
    unsigned n = (w0 >> 12) & 0xff, size = hle_dm16(t, F2_LIGHTS_SIZE - 1), nl = size / FBG_LIGHT_STRIDE, endslot, first, s4, i, v0;
    struct fx2_vctx c;
    struct fbg_ctx bc;
    static struct fx2_vin in[34];
    static struct fbg_vin bin[34];
    static struct fx2_vtx out[34];

    if (n == 0 || n > 32)
        return "G_VTX with an unusual count";
    if (size % FBG_LIGHT_STRIDE || !(nl & 1) || nl > FBG_MAX_LIGHTS)
        return "F3DEXBG: unusual light count";
    if (!hle_dm8(t, F2_LIGHTS_VALID)) {                 /* the lights have changed: the setup normalises the directions, as in F3DEX2 (the slots are 48 bytes apart) */
        struct fx2_mtx mv;
        int8_t raw[FBG_SLOTS][3], o[FBG_SLOTS][3];
        unsigned m, j;
        f2_ld_mtx(t, F2_MV, &mv);
        for (m = 0; m <= nl + 1; m++)
            for (j = 0; j < 3; j++)
                raw[m][j] = (int8_t)hle_dm8(t, F2_LIGHT_RAW + FBG_LIGHT_STRIDE * m + j + MUT(115, 0, 1));
        fx2_light_setup(&mv, raw, nl, (int16_t)hle_dm16(t, f2L->light_scale), o);
        hle_st8(t, F2_LIGHTS_VALID, (uint8_t)w0);
        for (m = 0; m <= nl + 1; m++)
            for (j = 0; j < 3; j++) {
                hle_st8(t, F2_LIGHT_BASE + FBG_LIGHT_STRIDE * m + j, (uint8_t)o[m][j]);
                hle_st8(t, F2_LIGHT_BASE + FBG_LIGHT_STRIDE * m + 4 + j, (uint8_t)o[m][j]);
            }
    }
    if (!hle_dm8(t, F2_MVP_VALID)) {
        struct fx2_mtx mv, p, r;
        f2_ld_mtx(t, F2_MV, &mv);
        f2_ld_mtx(t, F2_P, &p);
        fx2_mtx_mul(&mv, &p, &r);
        f2_st_mtx(t, F2_MVP, &r);
        hle_st8(t, F2_MVP_VALID, (uint8_t)w0);
    }
    endslot = hle_dm16(t, F2_VTAB + (w0 & 0xfff));
    first = hle_dm16(t, F2_VTAB + ((w0 - n * 2) & 0xfff));
    s4 = endslot - n * 16;
    v0 = (w0 & 0xfff) / 2 - n;                          /* the slot number of the first vertex */
    f2_ld_vctx(t, &c);
    c.mode &= ~(unsigned)FX2_LIGHTING;                  /* the standard lighting stays off: fbg_lit_pass() does it */
    for (i = 0; i < n + (n & 1); i++) {
        unsigned flagnz;
        if (lean && i < n) {
            uint32_t w1 = hle_rd32(t, va + 16 * i + 4);
            f2_ld_vin_rdram(t, va + 16 * i, &in[i]);
            flagnz = w1 & 0xffff;
        } else {
            if (lean && s4 + 16 * i == F2_IN)           /* an odd count that ends at the last slot: the pair's phantom vertex is the first bytes of the input buffer, which the lean mode does not keep in DMEM */
                f2_ld_vin_rdram(t, dl - MUT(121, 0, 8), &in[i]);
            else
                f2_ld_vin(t, s4 + 16 * i, &in[i]);
            flagnz = hle_dm16(t, s4 + 16 * i + 6);
        }
        bin[i].n[0] = (int8_t)hle_dm8(t, FBG_NORMALS + 2 * (v0 + i));
        bin[i].n[1] = (int8_t)hle_dm8(t, FBG_NORMALS + 2 * (v0 + i) + 1);
        bin[i].n[2] = (int8_t)MUT(116, flagnz, flagnz >> 8);
        bin[i].flag = (int8_t)(flagnz >> 8);
        memcpy(bin[i].col, in[i].c, 4);
    }
    fx2_vtx_xform(&c, in, n, out);
    fbg_ld_ctx(t, nl, &bc);
    fbg_lit_pass(&c, &bc, bin, n, out);
    for (i = 0; i < n; i++)
        f2_st_vtx(t, first + F2_VSTRIDE * i, &out[i]);
    /* the scratch words the lighting code leaves (the last pair's: its fractions of the clip coordinates swapped by halves, its input colours, and the first two rows of the MVP matrix's integer
     * part, twice each) */
    {
        unsigned a = (n - 1) & ~1u, b = a + 1, k;
        uint16_t cfa[4], cfb[4];
        memcpy(cfa, out[a].cf, 8);
        if (b < n)
            memcpy(cfb, out[b].cf, 8);
        else
            fbg_clip_frac(&c, &in[b], cfb);
        for (k = 0; k < 4; k++) {
            hle_st16(t, 0x200 + 2 * k, MUT(117, cfb[k], cfa[k]));
            hle_st16(t, 0x208 + 2 * k, cfa[k]);
            hle_st8(t, 0x258 + k, in[b].c[k]);
            hle_st8(t, 0x25c + k, MUT(118, in[a].c[k], in[b].c[k]));
        }
        for (k = 0; k < 4; k++) {
            hle_st16(t, 0x1b0 + 2 * k, hle_dm16(t, F2_MVP + 2 * k));
            hle_st16(t, 0x1b8 + 2 * k, hle_dm16(t, F2_MVP + 2 * k));
            hle_st16(t, 0x1c0 + 2 * k, hle_dm16(t, F2_MVP + 8 + 2 * k));
            hle_st16(t, 0x1c8 + 2 * k, hle_dm16(t, F2_MVP + MUT(119, 8, 0) + 2 * k));
        }
    }
    return NULL;
}

#endif
