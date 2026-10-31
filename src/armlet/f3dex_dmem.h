/*
 * arMLEt - f3dex_dmem.h: filling the f3dex_fx.h structs from a DMEM (or RDRAM) image and writing results back
 *
 * Both images are arrays of 32-bit host words that hold the big-endian words (byte a is at ((uint8_t *)img)[a ^ 3]), the layout hle.h's accessors use. The DMEM addresses are a build's layout
 * (struct f3dex_layout, the record's DMEM part: docs/ucode-f3dex.md section 3, docs/native-f3dex.md section 3); f3dex_lay_std is that of the 1.2x builds, b->lay == NULL means it (the tools).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEX_DMEM_H
#define ARMLET_F3DEX_DMEM_H

#include <string.h>

#include "f3dex_fx.h"

static inline uint8_t xd8(const uint32_t *m, unsigned a) { return ((const uint8_t *)m)[(a & 0xfff) ^ 3]; }
static inline uint16_t xd16(const uint32_t *m, unsigned a) { return ((const uint16_t *)m)[((a & 0xfff) ^ 2) >> 1]; }
static inline uint32_t xd32(const uint32_t *m, unsigned a) { return m[(a & 0xfff) >> 2]; }
static inline void xs8(uint32_t *m, unsigned a, uint8_t v) { ((uint8_t *)m)[(a & 0xfff) ^ 3] = v; }
static inline void xs16(uint32_t *m, unsigned a, uint16_t v) { ((uint16_t *)m)[((a & 0xfff) ^ 2) >> 1] = v; }
static inline void xs32(uint32_t *m, unsigned a, uint32_t v) { m[(a & 0xfff) >> 2] = v; }
/* RDRAM (23-bit addresses) */
static inline uint8_t xr8(const uint32_t *m, unsigned a) { return ((const uint8_t *)m)[a & 0x7fffff]; }
static inline uint16_t xr16(const uint32_t *m, unsigned a) { const uint8_t *p = (const uint8_t *)m + (a & 0x7ffffe); return (uint16_t)(p[0] << 8 | p[1]); }

/* The DMEM layout of a build. The first F3DEX_LAY_NKNOB fields are addresses the derivation reads from the code (tools/derive_f1d.py: operands, also relative to the base register $29 = 0x110);
 * the rest are constants of the data segment's first 0xc0 bytes that it copies from the reference. Fixed structure of the family (not fields): light records of 32 bytes from `light` (the texgen
 * weights at +0, colours +0x10, raw directions +0x18, model-space directions +0x20), G_TEXTURE's tile byte at texture + 2 and its scales at + 4, the FIFO end at fifo + 4, the vertex records of 0x28
 * bytes from vbuf, the window of 320 bytes, the DMA table at 0 (overlay k at 8 + 8k), the OSTask header at 0xfc0. */
#define F3DEX_LAY_FIELDS(X) \
    X(jt0) X(jt2) X(half1) X(persp) X(geom) X(omh) X(oml) X(texture) X(rdp_wp) X(lights) X(mtx_sp) X(mtx_lim) X(depth) X(fifo) X(segs) X(light) X(mm_tab) X(mw_tab) X(vtab) \
    X(viewport) X(fog) X(retstack) X(mv) X(proj) X(mvp) X(vbuf) X(window) X(operand) X(outbuf) \
    X(lim) X(tgbias) X(vsign) X(clip_k) X(planes) X(tg) X(plane_masks)
#define F3DEX_LAY_NKNOB 29
#define F3DEX_LAY_N 36
struct f3dex_layout {
#define X(f) uint16_t f;
    F3DEX_LAY_FIELDS(X)
#undef X
};
extern const struct f3dex_layout f3dex_lay_std;
static inline const struct f3dex_layout *f3dex_lay(const struct f3dex_build *b) { return b && b->lay ? b->lay : &f3dex_lay_std; }

/* the standard layout's addresses (the tools) */
#define F3DEX_DM_MV 0x3e0
#define F3DEX_DM_PROJ 0x420
#define F3DEX_DM_MVP 0x460
#define F3DEX_DM_VBUF 0x4a0
#define F3DEX_DM_VIN 0xae0
#define F3DEX_VSLOT(k) (F3DEX_DM_VBUF + 0x28u * (k))

/* a matrix: 16 integer halfwords (row-major, 8 bytes per row) then 16 fractions at +0x20 */
static inline void f3dex_load_mtx(struct f3dex_mtx *m, const uint32_t *img, unsigned a, int rdram)
{
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            m->i[r][c] = (int16_t)(rdram ? xr16(img, a + 8 * r + 2 * c) : xd16(img, a + 8 * r + 2 * c));
            m->f[r][c] = rdram ? xr16(img, a + 0x20 + 8 * r + 2 * c) : xd16(img, a + 0x20 + 8 * r + 2 * c);
        }
}
static inline void f3dex_store_mtx(const struct f3dex_mtx *m, uint32_t *dmem, unsigned a)
{
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            xs16(dmem, a + 8 * r + 2 * c, (uint16_t)m->i[r][c]);
            xs16(dmem, a + 0x20 + 8 * r + 2 * c, m->f[r][c]);
        }
}

/* an input vertex at byte address a (the 16 bytes of the display list) */
static inline void f3dex_load_vin(struct f3dex_vin *v, const uint32_t *img, unsigned a, int rdram)
{
    uint16_t (*r16)(const uint32_t *, unsigned) = rdram ? xr16 : xd16;
    uint8_t (*r8)(const uint32_t *, unsigned) = rdram ? xr8 : xd8;
    v->x = (int16_t)r16(img, a);
    v->y = (int16_t)r16(img, a + 2);
    v->z = (int16_t)r16(img, a + 4);
    v->s = (int16_t)r16(img, a + 8);
    v->t = (int16_t)r16(img, a + 10);
    for (int i = 0; i < 4; i++)
        v->c[i] = r8(img, a + 12 + i);
}

/* a vertex record (DMEM +0 .. +37) */
static inline void f3dex_load_vtx(struct f3dex_vtx *v, const uint32_t *dmem, unsigned a)
{
    for (int k = 0; k < 4; k++) {
        v->cx[k] = (int16_t)xd16(dmem, a + 2 * k);
        v->cf[k] = xd16(dmem, a + 8 + 2 * k);
        v->col[k] = xd8(dmem, a + 16 + k);
    }
    v->s = (int16_t)xd16(dmem, a + 20);
    v->t = (int16_t)xd16(dmem, a + 22);
    v->sx = (int16_t)xd16(dmem, a + 24);
    v->sy = (int16_t)xd16(dmem, a + 26);
    v->sz = (int16_t)xd16(dmem, a + 28);
    v->szf = xd16(dmem, a + 30);
    v->rwi = (int16_t)xd16(dmem, a + 32);
    v->rwf = xd16(dmem, a + 34);
    v->flags = xd16(dmem, a + 36);
}
static inline void f3dex_store_vtx(const struct f3dex_vtx *v, uint32_t *dmem, unsigned a)
{
    for (int k = 0; k < 4; k++) {
        xs16(dmem, a + 2 * k, (uint16_t)v->cx[k]);
        xs16(dmem, a + 8 + 2 * k, v->cf[k]);
        xs8(dmem, a + 16 + k, v->col[k]);
    }
    xs16(dmem, a + 20, (uint16_t)v->s);
    xs16(dmem, a + 22, (uint16_t)v->t);
    xs16(dmem, a + 24, (uint16_t)v->sx);
    xs16(dmem, a + 26, (uint16_t)v->sy);
    xs16(dmem, a + 28, (uint16_t)v->sz);
    xs16(dmem, a + 30, v->szf);
    xs16(dmem, a + 32, (uint16_t)v->rwi);
    xs16(dmem, a + 34, v->rwf);
    xs16(dmem, a + 36, v->flags);
}

/* The G_VTX state from DMEM without the lights (what the clipper and the triangle code need). */
static inline void f3dex_load_vctx_geom(struct f3dex_vctx *c, const uint32_t *dmem, const struct f3dex_build *b)
{
    const struct f3dex_layout *L = f3dex_lay(b);
    c->b = b;
    f3dex_load_mtx(&c->mp, dmem, L->mvp, 0);
    for (int k = 0; k < 4; k++) {
        c->vscale[k] = (int16_t)xd16(dmem, L->viewport + 2 * k);
        c->vtrans[k] = (int16_t)xd16(dmem, L->viewport + 8 + 2 * k);
        c->vsign[k] = (int16_t)xd16(dmem, L->vsign + 2 * k);
        c->lim[k] = (int16_t)xd16(dmem, L->lim + 2 * k);
    }
    c->persp = xd16(dmem, L->persp);
    c->clip_k = (int16_t)xd16(dmem, L->clip_k);
    c->tscale[0] = xd16(dmem, L->texture + 4);
    c->tscale[1] = xd16(dmem, L->texture + 6);
    c->geom = xd32(dmem, L->geom);
    for (int k = 0; k < 3; k++)
        c->fog[k] = (int16_t)xd16(dmem, L->fog + 2 * k);
    c->tgbias = (int16_t)xd16(dmem, L->tgbias);
    c->vco_carry = 0;
    for (int p = 0; p < 6; p++) {
        for (int k = 0; k < 4; k++)
            c->plane[p][k] = (int16_t)xd16(dmem, L->planes - 8 * p + 2 * k);
        c->plane_mask[p] = xd16(dmem, L->plane_masks - 2 * p);
    }
    c->nrec = 0;
}
/* The G_VTX state from DMEM. Returns nonzero when lighting is on and the light words are in a state the functions do not model (a pending relight (word `lights` negative), a count that is not a multiple of
 * 32 or more than F3DEX_MAX_REC records); without lighting the light words are not read. */
static inline int f3dex_load_vctx(struct f3dex_vctx *c, const uint32_t *dmem, const struct f3dex_build *b)
{
    const struct f3dex_layout *L = f3dex_lay(b);
    f3dex_load_vctx_geom(c, dmem, b);
    if (c->geom & F3DEX_G_LIGHTING) {
        uint32_t l = xd32(dmem, L->lights);
        if ((int32_t)l < 0 || (l & 31) || (l >> 5) > F3DEX_MAX_REC)
            return 1;
        c->nrec = l >> 5;
        for (unsigned k = 0; k <= c->nrec + 1; k++)
            for (int i = 0; i < 8; i++)
                c->lcol[k][i] = xd8(dmem, L->light + 0x10 + 32 * k + i);
        for (unsigned k = 0; k <= c->nrec; k++)
            for (int i = 0; i < 8; i++)
                c->ldir[k][i] = (int8_t)xd8(dmem, L->light + 0x20 + 32 * k + i);
        for (int i = 0; i < 8; i++)
            c->lk[i] = xd8(dmem, L->light + i);
        for (int i = 0; i < 3; i++)
            c->tg[i] = (int16_t)xd16(dmem, L->tg + 2 * i);
    }
    return 0;
}
#endif
