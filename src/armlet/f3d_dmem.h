/*
 * arMLEt - f3d_dmem.h: filling the f3d_fx.h structs from a DMEM (or RDRAM) image and writing results back
 *
 * Both images are arrays of 32-bit host words that hold the big-endian words (byte a is at ((uint8_t *)img)[a ^ 3]), the layout hle.h's accessors use. The offsets are those of the
 * microcode's builds (docs/ucode-f3d.md 5 and 10.3); the build-dependent ones are in struct f3d_layout.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3D_DMEM_H
#define ARMLET_F3D_DMEM_H

#include <string.h>

#include "f3d_fx.h"

/* The DMEM layout of a build (the record's DMEM part: docs/ucode-f3d.md 5 and 10.3, docs/ucode-derive.md "F3D records that run"). The first F3D_LAY_NKNOB fields are addresses the derivation
 * reads from the code (tools/derive_f3d.py: operands, also relative to the base register $29 = 0x110), jt0 / jt2 are read from the two dispatchers, tri4_tab (GoldenEye's nibble-to-vertex-byte
 * table of G_TRI4; 0: none) from that handler, the rest are constants of the data segment's first
 * 0xc0 bytes that it copies from the reference. vret is the halfword where G_VTX and the clipper keep their return address (scratch: a dead range of the build). Fixed structure of the family (not fields): light records of 32 bytes from `light` (the texgen weights at +0, colours +0x10, raw directions +0x18,
 * model-space directions +0x20; the clamp of the relight at light + 0x68), G_TEXTURE's level byte at texture + 1, its tile byte at + 2 and its scales at + 4, the FIFO end at fifo + 4, the projection
 * at mv + 0x40, vertex records of 40 bytes from vtx_base up to the command buffer, the command buffer of 320 bytes, the DMA table at 0 (overlay k at 8 + 8k), the OSTask header at 0xfc0. */
#define F3D_LAY_FIELDS(X) \
    X(half1) X(persp) X(geom) X(omh) X(oml) X(texture) X(rdp_wp) X(lights) X(mtx_sp) X(mtx_lim) X(depth) X(fifo) X(segs) X(mm_tab) X(mw_tab) X(retstack) \
    X(viewport) X(fog) X(mv) X(mp) X(vtx_base) X(cmdbuf) X(geombuf) X(outbuf) X(ring_end) X(vret) \
    X(jt0) X(jt2) X(tri4_tab) \
    X(light) X(lim) X(vsign) X(clip_k) X(planes) X(tg) X(plane_masks)
#define F3D_LAY_NKNOB 26
#define F3D_LAY_NSIG 29
#define F3D_LAY_N 36
#define F3D_VSTRIDE 40
struct f3d_layout {
#define X(f) uint16_t f;
    F3D_LAY_FIELDS(X)
#undef X
    uint8_t flags_variant;              /* F3D_FLAGS_*: how the vertex code forms the clip flag word (the record's attribute, kept here for the loaders) */
};
static inline unsigned f3d_nslots(const struct f3d_layout *l) { return (unsigned)(l->cmdbuf - l->vtx_base) / F3D_VSTRIDE; }   /* the vertex buffer ends where the command buffer starts */

/* SM64 and the 2.0D builds of family A (docs/ucode-f3d.md 10.3.2) */
static const struct f3d_layout f3d_layout_a = {
    .half1 = 0x10c, .persp = 0x112, .geom = 0x114, .omh = 0x118, .oml = 0x11c, .texture = 0x120, .rdp_wp = 0x128, .lights = 0x12c, .mtx_sp = 0x134, .mtx_lim = 0x15c, .depth = 0x110, .fifo = 0x150, .segs = 0x160, .mm_tab = 0x2f0, .mw_tab = 0x30e, .retstack = 0x336, .viewport = 0x320, .fog = 0x330, .mv = 0x360, .mp = 0x3e0, .vtx_base = 0x420, .cmdbuf = 0x6a0, .geombuf = 0x7e0, .outbuf = 0x9e0, .ring_end = 0xde0, .vret = 0x106, .jt0 = 0x0c4, .jt2 = 0x076, .tri4_tab = 0x000, .light = 0x1a0, .lim = 0x028, .vsign = 0x050, .clip_k = 0x076, .planes = 0x098, .tg = 0x0a2, .plane_masks = 0x0b2,
    .flags_variant = F3D_FLAGS_SM64 };
/* Pilotwings: family A with its own clip flag word */
static const struct f3d_layout f3d_layout_pw = {
    .half1 = 0x10c, .persp = 0x112, .geom = 0x114, .omh = 0x118, .oml = 0x11c, .texture = 0x120, .rdp_wp = 0x128, .lights = 0x12c, .mtx_sp = 0x134, .mtx_lim = 0x15c, .depth = 0x110, .fifo = 0x150, .segs = 0x160, .mm_tab = 0x2f0, .mw_tab = 0x30e, .retstack = 0x336, .viewport = 0x320, .fog = 0x330, .mv = 0x360, .mp = 0x3e0, .vtx_base = 0x420, .cmdbuf = 0x6a0, .geombuf = 0x7e0, .outbuf = 0x9e0, .ring_end = 0xde0, .vret = 0x106, .jt0 = 0x0c4, .jt2 = 0x076, .tri4_tab = 0x000, .light = 0x1a0, .lim = 0x028, .vsign = 0x050, .clip_k = 0x076, .planes = 0x098, .tg = 0x0a2, .plane_masks = 0x0b2,
    .flags_variant = F3D_FLAGS_PW };
/* the 2.0H builds of family A (Mischief Makers, Tetrisphere): the G_DL depth at 0x15a, the class-2 table 2 lower */
static const struct f3d_layout f3d_layout_h = {
    .half1 = 0x10c, .persp = 0x112, .geom = 0x114, .omh = 0x118, .oml = 0x11c, .texture = 0x120, .rdp_wp = 0x128, .lights = 0x12c, .mtx_sp = 0x134, .mtx_lim = 0x15c, .depth = 0x15a, .fifo = 0x150, .segs = 0x160, .mm_tab = 0x2f0, .mw_tab = 0x30e, .retstack = 0x336, .viewport = 0x320, .fog = 0x330, .mv = 0x360, .mp = 0x3e0, .vtx_base = 0x420, .cmdbuf = 0x6a0, .geombuf = 0x7e0, .outbuf = 0x9e0, .ring_end = 0xde0, .vret = 0x104, .jt0 = 0x0c4, .jt2 = 0x074, .tri4_tab = 0x000, .light = 0x1a0, .lim = 0x028, .vsign = 0x050, .clip_k = 0x076, .planes = 0x098, .tg = 0x0a2, .plane_masks = 0x0b2,
    .flags_variant = F3D_FLAGS_SM64 };
/* GoldenEye (2.0G): Pilotwings' clip flag word, the G_DL depth byte of the 2.0H builds, G_TRI4's nibble table */
static const struct f3d_layout f3d_layout_ge = {
    .half1 = 0x10c, .persp = 0x112, .geom = 0x114, .omh = 0x118, .oml = 0x11c, .texture = 0x120, .rdp_wp = 0x128, .lights = 0x12c, .mtx_sp = 0x134, .mtx_lim = 0x15c, .depth = 0x15a, .fifo = 0x150, .segs = 0x160, .mm_tab = 0x2f0, .mw_tab = 0x30e, .retstack = 0x336, .viewport = 0x320, .fog = 0x330, .mv = 0x360, .mp = 0x3e0, .vtx_base = 0x420, .cmdbuf = 0x6a0, .geombuf = 0x7e0, .outbuf = 0x9e0, .ring_end = 0xde0, .vret = 0x106, .jt0 = 0x0c4, .jt2 = 0x076, .tri4_tab = 0x2d0, .light = 0x1a0, .lim = 0x028, .vsign = 0x050, .clip_k = 0x076, .planes = 0x098, .tg = 0x0a2, .plane_masks = 0x0b2,
    .flags_variant = F3D_FLAGS_PW };
/* Wave Race: the buffers 0x280 higher, so 32 vertex slots */
static const struct f3d_layout f3d_layout_wr = {
    .half1 = 0x10c, .persp = 0x112, .geom = 0x114, .omh = 0x118, .oml = 0x11c, .texture = 0x120, .rdp_wp = 0x128, .lights = 0x12c, .mtx_sp = 0x134, .mtx_lim = 0x15c, .depth = 0x110, .fifo = 0x150, .segs = 0x160, .mm_tab = 0x2f0, .mw_tab = 0x30e, .retstack = 0x336, .viewport = 0x320, .fog = 0x330, .mv = 0x360, .mp = 0x3e0, .vtx_base = 0x420, .cmdbuf = 0x920, .geombuf = 0xa60, .outbuf = 0xc60, .ring_end = 0xde0, .vret = 0x106, .jt0 = 0x0c4, .jt2 = 0x076, .tri4_tab = 0x000, .light = 0x1a0, .lim = 0x028, .vsign = 0x050, .clip_k = 0x076, .planes = 0x098, .tg = 0x0a2, .plane_masks = 0x0b2,
    .flags_variant = F3D_FLAGS_SM64 };

static inline uint8_t fd8(const uint32_t *m, unsigned a) { return ((const uint8_t *)m)[(a & 0xfff) ^ 3]; }
static inline uint16_t fd16(const uint32_t *m, unsigned a) { return ((const uint16_t *)m)[((a & 0xfff) ^ 2) >> 1]; }
static inline uint32_t fd32(const uint32_t *m, unsigned a) { return m[(a & 0xfff) >> 2]; }
static inline void fs8(uint32_t *m, unsigned a, uint8_t v) { ((uint8_t *)m)[(a & 0xfff) ^ 3] = v; }
static inline void fs16(uint32_t *m, unsigned a, uint16_t v) { ((uint16_t *)m)[((a & 0xfff) ^ 2) >> 1] = v; }
static inline void fs32(uint32_t *m, unsigned a, uint32_t v) { m[(a & 0xfff) >> 2] = v; }
/* the same for RDRAM (23-bit addresses) */
static inline uint16_t fr16(const uint32_t *m, unsigned a) { const uint8_t *p = (const uint8_t *)m + (a & 0x7ffffe); return (uint16_t)(p[0] << 8 | p[1]); }
static inline uint8_t fr8(const uint32_t *m, unsigned a) { return ((const uint8_t *)m)[a & 0x7fffff]; }

/* a matrix: 16 integer halfwords (row-major, 8 bytes per row) then 16 fractions at +0x20 */
static inline void f3d_load_mtx(struct f3d_mtx *m, const uint32_t *img, unsigned a, int rdram)
{
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            m->i[r][c] = (int16_t)(rdram ? fr16(img, a + 8 * r + 2 * c) : fd16(img, a + 8 * r + 2 * c));
            m->f[r][c] = rdram ? fr16(img, a + 0x20 + 8 * r + 2 * c) : fd16(img, a + 0x20 + 8 * r + 2 * c);
        }
}
static inline void f3d_store_mtx(const struct f3d_mtx *m, uint32_t *dmem, unsigned a)
{
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            fs16(dmem, a + 8 * r + 2 * c, (uint16_t)m->i[r][c]);
            fs16(dmem, a + 0x20 + 8 * r + 2 * c, m->f[r][c]);
        }
}

/* an input vertex at byte address a of an image */
static inline void f3d_load_vin(struct f3d_vin *v, const uint32_t *img, unsigned a, int rdram)
{
    uint16_t (*r16)(const uint32_t *, unsigned) = rdram ? fr16 : fd16;
    uint8_t (*r8)(const uint32_t *, unsigned) = rdram ? fr8 : fd8;
    v->x = (int16_t)r16(img, a);
    v->y = (int16_t)r16(img, a + 2);
    v->z = (int16_t)r16(img, a + 4);
    v->s = (int16_t)r16(img, a + 8);
    v->t = (int16_t)r16(img, a + 10);
    for (int i = 0; i < 4; i++)
        v->c[i] = r8(img, a + 12 + i);
}

/* a vertex record (DMEM +0 .. +37) */
static inline void f3d_load_vtx(struct f3d_vtx *v, const uint32_t *dmem, unsigned a)
{
    for (int k = 0; k < 4; k++) {
        v->cx[k] = (int16_t)fd16(dmem, a + 2 * k);
        v->cf[k] = fd16(dmem, a + 8 + 2 * k);
        v->col[k] = fd8(dmem, a + 16 + k);
    }
    v->s = (int16_t)fd16(dmem, a + 20);
    v->t = (int16_t)fd16(dmem, a + 22);
    v->sx = (int16_t)fd16(dmem, a + 24);
    v->sy = (int16_t)fd16(dmem, a + 26);
    v->sz = (int16_t)fd16(dmem, a + 28);
    v->szf = fd16(dmem, a + 30);
    v->rwi = (int16_t)fd16(dmem, a + 32);
    v->rwf = fd16(dmem, a + 34);
    v->flags = fd16(dmem, a + 36);
}
static inline void f3d_store_vtx(const struct f3d_vtx *v, uint32_t *dmem, unsigned a)
{
    for (int k = 0; k < 4; k++) {
        fs16(dmem, a + 2 * k, (uint16_t)v->cx[k]);
        fs16(dmem, a + 8 + 2 * k, v->cf[k]);
        fs8(dmem, a + 16 + k, v->col[k]);
    }
    fs16(dmem, a + 20, (uint16_t)v->s);
    fs16(dmem, a + 22, (uint16_t)v->t);
    fs16(dmem, a + 24, (uint16_t)v->sx);
    fs16(dmem, a + 26, (uint16_t)v->sy);
    fs16(dmem, a + 28, (uint16_t)v->sz);
    fs16(dmem, a + 30, v->szf);
    fs16(dmem, a + 32, (uint16_t)v->rwi);
    fs16(dmem, a + 34, v->rwf);
    fs16(dmem, a + 36, v->flags);
}

/* The G_VTX state from DMEM. Returns nonzero when the lights are in a state the functions do not model (a pending relight (D32[0x12c] < 0), a count that is not a multiple of 32, fewer than one
 * or more than F3D_MAX_SLOTS slots) and lighting is on; without lighting the light words are not read. */
static inline int f3d_load_vctx(struct f3d_vctx *c, const uint32_t *dmem, const struct f3d_layout *lay)
{
    f3d_load_mtx(&c->mp, dmem, lay->mp, 0);
    for (int k = 0; k < 4; k++) {
        c->vscale[k] = (int16_t)fd16(dmem, lay->viewport + 2 * k);
        c->vtrans[k] = (int16_t)fd16(dmem, lay->viewport + 8 + 2 * k);
        c->vsign[k] = (int16_t)fd16(dmem, lay->vsign + 2 * k);
        c->lim[k] = (int16_t)fd16(dmem, lay->lim + 2 * k);
    }
    c->persp = fd16(dmem, lay->persp);
    c->clip_k = (int16_t)fd16(dmem, lay->clip_k);
    c->tscale[0] = fd16(dmem, lay->texture + 4);
    c->tscale[1] = fd16(dmem, lay->texture + 6);
    c->geom = fd32(dmem, lay->geom);
    for (int k = 0; k < 3; k++)
        c->fog[k] = (int16_t)fd16(dmem, lay->fog + 2 * k);
    for (int p = 0; p < 6; p++) {
        for (int k = 0; k < 4; k++)
            c->plane[p][k] = (int16_t)fd16(dmem, lay->planes - 8 * p + 2 * k);
        c->plane_mask[p] = fd16(dmem, lay->plane_masks - 2 * p);
    }
    c->flags_variant = lay->flags_variant;
    c->nslot = 0;
    if (c->geom & F3D_G_LIGHTING) {
        uint32_t l = fd32(dmem, lay->lights);
        if ((int32_t)l < 0 || (l & 31) || (l >> 5) < 1 || (l >> 5) > F3D_MAX_SLOTS)
            return 1;
        c->nslot = l >> 5;
        for (unsigned k = 0; k <= c->nslot + 1; k++)
            for (int i = 0; i < 8; i++)
                c->lcol[k][i] = fd8(dmem, lay->light + 0x10 + 32 * k + i);
        for (unsigned k = 0; k <= c->nslot; k++)
            for (int i = 0; i < 8; i++)
                c->ldir[k][i] = (int8_t)fd8(dmem, lay->light + 0x20 + 32 * k + i);
        for (int i = 0; i < 8; i++)
            c->lk[i] = (int8_t)fd8(dmem, lay->light + i);
        for (int i = 0; i < 3; i++)
            c->tg[i] = (int16_t)fd16(dmem, lay->tg + 2 * i);
    }
    return 0;
}

#endif
