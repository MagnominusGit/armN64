/*
 * arMLEt - f3dex2_dmem.h: the F3DEX2 structs (f3dex2_fx.h) from and to DMEM
 *
 * DMEM is held as host words that contain the big-endian words: the halfword at an even byte address a is the high half of the word at a & ~3 when a & 2 is clear and the low half
 * otherwise. The adapters below know the microcode's layout (docs/ucode-f3dex2.md); the functions of f3dex2_fx.h know nothing of it.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEX2_DMEM_H
#define ARMLET_F3DEX2_DMEM_H

#include <string.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include "f3dex2_fx.h"
#include "hle.h"
#include "hle_fx.h"

/* The layout. The four standard builds share it; a dialect that moved the microcode's DMEM state (F3DEXBG) has its own instance (hle_f3dex2.c sets f2L per task). The members are
 * DMEM addresses; the F2_ names are the ones the rest of the front-end uses. */
struct f3;
/* G_ZBUFFER, G_SHADE, G_CULL_FRONT, G_CULL_BACK, G_FOG, G_LIGHTING, G_TEXTURE_GEN, G_TEXTURE_GEN_LINEAR, G_LOD, G_SHADING_SMOOTH, G_CLIPPING */
#define F2_GEOM_DEF (0x1u | 0x4u | 0x200u | 0x400u | 0x10000u | 0x20000u | 0x40000u | 0x80000u | 0x100000u | 0x200000u | 0x800000u)
struct f2_layout {
    uint16_t mv, p, mvp;
    uint16_t persp_norm, viewport_scale, viewport_trans;
    uint16_t clip_ratio, const31, const30;
    uint16_t fog_mul, fog_ofs, texture, tex_scale_s, tex_scale_t, geom;
    uint16_t mvp_valid, lights_valid, lights_size;
    uint16_t light_raw, light_base;
    uint16_t tg;                    /* the three texture-generation polynomial constants */
    uint16_t light_scale, light_scale_zex;
    uint16_t vtab, clipmask, culltab;
    uint16_t in, in_end;
    uint16_t jump_table;
    uint32_t geom_def;              /* the geometry mode bits the build gives a meaning to */
    uint8_t mw_max;                 /* the last index of the G_MOVEWORD table (even; 14, F3DEXBG's 16) */
    uint16_t movemem_tab;           /* halfword table by (w0 & 0xfe): destination, and +2 the second address of G_MTX */
    uint16_t moveword_adj;          /* the microcode's index arithmetic of G_MOVEWORD: table address = (w0 >> 16) + this, mod 0x1000 */
    uint16_t desc_light, desc_clip;
    uint16_t clip_planes, clip_masks;
    uint16_t out_bytes, out_toggle;
    uint8_t pipe_sync;              /* every flush check of the build is preceded by a G_RDPPIPESYNC written at the output pointer (S2DEX 2.05 [Safe]: the entry 0x640 / 0x644 of its flush routine): 8 bytes, one of them written */
    uint16_t ring_start;            /* with xbus: where the ring begins (0x9f8 in the Rej texts, 0xba8 in the F3DEX2.NoN xbus text Excitebike 64 loads, 0x4c0 in the S2DEX xbus text StarCraft 64 runs) */
    uint16_t ring_end, ring_gap;    /* ... the output pointer past ring_end wraps it (0xf10; StarCraft 64's S2DEX: 0xee0), the commands may run over by 0xb0; the RDP must stay more than ring_gap bytes behind the pointer (176; 224) */
    uint8_t rej_flags;              /* the Rej texts' variants (rej_fx.h: REJ_YQ) */
    uint8_t xbus;                   /* the RDP list goes through a ring in DMEM that the RDP reads (DPC_STATUS.XBUS; the Rej builds' `xbus` texts): no output buffers, every command that writes sets DPC_END (hle_rej.c) */
    uint8_t no_lit;                 /* the build's lighting code is another (F3DEXBG's 48-byte light slots): lit vertex commands do not use the standard lighting */
    int (*lit_vtx)(struct f3 *f);   /* ... they are run by this instead (the whole G_VTX), or decline when NULL */
};
extern const struct f2_layout *f2L;

#define F2_MV (f2L->mv)
#define F2_P (f2L->p)
#define F2_MVP (f2L->mvp)
#define F2_PERSP_NORM (f2L->persp_norm)
#define F2_VIEWPORT_SCALE (f2L->viewport_scale)
#define F2_VIEWPORT_TRANS (f2L->viewport_trans)
#define F2_CLIP_RATIO (f2L->clip_ratio)
#define F2_CONST31 (f2L->const31)
#define F2_CONST30 (f2L->const30)
#define F2_FOG_MUL (f2L->fog_mul)
#define F2_FOG_OFS (f2L->fog_ofs)
#define F2_TEXTURE (f2L->texture)
#define F2_TEX_SCALE_S (f2L->tex_scale_s)
#define F2_TEX_SCALE_T (f2L->tex_scale_t)
#define F2_GEOM (f2L->geom)
#define F2_MVP_VALID (f2L->mvp_valid)
#define F2_LIGHTS_VALID (f2L->lights_valid)
#define F2_LIGHTS_SIZE (f2L->lights_size)
#define F2_LIGHT_RAW (f2L->light_raw)
#define F2_LIGHT_BASE (f2L->light_base)
#define F2_VTAB (f2L->vtab)
#define F2_CLIPMASK (f2L->clipmask)
#define F2_CULLTAB (f2L->culltab)
#define F2_IN (f2L->in)
#define F2_IN_END (f2L->in_end)
enum { F2_LIGHT_STRIDE = 24, F2_VSTRIDE = 0x28, F2_IN_REFILL = 168 };

static inline uint16_t f2_bswap16(uint16_t v) { return (uint16_t)(v >> 8 | v << 8); }

/* a matrix: 8 words of integer parts, 8 of fractions */
static inline void f2_ld_mtx(const struct hle_task *t, unsigned a, struct fx2_mtx *m)
{
    unsigned k;
    for (k = 0; k < 8; k++) {
        uint32_t wi = t->dmem[((a & 0xfff) >> 2) + k], wf = t->dmem[(((a + 0x20) & 0xfff) >> 2) + k];
        ((int16_t *)m->i)[2 * k] = (int16_t)(wi >> 16);
        ((int16_t *)m->i)[2 * k + 1] = (int16_t)wi;
        ((uint16_t *)m->f)[2 * k] = (uint16_t)(wf >> 16);
        ((uint16_t *)m->f)[2 * k + 1] = (uint16_t)wf;
    }
}
static inline void f2_st_mtx(struct hle_task *t, unsigned a, const struct fx2_mtx *m)
{
    unsigned k;
    for (k = 0; k < 8; k++) {
        t->dmem[((a & 0xfff) >> 2) + k] = (uint32_t)(uint16_t)((const int16_t *)m->i)[2 * k] << 16 | (uint16_t)((const int16_t *)m->i)[2 * k + 1];
        t->dmem[(((a + 0x20) & 0xfff) >> 2) + k] = (uint32_t)((const uint16_t *)m->f)[2 * k] << 16 | ((const uint16_t *)m->f)[2 * k + 1];
    }
}

/* a vertex record (+0 .. +0x27): ten words */
static inline void f2_ld_vtx_lite(const struct hle_task *t, unsigned s, struct fx2_vtx *v)
{
    const uint32_t *w = &t->dmem[(s & 0xfff) >> 2];
    uint32_t c = w[4];
    v->col[0] = (uint8_t)(c >> 24);
    v->col[1] = (uint8_t)(c >> 16);
    v->col[2] = (uint8_t)(c >> 8);
    v->col[3] = (uint8_t)c;
    v->s = (int16_t)(w[5] >> 16);
    v->t = (int16_t)w[5];
    v->sx = (int16_t)(w[6] >> 16);
    v->sy = (int16_t)w[6];
    v->szi = (int16_t)(w[7] >> 16);
    v->szf = (uint16_t)w[7];
    v->rwi = (int16_t)(w[8] >> 16);
    v->rwf = (uint16_t)w[8];
    v->flags = w[9];
}
static inline void f2_ld_vtx(const struct hle_task *t, unsigned s, struct fx2_vtx *v)
{
    const uint32_t *w = &t->dmem[(s & 0xfff) >> 2];
    v->cx[0] = (int16_t)(w[0] >> 16);
    v->cx[1] = (int16_t)w[0];
    v->cx[2] = (int16_t)(w[1] >> 16);
    v->cx[3] = (int16_t)w[1];
    v->cf[0] = (uint16_t)(w[2] >> 16);
    v->cf[1] = (uint16_t)w[2];
    v->cf[2] = (uint16_t)(w[3] >> 16);
    v->cf[3] = (uint16_t)w[3];
    f2_ld_vtx_lite(t, s, v);
}
#ifdef __aarch64__
/* The words of DMEM hold the record's halfword pairs swapped against the host layout of struct fx2_vtx (the colour bytes reversed, the flag word as it is): three vector loads and the
 * lane reversals (the scalar form below is the reference; the device equivalence runs check this one). */
static inline void f2_st_vtx(struct hle_task *t, unsigned s, const struct fx2_vtx *v)
{
    uint8_t *d = (uint8_t *)&t->dmem[(s & 0xfff) >> 2];
    const uint8_t *p = (const uint8_t *)v;
    uint8x16_t a = vld1q_u8(p), b = vld1q_u8(p + 16);
    uint8x8_t c = vld1_u8(p + 32);
    uint8x16_t b16 = vreinterpretq_u8_u16(vrev32q_u16(vreinterpretq_u16_u8(b)));
    uint32x4_t bw = vsetq_lane_u32(vgetq_lane_u32(vreinterpretq_u32_u8(vrev32q_u8(b)), 0), vreinterpretq_u32_u8(b16), 0);
    uint32x2_t cw = vset_lane_u32(vget_lane_u32(vreinterpret_u32_u8(c), 1), vreinterpret_u32_u16(vrev32_u16(vreinterpret_u16_u8(c))), 1);
    vst1q_u8(d, vreinterpretq_u8_u16(vrev32q_u16(vreinterpretq_u16_u8(a))));
    vst1q_u32((uint32_t *)(d + 16), bw);
    vst1_u32((uint32_t *)(d + 32), cw);
}
#else
static inline void f2_st_vtx(struct hle_task *t, unsigned s, const struct fx2_vtx *v)
{
    uint32_t *w = &t->dmem[(s & 0xfff) >> 2];
    w[0] = (uint32_t)(uint16_t)v->cx[0] << 16 | (uint16_t)v->cx[1];
    w[1] = (uint32_t)(uint16_t)v->cx[2] << 16 | (uint16_t)v->cx[3];
    w[2] = (uint32_t)v->cf[0] << 16 | v->cf[1];
    w[3] = (uint32_t)v->cf[2] << 16 | v->cf[3];
    w[4] = (uint32_t)v->col[0] << 24 | (uint32_t)v->col[1] << 16 | (uint32_t)v->col[2] << 8 | v->col[3];
    w[5] = (uint32_t)(uint16_t)v->s << 16 | (uint16_t)v->t;
    w[6] = (uint32_t)(uint16_t)v->sx << 16 | (uint16_t)v->sy;
    w[7] = (uint32_t)(uint16_t)v->szi << 16 | v->szf;
    w[8] = (uint32_t)(uint16_t)v->rwi << 16 | v->rwf;
    w[9] = v->flags;
}
#endif

#ifdef __aarch64__
/* the 16 bytes of an input vertex in host-order words (DMEM or RDRAM) into the record: the halfword pairs swapped, the colour bytes reversed */
static inline void f2_vin_from_words(const void *p, struct fx2_vin *v)
{
    uint8x16_t w = vld1q_u8((const uint8_t *)p);
    uint32x4_t r = vsetq_lane_u32(vgetq_lane_u32(vreinterpretq_u32_u8(vrev32q_u8(w)), 3), vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u8(w))), 3);
    vst1q_u8((uint8_t *)v, vreinterpretq_u8_u32(r));
}
#endif
/* an input vertex as the DMA left it */
static inline void f2_ld_vin(const struct hle_task *t, unsigned a, struct fx2_vin *v)
{
#ifdef __aarch64__
    if ((a & 0xfff) <= 0xff0) {
        f2_vin_from_words(&t->dmem[(a & 0xfff) >> 2], v);
        return;
    }
#endif
    v->x = (int16_t)hle_dm16(t, a);
    v->y = (int16_t)hle_dm16(t, a + 2);
    v->z = (int16_t)hle_dm16(t, a + 4);
    v->s = (int16_t)hle_dm16(t, a + 8);
    v->t = (int16_t)hle_dm16(t, a + 10);
    {
        uint32_t c = hle_dm32(t, a + 12);
        v->c[0] = (uint8_t)(c >> 24);
        v->c[1] = (uint8_t)(c >> 16);
        v->c[2] = (uint8_t)(c >> 8);
        v->c[3] = (uint8_t)c;
    }
}

/* an input vertex from RDRAM (big-endian words) */
static inline void f2_ld_vin_rdram(const struct hle_task *t, uint32_t a, struct fx2_vin *v)
{
#ifdef __aarch64__
    if ((a & 0x7ffffc) <= 0x7ffff0) {
        uint32_t w[4];
        hle_ram_to_words(w, t->rdram, a & 0x7ffffc, 16);
        f2_vin_from_words(w, v);
        return;
    }
#endif
    uint32_t w0 = hle_rd32(t, a), w1 = hle_rd32(t, a + 4), w2 = hle_rd32(t, a + 8), w3 = hle_rd32(t, a + 12);
    v->x = (int16_t)(w0 >> 16);
    v->y = (int16_t)w0;
    v->z = (int16_t)(w1 >> 16);
    v->s = (int16_t)(w2 >> 16);
    v->t = (int16_t)w2;
    v->c[0] = (uint8_t)(w3 >> 24);
    v->c[1] = (uint8_t)(w3 >> 16);
    v->c[2] = (uint8_t)(w3 >> 8);
    v->c[3] = (uint8_t)w3;
}

/* the lights as the lit-vertex code reads them: the slots at 0x200 + 24 m, m = 0 .. nlights + 1 */
static inline void f2_ld_lights(const struct hle_task *t, unsigned nlights, struct fx2_lights *L)
{
    unsigned g, m, k;
    L->top = nlights + 1;
    for (g = 0; g < 2; g++)
        for (m = 0; m <= L->top; m++) {
            unsigned S = F2_LIGHT_BASE + F2_LIGHT_STRIDE * m;
            for (k = 0; k < 3; k++) {
                L->d[g][m][k] = (int16_t)((int8_t)hle_dm8(t, S + (g ? 4 : 0) + k) * 256);
                L->c[g][m][k] = (int16_t)(hle_dm8(t, S + (g ? 12 : 8) + k) << 7);
            }
            L->d[g][m][3] = 0;
            L->c[g][m][3] = 0;
        }
}

/* the vertex state */
static inline void f2_ld_vctx(const struct hle_task *t, struct fx2_vctx *c)
{
    unsigned k;
    f2_ld_mtx(t, F2_MVP, &c->mp);
    for (k = 0; k < 4; k++) {
        c->scale[k] = (int16_t)hle_dm16(t, F2_VIEWPORT_SCALE + 2 * k);
        c->trans[k] = (int16_t)hle_dm16(t, F2_VIEWPORT_TRANS + 2 * k);
    }
    c->scale[1] = hle_clamp16(-(int32_t)c->scale[1]);
    c->scale[3] = (int16_t)hle_dm16(t, F2_FOG_MUL);      /* the fog multiplier and offset (G_MV_FOG) take the fourth lane of the viewport scale and translation */
    c->trans[3] = (int16_t)hle_dm16(t, F2_FOG_OFS);
    c->pn = hle_dm16(t, F2_PERSP_NORM);
    c->ratio = hle_dm16(t, F2_CLIP_RATIO);
    c->sc_s = hle_dm16(t, F2_TEX_SCALE_S);
    c->sc_t = hle_dm16(t, F2_TEX_SCALE_T);
    c->c31_3 = hle_dm16(t, F2_CONST31 + 6);
    c->c31_7 = (int16_t)hle_dm16(t, F2_CONST31 + 14);
    for (k = 0; k < 3; k++)
        c->tg[k] = (int16_t)hle_dm16(t, f2L->tg + 2 * k);
    c->mode = hle_dm16(t, F2_GEOM) & 15u;
    c->gen204 = 0;
}

#endif
