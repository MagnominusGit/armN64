/*
 * arMLEt - rej_fx.h: the geometry of the "Rej" microcodes (F3DLX.Rej, F3DFLX.Rej, F3DEX.Rej) as pure C on plain structs
 *
 * The Rej builds are F3DEX2-class microcode that rejects instead of clipping (docs/ucode-rej.md): a vertex outside the guard band is not clipped but marked, and every triangle
 * that uses a marked vertex is dropped; the vertex record has 20 bytes and no clip coordinates, the triangle code is the F3DEX2 setup without the clipper. The vertex pipeline up to
 * the clip coordinates and the lighting are the F3DEX2 ones (f3dex2_fx.h); this header has what is different: the projection and the record, the triangle decisions.
 *
 * Every function has a scalar form (rej_*_scalar, the reference, plain C11); on aarch64 the vertex and the triangle forms have a NEON form with identical results where one exists
 * (tools/hle/test_rej.c). tools/hle/test_rej.c also checks the closed forms against the microcode (rsp2c oracles, docs/testing.md, "Rej").
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_REJ_FX_H
#define ARMLET_REJ_FX_H

#include <stdint.h>

#include "f3dex2_fx.h"

#ifdef __cplusplus
extern "C" {
#endif

/* a vertex record as G_VTX leaves it in its slot (DMEM +0 .. +0x13) */
struct rej_vtx {
    uint8_t col[4];                                 /* r g b a (+0): lit, with the fog value in a when fog is on */
    int16_t s, t;                                   /* scaled texture coordinates (+4) */
    int16_t sx;                                     /* screen x (+8) */
    int16_t sy;                                     /* screen y with the two low bits cleared (+10): 0x7ffc is the "rejected" mark */
    int16_t szi;                                    /* screen z, integer part as computed (+12) */
    uint16_t szf;                                   /* its fraction (+14) */
    int16_t rwi;                                    /* 1/W integer part (+16) */
    uint16_t rwf;                                   /* fraction (+18) */
};
/* flags of the variants of the Rej texts (f2_layout.rej_flags). REJ_FLAT_A: a flat-shaded triangle takes the alpha of its first vertex too (the others take the alpha of each vertex). REJ_LIT_FLX: the lighting is F3DFLX.Rej's (rej_flx_lit); REJ_LIT_LX: F3DLX.Rej 2.03's (the F3DEX2 lighting loop with the unsigned multiply-accumulates: rej_lx_lit_vertex) */
enum { REJ_YQ = 1, REJ_FLAT_A = 2, REJ_LIT_FLX = 4, REJ_LIT_LX = 8 };                                /* the screen y is rounded to whole pixels: the viewport's y translation + 2, the two low bits cleared; the mark of a rejected vertex is then 0x7ffc, else 0x7fff (F3DEX.Rej 2.06 does neither) */
enum { REJ_VSTRIDE = 20, REJ_NSLOTS = 64, REJ_MARK_Y = 0x7ffc };                /* (64 vertex slots of 20 bytes: DMEM 0x410 .. 0x910) */

/* n vertices (no pair partner is needed: the records carry no clip bits). out[i] is the record of in[i]. */
/* F3DFLX.Rej's lighting (docs/ucode-rej.md, section 6.2): an ambient colour and ONE directional light, for the two lane groups of the vector unit (group 1 = the first, even vertex of a pair, group 0 = the odd one) */
struct rej_flx_lit {
    int16_t dir[2][3];                              /* the direction as the setup leaves it (s16 lanes) */
    uint8_t amb[2][3], col[2][3];                   /* the ambient and the light colour */
    int16_t k7;                                     /* v31[7]: the factor the ambient is multiplied by (0x7fff) */
    int texgen;                                     /* geometry mode bit 18: the vertex's alpha is looked up in a table by the normal's dot product with `dir1` (F-Zero X's reflections) */
    int16_t dir1[2][3];                             /* the setup's first direction (the one that is not the light) */
    uint8_t tab[256];                               /* the 256 bytes at DMEM 0x8b0 (loaded by G_DMA_IO), by the signed byte as an unsigned number */
};
void rej_flx_lit_vertex(const struct rej_flx_lit *fl, unsigned g, const int8_t n[3], uint8_t out[4]);       /* out[3]: the table byte with texture generation, else untouched */
/* the setup of one direction d (s16 lanes; the light's bytes << 8): through the upper 3x3 of the model-view matrix, normalised (the reciprocal square root of the sum of squares), scaled by `scale`, s16 lanes */
void rej_light_dir(const struct fx2_mtx *mv, const int16_t d[3], int16_t scale, int16_t out[3]);

/* The lit colour of one vertex of F3DLX.Rej 2.03 (fx2_lit_vertex_scalar with the unsigned multiplies VMULU / VMACU in the colour loop, the lambert terms unmasked); same arguments */
void rej_lx_lit_vertex(const struct fx2_vctx *cx, unsigned g, const int8_t nb[3], uint8_t alpha, uint8_t out[4], int texgen, int16_t sgen[2]);

/* fl: F3DFLX.Rej's lighting when the geometry mode lights (c->mode), else NULL (the F3DEX2 lighting of c->L) */
void rej_vtx_xform_scalar(const struct fx2_vctx *c, unsigned flags, const struct rej_flx_lit *fl, const struct fx2_vin *in, unsigned n, struct rej_vtx *out);
void rej_vtx_xform(const struct fx2_vctx *c, unsigned flags, const struct rej_flx_lit *fl, const struct fx2_vin *in, unsigned n, struct rej_vtx *out);

/* ---- triangles ----------------------------------------------------------------------------------------------------------------------------------------------------------------- */
struct rej_tri_in {
    const struct rej_vtx *v[3];                     /* the vertices in command order */
    uint32_t g;                                     /* the geometry mode word */
    unsigned tex, tile;                             /* the G_TEXTURE bytes at 0x1e7 / 0x1e6 */
    unsigned pass_neg, pass_pos;                    /* the cull table (DMEM 0x402 + the cull byte of the geometry mode) lets a triangle with a negative / positive cross product through */
    unsigned flags;                                 /* REJ_* */
    uint16_t rcp_k, dx_k;                           /* the constants of the triangle code that come from the vector constants (0x1b0 lanes 2 and 5: 8, 0x4000) */
};
/* The decisions and the RDP triangle command. FX2_TRI_DRAW: *len bytes (a multiple of 8, at most 0xb0), in cmd as the host words of DMEM hold them; FX2_TRI_SKIP: nothing is emitted (a vertex was rejected, the
 * triangle is culled or has no area). */
int rej_tri_setup_scalar(const struct rej_tri_in *in, uint32_t *cmd, unsigned *len);
int rej_tri_setup(const struct rej_tri_in *in, uint32_t *cmd, unsigned *len);
int rej_tri_setup_core(const struct rej_tri_in *in, uint32_t *cmd, unsigned *len, int neon);     /* (the NEON attribute forms where there are any) */

#ifdef __cplusplus
}
#endif

#endif
