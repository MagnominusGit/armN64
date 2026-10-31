/*
 * arMLEt - f3dexbg_fx.h: the lit vertices of F3DEXBG.NoN 2.08 (Conker's Bad Fur Day) as pure C on plain structs
 *
 * The microcode's lit G_VTX is the F3DEX2 vertex pipeline (matrix, projection, clip flags, viewport: f3dex2_fx.h, unchanged) with another lighting stage. That stage works in the view space the
 * game's projection comes from, on 48-byte light slots with a position and an attenuation, and takes the vertex normals from a table of the command (G_MOVEMEM 14, DMEM 0x510) and the vertex's
 * pad bytes instead of its colour (docs/ucode-f3dex2-dialects.md, section 2.4). It is computed from the clip coordinates the pipeline leaves in the vertex records, so it is a pass over them:
 * fx2_vtx_xform() with the standard lighting off, then fbg_lit_pass(), which sets the colour and (with texture generation) the texture coordinates.
 *
 * The arithmetic is hle_fx.h's; tools/hle/test_f3dexbg.c checks the NEON vertex form and the scalar one under the pass, and the NEON lit pass (f3dexbg_fx_neon.h, aarch64) against the scalar one; the equivalence with the microcode is docs/testing.md ("Regenerating the microcode oracles") and the replays of docs/testing.md.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEXBG_FX_H
#define ARMLET_F3DEXBG_FX_H

#include <stdint.h>

#include "f3dex2_fx.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FBG_MAX_LIGHTS 11
#define FBG_SLOTS (FBG_MAX_LIGHTS + 2)              /* slots 0 and 1 hold the look-at vectors, the lights use slots 2 .. nl + 1, the ambient colour is in slot nl + 1 */

/* a light slot as the lighting loop reads it, per lane group (group 0 = the pair's odd vertex, group 1 = its even one; the copies are normally equal) */
struct fbg_slot {
    int16_t d[2][3];                                /* direction, byte << 8 (the setup normalises it) */
    int16_t pos[2][4];                              /* position x y z w, view space */
    int16_t col[2][3];                              /* colour, byte << 7 */
    uint8_t atten;                                  /* attenuation: (a >> 4) + (a & 15) / 16 is the numerator of the 1 / distance^2 falloff */
};

struct fbg_ctx {
    unsigned top;                                   /* nlights + 1: the slot of the ambient colour; the lights are the slots top .. 2 (direction of slot m, colour of slot m - 1) */
    unsigned positional;                            /* geometry mode bit 6 (G_LIGHTING_POSITIONAL): the lights are attenuated by their distance and lambert-weighted, else the attenuated colours are added */
    struct fbg_slot s[FBG_SLOTS];
    /* the reconstruction of the view-space position from the clip coordinates: p * k per axis (32-bit k = khi:klo), z + zoff first */
    int16_t khi[2][4], klo[2][4];
    int16_t zoff[2][4];
};

/* per input vertex: the normal (table bytes x y, pad byte z), the flag byte (negative: not lit, the colour stays), the colour bytes (the base colour the lit result is multiplied by) */
struct fbg_vin {
    int8_t n[3];
    int8_t flag;
    uint8_t col[4];
};

/* out[i] is the record fx2_vtx_xform() made for the vertex with the standard lighting off (its colour the input colour); sets the colour (or leaves the input colour when the flag says so) and, when
 * the geometry mode has texture generation, s and t. n vertices; lane group of vertex i: (i & 1) ? 0 : 1. */
void fbg_lit_pass(const struct fx2_vctx *vc, const struct fbg_ctx *bc, const struct fbg_vin *in, unsigned n, struct fx2_vtx *out);
/* the scalar reference form (fbg_lit_pass is the NEON form on aarch64, tools/hle/test_f3dexbg.c checks it against this) */
void fbg_lit_pass_scalar(const struct fx2_vctx *vc, const struct fbg_ctx *bc, const struct fbg_vin *in, unsigned n, struct fx2_vtx *out);

/* The fractions of the clip coordinates of an input vertex (what the microcode's pair loop leaves in its scratch words for the vertex it computes only for the pair's sake). */
void fbg_clip_frac(const struct fx2_vctx *vc, const struct fx2_vin *vi, uint16_t cf[4]);

#ifdef __cplusplus
}
#endif

#endif
