/*
 * arMLEt - f3dex2_fx_int.h: what the scalar and the NEON forms of f3dex2_fx.h share
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEX2_FX_INT_H
#define ARMLET_F3DEX2_FX_INT_H

#include <stdint.h>

#include "f3dex2_fx.h"

struct fx2_pair {
    int16_t i;
    uint16_t f;
};

/* what the first part of the triangle setup leaves for the attribute part */
struct fx2_tri_state {
    const struct fx2_vtx *H, *M, *L;                /* the sorted vertices */
    int16_t dyMH, nyLH, dxLH, nxMH;
    int32_t area;
    struct fx2_pair slope[3];                       /* DxLDy, DxMDy, DxHDy (lanes 0, 2, 3 of the microcode) */
    struct fx2_pair C0;                             /* -(fraction of H.y / 4) */
};

/* the lit colour (and generated texture coordinates) of one vertex */
void fx2_lit_vertex_scalar(const struct fx2_vctx *cx, unsigned g, const int8_t nb[3], uint8_t alpha, uint8_t out[4], int texgen, int16_t sgen[2]);

int16_t fx2_texgen_linear(const struct fx2_vctx *c, int16_t d);

/* decisions, sort, slopes, the first 32 bytes of the command; FX2_TRI_SKIP / CLIP / DRAW */
int fx2_tri_prepare(const struct fx2_tri_in *in, uint32_t *cmd, struct fx2_tri_state *st);
/* the inputs of the lane loop: att[vertex H M L][lane] (lanes 0..3 colour, 4..6 s t w, 7 z) and R1 = 1 / area; returns the mask of the lanes that are needed */
unsigned fx2_tri_attr_prep(const struct fx2_tri_in *in, const struct fx2_tri_state *st, struct fx2_pair att[3][8], struct fx2_pair *R1);
/* the shade, texture and z blocks; returns the length of the command */
unsigned fx2_tri_attrs_scalar(const struct fx2_tri_in *in, const struct fx2_tri_state *st, uint32_t *cmd);
#ifdef __aarch64__
unsigned fx2_tri_attrs_neon(const struct fx2_tri_in *in, const struct fx2_tri_state *st, uint32_t *cmd);
#endif

#endif
