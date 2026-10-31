/*
 * arMLEt - f3d_fx.h: the Fast3D (F3D family) geometry functions as pure C on plain structs
 *
 * The functions compute what the microcode's vertex, light, triangle, matrix and clip code computes (docs/ucode-f3d.md), bit for bit, from plain structs: no DMEM
 * addresses, no pcs, no vector registers. A hook that replaces the microcode's code at a pc and a front-end that runs a whole task both fill the structs from DMEM (f3d_dmem.h)
 * and write the results back. The arithmetic is hle_fx.h's (the closed forms of the VU's multiply-accumulate chains over the 48-bit accumulator).
 *
 * Every function has a scalar form (f3d_*_scalar, the reference, plain C11) and, on aarch64, a NEON form (f3d_*) with identical results; on other targets f3d_* is the scalar form.
 * The scalar forms are checked against the JIT running the real microcode (tools/f3d/test_fx.cpp), the NEON forms against the scalar forms.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3D_FX_H
#define ARMLET_F3D_FX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- matrices: s15.16 as the integer parts and the fractions (the DMEM and RDRAM format: 16 integer halfwords, then 16 fraction halfwords) ------------------------------------ */
struct f3d_mtx {
    int16_t i[4][4];
    uint16_t f[4][4];
};

/* out = a x b (the row-vector convention of G_MTX: the new matrix is the first operand): out[r][c] = sum over k of a[r][k] * b[k][c], each product term as the microcode's chain forms it, the sum
 * in the 48-bit accumulator, the integer part clamped to s16 and the fraction clamped to 0 / 0xffff when the 32-bit value overflows. out may alias a or b. */
void f3d_mtx_mul_scalar(const struct f3d_mtx *a, const struct f3d_mtx *b, struct f3d_mtx *out);
void f3d_mtx_mul(const struct f3d_mtx *a, const struct f3d_mtx *b, struct f3d_mtx *out);

/* ---- vertices ------------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* an input vertex (RDRAM layout, host byte order values) */
struct f3d_vin {
    int16_t x, y, z;
    int16_t s, t;
    uint8_t c[4];                   /* r g b a, or (lighting) nx ny nz (signed bytes) and a */
};

/* a vertex record as G_VTX leaves it in the vertex buffer (DMEM +0 .. +37; the microcode's +38 / +39 are junk) */
struct f3d_vtx {
    int16_t cx[4];                  /* clip X Y Z W, integer parts (+0) */
    uint16_t cf[4];                 /* fractions (+8) */
    uint8_t col[4];                 /* r g b a as stored (+16): lit, with the fog value in a when fog is on */
    int16_t s, t;                   /* scaled texture coordinates (+20) */
    int16_t sx, sy, sz;             /* screen x y z (+24); x, y in 10.2 pixels */
    uint16_t szf;                   /* the fraction of screen z (+30) */
    int16_t rwi;                    /* 1/W' integer part (+32) */
    uint16_t rwf;                   /* fraction (+34) */
    uint16_t flags;                 /* clip flags (+36) */
};

/* the clip flag word variants: SM64's, and Pilotwings' (bit 6, the near-plane flag, is replaced by "W < 0" and carried by the second vertex of each pair only; the first vertex's W<0 lands in bit 2 of the second) */
enum { F3D_FLAGS_SM64 = 0, F3D_FLAGS_PW = 1 };

/* geometry mode bits the vertex code reads */
enum { F3D_G_ZBUFFER = 0x1, F3D_G_TEXTURE = 0x2, F3D_G_SHADE = 0x4, F3D_G_SMOOTH = 0x200, F3D_G_CULL_FRONT = 0x1000, F3D_G_CULL_BACK = 0x2000,
       F3D_G_FOG = 0x10000, F3D_G_LIGHTING = 0x20000, F3D_G_TEXGEN = 0x40000, F3D_G_TEXGEN_LINEAR = 0x80000 };

#define F3D_MAX_SLOTS 16            /* light slots above the two look-at slots: nslot <= 16 (DMEM 0x12c & 0xfff <= 512) */

/* the state G_VTX reads (everything the microcode takes from DMEM) */
struct f3d_vctx {
    struct f3d_mtx mp;              /* the model-view-projection matrix (DMEM 0x3e0) */
    int16_t vscale[4], vtrans[4];   /* the viewport (0x320, 0x328) as G_MOVEMEM stored it (not yet multiplied by the sign vector) */
    int16_t vsign[4];               /* the sign vector (0x50): 1, -1, 1, 1 */
    int16_t lim[4];                 /* screen clamps (0x28): x, y low / high are lim[1] / lim[0], z low / high lim[3] / lim[2] */
    uint16_t persp;                 /* perspective normalisation (0x112) */
    int16_t clip_k;                 /* the clip ratio (0x76), 1 in practice */
    uint16_t tscale[2];             /* G_TEXTURE scale s, t (0x124, 0x126) */
    uint32_t geom;                  /* geometry mode (0x114) */
    int16_t fog[3];                 /* multiplier, offset, maximum (0x330..) */
    /* lighting (geom & F3D_G_LIGHTING): the microcode walks the slots a = 32 nslot down to 0 (the look-at Y slot); slot k has colour bytes lcol[k] (0x1b0 + 32k, eight bytes: the colour, the colour again for the
     * second vertex of a pair) and the direction bytes ldir[k] (0x1c0 + 32k, four bytes then the same four); the ambient colour is lcol[nslot + 1]. nslot = (D32[0x12c] & 0xfff) / 32 >= 1. */
    unsigned nslot;
    uint8_t lcol[F3D_MAX_SLOTS + 2][8];
    int8_t ldir[F3D_MAX_SLOTS + 1][8];
    int8_t lk[8];                   /* the texgen look-at weights (0x1a0) */
    int16_t tg[3];                  /* the linear texgen polynomial (0xa2, 0xa4, 0xa6) */
    /* the clipper's planes in processing order (z+w, z-w, y-Kw, x-Kw, y+Kw, x+Kw): the coefficient vectors (0x98, 0x90, ..., 0x70) and the flag masks (0xb2, 0xb0, ..., 0xa8) */
    int16_t plane[6][4];
    uint16_t plane_mask[6];
    uint8_t flags_variant;          /* F3D_FLAGS_*: how the clip flag word is formed (docs/ucode-f3d.md 10.5.1) */
    /* the Rare builds (docs/ucode-rare.md): a 32-bit offset vector added to the clip coordinates (the S15.16 sums) before the flags and the projection, VADDC + VADD per lane */
    int has_off;
    int16_t off_i[4];
    uint16_t off_f[4];
    /* Indiana Jones (docs/ucode-indy.md): the fog value from five halfwords (DMEM 0x160): a 32-bit multiplier (fogw[0] : fogw[1]), a 32-bit offset (fogw[2] : fogw[3]) and a 16-bit factor that is also the
     * maximum (fogw[4]); fog_wide selects it in place of fog[] */
    int fog_wide;
    int16_t fogw[5];
    /* the clipper's rounding constant of the weight (the accumulator lane the microcode starts that sum from): 3 unless set (Indy: 1) */
    int clip_acc;
};

/* The F3D vertex pipeline for n vertices: clip coordinates, clip flags, 1/W, screen coordinates, texture scale, lighting, texture generation, fog. out[i] is the record of in[i]. The results do not depend on
 * the other vertices (the microcode's pairing is invisible) except through nothing: every vertex is independent. Returns 0, or nonzero when the state is outside what the code supports
 * (lighting with nslot < 1 or > F3D_MAX_SLOTS). */
int f3d_vtx_xform_scalar(const struct f3d_vctx *c, const struct f3d_vin *in, unsigned n, struct f3d_vtx *out);
int f3d_vtx_xform(const struct f3d_vctx *c, const struct f3d_vin *in, unsigned n, struct f3d_vtx *out);

/* ---- light directions ("relight", overlay 2): the model-space light directions transformed by the model-view matrix ---------------------------------------------------------------- */
/* raw[k][0..2] = the signed direction bytes of slot k (DMEM 0x1b8 + 32k + 0..2), k = 0 .. nslot (slot 0 is the look-at Y vector, slot 1 look-at X). The model-view matrix is mv (DMEM 0x360, 0x380; rows 0..2
 * and columns 0..2 are read). clamp[3] = the constants at DMEM 0x208 (-8191, 8191, 4). out[k][0..2] = the transformed, normalised direction bytes the microcode stores at 0x1c0 + 32k; out[k][3] is the
 * fourth byte, which depends on leftover registers and is never read by the lit loop: set to 0. */
void f3d_relight_scalar(const struct f3d_mtx *mv, const int8_t raw[][3], unsigned nslot, const int16_t clamp[3], int8_t out[][4]);
void f3d_relight(const struct f3d_mtx *mv, const int8_t raw[][3], unsigned nslot, const int16_t clamp[3], int8_t out[][4]);

/* ---- triangles ------------------------------------------------------------------------------------------------------------------------------------------------------------ */
enum { F3D_TRI_OK = 0, F3D_TRI_REJECT = 1, F3D_TRI_CLIP = 2, F3D_TRI_CULL = 3 };

/* the triangle code variants: SM64's, Wave Race's (no front-face culling, no flat shading, the shade block when any of the low three geometry bits is set; docs/ucode-f3d.md 10.5.2) and Rare's (the cull
 * bits mean something else, the texture block is not scaled, no minimum z slope; docs/ucode-rare.md) */
enum { F3D_TRIV_SM64 = 0, F3D_TRIV_WR = 1, F3D_TRIV_RARE = 2, F3D_TRIV_INDY = 3 };

struct f3d_tri_in {
    const struct f3d_vtx *v[3];     /* the three vertices in command order */
    const struct f3d_vtx *flat;     /* the vertex whose colour a flat-shaded triangle uses (the one the flag byte selects) */
    uint32_t geom;                  /* geometry mode */
    uint16_t persp;                 /* perspective normalisation */
    uint8_t tile_byte;              /* DMEM 0x122: level << 3 | tile of G_TEXTURE */
    uint8_t zclamp;                 /* DMEM 0x121 */
    uint16_t reject_mask, clip_mask;/* 0x7070 and 0x4343 (SM64); the build's */
    int vco_carry;                  /* the VCO carry of lane 1 at entry: 0 */
    uint8_t variant;                /* F3D_TRIV_* */
    /* Indiana Jones sorts the vertices by a table (docs/ucode-indy.md): with sort_given ord[] are the indices into v[] of the top, middle and bottom vertex and par is the parity of the swaps (1: the area is
     * negated); no front-face culling, every vertex shades with its own colour */
    int sort_given;
    uint8_t ord[3];
    int par;
};

/* The triangle command: the decision (flags, area, facing) and the RDP triangle bytes. On F3D_TRI_OK, cmd[0 .. *len) (a multiple of 8, at most 0xb0) is the RDP command, big endian. On the other results nothing is
 * written; F3D_TRI_CLIP means the clipper must run (f3d_clip_tri) and feed each of its triangles to f3d_tri_setup_*_noclip. */
int f3d_tri_setup_scalar(const struct f3d_tri_in *in, uint8_t *cmd, unsigned *len);
int f3d_tri_setup(const struct f3d_tri_in *in, uint8_t *cmd, unsigned *len);
/* the same without the flag tests (the clipper's triangles) */
int f3d_tri_setup_noclip_scalar(const struct f3d_tri_in *in, uint8_t *cmd, unsigned *len);
int f3d_tri_setup_noclip(const struct f3d_tri_in *in, uint8_t *cmd, unsigned *len);

/* ---- the clipper ------------------------------------------------------------------------------------------------------------------------------------------------------------ */
#define F3D_CLIP_MAX_NEW 12
struct f3d_clip_out {
    unsigned ntri;                  /* 0: the whole triangle was discarded */
    unsigned nnew;                  /* new vertices created (also when the triangle ends up discarded) */
    struct f3d_vtx nv[F3D_CLIP_MAX_NEW];
    const struct f3d_vtx *tri[7][3];/* the fan triangles in submission order, vertices as pointers to the inputs or to nv[] */
    const struct f3d_vtx *poly[10]; /* f3d_clip_tri_seq: the polygon */
    unsigned npoly;
};
/* Clips the triangle v[0..2] (command order) against the six planes. c supplies the clip ratio, persp, viewport, fog and geometry state the new vertices are completed with; the colours and texture coordinates are
 * interpolated. */
void f3d_clip_tri_scalar(const struct f3d_vctx *c, const struct f3d_vtx *const v[3], struct f3d_clip_out *o);
/* Indiana Jones's clipper (docs/ucode-indy.md): the same cuts and new vertices, but the polygon is walked in the order of the input list (v[0], v[1], v[2]), a new vertex is made where the walk crosses a
 * plane (the one on the inside is the interpolation's anchor), and the fan is (poly[0], poly[i + 1], poly[i + 2]); o->poly / o->npoly is the final polygon, tri[] the fan. The planes are separated by
 * the value of the masked flags, not by a boolean. */
void f3d_clip_tri_seq(const struct f3d_vctx *c, const struct f3d_vtx *const v[3], struct f3d_clip_out *o);
void f3d_clip_tri(const struct f3d_vctx *c, const struct f3d_vtx *const v[3], struct f3d_clip_out *o);

#ifdef __cplusplus
}
#endif

#endif
