/*
 * arMLEt - f3dex_fx.h: the F3DEX 1.x / F3DLX geometry functions as pure C on plain structs
 *
 * The functions compute what the microcode's matrix, vertex, light, triangle and clip code computes (docs/ucode-f3dex.md, docs/ucode-f3dex/), bit for bit, from plain structs: no DMEM
 * addresses, no pcs, no vector registers. A whole-task front-end fills the structs from its state (f3dex_dmem.h does it from a DMEM image) and writes the results back. The arithmetic is
 * hle_fx.h's: the closed forms of the VU's multiply-accumulate chains over the 48-bit accumulator.
 *
 * Every function has a scalar form (f3dex_*_scalar, the reference, plain C11) and, where it is a hot path, a NEON form (f3dex_*) with identical results; without NEON f3dex_* is the scalar form.
 *
 * The builds are parameters (struct f3dex_build): F3DEX 1.21 / 1.23 / 0.95 are one arithmetic; F3DEX.NoN differs in the clip flags; F3DLX in the clip flags (optional), the rounded screen y and the
 * triangle code. Mapping the ucode id to a build is the caller's job (f3dex_build_of_ucode).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEX_FX_H
#define ARMLET_F3DEX_FX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- the builds -------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
struct f3dex_layout;
struct f3dex_build {
    uint16_t reject_mask;           /* trivial-reject clip-flag mask: 0x7070, NoN 0x7030 */
    uint8_t non;                    /* F3DEX.NoN: no "z <= -w" bit in the first compare; the sign of w instead (clip flags of the second vertex of a pair) */
    uint8_t dlx;                    /* F3DLX: G_CLIPPING (geometry bit 23) optional, screen y rounded, the F3DLX triangle code */
    const struct f3dex_layout *lay; /* the DMEM layout (f3dex_dmem.h; not used by the functions of this file, only by the loaders); NULL: f3dex_lay_std */
};
extern const struct f3dex_build f3dex_build_dex;        /* F3DEX 1.21, 1.23, 0.95 */
extern const struct f3dex_build f3dex_build_non;        /* F3DEX.NoN 1.22 */
extern const struct f3dex_build f3dex_build_dlx;        /* F3DLX 1.21, 1.23, 0.95 */
extern const struct f3dex_build f3dex_build_dlx_non;    /* F3DLX.NoN 1.21, 1.23 */
/* the ucode ids of gfx_ucode.c; returns NULL for any other ucode */
const struct f3dex_build *f3dex_build_of_ucode(uint64_t id);

/* ---- matrices: s15.16 as the integer parts and the fractions (the DMEM and RDRAM format: 16 integer halfwords, then 16 fraction halfwords) ------------------------------------------- */
struct f3dex_mtx {
    int16_t i[4][4];
    uint16_t f[4][4];
};

/* out = a x b (row vectors: out[r][c] = clamp32(sum over k of floor(a[r][k] * b[k][c] / 65536)), each term floored separately, the sum in the 48-bit accumulator, the integer half
 * clamped to s16, the fraction to 0 / 0xffff when the 32-bit value overflows). G_MTX multiply: a = the command's matrix, b = the current one; the MVP: a = modelview, b = projection.
 * out may alias a or b. */
void f3dex_mtx_mul_scalar(const struct f3dex_mtx *a, const struct f3dex_mtx *b, struct f3dex_mtx *out);
void f3dex_mtx_mul(const struct f3dex_mtx *a, const struct f3dex_mtx *b, struct f3dex_mtx *out);

/* ---- vertices ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* an input vertex (the 16 bytes of the display list; host-order values) */
struct f3dex_vin {
    int16_t x, y, z;
    int16_t s, t;
    uint8_t c[4];                   /* r g b a, or (lighting) nx ny nz (signed bytes) and a */
};

/* a vertex record as G_VTX leaves it in the vertex buffer (the microcode's 40 bytes; +38 is a leftover and not modelled) */
struct f3dex_vtx {
    int16_t cx[4];                  /* clip X Y Z W, integer parts (+0) */
    uint16_t cf[4];                 /* fractions (+8) */
    uint8_t col[4];                 /* r g b a (+16): lit, with the fog value in a when fog is on */
    int16_t s, t;                   /* scaled texture coordinates (+20) */
    int16_t sx, sy, sz;             /* screen x y z (+24); x, y in 10.2 pixels; F3DLX: sy = (y + 2) & ~3 */
    uint16_t szf;                   /* the fraction of screen z (+30) */
    int16_t rwi;                    /* 1/W' integer part (+32; raw: may be negative) */
    uint16_t rwf;                   /* fraction (+34) */
    uint16_t flags;                 /* clip flags (+36); F3DLX without G_CLIPPING: 0 (the microcode leaves junk there that no consumer reads except G_CULLDL) */
};

enum { F3DEX_G_ZBUFFER = 0x1, F3DEX_G_TEXTURE = 0x2, F3DEX_G_SHADE = 0x4, F3DEX_G_SMOOTH = 0x200, F3DEX_G_CULL_FRONT = 0x1000, F3DEX_G_CULL_BACK = 0x2000,
       F3DEX_G_FOG = 0x10000, F3DEX_G_LIGHTING = 0x20000, F3DEX_G_TEXGEN = 0x40000, F3DEX_G_TEXGEN_LINEAR = 0x80000, F3DEX_G_CLIPPING = 0x800000,
       F3DEX_G_DEFINED = 0x1 | 0x2 | 0x4 | 0x200 | 0x1000 | 0x2000 | 0x10000 | 0x20000 | 0x40000 | 0x80000 | 0x800000 };

#define F3DEX_MAX_REC 16            /* light records above the look-at records: nrec <= F3DEX_MAX_REC (the word at DMEM 0x12c is (nrec) * 32) */

/* the state G_VTX (and the clipper's new vertices) read */
struct f3dex_vctx {
    const struct f3dex_build *b;
    struct f3dex_mtx mp;            /* the model-view-projection matrix (DMEM 0x460) */
    int16_t vscale[4], vtrans[4];   /* the viewport as G_MOVEMEM stored it (DMEM 0x380): not yet multiplied by the sign vector */
    int16_t vsign[4];               /* the sign vector (DMEM 0x50): 1, -1, 1, 1 */
    int16_t lim[4];                 /* screen clamps (DMEM 0x28): x, y high / low are lim[0] / lim[1], z high / low lim[2] / lim[3] (4090, -4090, 0x7fff, 0) */
    uint16_t persp;                 /* perspective normalisation (DMEM 0x112) */
    int16_t clip_k;                 /* the clip ratio (DMEM 0x76) */
    uint16_t tscale[2];             /* G_TEXTURE scale s, t (DMEM 0x124, 0x126) */
    uint32_t geom;                  /* geometry mode (DMEM 0x114) */
    int16_t fog[3];                 /* multiplier, offset, maximum (DMEM 0x390) */
    uint8_t vco_carry;              /* F3DLX without G_CLIPPING, unlit: the VCO carry bits of the vector unit at entry (lane n = bit n) that the microcode adds to the integer position of the
                                       first pair; 0 in the normal flow (the previous command's last compare cleared them). Unobservable for the triangle code except through the w the texture scale uses. */
    /* lighting (geom & F3DEX_G_LIGHTING): records 0 .. nrec (0 = look-at Y, 1 = look-at X, 2.. the lights), the ambient record nrec + 1. The microcode walks records nrec down to 0. Per record: the colour bytes
     * lcol[k][0..3] (vertex A) and [4..7] (vertex B: the game's data duplicates them), the model-space direction bytes ldir[k][0..2] (A) and [4..6] (B) (DMEM 0x1c0 + 32k: as the relight left them).
     * nrec = (DMEM[0x12c] & 0xfff) / 32. */
    unsigned nrec;
    uint8_t lcol[F3DEX_MAX_REC + 2][8];
    int8_t ldir[F3DEX_MAX_REC + 1][8];
    uint8_t lk[8];                  /* the texgen weights (DMEM 0x1a0: bytes, A 0..3, B 4..7) */
    int16_t tg[3];                  /* the linear texgen polynomial (DMEM 0xa2, 0xa4, 0xa6) */
    int16_t tgbias;                 /* 0x4000 (DMEM 0x38) */
    /* the clipper's planes in processing order (near, far, y+, x+, y-, x-): coefficient vectors (x y z w) and flag masks */
    int16_t plane[6][4];
    uint16_t plane_mask[6];
};

/* The G_VTX pipeline for n vertices of one command: clip coordinates, clip flags, 1/W, screen coordinates, lighting, texgen, fog, texture scale. out[i] is the record of in[i]. The microcode works on
 * pairs (in[0], in[1]), (in[2], in[3]), ...: the vertex index parity selects the second half of the light records, and F3DEX.NoN flags couple a pair.
 * Returns 0, or nonzero when the state is outside what the function models: lighting with nrec > F3DEX_MAX_REC, or texgen with lighting on and nrec == 0 (the microcode then reads a stale register),
 * or n == 0 / n > 32. */
int f3dex_vtx_xform_scalar(const struct f3dex_vctx *c, const struct f3dex_vin *in, unsigned n, struct f3dex_vtx *out);
int f3dex_vtx_xform(const struct f3dex_vctx *c, const struct f3dex_vin *in, unsigned n, struct f3dex_vtx *out);

/* ---- light directions ("relight"): the model-space light directions the lit loop uses, from the model-view matrix ---------------------------------------------------------------------- */
/* raw[k][0..2] = the signed direction bytes of record k (DMEM 0x1b8 + 32k), k = 0 .. nrec (the look-at records and the ambient record included: the microcode transforms every record from nrec down to 0).
 * mv = the current model-view matrix. out[k][0..2] = the transformed, normalised direction bytes (DMEM 0x1c0 + 32k), out[k][3] = 0 (the microcode's fourth byte is a leftover of stale registers that no
 * consumer reads). The constants of DMEM 0x208 are -8191, 8191, 4. */
void f3dex_relight_scalar(const struct f3dex_mtx *mv, const int8_t raw[][3], unsigned nrec, int8_t out[][4]);
void f3dex_relight(const struct f3dex_mtx *mv, const int8_t raw[][3], unsigned nrec, int8_t out[][4]);

/* ---- triangles --------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* CULL: back/front-face culled (the microcode returns with $31 = 0, which ends a G_QUAD); DEGEN: area zero (nothing drawn, $31 not cleared); REJECT: all three vertices outside one plane */
enum { F3DEX_TRI_OK = 0, F3DEX_TRI_REJECT = 1, F3DEX_TRI_CLIP = 2, F3DEX_TRI_CULL = 3, F3DEX_TRI_DEGEN = 4 };

struct f3dex_tri_in {
    const struct f3dex_build *b;
    const struct f3dex_vtx *v[3];   /* the three vertices in command order */
    const struct f3dex_vtx *flat;   /* the vertex whose colour a flat-shaded triangle uses (the command's first vertex; for a clipped triangle's pieces also the original first) */
    uint32_t geom;                  /* geometry mode */
    uint16_t persp;                 /* perspective normalisation (DMEM 0x112) */
    uint8_t tile_byte;              /* DMEM 0x122: level << 3 | tile of G_TEXTURE */
};

/* The triangle command: the decision (flags, area, facing) and the RDP triangle bytes. On F3DEX_TRI_OK, cmd[0 .. *len) (a multiple of 8: 0x20 + 0x40 shade + 0x40 texture + 0x10 z, at most 0xb0) is the RDP
 * command, big endian. Otherwise nothing is written. F3DEX_TRI_CLIP means the clipper must run (f3dex_clip_tri) and feed each of its triangles to f3dex_tri_setup_noclip. */
int f3dex_tri_setup_scalar(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len);
int f3dex_tri_setup(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len);
/* the same without the clip-flag tests (the clipper's pieces); the culls and the degenerate test still apply */
int f3dex_tri_setup_noclip_scalar(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len);
int f3dex_tri_setup_noclip(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len);

/* ---- the clipper ----------------------------------------------------------------------------------------------------------------------------------------------------------------- */
#define F3DEX_CLIP_MAX_NEW 12
struct f3dex_clip_out {
    unsigned ntri;                  /* 0: the whole triangle was discarded (no inside vertex for some plane, or fewer than three vertices left) */
    unsigned nnew;                  /* new vertices created (also when the triangle ends up discarded) */
    struct f3dex_vtx nv[F3DEX_CLIP_MAX_NEW];
    const struct f3dex_vtx *tri[7][3];  /* the fan triangles in submission order (v[0], v[1], v[2] of f3dex_tri_in), pointers to the inputs or to nv[] */
};
/* Clips the triangle v[0..2] (command order) against the six planes (docs/ucode-f3dex/clip.md). c supplies the planes, the clip ratio, persp, viewport, fog and geometry state the new vertices are
 * completed with; positions, colours and texture coordinates are interpolated. */
void f3dex_clip_tri_scalar(const struct f3dex_vctx *c, const struct f3dex_vtx *const v[3], struct f3dex_clip_out *o);
void f3dex_clip_tri(const struct f3dex_vctx *c, const struct f3dex_vtx *const v[3], struct f3dex_clip_out *o);

#ifdef __cplusplus
}
#endif

#endif
