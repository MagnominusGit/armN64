/*
 * arMLEt - f3dex2_fx.h: the F3DEX2 family's geometry as pure C on plain structs
 *
 * The functions compute what the microcode's matrix, vertex, light and triangle code leaves in DMEM / the RDP buffer (docs/ucode-f3dex2.md), bit for bit, from plain structs: no DMEM
 * addresses, no pcs, no vector registers. The front-end (hle_f3dex2.c) fills the structs from DMEM through f3dex2_dmem.h and writes the results back. The arithmetic is hle_fx.h's (the
 * closed forms of the VU's multiply-accumulate chains over the 48-bit accumulator; Fleet B's), the shape of the interface follows Fleet C's f3d_fx.h.
 *
 * Every function has a scalar form (fx2_*_scalar, the reference, plain C11) and, on aarch64, a NEON form (fx2_*) with identical results; elsewhere fx2_* is the scalar form. tools/hle/test_fx2.c
 * checks the scalar forms against the interpreter running the real microcode on captured tasks and the NEON forms against the scalar forms.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEX2_FX_H
#define ARMLET_F3DEX2_FX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- matrices: s15.16 as the integer parts and the fractions (the DMEM and RDRAM format: 16 integer halfwords, then 16 fraction halfwords) ------------------------------------ */
struct fx2_mtx {
    int16_t i[4][4];
    uint16_t f[4][4];
};
/* out = a x b, each product the full 32 x 32 chain, the four terms in the 48-bit accumulator; rows 0 and 1 of the fraction are read after the last term, rows 2 and 3 before its integer product */
void fx2_mtx_mul_scalar(const struct fx2_mtx *a, const struct fx2_mtx *b, struct fx2_mtx *out);
void fx2_mtx_mul(const struct fx2_mtx *a, const struct fx2_mtx *b, struct fx2_mtx *out);

/* ---- lights ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ */
#define FX2_MAX_LIGHTS 7
#define FX2_LSLOTS (FX2_MAX_LIGHTS + 2)             /* slot 0 and 1 hold the look-at vectors, then the lights, then the ambient slot */
/* The first G_VTX after the lights changed transforms the direction of every slot by the upper 3x3 of the model-view matrix, scales it to unit length and stores it as signed bytes.
 * raw[m] = the direction bytes of slot m - 1 (m = 0 .. nlights + 1); out[m][0..2] = the bytes stored in slot m. scale = the halfword at 0x1c6. */
void fx2_light_setup_scalar(const struct fx2_mtx *mv, const int8_t raw[][3], unsigned nlights, int16_t scale, int8_t out[][3]);
void fx2_light_setup(const struct fx2_mtx *mv, const int8_t raw[][3], unsigned nlights, int16_t scale, int8_t out[][3]);

struct fx2_lights {
    unsigned top;                                   /* the slot the lighting loop starts at: nlights + 1 */
    int16_t d[2][FX2_LSLOTS][4];                    /* per lane group (0: the pair's odd vertex, 1: its even vertex), per slot: the direction as the vector lane holds it (byte << 8); [3] = 0 */
    int16_t c[2][FX2_LSLOTS][4];                    /* the colour (byte << 7); [3] = 0 */
};

/* ---- vertices ----------------------------------------------------------------------------------------------------------------------------------------------------------------- */
struct fx2_vin {                                    /* an input vertex (RDRAM layout, 16 bytes: the NEON form loads it as one vector) */
    int16_t x, y, z;
    int16_t pad;                                    /* the flag halfword of the record: unused */
    int16_t s, t;
    uint8_t c[4];                                   /* r g b a, or (lighting) nx ny nz (signed bytes) and a */
};
struct fx2_vtx {                                    /* a vertex record as G_VTX leaves it in its slot (DMEM +0 .. +0x27) */
    int16_t cx[4];                                  /* clip X Y Z W, integer parts (+0) */
    uint16_t cf[4];                                 /* fractions (+8) */
    uint8_t col[4];                                 /* r g b a as stored (+0x10): lit, with the fog value in a when fog is on */
    int16_t s, t;                                   /* scaled texture coordinates (+0x14) */
    int16_t sx, sy;                                 /* screen x y (+0x18) */
    int16_t szi;                                    /* screen z, integer part, negatives set to 0 (+0x1c; F3DEX2 2.04: as computed) */
    uint16_t szf;                                   /* its fraction (+0x1e) */
    int16_t rwi;                                    /* 1/W integer part (+0x20) */
    uint16_t rwf;                                   /* fraction (+0x22) */
    uint32_t flags;                                 /* the clip-flag word (+0x24): the compare of x y z w against w in the low halfword, against the guard-band w in the high one, pair-merged */
};
/* geometry mode bits (the high halfword of the word at 0x1ec: DMEM16[0x1ec] = mode >> 16) */
enum { FX2_FOG = 1, FX2_LIGHTING = 2, FX2_TEXGEN = 4, FX2_TEXGEN_LINEAR = 8 };

struct fx2_vctx {
    struct fx2_mtx mp;                              /* the model-view-projection matrix */
    int16_t scale[4], trans[4];                     /* the viewport; lane 1 of the scale already negated, lane 3 the fog multiplier / offset */
    uint16_t pn;                                    /* the perspective normalisation */
    uint16_t ratio;                                 /* the guard-band ratio (G_MW_CLIP) */
    uint16_t sc_s, sc_t;                            /* the G_TEXTURE scales */
    uint16_t c31_3;                                 /* v31[3] = 0x7f00 */
    int16_t tg[3];                                  /* the linear texture generation polynomial (DMEM 0x1d0 ..) */
    int16_t c31_7;                                  /* v31[7] = 0x7fff */
    unsigned mode;                                  /* FX2_FOG | FX2_LIGHTING | FX2_TEXGEN */
    unsigned gen204;                                /* F3DEX2 2.04 / 2.04H: the stored screen z keeps its sign (the later builds set a negative integer part to 0), and the linear texture generation's
                                                     * polynomial has 0x4000 in place of x in its linear term (the lighting overlay's `VMUDH v22` that the later builds changed to v21 overwrites x) */
    struct fx2_lights L;
};

/* n vertices; in[] holds n + (n & 1) entries (an odd count: the pair's second vertex is computed from the next 16 bytes of DMEM, only its clip bits are used). out[i] is the record of in[i]. */
void fx2_vtx_xform_scalar(const struct fx2_vctx *c, const struct fx2_vin *in, unsigned n, struct fx2_vtx *out);
void fx2_vtx_xform(const struct fx2_vctx *c, const struct fx2_vin *in, unsigned n, struct fx2_vtx *out);

/* ---- triangles ----------------------------------------------------------------------------------------------------------------------------------------------------------------- */
enum { FX2_TRI_SKIP = 0, FX2_TRI_DRAW = 1, FX2_TRI_CLIP = 2 };
struct fx2_tri_in {
    const struct fx2_vtx *v[3];                     /* the vertices in command order */
    const struct fx2_vtx *flat;                     /* the vertex whose colour a flat-shaded triangle takes, or NULL: v[0]. The microcode keeps the first vertex of the triangle command for the triangles the clipper makes of it */
    uint32_t g;                                     /* the geometry mode word */
    unsigned tex, tile;                             /* the G_TEXTURE bytes at 0x1e7 / 0x1e6 */
    uint32_t reject_mask;                           /* a triangle whose three flag words have a common bit in it is rejected (the build's) */
    uint32_t clip_mask;                             /* flag bits that make the clipper run (DMEM32[0x3cc]; 0: none) */
    uint32_t cull;                                  /* the cull table word for the geometry mode's cull bits, word at 0x3c2 + byte at 0x1ee */
    /* constants of the triangle code that the builds take from different lanes of the constant vectors (DMEM 0x1b0 / 0x1c0); the F3DZEX code has no mask on the slope fractions (frac_mask 0xffff) */
    uint16_t rcp_k;                                 /* the reciprocal of a dy is scaled by it (0x20; F3DZEX 8) */
    uint16_t dx_k;                                  /* dx is scaled by it (0x1000; F3DZEX 0x4000) */
    int16_t slope_lim;                              /* the slope's integer part is clamped to +-lim (0x100; F3DZEX 0x1cc) */
    uint16_t frac_mask;                             /* the slope fraction the edge start uses is masked by it (0xfff8) */
};
/* The decision and the RDP triangle command: FX2_TRI_DRAW: *len bytes (a multiple of 8, at most 0xb0), in cmd as the host words of DMEM hold them (word k = the big-endian value of bytes 4k .. 4k+3); FX2_TRI_SKIP: nothing is emitted (rejected, culled, degenerate); FX2_TRI_CLIP: the clipper runs. */
int fx2_tri_setup_scalar(const struct fx2_tri_in *in, uint32_t *cmd, unsigned *len);
int fx2_tri_setup(const struct fx2_tri_in *in, uint32_t *cmd, unsigned *len);

/* ---- the clipper ------------------------------------------------------------------------------------------------------------------------------------------------------------ */
#define FX2_CLIP_MAX_NEW 12             /* the microcode keeps the new vertices in the 480 bytes at 0x9c8 */
struct fx2_clip_ctx {
    const struct fx2_vctx *v;           /* the viewport, perspective norm and guard-band ratio the new vertices are completed with */
    int16_t plane[6][4];                /* the six planes in the order the code takes them (DMEM 0x1a8, 0x1a0, 0x198, 0x190, 0x188, 0x180): x y z w coefficients */
    uint32_t mask[6];                   /* the flag bits that say "outside" (DMEM 0x40c, 0x408, ... 0x3f8) */
    /* The clipper of F3DEX2 2.04 / 2.04H differs from the later builds' in two places (docs/ucode-f3dex2-dialects.md, section 5): */
    unsigned sign0;                     /* the scale of the intersection's reciprocal has the sign of the distance difference's integer part, and is 0 when that is 0 (later: the sign of `int | 1`, never 0) */
    unsigned fan_l0;                    /* the polygon is drawn as the fan around its first vertex, last triangle first: (L[0], L[n-2-k], L[n-1-k]) (later: (L[k], L[k+1], L[n-1])) */
};
struct fx2_clip_out {
    unsigned ntri;                      /* 0: the whole triangle was discarded */
    unsigned nnew;                      /* new vertices created (also when the triangle ends up discarded) */
    struct fx2_vtx nv[FX2_CLIP_MAX_NEW];
    const struct fx2_vtx *tri[7][3];    /* the fan triangles in submission order: vertices as pointers to the inputs or to nv[] */
};
/* Clips the triangle v[0..2] (command order) against the six planes (Sutherland-Hodgman, as the microcode's vertex lists), completes the new vertices with the vertex pipeline from their clip
 * coordinates (the colour and texture coordinates interpolated), and returns the fan (L[k], L[k+1], L[last]). Returns nonzero when more than FX2_CLIP_MAX_NEW vertices would be needed. */
int fx2_clip_tri_scalar(const struct fx2_clip_ctx *c, const struct fx2_vtx *const v[3], struct fx2_clip_out *o);
int fx2_clip_tri(const struct fx2_clip_ctx *c, const struct fx2_vtx *const v[3], struct fx2_clip_out *o);

#ifdef __cplusplus
}
#endif

#endif
