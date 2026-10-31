/*
 * arMLEt - f3dex_fx_int.h: helpers shared by the f3dex_*_fx.c files (not part of the interface)
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEX_FX_INT_H
#define ARMLET_F3DEX_FX_INT_H

#include "f3dex_fx.h"
#include "hle_fx.h"

#ifndef F3DEX_FX_MUTATE
#define F3DEX_FX_MUTATE 0
#endif
/* deliberate errors for the mutation checks (-DF3DEX_FX_MUTATE=n): ok when n is not selected */
#define FMUT(n, ok, bad) (F3DEX_FX_MUTATE == (n) ? (bad) : (ok))

/* hle_recip_refine written as the 64-bit products they are (the same value, fewer operations: the hot path of every vertex). Checked equal to hle_recip_refine by test_fxn. */
static inline struct hle_pair f3dex_recip_refine(int32_t W)
{
    int64_t r2 = 2 * (int64_t)(int32_t)hle_rcp32(W);
    int32_t R2 = r2 > 0x7fffffffll ? 0x7fffffff : r2 < -0x80000000ll ? (int32_t)0x80000000u : (int32_t)r2;
    int64_t r2h = (int64_t)(R2 >> 16);
    int64_t acc = ((int64_t)R2 * W) >> 16;
    uint16_t plo = hle_uclamp(acc - ((r2h * (int16_t)(W >> 16)) << 16));
    uint16_t phi = (uint16_t)hle_sclamp(acc);
    uint16_t nlo = (uint16_t)(0 - plo);
    uint16_t nhi = (uint16_t)hle_clamp16(2 - (int16_t)phi - (plo != 0));
    int32_t N = (int32_t)((uint32_t)nhi << 16 | nlo);
    acc = ((int64_t)R2 * N) >> 16;
    struct hle_pair r;
    r.f = hle_uclamp(acc - ((r2h * (int16_t)nhi) << 16));
    r.i = hle_sclamp(acc);
    return r;
}

/* the clip flags of a clip-space position: the unscaled compare (f1: bits 0..2 = x y z "X <= -W", 8..10 "X >= W") and the K-scaled compare (f2); wneg = the sign of the integer half of w */
struct f3dex_flagparts {
    unsigned f1, f2;
    int wneg;
};
void f3dex_vtx_flagparts(const struct f3dex_vctx *c, const int16_t Pi[4], const uint16_t Pf[4], struct f3dex_flagparts *fp);

/* what the part of the pipeline after the clip position produces beyond the record: the projected z (fog) */
struct f3dex_tail_out {
    int16_t qzi;
    uint16_t qzf;
};
/* the tail shared by G_VTX and the clipper's new vertices: 1/w, projection, viewport. Pi / Pf = the clip-space position as stored and projected; wi = the integer half of w that the reciprocal uses
 * (Pi[3], except for F3DLX without G_CLIPPING where Pi is the carry-adjusted copy). Fills o->rwi .. o->szf (sx sy sz szf rwi rwf), not
 * the colour, the texture coordinates or the flags. */
void f3dex_vtx_tail(const struct f3dex_vctx *c, const int16_t Pi[4], int16_t wi, const uint16_t Pf[4], struct f3dex_vtx *o, struct f3dex_tail_out *t);
/* the fog byte from the projected z */
uint8_t f3dex_fog_byte(const struct f3dex_vctx *c, int16_t qzi, uint16_t qzf);

#endif
