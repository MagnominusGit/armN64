/*
 * arMLEt - f3d_fx_int.h: helpers shared by the f3d_*_fx.c files (not part of the interface)
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3D_FX_INT_H
#define ARMLET_F3D_FX_INT_H

#include "hle_fx.h"

/* hle_recip_refine written as the 64-bit products they are (the same value, fewer operations: the hot path of every vertex): R1 = rcp32(W); R2 = 2 R1 with the halves saturating (here: as one s32);
 * P = R2 W >> 16 (the low word read before the high product is added); N = 2 - P; R = R2 N >> 16 (the same rule). Checked equal to hle_recip_refine by test_fx / test_fxn. */
static inline struct hle_pair f3d_recip_refine(int32_t W)
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

#endif
