/*
 * arMLEt - f3d_relight_fx.c: the F3D light re-transform (overlay 2) as a pure function; docs/ucode-f3d.md 7.2
 *
 * For every light slot: the three signed direction bytes are scaled to 128 * byte, multiplied by the 3x3 part of the model-view matrix (transposed sense: out[l] = sum_c M[l][c] * d[c], an s15.16
 * result), normalised with the reciprocal square root of the squared length (the ROM table and the one-step scaling of the divide unit), clamped and stored as signed bytes.
 * The arithmetic is the verified model of the spec (relight_fast); the fourth byte of a slot depends on leftover registers and is not computed (out[k][3] = 0).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "f3d_fx.h"
#include "hle_fx.h"

#ifndef F3D_FX_MUTATE
#define F3D_FX_MUTATE 0
#endif

static inline int clamp16_64(int64_t x) { return x > 32767 ? 32767 : x < -32768 ? -32768 : (int)x; }
/* VMADN-style low word: the low 16 bits while the accumulator fits in 32 bits, else 0xffff / 0 */
static inline uint16_t sat_lo(int64_t acc) { return acc >= 0x80000000LL ? 0xffff : acc < -0x80000000LL ? 0 : (uint16_t)acc; }
static inline int sat_md(int64_t acc) { return clamp16_64(acc >> 16); }

void f3d_relight_scalar(const struct f3d_mtx *mv, const int8_t raw[][3], unsigned nslot, const int16_t clamp[3], int8_t out[][4])
{
    for (unsigned k = 0; k <= nslot; k++) {
        int dd[3];
        for (int c = 0; c < 3; c++)
            dd[c] = 128 * raw[k][c];
        int hi[3];
        uint32_t lo[3];
        for (int l = 0; l < 3; l++) {
            int64_t S = 0, I = 0;
            for (int c = 0; c < 3; c++) {
                S += (int64_t)mv->f[l][c] * dd[c];
                I += (int64_t)mv->i[l][c] * dd[c];
            }
            int64_t T = 2 * ((int64_t)clamp16_64(S >> 16) + I);
            hi[l] = sat_md(T);
            lo[l] = sat_lo(T);
        }
        /* squared length: A3 = floor(lo^2 / 65536) + 2 hi lo is read for its low word before hi^2 is added */
        uint32_t sq_lo[3];
        int sq_hi[3];
        for (int l = 0; l < 3; l++) {
            int64_t A3 = ((int64_t)lo[l] * lo[l] >> 16) + 2 * (int64_t)hi[l] * (int64_t)lo[l];
            sq_lo[l] = sat_lo(A3);
            sq_hi[l] = sat_md(A3 + (int64_t)hi[l] * hi[l] * 65536);
        }
        uint32_t s01 = sq_lo[0] + sq_lo[1];
        int c01 = s01 > 0xffff;
        int h01 = clamp16_64((int64_t)sq_hi[0] + sq_hi[1] + c01);
        uint32_t s2 = sq_lo[2] + (s01 & 0xffff);
        int c2 = s2 > 0xffff;
        int h2 = clamp16_64((int64_t)sq_hi[2] + h01 + (F3D_FX_MUTATE == 1 ? 0 : c2));
        int32_t R = (int32_t)hle_rsq32((int32_t)((uint32_t)(uint16_t)h2 << 16 | (s2 & 0xffff)));
        int64_t Kacc = (((int64_t)(R & 0xffff) * 0x40) >> 16) + (int64_t)(int16_t)(R >> 16) * 0x40;
        int Khi = sat_md(Kacc);
        uint32_t Klo = sat_lo(Kacc);
        for (int l = 0; l < 3; l++) {
            int64_t N = (((int64_t)lo[l] * Klo) >> 16) + (int64_t)hi[l] * Klo + (int64_t)lo[l] * Khi + (int64_t)hi[l] * Khi * 65536;
            int16_t x = (int16_t)sat_lo(N);
            if (x < clamp[0]) x = clamp[0];
            if (x > clamp[1]) x = clamp[1];
            out[k][l] = (int8_t)((int16_t)(uint16_t)((uint16_t)x * (int64_t)clamp[2]) >> (F3D_FX_MUTATE == 2 ? 7 : 8));
        }
        out[k][3] = 0;
    }
}

/* the NEON forms of these functions are not written yet: the scalar form serves */
void f3d_relight(const struct f3d_mtx *mv, const int8_t raw[][3], unsigned nslot, const int16_t clamp[3], int8_t out[][4]) { f3d_relight_scalar(mv, raw, nslot, clamp, out); }
