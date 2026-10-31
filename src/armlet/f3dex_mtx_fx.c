/*
 * arMLEt - f3dex_mtx_fx.c: the F3DEX 1.x / F3DLX matrix product and the light-direction recomputation (relight)
 *
 * docs/ucode-f3dex/mtx.md: 1.1 (the product, the subroutine at 0x5fc) and 4.2 (the relight, lighting overlay 2). Plain C on the structs of f3dex_fx.h; the sums are hle_fx.h's closed forms.
 *
 * Mutations (-DF3DEX_FX_MUTATE=n, tools/f3dex/fxn_mtx.c and the spec.f3dex.fxmtx.* routines must fail): 20 operands swapped, 21 each term rounded instead of floored, 22 floor of the sum instead of
 * the sum of the floors, 23 relight upper clamp 8127 instead of 8191 (8190 would be an equivalent mutant: the byte is v >> 6), 24 relight direction scale 127 instead of 128, 25 relight Q integer half
 * without the high x high term, 26 relight output shift 7, 27 relight reads the matrix not transposed, 28 relight skips the carry of the second sum, 29 relight lower clamp -8127,
 * 33 relight Q fraction read after the high x high term (the "lo3" rule); 30 .. 32 the NEON form (30 rounding shift of a term (a logical shift is an equivalent mutant), 31 no 48-bit wrap, 32 fraction saturation to 0x7fff).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "f3dex_fx_int.h"

/* ---- the matrix product (1.1) ------------------------------------------------------------------------------------------------------------------------------------------------- */
void f3dex_mtx_mul_scalar(const struct f3dex_mtx *a, const struct f3dex_mtx *b, struct f3dex_mtx *out)
{
    const struct f3dex_mtx *x = FMUT(20, a, b), *y = FMUT(20, b, a);       /* mutation 20: operands swapped */
    struct f3dex_mtx r;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            hle_acc acc = 0;
            int64_t exact = 0;
            for (int k = 0; k < 4; k++) {
                int32_t A = hle_fx(x->i[i][k], x->f[i][k]), B = hle_fx(y->i[k][j], y->f[k][j]);
                acc += FMUT(21, hle_mac_32x32(A, B), hle_wrap48((((int64_t)A * B) + 0x8000) >> 16));      /* mutation 21: rounded terms */
                exact += (int64_t)A * B;
            }
            acc = FMUT(22, acc, hle_wrap48(exact >> 16));                                                /* mutation 22: floor of the sum */
            acc = hle_wrap48(acc);
            r.i[i][j] = hle_sclamp(acc);
            r.f[i][j] = hle_uclamp(acc);
        }
    *out = r;
}

#ifdef __ARM_NEON
#include <arm_neon.h>

/* One row of the result at a time, the four columns in two int64x2 accumulators: the 32 x 32 -> 64-bit products (vmull) of the row's elements, broadcast, with the rows of b (s15.16 as int32), each
 * shifted right (arithmetic, so floored) by 16 before it is added. The accumulator wraps to 48 bits, then the integer half saturates to s16 and the fraction saturates to 0 / 0xffff when the sum
 * does not fit in s32. */
void f3dex_mtx_mul(const struct f3dex_mtx *a, const struct f3dex_mtx *b, struct f3dex_mtx *out)
{
    struct f3dex_mtx r;
    int32x4_t B[4];
    for (int k = 0; k < 4; k++)
        B[k] = vreinterpretq_s32_u32(vorrq_u32(vshll_n_u16(vld1_u16((const uint16_t *)b->i[k]), 16), vmovl_u16(vld1_u16(b->f[k]))));
    for (int i = 0; i < 4; i++) {
        int64x2_t acc0 = vdupq_n_s64(0), acc1 = vdupq_n_s64(0);
        for (int k = 0; k < 4; k++) {
            int32_t A = (int32_t)((uint32_t)(uint16_t)a->i[i][k] << 16 | a->f[i][k]);
            int64x2_t p0 = vmull_n_s32(vget_low_s32(B[k]), A), p1 = vmull_n_s32(vget_high_s32(B[k]), A);
            /* mutation 30: a rounding shift (a logical shift would be an equivalent mutant: the difference is 2^48 per negative term, which the 48-bit wrap removes) */
            if (F3DEX_FX_MUTATE == 30) {
                p0 = vrshrq_n_s64(p0, 16);
                p1 = vrshrq_n_s64(p1, 16);
            } else {
                p0 = vshrq_n_s64(p0, 16);
                p1 = vshrq_n_s64(p1, 16);
            }
            acc0 = vaddq_s64(acc0, p0);
            acc1 = vaddq_s64(acc1, p1);
        }
        if (F3DEX_FX_MUTATE != 31) {                                            /* mutation 31: no wrap to 48 bits */
            acc0 = vshrq_n_s64(vshlq_n_s64(acc0, 16), 16);
            acc1 = vshrq_n_s64(vshlq_n_s64(acc1, 16), 16);
        }
        /* integer half: bits 47..16 saturated to s16 (the narrows saturate s64 -> s32 -> s16) */
        int32x4_t hi32 = vcombine_s32(vqmovn_s64(vshrq_n_s64(acc0, 16)), vqmovn_s64(vshrq_n_s64(acc1, 16)));
        vst1_s16(r.i[i], vqmovn_s32(hi32));
        /* fraction: the low 16 bits when the sum is in [-2^31, 2^31), else 0xffff (positive) / 0 (negative) */
        const int64x2_t bias = vdupq_n_s64(0x80000000ll);
        uint32x4_t ok = vcombine_u32(vmovn_u64(vceqzq_u64(vshrq_n_u64(vreinterpretq_u64_s64(vaddq_s64(acc0, bias)), 32))),
                                     vmovn_u64(vceqzq_u64(vshrq_n_u64(vreinterpretq_u64_s64(vaddq_s64(acc1, bias)), 32))));
        uint32x4_t low = vcombine_u32(vmovn_u64(vreinterpretq_u64_s64(acc0)), vmovn_u64(vreinterpretq_u64_s64(acc1)));
        uint32x4_t neg = vcombine_u32(vmovn_u64(vcltzq_s64(acc0)), vmovn_u64(vcltzq_s64(acc1)));
        uint32x4_t sat = vmvnq_u32(neg);
        if (F3DEX_FX_MUTATE == 32)                                              /* mutation 32: positive overflow saturates to 0x7fff */
            sat = vandq_u32(sat, vdupq_n_u32(0x7fff));
        vst1_u16(r.f[i], vmovn_u32(vbslq_u32(ok, low, sat)));
    }
    *out = r;
}
#else
void f3dex_mtx_mul(const struct f3dex_mtx *a, const struct f3dex_mtx *b, struct f3dex_mtx *out) { f3dex_mtx_mul_scalar(a, b, out); }
#endif

/* ---- the light directions (4.2) ------------------------------------------------------------------------------------------------------------------------------------------------ */
/* The clamp constants of DMEM 0x208..0x20d are the same in all six builds. */
#define RELIGHT_LO (-8191)
#define RELIGHT_HI 8191
#define RELIGHT_MUL 4

void f3dex_relight_scalar(const struct f3dex_mtx *mv, const int8_t raw[][3], unsigned nrec, int8_t out[][4])
{
    for (unsigned k = 0; k <= nrec; k++) {
        /* dd = 128 * d (LPV puts d << 8, VMULF by 0x4000 halves it) */
        int16_t dd[3];
        for (int c = 0; c < 3; c++)
            dd[c] = (int16_t)(raw[k][c] * FMUT(24, 128, 127));
        /* N_j = clamp32(256 * (M row j . d)): the fractions first (VMUDN VMADN VMADN), their s16 integer clamp is added doubled (VMUDM by 2) to twice the integer products (VMACF x3) */
        int16_t Nh[3], Qh[3];
        uint16_t Nl[3], Ql[3];
        int32_t N[3];
        for (int j = 0; j < 3; j++) {
            hle_acc accf = 0;
            for (int c = 0; c < 3; c++)
                accf += (hle_acc)FMUT(27, mv->f[j][c], mv->f[c][j]) * dd[c];
            hle_acc acc = (hle_acc)2 * hle_sclamp(accf);
            for (int c = 0; c < 3; c++)
                acc += (hle_acc)2 * FMUT(27, mv->i[j][c], mv->i[c][j]) * dd[c];
            acc = hle_wrap48(acc);
            Nh[j] = hle_sclamp(acc);
            Nl[j] = hle_uclamp(acc);
            N[j] = hle_fx(Nh[j], Nl[j]);
            /* Q = floor(N * N / 65536); the fraction is read before the high x high term */
            Ql[j] = FMUT(33, hle_uclamp(hle_mac_32x32_before_h(N[j], N[j])), hle_uclamp(hle_mac_32x32(N[j], N[j])));
            Qh[j] = hle_sclamp(FMUT(25, hle_mac_32x32(N[j], N[j]), hle_mac_32x32_before_h(N[j], N[j])));
        }
        /* S = Q0 + Q1 + Q2: VADDC / VADD on lanes (0,1), then lane 2 + that: the integer halves saturate */
        uint32_t s01 = (uint32_t)Ql[0] + Ql[1];
        int16_t h01 = hle_clamp16((int32_t)Qh[0] + Qh[1] + (int32_t)(s01 >> 16));
        uint32_t s2 = (uint32_t)Ql[2] + (uint16_t)s01;
        int16_t h2 = hle_clamp16((int32_t)Qh[2] + h01 + FMUT(28, (int32_t)(s2 >> 16), 0));
        int32_t S = hle_fx(h2, (uint16_t)s2);
        int32_t R = (int32_t)hle_rsq32(S);
        /* T = floor(R * 64 / 65536) */
        int32_t T = hle_pair32(hle_pair_acc(hle_mac_32x32(R, 64)));
        for (int j = 0; j < 3; j++) {
            /* U = floor(N * T / 65536): only the fraction half is used */
            int16_t u = (int16_t)hle_uclamp(hle_mac_32x32(N[j], T));
            if (u < FMUT(29, RELIGHT_LO, -8127)) u = FMUT(29, RELIGHT_LO, -8127);
            if (u > FMUT(23, RELIGHT_HI, 8127)) u = FMUT(23, RELIGHT_HI, 8127);
            out[k][j] = (int8_t)((int16_t)(uint16_t)(u * RELIGHT_MUL) >> FMUT(26, 8, 7));
        }
        out[k][3] = 0;
    }
}

/* the relight runs once per matrix change: the scalar form serves (no NEON form) */
void f3dex_relight(const struct f3dex_mtx *mv, const int8_t raw[][3], unsigned nrec, int8_t out[][4]) { f3dex_relight_scalar(mv, raw, nrec, out); }
