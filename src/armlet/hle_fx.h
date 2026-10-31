/*
 * arMLEt - hle_fx.h: the RSP's fixed-point arithmetic as plain integer functions (the scalar reference of the task-level native path)
 *
 * The microcode keeps 32-bit numbers as two vectors, an integer half (signed 16 bits) and a fraction half (unsigned 16 bits); the value of the pair is
 * I * 65536 + F, held here as an int32_t ("S15.16" when it has 16 fraction bits, a plain 32-bit integer otherwise). The vector unit builds products
 * of such pairs from chains of multiply-accumulate instructions over a 48-bit accumulator; each function below is the closed form of one chain, with the
 * result as the accumulator (hle_acc) and the readouts (sclamp / uclamp) that the chain's instructions return. The closed forms and the tests that check
 * them against the exact VU instruction sequences (tools/hle/prim_test.c, 200 000 trials x 8 lanes per primitive, 0 differences) were written and verified
 * by Fleet B (docs/ucode-f3dex.md, section 1); this header is those definitions made into the shared library.
 *
 * Sums wrap modulo 2^48 (hle_wrap48) as the accumulator does. The NEON versions (hle_fx_neon.h) must agree with these functions bit for bit.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_HLE_FX_H
#define ARMLET_HLE_FX_H

#include <stdint.h>

/* the divide unit's reciprocal ROM (the fork's copy, ext/parallel-rsp/rsp/reciprocal.cpp) */
extern const uint16_t rsp_reciprocal_rom[1024];

#define HLE_FX_I static inline __attribute__((always_inline))

/* test builds only (-DHLE_FX_MUTATE=n): deliberate errors, to show that tools/hle/prim_test.c catches them */
#ifdef HLE_FX_MUTATE
#define HLE_FX_MUT(n, ok, bad) ((HLE_FX_MUTATE) == (n) ? (bad) : (ok))
#else
#define HLE_FX_MUT(n, ok, bad) (ok)
#endif

/* ---- the accumulator ----------------------------------------------------------------------------------------------------------------------------------------- */
typedef int64_t hle_acc;                /* signed, 48 bits (kept sign-extended by hle_wrap48) */

HLE_FX_I hle_acc hle_wrap48(hle_acc a) { return (hle_acc)((uint64_t)a << 16) >> 16; }

/* VMULF VMACF VMUDH VMADH VMUDM VMADM: bits 47..16 of the accumulator, saturated to s16 */
HLE_FX_I int16_t hle_sclamp(hle_acc a)
{
    hle_acc v = a >> 16;
    return v > 32767 ? 32767 : v < -32768 ? HLE_FX_MUT(1, -32768, -32767) : (int16_t)v;
}
/* VMADN VMADL: bits 15..0 when the accumulator fits in s32, else 0xffff (positive) / 0 (negative) */
HLE_FX_I uint16_t hle_uclamp(hle_acc a)
{
    if (a >= -(hle_acc)2147483648ll && a < (hle_acc)2147483648ll)
        return (uint16_t)a;
    return a < 0 ? 0 : HLE_FX_MUT(2, 0xffff, 0x7fff);
}
/* VSAR 8 / 9 / 10 */
HLE_FX_I uint16_t hle_acc_hi(hle_acc a) { return (uint16_t)(a >> 32); }
HLE_FX_I uint16_t hle_acc_md(hle_acc a) { return (uint16_t)(a >> 16); }
HLE_FX_I uint16_t hle_acc_lo(hle_acc a) { return (uint16_t)a; }

/* ---- 32-bit numbers as (integer, fraction) pairs ------------------------------------------------------------------------------------------------------------- */
struct hle_pair { int16_t i; uint16_t f; };

HLE_FX_I int32_t hle_fx(int16_t i, uint16_t f) { return (int32_t)((uint32_t)(uint16_t)i << 16 | f); }
HLE_FX_I int16_t hle_fx_i(int32_t v) { return (int16_t)(v >> 16); }
HLE_FX_I uint16_t hle_fx_f(int32_t v) { return (uint16_t)v; }
HLE_FX_I int32_t hle_pair32(struct hle_pair p) { return hle_fx(p.i, p.f); }
/* the usual end of a chain: integer half sclamp, fraction half uclamp */
HLE_FX_I struct hle_pair hle_pair_acc(hle_acc a) { struct hle_pair p = { hle_sclamp(a), hle_uclamp(a) }; return p; }

/* ---- products (docs/ucode-f3dex.md 1.2) ---------------------------------------------------------------------------------------------------------------------- */
/* VMUDL VMADM VMADN VMADH [VMADN 0]: A x B, accumulator = floor(A * B / 2^16) (the dropped part is the low 16 bits of af*bf) */
HLE_FX_I hle_acc hle_mac_32x32(int32_t A, int32_t B) { return hle_wrap48(((int64_t)A * B) >> 16); }
/* the fraction half read by the VMADN in the middle of that chain (before the ai*bi<<16 term) */
HLE_FX_I hle_acc hle_mac_32x32_before_h(int32_t A, int32_t B)
{
    return hle_mac_32x32(A, B) - HLE_FX_MUT(3, (((hle_acc)hle_fx_i(A) * hle_fx_i(B)) << 16), 0);
}
/* VMUDL VMADM VMADN: A x c, c an unsigned 16-bit fraction */
HLE_FX_I hle_acc hle_mac_32xu16(int32_t A, uint16_t c) { return hle_wrap48(((int64_t)A * (int64_t)c) >> 16); }
/* VMUDM VMADN 0: s16 x u16, the full 32-bit product (integer half = p >> 16, fraction half = p & 0xffff) */
HLE_FX_I int64_t hle_mul_s16u16(int16_t x, uint16_t c) { return (int64_t)x * c; }
/* VMUDN VMADH VMADN VMADH: A*x + B*y with signed 16-bit x, y: exact (VSAR gives the three parts, sclamp the last VMADH) */
HLE_FX_I hle_acc hle_dot32x16(int32_t A, int16_t x, int32_t B, int16_t y)
{
    return hle_wrap48((hle_acc)A * x + (hle_acc)B * y);
}
/* a*b + c*d of s16 values, as the 32-bit sum VMUDH VMADH + VSAR 8/9 give (the accumulator is that << 16) */
HLE_FX_I int32_t hle_dot16(int16_t a, int16_t b, int16_t c, int16_t d) { return (int32_t)((int64_t)a * b + (int64_t)c * d); }
HLE_FX_I hle_acc hle_dot16_acc(int16_t a, int16_t b, int16_t c, int16_t d) { return hle_wrap48(((hle_acc)a * b + (hle_acc)c * d) << 16); }
/* VMULF: (2ab + 0x8000), VMACF adds 2cd; read with hle_sclamp */
HLE_FX_I hle_acc hle_mulf(int16_t a, int16_t b) { return (hle_acc)2 * a * b + 0x8000; }
HLE_FX_I hle_acc hle_macf(hle_acc acc, int16_t c, int16_t d) { return hle_wrap48(acc + (hle_acc)2 * c * d); }
/* VMUDN VMADH VMADN 0 with the constant 2: 2*A with the integer half saturating */
HLE_FX_I struct hle_pair hle_double32(int32_t A) { return hle_pair_acc(hle_wrap48((hle_acc)A * 2)); }

/* ---- 32-bit add, subtract, min, max, abs (1.3) --------------------------------------------------------------------------------------------------------------- */
HLE_FX_I int16_t hle_clamp16(int32_t v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t)v; }
/* VADDC VADD / VSUBC VSUB: only the integer half saturates */
HLE_FX_I struct hle_pair hle_add32(int32_t A, int32_t B)
{
    unsigned carry = (unsigned)hle_fx_f(A) + hle_fx_f(B) > 0xffff;
    struct hle_pair p = { hle_clamp16((int32_t)hle_fx_i(A) + hle_fx_i(B) + (int32_t)carry), (uint16_t)(hle_fx_f(A) + hle_fx_f(B)) };
    return p;
}
HLE_FX_I struct hle_pair hle_sub32(int32_t A, int32_t B)
{
    unsigned borrow = HLE_FX_MUT(4, hle_fx_f(A) < hle_fx_f(B), hle_fx_f(A) <= hle_fx_f(B));
    struct hle_pair p = { hle_clamp16((int32_t)hle_fx_i(A) - hle_fx_i(B) - (int32_t)borrow), (uint16_t)(hle_fx_f(A) - hle_fx_f(B)) };
    return p;
}
/* VSUBC VLT VMRG / VSUBC VGE VMRG */
HLE_FX_I int32_t hle_min32(int32_t A, int32_t B) { return A < B ? A : B; }
HLE_FX_I int32_t hle_max32(int32_t A, int32_t B) { return A >= B ? A : B; }
/* VABS x,x */
HLE_FX_I int16_t hle_abs16(int16_t a) { int32_t m = a < 0 ? -(int32_t)a : a; return m > 32767 ? 32767 : (int16_t)m; }

/* ---- the clip-flag pair VCH + VCL (1.4) ---------------------------------------------------------------------------------------------------------------------- */
/* X, W: 32-bit coordinate and w. "opposite" means X < 0 differs from W < 0. le = VCC bit k (X <= -W for W >= 0), ge = VCC bit 8+k (X >= W for W >= 0). */
HLE_FX_I int hle_clip_le(int32_t X, int32_t W)
{
    int opp = (X < 0) != (W < 0);
    return opp ? ((int64_t)X + W <= HLE_FX_MUT(5, 0, -1)) : (W < 0);
}
HLE_FX_I int hle_clip_ge(int32_t X, int32_t W)
{
    int opp = (X < 0) != (W < 0);
    return opp ? (W < 0) : ((int64_t)X - W >= 0);
}

/* ---- the divide unit (1.5, 1.6) ------------------------------------------------------------------------------------------------------------------------------ */
/* VRCPH VRCPL VRCPH: the 32-bit reciprocal of a 32-bit input (double-precision form) */
HLE_FX_I uint32_t hle_rcp32(int32_t input)
{
    int32_t mask = input >> 31, data = input ^ mask, result;
    unsigned shift, idx;
    if (input > -32768)
        data -= mask;
    if (data == 0)
        return 0x7fffffffu;
    if (input == -32768)
        return 0xffff0000u;
    shift = (unsigned)__builtin_clz((uint32_t)data);
    idx = (unsigned)((((uint64_t)(uint32_t)data << shift) & 0x7FC00000u) >> 22);
    result = rsp_reciprocal_rom[idx];
    result = (int32_t)(((0x10000 | result) << 14) >> HLE_FX_MUT(6, 31 - shift, 30 - shift));
    return (uint32_t)(result ^ mask);
}
/* VRSQH VRSQL VRSQH: the reciprocal square root, same input form */
HLE_FX_I uint32_t hle_rsq32(int32_t input)
{
    int32_t mask = input >> 31, data = input ^ mask, result;
    unsigned shift, idx;
    if (input > -32768)
        data -= mask;
    if (data == 0)
        return 0x7fffffffu;
    if (input == -32768)
        return 0xffff0000u;
    shift = (unsigned)__builtin_clz((uint32_t)data);
    idx = (unsigned)((((uint64_t)(uint32_t)data << shift) & 0x7FC00000u) >> 22);
    idx = ((idx | 0x200) & 0x3FE) | HLE_FX_MUT(7, shift & 1, 0);
    result = rsp_reciprocal_rom[idx];
    result = (int32_t)(((0x10000 | result) << 14) >> ((31 - shift) >> 1));
    return (uint32_t)(result ^ mask);
}
/* The reciprocal of a 32-bit S15.16 value as the vertex and triangle code compute it: rcp32, doubled, then one Newton step of the chains above. The fraction
 * of T is read before the ai*bi term, and E = 2.0 - T uses the borrow rule (hle_sub32 semantics on the pair (2:0) - T). Returns the result as (integer, fraction). */
HLE_FX_I struct hle_pair hle_recip_refine(int32_t w)
{
    int32_t R = (int32_t)hle_rcp32(w);
    struct hle_pair D = hle_double32(R);
    int32_t Dv = hle_pair32(D);
    hle_acc t_acc = hle_mac_32x32(Dv, w);
    uint16_t tf = hle_uclamp(hle_mac_32x32_before_h(Dv, w));
    int16_t ti = hle_sclamp(t_acc);
    struct hle_pair E = hle_sub32(hle_fx(2, 0), hle_fx(ti, tf));
    int32_t Ev = hle_pair32(E);
    struct hle_pair r;
    r.i = hle_sclamp(hle_mac_32x32(Dv, Ev));
    r.f = hle_uclamp(hle_mac_32x32_before_h(Dv, Ev));
    return r;
}

#endif
