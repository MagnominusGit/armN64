/*
 * arMLEt - s2dex2_int.h: what the S2DEX 2.x closed forms (s2dex2_bg.c, s2dex2_rect.c) share: DMEM access, the 16-bit saturation, the microcode's vector constants
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_S2DEX2_INT_H
#define ARMLET_S2DEX2_INT_H

#include <stdint.h>

/* test builds only (scripts/build.sh native --tools --hmut=N, tools/hle/test_s2dex2.sh): deliberate errors, to show that the tests catch them (docs/testing.md) */
#ifdef ARMLET_HLE_MUTATE
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#define MUT(n, ok, bad) (ok)
#endif

/* ---- DMEM, big-endian, any byte address (wrapping inside DMEM; the image is the host words that hold the big-endian words, as everywhere in src/hle) ------------------------------ */
static inline unsigned m8(const uint32_t *dm, unsigned a) { return ((const uint8_t *)dm)[(a ^ 3) & 0xfff]; }
static inline unsigned m16(const uint32_t *dm, unsigned a) { return m8(dm, a) << 8 | m8(dm, a + 1); }
static inline uint32_t m32(const uint32_t *dm, unsigned a) { return (uint32_t)m16(dm, a) << 16 | m16(dm, a + 2); }
static inline void w8(uint32_t *dm, unsigned a, unsigned v) { ((uint8_t *)dm)[(a ^ 3) & 0xfff] = (uint8_t)v; }
static inline void w16(uint32_t *dm, unsigned a, unsigned v) { w8(dm, a, v >> 8); w8(dm, a + 1, v); }
static inline void w32(uint32_t *dm, unsigned a, uint32_t v) { w16(dm, a, v >> 16); w16(dm, a + 2, v); }

static inline int32_t sat16(int32_t v) { return v > 32767 ? 32767 : v < -32768 ? -32768 : v; }

/* The DMEM layout the kernels work on. S2DEX 2.x is the default; a translation unit that defines S2_L1 before it includes a kernel file (s2dex1_rect.c, s2dex1_bg.c) gets S2DEX 1.06 / 1.07's: the same arithmetic
 * (the code of the rectangle and of G_BG_1CYC is word for word the same but for these addresses), the object block 0x170 higher, the rest elsewhere (docs/ucode-s2dex1.md, section 2). */
#ifdef S2_L1
#define S2FN(n) s2dex1_##n
enum { L_CONSTS = 0x170, L_D = 0x170, L_SCRATCH = 0x1b0, L_SCISSOR = 0xc0, L_TILE_STATE = 0xd2, L_TILE_TOGGLE = 0xd3, L_RM_WORDS = 0x1e8, L_RM_PTR_A = 0x252, L_RM_PTR_B = 0x254, L_RM_BYTE = 0x250,
       L_TILE_W1 = 0x24c, L_TILE_BY_SIZ = 0x280, L_TILE_BY_FMT = 0x290 };
#else
#define S2FN(n) s2dex2_##n
enum { L_CONSTS = 0x180, L_D = 0, L_SCRATCH = 0x1c0, L_SCISSOR = 0x204, L_TILE_STATE = 0x2ae, L_TILE_TOGGLE = 0x2af, L_RM_WORDS = 0x1f8, L_RM_PTR_A = 0x244, L_RM_PTR_B = 0x246, L_RM_BYTE = 0x268,
       L_TILE_W1 = 0x264, L_TILE_BY_SIZ = 0x294, L_TILE_BY_FMT = 0x2a4 };
#endif

/* The microcode's vector constants v31 .. v28 (DMEM 0x180 .. 0x1bf in S2DEX 2.x, 0x170 .. 0x1af in 1.x: the data segment of every task). The closed forms have them folded in and decline any other. */
static const uint32_t k_consts[16] = {
    0x08000040, 0x0002ffff, 0x40000003, 0x0008fff8,         /* v31 */
    0x01008000, 0xfe000000, 0x10000400, 0x20000200,         /* v30 */
    0x3d100001, 0x32000000, 0x27000000, 0x35242628,         /* v29: RDP command ids */
    0xfffc0400, 0x00200010, 0xfc0003ff, 0x001fffe0,         /* v28 */
};

#endif
