/*
 * arMLEt - f3dex_bld.h: the build record of the F3DEX 1.x / F3DLX front-end (f3dex.c), which dbuild.c also fills from a derived record (docs/native-f3dex.md section 3, docs/ucode-derive.md)
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3DEX_BLD_H
#define ARMLET_F3DEX_BLD_H

#include <stdint.h>

#include "f3dex_dmem.h"
#include "f3dex_fx.h"
#include "hle.h"

/* the handler pcs of a build, in the order of f3dex1_build.pc[] (and of tools/derive_f1d.py's F1_PC_FIELDS). head: where the start-up code reaches the command loop (the interpreter stops there);
 * nop: the loop's return point, which the no-op commands lead to (a class-0 one after its operand DMA) */
#define F3DEX_PC_FIELDS(X) \
    X(head) X(nop) X(mtx) X(movemem) X(vtx) X(dl) X(loaduc) X(branchz) X(tri2) X(modvtx) X(half2) X(half1) X(persp) X(quad) X(clrgeom) X(setgeom) X(enddl) X(oml) X(omh) X(texture) \
    X(moveword) X(popmtx) X(culldl) X(tri1)
enum {
#define X(f) F1PC_##f,
    F3DEX_PC_FIELDS(X)
#undef X
    F1_NPC
};

/* S2DEX 1.06 / 1.07 (docs/ucode-s2dex1.md): the 2D microcode on the skeleton of F3DEX 1.x, run by the same front-end (f3dex.c, hle_s2dex1.h) in its own mode. The handler pcs of a build, in the order of s2d1_build.pc[]: the commands
 * are told apart by the pc the build's jump table (DMEM s2d1_build.jt, by signed opcode) holds for the opcode. nop: the command loop's head, where the no-ops and the load / rectangle continuation of G_OBJ_LOADTXTR (the
 * first cell of its table) lead to; ldtx_spr / ldtx_rect: where the table sends LDTX_SPRITE and LDTX_RECT. */
#define S2D1_PC_FIELDS(X) \
    X(nop) X(half0) X(half1) X(half2) X(enddl) X(omh) X(oml) X(moveword) X(rendermode) X(setom) X(segcopy) X(scissor) X(sync) X(copy) X(seldl) X(dl) X(movemem) X(bg1cyc) X(rect) X(sprite) X(bgcopy) X(loadtxtr) X(loaduc) \
    X(ldtx_spr) X(ldtx_rect)
enum {
#define X(f) S2P_##f,
    S2D1_PC_FIELDS(X)
#undef X
    S2P_N
};
struct s2d1_build {
    uint16_t jt;                        /* the jump table: the pc of opcode op is the halfword at jt + 2 * (int8_t)op */
    uint16_t pc[S2P_N];                 /* by S2P_* */
    uint8_t bgc107;                     /* G_BG_COPY is the routine of S2DEX 2.x (1.07), not 1.06's own */
};

struct f3dex1_build {
    struct f3dex_build fx;              /* reject mask, NoN, LX, the DMEM layout */
    uint16_t pc[F1_NPC];                /* by F1PC_*; 0: the build has no such command */
    uint8_t culldl_off;                 /* G_CULLDL's operands are byte offsets into the vertex buffer and the last vertex is not tested (the 0.95 builds), not vertex numbers */
    uint64_t load_key;                  /* FNV-1a-64 of the text's first 0xf48 bytes (the derivation's record key; 0: no dump of the text) */
    const struct s2d1_build *s2;        /* S2DEX 1.x: the commands are the 2D ones (hle_s2dex1.h); NULL for the F3DEX 1.x / F3DLX builds */
};

#ifdef ARMLET_NATIVE
extern const struct hle_family f3dex1_fam;
#endif

#endif
