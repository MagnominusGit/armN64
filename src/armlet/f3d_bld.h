/*
 * arMLEt - f3d_bld.h: the build record of the Fast3D front-end (f3d.c), which dbuild.c also fills from a derived record (docs/ucode-f3d.md section 15, docs/ucode-derive.md)
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_F3D_BLD_H
#define ARMLET_F3D_BLD_H

#include <stdint.h>

#include "f3d_dmem.h"
#include "f3d_fx.h"
#include "hle.h"

/* the handler pcs of a build, in the order of f3d_build.pc[] (and of tools/derive_f3d.py's F3_PC_FIELDS). head: where the start-up code reaches the command loop (the interpreter stops there);
 * nop: the loop's return point, which the no-op commands lead to (a class-0 one after its operand DMA); brk: the BREAK of the exit code (overlay 3) that ends a task; halfc / half2: the two
 * commands that send the RDPHALF word with w1 (G_RDPHALF_CONT and G_RDPHALF_2); half1: the one that stores it; tri4: GoldenEye's (2.0G) four triangles from nibbles (op 0xb1; the vertex
 * bytes through the table at lay.tri4_tab) */
#define F3D_PC_FIELDS(X) \
    X(head) X(nop) X(brk) X(mtx) X(movemem) X(vtx) X(dl) X(tri1) X(quad) X(halfc) X(half2) X(half1) X(persp) X(clrgeom) X(setgeom) X(enddl) X(oml) X(omh) X(texture) X(moveword) X(popmtx) X(culldl) \
    X(tri4)
enum {
#define X(f) F3PC_##f,
    F3D_PC_FIELDS(X)
#undef X
    F3_NPC
};

/* The record. Everything that differs between the builds is here; the front-end has no other knowledge of a build. The attributes are behaviour that a layout or a pc cannot express: */
struct f3d_build {
    const struct f3d_layout *lay;       /* the DMEM layout (with the clip flag word's variant, F3D_FLAGS_*) */
    uint16_t pc[F3_NPC];                /* by F3PC_*; 0: the build has no such command (the opcodes its tables lead there decline) */
    uint16_t clip_mask, reject_mask;    /* G_TRI1: clip request / reject masks over the clip flag word */
    uint16_t culldl_mask;               /* G_CULLDL: the flag bits that all vertices must have for the list to end (docs/ucode-f3d.md 10.3.1) */
    uint8_t vptr_mul;                   /* G_TRI1: the vertex pointer is vtx_base + vptr_mul * byte (the encoding scales the index: 10 i with 4, 5 i with 8) */
    uint8_t vtx_wr;                     /* G_VTX encoding of Wave Race: 04 | 5 v0 | n << 9 | 16 n - 1 */
    uint8_t gp_end;                     /* the command loop's register gp is the end address of the command buffer (and k1 the next command) instead of the byte count left */
    uint8_t tri_wr;                     /* the triangle code of Wave Race (F3D_TRIV_WR: no front-face culling, no flat shading, the shade block with any of the low three geometry bits) */
    uint8_t popmtx_stale;               /* G_POPMTX sets the lights-stale flag */
    uint8_t xbus;                       /* the RDP reads its lists from a ring in DMEM (outbuf .. ring_end): every command is written at the ring pointer and handed over on its own */
    uint64_t load_key;                  /* FNV-1a-64 of the text's first 0xf48 bytes (the derivation's record key; 0: no dump of the text) */
};

#ifdef ARMLET_NATIVE
extern const struct hle_family f3d_fam;
/* the RDP command sizes of a build with Wave Race's triangle code (its triangles carry a shade block the command byte does not announce): hle_ucode.rdp_cmd_size */
unsigned f3d_wr_cmd_size(unsigned first_byte);
#endif

#endif
