/*
 * arMLEt - hle_f3dex2_int.h: what the F3DEX2 front-end (hle_f3dex2.c) shares with its dialects (hle_f3dex2_dialects.c, hle_s2dex2.c): the build tables' type, the command kinds, the
 * task state and the few helpers the dialects' handlers call
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_HLE_F3DEX2_INT_H
#define ARMLET_HLE_F3DEX2_INT_H

#include <stdint.h>

#include "f3dex2_dmem.h"
#include "f3dex2_fx.h"
#include "hle.h"

/* test builds only (scripts/build.sh native --tools --hmut=N, tools/hle/mutcheck.sh): deliberate errors, to show that the tests catch them (docs/testing.md) */
#ifdef ARMLET_HLE_MUTATE
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#define MUT(n, ok, bad) (ok)
#endif

struct f3;
struct f2_pcs;
struct f3dex2_build {
    uint16_t head;                  /* the command loop: mfc0 at (the fetch of the next command) */
    uint16_t copy, sync, segcopy;   /* handlers: copy a command to the RDP buffer / copy only its first word / resolve the segment address, then copy */
    uint16_t ef, ed;                /* G_RDPSETOTHERMODE and G_SETSCISSOR: remember the command, then copy */
    uint16_t tri1, tri2;            /* G_TRI1; G_TRI2 and G_QUAD */
    uint16_t vtx, mtx, geom, enddl, half2, modvtx, culldl, branchz;
    uint16_t mvp;                   /* a handler that concatenates the matrices now (G_SPECIAL_1 of F3DEX.NoN 2.08H), or 0 */
    unsigned gen204;                /* F3DEX2 2.04 / 2.04H, the generation before 2.05: the vertex pipeline's unclamped screen z, the clipper's fan and its new vertices' stores (fx2_vctx.gen204, fx2_clip_ctx.fan_l0; docs/ucode-f3dex2-dialects.md section 5) */
    unsigned clip204;               /* the clipper's 2.04 reciprocal (fx2_clip_ctx.sign0); equal to gen204 except in F3DEX fifo 2.05 (Chameleon Twist 2: the 2.04 reciprocal, the 2.06 fan and stores) */
    unsigned zex;                   /* F3DZEX: other lanes of the constant vectors, no mask on the slope fractions */
    unsigned bz_w;                  /* G_BRANCH_Z compares the integer clip w (`lh 6(vertex)`) although zex is 0 (F3DZEX.NoN 2.08I, Majora's Mask: Paper Mario's 2.08 text with one changed word) */
    unsigned bz_z;                  /* with zex: G_BRANCH_Z still compares the screen z (`lw 0x1c(vertex)`), not the integer part of the clip w that the F3DZEX builds compare (F3DEX 2.06: Blues Brothers 2000's text has the F3DZEX triangle code and the F3DEX branch) */
    uint64_t load_key;              /* FNV-1a-64 (armlet_ucode_hash) of the first 0xf48 bytes of the text: the part a G_LOAD_UCODE copies to IMEM, which names the build that command loads (loaded_build) */
    uint32_t reject_mask;           /* a triangle whose three clip-flag words have a common bit in this mask is rejected */
    uint16_t tri4;                  /* G_TRI4 (F3DEXBG): four triangles in one command, or 0 */
    uint16_t ldtx_spr, ldtx_rect;   /* S2DEX: where G_OBJ_LOADTXTR's continuation table (DMEM 0x202) sends the LDTX_SPRITE and LDTX_RECT commands (the sprite and rectangle code behind the overlay's stubs), or 0 */
    const struct f2_layout *lay;    /* the DMEM layout when it is not the standard one */
    const struct f2_pcs *pcs;       /* the handlers of the commands below 0x170 when they are not at the standard pcs (S2DEX), or NULL */
    const uint16_t (*ext)[2];       /* dialect commands: { handler pc, kind (>= K_EXT) } pairs ending in { 0xffff, 0 }, run by ext_cmd */
    int (*ext_cmd)(struct f3 *f, unsigned kind);
};

/* The handlers below 0x170 sit at the same pcs in all the builds: the display list stack, the state words the other handlers read, the other-mode registers */
enum { PC_DL = 0x000, PC_TEXTURE = 0x028, PC_TEXRECT = 0x02c, PC_HALF1 = 0x030, PC_MOVEWORD = 0x038, PC_POPMTX = 0x04c, PC_MOVEMEM = 0x120, PC_OMH = 0x13c, PC_OML = 0x140 };

/* the handler pcs of the commands the standard builds share (PC_*), for a build that has them elsewhere; 0xffff: the build has no such command */
struct f2_pcs {
    uint16_t dl, texture, texrect, half1, moveword, popmtx, movemem, omh, oml;
};
enum { K_EXT = 0x40 };                                  /* kinds from here on belong to a dialect (ext_cmd) */

enum { K_BAD = 0, K_NOP, K_DL, K_ENDDL, K_TEXTURE, K_TEXRECT, K_HALF1, K_HALF2, K_MOVEWORD, K_MOVEMEM, K_OMH, K_OML, K_GEOM, K_COPY, K_SYNC, K_SEGCOPY, K_EF, K_ED, K_VTX, K_MTX, K_POPMTX,
       K_TRI1, K_TRI2, K_MODVTX, K_CULLDL, K_BRANCHZ, K_MVP, K_TRI4, K_LOADUCODE };

/* DMEM layout (the same in all the builds): jump table at 0x36e by opcode; segments at 0xf8; RDP FIFO pointer 0xf0, FIFO bounds 0xfe8 / 0xfec; the display list stack depth at 222 (x4) and
 * the stack at 0x138; the RDP buffers: two of 344 bytes, s7 the write pointer, s6 the end of the current one (xor 0x208 toggles) */
enum { SEGMENTS = 0xf8, FIFO_PTR = 0xf0, FIFO_START = 0xfe8, FIFO_END = 0xfec, DL_DEPTH = 222, DL_STACK = 0x138,
       SIG0 = 0x80, DPC_BUSY_BIT = 0x400, POLL_LIMIT = 30000 };

enum { OV_CLIP = 0, OV_LIGHT = 1, OV_UNKNOWN = 2 };
#define JUMP_TABLE (f2L->jump_table)
#define OUT_BYTES (f2L->out_bytes)
#define OUT_TOGGLE (f2L->out_toggle)
#define DESC_LIGHT (f2L->desc_light)
#define DESC_CLIP (f2L->desc_clip)

struct f3 {
    struct hle_task *t;
    const struct f3dex2_build *b;
    uint32_t k0, k1, s6, s7;        /* the microcode's registers */
    unsigned polls;                 /* SP_STATUS polls the microcode would have made (the JIT halts a task at 32767) */
    int ov;                         /* which overlay IMEM holds at [0x2d0, ...) */
    uint8_t kind[256];
    uint32_t w0, w1;
    struct fx2_clip_out co;
    struct fx2_tri_in tin;          /* the constants of the triangle setup */
    uint32_t clip_mask;
    int rc;                         /* a result an ext command's helper left (hle_rej.c)  */
    int ho_req;                     /* an ext command ended the outermost display list: the loop hands the task back (f3_enddl) */
    int cut;                        /* the hand-over is a G_DL to itself (PRSP_POLL_IDLE): the interpreter runs it to the JIT's idle cut, which ends the call */
    int lean;                       /* the external-effects mode: no DMA the effects do not need (the display list and the vertices are read from RDRAM, the overlays are loaded once at the end) */
    int ov0;                        /* the overlay IMEM held at the start */
    int trace, watch, dump_imem;
    uint32_t nsrc;                  /* with nb != b: the address of the text it is loaded from (uc0 after the switch) */
    uint32_t uc0;                   /* the address of the text this run is for: OSTask.ucode at the start, the text address of the last G_LOAD_UCODE that switched builds */
    const struct hle_ucode *nu;     /* the hle_ucode of nb (its gate: hle_bind at the switch) */
    const struct f3dex2_build *nb;  /* the build a G_LOAD_UCODE that was handed to the interpreter loads (b: the same text), taken over when the reloaded microcode reaches its first command fetch */
    unsigned ho_pc;                 /* where the interpreter takes over when the loop hands the task back: the G_ENDDL handler, or 0xfac (G_LOAD_UCODE) */
    uint16_t wval;
    unsigned ncmd;
    uint32_t total;
    const char *bad;                /* a reason to decline found where the function could not return one (checked at the head of the loop) */
};

/* ---- shared with the dialects ------------------------------------------------------------------------------------------------------------------------------------------------ */
extern const struct hle_family f3dex2_fam;
const struct f2_layout *f3dex2_std_layout(void);

/* the dead ranges of the standard builds (the dialects that keep the standard buffers use them too) */
#define F3DEX2_DEAD_RANGES { 0x3ca, 0x408 }, { 0x920, 0xfb8 }, { 0xfc8, 0xfcc }, \
    { 0x203, 0x204 }, { 0x207, 0x208 }, { 0x21b, 0x21c }, { 0x21f, 0x220 }, { 0x233, 0x234 }, { 0x237, 0x238 }, { 0x24b, 0x24c }, { 0x24f, 0x250 }, \
    { 0x263, 0x264 }, { 0x267, 0x268 }, { 0x27b, 0x27c }, { 0x27f, 0x280 }, { 0x293, 0x294 }, { 0x297, 0x298 }, { 0x2ab, 0x2ac }, { 0x2af, 0x2b0 }, { 0x2c3, 0x2c4 }, \
    { 0x2c7, 0x2c8 }
extern const struct hle_dead_range f3dex2_dead_out[];
enum { F3DEX2_N_DEAD_OUT = 21 };
#define F3DEX2_DEAD .ring_off = 0xfe8, .dead = f3dex2_dead_out, .n_dead = F3DEX2_N_DEAD_OUT

int f3_flush(struct f3 *f);
int f3_flush_xbus(struct f3 *f);                    /* hle_rej.c: the room check and DPC_END of the XBUS builds' flush routine (f2L->xbus) */
void f3_dma_rd(struct f3 *f, unsigned mem, uint32_t dram, uint32_t len);
int f3_store_unmodelled(const struct f3 *f, unsigned a, unsigned len);   /* a store of [a, a + len] the lean mode cannot model (display list buffer, jump table, overlay descriptors, FIFO pointer ..., or IMEM) */
void f3_overlay_load(struct f3 *f, int which);
void f3_do_dl(struct f3 *f, uint32_t w0, uint32_t w1);
int f3_enddl(struct f3 *f);                         /* G_ENDDL's effect for a dialect's command (f->ho_req set when it was the outermost list's) */
int f3_branch(struct f3 *f, uint32_t addr);         /* a branch to the segment address `addr`, no stack */

/* after a command that wrote to the buffer (the build's flush entry: a "safe" build writes a pipe sync first) */
static inline int after_emit(struct f3 *f)
{
    if (f2L->pipe_sync) {
        hle_st8(f->t, f->s7, MUT(300, 0xe7, 0xe6));
        f->s7 += MUT(301, 8, 0);
    }
    if (f2L->xbus)
        return f3_flush_xbus(f);
    if ((int32_t)(f->s7 - f->s6) > 0)
        return f3_flush(f);
    return HLE_OK;
}

static inline uint32_t seg_addr(const struct hle_task *t, uint32_t w) { return (w & 0xffffff) + hle_dm32(t, SEGMENTS + ((w >> 22) & 0x3c)); }

#endif
