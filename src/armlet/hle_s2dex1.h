/*
 * arMLEt - hle_s2dex1.h: S2DEX 1.06 / 1.07 (the 2D microcode on the skeleton of F3DEX 1.x: docs/ucode-s2dex1.md) in the F3DEX 1.x front-end
 *
 * Included by f3dex.c (it works on that file's task state and its staged RDP output); not a translation unit of its own. A build with `f3dex1_build.s2` set runs this mode: the same task start (the
 * boot code and the initialisation on the interpreter up to the command loop, a G_LOAD_UCODE in either direction), another loop. The display list comes in 320-byte windows, as in F3DEX 1.x, from RDRAM;
 * the RDP commands are assembled in DMEM at the output pointer s7 (0x588 ..) and handed to the FIFO when it is above 0x8a8 (the microcode's flush check after every command), as one chunk, so the wrap
 * of the FIFO is decided per chunk as the microcode does; the end of the task (the outermost G_ENDDL) is the microcode's own code, run by the interpreter after the last chunk (s2_end).
 *
 * Declined (the JIT runs the task): G_OBJ_SPRITE and the rotated forms (G_OBJ_RECTANGLE_R, LDTX_SPRITE, LDTX_RECT_R), and what a game does not write (docs/ucode-s2dex1.md, section 5).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_HLE_S2DEX1_H
#define ARMLET_HLE_S2DEX1_H

#include "s2dex2_fx.h"

/* the command kinds (by the pc the jump table holds for the opcode) and what s2_cmd returns besides HLE_OK / HLE_DECLINE */
enum { S2K_BAD = 0, S2K_NOP, S2K_HALF0, S2K_HALF1, S2K_HALF2, S2K_ENDDL, S2K_OMH, S2K_OML, S2K_MOVEWORD, S2K_RENDERMODE, S2K_SETOM, S2K_SEGCOPY, S2K_SCISSOR, S2K_SYNC, S2K_COPY, S2K_SELDL, S2K_DL,
       S2K_MOVEMEM, S2K_BG1CYC, S2K_RECT, S2K_SPRITE, S2K_BGCOPY, S2K_LOADTXTR, S2K_LOADUC };
enum { S2R_END = 2, S2R_FINISH = 3 };                       /* the outermost G_ENDDL; the task ended inside the command (a G_LOAD_UCODE whose initialisation ended it) */

/* the DMEM of S2DEX 1.x that the commands use (the same in 1.06 and 1.07: docs/ucode-s2dex1.md, section 1) */
enum {
    S2_IN = 0x3c0, S2_IN_END = 0x500, S2_OUT = 0x588, S2_THR = 0x8a8, S2_OUT_END = 0x960,
    S2_SCISSOR = 0xc0, S2_OMH = 0xc8, S2_OML = 0xcc, S2_DEPTH = 0xd0, S2_TSTATE = 0xd2, S2_WP = 0xd4, S2_H0 = 0xd8, S2_H1 = 0xe0, S2_SEGS = 0xe8, S2_STACK = 0x128,
    S2_CONSTS = 0x170, S2_RMBYTE = 0x250, S2_FLAGS = 0x2a0, S2_OVL = 0x2f0, S2_JT_LO = 0x2aa, S2_JT_HI = 0x4aa, S2_MWTAB = 0x3b6, S2_MMTAB = 0x3b8,
    S2_TXTR = 0x518, S2_SPRITE = 0x530, S2_BGOBJ = 0x520, S2_V29 = 0x190, S2_OV_ADDR = 0x238, S2_OV_LEN = 0x730
};

static inline uint32_t s2_ld32(const struct hle_task *t, unsigned a)     /* a word at any address (the RSP's LW wraps inside DMEM a byte at a time) */
{
    return (uint32_t)hle_dm8(t, a) << 24 | (uint32_t)hle_dm8(t, a + 1) << 16 | (uint32_t)hle_dm8(t, a + 2) << 8 | hle_dm8(t, a + 3);
}
static inline unsigned s2_ld16(const struct hle_task *t, unsigned a) { return (unsigned)hle_dm8(t, a) << 8 | hle_dm8(t, a + 1); }

static unsigned s2_classify(const struct s2d1_build *b, unsigned h)
{
    static const uint8_t kind_of_pc[S2P_N] = { S2K_NOP, S2K_HALF0, S2K_HALF1, S2K_HALF2, S2K_ENDDL, S2K_OMH, S2K_OML, S2K_MOVEWORD, S2K_RENDERMODE, S2K_SETOM, S2K_SEGCOPY, S2K_SCISSOR, S2K_SYNC, S2K_COPY,
                                               S2K_SELDL, S2K_DL, S2K_MOVEMEM, S2K_BG1CYC, S2K_RECT, S2K_SPRITE, S2K_BGCOPY, S2K_LOADTXTR, S2K_LOADUC, S2K_BAD, S2K_BAD };
    unsigned k;
    for (k = 0; k <= S2P_loaduc; k++)                       /* (G_RDPHALF_0's handler is at pc 0: the cells of the undefined opcodes that hold 0 lead to it, as in the microcode) */
        if (h == b->pc[k])
            return kind_of_pc[k];
    return S2K_BAD;
}

/* The opcodes through the build's jump table in DMEM (a cell per opcode, signed). The display list window lies over the cells of the opcodes 0x0b .. 0x7f: those are not commands (the microcode would jump to
 * a halfword of the display list). */
static void s2_make_kinds(struct f3dex_task *f)
{
    unsigned op;
    for (op = 0; op < 256; op++) {
        unsigned ea = (f->s2->jt + 2 * (int)(int8_t)op) & 0xfff;
        f->kind[op] = (ea + 1 >= S2_IN && ea < S2_IN_END) ? (uint8_t)S2K_BAD : (uint8_t)s2_classify(f->s2, hle_dm16(f->t, ea) & 0xffc);
    }
}

/* a store of n bytes at a over the first cells of the jump table (the flags words of index 8 end in the cell of opcode 0x80): the opcodes whose cell it changed are classified anew, as the microcode reads the cell when the command comes */
static void s2_cells_changed(struct f3dex_task *f, unsigned a, unsigned n)
{
    unsigned op;
    if (MUT(534, 0, 1))
        return;
    for (op = 0x80; op < 0x84; op++) {
        unsigned ea = (f->s2->jt + 2 * (int)(int8_t)op) & 0xfff;
        if (ea < a + n && ea + 2 > a)
            f->kind[op] = (uint8_t)s2_classify(f->s2, hle_dm16(f->t, ea) & 0xffc);
    }
}

static int s2_ov_match(struct f3dex_task *f, int k)
{
    struct hle_task *t = f->t;
    unsigned d = S2_OVL + 8 * k;
    uint32_t dram = hle_dm32(t, d) & 0x7ffff8;
    unsigned len = hle_dm16(t, d + 4) + 1;
    if (len != S2_OV_LEN || dram + len > 0x800000)
        return 0;
    return hle_ram_cmp_words(t->imem + (S2_OV_ADDR >> 2), t->rdram, dram, len) == 0;
}

/* the state of the loop from the interpreter at the first fetch of S2DEX's command loop (the start of the task and after a G_LOAD_UCODE) */
static int s2_enter(struct f3dex_task *f, const struct hle_cpu *c)
{
    struct hle_task *t = f->t;
    const struct f3dex_layout *L = f->L;
    f->k0 = c->r[HLE_REG_K0] - 320;
    f->gp = 320;
    f->ov = OV_TEXT;
    f->s7 = c->r[HLE_REG_S7];
    f->wp = hle_dm32(t, L->rdp_wp);
    f->fstart = hle_dm32(t, L->fifo);
    f->fend = hle_dm32(t, L->fifo + 4);
    if (f->wp & 7)
        return hle_decline(t, "RDP write pointer not 8-byte aligned");
    if (*t->cr[11] & 1)
        return hle_decline(t, "XBUS still set after the start of the task");
    if ((f->k0 & 7) || c->r[HLE_REG_K1] != (uint32_t)-320 || f->s7 != S2_OUT)
        return hle_decline(t, "unusual state at the command loop");
    f->s2ov = s2_ov_match(f, 1) ? 1 : s2_ov_match(f, 2) ? 2 : 0;
    if (!f->s2ov)
        return hle_decline(t, "unknown overlay");
    s2_make_kinds(f);
    return HLE_OK;
}

/* the overlay k (1: sprites and rectangles, 2: backgrounds) in IMEM 0x238: the stubs of the other one load it (DMA table at DMEM 0x2f0 + 8k) */
static void s2_ovl(struct f3dex_task *f, int k)
{
    struct hle_task *t = f->t;
    unsigned d = S2_OVL + 8 * k;
    if (f->s2ov == k)
        return;
    hle_dma_read(t, hle_dm16(t, d + 6), hle_dm32(t, d), hle_dm16(t, d + 4));
    f->s2ov = k;
}

/* an object into DMEM, as the microcode's DMA does (len = the register value) */
static void s2_dma_rd(struct f3dex_task *f, unsigned mem, uint32_t dram, uint32_t len)
{
    if (MUT(243, ring_hit(f, dram, ((len & 0xfff) | 7) + 1), 0))
        f->bad = "a read from inside the RDP output ring";
    hle_dma_read(f->t, mem, dram, len);
}

/* the commands written at the output pointer go to the FIFO when it is above the threshold (the flush check after every command: db0 / dbc), as one chunk */
static int s2_flush(struct f3dex_task *f)
{
    struct hle_task *t = f->t;
    unsigned n = f->s7 - S2_OUT;
    int w;
    if (f->exact) {
        if (flush(f, n) != HLE_OK)
            return HLE_DECLINE;
    } else {
        w = out_reserve(f, n);
        if (w < 0)
            return HLE_DECLINE;
        memcpy(&f->stage[w], t->dmem + (S2_OUT >> 2), n);
        *t->cr[0] = S2_OUT + n;                             /* (the registers the DMA write leaves, hle_dma_write) */
        *t->cr[1] = f->wp;
        *t->cr[3] = 0xff8;
    }
    f->s7 = S2_OUT;
    return HLE_OK;
}
static inline int s2_chk(struct f3dex_task *f) { return (int32_t)(f->s7 - MUT(520, S2_THR, S2_THR + 8)) > 0 ? s2_flush(f) : HLE_OK; }

static inline void s2_put(struct f3dex_task *f, uint32_t w0, uint32_t w1)
{
    hle_st32(f->t, f->s7, w0);
    hle_st32(f->t, f->s7 + 4, w1);
    f->s7 += 8;
}

/* the callbacks of the kernels: the segment address resolution, and the flush check after a strip */
static uint32_t s2c_seg(void *ctx, uint32_t a) { return seg((struct f3dex_task *)ctx, a); }
static int s2c_flush(void *ctx, uint32_t *s7, uint32_t *s6)
{
    struct f3dex_task *f = ctx;
    int rc;
    (void)s6;
    f->s7 = *s7;
    rc = s2_chk(f);
    *s7 = f->s7;
    return rc != HLE_OK;
}

/* a display list call / branch to the (segment) address `a`; w0's byte 1 is the flag (0: call) */
static int s2_do_dl(struct f3dex_task *f, uint32_t w0, uint32_t a)
{
    struct hle_task *t = f->t;
    unsigned d = hle_dm8(t, S2_DEPTH);
    if (((w0 >> 16) & 0xff) == 0) {
        if (MUT(521, d >= 0x48, 0))                         /* (the stack has room for four entries before it runs over the microcode's banner and constants; a game nests two) */
            return hle_decline(t, "G_DL: display list stack deeper than a game's");
        hle_st32(t, S2_STACK + d, f->k0);
        hle_st8(t, S2_DEPTH, (uint8_t)(d + 4));
    }
    return jump_to(f, seg(f, a));
}

/* G_OBJ_RECTANGLE (and the rectangle part of LDTX_RECT): the 24-byte uObjSprite at w1 into DMEM 0x530 (`fresh`: still to be fetched), then the closed form */
static int s2_rect(struct f3dex_task *f, uint32_t w1, int fresh, int rot)
{
    struct hle_task *t = f->t;
    unsigned s7 = f->s7;
    int rc;
    s2_ovl(f, 1);
    if (fresh)
        s2_dma_rd(f, S2_SPRITE, seg(f, w1), 23);
    rc = rot ? s2dex1_obj_rect_r(t->dmem, &s7) : s2dex1_obj_rect(t->dmem, &s7);
    if (rc == S2_UNSUPPORTED)
        return hle_decline(t, "S2DEX rectangle");
    if (rc == S2_CLIPPED)
        return HLE_OK;                                      /* (outside the scissor: back to the loop head, no flush check) */
    f->s7 = s7;
    return s2_chk(f);
}

/* G_OBJ_SPRITE (and the sprite part of LDTX_SPRITE): the sprite through the object matrix, two triangles (s2dex1_sprite.c) */
static int s2_sprite(struct f3dex_task *f, uint32_t w1, int fresh)
{
    struct hle_task *t = f->t;
    unsigned s7 = f->s7;
    int rc;
    s2_ovl(f, 1);
    if (fresh)
        s2_dma_rd(f, S2_SPRITE, seg(f, w1), 23);
    rc = s2dex1_obj_sprite(t->dmem, &s7);
    if (rc == S2_UNSUPPORTED)
        return hle_decline(t, "S2DEX sprite");
    if (rc == S2_CLIPPED)
        return HLE_OK;                                      /* (no corner inside the scissor: back to the loop head; the scratch word the microcode writes is written) */
    f->s7 = s7;
    return s2_chk(f);
}

/* G_OBJ_LOADTXTR (and the load part of the LDTX_ commands): the texture object (uObjTxtr, 24 bytes; 48 with the sprite behind it) to DMEM 0x518; when its load bits are not in the loaded-texture word yet (0x2a0 +
 * its class), SET_TEXTURE_IMAGE, SET_TILE, SYNC_LOAD and the load are emitted (after a SYNC_TILE when the state byte 0xd2 says so). The words are the 2.x routine's (hle_s2dex2.c), addresses 0x170 higher. */
static int s2_loadtxtr(struct f3dex_task *f, uint32_t w0, uint32_t w1, int *early)
{
    struct hle_task *t = f->t;
    uint32_t t8, a0, a1, a2, a3, t0, t1, v1;
    int32_t at;
    unsigned s7, k;
    *early = 0;
    t8 = seg(f, w1);
    if (MUT(522, (w0 & 255) >= 0x70, 0))                     /* the object's length is the command's low byte: 24 or 48 in a game; more runs over the output buffer */
        return hle_decline(t, "S2DEX texture object longer than its buffer");
    s2_dma_rd(f, S2_TXTR, t8, w0 & 255);
    at = (int8_t)hle_dm8(t, S2_TSTATE);
    v1 = hle_dm8(t, S2_TXTR + 15);
    if (MUT(523, v1 > 8, 0))                                 /* the loaded-texture word of the object's class: a game's classes are 0 and 8 (the word of 8 ends in the cell of opcode 0x80, which no game sends); further ones store over the microcode's own state (the jump table ...) */
        return hle_decline(t, "S2DEX texture class outside the state words");
    a3 = s2_ld32(t, S2_TXTR + 20);
    a1 = s2_ld32(t, S2_TXTR + 16);
    a0 = s2_ld32(t, S2_FLAGS + v1);
    a2 = a1 & a3;
    t1 = a0 & a3;
    if (t1 == a1) {                                          /* the bits are loaded already */
        *early = 1;
        return HLE_OK;
    }
    t1 = (a0 & ~a3) | a2;
    st32u(t, S2_FLAGS + v1, t1);
    s2_cells_changed(f, S2_FLAGS + v1, 4);
    hle_st8(t, S2_TSTATE, (uint8_t)MUT(524, -127, -126));
    s7 = f->s7;
    if (at == -127) {
        hle_st8(t, s7, hle_dm8(t, S2_V29 + 15));
        s7 += 8;
    }
    a0 = s2_ld16(t, S2_TXTR + 10);
    hle_st8(t, s7, hle_dm8(t, S2_V29));
    hle_st8(t, s7 + 1, hle_dm8(t, S2_V29 + 1));
    t8 = seg(f, s2_ld32(t, S2_TXTR + 4));
    hle_st16(t, s7 + 2, (uint16_t)a0);
    v1 = s2_ld16(t, S2_TXTR + 8);
    st32u(t, s7 + 4, t8);
    t0 = (uint32_t)(int32_t)(int8_t)hle_dm8(t, S2_TXTR + 1);
    a3 = hle_dm8(t, S2_TXTR + 2);
    a1 = hle_dm8(t, S2_TXTR + 3);
    a2 = ((a0 + 1) & t0) << 7;
    a3 <<= MUT(525, 16, 15);
    v1 |= a2;
    v1 |= a3;
    st32u(t, s7 + 8, v1);
    hle_st8(t, s7 + 8, hle_dm8(t, S2_V29 + 12));
    for (k = 0; k < 4; k++)                                 /* SLV v29, 8: the bytes 8 .. 11 */
        hle_st8(t, s7 + 12 + k, hle_dm8(t, S2_V29 + 8 + k));
    if (MUT(526, at >= 0, at > 0)) {
        hle_st8(t, s7 + 16, hle_dm8(t, S2_V29 + 14));
        s7 += 8;
    }
    at = (int32_t)s2_ld16(t, S2_TXTR + 12);
    a1 <<= 24;
    st32u(t, s7 + 16, a1);
    a0 = (a0 << 14) | (uint32_t)at;
    st32u(t, s7 + 20, a0);
    hle_st8(t, s7 + 20, hle_dm8(t, S2_V29 + 8));
    f->s7 = s7 + 24;
    return s2_chk(f);
}

/* G_LOAD_UCODE: the microcode's handler on the interpreter (load_ucode: the registers as the loop leaves them) */
static int s2_loaduc(struct f3dex_task *f, struct hle_cpu *c, uint32_t w0, uint32_t w1)
{
    int r = load_ucode(f, c, w0, w1);
    return r == HLE_STOP ? HLE_OK : r == HLE_OK ? S2R_FINISH : HLE_DECLINE;
}

static int s2_cmd(struct f3dex_task *f, struct hle_cpu *c, unsigned op, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    const struct s2d1_build *b = f->s2;
    unsigned kd = f->kind[op];
    switch (kd) {
    case S2K_NOP:
        return HLE_OK;
    case S2K_HALF0:                                         /* the first words of G_TEXRECT / G_SELECT_DL's address: kept at 0xd8 / 0xdc */
        hle_st32(t, S2_H0, w0);
        hle_st32(t, S2_H0 + 4, w1);
        return HLE_OK;
    case S2K_HALF1:
        hle_st32(t, S2_H1, w1);
        return HLE_OK;
    case S2K_HALF2:                                         /* completes a texture rectangle: 16 bytes */
        hle_st32(t, f->s7, hle_dm32(t, S2_H0));
        hle_st32(t, f->s7 + 4, hle_dm32(t, S2_H0 + 4));
        hle_st32(t, f->s7 + 8, hle_dm32(t, MUT(527, S2_H1, S2_H1 + 4)));
        hle_st32(t, f->s7 + 12, w1);
        f->s7 += 16;
        return s2_chk(f);
    case S2K_ENDDL: {
        int e = enddl(f);
        if (e > 0)
            return S2R_END;
        if (e < 0)
            return hle_decline(t, "display list address not 8-byte aligned");
        return HLE_OK;
    }
    case S2K_OMH: case S2K_OML: {                           /* the word at pc + 0x78: 0xc8 / 0xcc; the new bits are merged and the command is emitted unless another G_SETOTHERMODE_H / _L follows in the window */
        unsigned a = (b->pc[kd == S2K_OML ? S2P_oml : S2P_omh] + 0x78) & 0xfff;
        uint32_t a0 = ~0u << ((w0 >> 8) & 31), v1 = (a0 << ((w0 - 1) & 31)) << 1;
        v1 = ~(v1 ^ a0);
        hle_st32(t, a, (v1 & hle_dm32(t, a)) | w1);
        if (f->gp > 0) {
            unsigned nx = hle_rd32(t, f->k0) >> 24;
            if (MUT(528, nx == 0xb9 || nx == 0xba, 0))
                return HLE_OK;
        }
        s2_put(f, hle_dm32(t, S2_OMH), hle_dm32(t, S2_OML));
        return s2_chk(f);
    }
    case S2K_MOVEWORD: {                                    /* the word goes to the table entry of the index + the offset: the segment table (index 6) or the select-DL flags */
        unsigned ii = w0 & 0xff, dst;
        if ((ii & 1) || ii > 8)
            return hle_decline(t, "G_MOVEWORD index outside the table");
        dst = (hle_dm16(t, S2_MWTAB + ii) + (w0 >> 8)) & 0xfff;
        if (MUT(529, !((dst >= S2_SEGS && dst + 3 < S2_STACK) || (dst >= S2_FLAGS && dst + 3 < S2_JT_LO)), 0))     /* (a store elsewhere would be over the microcode's own state: the jump table, the display list window ..) */
            return hle_decline(t, "G_MOVEWORD outside what a game writes");
        st32u(t, dst, w1);
        return HLE_OK;
    }
    case S2K_RENDERMODE: {                                  /* the render mode word w1 selects the other-mode words (0x1e8, 0x1ec) and the two table pointers (0x252, 0x254) */
        unsigned at;
        uint32_t v1;
        hle_st8(t, S2_RMBYTE, (uint8_t)w1);
        hle_st16(t, S2_RMBYTE + 2, (uint16_t)(0x230 + (w1 & 8)));
        hle_st16(t, S2_RMBYTE + 4, (uint16_t)(0x240 + (w0 & 8)));
        at = (w1 & 24) >> 1;
        v1 = s2_ld32(t, MUT(530, 0x220, 0x224) + at);
        hle_st32(t, 0x1ec, v1);
        at = (w1 & 112) >> 2;
        hle_st32(t, 0x1e8, s2_ld32(t, 0x200 + at));
        return HLE_OK;
    }
    case S2K_SETOM:
        hle_st32(t, S2_OMH, w0);
        hle_st32(t, S2_OML, w1);
        s2_put(f, w0, w1);
        return s2_chk(f);
    case S2K_SEGCOPY:
        s2_put(f, w0, seg(f, w1));
        return s2_chk(f);
    case S2K_SCISSOR:                                       /* the four fields are decoded for the sprite clipping (0xc0 .. 0xc6), then the command is copied */
        hle_st16(t, S2_SCISSOR + 4, (uint16_t)(w0 & 0xfff));
        hle_st16(t, S2_SCISSOR, (uint16_t)((w0 >> MUT(531, 12, 13)) & 0xfff));
        hle_st16(t, S2_SCISSOR + 6, (uint16_t)(w1 & 0xfff));
        hle_st16(t, S2_SCISSOR + 2, (uint16_t)((w1 >> 12) & 0xfff));
        s2_put(f, w0, w1);
        return s2_chk(f);
    case S2K_SYNC:                                          /* the syncs forget the tile state byte */
        hle_st8(t, S2_TSTATE, 0);
        s2_put(f, w0, w1);
        return s2_chk(f);
    case S2K_COPY:
        s2_put(f, w0, w1);
        return s2_chk(f);
    case S2K_SELDL: {                                       /* a display list call / branch, taken only when the flags selected by w1 change the flags word; the index byte is the one left at 0xd9 by the earlier commands */
        unsigned at = hle_dm8(t, S2_H0 + 1);
        uint32_t v1 = hle_dm32(t, S2_H0 + 4), v0 = s2_ld32(t, S2_FLAGS + at), a0 = v1 & w1, a2 = v0 & w1;
        hle_st16(t, S2_H0, (uint16_t)w0);
        if (a2 == v1)
            return HLE_OK;
        if (MUT(532, at > 8, 0))                            /* (the flags word of a game's index lies at 0x2a0 .. 0x2a7, as the texture state words; beyond 0x2ab a store is over the jump table) */
            return hle_decline(t, "G_SELECT_DL flags word over the microcode's own state");
        st32u(t, S2_FLAGS + at, (v0 & ~w1) | a0);
        s2_cells_changed(f, S2_FLAGS + at, 4);
        return s2_do_dl(f, w0, hle_dm32(t, S2_H0));
    }
    case S2K_DL:
        return s2_do_dl(f, w0, w1);
    case S2K_MOVEMEM: {                                     /* G_OBJ_MOVEMEM: the byte 1 + 1 bytes from the segment address to the table entry of w0 */
        unsigned dst = hle_dm16(t, S2_MMTAB + (w0 & 0xfff)), len = (w0 >> 16) & 0xff;
        if (MUT(533, dst + len >= S2_JT_LO && dst < S2_JT_HI, 0))   /* (not the jump table or the window: the loop works from what it classified) */
            return hle_decline(t, "G_OBJ_MOVEMEM into the microcode's own state");
        s2_dma_rd(f, dst, seg(f, w1), len);
        return HLE_OK;
    }
    case S2K_BG1CYC: {
        struct s2_ctx cx = { f, s2c_seg, s2c_flush };
        uint32_t s7 = f->s7, s6 = 0;
        s2_ovl(f, 2);
        s2_dma_rd(f, S2_BGOBJ, seg(f, w1), 39);
        if (s2dex1_obj_bg1cyc(t->dmem, &s7, &s6, &cx) != S2_DONE)
            return hle_decline(t, "S2DEX BG_1CYC routine");
        f->s7 = s7;
        return HLE_OK;
    }
    case S2K_BGCOPY: {                                      /* the routine is in the resident code (no overlay) */
        struct s2_ctx cx = { f, s2c_seg, s2c_flush };
        uint32_t s7 = f->s7, s6 = 0;
        s2_dma_rd(f, S2_BGOBJ, seg(f, w1), 39);
        if ((b->bgc107 ? s2dex1_obj_bgcopy(t->dmem, &s7, &s6, &cx) : s2dex1_obj_bgcopy106(t->dmem, &s7, &s6, &cx)) != S2_DONE)
            return hle_decline(t, "S2DEX BG_COPY routine");
        f->s7 = s7;
        return HLE_OK;
    }
    case S2K_RECT:
        return s2_rect(f, w1, 1, (int32_t)w0 < 0);
    case S2K_LOADTXTR: {                                    /* the continuation (the table at 0x1f8 by the opcode): the loop head, the sprite code, the rectangle code (the rotated one when the entry is negative) */
        int32_t c0 = (int16_t)s2_ld16(t, 0x76 + (w0 >> 23));
        unsigned cont = (unsigned)c0 & 0xffc;
        int early;
        if (cont != b->pc[S2P_nop] && cont != b->pc[S2P_ldtx_rect] && cont != b->pc[S2P_ldtx_spr])
            return hle_decline(t, "S2DEX LOADTXTR continuation");
        if (s2_loadtxtr(f, w0, w1, &early) != HLE_OK)
            return HLE_DECLINE;
        if (cont == b->pc[S2P_nop])
            return HLE_OK;
        if (cont == b->pc[S2P_ldtx_spr])
            return s2_sprite(f, w1, 0);
        return s2_rect(f, w1, 0, c0 < 0);
    }
    case S2K_LOADUC:
        return s2_loaduc(f, c, w0, w1);
    case S2K_SPRITE:
        return s2_sprite(f, w1, 1);
    default: {
        static char why[8][40];
        static unsigned wn;
        char *w = why[wn++ & 7];
        snprintf(w, sizeof(why[0]), "S2DEX command %02x", op);
        return hle_decline(t, w);
    }
    }
}

/* The end of the task (the outermost G_ENDDL): the chunks staged so far go to the FIFO, and the microcode's own code finishes the task on the interpreter (it flushes what is in the output buffer, sets SP_STATUS and breaks) */
static int s2_end(struct f3dex_task *f, struct hle_cpu *c)
{
    struct hle_task *t = f->t;
    sync_out(f);
    c->r[HLE_REG_S7] = f->s7;
    c->pc = f->s2->pc[S2P_enddl];
    c->stop_pc = 0x1000;
    (void)t;
    return hle_rsp_run(t, c);
}

#endif
