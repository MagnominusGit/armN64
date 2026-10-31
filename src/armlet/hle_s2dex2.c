/*
 * arMLEt - hle_s2dex2.c: S2DEX 2.x (fifo 2.05 [Safe] / 2.06 / 2.08: 2D sprites and backgrounds) as a dialect of the F3DEX2 front-end (hle_f3dex2.c)
 *
 * The microcode is built on the F3DEX2 skeleton (display list refill, flush of the RDP buffer, DL stack, the RDP command copies); the handlers sit at other pcs, there is
 * no vertex, matrix or triangle code, and the sprite / background commands are its own. docs/ucode-s2dex2.md has the semantics.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "hle_f3dex2_int.h"
#include "s2dex2_fx.h"

#ifdef ARMLET_NATIVE

enum { K_S2_SELDL = K_EXT, K_S2_LOADTXTR, K_S2_BG1CYC, K_S2_BGCOPY, K_S2_RENDERMODE, K_S2_RECT, K_S2_SPRITE, K_S2_MOVEMEM, K_S2_SCISSOR, K_S2_HALF2 };

/* the layout of S2DEX 2.x: the jump table, the display list input buffer (refilled in 168-byte chunks) and the RDP output buffer (800 bytes + slack, two of them 0xc00 apart) */
static const struct f2_layout l_lay_s2 = {
    .jump_table = 0x370, .mw_max = 14, .geom_def = F2_GEOM_DEF, .in = 0x418, .in_end = 0x4c0, .moveword_adj = 10374, .out_bytes = 800, .out_toggle = 0xc00,
    .desc_clip = 0x314, .desc_light = 0x31c };       /* the two code overlays of the microcode at IMEM 0x6c8 (sprites, rectangles) and (backgrounds) take the slots of the clipping and lighting overlays */

static const struct f2_pcs l_pcs_s2 = { .dl = 0x054, .texture = MUT(248, 0xffff, 0xfff), .texrect = 0x018, .half1 = 0x01c, .moveword = 0x090, .popmtx = 0xffff, .movemem = 0xffff, .omh = 0x0b4, .oml = 0x0b8 };
static const uint16_t l_ext_s2[][2] = {
    { 0x024, K_S2_SELDL }, { 0x558, K_S2_LOADTXTR }, { 0xb7c, K_S2_BG1CYC }, { 0x210, K_S2_BGCOPY }, { 0x0e8, K_S2_RENDERMODE }, { 0xb88, K_S2_RECT }, { 0x6c8, K_S2_SPRITE },
    { 0x0a4, K_S2_MOVEMEM }, { 0x1d8, K_S2_SCISSOR }, { 0x128, K_S2_HALF2 }, { 0xffff, 0 } };

/* overlay A (sprites, rectangles) is OV_CLIP, overlay B (backgrounds) is OV_LIGHT: the descriptors are at DMEM 0x314 and 0x31c */
enum { S2_OV_A = OV_CLIP, S2_OV_B = OV_LIGHT };

/* append n bytes (a multiple of 8) of RDP command to the output buffer: the handlers' own stores, then the flush check of the code at 0x648 / 0x64c */
static int s2_emit_done(struct f3 *f) { return after_emit(f); }

/* halfword / word accesses at any byte address, as the RSP's LH / LHU / LW / SH / SW make them (a byte at a time, wrapping inside DMEM) */
static inline unsigned s2_lhu(const struct hle_task *t, unsigned a) { return hle_dm8(t, a) << 8 | hle_dm8(t, a + 1); }
static inline int32_t s2_lh(const struct hle_task *t, unsigned a) { return (int16_t)s2_lhu(t, a); }
static inline uint32_t s2_lw(const struct hle_task *t, unsigned a) { return s2_lhu(t, a) << 16 | s2_lhu(t, a + 2); }
static inline void s2_sh(struct hle_task *t, unsigned a, unsigned v) { hle_st8(t, a, (uint8_t)(v >> 8)); hle_st8(t, a + 1, (uint8_t)v); }
static inline void s2_sw(struct hle_task *t, unsigned a, uint32_t v) { s2_sh(t, a, v >> 16); s2_sh(t, a + 2, v); }

/* the constants of the microcode's vector register v29 (RDP command ids and a few fields), which it keeps in DMEM 0x1a0 .. 0x1af */
static inline unsigned s2_v29b(const struct hle_task *t, unsigned i) { return hle_dm8(t, 0x1a0 + i); }

/* G_OBJ_LOADTXTR (and the load part of the LDTX_ commands): the texture object (uObjTxtr) is read to DMEM 0x3a8; when its load bits are not in the loaded-texture word yet
 * (0x2b0 + the object's tmem class), SET_TEXTURE_IMAGE, SET_TILE, SYNC_LOAD and the load command are emitted (after a SYNC_TILE when the sync state byte at 0x2ae says so).
 * *cont = the pc the microcode continues at: 0x1ac (the loop head), 0x6e8 (sprite), 0xba8 (rectangle); *neg: the rectangle is the rotated one. */
static int s2_loadtxtr(struct f3 *f, unsigned *cont, int *neg)
{
    struct hle_task *t = f->t;
    uint32_t w0 = f->w0, w1 = f->w1, t8, a0, a1, a2, a3, t0, t1, v1;
    int32_t at;
    unsigned s7 = f->s7, k;
    int32_t c = s2_lh(t, 0x202 + (w0 >> 23));

    *cont = (unsigned)c & 0xffc;
    *neg = c < 0;
    t8 = seg_addr(t, w1);
    if (MUT(241, f3_store_unmodelled(f, 0x3a8, w0 & 255), 0))     /* the object's length is the command's low byte: 24 bytes in a game; up to 256 run over the display list buffer and the state behind it */
        return hle_decline(t, "S2DEX texture object longer than its buffer");
    f3_dma_rd(f, 0x3a8, t8, w0 & 255);
    at = (int8_t)hle_dm8(t, 0x2ae);
    v1 = hle_dm8(t, 0x3a8 + 15);
    if (MUT(247, f3_store_unmodelled(f, 0x2b0 + v1, 3), 0))   /* the loaded-texture word of the object's class: the object picks it by a byte, and a word past the two there stores over the microcode's own state (the jump table ...) */
        return hle_decline(t, "S2DEX texture class outside the state words");
    a3 = s2_lw(t, 0x3a8 + 20);
    a1 = s2_lw(t, 0x3a8 + 16);
    a0 = s2_lw(t, 0x2b0 + v1);
    t0 = ~a3;
    a2 = a1 & a3;
    t1 = a0 & a3;
    if (t1 == a1)                                       /* the bits are loaded already */
        return MUT(309, HLE_OK, after_emit(f));
    t1 = (a0 & t0) | a2;
    s2_sw(t, 0x2b0 + v1, t1);
    hle_st8(t, 0x2ae, (uint8_t)MUT(37, -127, -126));
    if (at == -127) {
        hle_st8(t, s7, (uint8_t)s2_v29b(t, 15));
        s7 += 8;
    }
    a0 = s2_lhu(t, 0x3a8 + 10);
    s2_sh(t, s7, s2_v29b(t, 0) << 8 | s2_v29b(t, 1));
    t8 = seg_addr(t, s2_lw(t, 0x3a8 + 4));
    s2_sh(t, s7 + 2, a0);
    v1 = s2_lhu(t, 0x3a8 + 8);
    s2_sw(t, s7 + 4, t8);
    t0 = (uint32_t)(int32_t)(int8_t)hle_dm8(t, 0x3a8 + 1);
    a3 = hle_dm8(t, 0x3a8 + 2);
    a1 = hle_dm8(t, 0x3a8 + 3);
    a2 = ((a0 + 1) & t0) << 7;
    a3 <<= MUT(35, 16, 15);
    v1 |= a2;
    v1 |= a3;
    s2_sw(t, s7 + 8, v1);
    hle_st8(t, s7 + 8, (uint8_t)s2_v29b(t, 12));
    for (k = 0; k < 4; k++)                              /* SLV v29, 8: the bytes 8 .. 11 */
        hle_st8(t, s7 + 12 + k, (uint8_t)s2_v29b(t, 8 + k));
    if (MUT(36, at >= 0, at > 0)) {
        hle_st8(t, s7 + 16, (uint8_t)s2_v29b(t, 14));
        s7 += 8;
    }
    at = (int32_t)s2_lhu(t, 0x3a8 + 12);
    a1 <<= 24;
    s2_sw(t, s7 + 16, a1);
    a0 = (a0 << 14) | (uint32_t)at;
    s2_sw(t, s7 + 20, a0);
    hle_st8(t, s7 + 20, (uint8_t)s2_v29b(t, 8));
    f->s7 = s7 + 24;
    return after_emit(f);
}

/* G_OBJ_RECTANGLE (and the rectangle part of LDTX_RECT): the 24-byte uObjSprite at w1 is read to DMEM 0x3c0 (LDTX_RECT: after the load; `fresh`: the object address still has to be resolved) */
static int s2_rect(struct f3 *f, uint32_t w1, int fresh)
{
    struct hle_task *t = f->t;
    unsigned s7;
    int rc;
    if (f->ov == OV_UNKNOWN)
        return hle_decline(t, "unknown overlay");
    if (f->ov != S2_OV_A)
        f3_overlay_load(f, S2_OV_A);
    if (fresh)
        f3_dma_rd(f, 0x3c0, seg_addr(t, w1), 23);
    s7 = f->s7;
    rc = s2dex2_obj_rect(t->dmem, &s7);
    if (rc != S2_DONE)
        return rc == S2_CLIPPED ? MUT(308, HLE_OK, after_emit(f)) : hle_decline(t, "S2DEX rectangle");
    f->s7 = s7;
    return after_emit(f);
}

/* the two pieces of the microcode that G_BG_1CYC calls: the segment address resolution (0x154) and the flush check of the RDP buffer (0x64c) */
static uint32_t s2_cb_seg(void *ctx, uint32_t a)
{
    struct f3 *f = ctx;
    return seg_addr(f->t, a);
}
static int s2_cb_flush(void *ctx, uint32_t *s7, uint32_t *s6)
{
    struct f3 *f = ctx;
    int rc;
    f->s7 = *s7;
    f->s6 = *s6;
    rc = after_emit(f);
    *s7 = f->s7;
    *s6 = f->s6;
    return rc != HLE_OK;
}

static int s2_ext_cmd(struct f3 *f, unsigned kind)
{
    struct hle_task *t = f->t;
    uint32_t w0 = f->w0, w1 = f->w1;
    switch (kind) {
    case K_S2_SCISSOR:                                  /* G_SETSCISSOR: the command is remembered (0xc0 / 0xc4) and its fields decoded for the sprite clipping (halfwords 0x204 .. 0x20a), then copied */
        hle_st32(t, 0xc0, w0);
        hle_st32(t, 0xc4, w1);
        hle_st16(t, 0x208, (uint16_t)(w0 & 0xfff));
        hle_st16(t, 0x204, (uint16_t)((w0 >> MUT(44, 12, 13)) & 0xfff));
        hle_st16(t, 0x20a, (uint16_t)(w1 & 0xfff));
        hle_st16(t, 0x206, (uint16_t)((w1 >> 12) & 0xfff));
        hle_st32(t, f->s7 + 4, w1);
        hle_st32(t, f->s7, w0);
        f->s7 += 8;
        return s2_emit_done(f);
    case K_S2_HALF2:                                    /* G_RDPHALF_2 completes a texture rectangle: the remembered scissor (0xc0), then the rectangle words (0xd0, 0xd4, 0xd8, w1): 24 bytes */
        hle_st32(t, f->s7, hle_dm32(t, 0xc0));
        hle_st32(t, f->s7 + 4, hle_dm32(t, 0xc4));
        hle_st32(t, f->s7 + 8, hle_dm32(t, 0xd0));
        hle_st32(t, f->s7 + 12, hle_dm32(t, 0xd4));
        hle_st32(t, f->s7 + 16, hle_dm32(t, MUT(39, 0xd8, 0xdc)));
        hle_st32(t, f->s7 + 20, w1);
        f->s7 += 24;
        return s2_emit_done(f);
    case K_S2_RENDERMODE: {                             /* G_OBJ_RENDERMODE: the render mode word w1 selects the other-mode words (0x1f8, 0x1fc) and the two table pointers (0x244, 0x246) */
        unsigned at;
        hle_st8(t, 0x268, (uint8_t)w1);
        hle_st16(t, 0x244, (uint16_t)(0x248 + (w1 & 8)));
        hle_st16(t, 0x246, (uint16_t)(0x258 + (w0 & 8)));
        at = (w1 & 24) >> 1;
        {
            uint32_t v1 = s2_lw(t, MUT(38, 0x234, 0x238) + at);
            at = (w1 & 112) >> 2;
            hle_st32(t, 0x1fc, v1);
            hle_st32(t, 0x1f8, s2_lw(t, 0x214 + at));
        }
        return HLE_OK;
    }
    case K_S2_SELDL: {                                  /* G_SELECT_DL: a display list call / branch, taken only when the flags selected by w1 change the flags word */
        unsigned at = hle_dm8(t, 0xd1);
        uint32_t v1 = hle_dm32(t, 0xd4), v0 = s2_lw(t, 0x2b0 + at), a0 = v1 & w1, a2;
        hle_st16(t, 0xd0, (uint16_t)w0);
        if ((v0 & w1) == v1)
            return HLE_OK;
        a2 = (v0 & ~w1) | a0;
        if (MUT(315, f3_store_unmodelled(f, 0x2b0 + at, 4), 0))     /* the flags word is addressed by a byte of w0 (a game's are the first few): 0x2b0 + 0xbe is the jump table's entry of opcode 0, whose classification the loop caches */
            return hle_decline(t, "G_SELECT_DL flags word over the microcode's own state");
        s2_sw(t, 0x2b0 + at, a2);
        f3_do_dl(f, w0, hle_dm32(t, 0xd0));
        return HLE_OK;
    }
    case K_S2_LOADTXTR: {                               /* (the continuation pcs are the build record's: the loop head, the sprite and the rectangle code (f3dex2_build.ldtx_spr / ldtx_rect)) */
        unsigned cont, loop = MUT(258, f->b->head, 0x1ac), spr = MUT(258, f->b->ldtx_spr, 0x6e8), rct = MUT(258, f->b->ldtx_rect + MUT(304, 0, 8), 0xba8);
        int neg, rc;
        int32_t c = s2_lh(t, 0x202 + (w0 >> 23));
        if (((unsigned)c & 0xffc) == spr || (((unsigned)c & 0xffc) == rct && c < 0))
            return hle_decline(t, "S2DEX LDTX_SPRITE / LDTX_RECT_R");
        rc = s2_loadtxtr(f, &cont, &neg);
        if (rc != HLE_OK || cont == loop)
            return rc;
        if (cont != rct)
            return hle_decline(t, "S2DEX LOADTXTR continuation");
        return s2_rect(f, w1, 0);
    }
    case K_S2_RECT:
        if ((int32_t)w0 < 0)
            return hle_decline(t, "S2DEX OBJ_RECTANGLE_R");
        return s2_rect(f, w1, 1);
    case K_S2_BG1CYC: {
        struct s2_ctx c = { f, s2_cb_seg, s2_cb_flush };
        uint32_t s7 = f->s7, s6 = f->s6;
        if (f->ov == OV_UNKNOWN)
            return hle_decline(t, "unknown overlay");
        if (f->ov != S2_OV_B)
            f3_overlay_load(f, S2_OV_B);                                 /* G_BG_1CYC: the uObjBg (40 bytes) to DMEM 0x3b0, then the background is drawn as strips of texture rectangles */
        f3_dma_rd(f, 0x3b0, seg_addr(t, w1), 39);
        if (getenv("S2_DUMP_BG1")) {
            FILE *fp = fopen(getenv("S2_DUMP_BG1"), "ab");
            if (fp) {
                uint32_t hdr[3] = { s7, s6, 0 };
                fwrite(hdr, 4, 3, fp);
                fwrite(t->dmem, 4, 1024, fp);
                fclose(fp);
            }
        }
        if (s2dex2_obj_bg1cyc(t->dmem, &s7, &s6, &c) != S2_DONE)
            return hle_decline(t, "S2DEX BG_1CYC routine");
        f->s7 = s7;
        f->s6 = s6;
        return HLE_OK;
    }
    case K_S2_BGCOPY: {                                 /* G_BG_COPY: the uObjBg (40 bytes) to DMEM 0x3b0, then the background is drawn as strips of texture rectangles (the code is in the resident part: no overlay) */
        struct s2_ctx c = { f, s2_cb_seg, s2_cb_flush };
        uint32_t s7 = f->s7, s6 = f->s6;
        int rc;
        f3_dma_rd(f, 0x3b0, seg_addr(t, w1), 39);
        if (getenv("S2_DUMP_BGC")) {
            FILE *fp = fopen(getenv("S2_DUMP_BGC"), "ab");
            if (fp) {
                uint32_t hdr[3] = { s7, s6, 0 };
                fwrite(hdr, 4, 3, fp);
                fwrite(t->dmem, 4, 1024, fp);
                fclose(fp);
            }
        }
        rc = s2dex2_obj_bgcopy(t->dmem, &s7, &s6, &c);
        if (rc != S2_DONE)
            return hle_decline(t, "S2DEX BG_COPY routine");
        f->s7 = s7;
        f->s6 = s6;
        return HLE_OK;
    }
    case K_S2_MOVEMEM: {                                /* G_OBJ_MOVEMEM: size from bits 16..23, the DMEM address from the table at 0x388 indexed by the low bits of w0 */
        unsigned dst = hle_dm16(t, w0 + 904);
        if (MUT(237, f3_store_unmodelled(f, dst, (w0 >> 16) & 0xff), 0))
            return hle_decline(t, "G_OBJ_MOVEMEM into the microcode's own state");
        f3_dma_rd(f, dst, seg_addr(t, w1), (w0 >> 16) & 0xff);
        return HLE_OK;
    }
    default: {
        static char why[8][48];
        static unsigned wn;
        char *w = why[wn++ & 7];
        snprintf(w, sizeof(why[0]), "S2DEX command %02x", w0 >> 24);
        return hle_decline(t, w);
    }
    }
}

static const struct f3dex2_build l_b_s2_208 = {
    .head = 0x1ac, .copy = 0x140, .sync = 0x144, .segcopy = 0x150, .ef = 0x170, .ed = MUT(248, 0xffff, 0xfff), .tri1 = 0, .tri2 = 0, .vtx = 0,
    .mtx = 0, .geom = 0, .enddl = 0x07c, .half2 = MUT(248, 0xffff, 0xfff), .modvtx = 0, .culldl = 0, .branchz = 0, .reject_mask = 0, .tri4 = 0, .ldtx_spr = 0x6e8, .ldtx_rect = 0xba8, .lay = &l_lay_s2, .pcs = &l_pcs_s2, .ext = l_ext_s2, .ext_cmd = s2_ext_cmd, .load_key = 0xe81031ed857b6664ull };

/* dead at the end of a task: the display list input buffer (0x418 .. 0x4c0; a lean run reads the list from RDRAM) and the two RDP output buffers with their slack (0x4c0 .. 0x890 and 0x8c0 .. 0xc90: 800 bytes each (792 in the 2.05 build),
 * 0xc00 apart, 0xb0 of slack for the longest command): the handlers write the commands there and the flush DMAs them to the FIFO. */
static const struct hle_dead_range l_dead_s2[] = { { 0x418, 0x890 }, { 0x8c0, 0xc90 }, { 0xfc8, 0xfcc } };
#define S2DEAD .ring_off = 0xfe8, .dead = l_dead_s2, .n_dead = sizeof(l_dead_s2) / sizeof(l_dead_s2[0])
/* The S2DEX 2.06 text (Kirby 64's, loaded by G_LOAD_UCODE in the middle of the F3DEX2 2.04H tasks: docs/ucode-f3dex2-dialects.md section 5.6; Resident Evil 2's, which starts tasks with it): Worms' 2.08 with its
 * handlers from 0x184 on 8 bytes lower (one initialisation instruction less, a nop less), the same DMEM layout, constants and overlay code; the continuation pcs of G_OBJ_LOADTXTR are shifted with them (ldtx_spr / ldtx_rect). */
static const uint16_t l_ext_s2k[][2] = {
    { 0x024, K_S2_SELDL }, { 0x550, K_S2_LOADTXTR }, { 0xb74, K_S2_BG1CYC }, { 0x208, K_S2_BGCOPY }, { 0x0e8, K_S2_RENDERMODE }, { 0xb80, K_S2_RECT }, { 0x6c0, K_S2_SPRITE },
    { 0x0a4, K_S2_MOVEMEM }, { 0x1d0, K_S2_SCISSOR }, { 0x128, K_S2_HALF2 }, { 0xffff, 0 } };
static const struct f3dex2_build l_b_s2_kirby = {
    .head = 0x1a4, .copy = 0x140, .sync = 0x144, .segcopy = 0x150, .ef = 0x170, .ed = MUT(248, 0xffff, 0xfff), .tri1 = 0, .tri2 = 0, .vtx = 0,
    .mtx = 0, .geom = 0, .enddl = 0x07c, .half2 = MUT(248, 0xffff, 0xfff), .modvtx = 0, .culldl = 0, .branchz = 0, .reject_mask = 0, .tri4 = 0, .ldtx_spr = 0x6e0, .ldtx_rect = 0xba0, .lay = &l_lay_s2, .pcs = &l_pcs_s2, .ext = l_ext_s2k, .ext_cmd = s2_ext_cmd, .load_key = 0x9748556461d85ab3ull };
HLE_UCODE(l_s2_kirby) = { .id = 0xc9655420d2f5f8cfull, .name = "S2DEX fifo 2.06 (Kirby 64, loaded by G_LOAD_UCODE)", .fam = &f3dex2_fam, .fam_data = &l_b_s2_kirby, S2DEAD };

/* Resident Evil 2's S2DEX (fifo 2.06 in its banner; 75 % of its graphics tasks start with it) is the same text as Kirby 64's, the first 0xf48 bytes word for word (the key is the same), and runs as a task of its own: the same build. */
HLE_UCODE(l_s2_re2) = { .id = 0x5f33af89e713878bull, .name = "S2DEX fifo 2.06 (Resident Evil 2)", .fam = &f3dex2_fam, .fam_data = &l_b_s2_kirby, S2DEAD };

/* The text Ocarina of Time's F3DZEX loads (928 declines in 1500 frames before this build; the banner in its data says "Gfx[Safe] S2DEX fifo 2.05", the survey called it 2.06): the 2.06 text with a G_RDPPIPESYNC written in front of the flush
 * check (three instructions at 0x644, and a nop at 0x6cc, so the code behind 0x650 sits 0xc, behind 0x6cc 0x10 higher than in Kirby 64's text) and output buffers of 792 bytes instead of 800 (the threshold, the toggled pointer, the byte count
 * of the flush; `s6` starts at 0x7d8: the interpreter's initialisation). Every handler that writes to the output buffer ends in the flush routine at 0x640 / 0x644, so a sync follows each of them (f2_layout.pipe_sync, after_emit).
 * docs/ucode-s2dex2.md section 7. */
static const struct f2_layout l_lay_s2_safe = {
    .jump_table = 0x370, .mw_max = 14, .geom_def = F2_GEOM_DEF, .in = 0x418, .in_end = 0x4c0, .moveword_adj = 10374, .out_bytes = MUT(302, 792, 800), .out_toggle = 0xc00, .pipe_sync = MUT(303, 1, 0),
    .desc_clip = 0x314, .desc_light = 0x31c };
static const uint16_t l_ext_s2o[][2] = {
    { 0x024, K_S2_SELDL }, { 0x550, K_S2_LOADTXTR }, { 0xb84, K_S2_BG1CYC }, { 0x208, K_S2_BGCOPY }, { 0x0e8, K_S2_RENDERMODE }, { 0xb90, K_S2_RECT }, { 0x6d0, K_S2_SPRITE },
    { 0x0a4, K_S2_MOVEMEM }, { 0x1d0, K_S2_SCISSOR }, { 0x128, K_S2_HALF2 }, { 0xffff, 0 } };
static const struct f3dex2_build l_b_s2_oot = {
    .head = 0x1a4, .copy = 0x140, .sync = 0x144, .segcopy = 0x150, .ef = 0x170, .ed = MUT(248, 0xffff, 0xfff), .tri1 = 0, .tri2 = 0, .vtx = 0,
    .mtx = 0, .geom = 0, .enddl = 0x07c, .half2 = MUT(248, 0xffff, 0xfff), .modvtx = 0, .culldl = 0, .branchz = 0, .reject_mask = 0, .tri4 = 0, .ldtx_spr = 0x6f0, .ldtx_rect = 0xbb0, .lay = &l_lay_s2_safe, .pcs = &l_pcs_s2, .ext = l_ext_s2o, .ext_cmd = s2_ext_cmd, .load_key = 0xaf535d1683a57fcbull };
/* (never a task's own microcode: the id is the key of the loaded text) */
HLE_UCODE(l_s2_oot) = { .id = 0xaf535d1683a57fcbull, .name = "S2DEX fifo 2.05 [Safe] (Ocarina of Time, loaded by G_LOAD_UCODE)", .fam = &f3dex2_fam, .fam_data = &l_b_s2_oot, S2DEAD };

HLE_UCODE(l_s2_208) = { .id = 0xfa171b4489d45ef2ull, .name = "S2DEX fifo 2.08 (Worms Armageddon)", .fam = &f3dex2_fam, .fam_data = &l_b_s2_208, S2DEAD };

/* S2DEX xbus 2.08 (StarCraft 64's, docs/ucode-s2dex2.md section 8): Worms' 2.08 with the XBUS transport: the RDP list goes through a ring in DMEM (0x4c0 .. 0xee0, the commands may run over by 0xb0; there is no
 * output buffer, so no out_toggle / out_bytes) that every call of the flush routine (0x648: DPC_END = the output pointer, the wrap, the wait while the RDP is within 224 bytes ahead) hands to the RDP. The init is
 * shorter, so the handlers sit 0x28 lower from 0x1ac on and the continuation pcs of G_OBJ_LOADTXTR with them (ldtx_spr / ldtx_rect); the overlays, constants and DMEM state are the 2.08's. */
static const struct f2_layout l_lay_s2_xbus = {
    .jump_table = 0x370, .mw_max = 14, .geom_def = F2_GEOM_DEF, .in = 0x418, .in_end = 0x4c0, .moveword_adj = 10374, .desc_clip = 0x314, .desc_light = 0x31c,
    .xbus = 1, .ring_start = 0x4c0, .ring_end = 0xee0, .ring_gap = 224 };
static const uint16_t l_ext_s2x[][2] = {
    { 0x024, K_S2_SELDL }, { 0x558, K_S2_LOADTXTR }, { 0xb54, K_S2_BG1CYC }, { 0x210, K_S2_BGCOPY }, { 0x0e8, K_S2_RENDERMODE }, { 0xb60, K_S2_RECT }, { 0x6a0, K_S2_SPRITE },
    { 0x0a4, K_S2_MOVEMEM }, { 0x1d8, K_S2_SCISSOR }, { 0x128, K_S2_HALF2 }, { 0xffff, 0 } };
static const struct f3dex2_build l_b_s2_xbus = {
    .head = 0x1ac, .copy = 0x140, .sync = 0x144, .segcopy = 0x150, .ef = 0x170, .ed = MUT(248, 0xffff, 0xfff), .tri1 = 0, .tri2 = 0, .vtx = 0,
    .mtx = 0, .geom = 0, .enddl = 0x07c, .half2 = MUT(248, 0xffff, 0xfff), .modvtx = 0, .culldl = 0, .branchz = 0, .reject_mask = 0, .tri4 = 0, .ldtx_spr = 0x6c0, .ldtx_rect = 0xb80, .lay = &l_lay_s2_xbus, .pcs = &l_pcs_s2, .ext = l_ext_s2x, .ext_cmd = s2_ext_cmd, .load_key = 0xe51b62cd9879aeccull };
/* dead at the end of a task: the display list input buffer and the ring with its slack (0x418 .. 0xf90) */
static const struct hle_dead_range l_dead_s2x[] = { { 0x418, 0xf90 }, { 0xfc8, 0xfcc } };
HLE_UCODE(l_s2_xbus) = { .id = 0x3ee636569037de5bull, .name = "S2DEX xbus 2.08 (StarCraft 64)", .fam = &f3dex2_fam, .flags = HLE_F_XBUS_OK, .fam_data = &l_b_s2_xbus, .dead = l_dead_s2x, .n_dead = 2 };

#endif
