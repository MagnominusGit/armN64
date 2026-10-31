/*
 * arMLEt - hle_f3dex2_dialects.c: further builds of the F3DEX2 family and its dialects, as DATA for the front-end of hle_f3dex2.c
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <stdint.h>

#include "f3dexbg_dmem.h"
#include "hle_f3dex2_int.h"

#ifdef ARMLET_NATIVE

static const struct f3dex2_build l_b_ex2_208 = {
    .head = 0x194, .copy = 0x210, .sync = 0x214, .segcopy = 0x220, .ef = 0x240, .ed = 0x24c, .tri1 = 0x7bc, .tri2 = 0x7b4, .vtx = 0x4f8,
    .mtx = 0x0f0, .geom = 0x1d8, .enddl = 0x1ec, .half2 = 0x200, .modvtx = 0xc80, .culldl = 0xc40, .branchz = 0xc68, .reject_mask = 0x7070, .load_key = 0xcf7664aee1658511ull };

HLE_UCODE(l_ex2_d2d1) = { .id = 0xd2d1f9f2e7dfe1abull, .name = "F3DEX 2.08 (WinBack, Mario Golf, Hydro Thunder, Dr. Mario 64, RR64 ...)", .fam = &f3dex2_fam, .flags = HLE_F_NEAR_CLIP, .fam_data = &l_b_ex2_208, F3DEX2_DEAD };

/* F3DEX2 2.04 (Buck Bumble, Cruis'n World) and 2.04H (Kirby 64, Super Smash Bros.) */
static const struct f3dex2_build l_b_ex2_204 = {
    .head = 0x18c, .copy = 0x208, .sync = 0x20c, .segcopy = 0x218, .ef = 0x238, .ed = 0x244, .tri1 = 0x794, .tri2 = 0x78c, .vtx = 0x4e8,
    .mtx = 0x0f0, .geom = 0x1d0, .enddl = 0x1e4, .half2 = 0x1f8, .modvtx = 0xc54, .culldl = 0xc14, .branchz = 0xc3c, .zex = 1, .gen204 = 1, .clip204 = 1, .reject_mask = 0x7070, .load_key = 0xc9eb7be30bec9b96ull };
static const struct f3dex2_build l_b_ex2_204h = {
    .head = 0x18c, .copy = 0x220, .sync = 0x224, .segcopy = 0x230, .ef = 0x250, .ed = 0x25c, .tri1 = 0x7ac, .tri2 = 0x7a4, .vtx = 0x500,
    .mtx = 0x0f0, .geom = 0x1e8, .enddl = 0x1fc, .half2 = 0x210, .modvtx = 0xc6c, .culldl = 0xc2c, .branchz = 0xc54, .mvp = 0x1b8, .zex = 1, .gen204 = 1, .clip204 = 1, .reject_mask = 0x7070, .load_key = 0xb5044ba83c39d3afull };
/* Cruis'n World's RDP ring (RDRAM 0x353380 ..) overlaps the last 0x120 bytes of the ucode data (0x352ca0 + 0x800): the lists a task writes there come back as data at the start of the next task,
 * into DMEM [0x6e0, 0x800), which is vertex slots 17 .. 22 (written before they are read). The lists differ from the JIT's in the stale second words of the sync commands (docs/design.md, "The RDP output
 * ring is a mask in RDRAM"), so those bytes of the next task's DMEM differ too; dead like the rest of the vertex area at the end of a task. */
static const struct hle_dead_range l_dead_204[] = { F3DEX2_DEAD_RANGES, { 0x6e0, 0x800 } };
HLE_UCODE(l_ex2_5dce) = { .id = 0x6635562b2dde31aeull, .name = "F3DEX 2.04 (Buck Bumble, Cruis'n World)", .fam = &f3dex2_fam, .flags = HLE_F_NEAR_CLIP, .fam_data = &l_b_ex2_204,
    .ring_off = 0xfe8, .dead = l_dead_204, .n_dead = F3DEX2_N_DEAD_OUT + 1 };
/* Kirby 64's tasks load S2DEX in the middle (G_LOAD_UCODE, docs/ucode-f3dex2-dialects.md section 5.6) and end with whatever DMEM the last build left: the S2DEX display list buffer and RDP output buffers are dead too
 * (the ranges of hle_s2dex2.c). The equivalence shim masks by the ucode a task starts with, so they are in this entry; Super Smash Bros., which shares the text, has them dead as well (the data segment of the next task
 * covers them, as in every build). */
static const struct hle_dead_range l_dead_204h[] = { F3DEX2_DEAD_RANGES, { 0x418, 0x890 }, { 0x8c0, 0xc90 } };
HLE_UCODE(l_ex2_e857) = { .id = 0xe857446754069ab5ull, .name = "F3DEX 2.04H (Kirby 64, Super Smash Bros.)", .fam = &f3dex2_fam, .flags = HLE_F_NEAR_CLIP, .fam_data = &l_b_ex2_204h,
    .ring_off = 0xfe8, .dead = l_dead_204h, .n_dead = F3DEX2_N_DEAD_OUT + 2 };

/* F3DEXBG.NoN 2.08 (Conker's Bad Fur Day): the F3DEX2 code with the state moved (the structures between 0x1d0 and 0x2e0 by 0x60 / 0x68, everything above by 0x260 / 0x298), a 16-byte output buffer
 * threshold (two buffers 320 bytes apart), a reject mask with the w lane instead of z, and G_TRI4 (opcodes 0x10 .. 0x1f). */

/* the lit G_VTX of F3DEXBG: the standard vertex pipeline with the build's own lighting (f3dexbg_fx.h) */
static int bg_lit_vtx(struct f3 *f)
{
    struct hle_task *t = f->t;
    uint32_t va = seg_addr(t, f->w1);
    const char *why;
    unsigned n = (f->w0 >> 12) & 0xff;
    if (f->ov == OV_UNKNOWN)
        return hle_decline(t, "unknown overlay");
    if (f->ov == OV_CLIP)                               /* the code at 0x2d8 (the clipping overlay's) loads the lighting overlay */
        f3_overlay_load(f, OV_LIGHT);
    if (f->lean) {
        if (va & 7)
            return hle_decline(t, "unaligned vertices");
    } else if (n >= 1 && n <= 32) {
        unsigned endslot = hle_dm16(t, F2_VTAB + (f->w0 & 0xfff));
        hle_dma_read(t, endslot - n * 16, va, n * 16 - 1);
    }
    why = fbg_cmd_vtx(t, f->w0, va, f->lean, f->k0 - F2_IN_REFILL);
    return why ? hle_decline(t, why) : HLE_OK;
}

static const struct f2_layout l_lay_bg = {
    .mv = 0x00, .p = 0x40, .mvp = 0x80, .persp_norm = 0xdc, .viewport_scale = 0xe0, .viewport_trans = 0xe8, .clip_ratio = 0x186, .const31 = 0x210, .const30 = 0x220,
    .fog_mul = 0x248, .fog_ofs = 0x24a, .texture = 0x24c, .tex_scale_s = 0x250, .tex_scale_t = 0x252, .geom = 0x254, .mvp_valid = 0x239, .lights_valid = 0x23c, .lights_size = 0x23f,
    .light_raw = 0x268, .light_base = 0x278, .tg = 0x230, .light_scale = 0x226, .light_scale_zex = 0x22e, .vtab = 0x618, .clipmask = 0x664, .culltab = 0x65a, .in = 0xbb8, .in_end = 0xc60,
    .jump_table = 0x5d6, .mw_max = 16, .geom_def = F2_GEOM_DEF | 0x400000, .movemem_tab = 0x550, .moveword_adj = 10848, .desc_light = 0x6a8, .desc_clip = 0x6b0, .clip_planes = 0x180, .clip_masks = 0x690, .out_bytes = 16, .out_toggle = 0x140, .no_lit = 1, .lit_vtx = bg_lit_vtx };
static const struct f3dex2_build l_b_bg = {
    .head = 0x194, .copy = 0x210, .sync = 0x214, .segcopy = 0x220, .ef = 0x240, .ed = 0x24c, .tri1 = 0xa70, .tri2 = 0xa68, .vtx = 0x6d8,
    .mtx = 0x0f0, .geom = 0x1d8, .enddl = 0x1ec, .half2 = 0x200, .modvtx = 0xf34, .culldl = 0xef4, .branchz = 0xf1c, .reject_mask = 0x70b0, .tri4 = 0x994, .lay = &l_lay_bg, .load_key = 0xe1ba4ab13432a1dfull };
/* dead at the end of a task, the standard table moved with the layout: the input buffer 0xbb8 .. 0xc60, the clipper's new vertices 0xc60 .. 0xe40, the two output buffers (16 bytes each, 0xe40 and 0xf00, with 0xb0 of
 * slack: up to 0xfb8), the clipper's polygon lists (0x662 .. 0x6a0), the OSTask word the initialisation stores, and the pad bytes of the light slots (13 slots of 48 bytes from 0x278: the lighting code stores whole vector lanes, the pad lane holds what the vector registers had) */
static const struct hle_dead_range l_dead_bg[] = { { 0x662, 0x6a0 }, { 0xbb8, 0xfb8 }, { 0xfc8, 0xfcc },
    { 0x27b, 0x27c }, { 0x27f, 0x280 }, { 0x2ab, 0x2ac }, { 0x2af, 0x2b0 }, { 0x2db, 0x2dc }, { 0x2df, 0x2e0 }, { 0x30b, 0x30c }, { 0x30f, 0x310 }, { 0x33b, 0x33c }, { 0x33f, 0x340 }, { 0x36b, 0x36c }, { 0x36f, 0x370 }, { 0x39b, 0x39c }, { 0x39f, 0x3a0 }, { 0x3cb, 0x3cc }, { 0x3cf, 0x3d0 }, { 0x3fb, 0x3fc }, { 0x3ff, 0x400 }, { 0x42b, 0x42c }, { 0x42f, 0x430 }, { 0x45b, 0x45c }, { 0x45f, 0x460 }, { 0x48b, 0x48c }, { 0x48f, 0x490 }, { 0x4bb, 0x4bc }, { 0x4bf, 0x4c0 } };
HLE_UCODE(l_bg) = { .id = 0xe48f5dc9bed2fe3dull, .name = "F3DEXBG.NoN 2.08 (Conker's Bad Fur Day)", .fam = &f3dex2_fam, .fam_data = &l_b_bg, .ring_off = 0xfe8, .dead = l_dead_bg, .n_dead = sizeof(l_dead_bg) / sizeof(l_dead_bg[0]) };

/* Builds derived offline from their texts (tools/derive_build.py, docs/ucode-derive.md; the jump tables and the load keys re-read from the dumps) and verified with ARMLET_HLE_VERIFY on the games that run them:
 * F3DEX.NoN 2.08 (the registered 2.08 with the NoN reject mask), F3DEX 2.06 (the pcs of OoT's F3DZEX.NoN 2.06H, the near-clipping reject mask, zex, but G_BRANCH_Z compares the screen z: bz_z; found by the synthetic `misc` case, 149 of 1000 seeds), F3DEX.NoN 2.07 (Donkey Kong 64's pcs, NoN reject mask).
 * Rocket's F3DEX 2.07 is Donkey Kong 64's text: the same id since the id covers the whole 4 KiB. */
static const struct f3dex2_build l_b_ex2_208n = {
    .head = 0x194, .copy = 0x210, .sync = 0x214, .segcopy = 0x220, .ef = 0x240, .ed = 0x24c, .tri1 = 0x7bc, .tri2 = 0x7b4, .vtx = 0x4f8,
    .mtx = 0x0f0, .geom = 0x1d8, .enddl = 0x1ec, .half2 = 0x200, .modvtx = 0xc80, .culldl = 0xc40, .branchz = 0xc68, .reject_mask = MUT(318, 0x70b0, 0x7070), .load_key = 0x6f36ef1549b31211ull };
HLE_UCODE(l_ex2_9875) = { .id = 0x9875370e17e420abull, .name = "F3DEX.NoN 2.08 (Banjo-Tooie, LEGO Racers, Star Wars Episode I Racer)", .fam = &f3dex2_fam, .fam_data = &l_b_ex2_208n, F3DEX2_DEAD };
static const struct f3dex2_build l_b_ex2_206 = {
    .head = 0x18c, .copy = 0x208, .sync = 0x20c, .segcopy = 0x218, .ef = 0x238, .ed = 0x244, .tri1 = 0x7b4, .tri2 = 0x7ac, .vtx = 0x4f0,
    .mtx = 0x0f0, .geom = 0x1d0, .enddl = 0x1e4, .half2 = 0x1f8, .modvtx = 0xc74, .culldl = 0xc34, .branchz = 0xc5c, .zex = 1, .bz_z = 1, .reject_mask = MUT(320, 0x7070, 0x70b0), .load_key = 0x549ebb163e63b994ull };
HLE_UCODE(l_ex2_0bbe) = { .id = 0x0bbec7c325e9cca8ull, .name = "F3DEX 2.06 (Blues Brothers 2000, Harvest Moon 64, Gauntlet Legends)", .fam = &f3dex2_fam, .flags = HLE_F_NEAR_CLIP, .fam_data = &l_b_ex2_206, F3DEX2_DEAD };
static const struct f3dex2_build l_b_ex2_207n = {
    .head = 0x18c, .copy = 0x208, .sync = 0x20c, .segcopy = 0x218, .ef = 0x238, .ed = 0x244, .tri1 = 0x7b4, .tri2 = 0x7ac, .vtx = 0x4f0,
    .mtx = 0x0f0, .geom = 0x1d0, .enddl = 0x1e4, .half2 = 0x1f8, .modvtx = 0xc78, .culldl = 0xc38, .branchz = 0xc60, .reject_mask = MUT(319, 0x70b0, 0x7070), .load_key = 0xa8d17cd78a5b4e14ull };
HLE_UCODE(l_ex2_d8bd) = { .id = 0xd8bde2727be1cf28ull, .name = "F3DEX.NoN 2.07 (Tonic Trouble)", .fam = &f3dex2_fam, .fam_data = &l_b_ex2_207n, F3DEX2_DEAD };
/* F3DZEX.NoN fifo 2.08I (Majora's Mask): Paper Mario's 2.08 text plus four words: the init `vor`, the NoN reject mask (twice) and the G_BRANCH_Z source `lh 6(vertex)` with zex = 0 (bz_w) */
static const struct f3dex2_build l_b_mm = {
    .head = 0x194, .copy = 0x210, .sync = 0x214, .segcopy = 0x220, .ef = 0x240, .ed = 0x24c, .tri1 = 0xa44, .tri2 = 0xa3c, .vtx = 0x780,
    .mtx = 0x0f0, .geom = 0x1d8, .enddl = 0x1ec, .half2 = 0x200, .modvtx = 0xf08, .culldl = 0xec8, .branchz = 0xef0, .bz_w = 1, .reject_mask = 0x70b0, .load_key = 0xc9efea08ae04999bull };
HLE_UCODE(l_mm) = { .id = 0x1c724a38139c0669ull, .name = "F3DZEX.NoN fifo 2.08I (Majora's Mask)", .fam = &f3dex2_fam, .fam_data = &l_b_mm, F3DEX2_DEAD };

#endif
