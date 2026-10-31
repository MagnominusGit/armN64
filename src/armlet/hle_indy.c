/*
 * arMLEt - hle_indy.c: the microcode of Indiana Jones and the Infernal Machine (Factor 5) on the task-level native path: the commands as plain C
 *
 * Fast3D's descendants again (docs/ucode-indy.md): the vertex record, the clipper and the triangle setup are F3D's (f3d_fx.h), what is its own is the data flow: the display list is a chain of
 * 0x108-byte chunks (a link word and 32 commands), the RDP commands go to a buffer in DMEM that is flushed to a FIFO in RDRAM, vertices come from a block loader that reads a compressed stream
 * of chunks, and the rest of the microcode (the clipper, the lighting, the indexed vertex fetch, the exit code) lives in code overlays that the microcode pages into IMEM by number.
 *
 * run(): the boot code and the initialisation (OS code and the overlay manager's first load) run on the interpreter up to the command loop; then a C loop takes the display list, keeps the
 * microcode's state in DMEM where the microcode keeps it, and does what each command does. The outermost end of the list goes back to the interpreter (the exit code). A command or an overlay
 * the loop does not know declines the task (the JIT runs it).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "f3d_dmem.h"
#include "f3d_fx.h"
#include "hle.h"
#include "hle_fx.h"
#include "hle_internal.h"

#ifdef ARMLET_NATIVE

#ifdef ARMLET_HLE_MUTATE
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#define MUT(n, ok, bad) (ok)
#endif

/* registers by their names in the microcode */
enum { R_AT = 1, R_V0 = 2, R_V1 = 3, R_A0 = 4, R_A1 = 5, R_A2 = 6, R_A3 = 7, R_T0 = 8, R_T1 = 9, R_T2 = 10, R_T3 = 11, R_T4 = 12, R_T5 = 13, R_T6 = 14, R_T7 = 15,
       R_S0 = 16, R_S1 = 17, R_S2 = 18, R_S3 = 19, R_S4 = 20, R_S5 = 21, R_S6 = 22, R_S7 = 23, R_T8 = 24, R_T9 = 25, R_S8 = 30, R_RA = 31 };

/* DMEM of the build (HEAD: the loop head is the instruction after the word fetch at 0x2c, which is the delay slot of the poll at 0x28 on some paths) */
enum { HEAD = 0x30, CMD_LO = 0x270, CMD_HI = 0x378, OUT_LO = 0xe80, OUT_FLUSH = 0xf70, PB = 0x110,
       NEXT_CHUNK = 0x5a0, THIS_CHUNK = 0x5a4, DEFERRED = 0x58c, STACK = 0xfe0, DEPTH = 0x142, SIG0 = 0x80, POLL_LIMIT = 30000 };

struct indy {
    struct hle_task *t;
    struct hle_cpu *c;
    int lean;
    int trace, hybrid, lockstep, ls_done, sub;
    unsigned ls_n0;
    int ovl_dev;
    uint32_t v0_hi;
    unsigned ncmd;
    uint32_t w0, w1;
    unsigned long fallbacks[256];
};

/* ---- DMEM ------------------------------------------------------------------------------------------------------------------------------------------------------------------ */
static inline uint32_t ld32u(const struct hle_task *t, unsigned a)
{
    if (!(a & 3))
        return hle_dm32(t, a);
    return (uint32_t)hle_dm8(t, a) << 24 | (uint32_t)hle_dm8(t, a + 1) << 16 | (uint32_t)hle_dm8(t, a + 2) << 8 | hle_dm8(t, a + 3);
}
static inline void st32u(struct hle_task *t, unsigned a, uint32_t v)
{
    if (!(a & 3)) {
        hle_st32(t, a, v);
        return;
    }
    hle_st8(t, a, (uint8_t)(v >> 24)); hle_st8(t, a + 1, (uint8_t)(v >> 16)); hle_st8(t, a + 2, (uint8_t)(v >> 8)); hle_st8(t, a + 3, (uint8_t)v);
}

#define REG(f, n) ((f)->c->r[n])

/* ---- the display list ------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* The list is a chain of chunks of 0x108 bytes, read into 0x270 .. 0x378: a link word (the RDRAM address of the next chunk) and 32 commands. s1 points at the next command; the chunk the buffer
 * holds is at DMEM[0x5a4], the link of the one after it at DMEM[0x5a0]. The lean mode reads the commands from RDRAM. */
static inline uint32_t chunk_word(struct indy *f, unsigned off)         /* the word at byte off of the command buffer */
{
    struct hle_task *t = f->t;
    if (!f->lean)
        return hle_dm32(t, CMD_LO + off);
    return hle_rd32(t, hle_dm32(t, THIS_CHUNK) + off);
}

/* ---- the output buffer ------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* The RDP commands are collected at DMEM 0xe80 .. and flushed to the FIFO in RDRAM (pc 0xdc): when the buffer holds more than 0xf70 - 0xe80 bytes after a command, and at the end of the task. The
 * FIFO bounds are in the pointer block (DMEM[0x110]: start, [0x114]: end, [0x118]: the write pointer); a chunk that does not fit before the end wraps to the start after the microcode has waited for
 * the RDP; with the video plugin synchronous nothing waits, a state in which it would wait for ever declines. */
static int flush(struct indy *f)
{
    struct hle_task *t = f->t;
    uint32_t s0 = REG(f, R_S0), cur = hle_dm32(t, PB + 8), n = s0 - OUT_LO, end = hle_dm32(t, PB + 4), cr;
    if (!n)
        return HLE_OK;                                  /* (0xe4: the buffer is empty) */
    if ((int32_t)(end - (cur + n)) < MUT(173, 0, 1)) {
        if (*t->cr[11] & 0x400)
            return hle_decline(t, "FIFO: DPC start valid while wrapping");
        cur = hle_dm32(t, PB);
        if (*t->cr[10] == cur)
            return hle_decline(t, "FIFO: the RDP stands at the start while wrapping");
        hle_mtc0(t, 8, cur);
    }
    cr = *t->cr[10];
    if ((int32_t)(cur - cr) < 0 && (int32_t)(cur + n - cr) >= 0)
        return hle_decline(t, "FIFO: the RDP is in the way");
    if (f->lean && !(cur & 7) && !(n & 7))
        hle_rdram_write(t, cur, &t->dmem[OUT_LO >> 2], n);
    else
        hle_dma_write(t, OUT_LO, cur, n - 1);
    hle_st32(t, PB + 8, cur + n);
    hle_mtc0(t, 9, cur + n);
    REG(f, R_S0) = OUT_LO;
    return HLE_OK;
}

/* ---- the commands --------------------------------------------------------------------------------------------------------------------------------------------------------- */
enum { K_FALLBACK = 0, K_NOP, K_POLL, K_RDP, K_REFILL, K_GCLR, K_GSET, K_TEXTURE, K_MOVEWORD, K_OTHER_L, K_OTHER_H, K_DL, K_BRANCH, K_ENDDL, K_LOAD, K_STREAM, K_TRI1, K_TRI2 };
enum { RET_DEFERRED = 1000, MAX_SUB = 3, RA_TEXRECT = 0xbc };

/* the table at DMEM 0xb6 (class 0) and, for the commands 0xa9 .. 0xdf of classes 1 and 2, the same entries from index op - 0xa9 */
static unsigned classify(unsigned pc)
{
    switch (pc) {
    case 0x020: return K_POLL;
    case 0x060: return K_NOP;                           /* (the entries that lead straight back to the loop) */
    case 0x068: return K_REFILL;
    case 0xa6c: return K_GCLR;
    case 0xa64: return K_GSET;
    case 0xa7c: return K_TEXTURE;
    case 0xa74: return K_MOVEWORD;
    case 0x7cc: return K_OTHER_L;
    case 0x7dc: return K_OTHER_H;
    case 0x750: return K_DL;
    case 0x768: return K_BRANCH;
    case 0x774: return K_ENDDL;
    case 0xc90: return K_LOAD;
    case 0xd18: return K_STREAM;
    case 0xb20: return K_TRI1;
    case 0xb40: return K_TRI2;
    }
    return K_FALLBACK;
}

static int cmd_other(struct indy *f, int conditional)
{
    struct hle_task *t = f->t;
    uint32_t w0 = f->w0, w1 = f->w1, s0 = REG(f, R_S0), word;
    unsigned a0;
    if (conditional && (((w0 >> MUT(171, 23, 22)) & 1) != hle_dm32(t, PB + 12)))
        return HLE_OK;
    a0 = (w0 >> 16) & 7;
    word = ld32u(t, 0x120 + a0);
    {
        uint32_t m = MUT(170, (uint32_t)((int32_t)0x80000000 >> (w0 & 31)), 0x80000000u >> (w0 & 31));         /* srav */
        m = m >> ((w0 >> 8) & 31);                                          /* srlv */
        word = (word & ~m) | w1;
    }
    st32u(t, 0x120 + a0, word);
    REG(f, R_S3) = hle_dm32(t, PB + 16);
    REG(f, R_S4) = hle_dm32(t, PB + 20);
    /* the pair of mode words goes to the output buffer as one SET_OTHER_MODES command, in place of the last command if that was one */
    if (MUT(172, s0 != OUT_LO && hle_dm8(t, s0 - 8) == 0xef, 0)) {
        hle_st32(t, s0 - 8, REG(f, R_S3));
        hle_st32(t, s0 - 4, REG(f, R_S4));
        return HLE_OK;
    }
    hle_st32(t, s0, REG(f, R_S3));
    hle_st32(t, s0 + 4, REG(f, R_S4));
    REG(f, R_S0) = s0 + 8;
    if (REG(f, R_S0) > OUT_FLUSH)                       /* (pc 0xd0, as for any other command written to the buffer) */
        return flush(f);
    return HLE_OK;
}

/* ---- the display list stack, the chunk reloads and the deferred list ------------------------------------------------------------------------------------------------------ */
/* pc 0x70 (and the code it returns to at 0x10): the chunk at RDRAM address `addr` becomes the current one, s1 points at its first command, the link of the chunk after it is kept */
static int set_list(struct indy *f, uint32_t addr)
{
    struct hle_task *t = f->t;
    if (f->lean && (addr & 7))
        return hle_decline(t, "a display list chunk that is not 8-byte aligned");
    hle_st32(t, THIS_CHUNK, addr);
    REG(f, R_S1) = CMD_LO + 8;
    if (!f->lean)
        hle_dma_read(t, CMD_LO, addr, 263);
    hle_st32(t, NEXT_CHUNK, chunk_word(f, 0));
    return HLE_OK;
}

/* pc 0x750: the display list command pushes the chunk and the command pointer (already past this command) and goes on at the address in w1 */
static int cmd_dl(struct indy *f, uint32_t target)
{
    struct hle_task *t = f->t;
    unsigned depth = hle_dm8(t, DEPTH);
    if (depth > 0x18)
        return hle_decline(t, "the display list stack is deeper than DMEM allows");
    hle_st32(t, STACK + depth, hle_dm32(t, THIS_CHUNK));
    hle_st32(t, STACK + depth + 4, REG(f, R_S1) + MUT(174, 0, 8));
    hle_st8(t, DEPTH, (uint8_t)(depth + 8));
    return set_list(f, target);
}

static int run_loop(struct indy *f, int *done);

/* pc 0x7b0 (called with ra = `ra`): the registers are saved in DMEM; a list set in DMEM[0x58c] runs first (it ends with an end-of-list whose w1 is not zero and that comes back here, pc 0x798),
 * the output buffer is flushed (pc 0xdc) */
static int call_deferred(struct indy *f, unsigned ra)
{
    struct hle_task *t = f->t;
    uint32_t v0 = hle_dm32(t, DEFERRED);
    int rc, done = 0;
    hle_st16(t, 0x590, (uint16_t)REG(f, R_S8));
    hle_st16(t, 0x592, (uint16_t)ra);
    hle_st16(t, 0x594, (uint16_t)REG(f, R_T3));
    if (!v0)
        return flush(f);
    if (f->sub >= MAX_SUB)
        return hle_decline(t, "deferred lists nest too deep");
    REG(f, R_S4) = v0;
    REG(f, R_RA) = ra;
    if ((rc = cmd_dl(f, v0)) != HLE_OK)
        return rc;
    f->sub++;
    rc = run_loop(f, &done);
    f->sub--;
    if (rc != RET_DEFERRED)
        return rc != HLE_OK ? rc : hle_decline(t, "the task ended inside a deferred list");
    if ((REG(f, R_RA) & 0xfff) != ra)
        return hle_decline(t, "a deferred list returns to another place");
    return HLE_OK;
}

/* pc 0x774: the end of a list: the stack is popped; with an empty stack the exit overlay (the caller hands over); an end whose w1 is not zero comes back from a deferred list */
static int cmd_end(struct indy *f, int *done)
{
    struct hle_task *t = f->t;
    int depth = (int)hle_dm8(t, DEPTH) - 8, rc;
    uint32_t s4 = f->w1, v0;
    if (depth >= 0 && s4 && !f->sub) {                  /* (a deferred list that was started by code the loop does not have: development only) */
        if (f->hybrid && !f->lean)
            return -1;
        return hle_decline(t, "an end of list that returns from a deferred list outside of one");
    }
    if (depth < 0) {
        REG(f, R_A1) = 30;                              /* (0x780: the exit overlay) */
        *done = 1;
        return HLE_OK;
    }
    hle_st8(t, DEPTH, (uint8_t)depth);
    v0 = hle_dm32(t, STACK + depth);
    if ((rc = set_list(f, v0)) != HLE_OK)
        return rc;
    REG(f, R_S1) = hle_dm32(t, STACK + depth + 4) + MUT(175, 0, 8);
    if (!s4)
        return HLE_OK;
    /* pc 0x798: back from the deferred list */
    REG(f, R_S8) = hle_dm16(t, 0x590);
    REG(f, R_RA) = hle_dm16(t, 0x592);
    REG(f, R_T3) = hle_dm16(t, 0x594);
    hle_st32(t, DEFERRED, 0);
    hle_st16(t, 0x590, (uint16_t)REG(f, R_S8));         /* (pc 0x7b0 again: the list is gone, the registers are saved again and the buffer is flushed) */
    hle_st16(t, 0x592, (uint16_t)REG(f, R_RA));
    hle_st16(t, 0x594, (uint16_t)REG(f, R_T3));
    if ((rc = flush(f)) != HLE_OK)
        return rc;
    return RET_DEFERRED;
}

/* a texture rectangle (0xe4, 0xe5; pc 0x98): the first word is stored, the deferred list and the flush come first, then the 16 bytes of the command go to the buffer */
static int cmd_texrect(struct indy *f, unsigned off)
{
    struct hle_task *t = f->t;
    uint32_t s0;
    unsigned i;
    int rc;
    if (off + 16 > CMD_HI - CMD_LO)
        return hle_decline(t, "a texture rectangle across the end of the command buffer");
    hle_st32(t, REG(f, R_S0), f->w0);
    REG(f, R_S1) += 8;
    if ((rc = call_deferred(f, RA_TEXRECT)) != HLE_OK)
        return rc;
    s0 = REG(f, R_S0);
    for (i = 0; i < 4; i++)
        hle_st32(t, s0 + 4 * i, chunk_word(f, off + 4 * i));
    REG(f, R_S0) = s0 + 16;
    if (REG(f, R_S0) > OUT_FLUSH)
        return flush(f);
    return HLE_OK;
}

/* ---- the overlay manager ---------------------------------------------------------------------------------------------------------------------------------------------- */
/* pc 0x164: overlay `n` is brought to IMEM unless it is the one that is there. The descriptor is three bytes at DMEM 0x6a + n: a halfword (index of the entry in the table at DMEM 100 in the top four bits,
 * the length in bytes - 1 below) and the next halfword (the RDRAM offset / 16 in the text). Returns the entry pc (the overlay's code is not run: the callers do what it does). */
static uint64_t hash_imem(const struct hle_task *t, unsigned lo, unsigned len)
{
    uint64_t h = 1469598103934665603ull;
    unsigned a;
    for (a = lo; a < lo + len; a++)
        h = (h ^ ((const uint8_t *)t->imem)[(a & 0xfff) ^ 3]) * 1099511628211ull;
    return h;
}

/* the images of the overlays that were seen (the hash of the bytes the manager transfers) */
static const struct { unsigned n; uint64_t hash; } known_ovl[] = {
    { 3, 0xad0f7aedf84d309dull },
    { 27, 0x50ec89682d4e16acull },
    { 42, 0xab09363dcbc9fd3bull },
    { 45, 0x7f894a46f34e7835ull },
};

static int overlay(struct indy *f, unsigned n)
{
    struct hle_task *t = f->t;
    unsigned d = (unsigned)hle_dm8(t, 0x6a + n) << 8 | hle_dm8(t, 0x6b + n), d2 = (unsigned)hle_dm8(t, 0x6b + n) << 8 | hle_dm8(t, 0x6c + n), idx = d >> 12, len = (d & 4092) - 1, imem, i;
    uint32_t src;
    if (f->trace)
        fprintf(stderr, "OVERLAY %u (loaded %u)\n", n, hle_dm8(t, 0x143));
    hle_st16(t, 0x16a, (uint16_t)REG(f, R_RA));
    if (hle_dm8(t, 0x143) != n) {
        hle_st8(t, 0x143, (uint8_t)n);
        src = hle_dm32(t, 0xfd0) + ((d2 << 4) & 16368);
        imem = (unsigned)hle_dm8(t, 100 + idx) << 8 | hle_dm8(t, 101 + idx);
        hle_dma_read(t, imem, src, len);
        if (f->ovl_dev) {
            fprintf(stderr, "OVERLAY_HASH %u %016llx len %u\n", n, (unsigned long long)hash_imem(t, imem, len + 1), len + 1);
        }
        for (i = 0; i < sizeof(known_ovl) / sizeof(known_ovl[0]); i++)
            if (known_ovl[i].n == n && known_ovl[i].hash == hash_imem(t, imem, len + 1))
                break;
        if (i == sizeof(known_ovl) / sizeof(known_ovl[0]) && !f->ovl_dev)
            return hle_decline(t, "an overlay image that was not seen");
    }
    return HLE_OK;
}

/* ---- vertices ------------------------------------------------------------------------------------------------------------------------------------------------------------- */
enum { POS = 0x170, VREC = 0x600, VSTRIDE = 40, MTX = 0x5c0, VOFF = 0x128, GEOM_FOG = 0x10000 };

/* the state the vertex transform and the clipper's new vertices are completed with */
static void indy_vctx(struct indy *f, struct f3d_vctx *c)
{
    struct hle_task *t = f->t;
    unsigned k;
    memset(c, 0, sizeof(*c));
    f3d_load_mtx(&c->mp, t->dmem, MTX, 0);
    for (k = 0; k < 4; k++) {
        c->vscale[k] = (int16_t)hle_dm16(t, 0x130 + 2 * k);
        c->vtrans[k] = (int16_t)hle_dm16(t, 0x138 + 2 * k);
        c->vsign[k] = MUT(201, 1, k == 1 ? -1 : 1);
    }
    c->lim[0] = c->lim[2] = MUT(200, 32767, (int16_t)hle_dm16(t, 0x50));                      /* (only the lower limits are applied: x, y at 0x52, z at 0x56; there is no VLT after the VGE) */
    c->lim[1] = (int16_t)hle_dm16(t, 0x52);
    c->lim[3] = (int16_t)hle_dm16(t, 0x56);
    c->persp = hle_dm16(t, 0x14e);
    c->clip_k = (int16_t)hle_dm16(t, 6);
    c->fog_wide = 1;
    for (k = 0; k < 5; k++)
        c->fogw[k] = (int16_t)hle_dm16(t, 0x160 + 2 * k);
    c->flags_variant = F3D_FLAGS_SM64;
    c->clip_acc = 1;                                    /* (the accumulator lane that the weight starts from holds 1 here) */
}

/* The vertex transform (pc 0x834 .. 0x9e4) of n vertices (at least one: the loop body runs once whatever n is) from the 8-byte positions at DMEM 0x170 to the records at 0x600 (40 bytes each). The
 * microcode takes them in pairs; an odd count transforms one more vertex (the next position in the buffer) of which it stores the fog value and keeps the clip flags in t3.
 * Per vertex: p = (x, y, z) + the offset vector (DMEM 0x128; 16-bit sums that wrap), the clip coordinates (x, y, z, 1) M with M the matrix at 0x5c0 (s15.16), the clip flags against W and K W, 1/W
 * normalised, the screen position, and with fog (geometry mode bit 16) the fog value in the alpha byte. These are f3d_vtx_xform's with the Rare offset moved before the multiplication and Indy's fog. */
static int cmd_vertices(struct indy *f, unsigned n)
{
    struct hle_task *t = f->t;
    unsigned cnt = n ? n : 1, tot = MUT(180, (cnt + 1) & ~1u, cnt), i, k;
    int fog = (REG(f, R_S5) & GEOM_FOG) != 0;
    struct f3d_vctx c;
    struct f3d_vin in[32];
    struct f3d_vtx out[32];
    if (cnt > 32)
        return hle_decline(t, "more vertices than the position buffer holds");
    memset(in, 0, sizeof(in));
    indy_vctx(f, &c);
    c.geom = fog ? F3D_G_FOG : 0;
    for (i = 0; i < tot; i++) {
        in[i].x = (int16_t)(hle_dm16(t, POS + 8 * i) + hle_dm16(t, VOFF));
        in[i].y = (int16_t)(hle_dm16(t, POS + 8 * i + 2) + MUT(179, hle_dm16(t, VOFF + 2), 0));
        in[i].z = (int16_t)(hle_dm16(t, POS + 8 * i + 4) + hle_dm16(t, VOFF + 4));
    }
    f3d_vtx_xform(&c, in, tot, out);
    for (i = 0; i < tot; i++) {
        unsigned a = VREC + VSTRIDE * i;
        if (i < cnt) {
            for (k = 0; k < 4; k++) {
                hle_st16(t, a + 2 * k, (uint16_t)out[i].cx[k]);
                hle_st16(t, a + 8 + 2 * k, out[i].cf[k]);
            }
            hle_st16(t, a + 24, (uint16_t)out[i].sx);
            hle_st16(t, a + 26, (uint16_t)out[i].sy);
            hle_st16(t, a + 28, (uint16_t)out[i].sz);
            hle_st16(t, a + 30, out[i].szf);
            hle_st16(t, a + 32, (uint16_t)out[i].rwi);
            hle_st16(t, a + 34, out[i].rwf);
            hle_st16(t, a + 36, out[i].flags);
            if (MUT(182, i & 1, 0))                     /* (the pair's word store at +76: the first vertex's flags land in the second record's padding) */
                hle_st16(t, a + 38, out[i - 1].flags);
        }
        if (MUT(181, fog, 0))
            hle_st8(t, a + 19, out[i].col[3]);
    }
    REG(f, R_T3) = (out[tot - 2].flags & MUT(183, 0x7070u, 0x7071u)) | ((uint32_t)(out[tot - 1].flags & 0x7070u) << 16);
    REG(f, R_S8) = 0x1060;
    if (f->hybrid && !f->lean) {                        /* (development: the triangle code on the interpreter reads the viewport and the normalisation from the vector registers that the vertex code loads: pc 0x9ec) */
        struct hle_cpu *cp = f->c;
        uint32_t ra = cp->r[R_RA];
        cp->r[R_RA] = HEAD;
        cp->pc = 0x9ec;
        cp->stop_pc = HEAD;
        hle_rsp_run(t, cp);
        cp->r[R_RA] = ra;
    }
    return HLE_OK;
}

/* pc 0xd60 of overlay 45: the vertices are fetched by number: the halfwords at DMEM 0x170 + 2 i index the table at DMEM[0x154] (6 bytes per vertex: x, y, z), the 16 bytes around each are read
 * by DMA and the three halfwords become the position (the fourth halfword of the 8 bytes stays) */
static int fetch_indexed(struct indy *f, unsigned n)
{
    struct hle_task *t = f->t;
    uint32_t base = hle_dm32(t, 0x154);
    unsigned i;
    if (!n)
        return hle_decline(t, "an indexed vertex fetch of no vertices");
    if (n > 32)
        return hle_decline(t, "more vertices than the position buffer holds");
    for (i = n; i-- > 0;) {
        unsigned idx = hle_dm16(t, POS + 2 * i), v0;
        uint32_t a0 = (idx << 1) + (idx << MUT(198, 2, 1)) + base;
        v0 = a0 & 7;
        hle_dma_read(t, 0xf80, a0, 15);
        hle_st32(t, POS + 8 * i, ld32u(t, 0xf80 + v0));
        hle_st16(t, POS + 8 * i + 4, (uint16_t)((hle_dm8(t, 0xf84 + v0) << 8) | hle_dm8(t, 0xf85 + v0)));
    }
    return HLE_OK;
}

/* ---- colours (pc 0xbd0) ----------------------------------------------------------------------------------------------------------------------------------------------- */
enum { COLS = 0x480, CTAB = 0xd40, SCALE = 0x158, SCALE_ON = 0x58b };

/* Type 2 of the data loader. A colour table of t0 = (w1 >> 24) & 127 entries was loaded to DMEM 0x480. With bit 23 of w0 (s3 & 2 after the shift) the entries are 16-bit 5-6-5 colours that are
 * first expanded to 32 bits in place (r5 g6 b5 at the top of each byte, alpha 255). Then, when w1 bit 31 or the byte at DMEM 0x58b is negative, the table is scaled by the bytes at 0x158 (one factor per channel) and copied to
 * 0xd40 (the colour table that the triangles read), four entries at a time; otherwise the lighting overlay (27, or 33 with bit 22 of w0) computes them. */
static int cmd_lights(struct indy *f, uint32_t w1);

static int cmd_colours(struct indy *f)
{
    struct hle_task *t = f->t;
    uint32_t w0 = f->w0, w1 = f->w1;
    int32_t a3 = (int32_t)w1 >> 24;
    unsigned n = (unsigned)a3 & 127, sh = w0 >> 22, i;
    if (sh & 2) {
        if (!n)
            return hle_decline(t, "a colour table of no entries");
        for (i = n; i-- > 0;) {
            unsigned v = hle_dm16(t, COLS + 2 * i);
            hle_st32(t, COLS + 4 * i, ((v & 0xf800u) << 16) | ((v & 0x7e0u) << MUT(176, 13, 12)) | ((v & 31u) << 11) | 255u);
        }
    }
    a3 |= (int8_t)hle_dm8(t, SCALE_ON);
    if (a3 >= 0) {
        int rc = overlay(f, (w0 >> 22 & 1) ? 33 : 27);
        if (rc != HLE_OK)
            return rc;
        if ((w0 >> 22) & 1)
            return hle_decline(t, "overlay 33");
        return cmd_lights(f, w1);
    }
    {
        unsigned src = COLS, dst = CTAB;
        int cnt = (int)n;
        do {
            for (i = 0; i < 16; i++) {
                unsigned x = hle_dm8(t, src + i);
                hle_st8(t, dst + i, (uint8_t)((x * hle_dm8(t, SCALE + (i & 7)) + MUT(177, 1, 0)) >> 8));     /* (VMULF lane by lane: the eight bytes at 0x158 are the factors of two colours) */
            }
            cnt -= 4;
            src += 16;
            dst += 16;
        } while (cnt > 0);
    }
    return HLE_OK;
}

/* The lighting overlay (27; pc 0xd60 .. 0xf9c). The colour table at 0xd40 (4 bytes per vertex, r g b a) is cleared and then, for each of the DMEM[0x58b] lights (24 bytes at 0xb00 + 24 k: a direction or
 * position at +0, a second direction at +8, the attenuation at +20, the colour at +16 and the type at +19), every pair of table entries gets base * light added: base is the colour loaded to 0x480
 * scaled by the bytes at 0x158, light the colour of the light times
 *   type 0 (ambient): 1
 *   type 1 (directional): max(0, direction . normal), the normal the signed bytes at 0x380 + 4 vertex (the direction of the second vertex of a pair may be the light's second one: 2 bits per pair
 *                         of the words at 0x5b0 / 0x5b4)
 *   type 2 ..  (point): the attenuation 1 - d^2 k when 0 < d^2 k < 1 (as the microcode forms it) of the distance d between the vertex (the position at 0x170 + the byte at 0x380 + 4 vertex + 3) and the light
 * all in the 16-bit fixed point of the vector unit (bytes << 7, VMULF's rounding, saturating adds); the alpha is the scaled base alpha OR-ed in. */
static inline int16_t i16(int32_t v) { return hle_clamp16(v); }
static inline int16_t luv(uint8_t b) { return (int16_t)(b << 7); }
static inline int16_t vmulf(int16_t a, int16_t b) { return (int16_t)hle_sclamp(hle_mulf(a, b)); }
static inline int16_t vmulf_add(int16_t a, int16_t b, int16_t c) { return (int16_t)hle_sclamp(hle_wrap48(hle_mulf(a, b) + ((hle_acc)c << 16))); }

static int cmd_lights(struct indy *f, uint32_t w1)
{
    struct hle_task *t = f->t;
    unsigned n = (w1 >> 24) & 127, end = CTAB + (((4 * n + 4) & 0xfff8u)), nl = hle_dm8(t, SCALE_ON), l, p, i, a3, c;
    int16_t scl[8];
    if (end == CTAB)
        return hle_decline(t, "a lit colour table of no entries");
    if (nl > 24)
        return hle_decline(t, "more lights than fit");
    for (c = CTAB; c < end || c == CTAB; c += 16)
        for (i = 0; i < 16; i++)
            hle_st8(t, c + i, 0);
    REG(f, R_T3) = c;                                   /* (the register t3 ends as the clearing loop's pointer, with lights as the end of the table) */
    for (i = 0; i < 8; i++)
        scl[i] = luv(hle_dm8(t, SCALE + i));
    if (nl)
        REG(f, R_T3) = end;
    for (l = 0, a3 = 0xb00; l < nl; l++, a3 += 24) {
        unsigned type = hle_dm8(t, a3 + 19);
        int16_t lc[8], dir0[4], dir1[4];
        uint32_t t4w = 0, t5w = 0;
        for (i = 0; i < 3; i++)
            lc[i] = lc[4 + i] = luv(hle_dm8(t, a3 + 16 + i));
        lc[3] = lc[7] = 0;
        for (i = 0; i < 4; i++) {
            dir0[i] = (int16_t)hle_dm16(t, a3 + 2 * i);
            dir1[i] = (int16_t)hle_dm16(t, a3 + 8 + 2 * i);
        }
        if (type == 1) {
            t4w = hle_dm32(t, 0x5b0);
            t5w = hle_dm32(t, 0x5b4);
        }
        for (p = 0; (CTAB + 8 * p) < end; p++) {
            int16_t sc[8], old[8], v0[8], v1[8];
            unsigned tb = CTAB + 8 * p, bb = COLS + 8 * p;
            for (i = 0; i < 8; i++) {
                sc[i] = vmulf(luv(hle_dm8(t, bb + i)), scl[i]);
                old[i] = luv(hle_dm8(t, tb + i));
            }
            if (type == 0) {
                for (i = 0; i < 8; i++)
                    v1[i] = lc[i];
            } else if (type == 1) {
                int16_t dir[8], pr[8], s[8], dot[2];
                unsigned nb = 0x380 + 8 * p, field = 0;
                if (t4w | t5w)
                    field = p < 16 ? (t4w >> (2 * p)) & 3 : (t5w >> (2 * (p - 16))) & 3;
                for (i = 0; i < 8; i++) {
                    unsigned bit = (t4w | t5w) ? (((0xfff00f00u >> (8 * field)) & 0xff) >> i) & 1 : 0;
                    dir[i] = bit ? dir1[i & 3] : dir0[i & 3];
                    pr[i] = (i & 3) == 3 ? vmulf(dir[i], 0) : vmulf(dir[i], (int16_t)((int8_t)hle_dm8(t, nb + i) << 8));
                }
                for (i = 0; i < 8; i++)
                    s[i] = i16((int32_t)pr[i] + pr[i | 1]);
                dot[0] = i16((int32_t)s[0] + s[2]);
                dot[1] = i16((int32_t)s[4] + s[6]);
                for (i = 0; i < 2; i++)
                    if (dot[i] < 0 && MUT(178, 1, 0))
                        dot[i] = 0;
                for (i = 0; i < 8; i++)
                    v1[i] = vmulf(lc[i], dot[i >> 2]);
            } else {
                uint16_t at[2];
                for (i = 0; i < 2; i++) {
                    unsigned pos = 0x170 + hle_dm8(t, 0x380 + 8 * p + 3 + 4 * i);
                    int32_t sq[4], s1[4], s2;
                    struct hle_pair S;
                    unsigned k;
                    int32_t F = hle_fx((int16_t)hle_dm16(t, a3 + 20), hle_dm16(t, a3 + 22));
                    for (k = 0; k < 4; k++) {
                        int16_t pv = k == 3 ? 0 : (int16_t)hle_dm16(t, pos + 2 * k);
                        int16_t d = i16((int32_t)pv - dir0[k]);
                        sq[k] = (int32_t)(((int64_t)d * d));
                        if (sq[k] < 0)
                            sq[k] = 0x7fffffff;
                    }
                    for (k = 0; k < 4; k++) {
                        S = hle_add32(sq[k], sq[k | 1]);
                        s1[k] = hle_pair32(S);
                    }
                    S = hle_add32(s1[0], s1[2]);
                    s2 = hle_pair32(S);
                    {
                        uint16_t lo = hle_uclamp(hle_mac_32x32_before_h(s2, F));
                        int16_t hi = (int16_t)hle_sclamp(hle_mac_32x32(s2, F));
                        at[i] = hi == -1 ? lo : 0;
                    }
                }
                for (i = 0; i < 8; i++)
                    v1[i] = (int16_t)hle_sclamp(hle_wrap48((hle_acc)lc[i] * at[i >> 2]));
            }
            for (i = 0; i < 8; i++) {
                v0[i] = vmulf_add(sc[i], v1[i], old[i]);
                if ((i & 3) == 3)
                    v0[i] |= sc[i];
                hle_st8(t, tb + i, (uint8_t)(v0[i] >> 7));
            }
        }
    }
    hle_st32(t, 0x5b0, 0);
    hle_st32(t, 0x5b4, 0);
    return HLE_OK;
}

/* ---- the data loader ------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* The vertex data comes in a stream of chunks: the state is DMEM[0xfd4] (the RDRAM address of the next byte), [0xfd8] (the address of the chunk after the current one) and [0xfdc] (the bytes left
 * in the current one). A chunk is an 8-byte header (the link and a word that is not used) and its data, 256 bytes unless the 0x02 command that started the stream says otherwise. */
enum { ST_PTR = 0xfd4, ST_LINK = 0xfd8, ST_LEFT = 0xfdc };

/* pc 0xd28: the header of the chunk at `addr` is read, the data follows it */
static void stream_open(struct indy *f, uint32_t addr)
{
    struct hle_task *t = f->t;
    hle_dma_read(t, ST_LINK, addr, 0);
    hle_st32(t, ST_LEFT, 256);
    hle_st32(t, ST_PTR, addr + 8);
}

/* pc 0xd18: a new stream starts with the chunk at w1; the size of its data is in w0 */
static int cmd_stream(struct indy *f)
{
    stream_open(f, f->w1);
    hle_st32(f->t, ST_LEFT, f->w0 & 511);
    return HLE_OK;
}

/* pc 0xc90: w0 & 0xff + 1 bytes go to DMEM[(w0 >> 8) & 0xfff]: from the address in w1, or from the stream when the low 24 bits of w1 are zero. Then the type in bits 20 .. 21 of w0 says what to do
 * with them (0: nothing, 1: vertices (w1 >> 24 of them), 2: colours / lights, 3: the byte at 0x58b). */
static int cmd_load(struct indy *f)
{
    struct hle_task *t = f->t;
    uint32_t w0 = f->w0, w1 = f->w1;
    unsigned a0 = w0 & 255, dst = (w0 >> 8) & 4095, type = (w0 >> 20) & 3;
    if (a0) {
        if (w1 << 8) {
            hle_dma_read(t, dst, w1, a0);
        } else {
            for (;;) {
                uint32_t left = hle_dm32(t, ST_LEFT), ptr = hle_dm32(t, ST_PTR);
                int32_t rest = (int32_t)(left - (a0 + 1));
                if (rest >= 0) {
                    hle_dma_read(t, dst, ptr, a0);
                    hle_st32(t, ST_LEFT, (uint32_t)rest);
                    hle_st32(t, ST_PTR, ptr + a0 + 1);
                    if (!rest)
                        stream_open(f, hle_dm32(t, ST_LINK));
                    break;
                }
                if (!left)
                    return hle_decline(t, "the stream chunk is empty");
                hle_dma_read(t, dst, ptr, left - 1);
                dst += left;
                a0 = a0 - left + MUT(199, 0, 1);
                stream_open(f, hle_dm32(t, ST_LINK));
            }
        }
    }
    REG(f, R_S4) = w1;
    switch (type) {
    case 0: break;
    case 1:
        if (w0 >> 22 & 1) {                             /* (pc 0xbb4: overlay 45 fetches the positions by index first) */
            unsigned rc;
            REG(f, R_A1) = 45;
            if ((rc = overlay(f, 45)) != HLE_OK)
                return (int)rc;
            if ((rc = (unsigned)fetch_indexed(f, w1 >> 24)) != HLE_OK)
                return (int)rc;
        }
        return cmd_vertices(f, w1 >> 24);
    case 2: return cmd_colours(f);
    case 3: hle_st8(t, 0x58b, (uint8_t)(w1 >> 24)); break;
    default: return -1;
    }
    return HLE_OK;
}

/* ---- triangles ------------------------------------------------------------------------------------------------------------------------------------------------------------- */
enum { TBL = 0xd40, STAGE = 0xf80, SKIP = 0x58a, VMIN = VREC, VMAX = TBL };

/* pc 0xb88: the skip counter. A triangle command with the counter above zero is skipped (and counted down); otherwise the counter is set from the low byte of w0 (the number of triangles to skip when this
 * one is culled; a triangle that is drawn or rejected resets it). Returns 1 when the command is skipped (the loop goes on at pc 0x60) */
static int tri_skip(struct indy *f, int *skipped)
{
    struct hle_task *t = f->t;
    unsigned v0 = hle_dm8(t, SKIP);
    REG(f, R_S1) += 8;
    *skipped = 0;
    if (!v0) {
        hle_st8(t, SKIP, (uint8_t)MUT(184, f->w0, f->w0 >> 1));
        return HLE_OK;
    }
    hle_st8(t, SKIP, (uint8_t)(v0 - 1));
    *skipped = 1;
    if (f->w0 & 512)
        REG(f, R_S1) += 16;
    return HLE_OK;
}

/* pc 0xa90: the colours. The table of colours at DMEM 0xd40 (4 bytes each, indexed by the bytes of the command) gives each vertex its colour word (+16); with fog the alpha of the table entries is first
 * set from the fog value in the vertices' alpha (+19). Then the texture coordinates (bit 9 of w0): 16 bytes that follow the command, one word each for the three vertices (and a fourth for a quad);
 * bit 11 has overlay 42 compute them. Returns the colour word of the fourth entry in *c4. */
static int env_map(struct indy *f, unsigned off, uint32_t out[4]);
static int tri_colours(struct indy *f, unsigned off, unsigned t3, unsigned t4, unsigned t5, unsigned t7, uint32_t *c4)
{
    struct hle_task *t = f->t;
    uint32_t w2 = chunk_word(f, off + 8), w0 = f->w0;
    unsigned t9 = w2 >> 24, s6 = (w2 >> 16) & 255, s7 = (w2 >> 8) & 255, t8 = w2 & 255;
    uint32_t c0, c1, c2;
    if (REG(f, R_S5) & GEOM_FOG) {
        uint8_t a4 = hle_dm8(t, t7 + 19), a3 = hle_dm8(t, t3 + 19), a_4 = hle_dm8(t, t4 + 19), a5 = hle_dm8(t, t5 + 19);
        hle_st8(t, TBL + 3 + t9, a4);
        hle_st8(t, TBL + 3 + s6, MUT(185, a3, a_4));
        hle_st8(t, TBL + 3 + s7, a_4);
        hle_st8(t, TBL + 3 + t8, a5);
    }
    c0 = ld32u(t, TBL + s6);
    c1 = ld32u(t, TBL + s7);
    c2 = ld32u(t, TBL + t8);
    *c4 = ld32u(t, TBL + t9);
    st32u(t, t3 + 16, c0);
    st32u(t, t4 + 16, c1);
    st32u(t, t5 + 16, c2);
    if (!(w0 & 512))
        return HLE_OK;
    if (off + 32 > CMD_HI - CMD_LO)
        return hle_decline(t, "texture coordinates across the end of the command buffer");
    REG(f, R_S1) += 16;
    if (w0 & 2048) {                                    /* (pc 0xaf8: overlay 42) */
        uint32_t w[4];
        unsigned i;
        int rc;
        REG(f, R_A1) = 42;
        REG(f, R_RA) = t7 ? 0xb50 : 0xb30;
        REG(f, R_S8) = REG(f, R_RA);
        if ((rc = overlay(f, 42)) != HLE_OK)
            return rc;
        if ((rc = env_map(f, off, w)) != HLE_OK)
            return rc;
        for (i = 0; i < 4; i++)
            hle_st32(t, STAGE + 4 * i, w[i]);
        st32u(t, t3 + 20, w[0]);
        st32u(t, t4 + 20, w[1]);
        st32u(t, t5 + 20, w[2]);
        f->v0_hi = w[3];
        return HLE_OK;
    }
    st32u(t, t3 + 20, chunk_word(f, off + 16));
    st32u(t, t4 + 20, chunk_word(f, off + MUT(186, 20, 24)));
    st32u(t, t5 + 20, chunk_word(f, off + 24));
    f->v0_hi = chunk_word(f, off + 28);
    return HLE_OK;
}

/* Overlay 42 (pc 0xd60 .. 0xe5c): the texture coordinates of the four vertices of the next 16 bytes of the list (the byte 3 of each 4 is a position offset: the position is the one at 0x170 + the byte)
 * come from the environment: the position times the matrix at 0xe40 (rows 0 .. 3 integer parts, then the fractions), its integer part less the vector at DMEM 0xf0 (wrapping, VSUBC), normalised
 * (1/sqrt of the sum of squares times 171, products in the lo3 form), s = (DMEM[0xec] x x) >> 16, t = (DMEM[0xee] x y) >> 16 of the normalised vector. The four words (s : t) are left at DMEM 0xf80. */
static inline int32_t mulc(int32_t a, int32_t b)
{
    return (int32_t)((uint32_t)(uint16_t)hle_sclamp(hle_mac_32x32(a, b)) << 16 | hle_uclamp(hle_mac_32x32_before_h(a, b)));
}

static int env_map(struct indy *f, unsigned off, uint32_t out[4])
{
    struct hle_task *t = f->t;
    struct f3d_mtx M;
    int16_t cam[4], sc[2];
    unsigned v, j, k;
    f3d_load_mtx(&M, t->dmem, 0xe40, 0);
    for (k = 0; k < 4; k++)
        cam[k] = (int16_t)hle_dm16(t, 0xf0 + 2 * k);
    sc[0] = (int16_t)hle_dm16(t, MUT(187, 0xec, 0xee));
    sc[1] = (int16_t)hle_dm16(t, MUT(187, 0xee, 0xec));
    for (v = 0; v < 4; v++) {
        unsigned pos = 0x170 + (chunk_word(f, off + 16 + 4 * v) & 0xff);
        int16_t px[3];
        int32_t V[4], sq[4], N[4], R, K;
        struct hle_pair S01, S23, S, Kp;
        for (j = 0; j < 3; j++)
            px[j] = (int16_t)((unsigned)hle_dm8(t, pos + 2 * j) << 8 | hle_dm8(t, pos + 2 * j + 1));      /* (LDV from any byte address) */
        for (k = 0; k < 4; k++) {
            hle_acc acc = 0;
            uint16_t hi, lo;
            for (j = 0; j < 3; j++)
                acc += (hle_acc)M.f[j][k] * px[j] + (((hle_acc)M.i[j][k] * px[j]) << 16);
            acc = hle_wrap48(acc + M.f[3][k]);
            lo = hle_uclamp(acc);
            hi = hle_sclamp(hle_wrap48(acc + ((hle_acc)M.i[3][k] << 16)));
            hi = (uint16_t)(hi - MUT(188, (uint16_t)cam[k], 0));
            V[k] = (int32_t)((uint32_t)hi << 16 | lo);
            sq[k] = k == 3 ? 0 : mulc(V[k], V[k]);
        }
        S01 = hle_add32(sq[0], sq[1]);
        S23 = hle_add32(sq[2], sq[3]);
        S = hle_add32(hle_pair32(S01), hle_pair32(S23));
        R = (int32_t)hle_rsq32(hle_pair32(S));
        Kp = hle_pair_acc(hle_mac_32xu16(R, 171));
        K = hle_pair32(Kp);
        for (k = 0; k < 2; k++)
            N[k] = mulc(V[k], K);
        out[v] = (uint32_t)(uint16_t)hle_sclamp(hle_wrap48((hle_acc)sc[0] * N[0])) << 16 | (uint16_t)hle_sclamp(hle_wrap48((hle_acc)sc[1] * N[1]));
    }
    return HLE_OK;
}

static int vtx_ok(unsigned a) { return (a >= VMIN && a + 40 <= VMAX && !((a - VMIN) % 40)) || (a >= 0x380 && a + 40 <= 0x560 && !((a - 0x380) % 40)); }

/* pc 0x1a8: one triangle from the vertex records at t3, t4, t5. `ra` is the return address of the call, which the microcode keeps in s8. Flags: all three outside one plane: rejected; any outside the
 * near / far or the w planes: the clipper. Then the area (command order), the cull test (bit 13 of the geometry mode: negative area), the sort by Y (table driven), and the F3D triangle setup. */
static int tri_clip(struct indy *f, unsigned t3, unsigned t4, unsigned t5, unsigned ra);

/* pc 0x1d8 on: the drawing of the triangle of the vertex records a[0..2] (command order) */
static int tri_draw(struct indy *f, const unsigned a[3], unsigned ra)
{
    struct hle_task *t = f->t;
    struct f3d_vtx v[3];
    struct f3d_tri_in ti;
    uint8_t cmd[0xc0];
    unsigned len = 0, code, k, s0;
    int rc;
    REG(f, R_S8) = ra;
    for (k = 0; k < 3; k++)
        f3d_load_vtx(&v[k], t->dmem, a[k]);
    memset(&ti, 0, sizeof(ti));
    for (k = 0; k < 3; k++)
        ti.v[k] = &v[k];
    ti.geom = REG(f, R_S5);
    ti.persp = hle_dm16(t, 0x14e);
    ti.tile_byte = hle_dm8(t, 0x14a);
    ti.reject_mask = 0x7070;
    ti.clip_mask = 0x4343;
    ti.variant = F3D_TRIV_INDY;
    ti.sort_given = 1;
    code = (unsigned)MUT(191, v[2].sy < v[1].sy, v[2].sy <= v[1].sy) << 2 | (unsigned)(v[1].sy < v[0].sy) << 1 | (unsigned)(v[0].sy < v[2].sy);
    ti.ord[2] = hle_dm8(t, 0x9a + code) >> 1;
    ti.ord[1] = hle_dm8(t, 0xa1 + code) >> 1;
    ti.ord[0] = hle_dm8(t, 0xa8 + code) >> 1;
    ti.par = MUT(192, hle_dm8(t, 0xaf + code) != 0, 0);
    for (k = 0; k < 3; k++)
        if (ti.ord[k] > 2 || ((hle_dm8(t, k == 0 ? 0xa8 + code : k == 1 ? 0xa1 + code : 0x9a + code)) & 1))
            return hle_decline(t, "the sort table is not what it was");
    hle_st16(t, STAGE, (uint16_t)a[0]);
    hle_st16(t, STAGE + 2, (uint16_t)a[1]);
    hle_st16(t, STAGE + 4, (uint16_t)a[2]);
    rc = f3d_tri_setup_noclip(&ti, cmd, &len);
    if (f->trace)
        fprintf(stderr, "  KERNEL rc %d geom %08x par %d ord %u%u%u sy %04x %04x %04x sx %04x %04x %04x\n", rc, ti.geom, ti.par, ti.ord[0], ti.ord[1], ti.ord[2], (uint16_t)v[0].sy, (uint16_t)v[1].sy, (uint16_t)v[2].sy, (uint16_t)v[0].sx, (uint16_t)v[1].sx, (uint16_t)v[2].sx);
    if (rc == F3D_TRI_CULL)                             /* (pc 0x9e4: back to the caller, the counter stays) */
        return HLE_OK;
    REG(f, R_T3) = a[ti.ord[0]];
    if (rc != F3D_TRI_OK) {
        hle_st8(t, SKIP, 0);
        return HLE_OK;
    }
    if ((ti.geom & 6) != 6)                             /* (the game draws shaded, textured triangles: behind a command that lacks a block the microcode leaves rows from its vector registers in the buffer) */
        return hle_decline(t, "a triangle without the shade or the texture block");
    /* pc 0x2a4: the deferred list (if any) runs and the output buffer is flushed. The game's deferred lists only set up the RDP; one that changes what the triangle is built from (the geometry mode, the tile,
     * a vertex record) is not modelled: the microcode's scratch rows behind such a command and the values it reads again are not known */
    {
        uint32_t s5 = REG(f, R_S5), tb = hle_dm8(t, 0x14a), keep[3][10];
        unsigned q;
        if (hle_dm32(t, DEFERRED)) {
            for (k = 0; k < 3; k++)
                for (q = 0; q < 10; q++)
                    keep[k][q] = hle_dm32(t, a[k] + 4 * q);
            if ((rc = call_deferred(f, 0x2ac)) != HLE_OK)
                return rc;
            if (REG(f, R_S5) != s5 || hle_dm8(t, 0x14a) != tb)
                return hle_decline(t, "a deferred list changed the geometry mode or the tile");
            for (k = 0; k < 3; k++)
                for (q = 0; q < 10; q++)
                    if (hle_dm32(t, a[k] + 4 * q) != keep[k][q] && MUT(193, 1, 0))
                        return hle_decline(t, "a deferred list changed a vertex record");
        } else if ((rc = call_deferred(f, 0x2ac)) != HLE_OK)
            return rc;
        if ((REG(f, R_S5) & 0x1002) == 0x1002)         /* (bit 12 with textures: the texture coordinates are not scaled and their fractions and w are lanes the vector registers carry from the vertex load and the earlier triangles: ucode-indy.md section 7) */
            return hle_decline(t, "a textured triangle with geometry mode bit 12");
    }
    hle_st8(t, SKIP, 0);
    s0 = REG(f, R_S0);
    for (k = 0; k < len; k += 4)
        hle_st32(t, s0 + k, (uint32_t)cmd[k] << 24 | (uint32_t)cmd[k + 1] << 16 | (uint32_t)cmd[k + 2] << 8 | cmd[k + 3]);
    REG(f, R_S0) = s0 + len;
    return HLE_OK;
}

static int tri_setup(struct indy *f, unsigned t3, unsigned t4, unsigned t5, unsigned ra)
{
    struct hle_task *t = f->t;
    unsigned a[3] = { t3, t4, t5 };
    uint16_t fa = hle_dm16(t, t5 + 36), fb = hle_dm16(t, t4 + 36), fc = hle_dm16(t, t3 + 36);
    if (f->trace)
        fprintf(stderr, "  TRI %03x %03x %03x flags %04x %04x %04x\n", t3, t4, t5, fc, fb, fa);
    if (!vtx_ok(t3) || !vtx_ok(t4) || !vtx_ok(t5))
        return hle_decline(t, "a triangle vertex outside the vertex records");
    if ((fa & fb & fc) & MUT(189, 0x7070, 0x7060)) {                      /* (pc 0x744) */
        hle_st8(t, SKIP, 0);
        REG(f, R_T3) = t3;
        return HLE_OK;
    }
    if ((fa | fb | fc) & MUT(190, 0x4343, 0x4342))
        return tri_clip(f, t3, t4, t5, ra);
    return tri_draw(f, a, ra);
}

/* Overlay 3, the clipper (pc 0xd60 ..): the triangle is clipped against the six planes (the planes at DMEM 0x00 .. 0x28 and their flag masks at 0x58 .. 0x62, in the order z + w, z - w, y - K w, x - K w,
 * y + K w, x + K w), the new vertices are records at DMEM 0x380 + 40 i (completed by the vertex code: flags, 1/W, the screen position), the polygon is cut into the fan of triangles from its first vertex and
 * each is drawn as above (the code behind the flag tests) and flushed. f3d_clip_tri does the geometry. */
static int tri_clip(struct indy *f, unsigned t3, unsigned t4, unsigned t5, unsigned ra)
{
    struct hle_task *t = f->t;
    struct f3d_vctx c;
    struct f3d_clip_out co;
    struct f3d_vtx v[3];
    const struct f3d_vtx *vp[3];
    unsigned a[3] = { t3, t4, t5 }, i, k, p, rc;
    if (REG(f, R_S5) & GEOM_FOG)
        return hle_decline(t, "clipping with fog");
    hle_st8(t, SKIP, MUT(195, 0, 1));
    REG(f, R_A1) = 3;
    if ((rc = (unsigned)overlay(f, 3)) != HLE_OK)
        return (int)rc;
    hle_st16(t, 0x140, (uint16_t)ra);
    hle_st16(t, 0x588, (uint16_t)t3);
    REG(f, R_T3) = t3;
    for (k = 0; k < 3; k++) {
        f3d_load_vtx(&v[k], t->dmem, a[k]);
        vp[k] = &v[k];
    }
    indy_vctx(f, &c);
    for (p = 0; p < 6; p++) {
        for (k = 0; k < 4; k++)
            c.plane[p][k] = (int16_t)hle_dm16(t, 0x28 - 8 * p + 2 * k);
        c.plane_mask[p] = hle_dm16(t, 0x62 - 2 * MUT(194, p, p ^ 1));
    }
    f3d_clip_tri_seq(&c, vp, &co);
    if (!co.ntri)
        return hle_decline(t, "a triangle that the clipper discards");
    for (i = 0; i < co.nnew; i++)
        f3d_store_vtx(&co.nv[i], t->dmem, 0x380 + 40 * i);
    for (i = 0; i < co.ntri; i++) {
        unsigned fa[3];
        int rcx;
        for (k = 0; k < 3; k++) {
            const struct f3d_vtx *q = co.tri[i][k];
            if (q >= v && q < v + 3)
                fa[k] = a[q - v];
            else
                fa[k] = 0x380 + 40 * (unsigned)(q - co.nv);
        }
        REG(f, R_T3) = fa[0];
        if (f->trace)
            fprintf(stderr, "  FAN %03x %03x %03x\n", fa[0], fa[1], fa[2]);
        if ((rcx = tri_draw(f, fa, 0xde4)) != HLE_OK)
            return rcx;
        if ((rcx = flush(f)) != HLE_OK)
            return rcx;
    }
    REG(f, R_T3) = co.poly[0] >= v && co.poly[0] < v + 3 ? a[co.poly[0] - v] : 0x380 + 40 * (unsigned)(co.poly[0] - co.nv);     /* (the fan loop loads the first vertex of the polygon again before it finds the end) */
    return HLE_OK;
}

/* the triangle commands: 0xbf (pc 0xb20) one triangle, 0xb4 (pc 0xb40) two, (a, b, c) and (a, c, d) */
static int cmd_tri(struct indy *f, unsigned off, int quad)
{
    struct hle_task *t = f->t;
    uint32_t w1 = chunk_word(f, off + 4), w3 = chunk_word(f, off + 12);
    unsigned t3 = w1 >> 16, t4 = w1 & 0xffff, t5 = w3 >> 16, t7 = quad ? (w3 & 0xffff) : 0, skipped;
    uint32_t c4;
    int rc;
    if (off + 16 > CMD_HI - CMD_LO)
        return hle_decline(t, "a triangle command across the end of the command buffer");
    if (quad && !(f->w0 & 512))
        return -1;                                      /* (SLV of a vector register that holds what an earlier command left) */
    tri_skip(f, (int *)&skipped);
    if (skipped)
        return HLE_OK;
    REG(f, R_RA) = 0xb30;
    if ((rc = tri_colours(f, off, t3, t4, t5, t7, &c4)) != HLE_OK)
        return rc;
    REG(f, R_T3) = t3;
    if (!quad) {
        if ((rc = tri_setup(f, t3, t4, t5, 0xb38)) != HLE_OK)
            return rc;
        return flush(f);
    }
    hle_st16(t, 0x37c, (uint16_t)t7);
    hle_st16(t, 0x378, (uint16_t)t3);
    hle_st16(t, 0x37a, (uint16_t)t5);
    if (!vtx_ok(t7))
        return hle_decline(t, "a triangle vertex outside the vertex records");
    st32u(t, t7 + 16, MUT(197, c4, 0));
    st32u(t, t7 + 20, f->v0_hi);
    if ((rc = tri_setup(f, t3, t4, t5, 0xb68)) != HLE_OK)
        return rc;
    if ((rc = flush(f)) != HLE_OK)
        return rc;
    REG(f, R_T3) = t3;
    if ((rc = tri_setup(f, t3, MUT(196, t5, t4), t7, 0xb80)) != HLE_OK)
        return rc;
    return flush(f);
}

/* one iteration of the command loop: returns HLE_OK, or a decline; *done is set when the outermost list ended (the caller hands over to the interpreter) */
static int step(struct indy *f, int *done)
{
    struct hle_task *t = f->t;
    unsigned s1 = REG(f, R_S1), off = s1 - CMD_LO, op, kind;
    uint32_t w0, w1;
    int rc = HLE_OK;

    if (s1 < CMD_LO + 8 || s1 >= CMD_HI)
        return hle_decline(t, "the command pointer is outside the command buffer");
    w0 = chunk_word(f, off);
    w1 = chunk_word(f, off + 4);
    f->w0 = w0;
    f->w1 = w1;
    REG(f, R_S3) = w0;
    REG(f, R_S4) = w1;
    REG(f, R_S1) = s1 + 8;
    op = w0 >> 24;
    if (f->trace)
        fprintf(stderr, "CMD %5u s1=%03x w0=%08x w1=%08x s0=%03x s5=%08x\n", f->ncmd, s1, w0, w1, REG(f, R_S0), REG(f, R_S5));
    if ((op >> 6) == 3) {                               /* an RDP command: written to the output buffer as it is (a texture rectangle takes the next 8 bytes too) */
        uint32_t s0 = REG(f, R_S0);
        REG(f, R_RA) = 0x1060;
        if ((w0 >> 25) == 0x72) {
            rc = cmd_texrect(f, off);
            if (rc != HLE_OK)
                return rc;
        } else {
            hle_st32(t, s0, w0);
            hle_st32(t, s0 + 4, w1);
            REG(f, R_S0) = s0 + 8;
            if (REG(f, R_S0) > OUT_FLUSH && (rc = flush(f)) != HLE_OK)
                return rc;
        }
        kind = K_RDP;
    } else {
        unsigned pc = (op >> 6) ? hle_dm16(t, ((2 * op) & 0x1fe) - 0x9c) : hle_dm16(t, 0xb6 + ((2 * op) & 0x1fe));       /* (the table at DMEM 0xb6; classes 1 and 2 enter it at 0xb6 - 0x9c + 2 (op - 0x40)) */
        kind = classify(pc & 0xfff);
        switch (kind) {
        case K_NOP: break;
        case K_POLL: return HLE_OK;                     /* (pc 0x20: back to the loop without the end of buffer check) */
        case K_REFILL:
            rc = set_list(f, hle_dm32(t, NEXT_CHUNK));
            return rc;
        case K_GCLR: REG(f, R_S5) &= w1; break;
        case K_GSET: REG(f, R_S5) |= w1; break;
        case K_TEXTURE:
            hle_st32(t, PB + 0x38, w0);
            REG(f, R_S5) ^= (REG(f, R_S5) ^ w0) & 2;
            break;
        case K_MOVEWORD: st32u(t, w0 & 0xfff, w1); break;
        case K_OTHER_L: rc = cmd_other(f, 1); break;
        case K_OTHER_H: rc = cmd_other(f, 0); break;
        case K_DL:
            return cmd_dl(f, w1);
        case K_BRANCH:
            REG(f, R_RA) = 0x1010;
            return set_list(f, w1);
        case K_ENDDL:
            rc = cmd_end(f, done);
            if (rc == -1) {
                f->fallbacks[op]++;
                REG(f, R_S1) = s1;
            }
            return rc;
        case K_LOAD:
            if ((rc = cmd_load(f)) == -1) {
                f->fallbacks[op]++;
                REG(f, R_S1) = s1;
                return -1;
            }
            break;
        case K_STREAM: rc = cmd_stream(f); break;
        case K_TRI1:
        case K_TRI2:
            rc = cmd_tri(f, off, kind == K_TRI2);
            if (rc == -1) {
                f->fallbacks[op]++;
                REG(f, R_S1) = s1;
                return -1;
            }
            break;
        default:
            f->fallbacks[op]++;
            REG(f, R_S1) = s1;
            return -1;
        }
        if (rc != HLE_OK)
            return rc;
    }
    if (REG(f, R_S1) == CMD_HI)
        return set_list(f, hle_dm32(t, NEXT_CHUNK));
    return HLE_OK;
}

/* ---- development: the interpreter and the lockstep ------------------------------------------------------------------------------------------------------------------- */
#define ENDED 100
static int interp_step(struct indy *f)
{
    struct hle_cpu *c = f->c;
    int rc;
    c->pc = HEAD;
    c->stop_pc = HEAD;
    c->steps = 0;
    if (f->trace)
        fprintf(stderr, "ISTEP s1=%03x\n", c->r[R_S1]);
    rc = hle_rsp_run(f->t, c);
    if (rc == HLE_STOP)
        return HLE_OK;
    return rc == HLE_OK ? ENDED : rc;                  /* the task ended (the exit code's break) */
}

/* HLE_INDY_LOCKSTEP=1 (with ARMLET_HLE_EXACT=1 and HLE_INDY_HYBRID=1): every command that the C code implements is run by it, the state is put back, the interpreter runs the same command and the
 * end states are compared (DMEM outside the dead ranges, the registers the loop keeps, the DPC registers). */
static struct { uint8_t d0[4096], dn[4096], i0[4096]; size_t mark[2]; uint32_t cr[16], r0[32]; } l_ls;
static const unsigned ls_regs[] = { R_S0, R_S1, R_S2, R_S5, R_T3, R_S8 };

static void ls_begin(struct indy *f)
{
    struct hle_task *t = f->t;
    unsigned i;
    f->ls_n0 = f->ncmd - 1;
    memcpy(l_ls.d0, t->dmem, 4096);
    memcpy(l_ls.i0, t->imem, 4096);
    hle_dev_mark(l_ls.mark);
    for (i = 0; i < 16; i++)
        l_ls.cr[i] = t->cr[i] ? *t->cr[i] : 0;
    memcpy(l_ls.r0, f->c->r, sizeof(l_ls.r0));
}

/* after the C code: puts the state back and runs the interpreter; compares */
static int ls_end(struct indy *f, int native_rc)
{
    struct hle_task *t = f->t;
    struct hle_cpu *c = f->c;
    unsigned i, nd = 0, k;
    uint32_t crn[16], rn[32];
    int rc;
    memcpy(l_ls.dn, t->dmem, 4096);
    memcpy(rn, c->r, sizeof(rn));
    for (i = 0; i < 16; i++)
        crn[i] = t->cr[i] ? *t->cr[i] : 0;
    memcpy(t->dmem, l_ls.d0, 4096);
    memcpy(t->imem, l_ls.i0, 4096);
    hle_dev_rewind(l_ls.mark);
    for (i = 8; i < 12; i++)
        *t->cr[i] = l_ls.cr[i];
    memcpy(c->r, l_ls.r0, sizeof(l_ls.r0));
    for (i = 0, rc = HLE_OK; i < f->ncmd - f->ls_n0 && rc == HLE_OK; i++)
        rc = interp_step(f);
    if (rc == ENDED) {
        if (native_rc == HLE_OK && f->ls_done)
            return ENDED;
        fprintf(stderr, "LOCKSTEP: command %u (w0 %08x w1 %08x): the interpreter ended the task, C rc %d\n", f->ncmd, f->w0, f->w1, native_rc);
        return HLE_DECLINE;
    }
    if (rc != HLE_OK) {
        fprintf(stderr, "LOCKSTEP: command %u (w0 %08x w1 %08x): the interpreter did not come back to the loop (rc %d)\n", f->ncmd, f->w0, f->w1, rc);
        return HLE_DECLINE;
    }
    for (i = 0; i < 4096; i++) {
        unsigned a = i ^ 3;
        if (((uint8_t *)t->dmem)[i] != l_ls.dn[i]) {
            int dead = 0;
            for (k = 0; k < t->uc->n_dead; k++)
                if (a >= t->uc->dead[k].lo && a < t->uc->dead[k].hi)
                    dead = 1;
            if (!dead && nd++ < (getenv("HLE_LS_ALL") ? 100000u : 16u))
                fprintf(stderr, "LOCKSTEP: command %u (w0 %08x w1 %08x): DMEM[%03x] interpreter %02x, C %02x\n", f->ncmd, f->w0, f->w1, a, ((uint8_t *)t->dmem)[i], l_ls.dn[i]);
        }
    }
    if (nd && getenv("HLE_LS_DUMP")) {
        unsigned a0 = (unsigned)strtoul(getenv("HLE_LS_DUMP"), NULL, 16), j;
        fprintf(stderr, "  before     :"); for (j = 0; j < 64; j++) fprintf(stderr, "%02x", ((uint8_t *)l_ls.d0)[(a0 + j) ^ 3]); fprintf(stderr, "\n");
        fprintf(stderr, "  interpreter:"); for (j = 0; j < 64; j++) fprintf(stderr, "%02x", ((uint8_t *)t->dmem)[(a0 + j) ^ 3]); fprintf(stderr, "\n");
        fprintf(stderr, "  C          :"); for (j = 0; j < 64; j++) fprintf(stderr, "%02x", ((uint8_t *)l_ls.dn)[(a0 + j) ^ 3]); fprintf(stderr, "\n");
    }
    for (i = 0; i < sizeof(ls_regs) / sizeof(ls_regs[0]); i++)
        if (c->r[ls_regs[i]] != rn[ls_regs[i]]) {
            fprintf(stderr, "LOCKSTEP: command %u (w0 %08x w1 %08x): r%u interpreter %08x, C %08x\n", f->ncmd, f->w0, f->w1, ls_regs[i], c->r[ls_regs[i]], rn[ls_regs[i]]);
            nd++;
        }
    for (i = 8; i < 12; i++)
        if (*t->cr[i] != crn[i] && i != 10) {
            fprintf(stderr, "LOCKSTEP: command %u: DPC register %u interpreter %08x, C %08x\n", f->ncmd, i, *t->cr[i], crn[i]);
            nd++;
        }
    return nd ? HLE_DECLINE : HLE_OK;
}

static int run_loop(struct indy *f, int *done)
{
    struct hle_task *t = f->t;
    for (;;) {
        int rc;
        if (++f->ncmd > POLL_LIMIT)
            return hle_decline(t, "more commands than the status polls allow");
        if (f->lockstep && !f->sub)
            ls_begin(f);
        rc = step(f, done);
        if (f->lockstep && !f->sub && rc != -1 && rc != -2) {
            if (rc == HLE_OK || rc == RET_DEFERRED) {
                f->ls_done = *done;
                rc = ls_end(f, rc);
                if (rc == ENDED) {
                    *done = 2;
                    return HLE_OK;
                }
                if (rc != HLE_OK)
                    return rc;
                continue;
            }
        }
        if (rc == -1 || rc == -2) {
            if (!f->hybrid || f->lean)
                return hle_decline(t, "a command the native loop does not implement");
            if (f->lockstep && !f->sub) {
                memcpy(t->dmem, l_ls.d0, 4096);
                memcpy(t->imem, l_ls.i0, 4096);
                hle_dev_rewind(l_ls.mark);
                memcpy(f->c->r, l_ls.r0, sizeof(l_ls.r0));
            }
            rc = interp_step(f);
            if (rc == ENDED) {
                *done = 2;
                return HLE_OK;
            }
        }
        if (rc == RET_DEFERRED)
            return rc;
        if (rc != HLE_OK)
            return rc;
        if (*done)
            return HLE_OK;
    }
}

static int indy_run(struct hle_task *t, const struct hle_ucode *u)
{
    struct hle_cpu c;
    struct indy f;
    int rc, done = 0;
    unsigned i;
    (void)u;
    hle_cpu_init(t, &c, t->start_pc, 1u << 22);
    c.stop_pc = HEAD;
    rc = hle_rsp_run(t, &c);
    if (rc != HLE_STOP)
        return rc;
    if (*t->cr[4] & SIG0)
        return hle_decline(t, "the task yields");
    memset(&f, 0, sizeof(f));
    f.t = t;
    f.c = &c;
    f.lean = !hle_exact();
    f.trace = getenv("HLE_INDY_TRACE") != NULL;
    f.hybrid = getenv("HLE_INDY_HYBRID") != NULL;
    f.lockstep = getenv("HLE_INDY_LOCKSTEP") != NULL;
    f.ovl_dev = getenv("HLE_INDY_OVL") != NULL;
    rc = run_loop(&f, &done);
    if (f.hybrid)
        for (i = 0; i < 256; i++)
            if (f.fallbacks[i])
                fprintf(stderr, "INDY_FALLBACK op %02x: %lu\n", i, f.fallbacks[i]);
    if (rc != HLE_OK)
        return rc;
    if (done == 2)
        return HLE_OK;                                 /* (development: the interpreter ran the end of the task) */
    c.pc = 0x164;                                      /* the exit overlay (a1 = 30) through the overlay manager, on the interpreter */
    c.stop_pc = 0x1000;
    return hle_rsp_run(t, &c);
}

static const struct hle_family l_fam = { .name = "indy", .run = indy_run };

/* DMEM dead at the end of a task: the return address the overlay manager parks (0x16a), the command buffer (the lean mode does not fill it), the registers the deferred list saves (0x590 ..), the scratch of the overlays and of the triangle code (0xf80 ..), the clipper's polygon lists (0x560 ..) */
static const struct hle_dead_range l_dead[] = { { 0x16a, 0x16c }, { 0x270, 0x378 }, { 0x590, 0x596 }, { 0xf80, 0xfd0 }, { 0x560, 0x58a } };

HLE_UCODE(l_indy_new) = { .id = 0xa6c64e8d82ca6346ull, .name = "Indiana Jones and the Infernal Machine", .fam = &l_fam, .flags = HLE_F_REGS_LIVE, .live_sregs = (1u << 11) | (1u << 30), .ring_off = 0xfe8, .dead = l_dead, .n_dead = 5 };

#endif
