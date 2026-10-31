/*
 * arMLEt - hle_rare.c: Rare's graphics microcode (Diddy Kong Racing, Jet Force Gemini) on the task-level native path: the commands as plain C
 *
 * The microcode is a Fast3D descendant (docs/ucode-rare.md): the vertex record, the clipper and most of the triangle setup are F3D's, so the arithmetic is that of f3d_fx.h (f3d_vtx_xform,
 * f3d_tri_setup with the F3D_TRIV_RARE rules, f3d_clip_tri); what is Rare's own is the data flow around them: every command that has data DMAs it itself from the address in w1, the RDP
 * lists go through a ring in DMEM (DPC_STATUS.XBUS, every command a list of its own), and the matrices, the vertex offset and the viewport are loaded into DMEM slots by commands of their own.
 *
 * run(): the boot code and the initialisation (OS code) run on the interpreter up to the command loop; then a C loop takes the display list from RDRAM (the lean mode; with ARMLET_HLE_EXACT=1
 * through the microcode's own 320-byte refills of the command buffer and the DMAs it makes), keeps the microcode's state where the microcode keeps it (the state block at 0x110, the
 * matrix slots, the viewport, the vertex records, the display list stack, the ring), and does what each command does. The outermost end of the display list goes back to the interpreter, which
 * runs the microcode's own exit (the overlay that saves the state and ends the task). A command the loop does not know, a state it does not support or a situation in which the microcode
 * would wait for the RDP makes the whole task decline (the JIT runs it).
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
#include "hle_internal.h"

#ifdef ARMLET_NATIVE

#ifdef ARMLET_HLE_MUTATE
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#define MUT(n, ok, bad) (ok)
#endif

/* ---- the builds ----------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* The handlers of both builds, by name; the builds differ in their addresses, in the class-0 prologue (the address of a matrix or vertex command is relative to a base word in Jet Force Gemini),
 * the matrix command (it can multiply), the vertex command (the first slot is in w0), and in three commands Jet Force Gemini has and Diddy Kong Racing has not. */
enum { K_BAD = 0, K_NOP, K_MTX, K_VEC, K_VTX, K_TRI, K_DL, K_BLK, K_HALF1, K_HALF2, K_HALF3, K_PERSP, K_GCLR, K_GSET, K_ENDDL, K_OML, K_OMH, K_TEXTURE, K_MOVEWORD, K_TEXLOAD, K_BASES };

struct rare_build {
    const char *name;
    int jfg;
    uint16_t head;                  /* the command loop: `lw t9, 0(k1)` */
    uint16_t cmdbuf, databuf;       /* the command buffer (320 bytes) and the buffer the class-0 commands DMA to (s6, at most 512 bytes) */
    uint16_t ring_lo, ring_hi;      /* the RDP ring in DMEM */
    uint16_t vtx;                   /* the vertex records (40 bytes each); slot 0 is also the vertex offset vector */
    uint16_t mtx;                   /* the matrix slots (64 bytes each) */
    uint16_t dlstack;               /* the display list stack (words at dlstack + 4 depth) */
    uint16_t enddl;                 /* the pc the outermost end of list is handed over at */
    struct { uint16_t pc; uint8_t kind; } h[24];
};

static const struct rare_build l_dkr = {
    "Diddy Kong Racing", 0, 0x060, 0x6d0, 0x810, 0xa10, 0xe10, 0x2d0, 0x210, 0x1e6, 0x29c,
    { { 0x0a8, K_NOP }, { 0x208, K_NOP }, { 0x210, K_NOP }, { 0x358, K_MTX }, { 0x45c, K_VEC }, { 0x46c, K_VTX }, { 0x674, K_TRI }, { 0x7a4, K_DL }, { 0x76c, K_BLK },
      { 0x2f4, K_HALF1 }, { 0x2f8, K_HALF2 }, { 0x2ec, K_HALF3 }, { 0x2e4, K_PERSP }, { 0x2cc, K_GCLR }, { 0x2bc, K_GSET }, { 0x29c, K_ENDDL }, { 0x25c, K_OML }, { 0x254, K_OMH },
      { 0x230, K_TEXTURE }, { 0x218, K_MOVEWORD } } };
static const struct rare_build l_jfg = {
    "Jet Force Gemini", 1, 0x060, 0x730, 0x870, 0xa70, 0xe70, 0x370, 0x220, 0x1e8, 0x2b0,
    { { 0x0cc, K_NOP }, { 0x3a8, K_MTX }, { 0x4d4, K_TEXLOAD }, { 0x504, K_VEC }, { 0x514, K_VTX }, { 0x70c, K_TRI }, { 0x83c, K_DL }, { 0x804, K_BLK },
      { 0x308, K_HALF1 }, { 0x30c, K_HALF2 }, { 0x300, K_HALF3 }, { 0x2f8, K_PERSP }, { 0x2e0, K_GCLR }, { 0x2d0, K_GSET }, { 0x2b0, K_ENDDL }, { 0x270, K_OML }, { 0x268, K_OMH },
      { 0x244, K_TEXTURE }, { 0x22c, K_MOVEWORD }, { 0x314, K_BASES } } };

/* DMEM that both builds share */
enum { SP = 0x110, PERSP = 0x112, GEOM = 0x114, OTHER_H = 0x118, OTHER_L = 0x11c, TEXTURE_W0 = 0x120, HALF = 0x10c, MTX_SLOT = 0x148, VTX_COUNT = 0x14c, VOFF_ON = 0x144, SEGS = 0x160,
       SEGMASK = 0xb8, TAB0 = 0xc4, TAB2 = 0x76, MOVEW = 0x1be, VIEWPORT = 0x1d0, FOG = 0x1e0, SIG0 = 0x80, POLL_LIMIT = 30000 };

struct rbuf {                       /* the bytes the last class-0 DMA brought: a window of RDRAM (the lean mode) or of DMEM */
    const uint8_t *img;
    unsigned base, mask, len;
    unsigned swap;                  /* armN64: 3 for DMEM's host words, 0 for RDRAM's big-endian bytes */
};

struct rare {
    struct hle_task *t;
    const struct rare_build *b;
    uint32_t k0;                    /* RDRAM address of the next command */
    int gp;                         /* the microcode's byte count left in its command buffer */
    uint32_t k1;                    /* ... and the pointer into it */
    uint32_t s7;                    /* the ring write pointer */
    uint32_t a0;                    /* the scalar register the next task of Jet Force Gemini reads */
    uint32_t w0, w1;
    int lean;
    unsigned ncmd, ntri;
    uint8_t kind0[64], kind2[64];
    struct rbuf buf;
    struct f3d_vtx vb[64];          /* the vertex records as the microcode left them in DMEM, valid per slot in vb_ok */
    uint64_t vb_ok;
    uint64_t vb_wr;                 /* the slots a vertex command of this task wrote (and nothing has overwritten): the clipper is exact for the records the vertex code makes, not for stale or inconsistent ones */
    int clip_loaded;                /* the clipper overlay is (would be) in IMEM */
    int trace;                      /* development: HLE_RARE_TRACE=1 prints every command */
    int watch;                      /* development: HLE_RARE_WATCH=<hex address> reports the commands after which the word there changed */
    uint32_t wval;
    uint64_t total;                 /* bytes put into the ring */
    struct hle_cpu *cpu;            /* the interpreter, for the lockstep */
    int lockstep;                   /* development: HLE_RARE_LOCKSTEP=1 runs every command on the interpreter too (the exact mode) and reports the first difference */
};

/* development: HLE_RARE_PROF=1 sums the time of the interpreter (boot and exit) and of each kind of command over all tasks, printed at exit */
static uint64_t g_pt[24], g_pn[24];
static int g_prof = -1;
static inline uint64_t ticks(void)
{
#if defined(__aarch64__)
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    unsigned lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (uint64_t)hi << 32 | lo;
#endif
}
static void prof_dump(void)
{
    static const char *nm[] = { "bad", "nop/rdp", "mtx", "vec", "vtx", "tri", "dl", "blk", "half1", "half2", "half3", "persp", "gclr", "gset", "enddl", "oml", "omh", "texture", "moveword", "texload", "bases", "boot", "exit", "fetch" };
    unsigned i;
    uint64_t tot = 0;
    for (i = 0; i < 24; i++)
        tot += g_pt[i];
    for (i = 0; i < 24; i++)
        if (g_pn[i])
            fprintf(stderr, "RARE_PROF %-8s n=%-8llu ticks=%-12llu per=%-8.1f %4.1f%%\n", nm[i], (unsigned long long)g_pn[i], (unsigned long long)g_pt[i], (double)g_pt[i] / g_pn[i], 100.0 * g_pt[i] / tot);
}

/* development: HLE_RARE_STATS=1 counts what the commands did over all tasks (the coverage of a test set), printed at exit */
static struct { unsigned long cmd[24], tri_ok, tri_reject, tri_cull, tri_clip, clip_tris, clip_discard, shade, tex, z, flat, fog_vtx, off_vtx, vtx, mmul, wrap; unsigned long combo[8], nnew[16]; } g_st;
static int g_stats = -1;
static const char *g_stats_file;
static void stats_dump(void)
{
    if (g_stats_file) {                                 /* one line per task (the offline harness leaves by _exit): tools/rare/stats.py sums them */
        FILE *fp = fopen(g_stats_file, "a");
        unsigned i;
        if (fp) {
            for (i = 0; i < 21; i++)
                fprintf(fp, "cmd.%u=%lu ", i, g_st.cmd[i]);
            fprintf(fp, "tri_ok=%lu tri_reject=%lu tri_cull=%lu tri_clip=%lu clip_tris=%lu clip_discard=%lu shade=%lu tex=%lu z=%lu flat=%lu vtx=%lu fog_vtx=%lu off_vtx=%lu mmul=%lu", g_st.tri_ok, g_st.tri_reject,
                    g_st.tri_cull, g_st.tri_clip, g_st.clip_tris, g_st.clip_discard, g_st.shade, g_st.tex, g_st.z, g_st.flat, g_st.vtx, g_st.fog_vtx, g_st.off_vtx, g_st.mmul);
            for (i = 0; i < 8; i++)
                fprintf(fp, " combo.%u=%lu", i, g_st.combo[i]);
            for (i = 0; i < 16; i++)
                fprintf(fp, " nnew.%u=%lu", i, g_st.nnew[i]);
            fprintf(fp, "\n");
            fclose(fp);
        }
        return;
    }
    static const char *nm[] = { "bad", "nop", "mtx", "vec", "vtx", "tri", "dl", "blk", "half1", "half2", "half3", "persp", "gclr", "gset", "enddl", "oml", "omh", "texture", "moveword", "texload", "bases" };
    unsigned i;
    for (i = 0; i < 21; i++)
        if (g_st.cmd[i])
            fprintf(stderr, "RARE_STATS cmd %-8s %lu\n", nm[i], g_st.cmd[i]);
    fprintf(stderr, "RARE_STATS triangles: drawn %lu, rejected %lu, culled %lu, clipped %lu (%lu fan triangles, %lu discarded); blocks shade %lu tex %lu z %lu; flat %lu; combos s/t/z:", g_st.tri_ok, g_st.tri_reject, g_st.tri_cull, g_st.tri_clip,
            g_st.clip_tris, g_st.clip_discard, g_st.shade, g_st.tex, g_st.z, g_st.flat);
    for (i = 0; i < 8; i++)
        fprintf(stderr, " %lu", g_st.combo[i]);
    fprintf(stderr, "\nRARE_STATS new vertices per clipped triangle:");
    for (i = 0; i < 16; i++)
        fprintf(stderr, " %lu", g_st.nnew[i]);
    fprintf(stderr, "\nRARE_STATS vertices %lu (fog %lu, offset %lu), matrix products %lu\n", g_st.vtx, g_st.fog_vtx, g_st.off_vtx, g_st.mmul);
}

/* ---- DMEM and the data buffer -------------------------------------------------------------------------------------------------------------------------------------------- */
static inline uint32_t ld32u(const struct hle_task *t, unsigned a)      /* the RSP's lw: any alignment, wrapping in DMEM */
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
static inline unsigned ld16u(const struct hle_task *t, unsigned a) { return (unsigned)hle_dm8(t, a) << 8 | hle_dm8(t, a + 1); }

static inline unsigned bb8(const struct rbuf *b, unsigned off) { return b->img[((b->base + off) & b->mask) ^ b->swap]; }
static inline unsigned bb16(const struct rbuf *b, unsigned off) { return bb8(b, off) << 8 | bb8(b, off + 1); }
static inline uint32_t bb32(const struct rbuf *b, unsigned off) { return (uint32_t)bb16(b, off) << 16 | bb16(b, off + 2); }

/* segment resolution (the microcode's routine at 0x11c): the segment number in bits 25 .. 22 of the address (a word index), the offset masked with the word at 0xb8 */
static inline uint32_t seg(const struct hle_task *t, uint32_t v)
{
    return (v & MUT(50, hle_dm32(t, SEGMASK), 0x01ffffff)) + hle_dm32(t, SEGS + ((v >> 22) & 0x3c));
}

/* ---- the ring ------------------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* The room check before a command of n bytes (the microcode's routine at 0x178): when s7 + n passes the end of the ring, s7 goes back to its start after the microcode has waited for
 * DPC_STATUS.START_VALID to clear and for the RDP not to stand at the start of the ring, and DPC_START is written; then the RDP must not be inside the next n bytes (CURRENT < s7 or
 * CURRENT >= s7 + n + 1). The video plugin has taken every list when DPC_END is written, so nothing waits; a state in which the microcode would wait for ever declines. */
static int ring_room(struct rare *f, unsigned n)
{
    struct hle_task *t = f->t;
    uint32_t cur;
    if ((int32_t)(f->b->ring_hi - (f->s7 + n)) < 0) {
        if (*t->cr[11] & 0x400)
            return hle_decline(t, "ring: DPC start valid while wrapping");
        f->s7 = MUT(48, f->b->ring_lo, f->b->ring_lo + 8);
        if (*t->cr[10] == f->s7)
            return hle_decline(t, "ring: the RDP stands at the start while wrapping");
        hle_mtc0(t, 8, f->s7);
    }
    cur = *t->cr[10];
    if ((int32_t)(f->s7 - cur) >= 0)
        return HLE_OK;
    if ((int32_t)(f->s7 + n - cur) >= 0)
        return hle_decline(t, "ring: the RDP is in the way");
    return HLE_OK;
}

static int emit8(struct rare *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    if (ring_room(f, 8) != HLE_OK)
        return HLE_DECLINE;
    hle_st32(t, f->s7, w0);
    hle_st32(t, f->s7 + 4, w1);
    f->s7 += 8;
    f->total += 8;
    hle_mtc0(t, 9, f->s7);
    return HLE_OK;
}

static int emit_bytes(struct rare *f, const uint8_t *cmd, unsigned len)     /* a triangle: the room check is for the largest one (176 bytes), the list is what was written */
{
    struct hle_task *t = f->t;
    unsigned i;
    if (ring_room(f, 176) != HLE_OK)
        return HLE_DECLINE;
    for (i = 0; i < len; i += 4)
        hle_st32(t, f->s7 + i, (uint32_t)cmd[i] << 24 | (uint32_t)cmd[i + 1] << 16 | (uint32_t)cmd[i + 2] << 8 | cmd[i + 3]);
    f->s7 += len;
    f->total += len;
    hle_mtc0(t, 9, f->s7);
    return HLE_OK;
}

/* ---- the DMA of a class-0 command ------------------------------------------------------------------------------------------------------------------------------------ */
/* The microcode DMAs ((w0 & 0x1ff) + 1 bytes, rounded up to 8) from the address in w1 (rounded down to 8) to the data buffer before the handler runs. The lean mode reads the bytes
 * from RDRAM where the handler needs them; a handler that needs more than was transferred would read what an earlier command left in the buffer, and declines. */
static int fetch_data(struct rare *f, uint32_t addr)
{
    struct hle_task *t = f->t;
    unsigned len = ((f->w0 & 0x1ff) + 1 + 7) & ~7u;
    if (!f->lean) {
        hle_dma_read(t, f->b->databuf, addr, f->w0 & 0x1ff);
        f->buf.img = (const uint8_t *)t->dmem;
        f->buf.base = f->b->databuf;
        f->buf.mask = 0xfff;
        f->buf.swap = 3;
    } else {
        f->buf.img = t->rdram;
        f->buf.base = addr & 0x7ffff8;
        f->buf.mask = 0x7fffff;
        f->buf.swap = 0;
    }
    f->buf.len = len;
    return HLE_OK;
}

/* ---- the commands --------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* the vertex state the transform reads, from DMEM */
static void load_vctx(struct rare *f, struct f3d_vctx *c, int with_offset)
{
    struct hle_task *t = f->t;
    unsigned k, p;
    memset(c, 0, sizeof(*c));
    f3d_load_mtx(&c->mp, t->dmem, f->b->mtx + hle_dm32(t, MTX_SLOT), 0);
    for (k = 0; k < 4; k++) {
        c->vscale[k] = (int16_t)hle_dm16(t, VIEWPORT + 2 * k);
        c->vtrans[k] = (int16_t)hle_dm16(t, VIEWPORT + 8 + 2 * k);
        c->vsign[k] = (int16_t)hle_dm16(t, 0x50 + 2 * k);
        c->lim[k] = (int16_t)hle_dm16(t, 0x28 + 2 * k);
    }
    c->persp = hle_dm16(t, PERSP);
    c->clip_k = (int16_t)hle_dm16(t, 0x76);
    c->geom = hle_dm32(t, GEOM) & F3D_G_FOG;              /* (the other bits are Rare's own: no lighting, no texture generation) */
    for (k = 0; k < 3; k++)
        c->fog[k] = (int16_t)hle_dm16(t, FOG + 2 * k);
    for (p = 0; p < 6; p++) {
        for (k = 0; k < 4; k++)
            c->plane[p][k] = (int16_t)hle_dm16(t, 0x98 - 8 * p + 2 * k);
        c->plane_mask[p] = hle_dm16(t, 0xb2 - 2 * p);
    }
    c->flags_variant = F3D_FLAGS_SM64;
    if (with_offset && hle_dm32(t, VOFF_ON)) {
        c->has_off = 1;
        for (k = 0; k < 4; k++) {
            c->off_i[k] = (int16_t)hle_dm16(t, f->b->vtx + 2 * k);
            c->off_f[k] = hle_dm16(t, f->b->vtx + 8 + 2 * k);
        }
    }
}

static int slot_of(const struct rare *f, unsigned addr)       /* the vertex slot at a DMEM address, or -1 */
{
    if (addr < f->b->vtx || addr + 40 > f->b->ring_lo || (addr - f->b->vtx) % 40)
        return -1;
    return (int)((addr - f->b->vtx) / 40);
}

static void vb_load(struct rare *f, int slot)
{
    if (!(f->vb_ok >> slot & 1)) {
        f3d_load_vtx(&f->vb[slot], f->t->dmem, f->b->vtx + 40 * (unsigned)slot);
        f->vb_ok |= 1ull << slot;
    }
}

static void vw_clear(struct rare *f, unsigned addr, unsigned n)      /* DMEM [addr, addr + n) was written by something else than a vertex command */
{
    unsigned s;
    for (s = 0; s < 64; s++) {
        unsigned lo = f->b->vtx + 40 * s;
        if (addr < lo + 40 && addr + n > lo)
            f->vb_wr &= ~(1ull << s);
    }
}

/* the data buffer's quad k copied to DMEM, a LQV + SQV pair (aligned destinations only) */
static void copy_quads(struct rare *f, unsigned dst, unsigned nquads)
{
    unsigned i;
    vw_clear(f, dst, 16 * nquads);
    for (i = 0; i < 16 * nquads; i++)
        hle_st8(f->t, dst + i, (uint8_t)bb8(&f->buf, i));
}

static int cmd_mtx(struct rare *f)
{
    struct hle_task *t = f->t;
    unsigned b1 = (f->w0 >> 16) & 0xff;
    if (f->buf.len < 64)
        return hle_decline(t, "matrix: less than 64 bytes transferred");
    if (!f->b->jfg) {
        /* DKR: slot b1 gets the 64 bytes */
        if ((b1 & 15) || b1 > 0x80)
            return hle_decline(t, "matrix: unusual slot");
        hle_st32(t, MTX_SLOT, b1);
        copy_quads(f, f->b->mtx + b1, 4);
        f->vb_ok = 0;
        return HLE_OK;
    }
    {
        /* JFG: b1 = multiply << 7 | right slot << 3 | destination slot; without the multiply flag the destination gets the 64 bytes, with it the product (new x slot) */
        unsigned dst = f->b->mtx + ((b1 & 7) << 6), right = f->b->mtx + ((b1 & 0x38) << 3);
        hle_st32(t, MTX_SLOT, (b1 & 7) << 6);
        if (!(b1 & 0x80)) {
            copy_quads(f, dst, 4);
        } else {
            /* the product A x B (A the transferred matrix, B the slot, the result in the destination slot), two rows at a time: the second pair is computed from B as the first pair's store has left it
             * (a product into its own right-hand slot is not a plain matrix product) */
            struct f3d_mtx a, bm, r;
            unsigned rr, cc, pair;
            g_st.mmul++;
            vw_clear(f, dst, 64);
            for (rr = 0; rr < 4; rr++)
                for (cc = 0; cc < 4; cc++) {
                    a.i[rr][cc] = (int16_t)bb16(&f->buf, 8 * rr + 2 * cc);
                    a.f[rr][cc] = (uint16_t)bb16(&f->buf, 0x20 + 8 * rr + 2 * cc);
                }
            for (pair = 0; pair < 2; pair++) {
                if (pair == 0 || !MUT(56, 0, 1))
                    f3d_load_mtx(&bm, t->dmem, right, 0);
                if (MUT(55, 0, 1))
                    f3d_mtx_mul(&bm, &a, &r);
                else
                    f3d_mtx_mul(&a, &bm, &r);
                for (rr = 2 * pair; rr < 2 * pair + 2; rr++)
                    for (cc = 0; cc < 4; cc++) {
                        hle_st16(t, dst + 8 * rr + 2 * cc, (uint16_t)r.i[rr][cc]);
                        hle_st16(t, dst + 0x20 + 8 * rr + 2 * cc, r.f[rr][cc]);
                    }
            }
        }
        f->vb_ok = 0;
    }
    return HLE_OK;
}

static int cmd_vec(struct rare *f)         /* 16 bytes to the address in the halfword at 0x120 + b1 (a SQV: up to the next 16-byte boundary) */
{
    struct hle_task *t = f->t;
    unsigned a1 = ld16u(t, 0x120 + ((f->w0 >> 16) & 0xff)), n = 16 - (a1 & 15), i;
    if (f->buf.len < 16)
        return hle_decline(t, "load vector: less than 16 bytes transferred");
    vw_clear(f, a1, n);
    for (i = 0; i < n; i++)
        hle_st8(t, a1 + i, (uint8_t)bb8(&f->buf, i));
    f->vb_ok = 0;
    return HLE_OK;
}

static int cmd_vtx(struct rare *f)
{
    struct hle_task *t = f->t;
    const struct rare_build *b = f->b;
    unsigned b1 = (f->w0 >> 16) & 0xff, n = b->jfg ? ((b1 >> 3) ? (b1 >> 3) : 1) : (b1 >> 3) + 1, pad = MUT(40, b1 & 6, 0), slot0, a3, i;   /* (JFG counts one less; the first vertex is always done) */
    struct f3d_vctx c;
    struct f3d_vin in[32];
    struct f3d_vtx out[32];

    if (!b->jfg) {
        if (!(b1 & 1))
            hle_st16(t, VTX_COUNT, (uint16_t)n);
        slot0 = (b1 & 1) ? MUT(41, hle_dm16(t, VTX_COUNT), 0) : 0;
    } else
        slot0 = MUT(42, ((f->w0 >> 8) & 0xff) >> 1, ((f->w0 >> 8) & 0xff));
    a3 = b->vtx + 40 * slot0;
    if (a3 + 40 * n > b->cmdbuf)                        /* (the vertex buffer reaches into the command buffer: the microcode would overwrite its own commands) */
        return hle_decline(t, "vertices: the slots reach into the command buffer");
    if (pad + 10 * n > f->buf.len)
        return hle_decline(t, "vertices: more than was transferred");
    hle_st16(t, 0x106, hle_dm16(t, 0xbe));
    load_vctx(f, &c, 1);
    for (i = 0; i < n; i++) {
        unsigned o = pad + 10 * i, k;
        in[i].x = (int16_t)bb16(&f->buf, o);
        in[i].y = (int16_t)bb16(&f->buf, o + 2);
        in[i].z = (int16_t)bb16(&f->buf, o + 4);
        in[i].s = in[i].t = 0;
        for (k = 0; k < 4; k++)
            in[i].c[k] = (uint8_t)bb8(&f->buf, o + 6 + k);
    }
    f3d_vtx_xform(&c, in, n, out);
    if (g_stats > 0) {
        g_st.vtx += n; g_st.fog_vtx += (c.geom & F3D_G_FOG) ? n : 0; g_st.off_vtx += c.has_off ? n : 0;
    }
    for (i = 0; i < n; i++) {
        out[i].s = MUT(58, (int16_t)hle_dm16(t, b->vtx + 8), 0);            /* the fractions of the offset vector (SLV v18), whether or not it is added */
        out[i].t = (int16_t)hle_dm16(t, b->vtx + 10);
    }
    for (i = 0; i < n; i++) {
        unsigned slot = slot0 + i;
        f3d_store_vtx(&out[i], t->dmem, b->vtx + 40 * slot);
        f->vb[slot] = out[i];
        f->vb_ok |= 1ull << slot;
        f->vb_wr |= 1ull << slot;
        if (i & 1)                                              /* the pair's word store (sw at +76): the first vertex's flags land in the second record's padding */
            hle_st16(t, b->vtx + 40 * slot + 38, out[i - 1].flags);
    }
    return HLE_OK;
}

/* the clipper overlay: the descriptor at DMEM 0x10 (RDRAM address, length - 1 in the high half of the second word, IMEM address with bit 12), loaded by a stub the first time it is needed; what it
 * leaves in IMEM after the exit code has overwritten its start is part of the result */
static void load_clipper(struct rare *f)
{
    struct hle_task *t = f->t;
    if (f->trace)
        fprintf(stderr, "CLIPPER load: mem %04x dram %08x len %04x\n", hle_dm16(t, 0x16), hle_dm32(t, 0x10), hle_dm16(t, 0x14));
    hle_dma_read(t, hle_dm16(t, 0x16), hle_dm32(t, 0x10), hle_dm16(t, 0x14));
    f->clip_loaded = 1;
}

static int cmd_tri_one(struct rare *f, struct f3d_vtx *v[3], const unsigned addr[3])
{
    struct hle_task *t = f->t;
    struct f3d_tri_in ti;
    uint8_t cmd[0xc0];
    unsigned len = 0, k;
    int rc;
    memset(&ti, 0, sizeof(ti));
    for (k = 0; k < 3; k++)
        ti.v[k] = v[k];
    ti.flat = MUT(60, v[0], v[2]);
    ti.geom = hle_dm32(t, GEOM);
    ti.persp = hle_dm16(t, PERSP);
    ti.tile_byte = hle_dm8(t, 0x122);
    ti.zclamp = 0;
    ti.reject_mask = 0x7070;
    ti.clip_mask = 0x4343;
    ti.variant = F3D_TRIV_RARE;
    rc = f3d_tri_setup(&ti, cmd, &len);
    if (g_stats > 0) {
        g_st.tri_ok += rc == F3D_TRI_OK; g_st.tri_reject += rc == F3D_TRI_REJECT; g_st.tri_cull += rc == F3D_TRI_CULL; g_st.tri_clip += rc == F3D_TRI_CLIP;
        if (rc == F3D_TRI_OK) {
            g_st.combo[ti.geom & 7]++;
            g_st.shade += (ti.geom & 4) != 0; g_st.tex += (ti.geom & 2) != 0; g_st.z += (ti.geom & 1) != 0; g_st.flat += !(ti.geom & 0x200);
        }
    }
    if (rc == F3D_TRI_REJECT || rc == F3D_TRI_CULL)
        return HLE_OK;
    if (rc == F3D_TRI_OK) {
        if (f->trace > 1) {
            fprintf(stderr, "  TRI s7=%03x len=%u geom=%08x:", f->s7, len, ti.geom);
            for (k = 0; k < len; k++)
                fprintf(stderr, "%s%02x", (k & 31) ? "" : "\n    ", cmd[k]);
            fprintf(stderr, "\n");
        }
        return emit_bytes(f, cmd, len);
    }
    {   /* the clipper: new vertices at 0xe10 + 40 i (the microcode's), the fan triangles of the polygon */
        struct f3d_vctx c;
        struct f3d_clip_out co;
        const struct f3d_vtx *vp[3] = { v[0], v[1], v[2] };
        unsigned i;
        for (i = 0; i < 3; i++) {
            int sl = slot_of(f, addr[i]);
            if (sl < 0 || !(f->vb_wr >> sl & 1))
                return hle_decline(t, "clipper: a vertex record that this task did not make");
        }
        if (!f->clip_loaded && !f->lean)
            load_clipper(f);
        f->clip_loaded = 1;
        load_vctx(f, &c, 0);
        f3d_clip_tri(&c, vp, &co);
        if (g_stats > 0) {
            g_st.clip_tris += co.ntri;
            g_st.clip_discard += co.ntri == 0;
            g_st.nnew[co.nnew < 15 ? co.nnew : 15]++;
        }
        if (f->b->ring_hi + 40 * co.nnew > 0x1000)       /* (the microcode would run its new vertices over the end of DMEM, onto the overlay descriptors at 0x08: the JIT would crash) */
            return hle_decline(t, "clipper: more new vertices than DMEM has room for");
        for (i = 0; i < co.nnew; i++)
            f3d_store_vtx(&co.nv[i], t->dmem, f->b->ring_hi + 40 * i);
        for (i = 0; i < co.ntri; i++) {
            struct f3d_tri_in tj = ti;
            unsigned l2 = 0;
            for (k = 0; k < 3; k++)
                tj.v[k] = co.tri[i][k];
            if (f3d_tri_setup_noclip(&tj, cmd, &l2) == F3D_TRI_OK && emit_bytes(f, cmd, l2) != HLE_OK)
                return HLE_DECLINE;
        }
    }
    return HLE_OK;
}

static int cmd_tri(struct rare *f)
{
    struct hle_task *t = f->t;
    const struct rare_build *b = f->b;
    unsigned b1 = (f->w0 >> 16) & 0xff, n = (b1 >> 4) + 1, i;
    uint32_t geom = hle_dm32(t, GEOM);
    if (16 * n > f->buf.len)
        return hle_decline(t, "triangles: more than was transferred");
    geom = (geom & ~2u) | (b1 & 1) << 1;
    hle_st32(t, GEOM, geom);
    hle_st16(t, 0x194, (uint16_t)n);
    hle_st16(t, 0x196, b->databuf);
    for (i = 0; i < n; i++) {
        unsigned r = 16 * i, fl = bb8(&f->buf, r), a[3], k;
        struct f3d_vtx tv[3], *vp[3];
        int rc;
        (void)rc;
        if (f->ncmd + ++f->ntri > POLL_LIMIT)           /* (one poll per command at pc 0xa8 and one per triangle at 0x6a4; the JIT halts a task at 32767 polls) */
            return hle_decline(t, "more commands and triangles than the status polls allow");
        geom = (hle_dm32(t, GEOM) & ~0x2000u) | (((fl ^ MUT(49, 0x40, 0)) & 0x40) << 7);
        hle_st32(t, GEOM, geom);
        for (k = 0; k < 3; k++) {
            a[k] = b->vtx + 40 * bb8(&f->buf, r + 1 + k);
            if (slot_of(f, a[k]) < 0)                           /* (a vertex in the ring or the buffers would be overwritten by the command's own output while it is still read) */
                return hle_decline(t, "triangles: vertex outside the vertex buffer");
            hle_st32(t, b->ring_hi + 4 * k, a[k]);
        }
        f->a0 = a[0];
        for (k = 0; k < 3; k++) {                                 /* the texture coordinates of the three corners go into their records */
            int slot = slot_of(f, a[k]);
            uint32_t st = bb32(&f->buf, r + 4 + 4 * k);
            if (!MUT(59, 0, 1))
                hle_st32(t, a[k] + 20, st);
            if (f->vb_ok >> slot & 1) {
                f->vb[slot].s = (int16_t)(st >> 16);
                f->vb[slot].t = (int16_t)st;
            }
        }
        hle_st16(t, 0x180, (uint16_t)(b->databuf + r));
        hle_st16(t, 0x182, (uint16_t)(n - i));
        for (k = 0; k < 3; k++) {
            int slot = slot_of(f, a[k]);
            vb_load(f, slot);
            tv[k] = f->vb[slot];
            vp[k] = &tv[k];
        }
        rc = cmd_tri_one(f, vp, a);
        if (rc != HLE_OK)
            return rc;
        hle_st16(t, 0x194, (uint16_t)(n - i - 1));
        hle_st16(t, 0x196, (uint16_t)(b->databuf + r + 16));
    }
    return HLE_OK;
}

static int cmd_dl(struct rare *f)
{
    struct hle_task *t = f->t;
    unsigned b1 = (f->w0 >> 16) & 0xff;
    if (!b1) {
        int d = (int8_t)hle_dm8(t, SP);
        f->a0 = (uint32_t)(d - 36);
        if (d - 36 > 0)
            return HLE_OK;                                       /* the stack is full: the command is ignored */
        st32u(t, f->b->dlstack + d, MUT(43, f->k0, f->k0 - 8));
        hle_st8(t, SP, (uint8_t)(d + 4));
    }
    f->k0 = seg(t, f->w1);
    f->gp = 0;
    return HLE_OK;
}

static int cmd_blk(struct rare *f)         /* b1 RDP commands from the data buffer, each with its own room check and DPC_END */
{
    struct hle_task *t = f->t;
    unsigned n = (f->w0 >> 16) & 0xff, i;
    if (!n || 8 * n > f->buf.len)
        return hle_decline(t, "RDP block: more than was transferred");
    for (i = 0; i < n; i++)
        if (emit8(f, bb32(&f->buf, 8 * i), bb32(&f->buf, 8 * i + 4)) != HLE_OK)
            return HLE_DECLINE;
    return HLE_OK;
}

static int cmd_other(struct rare *f, int high)    /* SETOTHERMODE_H (0xba, word 0x118) / _L (0xb9, word 0x11c), then the pair as a SET_OTHER_MODES command */
{
    struct hle_task *t = f->t;
    unsigned a = MUT(45, high, !high) ? OTHER_H : OTHER_L;
    uint32_t len = f->w0 & 0xff, shift = (f->w0 >> 8) & 0xff, m = ((1u << (len & 31)) - 1) << (shift & 31);
    if (len > 31 || shift > 31)
        return hle_decline(t, "other mode: field outside the word");
    hle_st32(t, a, (hle_dm32(t, a) & ~m) | f->w1);
    return emit8(f, hle_dm32(t, OTHER_H), hle_dm32(t, OTHER_L));
}

static int cmd_moveword(struct rare *f)
{
    struct hle_task *t = f->t;
    unsigned idx = f->w0 & 0xff, off = (f->w0 >> 8) & 0xffff, a;
    a = (ld16u(t, MOVEW + idx) + MUT(46, off, 0)) & 0xfff;
    vw_clear(f, a, 4);
    if (a > 0xffc)
        vw_clear(f, 0, a - 0xffc);
    st32u(t, a, f->w1);
    f->vb_ok = 0;
    return HLE_OK;
}

static int cmd_texture(struct rare *f)
{
    struct hle_task *t = f->t;
    hle_st32(t, TEXTURE_W0, f->w0);
    hle_st32(t, TEXTURE_W0 + 4, f->w1);
    hle_st16(t, SP + 6, (uint16_t)((hle_dm16(t, SP + 6) & 0xfffd) | (MUT(47, f->w0 & 1, (f->w0 >> 1) & 1)) << 1));
    return HLE_OK;
}

/* the outermost end of the list is handled by the caller; here the pop of a call */
static int cmd_enddl(struct rare *f, int *done)
{
    struct hle_task *t = f->t;
    int d = (int8_t)hle_dm8(t, SP) - 4;
    if (d < 0) {
        *done = 1;
        return HLE_OK;
    }
    f->k0 = ld32u(t, f->b->dlstack + MUT(44, d, d + 4));
    hle_st8(t, SP, (uint8_t)d);
    f->gp = 0;
    return HLE_OK;
}

/* ---- the task ------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
static unsigned kind_of(const struct rare_build *b, unsigned pc)
{
    unsigned i;
    for (i = 0; i < sizeof(b->h) / sizeof(b->h[0]) && b->h[i].pc; i++)
        if (b->h[i].pc == pc)
            return b->h[i].kind;
    return K_BAD;
}

static void refill(struct rare *f)
{
    f->gp = 320;
    f->k1 = f->b->cmdbuf;
    if (!f->lean)
        hle_dma_read(f->t, f->b->cmdbuf, f->k0, 319);
}


/* ---- development: lockstep ------------------------------------------------------------------------------------------------------------------------------------------------ */
/* HLE_RARE_LOCKSTEP=1 (with ARMLET_HLE_EXACT=1: the command buffer must be real): every command is run by the C code, the DMEM and the registers are put back, the interpreter runs the same
 * command (from the loop head to the next loop head) and the two end states are compared. The first command after which they differ is reported with the differing bytes. */
static struct { uint8_t d0[4096], dn[4096]; size_t mark[2]; uint32_t cr[16], k0, k1, s7, gp; } l_ls;

static void ls_begin(struct rare *f)
{
    struct hle_task *t = f->t;
    unsigned i;
    memcpy(l_ls.d0, t->dmem, 4096);
    hle_dev_mark(l_ls.mark);
    for (i = 0; i < 16; i++)
        l_ls.cr[i] = t->cr[i] ? *t->cr[i] : 0;
    l_ls.k0 = f->k0; l_ls.k1 = f->k1; l_ls.s7 = f->s7; l_ls.gp = (uint32_t)f->gp;
}

static int ls_end(struct rare *f)
{
    struct hle_task *t = f->t;
    struct hle_cpu *c = f->cpu;
    unsigned i, nd = 0;
    uint32_t crn[16];
    int rc;
    memcpy(l_ls.dn, t->dmem, 4096);
    for (i = 0; i < 16; i++)
        crn[i] = t->cr[i] ? *t->cr[i] : 0;
    memcpy(t->dmem, l_ls.d0, 4096);
    hle_dev_rewind(l_ls.mark);
    for (i = 8; i < 12; i++)
        *t->cr[i] = l_ls.cr[i];
    c->r[HLE_REG_K0] = l_ls.k0; c->r[HLE_REG_K1] = l_ls.k1; c->r[HLE_REG_S7] = l_ls.s7; c->r[28] = l_ls.gp;
    c->pc = f->b->head;
    c->stop_pc = f->b->head;
    c->steps = 0;
    rc = hle_rsp_run(t, c);
    if (rc != HLE_STOP) {
        fprintf(stderr, "LOCKSTEP: command %u (w0 %08x w1 %08x): the interpreter did not come back to the loop (rc %d)\n", f->ncmd, f->w0, f->w1, rc);
        return HLE_DECLINE;
    }
    for (i = 0; i < 4096; i++) {
        unsigned a = i ^ 3;
        if (((uint8_t *)t->dmem)[i] != l_ls.dn[i]) {
            int dead = 0;
            unsigned k;
            for (k = 0; k < f->t->uc->n_dead; k++)
                if (a >= f->t->uc->dead[k].lo && a < f->t->uc->dead[k].hi)
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
    if (c->r[HLE_REG_S7] != f->s7 || c->r[HLE_REG_K0] != f->k0 || c->r[HLE_REG_K1] != f->k1 || (uint32_t)c->r[28] != (uint32_t)f->gp) {
        fprintf(stderr, "LOCKSTEP: command %u (w0 %08x w1 %08x): s7 %03x/%03x k0 %06x/%06x k1 %03x/%03x gp %d/%d (interpreter/C)\n", f->ncmd, f->w0, f->w1, c->r[HLE_REG_S7], f->s7,
                c->r[HLE_REG_K0], f->k0, c->r[HLE_REG_K1], f->k1, (int)c->r[28], f->gp);
        nd++;
    }
    for (i = 8; i < 12; i++)
        if (*t->cr[i] != crn[i] && i != 10) {
            fprintf(stderr, "LOCKSTEP: command %u: DPC register %u interpreter %08x, C %08x\n", f->ncmd, i, *t->cr[i], crn[i]);
            nd++;
        }
    return nd ? HLE_DECLINE : HLE_OK;
}

static int run_loop(struct rare *f, int *done)
{
    struct hle_task *t = f->t;
    const struct rare_build *b = f->b;
    for (;;) {
        uint32_t w0, w1;
        unsigned op, kind = K_NOP, cls;
        int rc = HLE_OK;
        uint64_t t_cmd = 0;

        if (++f->ncmd > POLL_LIMIT)
            return hle_decline(t, "more commands than the status polls allow");
        if (f->lockstep)
            ls_begin(f);
        if (f->lean) {
            w0 = hle_rd32(t, f->k0);
            w1 = hle_rd32(t, f->k0 + 4);
        } else {
            w0 = hle_dm32(t, f->k1);
            w1 = hle_dm32(t, f->k1 + 4);
        }
        f->w0 = w0;
        f->w1 = w1;
        f->k0 += 8;
        f->k1 += 8;
        f->gp -= 8;
        op = w0 >> 24;
        cls = op >> 6;
        if (g_prof > 0)
            t_cmd = ticks();
        if (cls == 0) {
            uint32_t addr = f->w1, base = 0;
            if (b->jfg)
                base = op == 1 ? hle_dm32(t, 0x210) : op == 4 ? hle_dm32(t, 0x214) : 0;
            addr = seg(t, base + addr);
            fetch_data(f, addr);
            kind = f->kind0[op];
        } else if (cls == 2) {
            kind = f->kind2[op - 0x80];
        } else if (cls == 1) {
            kind = K_NOP;
        } else {
            uint32_t w1r = w1;
            if (op >= MUT(51, 0xfd, 0xfe)) {
                if (b->jfg && op == 0xfd && hle_dm8(t, 0x1e7)) {          /* Jet Force Gemini: a table of offsets for the texture images */
                    unsigned i = hle_dm8(t, 0x1e6);
                    w1r += MUT(53, ld16u(t, 0x320 + i), 0);
                    hle_st8(t, 0x1e6, (uint8_t)(i + 2));
                }
                w1r = seg(t, w1r);
            }
            if (emit8(f, w0, w1r) != HLE_OK)
                return HLE_DECLINE;
            kind = K_NOP;
            if (g_prof > 0) {
                g_pt[1] += ticks() - t_cmd;
                g_pn[1]++;
            }
            goto next;
        }
        if (g_stats > 0)
            g_st.cmd[kind]++;
        if (f->trace)
            fprintf(stderr, "CMD %5u k0=%06x w0=%08x w1=%08x kind=%u s7=%03x out=%llu\n", f->ncmd, f->k0 - 8, w0, w1, kind, f->s7, (unsigned long long)f->total);
        switch (kind) {
        case K_NOP: break;
        case K_MTX: rc = cmd_mtx(f); break;
        case K_VEC: rc = cmd_vec(f); break;
        case K_VTX: rc = cmd_vtx(f); break;
        case K_TRI: rc = cmd_tri(f); break;
        case K_DL: rc = cmd_dl(f); break;
        case K_BLK: rc = cmd_blk(f); break;
        case K_HALF1: case K_HALF2:
            rc = emit8(f, MUT(52, hle_dm32(t, HALF), w0), w1);
            break;
        case K_HALF3: hle_st32(t, HALF, w1); break;
        case K_PERSP: hle_st16(t, PERSP, (uint16_t)w1); break;
        case K_GCLR: hle_st32(t, GEOM, hle_dm32(t, GEOM) & ~w1); break;
        case K_GSET: hle_st32(t, GEOM, hle_dm32(t, GEOM) | w1); break;
        case K_ENDDL:
            rc = cmd_enddl(f, done);
            if (*done)
                return HLE_OK;
            break;
        case K_OML: rc = cmd_other(f, 0); break;
        case K_OMH: rc = cmd_other(f, 1); break;
        case K_TEXTURE: rc = cmd_texture(f); break;
        case K_MOVEWORD: rc = cmd_moveword(f); break;
        case K_TEXLOAD:
            if (f->buf.len < 64)
                return hle_decline(t, "texture table: less than 64 bytes transferred");
            copy_quads(f, 0x320, 4);
            hle_st16(t, 0x1e6, 1);
            break;
        case K_BASES:
            hle_st32(t, 0x210, MUT(54, w0 & 0x00ffffff, w0));
            hle_st32(t, 0x214, w1);
            hle_st8(t, 0x1e7, 0);
            break;
        default:
            return hle_decline(t, "unknown command");
        }
        if (g_prof > 0) {
            g_pt[kind] += ticks() - t_cmd;
            g_pn[kind]++;
        }
        if (rc != HLE_OK)
            return rc;
    next:
        if (f->gp <= 0)
            refill(f);
        if (f->lockstep && ls_end(f) != HLE_OK)
            return HLE_DECLINE;
        if (f->watch >= 0 && ld32u(t, f->watch) != f->wval) {
            fprintf(stderr, "WATCH %03x: %08x -> %08x after command %u (w0 %08x w1 %08x)\n", f->watch, f->wval, ld32u(t, f->watch), f->ncmd, w0, w1);
            f->wval = ld32u(t, f->watch);
        }
    }
}

static int rare_run(struct hle_task *t, const struct hle_ucode *u)
{
    const struct rare_build *b = (const struct rare_build *)u->fam_data;
    struct hle_cpu c;
    struct rare f;
    unsigned op;
    int rc, done = 0;
    uint64_t t_boot;

    if (g_stats < 0) {
        const char *e = getenv("HLE_RARE_STATS");
        g_stats = e != NULL;
        g_stats_file = e && e[0] && strcmp(e, "1") ? e : NULL;
        if (g_stats && !g_stats_file)
            atexit(stats_dump);
    }
    if (g_stats_file)
        memset(&g_st, 0, sizeof(g_st));
    if (g_prof < 0) {
        g_prof = getenv("HLE_RARE_PROF") != NULL;
        if (g_prof)
            atexit(prof_dump);
    }
    t_boot = g_prof > 0 ? ticks() : 0;
    hle_cpu_init(t, &c, t->start_pc, 1u << 22);
    c.stop_pc = b->head;
    rc = hle_rsp_run(t, &c);
    if (g_prof > 0) {
        g_pt[21] += ticks() - t_boot;
        g_pn[21]++;
    }
    if (rc != HLE_STOP)
        return rc;
    if (*t->cr[4] & SIG0)
        return hle_decline(t, "the task yields");
    memset(&f, 0, sizeof(f));
    f.t = t;
    f.b = b;
    f.k0 = c.r[HLE_REG_K0];
    f.k1 = c.r[HLE_REG_K1];
    f.gp = (int)c.r[28];
    f.s7 = c.r[HLE_REG_S7];
    f.a0 = c.r[HLE_REG_A0];
    f.lean = !hle_exact();
    f.trace = getenv("HLE_RARE_TRACE") ? atoi(getenv("HLE_RARE_TRACE")) : 0;
    f.lockstep = getenv("HLE_RARE_LOCKSTEP") != NULL;
    f.cpu = &c;
    f.watch = getenv("HLE_RARE_WATCH") ? (int)strtoul(getenv("HLE_RARE_WATCH"), NULL, 16) : -1;
    f.wval = f.watch >= 0 ? ld32u(t, f.watch) : 0;
    if (f.k1 != b->cmdbuf || f.gp != 320 || (f.k0 & 7))
        return hle_decline(t, "unusual state at the command loop");
    for (op = 0; op < 64; op++) {
        f.kind0[op] = (uint8_t)kind_of(b, hle_dm16(t, TAB0 + 2 * op) & 0xfff);
        f.kind2[op] = (uint8_t)kind_of(b, hle_dm16(t, TAB2 + 2 * ((op + 0x80) & 0x7f)) & 0xfff);
    }
    rc = run_loop(&f, &done);
    if (f.trace)
        fprintf(stderr, "LOOP DONE rc %d done %d\n", rc, done);
    if (rc != HLE_OK)
        return rc;
    if (f.clip_loaded && f.lean && !MUT(57, 0, 1))
        load_clipper(&f);                               /* what the microcode would have left in IMEM */
    /* the outermost end of the list: the microcode's own exit. The registers as the loop leaves them. */
    c.r[HLE_REG_K0] = f.k0;
    c.r[HLE_REG_K1] = f.k1;
    c.r[28] = (uint32_t)f.gp;
    c.r[HLE_REG_S7] = f.s7;
    c.r[HLE_REG_A0] = f.a0;
    c.r[HLE_REG_AT] = 4;
    c.r[HLE_REG_T9] = f.w0;
    c.r[HLE_REG_T8] = f.w1;
    c.pc = b->enddl;
    c.stop_pc = 0x1000;
    if (f.trace)
        fprintf(stderr, "HANDOVER pc %03x sp %x depth %02x k0 %06x k1 %03x gp %d s7 %03x\n", c.pc, c.r[29], hle_dm8(t, SP), f.k0, f.k1, f.gp, f.s7);
    t_boot = g_prof > 0 ? ticks() : 0;
    rc = hle_rsp_run(t, &c);
    if (g_prof > 0) {
        g_pt[22] += ticks() - t_boot;
        g_pn[22]++;
        if (g_pn[22] % 100 == 0)                        /* (the offline harness leaves by _exit) */
            prof_dump();
    }
    if (g_stats_file && rc == HLE_OK)
        stats_dump();
    return rc;
}

static const struct hle_family l_fam = { .name = "rare", .run = rare_run };

/* DMEM dead at the end of a task: the microcode reloads its data segment at the start of every task, and only what a task leaves in the RDRAM by DMA (the exit code's save of
 * DMEM[0 .. 0x930) to the yield area, which no later task of a finished list reads) could matter. Dead are the command buffer (the lean mode reads the list from RDRAM), the buffer the class-0
 * commands DMA to (the triangle setup's scratch is in it too), the ring (the lists in it are compared as the RDP stream; the triangle code writes the 64 bytes of its shade rows behind the
 * command whether or not it has them, which depends on vector registers) and a few words the clipper uses for its own bookkeeping. */
static const struct hle_dead_range l_dead_dkr[] = { { 0x0b4, 0x0b6 }, { 0x106, 0x108 }, { 0x158, 0x15a }, { 0x6d0, 0xe10 } };
static const struct hle_dead_range l_dead_jfg[] = { { 0x0b4, 0x0b6 }, { 0x106, 0x108 }, { 0x158, 0x15a }, { 0x730, 0xe70 } };

HLE_UCODE(l_rare_dkr) = { .id = 0x909f82e8f8da6456ull, .name = "Diddy Kong Racing (Rare)", .fam = &l_fam, .flags = HLE_F_XBUS_OK, .fam_data = &l_dkr, .dead = l_dead_dkr, .n_dead = 4 };
HLE_UCODE(l_rare_jfg) = { .id = 0xc2aca403f282af1bull, .name = "Jet Force Gemini (Rare)", .fam = &l_fam, .flags = HLE_F_XBUS_OK | HLE_F_REGS_LIVE, .live_sregs = 1u << 4, .fam_data = &l_jfg, .dead = l_dead_jfg, .n_dead = 4 };

#endif
