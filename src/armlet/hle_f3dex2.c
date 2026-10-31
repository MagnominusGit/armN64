/*
 * arMLEt - hle_f3dex2.c: the F3DEX2 family (F3DEX2, F3DEX2.NoN, F3DZEX, F3DZEX.NoN) on the task-level path: the whole-task native front-end, and the builds as DATA
 *
 * run(): the boot code (the initialisation up to the first fetch of the command loop) runs on the interpreter (hle_rsp.cpp), exactly; then a C loop takes over the display list: it
 * reads the commands (in the lean mode straight from RDRAM, with ARMLET_HLE_EXACT=1 through the microcode's 168-byte refills of the input buffer in DMEM), keeps the microcode's own state
 * where the microcode keeps it (DMEM is the home of the matrices, the lights, the vertex records, the geometry mode ...), and does what each command does with the functions of f3dex2_fx.h
 * (struct level, scalar and NEON). The end of the task (G_ENDDL of the outermost list) goes back to the interpreter: the microcode's own exit path flushes the RDP buffer and ends the
 * task. A command the loop does not know, a state it does not support or a situation in which the microcode would wait for the RDP makes the whole task decline (the JIT runs it).
 *
 * What is compared with the interpreter / the JIT: docs/design.md, "What verify compares" (external effects: final DMEM outside the dead ranges, IMEM, the SP / DPC / MI registers, the
 * RDRAM the task wrote, the RDP command stream).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "armlet.h"
#include "gfx_ucode.h"
#include "hle_f3dex2_int.h"
#include "latch.h"
#include "probe.h"
#include "prsp_native.h"

#ifdef ARMLET_NATIVE

/* development: HLE_F3_PROF=1 sums the time of each command kind over all tasks and prints it at exit */
static uint64_t g_pt[40], g_pn[40], g_pvn[33];     /* (g_pvn: G_VTX commands by vertex count) */
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
    static const char *nm[] = { "bad", "nop", "dl", "enddl", "texture", "texrect", "half1", "half2", "moveword", "movemem", "omh", "oml", "geom", "copy", "sync", "segcopy", "ef", "ed", "vtx", "mtx", "popmtx", "tri1", "tri2", "modvtx", "culldl", "branchz", "mvp" };
    unsigned k;
    for (k = 0; k < 27; k++)
        if (g_pn[k])
            fprintf(stderr, "PROF %-9s n=%-9llu ticks=%-12llu per=%.1f\n", nm[k], (unsigned long long)g_pn[k], (unsigned long long)g_pt[k], (double)g_pt[k] / g_pn[k]);
    fprintf(stderr, "PROF vtx counts:");
    for (k = 1; k <= 32; k++)
        if (g_pvn[k])
            fprintf(stderr, " %u:%llu", k, (unsigned long long)g_pvn[k]);
    fprintf(stderr, "\n");
}

/* ---- the builds ------------------------------------------------------------------------------------------------------------------------------------------------------------------ */
/* the standard layout: F3DEX2, F3DEX2.NoN, F3DZEX in all their versions (the offsets in docs/ucode-f3dex2.md) */
static const struct f2_layout l_lay_std = {
    .mv = 0x00, .p = 0x40, .mvp = 0x80, .persp_norm = 0xdc, .viewport_scale = 0xe0, .viewport_trans = 0xe8, .clip_ratio = 0x186, .const31 = 0x1b0, .const30 = 0x1c0,
    .fog_mul = 0x1e0, .fog_ofs = 0x1e2, .texture = 0x1e4, .tex_scale_s = 0x1e8, .tex_scale_t = 0x1ea, .geom = 0x1ec, .mvp_valid = 0x1d9, .lights_valid = 0x1dc, .lights_size = 0x1dd,
    .light_raw = 0x1f8, .light_base = 0x200, .tg = 0x1d0, .light_scale = 0x1c6, .light_scale_zex = 0x1ce, .vtab = 0x380, .clipmask = 0x3cc, .culltab = 0x3c2, .in = 0x920, .in_end = 0x9c8,
    .jump_table = 0x36e, .mw_max = 14, .geom_def = F2_GEOM_DEF, .movemem_tab = 0x2f0, .moveword_adj = 10238, .desc_light = 0x410, .desc_clip = 0x418, .clip_planes = 0x180, .clip_masks = 0x3f8, .out_bytes = 344, .out_toggle = 0x208 };
const struct f2_layout *f2L = &l_lay_std;
const struct f2_layout *f3dex2_std_layout(void) { return &l_lay_std; }     /* (dbuild.c: the layout of a build whose table has none) */

/* ---- the RDP buffer --------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* The flush of the RDP buffer (the microcode's code after every command that writes to it): the buffer holds s7 - (s6 - 344) bytes; the FIFO pointer in DMEM says where in RDRAM they
 * go; DPC_END is moved up to the end of the chunk written before this one (the RDP is always one chunk behind); when the chunk does not fit before the end of the FIFO the pointer wraps to
 * the start (DPC_START) first. The microcode would wait for the RDP; with the video plugin synchronous there is nothing to wait for, and a state in which it would wait forever (it would
 * hang the JIT too) declines. */
int f3_flush(struct f3 *f)
{
    struct hle_task *t = f->t;
    uint32_t cur = hle_dm32(t, FIFO_PTR);
    uint32_t n = (f->s7 - f->s6) + OUT_BYTES;
    int32_t room;
    if (*t->cr[6])
        return hle_decline(t, "DMA busy while flushing");
    hle_mtc0(t, 9, cur);
    room = (int32_t)hle_dm32(t, FIFO_END) - (int32_t)(cur + n);
    if (room < 0) {
        if (*t->cr[11] & DPC_BUSY_BIT)
            return hle_decline(t, "DPC busy while wrapping the FIFO");
        cur = hle_dm32(t, FIFO_START);
        if (*t->cr[10] == cur)
            return hle_decline(t, "RDP at the start of the FIFO while wrapping");
        MUT(4, hle_mtc0(t, 8, cur), 0);
    }
    {
        int32_t d = (int32_t)*t->cr[10] - (int32_t)cur;
        if (d > 0 && d - (int32_t)n <= 0)
            return hle_decline(t, "FIFO full (the RDP would have to catch up)");
    }
    hle_st32(t, FIFO_PTR, cur + n);
    f->total += n;
    if (f->lean && !(cur & 7) && !(n & 7))
        hle_rdram_write(t, cur, &t->dmem[(f->s6 - OUT_BYTES) >> 2], n);
    else
        hle_dma_write(t, f->s6 - OUT_BYTES, cur, n - 1);
    f->s6 ^= MUT(3, OUT_TOGGLE, OUT_TOGGLE ^ 8);
    f->s7 = f->s6 - OUT_BYTES;
    return HLE_OK;
}

/* A G_MOVEMEM (or S2DEX's G_OBJ_MOVEMEM) DMA goes to an address the display list picks from a table, plus an offset in w0: anywhere in DMEM, and with bit 12 set into IMEM (new microcode). Lean mode
 * keeps no copy of the display list input buffer in DMEM (it reads the list from RDRAM), the opcode classification is taken from the jump table at the start (the entries of the opcodes 0xd0 .. 0xff
 * and 0 .. 9; the others decline), the overlays are tracked by their descriptors, and the display list stack, its depth, the segment table and the RDP FIFO pointer are what the loop works on. A store of
 * [a, a + len] into any of those cannot be modelled, and only a list that rewrites the microcode's own state does it (no game does): decline. (G_MOVEWORD's offsets are bounded instead, below.) */
int f3_store_unmodelled(const struct f3 *f, unsigned a, unsigned len)
{
    unsigned lo = a & 0xfff, hi = lo + len, dl = DESC_LIGHT < DESC_CLIP ? DESC_LIGHT : DESC_CLIP, dh = (DESC_LIGHT < DESC_CLIP ? DESC_CLIP : DESC_LIGHT) + 8;
    if ((a & 0x1000) || hi >= 0x1000)
        return 1;
    return (lo < F2_IN_END && hi >= F2_IN) || (lo < JUMP_TABLE + 0x14u && hi >= JUMP_TABLE - 0x60u) || (lo < dh && hi >= dl)
        || (lo < DL_DEPTH + 1u && hi >= DL_DEPTH) || (lo < FIFO_PTR + 4u && hi >= FIFO_PTR) || (lo < DL_STACK + 0x48u && hi >= SEGMENTS);
}

/* The vertex table (F2_VTAB, 32 halfwords, then the constants of the clipper) is indexed by the vertex number x 2 that the command carries: a number that is odd or past the table reads those constants
 * (and an end index past 32 slots or a count that does not fit below it, in G_VTX, loads vertices over other state). The slot a game addresses is always in the table: decline the rest. */
static inline int vidx_bad(uint32_t w) { return ((w & 0xfff) > 0x3e) | (w & 1); }

/* The RDP output ring is written by the task's own flushes, and the compare treats the bytes of those commands the RDP ignores (the second words of the syncs, the slack after a command) as don't care.
 * A display list, a vertex array, a matrix or a light read from inside the ring would read exactly those bytes: not a game's. Decline when a read of [a, a + n) touches the ring [FIFO_START, FIFO_END). */
static int ring_hit(const struct f3 *f, uint32_t a, uint32_t n)
{
    uint32_t lo = hle_dm32(f->t, FIFO_START) & 0x7fffff, hi = hle_dm32(f->t, FIFO_END) & 0xffffff;
    a &= 0x7ffff8;
    return a < hi && a + n > lo;
}

/* the display list refill that G_DL and G_ENDDL jump into, and the loop's own: the next 168 bytes from k0 */
/* a DMA read into DMEM as the microcode makes it: in the lean mode a plain copy (one row, to DMEM) with the DMA's rounding and clipping, else through the framework */
void f3_dma_rd(struct f3 *f, unsigned mem, uint32_t dram, uint32_t len)
{
    struct hle_task *t = f->t;
    if (MUT(243, ring_hit(f, dram, ((len & 0xfff) | 7) + 1), 0))
        f->bad = "a read from inside the RDP output ring";
    if (f->lean && !(mem & 0x1000) && !(len >> 12)) {
        unsigned n = ((len & 0xfff) + 1 + 7) & ~7u, m = mem & 0xff8, k;
        if (m + n > 0x1000)
            n = 0x1000 - m;
        for (k = 0; k < n; k += 4)
            t->dmem[(m + k) >> 2] = hle_rd32(t, (dram & 0x7ffff8) + k);
        return;
    }
    hle_dma_read(t, mem, dram, len);
}

static inline void refill(struct f3 *f)
{
    if (!f->lean)
        hle_dma_read(f->t, F2_IN, f->k0, F2_IN_REFILL - 1);
    f->k0 += F2_IN_REFILL;
    f->k1 = (uint32_t)-F2_IN_REFILL;
    f->polls++;
}

/* ---- overlays ------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* IMEM [0x2d0, ...) holds the clipping code or the lighting code; each has a descriptor in DMEM (RDRAM address, length register, IMEM address). Which one is resident is found by comparing IMEM with the
 * RDRAM the descriptor points to. */
static int overlay_match(const struct hle_task *t, unsigned desc)
{
    uint32_t dram = hle_dm32(t, desc);
    unsigned reg = hle_dm16(t, desc + 4), imem = hle_dm16(t, desc + 6) & 0xff8, len = ((reg & 0xfff) + 1 + 7) & ~7u;
    if (imem + len > 0x1000 || (dram & 0x7fffff) + len > 0x800000)
        return 0;
    return hle_ram_cmp_words(t->imem + (imem >> 2), t->rdram, dram & 0x7ffff8, len) == 0;
}

void f3_overlay_load(struct f3 *f, int which)
{
    unsigned desc = which == OV_LIGHT ? DESC_LIGHT : DESC_CLIP;
    struct hle_task *t = f->t;
    if (!f->lean)
        hle_dma_read(t, hle_dm16(t, desc + 6), hle_dm32(t, desc), hle_dm16(t, desc + 4));
    f->ov = which;
}

/* ---- the commands ------------------------------------------------------------------------------------------------------------------------------------------------------------- */
static void do_mtx_mul_mvp(struct f3 *f, uint32_t w0)
{
    struct fx2_mtx mv, p, r;
    f2_ld_mtx(f->t, F2_MV, &mv);
    f2_ld_mtx(f->t, F2_P, &p);
    fx2_mtx_mul(&mv, &p, &r);
    f2_st_mtx(f->t, F2_MVP, &r);
    hle_st8(f->t, F2_MVP_VALID, MUT(216, (uint8_t)w0, 0));
}

static inline unsigned first_slot(unsigned first, unsigned i) { return (first + F2_VSTRIDE * i - 0x420) / F2_VSTRIDE; }

static int cmd_vtx(struct f3 *f)
{
    struct hle_task *t = f->t;
    uint32_t w0 = f->w0, w1 = f->w1, a1 = hle_dm16(t, F2_GEOM);
    unsigned n = (w0 >> 12) & 0xff, endslot, first, s4, i, lit = a1 & 2;
    struct fx2_vctx c;
    struct fx2_vin in[34];
    struct fx2_vtx out[34];

    if (n == 0 || n > 32)
        return hle_decline(t, "G_VTX with an unusual count");
    if (lit && hle_cmd(t, LATCH_K_LIT))
        return HLE_DECLINE;
    if (MUT(224, hle_dm32(t, F2_GEOM) & ~f2L->geom_def, 0))        /* (as for the triangles, tri_consts) */
        return hle_decline(t, "geometry mode bits the build does not define");
    if (MUT(221, ((w0 & 0xfff) > 0x40 || (w0 & 1) || (w0 & 0xfff) < 2 * n || (w0 & 0xf00000)), 0))
        return hle_decline(t, "G_VTX with an unusual end index");
    if (lit && f2L->no_lit) {
        if (!f2L->lit_vtx)
            return hle_decline(t, "lit vertices of a build with another light layout");
        return f2L->lit_vtx(f);
    }
    if (lit) {
        unsigned lb = hle_dm8(t, F2_LIGHTS_SIZE);
        if (!(lb >= F2_LIGHT_STRIDE && lb % F2_LIGHT_STRIDE == 0 && lb <= 7 * F2_LIGHT_STRIDE))
        {
            static char why[48];
            snprintf(why, sizeof(why), "light count %u", lb);
            return hle_decline(t, why);
        }
        if (f->ov == OV_UNKNOWN)
            return hle_decline(t, "unknown overlay");
        if (!hle_dm8(t, F2_LIGHTS_VALID)) {             /* the lights have changed since the last lit vertex command: the lighting overlay's setup code */
            struct fx2_mtx mv;
            int8_t raw[FX2_LSLOTS][3], o[FX2_LSLOTS][3];
            unsigned nl = lb / F2_LIGHT_STRIDE, m, j;
            f2_ld_mtx(t, F2_MV, &mv);
            for (m = 0; m <= nl + 1; m++)
                for (j = 0; j < 3; j++)
                    raw[m][j] = (int8_t)hle_dm8(t, F2_LIGHT_RAW + F2_LIGHT_STRIDE * m + j);
            fx2_light_setup(&mv, raw, nl, (int16_t)hle_dm16(t, f->b->zex ? f2L->light_scale_zex : f2L->light_scale), o);
            hle_st8(t, F2_LIGHTS_VALID, (uint8_t)w0);
            for (m = 0; m <= nl + 1; m++)
                for (j = 0; j < 3; j++) {
                    hle_st8(t, F2_LIGHT_BASE + F2_LIGHT_STRIDE * m + j, (uint8_t)o[m][j]);
                    hle_st8(t, F2_LIGHT_BASE + F2_LIGHT_STRIDE * m + 4 + j, (uint8_t)o[m][j]);
                }
        }
    }
    if (!hle_dm8(t, F2_MVP_VALID))                      /* the MVP matrix is stale: modelview x projection */
        do_mtx_mul_mvp(f, w0);
    if (f->trace)
        fprintf(stderr, "  VTX n=%u geom=%04x lights=%u/%u mvpvalid=%02x ov=%d\n", n, a1, hle_dm8(t, F2_LIGHTS_SIZE), hle_dm8(t, F2_LIGHTS_VALID), hle_dm8(t, F2_MVP_VALID), f->ov);
    endslot = hle_dm16(t, F2_VTAB + (w0 & 0xfff));
    first = hle_dm16(t, F2_VTAB + ((w0 - n * 2) & 0xfff));
    s4 = endslot - n * 16;
    if (MUT(243, ring_hit(f, seg_addr(t, w1), n * 16), 0))
        return hle_decline(t, "vertices inside the RDP output ring");
    if (f->lean) {
        if (seg_addr(t, w1) & 7)
            return hle_decline(t, "unaligned vertices");
    } else
        hle_dma_read(t, s4, seg_addr(t, w1), n * 16 - 1);
    if (lit && f->ov == OV_CLIP)                        /* the code at 0x2d0 (the clipping overlay's) loads the lighting overlay */
        f3_overlay_load(f, OV_LIGHT);

    f2_ld_vctx(t, &c);
    c.gen204 = f->b->gen204;
    if (MUT(232, c.ratio & 0x8000, 0))                  /* the guard-band ratio multiplies as a signed lane in the microcode; the ratios of the GBI are 1 .. 6 */
        return hle_decline(t, "guard-band ratio out of range");
    if (lit)
        f2_ld_lights(t, hle_dm8(t, F2_LIGHTS_SIZE) / F2_LIGHT_STRIDE, &c.L);
    if (f->lean) {
        uint32_t va = seg_addr(t, w1);
        for (i = 0; i < n; i++)
            f2_ld_vin_rdram(t, va + 16 * i, &in[i]);
        if (n & 1) {                                    /* an odd count: the pair's second vertex is computed from whatever DMEM holds after the loaded ones; only its clip bits are used */
            unsigned a = s4 + 16 * n;
            if (a == F2_IN)                             /* (the last slot: the bytes after it are the display list's current chunk, which the lean mode does not keep in DMEM) */
                f2_ld_vin_rdram(t, f->k0 - MUT(213, F2_IN_REFILL, F2_IN_REFILL - 8), &in[n]);
            else
                f2_ld_vin(t, a, &in[n]);
        }
    } else
        for (i = 0; i < n + (n & 1); i++)
            f2_ld_vin(t, s4 + 16 * i, &in[i]);
    fx2_vtx_xform(&c, in, n, out);
    if (t->gate || t->cmds) {                           /* (the latch policy's data situations) */
        uint32_t fo = 0;
        int wn = 0;
        for (i = 0; i < n; i++) {
            fo |= out[i].flags;
            wn |= hle_wneg(out[i].cx[3], out[i].cf[3]);
        }
        if (hle_vtx_kinds(t, fo, wn))
            return HLE_DECLINE;
    }
    if (f->trace)
        for (i = 0; i < n; i++)
            fprintf(stderr, "    v%u in=(%d %d %d) clip=(%d.%04x %d.%04x %d.%04x %d.%04x) flags=%08x\n", first_slot(first, i), in[i].x, in[i].y, in[i].z, out[i].cx[0], out[i].cf[0], out[i].cx[1], out[i].cf[1], out[i].cx[2], out[i].cf[2], out[i].cx[3], out[i].cf[3], out[i].flags);
    for (i = 0; i < n; i++)
        f2_st_vtx(t, first + F2_VSTRIDE * i, &out[i]);
    return HLE_OK;
}

static int cmd_mtx(struct f3 *f)
{
    struct hle_task *t = f->t;
    uint32_t w0 = f->w0, w1 = f->w1;
    unsigned at = w0 & 0xfe, len = (w0 >> 16) & 0xff, dst = hle_dm16(t, f2L->movemem_tab + at), k;
    if (!((w0 & 0xfe) < 8 && ((w0 >> 5) & 0x1fff) == 0))
        return hle_decline(t, "G_MTX with an unusual index");
    if (MUT(230, ((len | 7) + 1) != 64, 0))              /* a short DMA leaves the rest of the matrix as it was in DMEM, a long one runs on over the state after it */
        return hle_decline(t, "G_MTX with an unusual length");
    if (f->trace)
        fprintf(stderr, "  MTX flags %02x dst %03x len %u sp %08x seg %08x\n", w0 & 0xff, dst, len + 1, hle_dm32(t, 0xf4), seg_addr(t, w1));
    if ((w0 & 5) == 0) {                                /* push: the current modelview matrix goes to the matrix stack in RDRAM (pointer at 0xf4, which moves up by 64) */
        uint32_t sp = hle_dm32(t, 0xf4);
        hle_dma_write(t, 0, sp, 63);
        hle_st32(t, 0xf4, sp + MUT(14, 64, 32));
    }
    for (k = 0; k < MUT(15, 4, 3); k++)
        hle_st8(t, F2_MVP_VALID + k, 0);                /* the MVP matrix and the lights are stale */
    f3_dma_rd(f, dst, seg_addr(t, w1), len);
    if (!(w0 & 2)) {                                    /* multiply: new x current */
        unsigned d = hle_dm16(t, f2L->movemem_tab + 2 + at);
        struct fx2_mtx a, b, r;
        f2_ld_mtx(t, MUT(16, dst, d), &a);
        f2_ld_mtx(t, MUT(16, d, dst), &b);
        fx2_mtx_mul(&a, &b, &r);
        f2_st_mtx(t, d, &r);
    }
    return HLE_OK;
}

static int cmd_popmtx(struct f3 *f)
{
    struct hle_task *t = f->t;
    uint32_t w0 = f->w0, w1 = f->w1, sp = hle_dm32(t, 0xf4), bottom = hle_dm32(t, 0xfe0), ns = sp - w1;
    unsigned at = w0 & 0xfe, k;
    if ((int32_t)(ns - bottom) < 0)
        ns = bottom;
    hle_st32(t, 0xf4, ns);
    if (ns == sp)
        return HLE_OK;
    if (!((w0 & 0xfe) < 8 && ((w0 >> 5) & 0x1fff) == 0))
        return hle_decline(t, "G_POPMTX with an unusual index");
    for (k = 0; k < 4; k++)
        hle_st8(t, F2_MVP_VALID + k, 0);
    f3_dma_rd(f, hle_dm16(t, f2L->movemem_tab + at), ns, (w0 >> 16) & 0xff);
    return HLE_OK;
}

/* the end of the current display list: pops the stack, or (outermost list) hands over to the interpreter, which runs the microcode's own end of the task */
static int do_enddl(struct f3 *f, int *handover)
{
    struct hle_task *t = f->t;
    unsigned at = hle_dm8(t, DL_DEPTH);
    if (at == 0) {
        *handover = 1;
        return HLE_OK;
    }
    at -= 4;
    f->k0 = hle_dm32(t, DL_STACK + at);
    hle_st8(t, DL_DEPTH, (uint8_t)at);
    refill(f);
    return HLE_OK;
}

/* for the commands of the dialects (hle_rej.c): the end of the display list (the outermost one asks the loop to hand the task back, f->ho_req), and a branch (no stack) to the address a command computed */
int f3_enddl(struct f3 *f)
{
    int h = 0, rc = do_enddl(f, &h);
    f->ho_req = h;
    return rc;
}

int f3_branch(struct f3 *f, uint32_t addr)
{
    f->k0 = seg_addr(f->t, addr);
    if (MUT(240, f->k0 & 7, 0))
        f->bad = "unaligned display list address";
    refill(f);
    return HLE_OK;
}

/* the constants of the triangle setup that the microcode reads from DMEM: once per triangle command. The culling table is indexed by bits 8 .. 15 of the geometry mode (cull front 0x200, cull back 0x400:
 * 0, 2, 4, 6); any other bit set indexes past the 10 bytes of the table, into the clipper's scratch, whose contents the front-end does not keep (a mode no game sets). */
static int tri_consts(struct f3 *f)
{
    struct hle_task *t = f->t;
    struct fx2_tri_in *in = &f->tin;
    unsigned ca = F2_CULLTAB + hle_dm8(t, F2_GEOM + 2);
    if (MUT(224, hle_dm32(t, F2_GEOM) & ~f2L->geom_def, 0))        /* a bit the build gives no meaning to: bits 8 .. 15 index the culling table (past its 10 bytes into the clipper's scratch, which the front-end does not keep), the low byte goes into the command id, and the rest is data of the code paths nobody tests */
        return hle_decline(t, "geometry mode bits the build does not define");
    if (MUT(233, (hle_dm8(t, F2_TEXTURE + 3) & 0xfd) || (hle_dm8(t, F2_TEXTURE + 2) & 0xc0), 0))   /* the texture on byte is 0 or 2 (the triangle command's id takes it ORed in), the tile byte level << 3 | tile (6 bits) */
        return hle_decline(t, "G_TEXTURE with an unusual on / tile byte");
    in->g = hle_dm32(t, F2_GEOM);
    in->tex = hle_dm8(t, F2_TEXTURE + 3);
    in->tile = hle_dm8(t, F2_TEXTURE + 2);
    in->reject_mask = f->b->reject_mask;
    in->cull = (uint32_t)hle_dm8(t, ca) << 24 | (uint32_t)hle_dm8(t, ca + 1) << 16 | (uint32_t)hle_dm8(t, ca + 2) << 8 | hle_dm8(t, ca + 3);
    if (f->b->zex) {
        in->rcp_k = hle_dm16(t, F2_CONST31 + 4);
        in->dx_k = hle_dm16(t, F2_CONST31 + 10);
        in->slope_lim = (int16_t)hle_dm16(t, F2_CONST30 + 4);
        in->frac_mask = 0xffff;
    } else {
        in->rcp_k = hle_dm16(t, F2_CONST30 + 14);
        in->dx_k = hle_dm16(t, F2_CONST30 + 4);
        in->slope_lim = (int16_t)hle_dm16(t, F2_CONST30 + 6);
        in->frac_mask = hle_dm16(t, F2_CONST30 + 10);
    }
    f->clip_mask = hle_dm32(t, F2_CLIPMASK);
    return HLE_OK;
}

/* the triangle setup for three vertex records; HLE_OK, with the RDP command (if any) in the buffer */
static int tri_setup(struct f3 *f, const struct fx2_vtx *const v[3], const struct fx2_vtx *flat, uint32_t clip_mask, int *needs_clip)
{
    struct hle_task *t = f->t;
    struct fx2_tri_in *in = &f->tin;
    uint32_t cmd[0xb0 / 4];
    unsigned len = 0;
    int r;

    in->v[0] = v[0];
    in->v[1] = v[1];
    in->v[2] = v[2];
    in->flat = MUT(214, flat, NULL);
    in->clip_mask = clip_mask;
    r = fx2_tri_setup(in, cmd, &len);
    if (f->trace)
        fprintf(stderr, "  tri_setup -> %d len %u g=%08x cull=%08x tex=%x tile=%x\n", r, len, in->g, in->cull, in->tex, in->tile);
    if (r == FX2_TRI_CLIP) {
        *needs_clip = 1;
        return HLE_OK;
    }
    if (r != FX2_TRI_DRAW)
        return hle_cmd(t, LATCH_K_TRI_CULL) ? HLE_DECLINE : HLE_OK;
    memcpy(&t->dmem[f->s7 >> 2], cmd, len);
    f->s7 += len;
    return after_emit(f);
}

/* the clipper (the clipping overlay's code, entered from the triangle code when a vertex has a flag bit of the clip mask) */
static int do_clip(struct f3 *f, const struct fx2_vtx *const v[3])
{
    struct hle_task *t = f->t;
    struct fx2_vctx vc;
    struct fx2_clip_ctx cc;
    struct fx2_clip_out *co = &f->co;
    unsigned p, k;
    int dummy = 0, rc;

    if (hle_cmd(t, LATCH_K_CLIP))
        return HLE_DECLINE;
    if ((hle_wneg(v[0]->cx[3], v[0]->cf[3]) || hle_wneg(v[1]->cx[3], v[1]->cf[3]) || hle_wneg(v[2]->cx[3], v[2]->cf[3])) && hle_cmd(t, LATCH_K_NEARCLIP))
        return HLE_DECLINE;
    if (f->ov == OV_UNKNOWN)
        return hle_decline(t, "unknown overlay");
    if (f->ov == OV_LIGHT)                              /* the lighting overlay's entry loads the clipping overlay */
        f3_overlay_load(f, OV_CLIP);
    f2_ld_vctx(t, &vc);
    vc.gen204 = f->b->gen204;
    cc.v = &vc;
    cc.sign0 = f->b->clip204;
    cc.fan_l0 = f->b->gen204;
    for (p = 0; p < 6; p++) {
        unsigned a1 = 20 - 4 * p;
        for (k = 0; k < 4; k++)
            cc.plane[p][k] = (int16_t)hle_dm16(t, f2L->clip_planes + 2 * a1 + 2 * k);
        cc.mask[p] = hle_dm32(t, f2L->clip_masks + a1);
    }
    if (fx2_clip_tri(&cc, v, co))
        return hle_decline(t, "clipper: too many vertices");
    for (k = 0; k < co->ntri; k++) {
        rc = tri_setup(f, co->tri[k], v[0], 0, &dummy);
        if (rc != HLE_OK)
            return rc;
    }
    return HLE_OK;
}

/* one triangle of a G_TRI1 / G_TRI2 / G_QUAD word (bytes 1 .. 3 the vertex numbers * 2): the RDP command into the buffer */
static int do_tri(struct f3 *f, uint32_t word)
{
    struct hle_task *t = f->t;
    unsigned sv[3] = { hle_dm16(t, F2_VTAB + ((word >> 16) & 0xff)), hle_dm16(t, F2_VTAB + ((word >> 8) & 0xff)), hle_dm16(t, F2_VTAB + (word & 0xff)) };
    struct fx2_vtx v[3];
    const struct fx2_vtx *vp[3] = { &v[0], &v[1], &v[2] };
    uint32_t fl[3];
    unsigned k;
    int clip = 0, rc;

    if (MUT(217, ((word >> 16) | (word >> 8) | word) & 0xc1, 0))     /* a vertex number outside the 32 slots (or odd): the table lookup reads the constants after it */
        return hle_decline(t, "vertex number out of range");
    for (k = 0; k < 3; k++)
        fl[k] = hle_dm32(t, sv[k] + 0x24);
    if (fl[0] & fl[1] & fl[2] & f->b->reject_mask)
        return hle_cmd(t, LATCH_K_TRI_REJECT) ? HLE_DECLINE : HLE_OK;
    for (k = 0; k < 3; k++)
        f2_ld_vtx_lite(t, sv[k], &v[k]);
    if (f->trace) {
        fprintf(stderr, "TRI");
        for (k = 0; k < 3; k++)
            fprintf(stderr, " [%03x s=%04x t=%04x xy=%08x rw=%08x fl=%08x col=%08x z=%08x]", sv[k], (uint16_t)v[k].s, (uint16_t)v[k].t, (uint32_t)(uint16_t)v[k].sx << 16 | (uint16_t)v[k].sy, (uint32_t)(uint16_t)v[k].rwi << 16 | v[k].rwf, v[k].flags,
                    (uint32_t)v[k].col[0] << 24 | (uint32_t)v[k].col[1] << 16 | (uint32_t)v[k].col[2] << 8 | v[k].col[3], (uint32_t)(uint16_t)v[k].szi << 16 | v[k].szf);
        fprintf(stderr, "\n");
    }
    rc = tri_setup(f, vp, NULL, f->clip_mask, &clip);
    if (rc != HLE_OK || !clip)
        return rc;
    for (k = 0; k < 3; k++)                             /* the clipper needs the clip coordinates too */
        f2_ld_vtx(t, sv[k], &v[k]);
    return do_clip(f, vp);
}

/* G_DL: w1 resolved through the segment table; a call (bit 16 of w0 clear) pushes the address of the next command on the display list stack, a branch does not; then the list is refilled from the target */
void f3_do_dl(struct f3 *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    unsigned d = hle_dm8(t, DL_DEPTH);
    uint32_t ret = f->k0 + f->k1, target = seg_addr(t, w1);
    if (f->trace)
        fprintf(stderr, "  DL depth=%u ret=%08x target=%08x\n", d, ret, target);
    if ((int32_t)(w0 << 15) >= 0) {
        hle_st32(t, DL_STACK + d, ret);
        hle_st8(t, DL_DEPTH, (uint8_t)(d + 4));
    }
    if ((int32_t)(w0 << 15) < 0 && target == ret - 8)  /* a branch to the command itself: the microcode spins on it until the CPU patches the list (Gauntlet Legends), 32767 polls of the JIT; the JIT gets the task now (with PRSP_POLL_IDLE the loop does not get here: f3_cut_dl) */
        f->bad = "the display list branches to itself (a wait for the CPU)";
    f->k0 = target;
    if (MUT(240, target & 7, 0))                        /* the refill DMA rounds the address down to 8 and the list starts at the rounded address; the lean mode reads from the address itself */
        f->bad = "unaligned display list address";
    refill(f);
}

/* Tasks that load another ucode text in the middle (F-Zero X's every task: F3DLX.Rej; one in three of Resident Evil 2's S2DEX tasks: F3DEX.Rej) decline when the command comes, after the native work before it. The text and data pointers of the
 * task are remembered with the number of tasks that ran with them and the number that declined at the load: when more than half of them did, the next tasks with the same pair decline at once, and every 64th runs again in case the game changed
 * (a pair that is mostly not a loader runs natively, the loading minority declines late and the counts, halved from 1024 runs on, follow the game). */
static struct { uint32_t uc, data; unsigned runs, loads, skip; } l_loads[8];
static unsigned l_nloads;

static int load_known(const struct hle_task *t)
{
    unsigned i;
    for (i = 0; i < l_nloads; i++)
        if (l_loads[i].uc == hle_dm32(t, 0xfd0) && l_loads[i].data == hle_dm32(t, 0xfd8)) {
            if (MUT(306, l_loads[i].loads * 2 > l_loads[i].runs, 1) && ++l_loads[i].skip % 64 != 0)
                return 1;
            if (++l_loads[i].runs >= 1024) {
                l_loads[i].runs /= 2;
                l_loads[i].loads /= 2;
            }
            return 0;
        }
    return 0;
}

static void load_note(const struct hle_task *t)
{
    unsigned i;
    for (i = 0; i < l_nloads; i++)
        if (l_loads[i].uc == hle_dm32(t, 0xfd0) && l_loads[i].data == hle_dm32(t, 0xfd8)) {
            l_loads[i].loads++;
            return;
        }
    if (l_nloads < sizeof(l_loads) / sizeof(l_loads[0])) {
        l_loads[l_nloads].uc = hle_dm32(t, 0xfd0);
        l_loads[l_nloads].data = hle_dm32(t, 0xfd8);
        l_loads[l_nloads].runs = 1;
        l_loads[l_nloads++].loads = 1;
    }
}

/* the load of these pointers ran through (the loaded text has a build now: a latch came since the notes): the pointers start afresh, or load_known would go on declining them, 63 tasks in 64, for
 * as long as the old loads outnumber the runs (NBA Jam 2000: 2200 declines after its loaded Rej text latched) */
static void load_took(const struct hle_task *t)
{
    unsigned i;
    for (i = 0; i < l_nloads; i++)
        if (l_loads[i].uc == hle_dm32(t, 0xfd0) && l_loads[i].data == hle_dm32(t, 0xfd8)) {
            l_loads[i].loads = 0;
            l_loads[i].runs = 1;
            l_loads[i].skip = 0;
        }
}

/* The build a G_LOAD_UCODE loads: the one whose text is the 3912 bytes the command copies to IMEM 0x80 (f3dex2_build.load_key, the FNV of exactly those bytes: the overlay code behind them is not part of
 * the identity, as in the ucode id of a task). Only a build of this front-end (f3dex2_fam: the F3DEX2 dialects and S2DEX 2.x) can be taken over. The keys of the last two texts are remembered (compared with
 * memcmp against RDRAM, as gfx_ucode.c does), so that a task that loads the same text every time does not hash it every time. NULL: not a registered (or latched) build. */
static const struct hle_ucode *loaded_build(const struct hle_task *t, uint32_t src)
{
    static struct { uint8_t text[0xf48]; uint32_t addr; uint64_t key; int used; } slot[2];
    static unsigned next;
    const struct hle_ucode *u = NULL;
    uint64_t key = 0;
    unsigned i;
    for (i = 0; i < 2; i++)
        if (slot[i].used && slot[i].addr == src && !memcmp(slot[i].text, t->rdram + src, 0xf48)) {
            key = slot[i].key;
            break;
        }
    if (i == 2) {
        key = armlet_ucode_hash(t->rdram, src, 0xf48);
        i = next++ & 1;
        memcpy(slot[i].text, t->rdram + src, 0xf48);
        slot[i].addr = src;
        slot[i].key = key;
        slot[i].used = 1;
    }
    key = MUT(250, key, ~key);
    /* a build the probe latched for this text (its id is the load key: probe.c) comes first, then the registered ones (the lookup is not memoised: a latch can come between two loads) */
    u = armlet_hle_find(key);
    if (u && !hle_build_usable(t, u))                   /* a derived build that is not latched is not taken over, and neither is a registered one in its place (ARMLET_PROBE_FORCE: the text is as unknown) */
        return NULL;
    if (u && u->fam == &f3dex2_fam && ((const struct f3dex2_build *)u->fam_data)->load_key == key) {
        static const struct hle_ucode *logged[8];
        static unsigned nlogged;
        unsigned q;
        for (q = 0; q < nlogged && logged[q] != u; q++)
            ;
        if (!t->dx && q == nlogged && nlogged < 8 && !strncmp(u->name, "derived", 7)) {   /* (once per latched build: the tests look for it) */
            logged[nlogged++] = u;
            armlet_log(M64MSG_INFO, "hle: G_LOAD_UCODE of %016llx runs the latched build \"%s\"", (unsigned long long)key, u->name);
        }
        return u;
    }
    for (u = armlet_hle_next(NULL); u; u = armlet_hle_next(u))
        if (u->fam == &f3dex2_fam && ((const struct f3dex2_build *)u->fam_data)->load_key == key && hle_build_usable(t, u))
            return u;
    return NULL;
}

/* ---- a wait for the CPU (PRSP_POLL_IDLE=N of the fork, docs/design.md "Waits for the CPU") ------------------------------------------------------------------------------------------------
 * Gauntlet Legends' display lists contain a G_DL branch to the command itself, which the microcode spins on until the CPU patches the list. With the policy on the JIT cuts such a task after N idle
 * polls of SP_STATUS: the poll is the delay slot of the `jal` that waits for the DMA of the refill (head - 4: `jal wait`, head: `mfc0 at, SP_STATUS`), so the cut leaves pc at the wait routine, with
 * ra = head + 4, the refill DMA done, the registers of the loop (k0 = the target + 168, k1 = -168, t9 = w0, t8 = the target, at = the status), and HALT set and cleared again for the core. The next call
 * (SP_PC at the wait pc) runs the DMA wait, returns to head + 4 and executes the command still in the input buffer (the stale one: the same G_DL), whose refill brings the patched list. */
/* the scalar registers the loop carries across commands or the wait routine and the handler leave (at, v0, v1, t3, t4, s3, s4, s6, s7, t8, t9, k0, k1, gp, ra): the verify compare takes them as the JIT leaves them. The others
 * (a0 .. a3, t0 .. t2, t5 .. t7, s0 .. s2, s5, sp, s8) are the temporaries of the vertex, matrix and triangle code the front-end runs in place of the microcode: written before read in every handler (the poison run of
 * docs/testing.md fills them at every cut and nothing changes) */
#define F3_CUT_SREGS ((1u << 1) | (1u << 2) | (1u << 3) | (1u << 11) | (1u << 12) | (1u << 19) | (1u << 20) | (1u << 22) | (1u << 23) | (1u << 24) | (1u << 25) | (1u << 26) | (1u << 27) | (1u << 28) | (1u << 31))

static unsigned f3_wait_pc_b(const struct f3dex2_build *b, const uint32_t *imem)
{
    uint32_t j, m;
    if (!b || b->pcs || b->lay || b->head < 4 || prsp_poll_idle_limit() <= 0)
        return 0;
    j = imem[(b->head - 4) >> 2];
    m = imem[b->head >> 2];
    if ((j >> 26) != 3 || (m >> 26) != 0x10 || ((m >> 21) & 31) != 0 || ((m >> 11) & 31) != 4)    /* jal ..., mfc0 rt, SP_STATUS */
        return 0;
    return ((j & 0x3ff) << 2) & 0xffc;
}

static unsigned f3_wait_pc(const struct hle_ucode *u, const uint32_t *imem) { return f3_wait_pc_b((const struct f3dex2_build *)u->fam_data, imem); }

/* G_DL (branch) to its own address, taken up at the loop head: the microcode's own code runs the handler and the loop to the JIT's cut (the interpreter ends the call there: hle_cut) */
static int f3_cut_dl(struct f3 *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    if (t->dx || (int32_t)(w0 << 15) >= 0 || seg_addr(t, w1) != f->k0 + f->k1 - 8 || !f3_wait_pc_b(f->b, t->imem))
        return 0;
    f->ho_pc = hle_dm16(t, JUMP_TABLE + 2 * (int)(int8_t)(w0 >> 24)) & 0xfff;       /* (the handler: t3 of the jump) */
    f->cut = 1;
    t->cut_ok = 1;
    return 1;
}

/* A call at the wait pc: the state the JIT's cut left, if it is that of a wait on a G_DL to itself (NULL, and the target of the branch), else the reason it is not */
static const char *f3_wait_state(struct hle_task *t, const struct f3dex2_build *b, const uint32_t *r, uint32_t *target)
{
    uint32_t k0 = r[HLE_REG_K0], k1 = r[HLE_REG_K1], w0, w1, tg;
    if (t->dx)
        return "a replay";
    if (r[HLE_REG_RA] != (uint32_t)b->head + 4)
        return "not at the loop head's wait (ra)";
    if (k1 != (uint32_t)-F2_IN_REFILL)
        return "the input buffer is not just refilled";
    if (*t->cr[4] & SIG0)
        return "the task yields";
    w0 = hle_dm32(t, F2_IN_END + k1);                   /* the command still in the input buffer: the stale one */
    w1 = hle_dm32(t, F2_IN_END + 4 + k1);
    if ((w0 >> 24) != 0xde || (int32_t)(w0 << 15) >= 0 || (hle_dm16(t, JUMP_TABLE + 2 * (int)(int8_t)(w0 >> 24)) & 0xfff) != PC_DL)
        return "the command in the input buffer is not a G_DL branch";
    tg = seg_addr(t, w1);
    if (tg != k0 + k1)
        return "the G_DL does not branch to itself";
    if ((tg & 7) || (tg & 0x7fffff) + F2_IN_REFILL > 0x800000)
        return "the G_DL address is unaligned or at the end of RDRAM";
    if (r[HLE_REG_T9] != w0 || r[HLE_REG_T8] != tg)
        return "the registers are not those of the cut";
    *target = tg & 0x7ffff8;
    return NULL;
}

/* nothing has changed in the list the CPU has not patched yet: the JIT's idle slice (the refill of the same bytes, the same loop, the same registers, `at` = the status of that poll) leaves the
 * state it started from. The tracker of the fork is left as the slice's N idle polls leave it. */
static int f3_wait_unchanged(struct hle_task *t, uint32_t target)
{
    return !MUT(481, hle_ram_cmp_words(&t->dmem[F2_IN >> 2], t->rdram, target, F2_IN_REFILL), 0);
}

static void f3_wait_apply(struct hle_task *t, uint32_t *r)
{
    r[HLE_REG_AT] = MUT(482, *t->cr[4], *t->cr[4] ^ 0x40);
    prsp_poll_idle_mark(r);
}

static int f3_idle(struct hle_task *t, const struct hle_ucode *u)
{
    const struct f3dex2_build *b = (const struct f3dex2_build *)u->fam_data;
    struct hle_cpu c;
    uint32_t target;
    if (hle_exact())                                    /* (the microcode's own loop: f3_resume) */
        return 0;
    f2L = &l_lay_std;                                   /* (wait_pc: b->lay is NULL) */
    hle_cpu_init(t, &c, t->start_pc, 0);
    if (f3_wait_state(t, b, c.r, &target) || !f3_wait_unchanged(t, target))
        return 0;
    f3_wait_apply(t, c.r);
    return 1;
}

/* The call that takes the task up at the wait pc. HLE_STOP: the interpreter has run the microcode to the loop head (the stale command's iteration included); the caller goes on with the loop as it does after the
 * boot code of a fresh task. HLE_OK: the wait was unchanged, the call ends at the cut again. */
static int f3_resume(struct hle_task *t, const struct f3dex2_build *b, struct hle_cpu *c)
{
    uint32_t target;
    const char *why;
    int rc;
    if (prsp_poll_idle_limit() <= 0)
        return hle_decline(t, "the idle-poll policy is off");
    if (*t->cr[4] & SIG0) {
        /* a yield request (SIG0): the microcode's own code from the wait routine on, interpreted to its BREAK: the poll at the loop head, the exit code's flush and the yield save
         * (the JIT ran the same ~200 instructions; what it saved the resumed task restores natively, dispatch.c) */
        c->stop_pc = 0x1000;
        return hle_rsp_run(t, c);
    }
    why = f3_wait_state(t, b, c->r, &target);
    if (why)
        return hle_decline(t, why);
    if (f3_wait_unchanged(t, target)) {
        if (hle_exact()) {                              /* the exact-events mode does what the microcode does, DMA for DMA: its loop, to the cut */
            c->stop_pc = 0x1000;
            t->cut_ok = 1;
            rc = hle_rsp_run(t, c);
        } else {
            f3_wait_apply(t, c->r);
            hle_cut(t, t->start_pc);
            rc = HLE_OK;
        }
        if (rc == HLE_OK)
            hle_cut_live(t, F2_IN, F2_IN_END, c->r[HLE_REG_S6] - OUT_BYTES, c->r[HLE_REG_S7], F3_CUT_SREGS);
        return rc;
    }
    rc = hle_rsp_run(t, c);                             /* the DMA wait and the return to the loop head */
    if (rc != HLE_STOP)
        return rc;
    if (MUT(483, 1, 0))
        rc = hle_rsp_run(t, c);                         /* the stale command (the G_DL to itself) again, with its refill: the patched list is in the buffer and the registers are the loop's */
    return rc;
}

static int run_loop(struct f3 *f, int *handover)
{
    struct hle_task *t = f->t;
    const struct f3dex2_build *b = f->b;

    for (;;) {
        uint32_t w0, w1, k, e0, e1;
        unsigned kd, at, addr;
        uint64_t t_cmd = 0;
        int rc = HLE_OK;

        if (f->polls > POLL_LIMIT)
            return hle_decline(t, "too many SP_STATUS polls (the JIT would halt)");
        if (f->bad)
            return hle_decline(t, f->bad);
        if (f->k1 == 0) {                               /* the loop head found the buffer empty: refill */
            refill(f);
            if (MUT(243, ring_hit(f, f->k0 - F2_IN_REFILL, F2_IN_REFILL), 0))
                f->bad = "the display list is inside the RDP output ring";
            continue;
        }
        if (f->lean) {
            f->w0 = w0 = hle_rd32(t, f->k0 + f->k1);
            f->w1 = w1 = hle_rd32(t, f->k0 + f->k1 + 4);
        } else {
            f->w0 = w0 = hle_dm32(t, F2_IN_END + f->k1);
            f->w1 = w1 = hle_dm32(t, F2_IN_END + 4 + f->k1);
        }
        f->polls++;
        f->k1 += 8;
        kd = f->kind[w0 >> 24];
        if (hle_cmd(t, w0 >> 24))                       /* (the latch policy's command-kind gate; the count of a replay under validation) */
            return HLE_DECLINE;
        t_cmd = g_prof ? ticks() : 0;
        if (f->trace)
            fprintf(stderr, "CMD %5u k0=%08x op=%02x kind=%u w0=%08x w1=%08x out=%u entry=%04x\n", f->ncmd++, f->k0 + f->k1, w0 >> 24, kd, w0, w1, f->total + (f->s7 - (f->s6 - OUT_BYTES)), hle_dm16(t, JUMP_TABLE + 2 * (int)(int8_t)(w0 >> 24)));
        switch (kd) {
        case K_NOP:
            break;
        case K_DL:
            if (f3_cut_dl(f, w0, w1)) {                 /* a wait for the CPU (PRSP_POLL_IDLE): the microcode's own code takes it up to the JIT's idle cut */
                *handover = 1;
                return HLE_OK;
            }
            f3_do_dl(f, w0, w1);
            break;
        case K_ENDDL:
            rc = do_enddl(f, handover);
            if (*handover)
                return rc;
            break;
        case K_LOADUCODE: {                             /* the microcode's own code (on the interpreter) loads a ucode text and data and starts it. The same text is a re-initialisation; the text of another registered build of this front-end switches the build
                                                         * the loop works with (f3_bind) once the interpreter has run the loaded microcode's initialisation; any other text declines */
            uint32_t src = w1 & 0x7ffff8, cur = f->uc0 & 0x7ffff8;
            if (src + 0x1000 > 0x800000 || cur + 0xf48 > 0x800000) {
                if (!t->dx)
                    load_note(t);
                return hle_decline(t, "G_LOAD_UCODE outside RDRAM");
            }
            f->nb = b;
            if (memcmp(t->rdram + src, t->rdram + cur, 0xf48)) {
                const struct hle_ucode *nu = loaded_build(t, src);
                /* the probe derives a loaded text too (the G_RDPHALF_1 address is the image of DMEM 0: docs/design.md, "The probe and the ucode cache"); a replay notes it as well: a latched
                 * derived build's task that loads an unknown text declines at the gate in the real run before it gets here, and its replay is the only one that does (NBA Jam 2000) */
                armlet_probe_note_load(src, hle_dm32(t, 0xd8), 0x180 + (w0 & 0xfff) + 1);
                if (f->trace)
                    fprintf(stderr, "  G_LOAD_UCODE text %08x key %016llx data %08x -> %s\n", src, (unsigned long long)armlet_ucode_hash(t->rdram, src, 0xf48), hle_dm32(t, 0xd8), nu ? nu->name : "(not registered)");
                if (!nu || (t->ctx && !armlet_gfx_allowed(&t->ctx->cfg, nu->id))) {
                    if (!t->dx)
                        load_note(t);
                    return hle_decline(t, "G_LOAD_UCODE of another ucode");
                }
                if (hle_cmd(t, LATCH_K_LOAD))
                    return HLE_DECLINE;
                if (!t->dx)
                    load_took(t);
                f->nu = nu;
                f->nb = (const struct f3dex2_build *)nu->fam_data;
                f->nsrc = src;
            }
            f->ho_pc = 0xfac;
            if (f->lean) {                              /* the stub reads the command's second word from the input buffer. The lean mode does not keep the buffer in DMEM, but the microcode that is loaded goes on from
                                                         * DMEM as it is: its output buffers lie over this one, and the bytes it does not write of its commands (the seven after a one-byte sync) are what the
                                                         * buffer held, so the last chunk of the display list (168 bytes before k0) is put there. */
                if (MUT(251, 1, 0))
                    hle_ram_to_words(&t->dmem[F2_IN >> 2], t->rdram, (f->k0 - F2_IN_REFILL) & 0x7ffff8, F2_IN_REFILL);
                hle_st32(t, F2_IN_END + f->k1 - 8, w0);
                hle_st32(t, F2_IN_END + f->k1 - MUT(66, 4, 8), w1);
            }
            *handover = 1;
            return HLE_OK;
        }
        case K_GEOM:
            hle_st32(t, F2_GEOM, (hle_dm32(t, F2_GEOM) & w0) | w1);
            break;
        case K_TEXTURE:
            hle_st32(t, F2_TEXTURE, w0);
            hle_st32(t, F2_TEXTURE + 4, w1);
            break;
        case K_TEXRECT:
            hle_st32(t, 0xd0, w0);
            hle_st32(t, 0xd4, w1);
            break;
        case K_HALF1:
            hle_st32(t, 0xd8, w1);
            break;
        case K_HALF2:
            hle_st32(t, f->s7, hle_dm32(t, 0xd0));
            hle_st32(t, f->s7 + 4, hle_dm32(t, 0xd4));
            hle_st32(t, f->s7 + 8, hle_dm32(t, 0xd8));
            hle_st32(t, f->s7 + 12, w1);
            f->s7 += 16;
            rc = after_emit(f);
            break;
        case K_MOVEWORD:
            if (MUT(235, ((w0 >> 16) & 0xff) > f2L->mw_max || (w0 & 0x10000) || (w0 & 3) || (w0 & 0xffff) > (((w0 >> 16) & 0xff) == 10 ? 0x1fc : 0x3c), 0))     /* the table has the entries 0, 2 .. mw_max, and a game's offsets are words within the group (segments 0 .. 0x3c, clip ratios 4 .. 0x1c, the light colours up to the last light) */
                return hle_decline(t, "G_MOVEWORD index or offset outside what a game writes");
            addr = (hle_dm8(t, ((w0 >> 16) + f2L->moveword_adj) & 0xfff) << 8 | hle_dm8(t, ((w0 >> 16) + f2L->moveword_adj + 1) & 0xfff)) + w0;
            hle_st8(t, addr, (uint8_t)(w1 >> 24));
            hle_st8(t, addr + 1, (uint8_t)(w1 >> 16));
            hle_st8(t, addr + 2, (uint8_t)(w1 >> 8));
            hle_st8(t, addr + 3, (uint8_t)w1);
            break;
        case K_MOVEMEM: {
            unsigned dst;
            if (MUT(231, (w0 & 0xfe) > 14, 0))                  /* the table has the entries 0 .. 14; the next ones are the G_MOVEWORD table's */
                return hle_decline(t, "G_MOVEMEM index outside the table");
            dst = hle_dm16(t, f2L->movemem_tab + (w0 & 0xfe)) + (w0 >> 5);
            if (MUT(219, f3_store_unmodelled(f, dst, (w0 >> 16) & 0xff), 0))
                return hle_decline(t, "G_MOVEMEM into the microcode's own buffers");
            f3_dma_rd(f, dst, seg_addr(t, w1), (w0 >> 16) & 0xff);
            break;
        }
        case K_OMH:
        case K_OML: {                                   /* the other-mode words at 0xc8 / 0xcc: replace the field (length - 1 in the low bits of w0, shift in bits 8..), then the pair goes to the list */
            unsigned a = kd == K_OMH ? 0xc8 : 0xcc;
            uint32_t mask = (uint32_t)((int32_t)0x80000000 >> (w0 & 31)) >> ((w0 >> 8) & 31);
            hle_st32(t, a, (hle_dm32(t, a) & ~mask) | w1);
            e0 = hle_dm32(t, 0xc8);
            e1 = hle_dm32(t, 0xcc);
            hle_st32(t, f->s7, e0);
            hle_st32(t, f->s7 + 4, e1);
            f->s7 += 8;
            rc = after_emit(f);
            break;
        }
        case K_COPY:
        case K_SYNC:
        case K_SEGCOPY:
        case K_EF:
        case K_ED:
            if (kd == K_SEGCOPY)
                w1 = seg_addr(t, w1);
            else if (kd == K_EF) {
                hle_st32(t, 0xc8, w0);
                hle_st32(t, 0xcc, w1);
            } else if (kd == K_ED) {
                hle_st32(t, 0xc0, w0);
                hle_st32(t, 0xc4, w1);
            }
            if (kd != K_SYNC)
                hle_st32(t, f->s7 + MUT(1, 4, 0), w1);

            hle_st32(t, f->s7, w0);
            f->s7 += 8;
            rc = after_emit(f);
            break;
        case K_VTX:
            rc = cmd_vtx(f);
            break;
        case K_MTX:
            rc = cmd_mtx(f);
            break;
        case K_POPMTX:
            rc = cmd_popmtx(f);
            break;
        case K_TRI1:
            rc = tri_consts(f);
            if (rc == HLE_OK)
                rc = do_tri(f, w0);
            break;
        case K_TRI2:                                    /* the second triangle first */
            rc = tri_consts(f);
            if (rc == HLE_OK)
                rc = do_tri(f, w1);
            if (rc == HLE_OK)
                rc = do_tri(f, w0);
            break;
        case K_MVP:
            do_mtx_mul_mvp(f, w0);
            break;
        case K_TRI4: {                                  /* four triangles: vertex numbers x 2 in 6-bit fields; the last one's third is split between the words */
            uint32_t g = w1, tw[4];
            tw[0] = ((g >> 24) & 62) << 16 | ((g >> 19) & 62) << 8 | ((g >> 14) & 62);
            tw[1] = ((g >> 9) & 62) << 16 | ((g >> 4) & 62) << 8 | ((g << 1) & 62);
            tw[2] = ((w0 >> 9) & 62) << 16 | ((w0 >> 4) & 62) << 8 | ((w0 << 1) & 62);
            tw[3] = ((w0 >> 22) & 62) << 16 | ((w0 >> 17) & 62) << 8 | (((w0 >> 12) & 56) | ((g >> MUT(65, 29, 28)) & 6));
            rc = tri_consts(f);
            for (at = 0; at < 4 && rc == HLE_OK; at++)
                rc = do_tri(f, tw[at]);
            break;
        }
        case K_MODVTX:
            if (MUT(218, vidx_bad(w0) || ((w0 >> 16) & 0xff) >= 0x28 || ((w0 >> 16) & 3), 0))      /* a vertex outside the 32 slots, an odd number, or a field outside the record: a store anywhere in DMEM */
                return hle_decline(t, "G_MODIFYVTX out of range");
            at = ((w0 >> 16) & 0xff) + hle_dm16(t, F2_VTAB + (w0 & 0xfff));       /* (the slot address is the delay slot's lhu) */
            hle_st8(t, at, (uint8_t)(w1 >> 24));
            hle_st8(t, at + 1, (uint8_t)(w1 >> 16));
            hle_st8(t, at + 2, (uint8_t)(w1 >> 8));
            hle_st8(t, at + 3, (uint8_t)w1);
            break;
        case K_CULLDL: {                                /* the display list ends when a clip-flag bit of the reject mask is set in all the vertices of the range */
            if (MUT(222, vidx_bad(w0) | vidx_bad(w1), 0))
                return hle_decline(t, "G_CULLDL with an unusual vertex number");
            unsigned a = hle_dm16(t, F2_VTAB + (w0 & 0xfff)), z = hle_dm16(t, F2_VTAB + (w1 & 0xfff));
            uint32_t m = b->reject_mask;
            for (;;) {
                m &= hle_dm32(t, a + 0x24);
                if (!m)
                    break;
                if (a == z) {
                    if (hle_cmd(t, LATCH_K_CULLDL))
                        return HLE_DECLINE;
                    rc = do_enddl(f, handover);
                    if (*handover)
                        return rc;
                    break;
                }
                a += 40;
            }
            break;
        }
        case K_BRANCHZ: {
            if (MUT(223, vidx_bad(w0), 0))
                return hle_decline(t, "G_BRANCH_Z with an unusual vertex number");
            unsigned vs = hle_dm16(t, F2_VTAB + (w0 & 0xfff));
            uint32_t z = (b->zex && MUT(259, !b->gen204, 1) && MUT(246, 1, 0) && MUT(317, !b->bz_z, 1)) || MUT(321, b->bz_w, 0) ? (uint32_t)(int32_t)(int16_t)hle_dm16(t, vs + 6) : hle_dm32(t, vs + 0x1c);   /* F3DZEX compares the integer part of the clip w (`lh 6(vertex)`), the others (F3DEX2 2.04 / 2.04H, which have the `zex` triangle constants, too) the screen z */
            if ((int32_t)(z - w1) < 0) {                /* a branch (no stack) to the address G_RDPHALF_1 left at 0xd8 */
                if (hle_cmd(t, LATCH_K_BRZ))
                    return HLE_DECLINE;
                f->k0 = seg_addr(t, hle_dm32(t, 0xd8));
                if (MUT(240, f->k0 & 7, 0))
                    f->bad = "unaligned display list address";
                refill(f);
            }
            break;
        }
        default: {
            if (kd >= K_EXT && b->ext_cmd) {
                rc = b->ext_cmd(f, kd);
                if (f->ho_req) {                        /* the command ended the outermost display list */
                    f->ho_req = 0;
                    *handover = 1;
                    return rc;
                }
                break;
            }
            static char why[8][48];
            static unsigned wn;
            char *w = why[wn++ & 7];
            snprintf(w, sizeof(why[0]), "unsupported command %02x (handler %03x)", w0 >> 24, hle_dm16(t, JUMP_TABLE + 2 * (int)(int8_t)(w0 >> 24)) & 0xfff);
            return hle_decline(t, w);
        }
        }
        if (g_prof) {
            g_pt[kd] += ticks() - t_cmd;
            g_pn[kd]++;
            if (kd == K_VTX)
                g_pvn[((w0 >> 12) & 0xff) < 33 ? (w0 >> 12) & 0xff : 0]++;
        }
        if (f->watch >= 0 && hle_dm16(t, f->watch) != f->wval) {
            f->wval = hle_dm16(t, f->watch);
            fprintf(stderr, "WATCH %03x = %04x after cmd %u op %02x\n", f->watch, f->wval, f->ncmd - 1, w0 >> 24);
        }
        (void)k;
        if (rc != HLE_OK)
            return rc;
    }
}

static unsigned classify(const struct f3dex2_build *b, unsigned h)
{
    const struct f2_pcs *p = b->pcs;
    unsigned i;
    if (h == b->head || h == (unsigned)b->head - 4)
        return K_NOP;
    if (h == b->copy)
        return K_COPY;
    if (h == b->sync)
        return K_SYNC;
    if (h == b->segcopy)
        return K_SEGCOPY;
    if (h == b->ef)
        return K_EF;
    if (h == b->ed)
        return K_ED;
    if (b->tri1 && h == b->tri1)
        return K_TRI1;
    if (b->tri2 && h == b->tri2)
        return K_TRI2;
    if (b->vtx && h == b->vtx)
        return K_VTX;
    if (b->mtx && h == b->mtx)
        return K_MTX;
    if (b->geom && h == b->geom)
        return K_GEOM;
    if (h == b->enddl)
        return K_ENDDL;
    if (h == b->half2)
        return K_HALF2;
    if (b->modvtx && h == b->modvtx)
        return K_MODVTX;
    if (b->culldl && h == b->culldl)
        return K_CULLDL;
    if (b->branchz && h == b->branchz)
        return K_BRANCHZ;
    if (h == 0xfac)
        return K_LOADUCODE;
    if (b->mvp && h == b->mvp)
        return K_MVP;
    if (b->tri4 && h == b->tri4)
        return K_TRI4;
    for (i = 0; b->ext && b->ext[i][0] != 0xffff; i++)
        if (b->ext[i][0] == h)
            return b->ext[i][1];
    if (p) {
        if (h == p->dl) return K_DL;
        if (h == p->texture) return K_TEXTURE;
        if (h == p->texrect) return K_TEXRECT;
        if (h == p->half1) return K_HALF1;
        if (h == p->moveword) return K_MOVEWORD;
        if (h == p->popmtx) return K_POPMTX;
        if (h == p->movemem) return K_MOVEMEM;
        if (h == p->omh) return K_OMH;
        if (h == p->oml) return K_OML;
        return K_BAD;
    }
    switch (h) {
    case PC_DL: return K_DL;
    case PC_TEXTURE: return K_TEXTURE;
    case PC_TEXRECT: return K_TEXRECT;
    case PC_HALF1: return K_HALF1;
    case PC_MOVEWORD: return K_MOVEWORD;
    case PC_POPMTX: return K_POPMTX;
    case PC_MOVEMEM: return K_MOVEMEM;
    case PC_OMH: return K_OMH;
    case PC_OML: return K_OML;
    }
    return K_BAD;
}

/* The jump table has an entry for every opcode, but only the defined ones point at a handler: the others hold other data of the microcode, or 0, which is the pc of G_DL, and some builds' point at the
 * sync handler (Conker's 0x09 .. 0x0f). A command that is not G_DL (0xde) and still lands on its handler is an undefined opcode, and so is a one-word copy (the second word of the RDP command stays
 * stale: the stream compare only knows that for the sync commands 0xe6 .. 0xe9 and the no-op 0) that is not one of those: decline. */
static unsigned classify_op(const struct f3dex2_build *b, unsigned op, unsigned h)
{
    unsigned k = classify(b, h);
    if (MUT(225, k == K_DL && op != 0xde, 0))
        return K_BAD;
    if (MUT(226, k == K_SYNC && op != 0 && (op < 0xe6 || op > 0xe9), 0))
        return K_BAD;
    return k;
}

/* The loop works with one build at a time (f->b, the layout f2L, the classification of the opcodes by the jump table in DMEM, the overlay IMEM holds, the pc the end of the task goes to): taken from the
 * state of the microcode when it is at the first fetch of its command loop, at the start of a task and after a G_LOAD_UCODE (the same build or another one). */
static void f3_bind(struct f3 *f, const struct f3dex2_build *b)
{
    struct hle_task *t = f->t;
    unsigned op;
    int sw = f->b != NULL && f->b != b;                 /* another build than the loop was working with (the first call and a reload of the same text have none) */

    (void)sw;                                           /* (only the mutation checks use it) */
    f->b = b;
    if (MUT(252, 1, !sw))
        f2L = b->lay ? b->lay : &l_lay_std;
    if (MUT(256, 1, !sw))
        f->ho_pc = b->enddl;
    if (MUT(257, 1, !sw)) {
        f->ov = overlay_match(t, DESC_CLIP) ? OV_CLIP : overlay_match(t, DESC_LIGHT) ? OV_LIGHT : OV_UNKNOWN;
        f->ov0 = f->ov;
    }
    if (MUT(253, 1, !sw))
        for (op = 0; op < 256; op++) {
            unsigned ea = (JUMP_TABLE + 2 * (int)(int8_t)op) & 0xfff;
            /* (S2DEX's display list buffer lies behind its jump table: the entries of the undefined opcodes 0x54 .. 0x7f are bytes of the display list, which the microcode reads when the command comes) */
            f->kind[op] = MUT(310, ea + 1 >= F2_IN && ea < F2_IN_END, 0) ? (uint8_t)K_BAD : (uint8_t)classify_op(b, op, hle_dm16(t, ea) & 0xfff);
        }
}

static int f3dex2_run(struct hle_task *t, const struct hle_ucode *u)
{
    const struct f3dex2_build *b = (const struct f3dex2_build *)u->fam_data;
    struct hle_cpu c;
    struct f3 f;
    int handover = 0, rc;

    if (g_prof < 0) {
        g_prof = getenv("HLE_F3_PROF") != NULL;
        if (g_prof)
            atexit(prof_dump);
    }
    if (!t->dx && !t->start_pc && load_known(t))
        return hle_decline(t, "an earlier task with these pointers loaded another ucode");
    f2L = b->lay ? b->lay : &l_lay_std;
    hle_cpu_init(t, &c, t->start_pc, 1u << 22);
    c.stop_pc = b->head + 4;                            /* the first fetch has polled SP_STATUS already */
    rc = t->start_pc ? f3_resume(t, b, &c) : hle_rsp_run(t, &c);
    if (rc != HLE_STOP)
        return rc;                                      /* the task ended before the loop, or declined (a call that continues a task: ended at the cut again, or yielded) */
    if (*t->cr[4] & SIG0) {                             /* a yield request at the first fetch: the microcode's own code from the loop head, interpreted to its BREAK */
        c.stop_pc = 0x1000;
        return hle_rsp_run(t, &c);
    }
    memset(&f, 0, sizeof(f));
    f.t = t;
    f.k0 = c.r[HLE_REG_K0];
    f.k1 = c.r[HLE_REG_K1];
    f.s6 = c.r[HLE_REG_S6];
    f.s7 = c.r[HLE_REG_S7];
    f.polls = 1;
    {   /* (the development switches, read once: a getenv per call was 1.7 % of the emulation thread in Gauntlet Legends) */
        static int dev_init, dev_trace, dev_watch, dev_dump;
        if (!dev_init) {
            dev_init = 1;
            dev_trace = getenv("HLE_F3_TRACE") != NULL;
            dev_watch = getenv("HLE_F3_WATCH") ? (int)strtoul(getenv("HLE_F3_WATCH"), NULL, 16) : -1;
            dev_dump = getenv("HLE_DUMP_IMEM") != NULL;
        }
        f.trace = dev_trace;
        f.watch = dev_watch;
        f.dump_imem = dev_dump;
    }
    if (f.trace)
        fprintf(stderr, "START k0=%08x k1=%08x s6=%04x s7=%04x\n", f.k0, f.k1, f.s6, f.s7);
    f.wval = f.watch >= 0 ? hle_dm16(t, f.watch) : 0;
    f.total = 0;
    f.lean = !hle_exact();
    f.uc0 = hle_dm32(t, 0xfd0);
    f3_bind(&f, b);
    for (;;) {
        rc = run_loop(&f, &handover);
        if (rc != HLE_OK)
            return rc;
        if (f.dump_imem) {                              /* (one file per hand-over: <file>, <file>.1, ...) */
            static unsigned dumps;
            char nm[512];
            FILE *fp;
            snprintf(nm, sizeof(nm), "%s.%u", getenv("HLE_DUMP_IMEM"), dumps);
            dumps++;
            fp = fopen(nm, "wb");
            if (fp) {
                fwrite(t->imem, 4, 1024, fp);
                fclose(fp);
            }
        }
        if (f.lean && f.ov != f.ov0 && f.ov != OV_UNKNOWN) {   /* the overlay the microcode would have left in IMEM */
            unsigned desc = f.ov == OV_LIGHT ? DESC_LIGHT : DESC_CLIP;
            hle_dma_read(t, hle_dm16(t, desc + 6), hle_dm32(t, desc), hle_dm16(t, desc + 4));
        }
        /* the outermost G_ENDDL (or a G_LOAD_UCODE): the microcode's own code. The registers as the loop head leaves them for the handler. */
        c.r[HLE_REG_K0] = f.k0;
        c.r[HLE_REG_K1] = f.k1;
        c.r[HLE_REG_S6] = f.s6;
        c.r[HLE_REG_S7] = f.s7;
        c.r[HLE_REG_AT] = 0;
        c.r[HLE_REG_T9] = f.w0;
        c.r[HLE_REG_T8] = f.w1;
        c.r[HLE_REG_T4] = (uint32_t)((int32_t)f.w0 >> 24);
        c.r[HLE_REG_T3] = f.ho_pc;
        c.pc = f.ho_pc;
        if (f.ho_pc != 0xfac)
            break;
        /* G_LOAD_UCODE: the stub of the microcode that is loaded now (overlay 0, IMEM 0) flushes the RDP buffer, saves the position in the display list (OSTask.data_ptr, DMEM 0xff0) and the text address
         * (0xfd0), copies 3912 bytes of text to IMEM 0x80 and the data segment (G_RDPHALF_1's word, 0xd8, the size in w0) to DMEM 0x180 .. , and starts the new text at 0x80, which runs its initialisation
         * (including the overlay descriptors: text address added) up to its first command fetch. What the microcode carries across is what that code does not write: DMEM below 0x180 (the matrices and
         * viewport of F3DEX2, the RDP state words at 0xc0 .. 0xd8, the FIFO pointer, the segment table, the display list stack and its depth, OSTask) and above the data it loads, IMEM outside the text, the
         * SP / DPC registers and RDRAM; the registers k0 / k1 / s6 / s7 are the new microcode's own, from its initialisation. The loop then takes the new build over (f3_bind). */
        c.stop_pc = MUT(255, f.nb->head, f.b->head) + 4;   /* the reloaded microcode starts again: run its initialisation up to the first fetch */
        rc = hle_rsp_run(t, &c);
        if (rc != HLE_STOP)
            return rc;
        if (*t->cr[4] & SIG0) {
            c.stop_pc = 0x1000;
            return hle_rsp_run(t, &c);
        }
        handover = 0;
        f.k0 = c.r[HLE_REG_K0];
        f.k1 = c.r[HLE_REG_K1];
        f.s6 = c.r[HLE_REG_S6];
        f.s7 = c.r[HLE_REG_S7];
        f.polls += MUT(249, 1, -(int)f.polls + 1);      /* (the count of SP_STATUS polls is the task's, not the build's: the JIT halts at 32767 of them whichever microcode makes them; the reloaded one polls once more before the first command) */
        if (f.nb != f.b) {
            t->switched = 1;
            hle_bind(t, f.nu);                          /* (the loaded build's gate, or its counting in a replay that validates it) */
            if (MUT(254, 1, 0))
                f.uc0 = f.nsrc;
        }
        f3_bind(&f, f.nb);
    }
    c.stop_pc = 0x1000;

    if (g_prof) {
        static unsigned runs;
        if (++runs % 400 == 0)
            prof_dump();
    }
    rc = hle_rsp_run(t, &c);
    if (rc == HLE_OK && f.cut) {
        /* the call ended at the cut, in the middle of the task. What the lean mode does not do for the flushes (the DMA write registers) and what is live though declared dead at the end of a task:
         * the display list input buffer and the RDP output not flushed yet */
        if (f.lean && f.total)
            *t->cr[3] = MUT(484, 0xff8, 0x1234);
        hle_cut_live(t, F2_IN, F2_IN_END, c.r[HLE_REG_S6] - OUT_BYTES, c.r[HLE_REG_S7], F3_CUT_SREGS);
        if (MUT(485, 0, 1))
            c.r[HLE_REG_S7] += 8;
        if (MUT(486, 0, 1))
            hle_st32(t, f2L->viewport_scale, hle_dm32(t, f2L->viewport_scale) ^ 0x10000);
    }
    return rc;
}

const struct hle_family f3dex2_fam = { .name = "f3dex2", .run = f3dex2_run, .wait_pc = f3_wait_pc, .idle = f3_idle, .yield_ok = 1 };

static const struct f3dex2_build l_b_dk64 = {
    .head = 0x18c, .copy = 0x208, .sync = 0x20c, .segcopy = 0x218, .ef = 0x238, .ed = 0x244, .tri1 = 0x7b4, .tri2 = 0x7ac, .vtx = 0x4f0,
    .mtx = 0x0f0, .geom = 0x1d0, .enddl = 0x1e4, .half2 = 0x1f8, .modvtx = 0xc78, .culldl = 0xc38, .branchz = 0xc60, .reject_mask = 0x7070, .load_key = 0x4d947651fb1d5494ull };
static const struct f3dex2_build l_b_oot = {
    .head = 0x18c, .copy = 0x208, .sync = 0x20c, .segcopy = 0x218, .ef = 0x238, .ed = 0x244, .tri1 = 0x7b4, .tri2 = 0x7ac, .vtx = 0x4f0,
    .mtx = 0x0f0, .geom = 0x1d0, .enddl = 0x1e4, .half2 = 0x1f8, .modvtx = 0xc74, .culldl = 0xc34, .branchz = 0xc5c, .zex = 1, .reject_mask = MUT(245, 0x70b0, 0x7030), .load_key = 0x70184057549a5640ull };
static const struct f3dex2_build l_b_pm = {
    .head = 0x194, .copy = 0x210, .sync = 0x214, .segcopy = 0x220, .ef = 0x240, .ed = 0x24c, .tri1 = 0xa44, .tri2 = 0xa3c, .vtx = 0x780,
    .mtx = 0x0f0, .geom = 0x1d8, .enddl = 0x1ec, .half2 = 0x200, .modvtx = 0xf08, .culldl = 0xec8, .branchz = 0xef0, .reject_mask = 0x7070, .load_key = 0xf3e64981ab1bc14bull };
static const struct f3dex2_build l_b_snap = {
    .head = 0x194, .copy = 0x228, .sync = 0x22c, .segcopy = 0x238, .ef = 0x258, .ed = 0x264, .tri1 = 0x7d4, .tri2 = 0x7cc, .vtx = 0x510,
    .mtx = 0x0f0, .geom = 0x1f0, .enddl = 0x204, .half2 = 0x218, .modvtx = 0xc98, .culldl = 0xc58, .branchz = 0xc80, .mvp = 0x1c0, .reject_mask = MUT(245, 0x70b0, 0x7030), .load_key = 0x1721044c22f4f0ecull };

/* DMEM dead at the end of a task (docs/design.md, "The F3DEX2 table of dead ranges"; docs/testing.md has the evidence): the display list input buffer (0x920 .. 0x9c8: the microcode's last refill;
 * a lean run reads the list from RDRAM), the clipper's polygon lists (0x3ca .. 0x408) and new vertices (12 records from 0x9c8; the clipper makes them anew for every triangle), the two RDP output buffers
 * with their slack (0xba8..0xd00 + 0xb0 slack, 0xdb0..0xf08 + 0xb0): the command handlers write the commands there, the flush DMAs [buffer start, write pointer) to the FIFO, and the bytes beyond the
 * pointer are scratch of the triangle setup (it writes its shade, texture and z blocks whether or not the command has them; their values depend on vector registers left by earlier commands) and the
 * stale second words of sync commands, which the RDP ignores (the verify stream zeroes them). At the end of a task everything below the pointer has been flushed. */
/* the ranges: F3DEX2_DEAD_RANGES (hle_f3dex2_int.h, which a dialect with a few more reuses). The word at 0xfc8 (OSTask.ucode_boot): the initialisation stores register a0 there (sw a0, 0xfc8), which holds
 * whatever the previous task left in it; nothing reads it. The padding byte of each colour of the light slots (struct Light_t: col[3] pad, colc[3] pad, dir[3] pad; 24 bytes from 0x200): the lighting code
 * stores whole vector lanes and the pad lane holds what the vector registers had, which depends on the commands before; nothing reads the pads. */
const struct hle_dead_range f3dex2_dead_out[] = { F3DEX2_DEAD_RANGES };

_Static_assert(sizeof(f3dex2_dead_out) / sizeof(f3dex2_dead_out[0]) == F3DEX2_N_DEAD_OUT, "F3DEX2_N_DEAD_OUT (hle_f3dex2_int.h)");
HLE_UCODE(l_dk64) = { .id = 0x48c7e653a6adcda8ull, .name = "F3DEX 2.07 (Donkey Kong 64)", .fam = &f3dex2_fam, .flags = HLE_F_NEAR_CLIP, .fam_data = &l_b_dk64, F3DEX2_DEAD };
HLE_UCODE(l_snap) = { .id = 0xd576cfcfed9d9c18ull, .name = "F3DEX.NoN 2.08H (Pokemon Snap)", .fam = &f3dex2_fam, .fam_data = &l_b_snap, F3DEX2_DEAD };
HLE_UCODE(l_pm) = { .id = 0xc734403d56034719ull, .name = "F3DEX 2.08 (Paper Mario)", .fam = &f3dex2_fam, .flags = HLE_F_NEAR_CLIP, .fam_data = &l_b_pm, F3DEX2_DEAD };
HLE_UCODE(l_oot) = { .id = 0x0d3f849c8dd309c4ull, .name = "F3DZEX.NoN 2.06H (Ocarina of Time)", .fam = &f3dex2_fam, .flags = HLE_F_ZEX, .fam_data = &l_b_oot, F3DEX2_DEAD };

#endif
