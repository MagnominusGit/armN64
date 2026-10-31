/*
 * arMLEt - f3d.c: the Fast3D family (F3D 2.0D / 2.0H) on the task-level native path: a display-list interpreter in C
 *
 * A whole fresh task is run natively: the boot code and the microcode's initialisation (OS code, no vector unit worth replacing) on the RSP interpreter up to the command loop, then the display
 * list by C code over the struct-level functions of f3d_fx.h, then the end of the task. The microcode's memory layout is kept (DMEM-resident state, docs/ucode-f3d.md 5), so what the
 * CPU and the RDP can observe, and DMEM outside the build's dead ranges, equals what the microcode leaves. Anything the code does not implement declines the task (the JIT runs it).
 *
 * Everything that differs between the builds is the build record (struct f3d_build, f3d_bld.h): the DMEM layout, the handler pcs, the masks and a few attributes; the opcodes reach their command
 * through the build's own handler tables (in DMEM). The registered records are data below; dbuild.c makes the same struct from a derived record (docs/ucode-derive.md).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hle.h"
#include "hle_internal.h"
#include "f3d_fx.h"
#include "f3d_dmem.h"
#include "f3d_bld.h"

/* test builds only (scripts/build.sh native --tools --hmut=N, tools/hle/mutcheck_fuzz.sh): a deliberate error N, to show that the tests catch it (docs/testing.md) */
#ifdef ARMLET_HLE_MUTATE
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#define MUT(n, ok, bad) (ok)
#endif

#ifdef ARMLET_NATIVE

/* F3D_PROF=1: where the time of a task goes (printed at exit), for development */
#include <time.h>
enum { PR_BOOT, PR_VTX, PR_TRI, PR_FLUSH, PR_MTX, PR_OTHER, PR_N };
static struct { double ns[PR_N]; unsigned long n[PR_N], steps, runs; int on, ops; } g_pr;
static inline double pr_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e9 + ts.tv_nsec; }
static void pr_dump(void)
{
    static const char *nm[PR_N] = { "boot+init (interpreter)", "G_VTX", "G_TRI1 (setup, clip)", "RDP flush (DMA, DPC)", "G_MTX/POPMTX", "other commands" };
    int i;
    for (i = 0; i < PR_N; i++)
        fprintf(stderr, "F3D_PROF %-26s %10.1f us  (%lu calls)\n", nm[i], g_pr.ns[i] / 1000.0, g_pr.n[i]);
    fprintf(stderr, "F3D_PROF interpreter steps %lu\n", g_pr.steps);
}
#define PR_BEGIN() double pr_t0 = g_pr.on ? pr_now() : 0
#define PR_END(k) do { if (g_pr.on) { g_pr.ns[k] += pr_now() - pr_t0; g_pr.n[k]++; } } while (0)

/* ---- the builds: struct f3d_build (f3d_bld.h) ---------------------------------------------------------------------------------------------------------------------------------- */
enum { OV_TEXT = 0, OV_CLIP = 1, OV_LIGHT = 2 };   /* what is in the overlay window: the text image (stubs that load the others), the clipper, the lighting code */

/* the command kinds: what a handler pc of the build's tables is (classify) */
enum { K3_BAD = 0, K3_NOP0, K3_NOP, K3_MTX, K3_MOVEMEM, K3_VTX, K3_DL, K3_TRI1, K3_QUAD, K3_HALF2, K3_HALF1, K3_PERSP, K3_CLRGEOM, K3_SETGEOM, K3_ENDDL, K3_OML, K3_OMH, K3_TEXTURE,
       K3_MOVEWORD, K3_POPMTX, K3_CULLDL, K3_TRI4 };

/* the per-task state */
struct f3d_task {
    struct hle_task *t;
    const struct f3d_build *b;
    const struct f3d_layout *L;
    uint8_t kind[256];
    unsigned geom_end;                  /* the end of the dead range that starts at the geometry buffer (the operand of a class-0 command may not run past it: the native code does not stage it) */
    uint32_t k0;                        /* RDRAM address of the next command */
    int gp;                             /* bytes left in the command buffer */
    uint32_t W, S, E;                   /* the RDP FIFO: write pointer, start, end */
    uint32_t s7;                        /* xbus: the ring write pointer (the microcode's s7) */
    int ov;
    unsigned ncmds;
    int trace;
    int exact;                          /* issue the DMA reads of the microcode (the SP register writes, DMEM staging) */
    unsigned pend;                      /* bytes of RDP commands collected in the output buffer, not yet written to the FIFO (W is where they go) */
    struct f3d_vctx ctx;                /* the vertex state, valid while ctx_ok (any command that changes it clears the flag) */
    int ctx_ok, ctx_bad;
    struct f3d_vtx vb[32];              /* the vertex records G_VTX made (DMEM has them too), valid per slot in vb_ok */
    const char *bad;                    /* a reason to decline found where the function could not return one (checked at the head of the loop) */
    uint32_t vb_ok;
    uint32_t phantom;                   /* slots the microcode's pair store overwrote with the flags of a vertex it computed from stale input (the partner of an odd count): their flags are not the record's */
};

static inline uint32_t seg(const struct f3d_task *f, uint32_t x) { return hle_dm32(f->t, f->L->segs + 4 * ((x >> 24) & 0xf)) + (x & 0xffffff); }

static inline uint32_t dm32u(struct hle_task *t, unsigned a)    /* unaligned big-endian word */
{
    return (uint32_t)hle_dm8(t, a) << 24 | (uint32_t)hle_dm8(t, a + 1) << 16 | (uint32_t)hle_dm8(t, a + 2) << 8 | hle_dm8(t, a + 3);
}
static inline void st32u(struct hle_task *t, unsigned a, uint32_t v)
{
    hle_st8(t, a, (uint8_t)(v >> 24)); hle_st8(t, a + 1, (uint8_t)(v >> 16)); hle_st8(t, a + 2, (uint8_t)(v >> 8)); hle_st8(t, a + 3, (uint8_t)v);
}


/* ---- the RDP FIFO ----------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* The microcode flushes every command on its own (a DMA write and a DPC_END each). What is observable afterwards is the bytes in the FIFO, the pointers and the lists the video plugin takes: the
 * commands are collected in the output buffer and written with one DMA and one DPC_END, with the wrap and wait rules of the microcode applied where a command starts a batch. */
static int flush_pending(struct f3d_task *f)
{
    struct hle_task *t = f->t;
    if (!f->pend)
        return HLE_OK;
    if (f->exact)
        hle_dma_write(t, f->L->outbuf, f->W, f->pend - 1);
    else
        hle_rdram_write(t, f->W, t->dmem + (f->L->outbuf >> 2), f->pend);
    f->W += f->pend;
    f->pend = 0;
    hle_st32(t, f->L->rdp_wp, f->W);
    hle_mtc0(t, 9, f->W);
    return HLE_OK;
}

/* room for a command of n bytes: returns its DMEM address in the output buffer, or 0 (the task is declined) */
static unsigned begin_cmd(struct f3d_task *f, unsigned n)
{
    struct hle_task *t = f->t;
    if (f->pend && (f->W + f->pend + n > f->E || f->pend + n > 0xfc0u - f->L->outbuf)) {
        flush_pending(f);
    }
    if (!f->pend) {
        uint32_t c;
        if (f->W + n > f->E) {                              /* does not fit: wrap to the start of the FIFO */
            if (*t->cr[11] & 0x400) { hle_decline(t, "DPC start valid while wrapping the FIFO"); return 0; }
            if (*t->cr[10] == f->S) { hle_decline(t, "RDP at the start of the FIFO while wrapping"); return 0; }
            hle_mtc0(t, 8, f->S);
            f->W = f->S;
        }
        c = *t->cr[10];
        if (!(f->W >= c) && !(f->W + n < c)) { hle_decline(t, "FIFO full (the RDP would have to catch up)"); return 0; }
    }
    return f->L->outbuf + f->pend;
}

/* The XBUS builds: a command is written at the ring pointer s7 in DMEM and DPC_END is set to the new s7 on its own. Before it the room check of the microcode (0x178 with the length of the command: 176 for a
 * triangle, 8 otherwise): when s7 + length passes ring_end (0xde0) the ring wraps to outbuf (0x9e0; DPC_START = outbuf), after waiting for START_VALID to clear and for the RDP not to stand at outbuf. The wait for the RDP to be
 * out of the way of the new bytes cannot trigger while the RDP has taken every list (CURRENT = the last END = s7). */
static int xbus_emit(struct f3d_task *f, const uint8_t *bytes, unsigned n, unsigned room)
{
    struct hle_task *t = f->t;
    unsigned i;
    if (f->s7 + room > f->L->ring_end) {
        if (*t->cr[11] & 0x400) return hle_decline(t, "XBUS ring: start valid while wrapping");
        if (*t->cr[10] == f->L->outbuf) return hle_decline(t, "XBUS ring: RDP at the start while wrapping");
        f->s7 = f->L->outbuf;
        hle_mtc0(t, 8, f->L->outbuf);
    }
    if (*t->cr[10] < f->s7 || *t->cr[10] >= f->s7 + room) { /* the microcode would test CURRENT against s7 .. s7 + length: only CURRENT == s7 (taken everything) is reproduced */
        if (*t->cr[10] != f->s7) return hle_decline(t, "XBUS ring: the RDP is not caught up");
    }
    for (i = 0; i < n; i++)
        hle_st8(t, f->s7 + i, bytes[i]);
    f->s7 += n;
    hle_mtc0(t, 9, f->s7);
    return HLE_OK;
}

static int emit8(struct f3d_task *f, uint32_t w0, uint32_t w1)
{
    unsigned a;
    if (f->b->xbus) {
        uint8_t b[8] = { (uint8_t)(w0 >> 24), (uint8_t)(w0 >> 16), (uint8_t)(w0 >> 8), (uint8_t)w0, (uint8_t)(w1 >> 24), (uint8_t)(w1 >> 16), (uint8_t)(w1 >> 8), (uint8_t)w1 };
        return xbus_emit(f, b, 8, 8);
    }
    a = begin_cmd(f, 8);
    if (!a)
        return HLE_DECLINE;
    hle_st32(f->t, a, w0);
    hle_st32(f->t, a + 4, w1);
    f->pend += 8;
    return f->exact ? flush_pending(f) : HLE_OK;       /* exact events: every command is a list of its own, like the microcode's */
}

/* the bytes of an RDP triangle (big-endian) into the output buffer */
static int emit_bytes(struct f3d_task *f, const uint8_t *cmd, unsigned len)
{
    if (f->b->xbus)
        return xbus_emit(f, cmd, len, 176);
    unsigned a = begin_cmd(f, len), i;
    if (!a)
        return HLE_DECLINE;
    for (i = 0; i < len; i += 4) {
        uint32_t w;
        memcpy(&w, cmd + i, 4);
        hle_st32(f->t, a + i, __builtin_bswap32(w));
    }
    f->pend += len;
    return f->exact ? flush_pending(f) : HLE_OK;
}

/* ---- overlays --------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
static void load_overlay(struct f3d_task *f, int k)             /* the descriptor at DMEM 8 + 8 k: dram, ((len - 1) << 16) | imem */
{
    struct hle_task *t = f->t;
    unsigned d = 8 + 8 * k;
    hle_dma_read(t, hle_dm16(t, d + 6), hle_dm32(t, d), hle_dm16(t, d + 4));
}

/* ---- the commands ------------------------------------------------------------------------------------------------------------------------------------------------------------------ */
static void mtx_recompute(struct hle_task *t, const struct f3d_layout *lay)
{
    struct f3d_mtx mv, pj, mp;
    f3d_load_mtx(&mv, t->dmem, lay->mv, 0);
    f3d_load_mtx(&pj, t->dmem, lay->mv + 0x40, 0);
    f3d_mtx_mul(&mv, &pj, &mp);
    f3d_store_mtx(&mp, t->dmem, lay->mp);
}

/* the source of a DMA read the microcode makes into the geometry buffer: with `exact` the DMA is issued and the data read from DMEM, else from RDRAM */
struct src { const uint32_t *img; unsigned base; int ram; };
/* The RDP output ring is written by the task's own flushes and the compare treats the bytes the RDP ignores as don't care: a display list, vertices, a matrix or a light read from inside it would read those
 * bytes. Not a game's: decline (docs/design.md). */
static inline int ring_hit(const struct f3d_task *f, uint32_t a, uint32_t n)
{
    a &= 0x7ffff8;
    return a < (f->E & 0xffffff) && a + n > (f->S & 0x7fffff);
}
static struct src fetch(struct f3d_task *f, uint32_t dram, uint32_t lenreg)
{
    struct src s;
    if (MUT(243, ring_hit(f, dram, ((lenreg & 0x1ff) | 7) + 1), 0))
        f->bad = "a read from inside the RDP output ring";
    if (f->exact) {
        hle_dma_read(f->t, f->L->geombuf, dram, lenreg);
        s.img = f->t->dmem; s.base = f->L->geombuf; s.ram = 0;
    } else {
        s.img = (const uint32_t *)f->t->rdram; s.base = dram & 0x7ffff8; s.ram = 1;
    }
    return s;
}

static int cmd_mtx(struct f3d_task *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    const struct f3d_layout *lay = f->L;
    unsigned pp = (w0 >> 16) & 0xff;
    struct src s;
    struct f3d_mtx N, T, R;
    if (MUT(229, (((w0 & 0x1ff) + 1 + 7) & ~7u) < 64, 0))   /* a short DMA leaves the rest of the matrix as the geometry buffer was (which the front-end does not keep in DMEM) */
        return hle_decline(t, "G_MTX operand shorter than a matrix");
    s = fetch(f, seg(f, w1), w0 & 0x1ff);
    unsigned tbase = (pp & 1) ? lay->mv + 0x40 : lay->mv;
    f3d_load_mtx(&N, s.img, s.base, s.ram);
    hle_st8(t, lay->lights, 0xff);
    if (!(pp & 1) && (pp & 4) && hle_dm32(t, lay->mtx_sp) != hle_dm32(t, lay->mtx_lim)) {
        flush_pending(f);
        if (MUT(244, ring_hit(f, hle_dm32(t, lay->mtx_sp), (((w0 & 0x1ff) | 7) + 1)), 0))       /* (the output is staged and written later: a write into the ring now would be ordered differently from the microcode's) */
            return hle_decline(t, "matrix push into the RDP output ring");
        hle_dma_write(t, lay->mv, hle_dm32(t, lay->mtx_sp), w0 & 0x1ff);
        hle_st32(t, lay->mtx_sp, hle_dm32(t, lay->mtx_sp) + 64);
    }
    if (pp & 2) {
        R = N;
    } else {
        f3d_load_mtx(&T, t->dmem, tbase, 0);
        f3d_mtx_mul(&N, &T, &R);
    }
    f3d_store_mtx(&R, t->dmem, tbase);
    mtx_recompute(t, lay);
    return HLE_OK;
}

static int cmd_popmtx(struct f3d_task *f)
{
    struct hle_task *t = f->t;
    const struct f3d_layout *lay = f->L;
    uint32_t cur = hle_dm32(t, lay->mtx_sp);
    if (f->b->popmtx_stale)
        hle_st8(t, lay->lights, 0xff);
    if (cur > hle_dm32(t, lay->mtx_lim) - 640) {
        unsigned i;
        if (f->exact)
            hle_dma_read(t, lay->mv, cur - 64, 63);
        else
            for (i = 0; i < 64; i += 4)
                hle_st32(t, lay->mv + i, hle_rd32(t, ((cur - 64 + i) & 0x7ffff8) | ((cur - 64 + i) & 7)));
        hle_st32(t, lay->mtx_sp, cur - 64);
        mtx_recompute(t, lay);
    }
    return HLE_OK;
}

static int cmd_movemem(struct f3d_task *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    unsigned ii = (w0 >> 16) & 0xff, dest, n, i;
    struct src s = fetch(f, seg(f, w1), w0 & 0x1ff);
    if (ii < 0x80 || ii > 0xa0 || (ii & 1))
        return hle_decline(t, "G_MOVEMEM index outside the table");
    dest = hle_dm16(t, f->L->mm_tab + (ii - 0x80));
    n = 16 - (dest & 15);                               /* LQV + SQV: only up to the next 16-byte boundary */
    if (n > (((w0 & 0x1ff) + 1 + 7) & ~7u))
        return hle_decline(t, "G_MOVEMEM length shorter than the bytes stored");
    for (i = 0; i < n; i++) {
        unsigned a = s.base + i;
        hle_st8(t, dest + i, s.ram ? ((const uint8_t *)s.img)[a & 0x7fffff] : hle_dm8(t, a));
    }
    return HLE_OK;
}

static int cmd_vtx(struct f3d_task *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    const struct f3d_layout *lay = f->L;
    const unsigned nslots = f3d_nslots(lay);
    unsigned at = (w0 >> 16) & 0xff, a2 = at & 15, n = (at >> 4) + 1, i;
    if (f->b->vtx_wr) {
        n = (w0 >> 9) & 0x7f;
        a2 = at / 5;
        if (at % 5 || n < 1 || n > 32)
        return hle_decline(t, "G_VTX: encoding unusual");
    }
    const struct f3d_vctx *ctxp;
    struct f3d_vin in[32];
    struct f3d_vtx out[32];
    struct src s;
    uint32_t geom = hle_dm32(t, lay->geom);
    if ((((w0 & 0x1ff) + 1 + 7) & ~7u) < 16 * n || a2 + n > nslots)
    {
        return hle_decline(t, "G_VTX length or slot range unusual");
    }
    s = fetch(f, seg(f, w1), w0 & 0x1ff);               /* the class-0 DMA comes first */
    if ((geom & F3D_G_LIGHTING) && hle_cmd(t, LATCH_K_LIT))
        return HLE_DECLINE;
    if (geom & F3D_G_LIGHTING) {
        uint32_t l = hle_dm32(t, lay->lights);
        if (f->ov == OV_TEXT || f->ov == OV_CLIP) {
            load_overlay(f, 2);
            f->ov = OV_LIGHT;
        }
        if ((int32_t)l < 0) {                           /* the lights are stale: transform the directions */
            unsigned W = l & 0xfff, ns = W >> 5, k;
            int8_t raw[F3D_MAX_SLOTS + 1][3], res[F3D_MAX_SLOTS + 1][4];
            int16_t cl[3];
            struct f3d_mtx mv;
            if ((W & 31) || ns < 1 || ns > F3D_MAX_SLOTS)
                return hle_decline(t, "G_VTX: light count unusual");
            for (k = 0; k <= ns; k++)
                for (i = 0; i < 3; i++)
                    raw[k][i] = (int8_t)hle_dm8(t, lay->light + 0x18 + 32 * k + i);
            for (i = 0; i < 3; i++)
                cl[i] = (int16_t)hle_dm16(t, lay->light + 0x68 + 2 * i);
            f3d_load_mtx(&mv, t->dmem, lay->mv, 0);
            f3d_relight(&mv, raw, ns, cl, res);
            for (k = 0; k <= ns; k++)
                for (i = 0; i < 8; i++)
                    hle_st8(t, lay->light + 0x20 + 32 * k + i, (uint8_t)res[k][i & 3]);
            hle_st32(t, lay->lights, W);
            f->ctx_ok = 0;
        }
    }
    if (!f->ctx_ok) {
        f->ctx_bad = f3d_load_vctx(&f->ctx, t->dmem, lay);
        f->ctx_ok = 1;
    }
    if (f->ctx_bad)
        return hle_decline(t, "G_VTX: light state unusual");
    ctxp = &f->ctx;
    for (i = 0; i < n; i++)
        f3d_load_vin(&in[i], s.img, s.base + 16 * i, s.ram);
    if (f3d_vtx_xform(ctxp, in, n, out))
        return hle_decline(t, "G_VTX: state outside what the native code supports");
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
    f->phantom &= ~(((n < 32 ? (1u << n) : 0u) - 1u) << a2);
    if ((n & 1) && a2 + n < nslots && lay->flags_variant == F3D_FLAGS_SM64)    /* (Pilotwings' flag word is the second vertex's alone) */
        f->phantom |= 1u << (a2 + n);
    for (i = 0; i < n; i++) {
        unsigned a = lay->vtx_base + F3D_VSTRIDE * (a2 + i);
        f3d_store_vtx(&out[i], t->dmem, a);
        f->vb[a2 + i] = out[i];
        f->vb_ok |= 1u << (a2 + i);   /* nslots <= 32 (f3d_run) */
        if (i & 1)
            hle_st16(t, a + 38, out[i - 1].flags);           /* the pair's word store: the first vertex's flags land in the second one's padding */
    }
    return HLE_OK;
}

static int cmd_tri3(struct f3d_task *f, const unsigned idx[3], unsigned flag)
{
    struct hle_task *t = f->t;
    const struct f3d_layout *lay = f->L;
    const unsigned vend = lay->vtx_base + f3d_nslots(lay) * F3D_VSTRIDE;
    struct f3d_vtx v[3];
    struct f3d_tri_in ti;
    unsigned k;
    uint8_t cmd[0xc0];
    unsigned len = 0;
    int rc;
    for (k = 0; k < 3; k++) {
        unsigned a = lay->vtx_base + f->b->vptr_mul * idx[k];
        if (a < (unsigned)lay->vtx_base || a + 40 > vend || (a - lay->vtx_base) % F3D_VSTRIDE)
            return hle_decline(t, "G_TRI1: vertex pointer outside the buffer");
        {
            unsigned slot = (a - lay->vtx_base) / F3D_VSTRIDE;
            if (MUT(242, f->phantom >> slot & 1, 0))
                return hle_decline(t, "triangle with the partner slot of an odd G_VTX");
            if (!(f->vb_ok >> slot & 1)) {
                f3d_load_vtx(&f->vb[slot], t->dmem, a);
                f->vb_ok |= 1u << slot;
            }
            v[k] = f->vb[slot];
        }
        ti.v[k] = &v[k];
    }
    ti.geom = hle_dm32(t, lay->geom);
    ti.variant = f->b->tri_wr ? F3D_TRIV_WR : F3D_TRIV_SM64;
    if (flag > 2 && !(ti.geom & 0x200) && !f->b->tri_wr)
        return hle_decline(t, "G_TRI1: flat-shading selector outside 0..2");   /* the microcode reads a leftover word at 0xde0 + 4 flag; with smooth shading the selector is not used */
    ti.flat = &v[flag > 2 ? 0 : flag];
    ti.persp = hle_dm16(t, lay->persp);
    ti.tile_byte = hle_dm8(t, lay->texture + 2);
    ti.zclamp = hle_dm8(t, lay->texture + 1);
    ti.reject_mask = f->b->reject_mask;
    ti.clip_mask = f->b->clip_mask;
    ti.vco_carry = 0;
    ti.sort_given = MUT(228, 0, 1);                     /* (Indiana Jones' sort by a table: the other fields of the struct are only read with it) */
    rc = f3d_tri_setup(&ti, cmd, &len);
    if (f->trace) {
        fprintf(stderr, "TRI3 rc=%d len=%u g=%08x idx %u %u %u flag %u:", rc, len, ti.geom, idx[0], idx[1], idx[2], flag);
        for (k = 0; k < 3; k++)
            fprintf(stderr, " [sx=%04x sy=%04x sz=%04x.%04x w=%04x.%04x fl=%04x s=%04x t=%04x col=%02x%02x%02x%02x]", (uint16_t)v[k].sx, (uint16_t)v[k].sy, (uint16_t)v[k].sz, v[k].szf, (uint16_t)v[k].rwi, v[k].rwf, v[k].flags, (uint16_t)v[k].s, (uint16_t)v[k].t, v[k].col[0], v[k].col[1], v[k].col[2], v[k].col[3]);
        fprintf(stderr, "\n");
    }
    if (rc == F3D_TRI_REJECT || rc == F3D_TRI_CULL)
        return hle_cmd(t, rc == F3D_TRI_REJECT ? LATCH_K_TRI_REJECT : LATCH_K_TRI_CULL) ? HLE_DECLINE : HLE_OK;
    if (rc == F3D_TRI_OK) {
        int r_ = emit_bytes(f, cmd, len);
        return r_;
    }
    {   /* clipping */
        struct f3d_clip_out co;
        if (hle_cmd(t, LATCH_K_CLIP))
            return HLE_DECLINE;
        if ((hle_wneg(v[0].cx[3], v[0].cf[3]) || hle_wneg(v[1].cx[3], v[1].cf[3]) || hle_wneg(v[2].cx[3], v[2].cf[3])) && hle_cmd(t, LATCH_K_NEARCLIP))
            return HLE_DECLINE;
        const struct f3d_vtx *vp[3] = { &v[0], &v[1], &v[2] };
        unsigned i;
        if (f->ov == OV_LIGHT || f->ov == OV_TEXT) {
            load_overlay(f, 1);
            f->ov = OV_CLIP;
        }
        if (!f->ctx_ok) {
            f->ctx_bad = f3d_load_vctx(&f->ctx, t->dmem, lay);
            f->ctx_ok = 1;
        }
        if (f->ctx_bad)
            return hle_decline(t, "G_TRI1: clipper with lights in an unusual state");
        f3d_clip_tri(&f->ctx, vp, &co);
        for (i = 0; i < co.ntri; i++) {
            struct f3d_tri_in tj = ti;
            unsigned l2 = 0;
            int r2;
            for (k = 0; k < 3; k++) tj.v[k] = co.tri[i][k];
            r2 = f3d_tri_setup_noclip(&tj, cmd, &l2);
            if (r2 == F3D_TRI_OK && emit_bytes(f, cmd, l2) != HLE_OK)
                return HLE_DECLINE;
        }
    }
    return HLE_OK;
}

static int cmd_tri(struct f3d_task *f, uint32_t w1)
{
    unsigned idx[3] = { (w1 >> 16) & 0xff, (w1 >> 8) & 0xff, w1 & 0xff };
    return cmd_tri3(f, idx, w1 >> 24);
}

/* GoldenEye's op 0xb1 (2.0G): up to four triangles, triangle k of vertex nibbles (w1 >> 8k) & 0xf, (w1 >> 8k + 4) & 0xf and (w0 >> 4k) & 0xf, each nibble through the byte table at lay.tri4_tab to
 * G_TRI1's vertex byte, flat-shading vertex 0. The microcode draws one through G_TRI1's code, writes the command back into its buffer with w1 >> 8 and the low half of w0 >> 4 and runs it again,
 * until w1 is 0 */
static int cmd_tri4(struct f3d_task *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    const struct f3d_layout *lay = f->L;
    if (!lay->tri4_tab || f->b->gp_end)
        return hle_decline(t, "G_TRI4 without its table");
    while (w1) {
        unsigned idx[3] = { hle_dm8(t, lay->tri4_tab + (w1 & 0xf)), hle_dm8(t, lay->tri4_tab + ((w1 >> 4) & 0xf)), hle_dm8(t, lay->tri4_tab + (w0 & 0xf)) };
        PR_BEGIN();
        int r = cmd_tri3(f, idx, 0);
        PR_END(PR_TRI);
        if (r != HLE_OK)
            return HLE_DECLINE;
        w1 >>= 8;
        w0 = (w0 & 0xffff0000u) | ((w0 >> 4) & 0xffff);
        if (f->exact) {                                 /* (the command written back into the buffer, as the microcode leaves it) */
            unsigned a = f->L->cmdbuf + 320 - (unsigned)f->gp - 8;
            hle_st16(t, a + 2, (uint16_t)w0);
            hle_st32(t, a + 4, w1);
        }
    }
    return HLE_OK;
}

/* Wave Race's op 0xb5: the quad (D = w1 byte 3, then A, B, C in w1 bytes 2 .. 0 of the microcode's reading order): the diagonal is chosen by the screen y of the corners, then two triangles */
static int cmd_quad(struct f3d_task *f, uint32_t w1)
{
    struct hle_task *t = f->t;
    const struct f3d_layout *lay = f->L;
    const unsigned vend = lay->vtx_base + f3d_nslots(lay) * F3D_VSTRIDE;
    unsigned q[4] = { (w1 >> 16) & 0xff, (w1 >> 8) & 0xff, w1 & 0xff, (w1 >> 24) & 0xff };   /* at, v0, v1, a0 */
    int y[4], k;
    for (k = 0; k < 4; k++) {
        unsigned a = lay->vtx_base + f->b->vptr_mul * q[k];
        if (a < (unsigned)lay->vtx_base || a + 40 > vend || (a - lay->vtx_base) % F3D_VSTRIDE)
            return hle_decline(t, "quad: vertex pointer outside the buffer");
        y[k] = (int16_t)hle_dm16(t, a + 26);
    }
    {
        int a1 = y[0] - y[2], a2 = y[1] - y[3];
        if (a1 <= 0) a1 = -a1;
        if (a2 <= 0) a2 = -a2;
        if (a2 - a1 <= 0) {                             /* the diagonal the other way round: (at, v0, v1, a0) = (v0, v1, a0, at) */
            unsigned at = q[0];
            q[0] = q[1]; q[1] = q[2]; q[2] = q[3]; q[3] = at;
        }
    }
    {
        unsigned t1[3] = { q[0], q[1], q[2] }, t2[3] = { q[2], q[3], q[0] };
        if (cmd_tri3(f, t1, 0) != HLE_OK)
            return HLE_DECLINE;
        return cmd_tri3(f, t2, 0);
    }
}

/* the end of the display list: pops a call, or ends the task. Returns 1 when the task ended. */
static int enddl(struct f3d_task *f)
{
    struct hle_task *t = f->t;
    int d = (int8_t)hle_dm8(t, f->L->depth) - 4;
    if (d < 0)
        return 1;
    f->k0 = dm32u(t, f->L->retstack + d);
    hle_st8(t, f->L->depth, (uint8_t)d);
    f->gp = 0;
    return 0;
}

/* The kind of a handler pc of the build's tables. Class 0 (0x00 .. 0x3f) DMAs the operand first, class 2 (0x80 .. 0xbf) does not: a kind that belongs to the other class is not a command (BAD). */
static unsigned classify(const struct f3d_build *b, int cls0, unsigned h)
{
    static const uint8_t kind_of_pc[F3_NPC] = { K3_BAD, K3_NOP, K3_BAD, K3_MTX, K3_MOVEMEM, K3_VTX, K3_DL, K3_TRI1, K3_QUAD, K3_HALF2, K3_HALF2, K3_HALF1, K3_PERSP, K3_CLRGEOM, K3_SETGEOM,
                                                 K3_ENDDL, K3_OML, K3_OMH, K3_TEXTURE, K3_MOVEWORD, K3_POPMTX, K3_CULLDL, K3_TRI4 };
    unsigned k;
    if (h == b->pc[F3PC_nop])
        return cls0 ? K3_NOP0 : K3_NOP;
    for (k = F3PC_mtx; k < F3_NPC; k++)
        if (b->pc[k] && h == b->pc[k]) {
            unsigned kd = kind_of_pc[k];
            int c0 = kd == K3_MTX || kd == K3_MOVEMEM || kd == K3_VTX || kd == K3_DL;
            return c0 == cls0 ? kd : K3_BAD;
        }
    return K3_BAD;
}

/* the opcodes of the GBI (class 0: 0x00 .. 0x09 at jt0 + 2 op, class 2: 0xb0 .. 0xbf at jt2 + 2 (op - 0x80): docs/ucode-f3d.md 2.3) through the build's handler tables in DMEM; the other cells of the
 * overlapping tables are not handler pointers (0x0a .. 0x18 run into the class-2 table, the rest are data), and the other opcodes no game issues: BAD */
static void make_kinds(struct f3d_task *f)
{
    unsigned op;
    memset(f->kind, K3_BAD, sizeof(f->kind));
    for (op = 0x00; op <= 0x09; op++)
        f->kind[op] = (uint8_t)classify(f->b, 1, hle_dm16(f->t, f->L->jt0 + 2 * op) & 0xfff);
    for (op = 0xb0; op <= 0xbf; op++)
        f->kind[op] = (uint8_t)classify(f->b, 0, hle_dm16(f->t, f->L->jt2 + 2 * (op - 0x80)) & 0xfff);
}

/* the end of the dead range of the build's table that starts at (or covers) the geometry buffer: the operand DMA may run up to there (0: none, every class-0 command with an operand declines) */
static unsigned geom_dead_end(const struct hle_ucode *u, unsigned a)
{
    unsigned i, grew = 1;
    if (!u->dead)
        return 0;
    while (grew) {
        grew = 0;
        for (i = 0; i < u->n_dead; i++)
            if (u->dead[i].lo <= a && u->dead[i].hi > a) {
                a = u->dead[i].hi;
                grew = 1;
            }
    }
    return a;
}

static int f3d_run(struct hle_task *t, const struct hle_ucode *u)
{
    const struct f3d_build *b = (const struct f3d_build *)u->fam_data;
    const struct f3d_layout *L = b->lay;
    struct f3d_task f;
    struct hle_cpu c;
    int rc;
    unsigned opn[256] = { 0 };

    memset(&f, 0, sizeof(f));
    f.t = t;
    f.b = b;
    f.L = L;
    { static int tr = -1; if (tr < 0) tr = getenv("F3D_TRACE") != NULL; f.trace = tr; }
    f.exact = hle_exact();                              /* the exact-events mode: every DMA and every DPC write the microcode makes, in its order */
    t->fam = &f;
    if (hle_dm32(t, 0xfc4) & 1)
        return hle_decline(t, "yielded task");
    if (L->cmdbuf <= L->vtx_base || f3d_nslots(L) > 32)
        return hle_decline(t, "the build's vertex buffer is not 1 .. 32 slots");
    f.geom_end = geom_dead_end(u, L->geombuf);

    /* the boot code and the microcode's initialisation, on the interpreter, up to the command loop */
    if (!g_pr.on && getenv("F3D_PROF")) g_pr.on = 1;
    if (!g_pr.ops && getenv("F3D_OPS")) g_pr.ops = 1;
    {
        PR_BEGIN();
        hle_cpu_init(t, &c, t->start_pc, 1u << 16);
        c.stop_pc = b->pc[F3PC_head];
        rc = hle_rsp_run(t, &c);
        PR_END(PR_BOOT);
        g_pr.steps += c.steps;
    }
    if (rc == HLE_OK)
        return HLE_OK;                                  /* the boot code ended the task (a pending signal): exactly as the microcode */
    if (rc != HLE_STOP)
        return HLE_DECLINE;
    f.k0 = c.r[HLE_REG_K0];
    if (b->gp_end)
        f.k0 -= c.r[28] - c.r[27];                        /* the next block has been counted already: k0 is the address after the commands in the buffer */
    f.gp = b->gp_end ? (int)(c.r[28] - c.r[27]) : (int)c.r[28];
    f.W = hle_dm32(t, L->rdp_wp);
    f.S = hle_dm32(t, L->fifo);
    f.E = hle_dm32(t, L->fifo + 4);
    f.ov = OV_TEXT;
    f.s7 = c.r[23];
    if ((f.k0 & 7) || f.gp != 320)
        return hle_decline(t, "unusual state at the command loop");
    make_kinds(&f);

    for (;;) {
        uint32_t w0, w1;
        unsigned op, kd;
        if (f.bad)
            return hle_decline(t, f.bad);
        if (++f.ncmds > 30000)
            return hle_decline(t, "more commands than the status polls allow");
        w0 = hle_rd32(t, f.k0);
        w1 = hle_rd32(t, f.k0 + 4);
        f.k0 += 8;
        f.gp -= 8;
        op = w0 >> 24;
        if (hle_cmd(t, op))                             /* (the latch policy's command-kind gate; the count of a replay under validation) */
            return HLE_DECLINE;
        if (f.trace) fprintf(stderr, "F3D cmd %u @%06x: %08x %08x\n", f.ncmds, f.k0 - 8, w0, w1);
        if (g_pr.ops) opn[op]++;
        if (op >= 0xc0) {                               /* the RDP commands: copied, the segment address of the image commands resolved */
            if (op >= 0xfd)
                w1 = seg(&f, w1);
            if (emit8(&f, w0, w1) != HLE_OK)
                return HLE_DECLINE;
            goto next;
        }
        if (op >= 0x40 && op < 0x80)                    /* class 1: a no-op without a DMA */
            goto next;
        kd = f.kind[op];
        if (op < 0x40) {                                /* class 0: a geometry DMA is started for every one of them; then the handler */
            /* (not issued in the lean mode; the operand of a game is a matrix or at most 16 vertices (length field 0x100, 264 bytes), and the buffer's dead range reaches past that: a longer one leaves live DMEM different) */
            if (MUT(236, (((w0 & 0x1ff) | 7) + 1) > (unsigned)(f.geom_end - L->geombuf) || f.geom_end <= L->geombuf, 0))
                return hle_decline(t, "class-0 operand longer than the geometry buffer's dead range");
            if (kd == K3_BAD)
                return hle_decline(t, "unsupported class-0 opcode");
        } else if (kd == K3_BAD)
            return hle_decline(t, "unsupported class-2 opcode");
        switch (kd) {
        case K3_NOP0:
            if (f.exact)
                hle_dma_read(t, L->geombuf, seg(&f, w1), w0 & 0x1ff);
            break;
        case K3_NOP:
            break;
        case K3_MTX: { PR_BEGIN(); int r = cmd_mtx(&f, w0, w1); PR_END(PR_MTX); f.ctx_ok = 0; if (r != HLE_OK) return HLE_DECLINE; break; }
        case K3_MOVEMEM: if (cmd_movemem(&f, w0, w1) != HLE_OK) return HLE_DECLINE; f.ctx_ok = 0; f.vb_ok = 0; break;
        case K3_VTX: { PR_BEGIN(); int r = cmd_vtx(&f, w0, w1); PR_END(PR_VTX); if (r != HLE_OK) return HLE_DECLINE; break; }
        case K3_DL: {
            int d = (int8_t)hle_dm8(t, L->depth);
            if (f.exact)
                hle_dma_read(t, L->geombuf, seg(&f, w1), w0 & 0x1ff);
            if (((w0 >> 16) & 0xff) == 0) {                  /* a call */
                if (d > 36)
                    break;                                  /* the stack is full: ignored */
                st32u(t, L->retstack + d, f.k0);
                hle_st8(t, L->depth, (uint8_t)(d + 4));
            }
            f.k0 = seg(&f, w1);
            f.gp = 0;
            break;
        }
        case K3_QUAD:
            if (cmd_quad(&f, w1) != HLE_OK)
                return HLE_DECLINE;
            break;
        case K3_TRI4:
            if (cmd_tri4(&f, w0, w1) != HLE_OK)
                return HLE_DECLINE;
            break;
        case K3_HALF2:
            if (emit8(&f, hle_dm32(t, L->half1), w1) != HLE_OK)
                return HLE_DECLINE;
            break;
        case K3_HALF1: hle_st32(t, L->half1, w1); break;
        case K3_PERSP: hle_st16(t, L->persp, (uint16_t)w1); f.ctx_ok = 0; break;
        case K3_CLRGEOM: hle_st32(t, L->geom, hle_dm32(t, L->geom) & ~w1); f.ctx_ok = 0; break;
        case K3_SETGEOM: hle_st32(t, L->geom, hle_dm32(t, L->geom) | w1); f.ctx_ok = 0; break;
        case K3_ENDDL:
            if (enddl(&f))
                goto end;
            break;
        case K3_OML: case K3_OMH: {
            unsigned a = kd == K3_OML ? L->oml : L->omh;
            uint32_t m = (uint32_t)(((1u << (w0 & 31)) - 1) << ((w0 >> 8) & 31));
            hle_st32(t, a, (hle_dm32(t, a) & ~m) | w1);
            if (emit8(&f, hle_dm32(t, L->omh), hle_dm32(t, L->oml)) != HLE_OK)
                return HLE_DECLINE;
            break;
        }
        case K3_TEXTURE:
            hle_st32(t, L->texture, w0);
            hle_st32(t, L->texture + 4, w1);
            hle_st32(t, L->geom, (hle_dm32(t, L->geom) & ~2u) | ((w0 & 1) << 1));
            f.ctx_ok = 0;
            break;
        case K3_MOVEWORD: {
            unsigned ii = w0 & 0xff;
            unsigned off = (w0 >> 8) & 0xffff;
            if (ii > 0x10 || (ii & 1))
                return hle_decline(t, "G_MOVEWORD index outside the table");
            {
                /* a game's offsets are words within the group: segments 0 .. 0x3c, the light colours up to the last light, G_MW_POINTS
                 * (gSPModifyVertex: Cruis'n USA) any word of the vertex buffer */
                unsigned lim = ii == 0x0a ? 0x1fc : ii == 0x0c ? f3d_nslots(L) * F3D_VSTRIDE - 4 : 0x3c;
                if (MUT(238, (off & 3) || off > lim, 0)) {
                    static char why[96];
                    snprintf(why, sizeof why, "G_MOVEWORD offset outside what a game writes (index %02x offset %04x)", ii, off);
                    return hle_decline(t, why);
                }
                if (ii == 0x0c && hle_dm16(t, L->mw_tab + ii) != (unsigned)L->vtx_base)
                    return hle_decline(t, "G_MW_POINTS table entry is not the vertex buffer");
            }
            {
                unsigned a = hle_dm16(t, L->mw_tab + ii) + ((w0 >> 8) & 0xffff);
                st32u(t, a, w1);
                f.ctx_ok = 0;
                if ((a & 0xfff) + 4 > (unsigned)L->vtx_base && (a & 0xfff) < (unsigned)(L->vtx_base + f3d_nslots(L) * F3D_VSTRIDE))
                    f.vb_ok = 0;
            }
            break;
        }
        case K3_POPMTX: cmd_popmtx(&f); f.ctx_ok = 0; break;
        case K3_CULLDL: {
            unsigned a = w0 & 0x3ff, guard = 0;
            uint32_t v = b->culldl_mask;
            if (w1 <= a || (w1 - a) % 40)
                return hle_decline(t, "G_CULLDL range unusual");
            if (MUT(242, a % 40 == 0 && w1 / 40 < 32 && (f.phantom & (((1u << (w1 / 40)) - 1u) & ~((1u << (a / 40)) - 1u))), 0))
                return hle_decline(t, "G_CULLDL over the partner slot of an odd G_VTX");
            do {
                v &= hle_dm16(t, L->vtx_base + 36 + a);
                a += 40;
            } while (a != w1 && ++guard < 64);
            if (v != 0 && hle_cmd(t, LATCH_K_CULLDL))
                return HLE_DECLINE;
            if (v != 0 && enddl(&f))
                goto end;
            break;
        }
        case K3_TRI1: {
            PR_BEGIN();
            int r = cmd_tri(&f, w1);
            PR_END(PR_TRI);
            if (r != HLE_OK)
                return HLE_DECLINE;
            break;
        }
        default:
            return hle_decline(t, "unsupported opcode");
        }
    next:
        if (f.gp <= 0) {                                /* the buffer is used up (or a list change): the next 40 commands */
            if (f.k0 & 7)
                return hle_decline(t, "display list address not 8-byte aligned");
            if (f.exact)
                hle_dma_read(t, L->cmdbuf, f.k0, 319);
            f.gp = 320;
            if (MUT(243, ring_hit(&f, f.k0, 320), 0))
                f.bad = "the display list is inside the RDP output ring";
        }
    }

end:
    if (g_pr.ops) {
        unsigned i;
        fprintf(stderr, "F3D_OPS");
        for (i = 0; i < 256; i++) if (opn[i]) fprintf(stderr, " %02x:%u", i, opn[i]);
        fprintf(stderr, "\n");
    }
    if (g_pr.on && ++g_pr.runs % 500 == 0)
        pr_dump();
    flush_pending(&f);
    /* the end of the task: the overlay 3 code is loaded and runs: SP_STATUS gets SIG2, BREAK */
    load_overlay(&f, 3);
    hle_mtc0(t, 4, 0x4000);
    hle_break(t, b->pc[F3PC_brk] + 4);
    return HLE_OK;
}

const struct hle_family f3d_fam = { .name = "f3d", .run = f3d_run };

/* Wave Race's triangles carry a shade block that the command byte does not announce when the shade bit is clear (10.5.2): the stream parser sizes the commands from their first byte */
unsigned f3d_wr_cmd_size(unsigned first_byte)
{
    unsigned op = first_byte & 0x3f, s = op & 7;
    if (op >= 0x08 && op <= 0x0f)
        return 32 + (s ? 64 : 0) + ((s & 2) ? 64 : 0) + ((s & 1) ? 16 : 0);
    return hle_rdp_cmd_size(op);
}

/* ---- the registered builds ---------------------------------------------------------------------------------------------------------------------------------------------------- */
/* .pc in the order of F3D_PC_FIELDS, read from each text's handler tables (docs/ucode-f3d.md 10.3.1; brk: the BREAK of overlay 3; load_key: the FNV-1a-64 of the text's first 0xf48 bytes):
 *          head   nop    brk    mtx    movemem vtx   dl     tri1   quad   halfc  half2  half1  persp  clrgeom setgeom enddl oml    omh    texture moveword popmtx culldl tri4 */
/* SM64 */
static const struct f3d_build l_sm64 = { .lay = &f3d_layout_a,
    .pc = { 0x060, 0x0a8, 0x780, 0x3dc, 0x558, 0x568, 0x734, 0x20c, 0, 0x380, 0x384, 0x378, 0x370, 0x358, 0x348, 0x328, 0x2cc, 0x2c4, 0x2a0, 0x288, 0x250, 0x30c, 0 },
    .clip_mask = 0x4343, .reject_mask = 0x7070, .culldl_mask = 0xffff, .vptr_mul = 4, .load_key = 0x021a806f5839b9ddull };
/* Pilotwings 64 (and Blast Corps' main text): the overlays at 0x788, G_TRI1 rejects with 0x7030 and G_CULLDL uses it too, its own clip flag word (10.5.1); the near-plane vector is DMEM data */
static const struct f3d_build l_pw = { .lay = &f3d_layout_pw,
    .pc = { 0x060, 0x0a8, 0x7a0, 0x3dc, 0x558, 0x568, 0x754, 0x20c, 0, 0x380, 0x384, 0x378, 0x370, 0x358, 0x348, 0x328, 0x2cc, 0x2c4, 0x2a0, 0x288, 0x250, 0x30c, 0 },
    .clip_mask = 0x4343, .reject_mask = 0x7030, .culldl_mask = 0x7030, .vptr_mul = 4, .load_key = 0x092eac93aa4c29c8ull };
/* Wave Race 64: the buffers 0x280 further up, the loop's gp the end address of the buffer, its own vertex and triangle encodings (the index scaled by 5), no front-face culling and no flat
 * shading, the shade block when any of the three low geometry bits is set (10.5.2), G_POPMTX sets the stale flag, op 0xb5 the quad; the overlays at 0x7c8 */
static const struct f3d_build l_wr = { .lay = &f3d_layout_wr,
    .pc = { 0x064, 0x09c, 0x7e0, 0x43c, 0x5b8, 0x5c8, 0x788, 0x278, 0x1fc, 0x3e0, 0x3e4, 0x3d8, 0x3d0, 0x3b8, 0x3a8, 0x388, 0x32c, 0x324, 0x300, 0x2e8, 0x2b0, 0x36c, 0 },
    .clip_mask = 0x4343, .reject_mask = 0x7070, .culldl_mask = 0x7070, .vptr_mul = 8, .vtx_wr = 1, .gp_end = 1, .tri_wr = 1, .popmtx_stale = 1, .load_key = 0x923f0d2666ac9355ull };
/* Mischief Makers (2.0H): Tetrisphere's code with the XBUS transport (the RDP reads the ring in DMEM; docs/ucode-f3d.md 10.6 4.4); the RDPHALF ops are one opcode higher (0xb2 .. 0xb4, 0xb1 is a no-op)
 * and there is no G_PERSPNORMALIZE (the perspective norm is G_MOVEWORD index 0x0e) */
static const struct f3d_build l_mm = { .lay = &f3d_layout_h,
    .pc = { 0x060, 0x0a8, 0x780, 0x3dc, 0x558, 0x568, 0x734, 0x208, 0, 0x378, 0x37c, 0x370, 0, 0x358, 0x348, 0x328, 0x2cc, 0x2c4, 0x2a0, 0x288, 0x24c, 0x30c, 0 },
    .clip_mask = 0x4343, .reject_mask = 0x7070, .culldl_mask = 0x7070, .vptr_mul = 4, .popmtx_stale = 1, .xbus = 1, .load_key = 0x5740f13da0cf9c89ull };
/* Tetrisphere (2.0H 02-12-97): the SM64 code with the G_DL depth at 0x15a, G_POPMTX setting the stale flag, the 2.0H RDPHALF ops, G_CULLDL mask 0x7070 (10.1, 10.3) */
static const struct f3d_build l_tt = { .lay = &f3d_layout_h,
    .pc = { 0x060, 0x0a8, 0x780, 0x3d8, 0x554, 0x564, 0x730, 0x20c, 0, 0x37c, 0x380, 0x374, 0, 0x35c, 0x34c, 0x32c, 0x2d0, 0x2c8, 0x2a4, 0x28c, 0x250, 0x310, 0 },
    .clip_mask = 0x4343, .reject_mask = 0x7070, .culldl_mask = 0x7070, .vptr_mul = 4, .popmtx_stale = 1, .load_key = 0x0edc2cbf3ad81e8bull };

/* GoldenEye 007 (2.0G 09-30-96): Pilotwings' clip flag word and masks, the overlays at 0x790, the 2.0H RDPHALF numbering and G_DL depth byte, no G_PERSPNORMALIZE (G_MOVEWORD 0x0e) and no G_POPMTX
 * (its cell leads to G_MOVEWORD's handler), G_TRI4 at 0xb1 (docs/ucode-f3d.md 15) */
static const struct f3d_build l_ge = { .lay = &f3d_layout_ge,
    .pc = { 0x060, 0x0a8, 0x7a8, 0x3e0, 0x55c, 0x56c, 0x758, 0x250, 0, 0x384, 0x388, 0x37c, 0, 0x364, 0x354, 0x334, 0x2d8, 0x2d0, 0x2ac, 0x294, 0, 0x318, 0x20c },
    .clip_mask = 0x4343, .reject_mask = 0x7030, .culldl_mask = 0x7030, .vptr_mul = 4, .load_key = 0x1e3311bf53a4ae12ull };

/* DMEM dead at the end of a task: scratch and leftovers (docs/ucode-f3d.md 11); the justification of each range is the poison proof (ARMLET_DMEM_POISON, docs/testing.md), the
 * references are to docs/ucode-f3d.md. The fourth byte of each transformed light direction (and of its copy) is a leftover of the vector registers the relight reads (7.2), never read:
 * slot k of the lookat and light vectors at 0x1b0 + 0x20 k, k = 0 .. 9 */
#define F3D_DEAD_LIGHT(k) { 0x1c3 + 0x20 * (k), 0x1c4 + 0x20 * (k) }, { 0x1c7 + 0x20 * (k), 0x1c8 + 0x20 * (k) }
#define F3D_DEAD_LIGHTS F3D_DEAD_LIGHT(0), F3D_DEAD_LIGHT(1), F3D_DEAD_LIGHT(2), F3D_DEAD_LIGHT(3), F3D_DEAD_LIGHT(4), F3D_DEAD_LIGHT(5), F3D_DEAD_LIGHT(6), F3D_DEAD_LIGHT(7), F3D_DEAD_LIGHT(8), F3D_DEAD_LIGHT(9)

#define F3D_DEAD_A \
    { 0x0b4, 0x0b6 },       /* the clipper's fan pivot (9) */ \
    { 0x6a0, 0x8e0 },       /* the command buffer and the geometry DMA buffer: dead in the default mode only (the native code reads RDRAM directly and does not stage them; ARMLET_HLE_EXACT=1 fills them like the microcode) */ \
    { 0x158, 0x15a },       /* the clipper's stale return address */ \
    { 0x8e0, 0x930 },       /* the triangle scratch */ \
    { 0x940, 0x968 },       /* the clipper's polygon buffers */ \
    { 0x9e0, 0xfc0 },       /* the RDP output buffer beyond the last command, the clipper's vertex slots, the matrix product scratch */ \
    F3D_DEAD_LIGHTS

static const struct hle_dead_range l_dead_sm64[] = {
    { 0x106, 0x108 },       /* the return address G_VTX and the clipper keep */
    F3D_DEAD_A
};
static const struct hle_dead_range l_dead_wr[] = {      /* the layout of Wave Race: everything from its command buffer up is scratch or the RDP output buffer (10.3.2) */
    { 0x106, 0x108 },
    { 0x0b4, 0x0b6 },
    { 0x158, 0x15a },
    { 0x920, 0xfc0 },       /* the command and geometry buffers, the triangle scratch, the clipper's polygon buffers and vertex area, the RDP output buffer */
    F3D_DEAD_LIGHTS
};
static const struct hle_dead_range l_dead_tt[] = {
    { 0x104, 0x106 },       /* the same slot, 2 bytes lower in the 2.0H builds of family A */
    F3D_DEAD_A
};

/* The OSTask output buffer bounds the RDP ring in RDRAM (the video plugin takes the lists; nothing reads the ring afterwards): a mask of the compare and the target of ARMLET_DMEM_POISON=ring */
#define F3D_UCODE_R(var, ident, nm, bld, dr, ring, ...) \
    HLE_UCODE(var) = { .id = ident, .name = nm, .fam = &f3d_fam, .fam_data = &bld, .ring_off = ring, .dead = dr, .n_dead = sizeof(dr) / sizeof(dr[0]), __VA_ARGS__ }
#define F3D_UCODE(var, ident, nm, bld, dr) F3D_UCODE_R(var, ident, nm, bld, dr, 0xfe8)

F3D_UCODE(l_f3d_sm64, 0x94c7cfdd2c872523ull, "Fast3D 2.0D (Super Mario 64)", l_sm64, l_dead_sm64);
F3D_UCODE(l_f3d_pw, 0x3e9b9c3a6ebde1e0ull, "Fast3D 2.0D (Pilotwings 64)", l_pw, l_dead_sm64);
F3D_UCODE_R(l_f3d_wr, 0xfb483893c5b0ad3dull, "Fast3D 2.0D (Wave Race 64)", l_wr, l_dead_wr, 0xfe8, .rdp_cmd_size = f3d_wr_cmd_size);
/* the lists go through a ring in DMEM (no ring in RDRAM) */
F3D_UCODE_R(l_f3d_mm, 0x4212d63f6e483348ull, "Fast3D 2.0H (Mischief Makers, War Gods; XBUS)", l_mm, l_dead_tt, 0, .flags = HLE_F_XBUS_OK);
F3D_UCODE(l_f3d_tt, 0x0b574ecb06d75978ull, "Fast3D 2.0H (Tetrisphere)", l_tt, l_dead_tt);
F3D_UCODE(l_f3d_ge, 0x538f79d8ff151c3eull, "Fast3D 2.0G (GoldenEye 007)", l_ge, l_dead_sm64);

#endif
