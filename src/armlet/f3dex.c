/*
 * arMLEt - f3dex.c: the F3DEX 1.x / F3DLX family (F3DEX 1.21, 1.23, 0.95, F3DEX.NoN 1.22, F3DLX 1.23, 0.95) on the task-level native path: a display-list interpreter in C
 *
 * A whole fresh task is run natively: the boot code and the microcode's initialisation (OS code, no vector unit worth replacing) on the RSP interpreter up to the command loop, then the display
 * list by C code over the struct-level functions of f3dex_fx.h, then the end of the task. The microcode's memory layout is kept (DMEM-resident state, docs/ucode-f3dex.md section 3), so what the
 * CPU and the RDP can observe, and DMEM outside the build's dead ranges, equals what the microcode leaves. Anything the code does not implement declines the task (the JIT runs it).
 *
 * The semantics are docs/ucode-f3dex/ (dl, rdp, state, mtx, light, clip) (dl.md: the loop and the task start, rdp.md: the flush, state.md, mtx.md, light.md, clip.md); the numbers of the sections in the comments refer to those.
 * A build is data (struct f3dex1_build, f3dex_bld.h): the reject mask, the NoN and LX variants, the DMEM layout and the handler pcs; the commands are dispatched through the build's own handler
 * tables in DMEM, so no code here tests which build it runs. The registered builds (ucode ids as gfx_ucode.c computes them) are at the end; dbuild.c makes one from a derived record.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "armlet.h"
#include "gfx_ucode.h"
#include "hle.h"
#include "latch.h"
#include "probe.h"
#include "f3dex_fx.h"
#include "f3dex_dmem.h"
#include "f3dex_bld.h"

/* test builds only (scripts/build.sh native --tools --hmut=N, tools/hle/mutcheck_fuzz.sh): a deliberate error N, to show that the tests catch it (docs/testing.md) */
#ifdef ARMLET_HLE_MUTATE
#define MUT(n, ok, bad) (ARMLET_HLE_MUTATE == (n) ? (bad) : (ok))
#else
#define MUT(n, ok, bad) (ok)
#endif

#ifdef ARMLET_NATIVE

#include <time.h>
#include <unistd.h>

/* ---- timing of the struct-level functions (F3DEX_PROF=1) and the scalar forms (F3DEX_SCALAR=1), for tools/f3dex/bench ------------------------------------------------------------------ */
static struct {
    int prof, scalar;
    uint64_t vcalls, verts, vns, tcalls[5], tns[5], mcalls, mns, ccalls, cns, rcalls, rns, ovh, tasks, ns_init, ns_flush, nflush, ns_task, ncmds, nsync, ns_sync, ndma, ns_dma, ns_end, nend, bytes_dma;
} P;
#define DT(t0) ({ uint64_t d_ = now_ns() - (t0); d_ > P.ovh ? d_ - P.ovh : 0; })
static inline uint64_t now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }
static void prof_init(void)
{
    uint64_t a, b, i;
    P.prof = getenv("F3DEX_PROF") != NULL;
    P.scalar = getenv("F3DEX_SCALAR") != NULL;
    if (P.prof) {                                         /* the cost of a timing pair */
        a = now_ns();
        for (i = 0; i < 1000; i++) b = now_ns();
        P.ovh = (b - a) / 1000;
    }
}
static void prof_report(void)
{
    int i;
    if (!P.prof) return;
    fprintf(stderr, "F3DEXP pid=%d scalar=%d vcalls=%llu verts=%llu vns=%llu", (int)getpid(), P.scalar, (unsigned long long)P.vcalls, (unsigned long long)P.verts, (unsigned long long)P.vns);
    for (i = 0; i < 5; i++)
        fprintf(stderr, " tc%d=%llu tn%d=%llu", i, (unsigned long long)P.tcalls[i], i, (unsigned long long)P.tns[i]);
    fprintf(stderr, " mc=%llu mn=%llu rc=%llu rn=%llu cc=%llu cn=%llu tasks=%llu init=%llu emitn=%llu emitc=%llu task=%llu cmds=%llu syncn=%llu syncc=%llu dman=%llu dmac=%llu dmab=%llu endn=%llu endc=%llu\n",
            (unsigned long long)P.mcalls, (unsigned long long)P.mns, (unsigned long long)P.rcalls, (unsigned long long)P.rns, (unsigned long long)P.ccalls, (unsigned long long)P.cns,
            (unsigned long long)P.tasks, (unsigned long long)P.ns_init, (unsigned long long)P.ns_flush, (unsigned long long)P.nflush, (unsigned long long)P.ns_task, (unsigned long long)P.ncmds,
            (unsigned long long)P.ns_sync, (unsigned long long)P.nsync, (unsigned long long)P.ns_dma, (unsigned long long)P.ndma, (unsigned long long)P.bytes_dma, (unsigned long long)P.ns_end, (unsigned long long)P.nend);
}
static int fx_vtx(const struct f3dex_vctx *c, const struct f3dex_vin *in, unsigned n, struct f3dex_vtx *out)
{
    uint64_t t0 = P.prof ? now_ns() : 0;
    int r = P.scalar ? f3dex_vtx_xform_scalar(c, in, n, out) : f3dex_vtx_xform(c, in, n, out);
    if (P.prof) { P.vns += DT(t0); P.vcalls++; P.verts += n; }
    return r;
}
static int fx_tri(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len)
{
    uint64_t t0 = P.prof ? now_ns() : 0;
    int r = P.scalar ? f3dex_tri_setup_scalar(in, cmd, len) : f3dex_tri_setup(in, cmd, len);
    if (P.prof) { P.tns[r] += DT(t0); P.tcalls[r]++; }
    return r;
}
static int fx_tri_noclip(const struct f3dex_tri_in *in, uint8_t *cmd, unsigned *len)
{
    uint64_t t0 = P.prof ? now_ns() : 0;
    int r = P.scalar ? f3dex_tri_setup_noclip_scalar(in, cmd, len) : f3dex_tri_setup_noclip(in, cmd, len);
    if (P.prof) { P.tns[r] += DT(t0); P.tcalls[r]++; }
    return r;
}
static void fx_mtx_mul(const struct f3dex_mtx *a, const struct f3dex_mtx *b, struct f3dex_mtx *o)
{
    uint64_t t0 = P.prof ? now_ns() : 0;
    if (P.scalar) f3dex_mtx_mul_scalar(a, b, o); else f3dex_mtx_mul(a, b, o);
    if (P.prof) { P.mns += DT(t0); P.mcalls++; }
}
static void fx_relight(const struct f3dex_mtx *mv, const int8_t raw[][3], unsigned nrec, int8_t out[][4])
{
    uint64_t t0 = P.prof ? now_ns() : 0;
    if (P.scalar) f3dex_relight_scalar(mv, raw, nrec, out); else f3dex_relight(mv, raw, nrec, out);
    if (P.prof) { P.rns += DT(t0); P.rcalls++; }
}
static void fx_clip(const struct f3dex_vctx *c, const struct f3dex_vtx *const v[3], struct f3dex_clip_out *o)
{
    uint64_t t0 = P.prof ? now_ns() : 0;
    if (P.scalar) f3dex_clip_tri_scalar(c, v, o); else f3dex_clip_tri(c, v, o);
    if (P.prof) { P.cns += DT(t0); P.ccalls++; }
}

/* ---- the builds ------------------------------------------------------------------------------------------------------------------------------------------------------------------ */
enum { OV_TEXT = 0, OV_CLIP = 1, OV_LIGHT = 2 };   /* what is in the overlay window: the text image (stubs that load the others), the clipper, the lighting code */

#define STAGE_MAX 4096                  /* bytes staged before a sync */

/* the command kinds: what a handler pc of the build's tables is (classify) */
enum { K1_BAD = 0, K1_NOP0, K1_NOP, K1_MTX, K1_MOVEMEM, K1_VTX, K1_DL, K1_LOADUC, K1_BRANCHZ, K1_TRI2, K1_MODVTX, K1_HALF2, K1_HALF1, K1_PERSP, K1_QUAD, K1_CLRGEOM, K1_SETGEOM, K1_ENDDL,
       K1_OML, K1_OMH, K1_TEXTURE, K1_MOVEWORD, K1_POPMTX, K1_CULLDL, K1_TRI1 };

struct f3dex_task {
    struct hle_task *t;
    const struct f3dex1_build *b;
    const struct f3dex_layout *L;
    uint8_t kind[256];
    uint32_t k0;                        /* RDRAM address of the next command */
    int gp;                             /* bytes left in the command window */
    int ov;
    unsigned ncmds;
    int trace;
    uint32_t wp, fstart, fend;           /* the RDRAM write pointer (DMEM 0x128) and the FIFO bounds, kept here */
    const char *bad;                    /* a reason to decline found where the function could not return one (checked at the head of the loop) */
    uint32_t stage[STAGE_MAX / 4];        /* the staged output as host words holding the big-endian words (the layout of hle_rdram_write) */
    uint32_t pend_ptr, pend_n, vend;      /* the output not yet written (staged in stage[]), and the DPC_END the microcode would have written last */
    int vend_dirty;
    uint32_t junk;                      /* bit k: slot k was loaded by an F3DLX G_VTX with G_CLIPPING off: the microcode's clip flags there are junk, ours are 0 */
    uint32_t uc;                        /* the RDRAM address of the text the microcode runs (OSTask.ucode, DMEM 0xfd0, which G_LOAD_UCODE rewrites) */
    uint32_t p_uc, p_data;              /* the task's own text and data pointers (the key of the tasks that decline at a load, load_known) */
    unsigned nloads;
    int exact;                          /* issue the DMA reads of the microcode (window, operand buffer): DMEM as the microcode leaves it, the SP register write events */
    const struct f3dex_build *fx;
    const struct hle_ucode *nu;         /* the hle_ucode of the build a G_LOAD_UCODE switches to (its gate: hle_bind) */
    const struct s2d1_build *s2;        /* S2DEX 1.x: the loop runs the 2D commands (hle_s2dex1.h); NULL for F3DEX 1.x */
    unsigned s7;                        /* S2DEX 1.x: the microcode's output pointer, DMEM 0x588 .. (the commands are assembled in DMEM and staged when it is above the threshold) */
    int s2ov;                           /* S2DEX 1.x: the overlay in IMEM 0x238 .. (1: sprites and rectangles, 2: backgrounds) */
};

static inline int f1_vidx_bad(uint32_t w) { return ((w & 0xfff) > 0x3e) | (w & 1); }       /* (the vertex pointer table has 32 entries of 2 bytes) */
static inline void st32u(struct hle_task *t, unsigned a, uint32_t v)       /* unaligned big-endian word, wrapping at 4 KiB */
{
    hle_st8(t, a, (uint8_t)(v >> 24)); hle_st8(t, a + 1, (uint8_t)(v >> 16)); hle_st8(t, a + 2, (uint8_t)(v >> 8)); hle_st8(t, a + 3, (uint8_t)v);
}
static inline uint32_t seg(const struct f3dex_task *f, uint32_t x) { return hle_dm32(f->t, f->L->segs + 4 * ((x >> 24) & 0xf)) + (x & 0xffffff); }

/* ---- the RDP FIFO: the microcode's flush of the output buffer (rdp.md 5, F3DLX: 9) ----------------------------------------------------------------------------------------------- */
/* The output side. The microcode DMAs every command to the FIFO at once and writes DPC_END; only the result is observable (the bytes in RDRAM, the stream the RDP sees, the final DPC and
 * DMEM words), not the chunking, so consecutive chunks that are contiguous in RDRAM are staged in DMEM 0x9a0.. (free in the non-exact mode) and sent by one DMA and one DPC_END write at the
 * next "sync": a FIFO wrap, a full stage, the end of the task. The wrap decision is made per chunk exactly as the microcode does (it depends on the chunk sizes). DPC_END is the value the
 * microcode would have written last (F3DEX: after every chunk; F3DLX: at the publishes W1 / W2 / W3 of rdp.md section 9, so it can lag the write pointer). */

static int sync_out_(struct f3dex_task *f);
static int sync_out(struct f3dex_task *f)
{
    uint64_t t0 = P.prof ? now_ns() : 0;
    int r = sync_out_(f);
    if (P.prof) { P.ns_sync += DT(t0); P.nsync++; }
    return r;
}
static int sync_out_(struct f3dex_task *f)
{
    struct hle_task *t = f->t;
    if (f->pend_n) {
        if (f->trace) fprintf(stderr, "F3DEX sync dma %08x n %u (end %08x dirty %d)\n", f->pend_ptr, f->pend_n, f->vend, f->vend_dirty);
        uint64_t d0 = P.prof ? now_ns() : 0;
        hle_rdram_write(t, f->pend_ptr, f->stage, f->pend_n);
        hle_st32(t, f->L->rdp_wp, f->pend_ptr + f->pend_n);
        if (P.prof) { P.ns_dma += DT(d0); P.ndma++; P.bytes_dma += f->pend_n; }
        f->pend_n = 0;
    }
    if (f->vend_dirty) {
        uint64_t e0 = P.prof ? now_ns() : 0;
        hle_mtc0(t, 9, f->vend);
        if (P.prof) { P.ns_end += DT(e0); P.nend++; }
        f->vend_dirty = 0;
    }
    return HLE_OK;
}

/* F3DLX: DPC_END := the DRAM write pointer (the callers' W1 / W2 / W3) */
static inline void dlx_publish(struct f3dex_task *f)
{
    if (f->fx->dlx) {
        f->vend = f->wp;
        f->vend_dirty = 1;
    }
}

/* the space for a chunk of n bytes (a multiple of 8) of RDP commands: the flush of the microcode's wrap decision. Returns the index of the first stage word, or -1 (declined) */
static int out_reserve(struct f3dex_task *f, unsigned n)
{
    struct hle_task *t = f->t;
    uint32_t ptr = f->wp;
    if ((int32_t)(f->fend - (ptr + n)) < 0) {               /* would run past the FIFO end: back to the start */
        sync_out(f);
        if (*t->cr[11] & 0x400)
            return hle_decline(t, "DPC start valid while wrapping the FIFO"), -1;
        if (*t->cr[10] == f->fstart)
            return hle_decline(t, "RDP at the start of the FIFO while wrapping"), -1;
        hle_mtc0(t, 8, f->fstart);
        ptr = f->fstart;
    } else {
        uint32_t cur = *t->cr[10];
        if (!((int32_t)(ptr - cur) >= 0) && !((int32_t)((ptr + n) - cur) < 0))
            return hle_decline(t, "FIFO full (the RDP would have to catch up)"), -1;
    }
    if (f->pend_n + n > STAGE_MAX)
        sync_out(f);
    if (!f->pend_n)
        f->pend_ptr = ptr;
    f->wp = ptr + n;
    if (!f->fx->dlx) {                                      /* F3DEX: the flush writes DPC_END */
        f->vend = f->wp;
        f->vend_dirty = 1;
    }
    {
        int i = (int)(f->pend_n >> 2);
        f->pend_n += n;
        return i;
    }
}
static int out_chunk(struct f3dex_task *f, const uint8_t *cmd, unsigned n)
{
    unsigned i;
    int w = out_reserve(f, n);
    if (w < 0)
        return HLE_DECLINE;
    for (i = 0; i < n; i += 4) {
        uint32_t x;
        memcpy(&x, cmd + i, 4);
        f->stage[w + (i >> 2)] = __builtin_bswap32(x);       /* (little-endian hosts: the command bytes are big-endian) */
    }
    return HLE_OK;
}

/* the microcode's flush of the output buffer at 0xce0 (exact mode: the DMA and DPC_END at once, as the microcode does) */
static int flush_(struct f3dex_task *f, unsigned n)
{
    struct hle_task *t = f->t;
    uint32_t ptr = hle_dm32(t, f->L->rdp_wp), fend = f->fend, fstart = f->fstart;
    if (n == 0)
        return HLE_OK;
    if ((int32_t)(fend - (ptr + n)) < 0) {
        if (*t->cr[11] & 0x400)
            return hle_decline(t, "DPC start valid while wrapping the FIFO");
        if (*t->cr[10] == fstart)
            return hle_decline(t, "RDP at the start of the FIFO while wrapping");
        hle_mtc0(t, 8, fstart);
        ptr = fstart;
    }
    {
        uint32_t cur = *t->cr[10];
        if (!((int32_t)(ptr - cur) >= 0) && !((int32_t)((ptr + n) - cur) < 0))
            return hle_decline(t, "FIFO full (the RDP would have to catch up)");
    }
    hle_dma_write(t, f->L->outbuf, ptr, n - 1);
    hle_st32(t, f->L->rdp_wp, ptr + n);
    if (!f->fx->dlx)
        hle_mtc0(t, 9, ptr + n);
    return HLE_OK;
}
static int flush(struct f3dex_task *f, unsigned n)
{
    uint64_t t0 = P.prof ? now_ns() : 0;
    int r = flush_(f, n);
    if (P.prof) { P.ns_flush += DT(t0); P.nflush++; }
    return r;
}

/* the bytes of an RDP command, with the F3DLX publishes around it (pass-through: W1 and W2; triangles: W3 before, nothing after) */
static int emit_bytes(struct f3dex_task *f, const uint8_t *cmd, unsigned len, int tri)
{
    unsigned i;
    int r;
    uint64_t t0 = P.prof ? now_ns() : 0;
    if (f->trace) { fprintf(stderr, "F3DEX emit %u:", len); for (i = 0; i < len && i < 16; i++) fprintf(stderr, " %02x", cmd[i]); fprintf(stderr, "\n"); }
    if (f->exact) {
        if (f->fx->dlx) { hle_mtc0(f->t, 9, hle_dm32(f->t, f->L->rdp_wp)); }
        for (i = 0; i < len; i += 4)
            hle_st32(f->t, f->L->outbuf + i, (uint32_t)cmd[i] << 24 | (uint32_t)cmd[i + 1] << 16 | (uint32_t)cmd[i + 2] << 8 | cmd[i + 3]);
        r = flush(f, len);
        if (r == HLE_OK && !tri && f->fx->dlx)
            hle_mtc0(f->t, 9, hle_dm32(f->t, f->L->rdp_wp));
        return r == HLE_OK ? HLE_OK : HLE_DECLINE;
    }
    dlx_publish(f);                                         /* W1 / W3 */
    r = out_chunk(f, cmd, len);
    if (r != HLE_OK)
        return HLE_DECLINE;
    if (!tri)
        dlx_publish(f);                                     /* W2 */
    if (P.prof) { P.ns_flush += DT(t0); P.nflush++; }
    return HLE_OK;
}
static int emit8(struct f3dex_task *f, uint32_t w0, uint32_t w1)
{
    if (!f->exact && !f->trace) {                           /* the common case: two words straight into the stage */
        int w;
        dlx_publish(f);
        w = out_reserve(f, 8);
        if (w < 0)
            return HLE_DECLINE;
        f->stage[w] = w0;
        f->stage[w + 1] = w1;
        dlx_publish(f);
        return HLE_OK;
    }
    {
        uint8_t b[8] = { (uint8_t)(w0 >> 24), (uint8_t)(w0 >> 16), (uint8_t)(w0 >> 8), (uint8_t)w0, (uint8_t)(w1 >> 24), (uint8_t)(w1 >> 16), (uint8_t)(w1 >> 8), (uint8_t)w1 };
        return emit_bytes(f, b, 8, 0);
    }
}

/* ---- overlays ------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
static void load_overlay(struct f3dex_task *f, int k)           /* the DMA table entry at DMEM 8 + 8k: dram word, length-1 halfword, memory address halfword */
{
    struct hle_task *t = f->t;
    unsigned d = 8 + 8 * k;
    hle_dma_read(t, hle_dm16(t, d + 6), hle_dm32(t, d), hle_dm16(t, d + 4));
}

/* ---- the operand of a class-0 command: the microcode's DMA into 0xae0, or (not exact) the RDRAM itself ----------------------------------------------------------------------- */
struct src { const uint32_t *img; unsigned base; int ram; };
/* The RDP output ring is written by the task's own flushes and the compare treats the bytes the RDP ignores (second words of syncs, slack) as don't care: a display list, vertices, a matrix or a light
 * read from inside it would read those bytes. Not a game's: decline (docs/design.md). */
static inline int ring_hit(const struct f3dex_task *f, uint32_t a, uint32_t n)
{
    a &= 0x7ffff8;
    return a < (f->fend & 0xffffff) && a + n > (f->fstart & 0x7fffff);
}
static struct src fetch(struct f3dex_task *f, uint32_t dram, uint32_t lenreg)
{
    struct src s;
    if (MUT(243, ring_hit(f, dram, ((lenreg & 0x3ff) | 7) + 1), 0))
        f->bad = "a read from inside the RDP output ring";
    if (f->exact) {
        hle_dma_read(f->t, f->L->operand, dram, lenreg);
        s.img = f->t->dmem; s.base = f->L->operand; s.ram = 0;
    } else {
        s.img = (const uint32_t *)f->t->rdram; s.base = dram & 0x7ffff8; s.ram = 1;
    }
    return s;
}
static inline unsigned dma_bytes(uint32_t w0) { return ((w0 & 0x3ff) | 7) + 1; }

/* ---- matrices (mtx.md) -------------------------------------------------------------------------------------------------------------------------------------------------------- */
static void mvp_recompute(const struct f3dex_task *f)
{
    struct f3dex_mtx mv, pj, mp;
    f3dex_load_mtx(&mv, f->t->dmem, f->L->mv, 0);
    f3dex_load_mtx(&pj, f->t->dmem, f->L->proj, 0);
    fx_mtx_mul(&mv, &pj, &mp);
    f3dex_store_mtx(&mp, f->t->dmem, f->L->mvp);
}

static int cmd_mtx(struct f3dex_task *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    unsigned fl = (w0 >> 16) & 0xff;
    struct src s;
    struct f3dex_mtx N, T, R;
    unsigned cur = (fl & 1) ? f->L->proj : f->L->mv;
    if (dma_bytes(w0) < 64)
        return hle_decline(t, "G_MTX operand shorter than a matrix");
    s = fetch(f, seg(f, w1), w0 & 0x3ff);
    f3dex_load_mtx(&N, s.img, s.base, s.ram);
    hle_st8(t, f->L->lights, 0xff);
    if (!(fl & 1) && (fl & 4)) {
        uint32_t ptr = hle_dm32(t, f->L->mtx_sp), lim = hle_dm32(t, f->L->mtx_lim);
        if (ptr != lim) {
            if (MUT(244, ring_hit(f, ptr, dma_bytes(w0)), 0))       /* (the output is staged and written later: a write into the ring now would be ordered differently from the microcode's) */
                return hle_decline(t, "matrix push into the RDP output ring");
            hle_st32(t, f->L->mtx_sp, ptr + 64);
            hle_dma_write(t, f->L->mv, ptr, w0 & 0x3ff);
        }
    }
    if (fl & 2) {
        R = N;
    } else {
        f3dex_load_mtx(&T, t->dmem, cur, 0);
        fx_mtx_mul(&N, &T, &R);
    }
    f3dex_store_mtx(&R, t->dmem, cur);
    mvp_recompute(f);
    return HLE_OK;
}

static int cmd_popmtx(struct f3dex_task *f)
{
    struct hle_task *t = f->t;
    uint32_t ptr = hle_dm32(t, f->L->mtx_sp), base = hle_dm32(t, 0xfe0);      /* (OSTask.dram_stack) */
    hle_st8(t, f->L->lights, 0xff);
    if ((int32_t)(base - ptr) >= 0)
        return HLE_OK;
    ptr -= 64;
    hle_dma_read(t, f->L->mv, ptr, 63);
    hle_st32(t, f->L->mtx_sp, ptr);
    mvp_recompute(f);
    return HLE_OK;
}

static int cmd_movemem(struct f3dex_task *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    unsigned idx = (w0 >> 16) & 0xff, dest, n, i;
    struct src s = fetch(f, seg(f, w1), w0 & 0x3ff);
    if (idx < 0x80 || idx > 0x9e || (idx & 1))
        return hle_decline(t, "G_MOVEMEM index outside the table");
    dest = hle_dm16(t, f->L->mm_tab + idx);
    n = 16 - (dest & 15);                               /* LQV + SQV: up to the next 16-byte boundary */
    if (n > dma_bytes(w0))
        return hle_decline(t, "G_MOVEMEM length shorter than the bytes stored");
    for (i = 0; i < n; i++) {
        unsigned a = s.base + i;
        hle_st8(t, dest + i, s.ram ? ((const uint8_t *)s.img)[a & 0x7fffff] : hle_dm8(t, a));
    }
    return HLE_OK;
}

/* ---- vertices (light.md) ------------------------------------------------------------------------------------------------------------------------------------------------------ */
static int cmd_vtx(struct f3dex_task *f, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    unsigned v0x2 = (w0 >> 16) & 0xff, n = (w0 >> 10) & 0x3f, cnt = n ? n : 1, i;
    struct f3dex_vctx ctx;
    struct f3dex_vin in[32];
    struct f3dex_vtx out[32];
    struct src s;
    uint32_t geom = hle_dm32(t, f->L->geom);
    if ((v0x2 & 1) || (v0x2 >> 1) + cnt > 32 || dma_bytes(w0) < 16 * cnt)
        return hle_decline(t, "G_VTX slot range or length unusual");
    if (MUT(239, geom & ~F3DEX_G_DEFINED, 0))               /* (bits the microcode's mode word has no meaning for steer its vertex and triangle code in ways the struct-level functions do not model) */
        return hle_decline(t, "geometry mode bits the build does not define");
    s = fetch(f, seg(f, w1), w0 & 0x3ff);               /* the class-0 DMA comes first */
    if (geom & F3DEX_G_LIGHTING) {
        uint32_t l = hle_dm32(t, f->L->lights);
        if (hle_cmd(t, LATCH_K_LIT))
            return HLE_DECLINE;
        if (f->ov != OV_LIGHT) {
            load_overlay(f, 2);
            f->ov = OV_LIGHT;
        }
        if ((int32_t)l < 0) {                           /* the lights are stale: transform the directions */
            unsigned nrec = (l & 0xfff) >> 5, k;
            int8_t raw[F3DEX_MAX_REC + 1][3], res[F3DEX_MAX_REC + 1][4];
            struct f3dex_mtx mv;
            if ((l & 31) || nrec > F3DEX_MAX_REC)
                return hle_decline(t, "G_VTX: light count unusual");
            for (k = 0; k <= nrec; k++)
                for (i = 0; i < 3; i++)
                    raw[k][i] = (int8_t)hle_dm8(t, f->L->light + 0x18 + 32 * k + i);
            f3dex_load_mtx(&mv, t->dmem, f->L->mv, 0);
            fx_relight(&mv, raw, nrec, res);
            for (k = 0; k <= nrec; k++)
                for (i = 0; i < 8; i++)
                    hle_st8(t, f->L->light + 0x20 + 32 * k + i, (uint8_t)res[k][i & 3]);
            hle_st32(t, f->L->lights, l & 0xfff);
        }
    }
    if (f3dex_load_vctx(&ctx, t->dmem, f->fx))
        return hle_decline(t, "G_VTX: light state unusual");
    for (i = 0; i < cnt; i++)
        f3dex_load_vin(&in[i], s.img, s.base + 16 * i, s.ram);
    if (fx_vtx(&ctx, in, cnt, out))
        return hle_decline(t, "G_VTX: state outside what the native code supports");
    if (t->gate || t->cmds) {                           /* (the latch policy's data situations) */
        uint32_t fo = 0;
        int wn = 0;
        for (i = 0; i < cnt; i++) {
            fo |= out[i].flags;
            wn |= hle_wneg(out[i].cx[3], out[i].cf[3]);
        }
        if (hle_vtx_kinds(t, fo, wn))
            return HLE_DECLINE;
    }
    for (i = 0; i < cnt; i++) {
        unsigned a = f->L->vbuf + 0x28u * ((v0x2 >> 1) + i);
        f3dex_store_vtx(&out[i], t->dmem, a);
        if (f->fx->dlx && !(geom & F3DEX_G_CLIPPING)) f->junk |= 1u << ((v0x2 >> 1) + i); else f->junk &= ~(1u << ((v0x2 >> 1) + i));
        if (f->trace && (v0x2 >> 1) + i == 19) fprintf(stderr, "F3DEX slot17 sy %04x (vtx cmd %08x %08x)\n", (uint16_t)out[i].sy, w0, w1);
        if (i & 1)
            hle_st16(t, a + 38, out[i - 1].flags);       /* the pair's word store: the first vertex's flags land in the second one's padding */
    }
    return HLE_OK;
}

/* ---- triangles (docs/ucode-f3dex.md 5, clip.md) ------------------------------------------------------------------------------------------------------------------------------ */
/* One call of the triangle code: the addresses are twice the vertex indices; flat = the address of the vertex whose colour a flat-shaded triangle uses ($4); *res = the setup result */
static int tri_one(struct f3dex_task *f, unsigned a0, unsigned a1, unsigned a2, unsigned aflat, int *res)
{
    struct hle_task *t = f->t;
    struct f3dex_vtx v[3], vf;
    struct f3dex_tri_in ti;
    uint8_t cmd[0xc0];
    unsigned len = 0, k, ad[3] = { a0, a1, a2 };
    int rc;
    if (MUT(239, hle_dm32(t, f->L->geom) & ~F3DEX_G_DEFINED, 0))
        return hle_decline(t, "geometry mode bits the build does not define");
    for (k = 0; k < 3; k++) {
        if ((ad[k] & 1) || ad[k] >= 64)
            return hle_decline(t, "G_TRI: vertex index outside the buffer");
        f3dex_load_vtx(&v[k], t->dmem, f->L->vbuf + 0x28u * (ad[k] >> 1));
        ti.v[k] = &v[k];
    }
    if ((aflat & 1) || aflat >= 64)
        return hle_decline(t, "G_TRI: vertex index outside the buffer");
    f3dex_load_vtx(&vf, t->dmem, f->L->vbuf + 0x28u * (aflat >> 1));
    if (f->fx->dlx && (hle_dm32(t, f->L->geom) & F3DEX_G_CLIPPING) && (f->junk & ((1u << (a0 >> 1)) | (1u << (a1 >> 1)) | (1u << (a2 >> 1)))))
        return hle_decline(t, "F3DLX triangle with clipping on uses a vertex loaded with clipping off");
    ti.b = f->fx;
    ti.flat = &vf;
    ti.geom = hle_dm32(t, f->L->geom);
    ti.persp = hle_dm16(t, f->L->persp);
    ti.tile_byte = hle_dm8(t, f->L->texture + 2);
    rc = fx_tri(&ti, cmd, &len);
    *res = rc;
    if (f->trace) fprintf(stderr, "F3DEX tri %u %u %u flat %u -> rc %d len %u geom %08x f=%04x %04x %04x\n", a0, a1, a2, aflat, rc, len, ti.geom, v[0].flags, v[1].flags, v[2].flags);
    if (rc == F3DEX_TRI_REJECT || rc == F3DEX_TRI_CULL || rc == F3DEX_TRI_DEGEN)
        return hle_cmd(t, rc == F3DEX_TRI_REJECT ? LATCH_K_TRI_REJECT : LATCH_K_TRI_CULL) ? HLE_DECLINE : HLE_OK;
    if (rc == F3DEX_TRI_OK)
        return emit_bytes(f, cmd, len, 1);
    {   /* clipping */
        struct f3dex_vctx ctx;
        struct f3dex_clip_out co;
        const struct f3dex_vtx *vp[3] = { &v[0], &v[1], &v[2] };
        unsigned i;
        if (hle_cmd(t, LATCH_K_CLIP))
            return HLE_DECLINE;
        if ((hle_wneg(v[0].cx[3], v[0].cf[3]) || hle_wneg(v[1].cx[3], v[1].cf[3]) || hle_wneg(v[2].cx[3], v[2].cf[3])) && hle_cmd(t, LATCH_K_NEARCLIP))
            return HLE_DECLINE;
        if (f->ov != OV_CLIP) {
            load_overlay(f, 1);
            f->ov = OV_CLIP;
        }
        f3dex_load_vctx_geom(&ctx, t->dmem, f->fx);
        fx_clip(&ctx, vp, &co);
        for (i = 0; i < co.ntri; i++) {
            struct f3dex_tri_in tj = ti;
            unsigned l2 = 0;
            for (k = 0; k < 3; k++)
                tj.v[k] = co.tri[i][k];
            if (fx_tri_noclip(&tj, cmd, &l2) == F3DEX_TRI_OK && emit_bytes(f, cmd, l2, 1) != HLE_OK)
                return HLE_DECLINE;
        }
    }
    return HLE_OK;
}

/* TRI1: the addresses are the bytes 1..3 of w1 (twice the vertex indices); TRI2: w1's triangle first, then w0's */
static int cmd_tri1(struct f3dex_task *f, uint32_t w1)
{
    int r;
    return tri_one(f, (w1 >> 16) & 0xff, (w1 >> 8) & 0xff, w1 & 0xff, (w1 >> 16) & 0xff, &r);
}
static int cmd_tri2(struct f3dex_task *f, uint32_t w0, uint32_t w1)
{
    int r;
    if (tri_one(f, (w1 >> 16) & 0xff, (w1 >> 8) & 0xff, w1 & 0xff, (w1 >> 16) & 0xff, &r) != HLE_OK)
        return HLE_DECLINE;
    return tri_one(f, (w0 >> 16) & 0xff, (w0 >> 8) & 0xff, w0 & 0xff, (w0 >> 16) & 0xff, &r);
}

/* F3DEX / F3DLX 0.95 G_QUAD (w1 bytes: v4, v1, v2, v3, twice the indices): two triangles split along the diagonal that is longer in y (v1-v3 against v2-v4); the second is skipped when the first was face-culled (the cull exit
 * returns with $31 = 0, which the quad code tests); the flat-shading vertex of both is v1 */
static int cmd_quad(struct f3dex_task *f, uint32_t w1)
{
    struct hle_task *t = f->t;
    unsigned a1 = (w1 >> 16) & 0xff, a2 = (w1 >> 8) & 0xff, a3 = w1 & 0xff, a4 = (w1 >> 24) & 0xff;
    int y[4], r, d13, d24;
    unsigned ad[4] = { a1, a2, a3, a4 }, i;
    for (i = 0; i < 4; i++) {
        if ((ad[i] & 1) || ad[i] >= 64)
            return hle_decline(t, "G_QUAD: vertex index outside the buffer");
        y[i] = (int16_t)hle_dm16(t, f->L->vbuf + 0x28u * (ad[i] >> 1) + 26);
    }
    d13 = y[2] - y[0]; if (d13 < 0) d13 = -d13;
    d24 = y[3] - y[1]; if (d24 < 0) d24 = -d24;
    {
        int swap = d13 - d24 > 0;                       /* (the disassembly prints subu's operands swapped: the code computes d13 - d24) */
        if (f->trace) fprintf(stderr, "F3DEX quad y %d %d %d %d d13 %d d24 %d swap %d\n", y[0], y[1], y[2], y[3], d13, d24, swap);
        if (tri_one(f, a1, a2, swap ? a4 : a3, a1, &r) != HLE_OK)
            return HLE_DECLINE;
        if (r == F3DEX_TRI_CULL)
            return HLE_OK;
        return tri_one(f, a3, a4, swap ? a2 : a1, a1, &r);
    }
}

/* ---- display-list control (dl.md 4, 5) ----------------------------------------------------------------------------------------------------------------------------------------- */
/* the end of the display list: pops a call, or ends the task. Returns 1 when the task ended, -1 to decline. */
static int enddl(struct f3dex_task *f)
{
    struct hle_task *t = f->t;
    int d = (int8_t)hle_dm8(t, f->L->depth) - 4;
    if (d < 0)
        return 1;
    f->k0 = hle_dm32(t, f->L->retstack + d);
    hle_st8(t, f->L->depth, (uint8_t)d);
    f->gp = 0;
    return (f->k0 & 7) ? -1 : 0;
}

/* the display-list address of a jump: the window is loaded from the 8-byte aligned address, which is also where the commands are */
static int jump_to(struct f3dex_task *f, uint32_t a)
{
    f->k0 = a;
    f->gp = 0;
    return (a & 7) ? hle_decline(f->t, "display list address not 8-byte aligned") : HLE_OK;
}

/* The kind of a handler pc of the build's tables. Class 0 (0x00 .. 0x3f) DMAs the operand first, class 2 (0x80 .. 0xbf) does not: a kind that belongs to the other class is not a command (BAD). */
static unsigned classify(const struct f3dex1_build *b, int cls0, unsigned h)
{
    static const uint8_t kind_of_pc[F1_NPC] = { K1_BAD, K1_NOP, K1_MTX, K1_MOVEMEM, K1_VTX, K1_DL, K1_LOADUC, K1_BRANCHZ, K1_TRI2, K1_MODVTX, K1_HALF2, K1_HALF1, K1_PERSP, K1_QUAD,
                                                 K1_CLRGEOM, K1_SETGEOM, K1_ENDDL, K1_OML, K1_OMH, K1_TEXTURE, K1_MOVEWORD, K1_POPMTX, K1_CULLDL, K1_TRI1 };
    unsigned k;
    if (h == b->pc[F1PC_nop])
        return cls0 ? K1_NOP0 : K1_NOP;
    for (k = F1PC_mtx; k < F1_NPC; k++)
        if (b->pc[k] && h == b->pc[k]) {
            unsigned kd = kind_of_pc[k];
            int c0 = kd == K1_MTX || kd == K1_MOVEMEM || kd == K1_VTX || kd == K1_DL;
            return c0 == cls0 ? kd : K1_BAD;
        }
    return K1_BAD;
}

/* the opcodes of the GBI (class 0: 0x00 .. 0x09, class 2: 0xaf .. 0xbf) through the build's handler tables (class 0 at jt0 + 2 op, class 2 at jt2 + 2 (op & 0x7f): dl.md 3.6); the other cells of the
 * overlapping tables are not handler pointers, and the other opcodes no game issues: BAD */
static void make_kinds(struct f3dex_task *f)
{
    unsigned op;
    memset(f->kind, K1_BAD, sizeof(f->kind));
    for (op = 0x00; op <= 0x09; op++)
        f->kind[op] = (uint8_t)classify(f->b, 1, hle_dm16(f->t, f->L->jt0 + 2 * op) & 0xffc);
    for (op = 0xaf; op <= 0xbf; op++)
        f->kind[op] = (uint8_t)classify(f->b, 0, hle_dm16(f->t, f->L->jt2 + 2 * (op & 0x7f)) & 0xffc);
}


static int load_ucode(struct f3dex_task *f, struct hle_cpu *c, uint32_t w0, uint32_t w1);
#include "hle_s2dex1.h"


/* ---- G_LOAD_UCODE (docs/native-f3dex.md section 5) ------------------------------------------------------------------------------------------------------------------------------- */
/* Tasks that load another text in the middle and decline at the command (SpaceStation Silicon Valley and Pokemon Puzzle League load S2DEX 1.x) have done native work for nothing: the text and data pointers of
 * a task are remembered with the number of tasks that ran with them and the number that declined at a load. When more than half did, the next tasks with the same pair decline at once, every 64th runs again
 * (the F3DEX2 front-end's rule, hle_f3dex2.c; the counts are halved from 1024 runs on). */
static struct { uint32_t uc, data; unsigned runs, loads, skip; } l_loads[8];
static unsigned l_nloads;

static int load_known(const struct hle_task *t)
{
    unsigned i;
    static int off = -1;
    if (off < 0)
        off = getenv("F3DEX_NO_LOADMEM") != NULL;          /* (a measurement aid: the memory off, tasks that load decline late) */
    if (off)
        return 0;
    for (i = 0; i < l_nloads; i++)
        if (l_loads[i].uc == hle_dm32(t, 0xfd0) && l_loads[i].data == hle_dm32(t, 0xfd8)) {
            if (MUT(466, l_loads[i].loads * 2 > l_loads[i].runs, 1) && ++l_loads[i].skip % 64 != 0)
                return 1;
            if (++l_loads[i].runs >= 1024) {
                l_loads[i].runs /= 2;
                l_loads[i].loads /= 2;
            }
            return 0;
        }
    return 0;
}

static void load_note(const struct f3dex_task *f)
{
    unsigned i;
    for (i = 0; i < l_nloads; i++)
        if (l_loads[i].uc == f->p_uc && l_loads[i].data == f->p_data) {
            l_loads[i].loads++;
            return;
        }
    if (l_nloads < sizeof(l_loads) / sizeof(l_loads[0])) {
        l_loads[l_nloads].uc = f->p_uc;
        l_loads[l_nloads].data = f->p_data;
        l_loads[l_nloads].runs = 1;
        l_loads[l_nloads++].loads = 1;
    }
}

/* the load of these pointers ran through (the loaded text has a build now): the pointers start afresh, or load_known would decline them 63 tasks in 64 for as long as the old loads outnumber the runs */
static void load_took(const struct f3dex_task *f)
{
    unsigned i;
    for (i = 0; i < l_nloads; i++)
        if (l_loads[i].uc == f->p_uc && l_loads[i].data == f->p_data) {
            l_loads[i].loads = 0;
            l_loads[i].runs = 1;
            l_loads[i].skip = 0;
        }
}

/* The build a G_LOAD_UCODE loads: the one whose text is the first 0xf48 bytes at `src` (f3dex1_build.load_key; a build the probe latched for this text has the key as its id: probe.c, and comes first, as in the
 * F3DEX2 front-end's loaded_build). Only a build of this front-end can be taken over. The keys of the last two texts are remembered (compared with memcmp against RDRAM). NULL: not a registered (or latched) build. */
static const struct hle_ucode *loaded_build(const struct hle_task *t, uint32_t src)
{
    static struct { uint8_t text[0xf48]; uint32_t addr; uint64_t key; int used; } slot[2];
    static unsigned next;
    const struct hle_ucode *u;
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
    key = MUT(467, key, ~key);
    u = armlet_hle_find(key);
    if (u && !hle_build_usable(t, u))                   /* a derived build that is not latched is not taken over, and neither is a registered one in its place (ARMLET_PROBE_FORCE) */
        return NULL;
    if (u && u->fam == &f3dex1_fam && ((const struct f3dex1_build *)u->fam_data)->load_key == key) {
        static const struct hle_ucode *logged[8];
        static unsigned nlogged;
        unsigned q;
        for (q = 0; q < nlogged && logged[q] != u; q++)
            ;
        if (!t->dx && q == nlogged && nlogged < 8 && !strncmp(u->name, "derived", 7)) {
            logged[nlogged++] = u;
            armlet_log(M64MSG_INFO, "hle: G_LOAD_UCODE of %016llx runs the latched build \"%s\"", (unsigned long long)key, u->name);
        }
        return u;
    }
    for (u = armlet_hle_next(NULL); u; u = armlet_hle_next(u))
        if (u->fam == &f3dex1_fam && key && hle_build_usable(t, u) && ((const struct f3dex1_build *)u->fam_data)->load_key == key)
            return u;
    return NULL;
}

/* The state of the loop from the interpreter at the first fetch of the microcode's command loop: at the start of the task and after the initialisation of a microcode that G_LOAD_UCODE started. Declines a state
 * the front-end does not know. */
static int enter(struct f3dex_task *f, const struct hle_cpu *c)
{
    struct hle_task *t = f->t;
    const struct f3dex_layout *L = f->L;
    f->s2 = f->b->s2;
    if (f->s2)
        return s2_enter(f, c);
    f->k0 = c->r[HLE_REG_K0] - 320;                     /* $26 = the address after the first window */
    f->gp = 320;
    f->ov = OV_TEXT;
    f->wp = hle_dm32(t, L->rdp_wp);
    f->fstart = hle_dm32(t, L->fifo);
    f->fend = hle_dm32(t, L->fifo + 4);
    if (f->wp & 7)
        return hle_decline(t, "RDP write pointer not 8-byte aligned");
    if (*t->cr[11] & 1)                                 /* (the microcode's start clears XBUS with DPC_STATUS <- 1: the lists are the FIFO's whatever the previous task left) */
        return hle_decline(t, "XBUS still set after the start of the task");
    *t->cr[11] |= MUT(490, 0, 1);
    if ((f->k0 & 7) || c->r[27] != L->window || c->r[28] != L->operand || c->r[23] != L->outbuf)
        return hle_decline(t, "unusual state at the command loop");
    make_kinds(f);
    return HLE_OK;
}

/* G_LOAD_UCODE (opcode 0xaf; w1: the text's RDRAM address; G_RDPHALF_1's word, DMEM L->half1: the data's; w0 & 0xffff: the data's DMA length register). The command's handler (pc f3dex1_build.pc[loaduc], in the
 * part of the text that stays in IMEM) saves the display list position (the next command's address, k0 + k1 - gp) in OSTask.data_ptr (DMEM 0xff0) and the text address in OSTask.ucode (0xfd0), copies the first
 * 0xf18 bytes of the text to IMEM 0x80 and the data to DMEM 0, waits for the DMA and starts the new text at IMEM 0x80. That is the start of a task that continues the old one: the initialisation (a "fresh" start, the
 * flags word of OSTask is clear) patches the DMA table with the new text address, sets the FIFO bounds and the RDP write pointer from OSTask (the data overwrites the DMEM words they are kept in, so they are
 * the data image's, which the task has to have prepared), fetches the first window from data_ptr and enters the loop. It runs on the interpreter, as at the start of the task, up to the new build's first command
 * fetch. What the microcode carries across is what that code does not write: DMEM above the data it loads (OSTask, the matrices and the vertex buffer when the data is shorter than they begin, the segments table
 * ...), IMEM outside the text copied (0 .. 0x7f and 0xf98 .. 0xfff: the loop head and the DMA routines, equal in the 1.2x builds: tails that differ decline), the SP / DPC registers and RDRAM. The loop's own
 * state is made anew from the interpreter's (enter), and the staged output goes to the FIFO first, so that RDRAM, DPC_END and DMEM have what the microcode has at the command. The same text is a re-initialisation
 * (the display list goes on where it was); the text of another registered (or latched) F3DEX 1.x build changes the record the loop works with. Anything else declines. Returns HLE_STOP to go on, or what f3dex_run returns. */
/* The part of a text that a G_LOAD_UCODE leaves in IMEM (0xf18 .. 0xf7f of the text, IMEM 0xf98 .. 0xfff; what lies behind 0xf80 is the overlay code the text is followed by) must be the same in the text that loads and the one that is loaded: the loop head and the DMA routines. The one place the F3DEX 1.2x
 * texts and S2DEX 1.x differ in is the wait for the DMA (text 0xf48 .. 0xf4f, IMEM 0xfc8 / 0xfcc): S2DEX spins on its own pc and reads the status once more in the delay slot, F3DEX branches back to the read. A wait ends at the
 * first read when the DMA is done, which it always is in the emulation, so the one routine is as good as the other (docs/ucode-s2dex1.md, section 6). */
static int tail_differs(const uint8_t *a, const uint8_t *b)
{
    return memcmp(a + 0xf18, b + 0xf18, 0xf48 - 0xf18) || memcmp(a + 0xf50, b + 0xf50, 0xf80 - 0xf50);
}

static int load_ucode(struct f3dex_task *f, struct hle_cpu *c, uint32_t w0, uint32_t w1)
{
    struct hle_task *t = f->t;
    const struct f3dex_layout *L = f->L;
    const struct f3dex1_build *nb = f->b;
    uint32_t src = w1 & 0x7ffff8, cur = f->uc & 0x7ffff8, size = ((w0 & 0xfff) | 7) + 1;
    int rc;

    if (++f->nloads > 64)
        return hle_decline(t, "too many G_LOAD_UCODE in a task");
    if (src + 0x1000 > 0x800000 || cur + 0x1000 > 0x800000) {
        if (!t->dx)
            load_note(f);
        return hle_decline(t, "G_LOAD_UCODE outside RDRAM");
    }
    if (memcmp(t->rdram + src, t->rdram + cur, 0x1000)) {
        const struct hle_ucode *nu = loaded_build(t, src);
        if (MUT(463, 1, 0))                             /* (the probe derives a loaded text too, from a replay as well: the data image is what DMA fills DMEM 0 with, `size` bytes) */
            armlet_probe_note_load(src, hle_dm32(t, L->half1), (w0 & 0xfff) + 1);
        if (f->trace)
            fprintf(stderr, "F3DEX G_LOAD_UCODE text %08x key %016llx data %08x -> %s\n", src, (unsigned long long)armlet_ucode_hash(t->rdram, src, 0xf48), hle_dm32(t, L->half1), nu ? nu->name : "(not registered)");
        if (!nu || (t->ctx && !armlet_gfx_allowed(&t->ctx->cfg, nu->id))) {
            if (!t->dx)
                load_note(f);
            return hle_decline(t, "G_LOAD_UCODE of another ucode");
        }
        if (hle_cmd(t, LATCH_K_LOAD))                   /* (the latch policy: a ucode load is a risky path of the host build) */
            return HLE_DECLINE;
        if (!t->dx)
            load_took(f);
        f->nu = nu;
        if (MUT(464, tail_differs(t->rdram + src, t->rdram + cur), 0)) {
            if (!t->dx)
                load_note(f);
            return hle_decline(t, "G_LOAD_UCODE: the part of the text that stays in IMEM differs");
        }
        nb = (const struct f3dex1_build *)nu->fam_data;
        if (MUT(470, size < 0x800, 0)) {                /* the data segments of the builds differ in a few constants (the end of the overlay window ..) that the other build's code reads: only a whole image is a state the front-end knows */
            if (!t->dx)
                load_note(f);
            return hle_decline(t, "G_LOAD_UCODE of another build with a partial data image");
        }
    }
    if (!nb->s2 && (f3dex_lay(&nb->fx)->vtab < f3dex_lay(&nb->fx)->mw_tab + 2 || !nb->pc[F1PC_head]))
        return hle_decline(t, "the build's G_MOVEWORD table has no room");
    if (f->junk && size > L->vbuf) {                   /* (only a decline: the slots the data overwrites would hold its flags and no longer be junk, which the native code would have to follow) */
        if (!t->dx)
            load_note(f);
        return hle_decline(t, "G_LOAD_UCODE data over F3DLX vertices loaded with clipping off");
    }

    if (MUT(462, 0, 1))                                 /* (a mutation: the command does nothing) */
        return HLE_STOP;
    sync_out(f);                                        /* RDRAM, the FIFO write pointer in DMEM and DPC_END as the microcode has them */
    c->r[HLE_REG_T9] = w0;                              /* the registers as the loop head leaves them for the handler */
    c->r[HLE_REG_T8] = w1;
    c->r[HLE_REG_K0] = f->k0 + f->gp;
    c->r[HLE_REG_K1] = L->operand - f->gp;
    c->r[28] = L->operand;
    if (f->s2) {                                        /* S2DEX 1.x's loop: k1 counts the bytes left in the window up to 0 (negative), the output pointer is live, the handler clears $28 itself */
        c->r[HLE_REG_K1] = (uint32_t)-f->gp;
        c->r[28] = 0;
        c->r[HLE_REG_S7] = f->s7;
    }
    if (MUT(460, 0, 1))
        c->r[HLE_REG_K0] = f->k0 + f->gp + 8;
    c->pc = f->b->pc[F1PC_loaduc];
    c->stop_pc = nb->pc[F1PC_head];
    rc = hle_rsp_run(t, c);
    if (rc == HLE_OK)
        return HLE_OK;                                  /* the task ended in the initialisation */
    if (rc != HLE_STOP)
        return HLE_DECLINE;
    if (*t->cr[4] & 0x80)
        return hle_decline(t, "yield request pending");
    f->uc = MUT(468, src, f->uc);
    if (nb != f->b) {
        t->switched = 1;
        hle_bind(t, f->nu);                             /* (the loaded build's gate, or its counting in a replay that validates it) */
    }
    f->b = MUT(461, nb, f->b);
    f->L = f3dex_lay(&f->b->fx);
    f->fx = &f->b->fx;
    if (enter(f, c) != HLE_OK)
        return HLE_DECLINE;
    return HLE_STOP;
}

static int f3dex_run(struct hle_task *t, const struct hle_ucode *u)
{
    const struct f3dex1_build *b = (const struct f3dex1_build *)u->fam_data;
    const struct f3dex_layout *L = f3dex_lay(&b->fx);
    struct f3dex_task f;
    struct hle_cpu c;
    int rc;
    static int exact = -1;

    if (exact < 0) {
        exact = 0;
        prof_init();
    }
    exact = hle_exact();      /* debugging: issue the microcode's DMA reads too (DMEM staging, the SP register write events) */
    memset(&f, 0, sizeof(f));
    f.t = t;
    f.b = b;
    f.L = L;
    f.fx = &b->fx;
    f.exact = exact;
    { static int tr = -1; if (tr < 0) tr = getenv("F3DEX_TRACE") != NULL; f.trace = tr; }
    t->fam = &f;
    if (hle_dm32(t, 0xfc4) & 1)
        return hle_decline(t, "yielded task");
    if (*t->cr[4] & 0x80)
        return hle_decline(t, "yield request pending");
    if (!b->s2 && L->vtab < L->mw_tab + 2)
        return hle_decline(t, "the build's G_MOVEWORD table has no room");
    f.uc = hle_dm32(t, 0xfd0);
    f.p_uc = f.uc;
    f.p_data = hle_dm32(t, 0xfd8);
    if (!t->dx && load_known(t))
        return hle_decline(t, "an earlier task with these pointers loaded another ucode");

    /* the boot code and the microcode's initialisation, on the interpreter, up to the command loop head */
    uint64_t tt0 = P.prof ? now_ns() : 0, ti0 = tt0;
    hle_cpu_init(t, &c, t->start_pc, 1u << 16);
    c.stop_pc = b->pc[F1PC_head];
    rc = hle_rsp_run(t, &c);
    if (P.prof) P.ns_init += now_ns() - ti0;
    if (rc == HLE_OK)
        return HLE_OK;                                  /* the boot code ended the task (a pending signal): exactly as the microcode */
    if (rc != HLE_STOP)
        return HLE_DECLINE;
    if (enter(&f, &c) != HLE_OK)
        return HLE_DECLINE;

    for (;;) {
        uint32_t w0, w1;
        unsigned op;
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
        if (f.trace) fprintf(stderr, "F3DEX cmd %u: %08x %08x\n", f.ncmds, w0, w1);
        if (f.s2) {                                     /* S2DEX 1.x (hle_s2dex1.h) */
            rc = s2_cmd(&f, &c, op, w0, w1);
            if (rc == S2R_END)
                return s2_end(&f, &c);
            if (rc == S2R_FINISH)
                return HLE_OK;
            if (rc != HLE_OK)
                return rc;
            b = f.b;                                    /* (a G_LOAD_UCODE may have switched the build) */
            L = f.L;
        } else if (op >= 0xc0) {                        /* the RDP commands: copied, the segment address of the image commands resolved */
            if (op >= 0xfd)
                w1 = seg(&f, w1);
            if (emit8(&f, w0, w1) != HLE_OK)
                return HLE_DECLINE;
        } else if (op >= 0x40 && op < 0x80) {
            /* class 1: a no-op without a DMA */
        } else {
            unsigned kd = f.kind[op];
            if (op < 0x40) {                            /* class 0: a DMA into the operand buffer is started for every one of them; then the handler */
                /* (the DMA is not issued in the lean mode, and the buffer and the output buffer after it are dead: a length that runs on into the live DMEM past the output buffer is not a game's) */
                if (MUT(227, dma_bytes(w0) > (unsigned)(L->outbuf - L->operand), 0))
                    return hle_decline(t, "class-0 operand longer than the operand buffer");
            }
            switch (kd) {
            case K1_NOP0:
                if (f.exact)
                    hle_dma_read(t, L->operand, seg(&f, w1), w0 & 0x3ff);
                break;
            case K1_NOP:
                break;
            case K1_MTX: if (cmd_mtx(&f, w0, w1) != HLE_OK) return HLE_DECLINE; break;
            case K1_MOVEMEM: if (cmd_movemem(&f, w0, w1) != HLE_OK) return HLE_DECLINE; break;
            case K1_VTX: if (cmd_vtx(&f, w0, w1) != HLE_OK) return HLE_DECLINE; break;
            case K1_DL: {
                int d = (int8_t)hle_dm8(t, L->depth);
                if (f.exact)
                    hle_dma_read(t, L->operand, seg(&f, w1), w0 & 0x3ff);
                if (((w0 >> 16) & 0xff) == 0) {          /* a call */
                    if (d < 0)
                        return hle_decline(t, "G_DL: negative stack depth");
                    if (d - 68 > 0)
                        break;                          /* the stack is full: the command is ignored */
                    hle_st32(t, L->retstack + d, f.k0);
                    hle_st8(t, L->depth, (uint8_t)(d + 4));
                }
                if (jump_to(&f, seg(&f, w1)) != HLE_OK)
                    return HLE_DECLINE;
                break;
            }
            case K1_HALF1:
                hle_st32(t, L->half1, w1);
                break;
            case K1_HALF2:
                if (emit8(&f, hle_dm32(t, L->half1), w1) != HLE_OK)
                    return HLE_DECLINE;
                break;
            case K1_LOADUC: {
                int r = load_ucode(&f, &c, w0, w1);
                if (r != HLE_STOP)
                    return r;
                b = f.b;
                L = f.L;
                break;
            }
            case K1_BRANCHZ: {
                if (MUT(472, f1_vidx_bad(w0), 0))      /* (as G_MODIFYVTX: the slot address of an offset past the table is staged data the lean mode does not have) */
                    return hle_decline(t, "G_BRANCH_Z vertex out of range");
                unsigned va = hle_dm16(t, L->vtab + (w0 & 0xfff));
                uint32_t vz = hle_dm32(t, va + 0x1c);
                if ((int32_t)(vz - w1) <= 0 && hle_cmd(t, LATCH_K_BRZ))
                    return HLE_DECLINE;
                if ((int32_t)(vz - w1) <= 0 && jump_to(&f, seg(&f, hle_dm32(t, L->half1))) != HLE_OK)
                    return HLE_DECLINE;
                break;
            }
            case K1_TRI2:
                if (cmd_tri2(&f, w0, w1) != HLE_OK)
                    return HLE_DECLINE;
                break;
            case K1_MODVTX: {
                if (MUT(471, f1_vidx_bad(w0) || ((w0 >> 16) & 0xff) >= 0x28 || ((w0 >> 16) & 3), 0))      /* a vertex outside the 32 slots, an odd number, or a field outside the record: the slot address is read from, and the store goes to, anywhere in DMEM (Toy Story 2 perturbed: 0x800 read the staging area the lean mode does not fill) */
                    return hle_decline(t, "G_MODIFYVTX out of range");
                unsigned va = hle_dm16(t, L->vtab + (w0 & 0xfff));
                st32u(t, ((w0 >> 16) & 0xff) + va, w1);
                break;
            }
            case K1_PERSP:
                hle_st16(t, L->persp, (uint16_t)w1);
                break;
            case K1_QUAD:
                if (cmd_quad(&f, w1) != HLE_OK)
                    return HLE_DECLINE;
                break;
            case K1_CLRGEOM: hle_st32(t, L->geom, hle_dm32(t, L->geom) & ~w1); break;
            case K1_SETGEOM: hle_st32(t, L->geom, hle_dm32(t, L->geom) | w1); break;
            case K1_ENDDL: {
                int e = enddl(&f);
                if (e > 0)
                    goto end;
                if (e < 0)
                    return hle_decline(t, "display list address not 8-byte aligned");
                break;
            }
            case K1_OML: case K1_OMH: {
                unsigned a = kd == K1_OML ? L->oml : L->omh;
                uint32_t m = (uint32_t)(((1u << (w0 & 31)) - 1) << ((w0 >> 8) & 31));
                hle_st32(t, a, (hle_dm32(t, a) & ~m) | w1);
                if (emit8(&f, hle_dm32(t, L->omh), hle_dm32(t, L->oml)) != HLE_OK)
                    return HLE_DECLINE;
                break;
            }
            case K1_TEXTURE:
                hle_st32(t, L->texture, w0);
                hle_st32(t, L->texture + 4, w1);
                hle_st32(t, L->geom, (hle_dm32(t, L->geom) & ~2u) | ((w0 & 1) << 1));
                break;
            case K1_MOVEWORD: {
                unsigned ii = w0 & 0xff;
                unsigned off = (w0 >> 8) & 0xffff;
                if ((ii & 1) || ii > (unsigned)(L->vtab - L->mw_tab - 2))      /* (the table of G_MOVEWORD ends where the vertex table begins: 8 entries, 7 in the 0.95 builds) */
                    return hle_decline(t, "G_MOVEWORD index outside the table");
                if (MUT(238, (off & 3) || off > (ii == 10 ? 0x1fc : 0x3c), 0))     /* a game's offsets are words within the group (segments 0 .. 0x3c, clip ratios, the light colours up to the last light) */
                    return hle_decline(t, "G_MOVEWORD offset outside what a game writes");
                st32u(t, (unsigned)(int16_t)hle_dm16(t, L->mw_tab + ii) + ((w0 >> 8) & 0xffff), w1);
                break;
            }
            case K1_POPMTX: cmd_popmtx(&f); break;
            case K1_CULLDL: {                               /* culled when some clip-flag bit is set in all the vertices of the range */
                uint32_t acc = f.fx->reject_mask;
                unsigned guard = 0;
                if (b->culldl_off) {                        /* byte offsets into the vertex buffer, the last vertex excluded */
                    unsigned p = w0 & 0x3ff;
                    if (w1 <= p || (w1 - p) % 40 || w1 > 40 * 32)
                        return hle_decline(t, "G_CULLDL range unusual");
                    do { if (f.junk >> (p / 40) & 1) return hle_decline(t, "G_CULLDL over F3DLX vertices loaded with clipping off"); acc &= hle_dm16(t, L->vbuf + 0x24 + p); p += 40; } while (p != w1 && ++guard < 64);
                } else {
                    unsigned pa = hle_dm16(t, L->vtab + (w0 & 0xfff)), pb = hle_dm16(t, L->vtab + (w1 & 0xfff)), p = pa;
                    if (pb < pa || (pb - pa) % 40 || pb > L->vbuf + 0x28u * 31)
                        return hle_decline(t, "G_CULLDL range unusual");
                    for (;;) {
                        if (p >= L->vbuf && (f.junk >> ((p - L->vbuf) / 40) & 1))     /* (a record below the buffer is not a slot: its flags are what DMEM holds, for both) */
                            return hle_decline(t, "G_CULLDL over F3DLX vertices loaded with clipping off");
                        acc &= (uint32_t)(int32_t)(int16_t)hle_dm16(t, p + 0x24);
                        if (p == pb)
                            break;
                        p += 40;
                    }
                }
                if (acc != 0) {
                    int e;
                    if (hle_cmd(t, LATCH_K_CULLDL))
                        return HLE_DECLINE;
                    e = enddl(&f);
                    if (e > 0)
                        goto end;
                    if (e < 0)
                        return hle_decline(t, "display list address not 8-byte aligned");
                }
                break;
            }
            case K1_TRI1:
                if (cmd_tri1(&f, w1) != HLE_OK)
                    return HLE_DECLINE;
                break;
            default:
                return hle_decline(t, op < 0x40 ? "unsupported class-0 opcode" : "unsupported class-2 opcode");
            }
        }
        if (f.gp <= 0) {                                /* the window is used up (or a list change): the next 40 commands */
            if (f.k0 & 7)
                return hle_decline(t, "display list address not 8-byte aligned");
            if (f.exact || f.s2)                        /* (S2DEX's DMEM state is its own: the window is read as the microcode reads it, which also leaves the DMA registers where the microcode does) */
                hle_dma_read(t, L->window, f.k0, 319);
            f.gp = 320;
            if (MUT(243, ring_hit(&f, f.k0, 320), 0))
                f.bad = "the display list is inside the RDP output ring";
        }
    }

end:
    if (!f.exact)
        sync_out(&f);
    /* the end of the task: the overlay 3 code is loaded and runs: SP_STATUS gets SIG2, BREAK */
    load_overlay(&f, 3);
    hle_mtc0(t, 4, 0x4000);
    hle_break(t, (hle_dm16(t, 0x26) & 0xfff) + 0x1c);          /* DMEM 0x26: the IMEM address of overlay 3 (the DMA table) = the window; its BREAK is at +0x18 */
    if (P.prof) { P.ns_task += now_ns() - tt0; P.tasks++; P.ncmds += f.ncmds; }
    prof_report();
    return HLE_OK;
}

const struct hle_family f3dex1_fam = { .name = "f3dex", .run = f3dex_run };

/* ---- the builds ---------------------------------------------------------------------------------------------------------------------------------------------------------------- */
/* Layouts other than f3dex_lay_std (f3dex_vtx_fx.c): Turok's F3DEX.NoN 1.00 has the class tables 4 bytes higher (the 0.95 places), the 0.95 builds also the vertex table 2 lower and the display-list
 * stack depth byte at 0x110 (0x110 .. 0x113 is the perspective word in the 1.2x builds) */
static const struct f3dex_layout l_lay_c4 = { .jt0 = 0xc4, .jt2 = 0x76, .half1 = 0x10c, .persp = 0x112, .geom = 0x114, .omh = 0x118, .oml = 0x11c, .texture = 0x120, .rdp_wp = 0x128, .lights = 0x12c,
    .mtx_sp = 0x134, .mtx_lim = 0x15c, .depth = 0x148, .fifo = 0x150, .segs = 0x160, .light = 0x1a0, .mm_tab = 0x270, .mw_tab = 0x30e, .vtab = 0x31e, .viewport = 0x380, .fog = 0x390,
    .retstack = 0x398, .mv = 0x3e0, .proj = 0x420, .mvp = 0x460, .vbuf = 0x4a0, .window = 0x9a0, .operand = 0xae0, .outbuf = 0xce0,
    .lim = 0x28, .tgbias = 0x38, .vsign = 0x50, .clip_k = 0x76, .planes = 0x98, .tg = 0xa2, .plane_masks = 0xb2 };
static const struct f3dex_layout l_lay_095 = { .jt0 = 0xc4, .jt2 = 0x76, .half1 = 0x10c, .persp = 0x112, .geom = 0x114, .omh = 0x118, .oml = 0x11c, .texture = 0x120, .rdp_wp = 0x128, .lights = 0x12c,
    .mtx_sp = 0x134, .mtx_lim = 0x15c, .depth = 0x110, .fifo = 0x150, .segs = 0x160, .light = 0x1a0, .mm_tab = 0x270, .mw_tab = 0x30e, .vtab = 0x31c, .viewport = 0x380, .fog = 0x390,
    .retstack = 0x398, .mv = 0x3e0, .proj = 0x420, .mvp = 0x460, .vbuf = 0x4a0, .window = 0x9a0, .operand = 0xae0, .outbuf = 0xce0,
    .lim = 0x28, .tgbias = 0x38, .vsign = 0x50, .clip_k = 0x76, .planes = 0x98, .tg = 0xa2, .plane_masks = 0xb2 };

/* .fx = { reject_mask, non, dlx, layout }; .pc in the order of F3DEX_PC_FIELDS:
 *          head   nop    mtx    movemem vtx   dl     loaduc branchz tri2  modvtx half2  half1  persp  quad   clrgeom setgeom enddl oml    omh    texture moveword popmtx culldl tri1
 * (read from each text's handler tables: docs/ucode-f3dex/dl.md section 3.6; load_key: the FNV-1a-64 of the text's first 0xf48 bytes, from the dumps) */
static const struct f3dex1_build l_b_121 = { .fx = { 0x7070, 0, 0, &f3dex_lay_std },                 /* F3DEX 1.21 */
    .pc = { 0x058, 0x09c, 0x554, 0x6d0, 0x3a8, 0x6e0, 0xf8c, 0x2f8, 0x1e4, 0x334, 0x350, 0x348, 0, 0, 0x320, 0x310, 0x2d8, 0x27c, 0x274, 0x250, 0x238, 0x204, 0x2b8, 0x1bc },
    .culldl_off = 0, .load_key = 0xe00c958ba9b578c0ull };
static const struct f3dex1_build l_b_non122 = { .fx = { 0x7030, 1, 0, &f3dex_lay_std },                 /* F3DEX.NoN 1.22 */
    .pc = { 0x058, 0x09c, 0x574, 0x6f0, 0x3a8, 0x700, 0xf8c, 0x2f8, 0x1e4, 0x334, 0x350, 0x348, 0, 0, 0x320, 0x310, 0x2d8, 0x27c, 0x274, 0x250, 0x238, 0x204, 0x2b8, 0x1bc },
    .culldl_off = 0, .load_key = 0x3e4037f1162004e8ull };
static const struct f3dex1_build l_b_123 = { .fx = { 0x7070, 0, 0, &f3dex_lay_std },                 /* F3DEX 1.23 */
    .pc = { 0x058, 0x09c, 0x54c, 0x6c8, 0x3a0, 0x6d8, 0xf8c, 0x2f0, 0x1dc, 0x32c, 0x348, 0x340, 0, 0, 0x318, 0x308, 0x2d0, 0x274, 0x26c, 0x248, 0x230, 0x1fc, 0x2b0, 0x1b4 },
    .culldl_off = 0, .load_key = 0x5e166557ca542f4cull };
static const struct f3dex1_build l_b_lx123 = { .fx = { 0x7070, 0, 1, &f3dex_lay_std },                 /* F3DLX 1.23 */
    .pc = { 0x058, 0x09c, 0x578, 0x6f4, 0x3b0, 0x704, 0xf80, 0x2e8, 0x1d4, 0x324, 0x340, 0x338, 0, 0, 0x310, 0x300, 0x2c8, 0x26c, 0x264, 0x240, 0x228, 0x1f4, 0x2a8, 0x1ac },
    .culldl_off = 0, .load_key = 0x6b7f51fed2105fb1ull };
static const struct f3dex1_build l_b_095 = { .fx = { 0x7070, 0, 0, &l_lay_095 },                 /* F3DEX 0.95 */
    .pc = { 0x058, 0x09c, 0x5dc, 0x758, 0x430, 0x768, 0, 0, 0x224, 0, 0x3d8, 0x3d0, 0x3c8, 0x244, 0x3b4, 0x3a4, 0x384, 0x32c, 0x324, 0x300, 0x2e8, 0x2b4, 0x368, 0x1fc },
    .culldl_off = 1, .load_key = 0xbd2eddb3d4739b35ull };
static const struct f3dex1_build l_b_lx095 = { .fx = { 0x7070, 0, 1, &l_lay_095 },                 /* F3DLX 0.95 (no dump: the pcs from dl.md section 3.6) */
    .pc = { 0x058, 0x09c, 0x608, 0x784, 0x440, 0x794, 0, 0, 0x21c, 0, 0x3d0, 0x3c8, 0x3c0, 0x23c, 0x3ac, 0x39c, 0x37c, 0x324, 0x31c, 0x2f8, 0x2e0, 0x2ac, 0x360, 0x1f4 },
    .culldl_off = 1, .load_key = 0 };
static const struct f3dex1_build l_b_lx121 = { .fx = { 0x7070, 0, 1, &f3dex_lay_std },                 /* F3DLX 1.21 */
    .pc = { 0x058, 0x09c, 0x580, 0x6fc, 0x3b8, 0x70c, 0xf80, 0x2f0, 0x1dc, 0x32c, 0x348, 0x340, 0, 0, 0x318, 0x308, 0x2d0, 0x274, 0x26c, 0x248, 0x230, 0x1fc, 0x2b0, 0x1b4 },
    .culldl_off = 0, .load_key = 0x7fbe76cb59dbef39ull };
static const struct f3dex1_build l_b_lxn121 = { .fx = { 0x7030, 1, 1, &f3dex_lay_std },                 /* F3DLX.NoN 1.21 */
    .pc = { 0x058, 0x09c, 0x5a0, 0x71c, 0x3b8, 0x72c, 0xf80, 0x2f0, 0x1dc, 0x32c, 0x348, 0x340, 0, 0, 0x318, 0x308, 0x2d0, 0x274, 0x26c, 0x248, 0x230, 0x1fc, 0x2b0, 0x1b4 },
    .culldl_off = 0, .load_key = 0x528d66df73e165c9ull };
static const struct f3dex1_build l_b_lxn123 = { .fx = { 0x7030, 1, 1, &f3dex_lay_std },                 /* F3DLX.NoN 1.23 */
    .pc = { 0x058, 0x09c, 0x598, 0x714, 0x3b0, 0x724, 0xf80, 0x2e8, 0x1d4, 0x324, 0x340, 0x338, 0, 0, 0x310, 0x300, 0x2c8, 0x26c, 0x264, 0x240, 0x228, 0x1f4, 0x2a8, 0x1ac },
    .culldl_off = 0, .load_key = 0x16cc2c957cb30d31ull };
static const struct f3dex1_build l_b_non100 = { .fx = { 0x7030, 1, 0, &l_lay_c4 },                 /* F3DEX.NoN 1.00 (Turok): 1.2x opcodes, no G_BRANCH_Z / G_LOAD_UCODE */
    .pc = { 0x058, 0x09c, 0x5bc, 0x738, 0x3f0, 0x748, 0, 0, 0x224, 0x37c, 0x398, 0x390, 0, 0, 0x368, 0x358, 0x338, 0x2dc, 0x2d4, 0x2b0, 0x298, 0x264, 0x318, 0x1fc },
    .culldl_off = 0, .load_key = 0x5db9ec31c25de879ull };

/* DMEM dead at the end of a task: scratch and leftovers (docs/ucode-f3dex.md section 8). CANDIDATES until the poison probe has established them for this family (ARMLET_DMEM_POISON); each
 * entry says what writes it. The window / operand / output buffer are only dead because the native code does not issue the microcode's DMA reads (F3DEX_EXACT=1 does). */
#define F1_DEAD_PADS_1 { 0x1c3, 0x1c4 }, { 0x1c7, 0x1c8 }, { 0x1e3, 0x1e4 }, { 0x1e7, 0x1e8 }, { 0x203, 0x204 }, { 0x207, 0x208 }, { 0x223, 0x224 }, { 0x227, 0x228 }, { 0x243, 0x244 }, { 0x247, 0x248 }, \
    { 0x263, 0x264 }, { 0x267, 0x268 }, { 0x283, 0x284 }, { 0x287, 0x288 },
#define F1_DEAD_PADS_2 { 0x2a3, 0x2a4 }, { 0x2a7, 0x2a8 },          /* (S2DEX 1.x's flags words: its list leaves these two out) */
#define F1_DEAD_PADS_3 { 0x2c3, 0x2c4 }, { 0x2c7, 0x2c8 }, { 0x2e3, 0x2e4 }, { 0x2e7, 0x2e8 },
#define F1_DEAD_MAIN \
    { 0x0b4, 0x0b6 },       /* the clipper's fan pivot */ \
    { 0x106, 0x108 },       /* the return address G_VTX and the clipper keep, the fan pointer */ \
    { 0x158, 0x15a },       /* the clipper's stale return address */ \
    { 0x9a0, 0xce0 },       /* the command window and the operand buffer */ \
    { 0xbe0, 0xc40 },       /* the triangle scratch (sorted vertex addresses, perspective-scaled s/t/w, texture scale inputs) */ \
    { 0xc40, 0xc70 },       /* the clipper's polygon lists */ \
    { 0xce0, 0xd90 },       /* the RDP output buffer and the triangle code's coefficient scratch beyond the command */ \
    { 0xde0, 0xfc0 },       /* (the vertex clip flags of every slot follow: junk after an F3DLX G_VTX with clipping off, internal to a task otherwise) */       /* the clipper's new vertices, the matrix product scratch (0xde0..0xe1f), the light-recompute scratch (0xdf0..0xe2f) */ \
    /* the clip flags (+0x24, +0x26) of the 32 vertex slots */ \
    { 0x4c4, 0x4c8 }, \
    { 0x4ec, 0x4f0 }, \
    { 0x514, 0x518 }, \
    { 0x53c, 0x540 }, \
    { 0x564, 0x568 }, \
    { 0x58c, 0x590 }, \
    { 0x5b4, 0x5b8 }, \
    { 0x5dc, 0x5e0 }, \
    { 0x604, 0x608 }, \
    { 0x62c, 0x630 }, \
    { 0x654, 0x658 }, \
    { 0x67c, 0x680 }, \
    { 0x6a4, 0x6a8 }, \
    { 0x6cc, 0x6d0 }, \
    { 0x6f4, 0x6f8 }, \
    { 0x71c, 0x720 }, \
    { 0x744, 0x748 }, \
    { 0x76c, 0x770 }, \
    { 0x794, 0x798 }, \
    { 0x7bc, 0x7c0 }, \
    { 0x7e4, 0x7e8 }, \
    { 0x80c, 0x810 }, \
    { 0x834, 0x838 }, \
    { 0x85c, 0x860 }, \
    { 0x884, 0x888 }, \
    { 0x8ac, 0x8b0 }, \
    { 0x8d4, 0x8d8 }, \
    { 0x8fc, 0x900 }, \
    { 0x924, 0x928 }, \
    { 0x94c, 0x950 }, \
    { 0x974, 0x978 }, \
    { 0x99c, 0x9a0 },
/* the fourth byte of each transformed light direction (and its copy), records 0..9 (the shim of tools/equiv takes 64 ranges per ucode): a leftover of the vector registers the relight reads (mtx.md 4.2), never read */
#define F1_DEAD_RANGES F1_DEAD_MAIN F1_DEAD_PADS_1 F1_DEAD_PADS_2 F1_DEAD_PADS_3
static const struct hle_dead_range l_dead[] = { F1_DEAD_RANGES };


#define U(var, uid, nm, bld) HLE_UCODE(var) = { .id = (uid), .name = (nm), .fam = &f3dex1_fam, .fam_data = &(bld), .dead = l_dead, .n_dead = sizeof(l_dead) / sizeof(l_dead[0]), .ring_off = 0xfe8, .flags = HLE_F_XBUS_OK }
U(l_bk, 0x208a567e9aead5eaull, "F3DEX 1.21 (Banjo-Kazooie, Glover, Bomberman Hero, Quest 64)", l_b_121);
U(l_sf, 0xfd5a31ccf6d40b62ull, "F3DEX.NoN 1.22 (Star Fox 64, Quake, Wetrix)", l_b_non122);
U(l_ts_ex, 0x047252e810e35d56ull, "F3DEX 1.23 (Toy Story 2, Earthworm Jim 3D, Mystical Ninja, SpaceStation, Pokemon Puzzle League)", l_b_123);
U(l_ts_lx, 0x7502cfb7098eeffbull, "F3DLX 1.23 (Toy Story 2, A Bug's Life)", l_b_lx123);
U(l_mk_ex, 0x34dad67335a54cdaull, "F3DEX 0.95 (Mario Kart 64)", l_b_095);
U(l_mk_lx, 0x5f33f7c2be924de0ull, "F3DLX 0.95", l_b_lx095);
U(l_wo, 0x76c763bc18ad3ab3ull, "F3DLX 1.21 (Wipeout 64)", l_b_lx121);
U(l_cf, 0x0a7cc6c93363c663ull, "F3DLX.NoN 1.21 (Clay Fighter SC)", l_b_lxn121);
U(l_ig, 0xcc09096cb6a48e7bull, "F3DLX.NoN 1.23 (Iggy's Reckin' Balls, Penny Racers)", l_b_lxn123);
U(l_tu, 0x5063f108118bf62aull, "F3DEX.NoN 1.00 (Turok)", l_b_non100);

/* ---- S2DEX 1.06 / 1.07 (hle_s2dex1.h; docs/ucode-s2dex1.md) ------------------------------------------------------------------------------------------------------------------------------------------ */
static const struct f3dex_layout l_lay_s2d1 = { .half1 = 0xe0, .omh = 0xc8, .oml = 0xcc, .rdp_wp = 0xd4, .depth = 0xd0, .fifo = 0xfe8, .segs = 0xe8, .retstack = 0x128, .window = 0x3c0, .outbuf = 0x588 };
/* the pcs by S2D1_PC_FIELDS: nop half0 half1 half2 enddl omh oml moveword rendermode setom segcopy scissor sync copy seldl dl movemem bg1cyc rect sprite bgcopy loadtxtr loaduc ldtx_spr ldtx_rect (read from the jump tables in the data images) */
static const struct s2d1_build l_s2_106 = { .jt = 0x3aa, .bgc107 = 0,
    .pc = { 0x158, 0x000, 0x00c, 0x014, 0x038, 0x050, 0x054, 0x0b0, 0x0c8, 0x108, 0x114, 0x1f4, 0x220, 0x224, 0x184, 0x1b4, 0x1dc, 0x6e8, 0x6f4, 0x238, 0x968, 0xc84, 0xf6c, 0x258, 0x714 } };
static const struct s2d1_build l_s2_107 = { .jt = 0x3aa, .bgc107 = 1,
    .pc = { 0x158, 0x000, 0x00c, 0x014, 0x038, 0x050, 0x054, 0x0b0, 0x0c8, 0x108, 0x114, 0x1f4, 0x220, 0x224, 0x184, 0x1b4, 0x1dc, 0x6e8, 0x6f4, 0x238, 0x968, 0xcb0, 0xf6c, 0x258, 0x714 } };
static const struct f3dex1_build l_b_s2_106 = { .fx = { 0, 1, 0, &l_lay_s2d1 }, .pc = { [F1PC_head] = 0x158, [F1PC_loaduc] = 0xf6c }, .load_key = 0x3a88b2c21c8ec254ull, .s2 = &l_s2_106 };
static const struct f3dex1_build l_b_s2_107 = { .fx = { 0, 1, 0, &l_lay_s2d1 }, .pc = { [F1PC_head] = 0x158, [F1PC_loaduc] = 0xf6c }, .load_key = 0x9f01e13abd30c872ull, .s2 = &l_s2_107 };
/* dead at the end of a task: the display list window (the lean mode reads the list from RDRAM) and the RDP output buffer with its slack (the commands are assembled there; the flush DMAs them to the FIFO), and the F3DEX 1.x ranges: the tasks of
 * these builds load an F3DEX 1.x text (Yoshi's Story: F3DEX.NoN 1.22, Pokemon Puzzle League: F3DEX 1.23) and back, and what it leaves in its window, output buffer and scratch is not kept (the lean F3DEX 1.x path) */
static const struct hle_dead_range l_dead_s2d1[] = { { 0x3c0, 0x500 }, { 0x588, 0x960 }, F1_DEAD_MAIN F1_DEAD_PADS_1 F1_DEAD_PADS_3 };
#define S2U(var, uid, nm, bld) HLE_UCODE(var) = { .id = (uid), .name = (nm), .fam = &f3dex1_fam, .fam_data = &(bld), .dead = l_dead_s2d1, .n_dead = sizeof(l_dead_s2d1) / sizeof(l_dead_s2d1[0]), .ring_off = 0xfe8 }
S2U(l_s2_y, 0x5b327e0de25d12b9ull, "S2DEX 1.06 (Yoshi's Story)", l_b_s2_106);
S2U(l_s2_p, 0x190d04dfa000cd4bull, "S2DEX 1.07 / S2DEXD 1.07 (Bangaioh, Magical Tetris Challenge, Pokemon Puzzle League)", l_b_s2_107);

#endif
