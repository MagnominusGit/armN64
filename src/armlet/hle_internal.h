/*
 * arMLEt - hle_internal.h: what hle_core.c, hle_task.c and the offline harness (tools/native/replay.c) share
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_HLE_INTERNAL_H
#define ARMLET_HLE_INTERNAL_H

#include <stdlib.h>

#include "hle.h"
#include "latch.h"
#include "native/native.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The RDP command bytes of a task as one stream (list boundaries are not observable, docs/design.md): FNV-1a-64 and length of the bytes of every list in
 * order, and the first STREAM_KEEP bytes for finding the first difference. */
#define HLE_STREAM_KEEP (32u << 20)
struct hle_stream {
    uint64_t hash, len;
    uint64_t hash_r;                            /* the same stream with bytes 1 .. 7 of NO_OP and the syncs zeroed (the relaxed comparison of a task that switched builds, hle_task.switched) */
    uint8_t *bytes;
    size_t cap;
    unsigned (*size)(unsigned first_byte);      /* the ucode's own sizing of a command, or NULL */
    uint32_t cmd_pos, cmd_end, ignore_from;     /* the command being parsed, carried over from one list to the next (a command split over lists normalises as one) */
    uint64_t keep;                              /* bytes kept at most (0: HLE_STREAM_KEEP) */
};

static inline void hle_stream_reset(struct hle_stream *s, const struct hle_ucode *uc)
{
    s->size = uc ? uc->rdp_cmd_size : NULL;
    s->hash = s->hash_r = 0xcbf29ce484222325ull;
    s->len = 0;
    s->cmd_pos = s->cmd_end = 0;
    s->ignore_from = ~0u;
}

/* The size in bytes of the RDP command that starts with `op` (the low 6 bits of its first byte): triangles 32 + 64 shade + 64 texture + 16 z, texture rectangles 16 (the RSP
 * writes the two half-words behind the command), everything else 8. */
static inline unsigned hle_rdp_cmd_size(unsigned op)
{
    if (op >= 0x08 && op <= 0x0f)
        return 32 + ((op & 4) ? 64 : 0) + ((op & 2) ? 64 : 0) + ((op & 1) ? 16 : 0);
    if (op == 0x24 || op == 0x25)
        return 16;
    return 8;
}
/* The second word of NO_OP and of the sync commands is ignored by the RDP; the microcode leaves whatever the output buffer held there (scratch of earlier commands), so the
 * stream comparison zeroes it (docs/design.md, "What verify compares"). Where a one-byte store writes the command (S2DEX), the three bytes behind it in the first word are scratch as well; they are
 * the same in a native run and the JIT's unless the task switched builds (a G_LOAD_UCODE in the middle: the buffers of the build it loaded lie over DMEM that the build before left, in ranges it
 * declares dead and does not keep exactly: the clipper's new vertices, the scratch of the triangle setup); the comparison of such a task zeroes them too (hash_r). */
static inline int hle_rdp_w1_ignored(unsigned op) { return op == 0x00 || (op >= 0x26 && op <= 0x29); }

/* the list [cur, end) as a CMD_END event reports it (status bit 0 = XBUS: the bytes are in DMEM); dmem / rdram are the host word images */
static inline void hle_stream_add(struct hle_stream *s, const PrspNativeEvent *e, const uint32_t *dmem, const uint32_t *rdram)
{
    uint32_t a;
    if (e->kind != PRSP_EV_CMD_END || e->b <= e->a || e->b - e->a > (1u << 20))
        return;
    for (a = e->a; a < e->b; a++) {
        uint8_t byte = e->c & 1 ? ((const uint8_t *)dmem)[(a & 0xfff) ^ 3] : ((const uint8_t *)rdram)[(a & 0x7fffff) ^ 3];
        if (s->cmd_pos == s->cmd_end) {         /* a command starts */
            unsigned op = byte & 0x3f;
            s->cmd_pos = 0;
            s->cmd_end = s->size ? s->size(byte) : hle_rdp_cmd_size(op);
            s->ignore_from = hle_rdp_w1_ignored(op) ? 4 : ~0u;
        }
        {
            uint8_t rb = s->ignore_from != ~0u && s->cmd_pos >= 1 ? 0 : byte;
            if (s->cmd_pos >= s->ignore_from && s->cmd_pos < s->ignore_from + 4)
                byte = 0;
            s->hash_r = (s->hash_r ^ rb) * 0x100000001b3ull;
        }
        s->cmd_pos++;
        s->hash = (s->hash ^ byte) * 0x100000001b3ull;
        if (s->len < (s->keep ? s->keep : HLE_STREAM_KEEP)) {
            if (s->len >= s->cap) {
                s->cap = s->cap ? s->cap * 2 : 1u << 16;
                s->bytes = (uint8_t *)realloc(s->bytes, s->cap);
                if (!s->bytes)
                    abort();
            }
            s->bytes[s->len] = byte;
        }
        s->len++;
    }
}

/* a task's snapshot file carries the JIT's stream as one more event after the recorded ones */
#define HLE_EV_STREAM 0x100u    /* a = length low, b = length high, hash = FNV-1a-64 */

/* The offline harness: run the family's front-end from `before` (the RDRAM rows it reads must be loaded), compare with the JIT's end state `after` and recorded
 * events `ev` (the last one the HLE_EV_STREAM). Returns the number of differences (0 = identical), or -1 when the front-end declined (the reason in out). */
struct hle_harness_result {
    int declined;
    double native_us;
};
int armlet_hle_harness(const struct hle_ucode *uc, const PrspNativeSnap *before, const PrspNativeSnap *after, const PrspNativeEvent *ev, size_t n_ev, char *out,
                       size_t outsz, struct hle_harness_result *res);

/* like armlet_hle_harness, but the oracle is a JIT run from the same state, made here by jit(ctx) (the state `before` is loaded, SP_PC set, parallelRSPDoRspCycles called), not a recorded
 * result: for snapshots that were changed after the capture (tools/rsp2c/fuzz_snaps.py) */
int armlet_hle_harness_jit(const struct hle_ucode *uc, const PrspNativeSnap *before, void (*jit)(void *), void *jctx, char *out, size_t outsz, struct hle_harness_result *res);

/* The probe (probe.c): a pair run of a build's front-end and the JIT on a canned task, compared. */
struct hle_probe_opts {
    int tol;                                    /* near-match tolerance for the RDP stream: shade coefficients only, 2 levels / 1/128 level per pixel (hle_core.c, stream_tolerable) */
    int loose_levels;                           /* armlet_hle_dx_compare with a latch_diff: a difference inside this many colour levels (shade coefficients, vertex colours) is NUMERIC, not structural; 0: none */
    /* a DMEM byte (address addr, native value na, JIT value nb) that a tolerance of `lim` colour levels admits: nonzero; dmem is the native end state */
    int (*dmem_ok)(const struct hle_ucode *u, const uint32_t *dmem, unsigned addr, unsigned na, unsigned nb, int lim);
    void (*perturb)(const struct hle_ucode *u, uint32_t *native_dmem);   /* test builds only: changes the native end state before the compare (mutations 352, 353) */
};
enum { HLE_PROBE_EQUAL = 0, HLE_PROBE_DECLINED, HLE_PROBE_MISMATCH, HLE_PROBE_INCONCLUSIVE, HLE_PROBE_TOLERANT };
struct hle_probe_out {
    int status;                                 /* HLE_PROBE_*: equal; the front-end declined (why); differences (report); the JIT did not finish; equal within the tolerance */
    const char *why;
    char report[2048];
    unsigned n_diff, n_events;
    double native_us, jit_us;
    uint64_t stream_len;
};
/* `rdram`: the canned task's own 8 MiB of RDRAM (the game's is not touched); the RDRAM compared: the output ring [win_lo, win_hi) and the matrix stack [win2_lo, win2_hi) */
int armlet_hle_probe_pair(const struct hle_ucode *uc, const PrspNativeSnap *before, const struct hle_probe_opts *opts, uint32_t win_lo, uint32_t win_hi, uint32_t win2_lo, uint32_t win2_hi,
                          uint32_t *rdram, struct hle_probe_out *out);
void armlet_hle_set_rdram(uint32_t *rdram);                      /* probe_rdram.cpp */
int armlet_hle_register_dynamic(const struct hle_ucode *u);
const struct hle_ucode *armlet_hle_find_registered(uint64_t id);  /* the static table only (not the builds the probe registered) */     /* a build derived at run time: found by armlet_hle_find and armlet_hle_next after the registered ones */
void armlet_hle_unregister_dynamic(const struct hle_ucode *u);

/* ---- the detached replay (validate.c; docs/design.md "The background validator and the latch policy") -------------------------------------------------------------------------
 * A front-end runs a captured task on memories and registers of its own (hle_task.dx): DMEM, IMEM, an 8 MiB RDRAM holding the rows the JIT's run read, the SP / DPC registers, MI_INTR, the
 * status-poll counters and an RSP register file and vector unit for the interpreter's parts. The SP register writes are done here as the fork's RSP_MTC0 does them (DMA, SP_STATUS, the
 * semaphore, DPC_START / END / STATUS; a DPC_END takes the list as the video plugin does, START = CURRENT = END), nothing reaches the fork, the video plugin or the game's RDRAM: it runs on any thread.
 * The front-ends' own static state is shared with the emulation thread: the replay holds the front-end lock (armlet_hle_fe_*), which the emulation thread only ever tries. */
struct hle_dx {
    void *cpu;                                  /* an RSP::CPUState of its own (hle_rsp.cpp) */
    uint32_t *dmem, *imem;                      /* 1024 host words each */
    uint8_t *rdram;                             /* 8 MiB */
    uint32_t crv[16], irq;
    uint32_t *cr[16];
    int16_t mfc0_count[32];
    int32_t timeout;
    const char *why;
    unsigned end_pc;
    int broke, switched;
    struct hle_stream st;                       /* the RDP lists, as the verify compare takes them */
    struct hle_dx_wr { uint32_t a, n; } *wr;    /* the RDRAM rows the run wrote (DMA, hle_rdram_write) */
    size_t n_wr, cap_wr;
    const struct hle_ucode *cand;               /* the build under validation (its commands are counted in hist) */
    const struct hle_ucode *bound;              /* the build the loop ran last (hle_bind): after a G_LOAD_UCODE switch the build the task ended in, whose dead DMEM the end-state compare accepts too */
    uint32_t hist[LATCH_NKINDS];
};
void *hle_dx_cpu_new(void);                     /* hle_rsp.cpp */
void hle_dx_cpu_free(void *cpu);
void hle_dx_cpu_load(void *cpu, const PrspNativeSnap *s, uint32_t *dmem, uint32_t *imem, uint32_t *rdram);
uint32_t *hle_dx_cpu_sr(void *cpu);
/* Runs build `uc` on the task that starts in `s0` (its RDRAM rows already in dx->rdram); dx->cand set by the caller. HLE_OK (the task ended: dx holds the end state) or HLE_DECLINE (dx->why). */
int armlet_hle_dx_run(struct hle_dx *dx, const struct hle_ucode *uc, const PrspNativeSnap *s0);
/* The JIT's run of the same task: its end state, its RDP stream (bytes kept or not), the RDRAM rows it wrote outside the ring with their final bytes */
struct hle_dx_jit {
    const PrspNativeSnap *s1;
    const struct hle_stream *st;
    const struct hle_dx_wr *wr;                 /* (a, n) of each row, the bytes one after the other in wr_bytes */
    size_t n_wr;
    const uint8_t *wr_bytes;
};
/* Compares the replay's end state with the JIT's as the verify compare does (external effects; the probe's near-match tolerance when opts->tol). Returns the number of differences
 * (0: equal); *tol_used: equal only within the tolerance. The first ones are described in rep. diff (or NULL): the classes of the differences for latch_grade (a loose build: HARD the task
 * ended otherwise, NUMERIC inside opts->loose_levels in the shade coefficients or vertex colours, structural the rest). */
unsigned armlet_hle_dx_compare(const struct hle_dx *dx, const struct hle_ucode *uc, const struct hle_dx_jit *j, const struct hle_probe_opts *opts, char *rep, size_t repsz, int *tol_used,
                               struct latch_diff *diff);
/* the front-end lock: the validator takes it around a replay; the emulation thread only tries it (the JIT runs the task when it is taken) */
void armlet_hle_fe_enable(int on);              /* off: no locking at all (no validator) */
int armlet_hle_fe_trylock(void);                /* 0: taken (or locking is off) */
int armlet_hle_fe_wait(unsigned ns);            /* the emulation thread's wait for a replay in progress, at most ns: 0 taken, -1 still busy */
void armlet_hle_fe_lock(void);
void armlet_hle_fe_unlock(void);
uint64_t armlet_hle_fe_busy(void);
void armlet_hle_core_init(void);
int hle_dead_dmem_of(const struct hle_ucode *u, unsigned byte);      /* the byte is in one of the build's dead DMEM ranges */                /* the path's settings (before a validator thread starts) */              /* tasks the emulation thread gave to the JIT because the lock was taken */
const char *armlet_hle_last_why(void);
extern const char hle_gate_why[];               /* the decline reason of the command-kind gate */

double armlet_hle_dmabench(const struct hle_ucode *uc, const PrspNativeSnap *before, unsigned iters, unsigned bytes, int record);
double armlet_hle_bench(const struct hle_ucode *uc, const PrspNativeSnap *before, unsigned iters);   /* ns per front-end run, or -1 when it declines */

#include <stdio.h>
void armlet_hle_print_masks(FILE *f);    /* the dead DMEM ranges of every registered ucode, for the equivalence shim */

#ifdef __cplusplus
}
#endif

#endif
