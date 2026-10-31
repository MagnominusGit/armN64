/*
 * arMLEt - hle.h: the task-level native graphics path (docs/design.md, "Task-level native path")
 *
 * A graphics task is run as a whole by C code that works on the RSP's real memories (DMEM image, IMEM, RDRAM), with the microcode's own memory layout, instead of by the
 * JIT. What the CPU and the RDP can observe afterwards must be what the microcode would have left: DMEM and IMEM (except the ranges a family's table declares dead),
 * the RDRAM the microcode writes, the RDP lists it submits, the SP and DPC registers, MI_INTR. The vector unit, the scalar registers and the accumulator are not part
 * of it: the next task reloads them. Anything a task does that the path does not implement makes the whole task decline: nothing is changed, and the JIT runs the task.
 *
 * Layers: a family (F3D, F3DEX, F3DEX2 ...) is a front-end, `struct hle_family`: it runs the task. A build of a microcode is DATA, `struct hle_ucode`, registered
 * with HLE_UCODE() and found by the ucode id (gfx_ucode.c). The shared library (hle_fx.h and the hle_*.c files) holds what is the same across families.
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_HLE_H
#define ARMLET_HLE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct armlet_ctx;
struct armlet_ucode_id;

#ifdef __cplusplus
extern "C" {
#endif

/* ---- results of a task / of a handler ---------------------------------------------------------------------------------------------------------------------------- */
enum { HLE_OK = 0, HLE_DECLINE = 1 };

/* ---- the layout of a build: DMEM offsets, by name. A family fills the names it uses (a missing one is 0 and the family must not ask for it). ------------------- */
enum hle_off {
    HLE_OFF_INPUT,          /* DL input buffer (the microcode fetches the display list in chunks of HLE_OFF_INPUT_SIZE bytes) */
    HLE_OFF_INPUT_SIZE,
    HLE_OFF_OUTPUT,         /* RDP output buffer (commands are assembled here, flushed to the RDRAM FIFO with a DMA) */
    HLE_OFF_OUTPUT_SIZE,
    HLE_OFF_VTX_BUF,        /* first vertex slot */
    HLE_OFF_VTX_STRIDE,     /* bytes per slot (0x28 in F3DEX2) */
    HLE_OFF_VTX_ADDR_TABLE, /* halfword table: slot number * 2 -> slot address */
    HLE_OFF_MVP,            /* the cached model-view-projection matrix (integer part, then fraction) */
    HLE_OFF_MATRIX_STACK,
    HLE_OFF_LIGHTS,
    HLE_OFF_VIEWPORT,
    HLE_OFF_SEGMENTS,       /* 16 words: segment number * 4 -> base */
    HLE_OFF_GEOM_MODE,
    HLE_OFF_OTHERMODE,
    HLE_OFF_FIFO_STATE,     /* RDP FIFO pointers and bounds */
    HLE_OFF_DL_STACK,
    HLE_OFF_DL_DEPTH,
    HLE_OFF_TASK,           /* the OSTask header (0xfc0 on every build) */
    HLE_OFF_COUNT = 64
};

struct hle_layout {
    uint16_t off[HLE_OFF_COUNT];
};

/* ---- flags of a build ----------------------------------------------------------------------------------------------------------------------------------------- */
enum {
    HLE_F_NEAR_CLIP = 1u << 0,      /* clips against the near plane (F3DEX2 without .NoN) */
    HLE_F_ZEX = 1u << 1,            /* F3DZEX (OoT / MM) differences */
    HLE_F_LEGACY_TRI = 1u << 2,     /* the two-level dispatch families */
    HLE_F_XBUS_OK = 1u << 3,        /* a task may start with DPC_STATUS.XBUS set (the build's lists go through a DMEM ring; the commit replays the list bytes it recorded) */
    HLE_F_REGS_LIVE = 1u << 4       /* the next task reads some of the scalar registers this one leaves (hle_ucode.live_sregs; ARMLET_HLE_POISON=1 changes the results of Jet Force Gemini and Indiana Jones
                                       if they are poisoned): verify compares those with the JIT's, the poison leaves them alone. The vector unit is never live. */
};

/* A DMEM range declared dead at the end of a task (scratch / leftovers). Each entry cites how it was established; see docs/design.md "DMEM masks". The verify compare,
 * the poison probe and tools/equiv all use exactly these. */
struct hle_dead_range {
    uint16_t lo, hi;                /* [lo, hi) bytes */
};

struct hle_task;
struct hle_ucode;
struct hle_gate;                    /* the command-kind gate of a derived build (probe.c, validate.c; docs/design.md "The background validator and the latch policy") */
struct hle_dx;                      /* a detached replay: the task runs on memories and registers of its own, off the emulation thread (validate.c) */

struct hle_family {
    const char *name;               /* "f3dex2" */
    /* Runs a whole fresh task: the boot code's effects, the display list, the end of the task. HLE_OK, or HLE_DECLINE with anything it changed undoable (the framework
     * restores DMEM, IMEM, the SP registers and RDRAM from its own snapshot and undo log; nothing reached the RDP). */
    int (*run)(struct hle_task *t, const struct hle_ucode *u);
    /* PRSP_POLL_IDLE (docs/design.md, "Waits for the CPU"): a family that can take a task up again after the JIT's idle-poll cut, and can end a call there itself, gives the pc the
     * cut leaves (the wait pc of the build, or 0: none); dispatch then offers a call that starts there to run(), with t->start_pc != 0. NULL: not resumable. */
    unsigned (*wait_pc)(const struct hle_ucode *u, const uint32_t *imem);
    /* the call at the wait pc when nothing changed since the cut, without the framework's snapshot and commit: 1 = done (the state is what the JIT's idle slice leaves), 0 = run() */
    int (*idle)(struct hle_task *t, const struct hle_ucode *u);
    /* a yielded task's resume (OSTask flags bit 0: the boot code restores the state the yield saved) may run natively: run() runs the boot code on the interpreter */
    int yield_ok;
};

struct hle_ucode {
    uint64_t id;                    /* FNV-1a-64 of the text, as gfx_ucode.c logs it */
    const char *name;
    const struct hle_family *fam;
    struct hle_layout lay;
    uint32_t flags;                 /* HLE_F_* */
    /* Variants WITHIN a family: the builds of F3DEX 1.x share one vertex and triangle code with other constants, F3DLX has another vertex record, another triangle
     * code and other clip flags. The numbers are the family's own (an enum in its front-end); the front-end picks the code path by them, the tables hold the values. */
    uint8_t vtx_variant;            /* the vertex record layout / vertex code */
    uint8_t tri_variant;            /* the triangle setup code */
    uint8_t clip_variant;           /* the clip-flag computation and the clipping code */
    unsigned (*rdp_cmd_size)(unsigned first_byte);   /* the size in bytes of the RDP command that starts with this byte, when the ucode's lists have commands the default sizing gets wrong (a triangle whose shade
                                       block is not announced by its command byte: Wave Race); NULL: hle_rdp_cmd_size (hle_internal.h) */
    uint16_t ring_off;              /* DMEM address of the two words that bound the RDP output ring in RDRAM (start, end) at the start of a task (the OSTask output buffer: 0xfe8 on F3DEX2), or 0.
                                       The ring is treated like a DMEM mask: its contents are not compared (the RDP stream is), because nothing reads them (ARMLET_DMEM_POISON=ring) */
    const void *fam_data;           /* the family's per-build constants (its own struct: PCs, cull bits, constants), or NULL */
    uint32_t live_sregs;            /* with HLE_F_REGS_LIVE: bit r = scalar register r is read by a later task */
    const struct hle_dead_range *dead;  /* DMEM masks, or NULL */
    unsigned n_dead;
    struct hle_gate *gate;          /* a build the probe derived: which command kinds it may run (latch policy); NULL for the registered builds (everything) */
};

/* Registration: a table entry in the section "hle_ucodes" (no shared file to edit). */
#define HLE_UCODE(var) \
    static const struct hle_ucode var; \
    static const struct hle_ucode *const var##_ptr __attribute__((used, section("hle_ucodes"))) = &var; \
    static const struct hle_ucode var

/* ---- the task context ------------------------------------------------------------------------------------------------------------------------------------------ */
struct hle_task {
    uint32_t *dmem;                 /* the real DMEM image: host words holding the big-endian words (byte a is at ((uint8_t *)dmem)[a ^ 3]) */
    uint32_t *imem;
    uint8_t *rdram;
    uint32_t *const *cr;            /* the SP / DPC registers by MTC0 number: 0 SP_MEM_ADDR, 1 SP_DRAM_ADDR, 2 RD_LEN, 3 WR_LEN, 4 SP_STATUS, 7 SEMAPHORE, 8 DPC_START, 9 DPC_END, 10 CURRENT, 11 STATUS */
    uint32_t *irq;                  /* MI_INTR */
    const struct hle_ucode *uc;
    struct armlet_ctx *ctx;
    void *fam;                      /* the family's own per-task state (it points it at storage of its own) */
    void *priv;                     /* the framework's */
    unsigned start_pc;              /* SP_PC at the start of the task */
    int log;                        /* nonzero: RDRAM writes and RDP register writes are logged for the commit (the task-level run; not the step routine) */
    int switched;                   /* set by a family that took another build over in the middle of the task (G_LOAD_UCODE): the verify compare of the RDP stream ignores the scratch bytes behind one-byte commands */
    struct hle_dx *dx;              /* a detached replay (validate.c): the SP registers, the DMA, the RDP lists and the interpreter's registers are its own; NULL for the real run */
    const struct hle_gate *gate;    /* the gate of the build the loop runs now (hle_bind), or NULL */
    int cut_ok;                     /* the interpreter may end the call at an idle-poll cut (the JIT's PRSP_POLL_IDLE halt of a wait for the CPU) instead of declining it: hle_cut */
    uint32_t *cmds;                 /* the command kinds the loop ran, counted (a detached replay of the build under validation), or NULL */
};

/* DMEM accessors (big-endian view). DMEM and RDRAM are arrays of 32-bit words that are also accessed as halfwords and bytes: through types that may alias. */
typedef uint16_t __attribute__((may_alias)) hle_u16a;
typedef uint32_t __attribute__((may_alias)) hle_u32a;
static inline uint8_t hle_dm8(const struct hle_task *t, unsigned a) { return ((const uint8_t *)t->dmem)[(a & 0xfff) ^ 3]; }
static inline uint16_t hle_dm16(const struct hle_task *t, unsigned a) { return ((const hle_u16a *)t->dmem)[((a & 0xfff) ^ 2) >> 1]; }
static inline uint32_t hle_dm32(const struct hle_task *t, unsigned a) { return t->dmem[(a & 0xfff) >> 2]; }
static inline void hle_st8(struct hle_task *t, unsigned a, uint8_t v) { ((uint8_t *)t->dmem)[(a & 0xfff) ^ 3] = v; }
static inline void hle_st16(struct hle_task *t, unsigned a, uint16_t v) { ((hle_u16a *)t->dmem)[((a & 0xfff) ^ 2) >> 1] = v; }
static inline void hle_st32(struct hle_task *t, unsigned a, uint32_t v) { t->dmem[(a & 0xfff) >> 2] = v; }
/* RDRAM accessors (big-endian view, byte address). armN64: RDRAM is big-endian bytes, so a host word of DMEM's layout is read from it a byte at a time, and the helpers below copy and compare
 * RDRAM against the DMEM / IMEM images. */
static inline uint32_t hle_rd32(const struct hle_task *t, uint32_t a)
{
    const uint8_t *p = t->rdram + (a & 0x7ffffc);
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}
static inline uint32_t hle_ram32(const uint8_t *rdram, uint32_t a)
{
    const uint8_t *p = rdram + (a & 0x7ffffc);
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}
static inline void hle_ram_to_words(uint32_t *words, const uint8_t *rdram, uint32_t a, uint32_t len)
{
    for (uint32_t i = 0; i < len; i += 4)
        words[i >> 2] = hle_ram32(rdram, a + i);
}
static inline int hle_ram_cmp_words(const uint32_t *words, const uint8_t *rdram, uint32_t a, uint32_t len)
{
    for (uint32_t i = 0; i < len; i += 4)
        if (words[i >> 2] != hle_ram32(rdram, a + i))
            return 1;
    return 0;
}

/* ---- shared library: the pieces every family uses (src/hle/hle_core.c; the arithmetic is in hle_fx.h) ---------------------------------------------------------------- */
/* A family gives up on the task: the framework restores everything and the JIT runs it. `why` is a static string (statistics count the reasons). Returns HLE_DECLINE. */
int hle_decline(struct hle_task *t, const char *why);
/* SP DMA through the fork's own DMA code (so the SP registers, IMEM dirty bits and RDRAM end exactly as the microcode leaves them); len is the register
 * value (bytes - 1 | rows << 12 | skip << 20). Reads from RDRAM into DMEM / IMEM (mem_addr bit 12), writes DMEM to RDRAM. */
void hle_dma_read(struct hle_task *t, unsigned mem_addr, unsigned dram_addr, uint32_t len);
void hle_dma_write(struct hle_task *t, unsigned mem_addr, unsigned dram_addr, uint32_t len);
/* A write of RDRAM that stands for an SP DMA write (nbytes a multiple of 8, dram 8-aligned) without going through the fork's DMA: the same effect for the commit (the log), the sandbox and
 * the verify, at a fraction of the cost. words = the bytes as the host words of DMEM / RDRAM hold them. */
void hle_rdram_write(struct hle_task *t, uint32_t dram, const uint32_t *words, uint32_t nbytes);
/* MTC0 to an SP / DPC register as the microcode does it. A write to DPC_END hands the list [DPC_CURRENT, end) to the RDP: buffered until the task commits, then
 * replayed in order with the DMA writes through the coalescing path (so the video plugin sees exactly what it sees after a JIT run). DPC_START and DPC_STATUS
 * writes are replayed too. Returns nonzero when the write ended the run (the SP halted). */
int hle_mtc0(struct hle_task *t, unsigned rd, uint32_t value);
/* MFC0 of SP_STATUS into register rt as the microcode does it (the poll counting): returns 0 and the value, or nonzero when the JIT would halt on the poll. */
int hle_poll_status(struct hle_task *t, unsigned rt, uint32_t *value);
/* The end of the task as the BREAK instruction makes it (SP_STATUS BROKE | HALT, MI_INTR when INTR_BREAK is set); end_pc is the pc the RSP is left at. */
void hle_break(struct hle_task *t, unsigned end_pc);
/* The end of a call at an idle-poll cut (PRSP_POLL_IDLE): the task goes on in a later call at end_pc. SP_STATUS is left as the JIT leaves it (HALT cleared again). */
void hle_cut(struct hle_task *t, unsigned end_pc);
/* What a cut leaves that the next call reads, and that the verify compare takes as the JIT leaves it though a task's end would not need it: the DMEM ranges [lo, hi) x 2 that the family declares dead at the end
 * of a task (the display list input buffer; the RDP output not flushed yet, which is a list of RDP commands: the words the RDP ignores are not compared), and the scalar registers (bit k = register k) the
 * microcode's loop carries from one command to the next. The others hold the temporaries of the commands the JIT ran and the front-end did not (the loop writes them before it reads them: docs/design.md). */
void hle_cut_live(struct hle_task *t, unsigned lo0, unsigned hi0, unsigned lo1, unsigned hi1, uint32_t sregs);

/* The command-kind hook of a front-end's loop (docs/design.md, "The background validator and the latch policy"): `kind` is the command's opcode byte, or a risky path (LATCH_K_CLIP,
 * LATCH_K_LIT, LATCH_K_LOAD, LATCH_K_CULLDL in latch.h). Counts it for a replay under validation; declines the task (nonzero: the caller returns HLE_DECLINE) when the build is a latched
 * derived build that has not run this kind in a validated task. Costs one test when neither applies. */
int hle_cmd_slow(struct hle_task *t, unsigned kind);
static inline int hle_cmd(struct hle_task *t, unsigned kind) { return (t->gate || t->cmds) ? hle_cmd_slow(t, kind) : 0; }
/* The data situations of a G_VTX (latch.h): the OR of the vertices' clip-flag words, and whether one has clip w <= 0. Nonzero: declined (as hle_cmd). */
int hle_vtx_kinds_slow(struct hle_task *t, uint32_t flags_or, int wneg);
static inline int hle_vtx_kinds(struct hle_task *t, uint32_t flags_or, int wneg) { return (t->gate || t->cmds) ? hle_vtx_kinds_slow(t, flags_or, wneg) : 0; }
static inline int hle_wneg(int16_t wi, uint16_t wf) { return wi < 0 || (wi == 0 && wf == 0); }
/* The loop works with build `u` from now on (the task's own at the start, a G_LOAD_UCODE's afterwards): its gate, and the counting of a replay under validation */
void hle_bind(struct hle_task *t, const struct hle_ucode *u);
/* Whether a derived build may be taken over by a G_LOAD_UCODE: latched (in a replay under validation: any that is not unlatched for good). Registered builds: always. */
int hle_build_usable(const struct hle_task *t, const struct hle_ucode *u);

/* nonzero in the exact-events mode (ARMLET_HLE_EXACT=1): a front-end then does every SP DMA and every IMEM load the microcode does, in its order, instead of only what the external effects need */
int hle_exact(void);
void hle_dev_mark(size_t m[2]);              /* development: the commit log's length ... */
void hle_dev_rewind(const size_t m[2]);      /* ... and back to it */

/* ---- the RSP interpreter (hle_rsp.cpp) ------------------------------------------------------------------------------------------------------------------------------ */
/* The microcode's own code, run by an interpreter on the real DMEM / IMEM and the JIT's register file (the vector unit through the fork's per-instruction helpers), with
 * the SP / DPC registers and the DMA going through the calls above. It is the exact reference of the task-level path: verify can compare a native front-end with it, the capture
 * harness replays tasks with it, and a family without native code can run entirely on it (GfxMode=hle with ARMLET_HLE_INTERP=1; slower than the JIT, never the default). */
enum { HLE_REG_AT = 1, HLE_REG_V0 = 2, HLE_REG_V1 = 3, HLE_REG_A0 = 4, HLE_REG_T0 = 8, HLE_REG_T3 = 11, HLE_REG_T4 = 12, HLE_REG_S3 = 19, HLE_REG_S4 = 20, HLE_REG_S6 = 22,
       HLE_REG_S7 = 23, HLE_REG_T8 = 24, HLE_REG_T9 = 25, HLE_REG_K0 = 26, HLE_REG_K1 = 27, HLE_REG_RA = 31 };
enum { HLE_STOP = 3 };          /* hle_rsp_run's other result: stop_pc was reached */

struct hle_cpu {
    uint32_t *r;                    /* the scalar registers: the JIT's own array, r[0] is kept 0 */
    unsigned pc;                    /* byte offset in IMEM, word aligned */
    unsigned steps, max_steps;
    unsigned stop_pc;               /* hle_rsp_run returns HLE_STOP when it reaches this pc after at least one instruction (0x1000: never) */
};
void hle_cpu_init(struct hle_task *t, struct hle_cpu *c, unsigned pc, unsigned max_steps);
/* Runs from c->pc until the task ends (HLE_OK: hle_break was called), or a reason to decline (HLE_DECLINE, hle_decline called). */
int hle_rsp_run(struct hle_task *t, struct hle_cpu *c);
/* A whole task on the interpreter, from the task's start pc (the family run() of a build with no native front-end) */
int hle_interp_task(struct hle_task *t);

/* the table (hle_core.c) */
const struct hle_ucode *armlet_hle_find(uint64_t id);
const struct hle_ucode *armlet_hle_next(const struct hle_ucode *prev);   /* the registered builds in turn (prev NULL: the first); NULL at the end */
/* the handler dispatch.c calls for a task of a ucode with a table entry (gfx_ucode.c): nonzero when the task was completed natively */
int armlet_hle_run(struct armlet_ctx *c, const struct armlet_ucode_id *u);

/* ARMLET_DMEM_POISON (hle_task.c): overwrites the ranges with a pattern at the end of a graphics task, whoever ran it; `uc` supplies the "mask" ranges (NULL: none) */
void armlet_hle_poison(const struct hle_ucode *uc);

/* framework entry points (hle_task.c); all compiled out without ARMLET_NATIVE */
void armlet_hle_init(struct armlet_ctx *c);        /* after parallelRSPInitiateRSP */
/* Before the JIT runs a graphics task (SP_PC at the start of a fresh task). Returns true when the task was run natively and committed. */
bool armlet_hle_task_begin(struct armlet_ctx *c);
/* After the JIT ran the task (not called when the native path handled it): finishes a capture or a verify. */
void armlet_hle_task_end(struct armlet_ctx *c);
void armlet_hle_reset(void);
void armlet_hle_core_reset(void);                  /* RomClosed: the statistics go to the log */

#ifdef __cplusplus
}
#endif

#endif
