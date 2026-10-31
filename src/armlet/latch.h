/*
 * arMLEt - latch.h: the latch policy of the probe's derived builds (docs/design.md, "The background validator and the latch policy")
 *
 * Pure bookkeeping, no I/O and no threads: the validator (validate.c) feeds it the outcome of each real task it replayed through a candidate build (a match with the task's
 * fingerprint and command kinds, or a mismatch), and it says when the build latches, when it is suspended and when it is unlatched for good, which command kinds a latched build
 * may run, and how often a latched build's tasks are still sampled. The numbers are tunables (struct latch_cfg; GfxLatchPolicy / ARMLET_LATCH_POLICY).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_LATCH_H
#define ARMLET_LATCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Command kinds: the opcode byte of a display list command (0 .. 255), then the risky paths and data situations a task can meet (counted once per occurrence):
 * a triangle sent to the clipper, lit vertices, a G_LOAD_UCODE of another text, a G_CULLDL that ended its list, a G_VTX with a vertex at or behind the eye (clip w <= 0), a clipped
 * triangle with such a vertex (the near plane), a triangle rejected (all vertices beyond one plane), a triangle culled or degenerate, a G_BRANCH_Z taken, and each bit of the vertex
 * clip-flag word (the planes and the guard band: low halfword LATCH_K_VFLAG + 0 .. 15, the high halfword of F3DEX2's + 16 .. 31) that a G_VTX produced */
enum { LATCH_K_CLIP = 256, LATCH_K_LIT, LATCH_K_LOAD, LATCH_K_CULLDL, LATCH_K_WNEG, LATCH_K_NEARCLIP, LATCH_K_TRI_REJECT, LATCH_K_TRI_CULL, LATCH_K_BRZ,
       LATCH_K_VFLAG = 272, LATCH_NKINDS = LATCH_K_VFLAG + 32 };
#define LATCH_KWORDS ((LATCH_NKINDS + 63) / 64)

struct latch_cfg {
    double evidence;            /* evidence an EXACT build needs to latch (default 24) */
    unsigned distinct;          /* distinct task fingerprints an EXACT build needs (default 8) */
    double near_mul;            /* a NEAR build needs near_mul times both (default 1.75) */
    unsigned strikes;           /* EXACT: the mismatch that makes this many strikes (this session and earlier ones) unlatches for good (default 2; the first suspends) */
    unsigned sample_first;      /* a latched build's tasks: one in sample_first is still validated at first (default 32) ... */
    unsigned sample_last;       /* ... halving the rate every sample_decay matches down to one in sample_last (default 256) */
    unsigned sample_decay;      /* (default 64) */
    unsigned queue;             /* samples waiting for the validator at most (default 4; more are not captured) */
    unsigned fp_max;            /* fingerprints remembered per build (default 1024; beyond, a new one counts as new) */
    double loose_mul;           /* a loose build (GfxProbe = loose: a NONE derivation admitted by the word bound) needs loose_mul times both (default 2.5) */
    double default_mul;         /* a default-record build (loose tier, a NONE derivation outside the word bound: the family's record as derived) needs default_mul times both (default 3) */
    double repeat_min;          /* the least evidence a repeat of a known fingerprint adds (default 0.05: 1 / (1 + n) alone decays to nothing, and a build whose tasks all look alike never moves) */
    unsigned resume;            /* a build latched in an earlier session starts its sampling as if it had this many matches since latching (capped by what the cache holds; default 128 = two halvings of the rate; 0: from the first rate again) */
    double override_mul;        /* evidence of override_mul times `evidence` (the base bar, whatever the tier) latches with 2 distinct fingerprints or more, whatever the distinct bar (default 4; 0:
                                   never): a build whose tasks are inherently uniform (a loaded text, a HUD) is validated hundreds of times without variety */
};
void latch_cfg_default(struct latch_cfg *c);
/* "key=value,key=value": evidence, distinct, near, strikes, sample, sample_last, decay, queue, fingerprints, loose, default, repeat, override, resume. Returns 0, or -1 with the first bad key in *bad (the rest still parsed). */
int latch_cfg_parse(struct latch_cfg *c, const char *s, const char **bad);

enum { LATCH_CAND = 0, LATCH_LATCHED = 1, LATCH_DEAD = 2 };
enum { LATCH_EV_NONE = 0, LATCH_EV_LATCH, LATCH_EV_SUSPEND, LATCH_EV_DEAD };

/* The grade of a mismatch (docs/design.md, "The loose tier"). STRUCTURAL: what a strict build's mismatch always is (a strike). NUMERIC: a loose build's difference that lies inside the loose
 * tolerance (shade coefficients, vertex colours) and nowhere else: counted, never a strike. HARD: the task ended differently (status, pc, interrupt, a DMA where the JIT made none, IMEM, the
 * registers): unlatched for good at once, whatever the tier. */
enum { LATCH_G_STRUCTURAL = 0, LATCH_G_NUMERIC, LATCH_G_HARD };
struct latch_diff {
    int hard;                   /* a difference of the HARD class was seen */
    int numeric;                /* a difference inside the loose tolerance was seen */
    int structural;             /* any other difference */
};
int latch_grade(const struct latch_diff *d);

struct latch_state {
    int near;                   /* a NEAR build: higher thresholds, one mismatch unlatches */
    int loose;                  /* a loose build: higher thresholds, mismatches graded (a NUMERIC one is counted, not a strike) */
    int deflt;                  /* (loose) a default-record build: the default_mul thresholds */
    int state;                  /* LATCH_* */
    double evidence;
    unsigned distinct, matches, mismatches;
    unsigned numeric;           /* mismatches of the NUMERIC grade (in mismatches too) */
    unsigned strikes;           /* EXACT mismatches so far (also of earlier sessions) */
    unsigned post;              /* matches since the build latched (this session): the sampling decay */
    uint64_t kinds[LATCH_KWORDS];   /* the command kinds that were in a validated task */
    uint64_t never[LATCH_KWORDS];   /* kinds a match never adds (the region gate of a loose build: a kind whose code region has unexplained words stays on the JIT) */
    uint64_t *fp;               /* the fingerprints seen in matches, with their counts: open addressing, fp_cap slots (a power of two), 0 = empty */
    uint32_t *fpn;
    unsigned fp_cap, fp_n;
};
/* near: the build is NEAR; loose: the build is loose; strikes: from the cache. 0, or -1 when there is no memory (the state is then a candidate that never latches). */
int latch_init(struct latch_state *s, const struct latch_cfg *c, int near, int loose, unsigned strikes);
void latch_free(struct latch_state *s);

/* The fingerprint of a task: the command kinds it ran with their counts bucketed (1, 2-3, 4-7, ... 128+), FNV-1a; never 0 */
uint64_t latch_fingerprint(const uint32_t hist[LATCH_NKINDS]);
/* the evidence a match with this fingerprint adds now: 1 / (1 + earlier matches of it), at least repeat_min */
double latch_weight(const struct latch_state *s, const struct latch_cfg *c, uint64_t fp);

/* kind k is never admitted by a match (set before the first match; a suspension's reset keeps it) */
void latch_never(struct latch_state *s, unsigned k);

/* A replayed task came out equal: adds its evidence and its kinds (except the never ones). Returns LATCH_EV_LATCH when the build latches with it, else LATCH_EV_NONE. */
int latch_match(struct latch_state *s, const struct latch_cfg *c, uint64_t fp, const uint32_t hist[LATCH_NKINDS]);
/* A replayed task differed, with grade LATCH_G_*: LATCH_EV_SUSPEND (EXACT, strikes left: evidence, fingerprints and kinds are reset, the build is a candidate again), LATCH_EV_DEAD (for good:
 * a NEAR build, the last strike, or a HARD grade), or LATCH_EV_NONE (a loose build's NUMERIC grade: counted only). A strict build passes LATCH_G_STRUCTURAL. */
int latch_mismatch(struct latch_state *s, const struct latch_cfg *c, int grade);

/* whether a latched build may run command kind k (it was in a validated task) */
static inline int latch_kind_ok(const uint64_t kinds[LATCH_KWORDS], unsigned k) { return k < LATCH_NKINDS && (kinds[k >> 6] >> (k & 63) & 1); }
/* one in how many of a latched build's tasks is validated now */
unsigned latch_sample_interval(const struct latch_state *s, const struct latch_cfg *c);
/* the thresholds that apply to this build */
double latch_need_evidence(const struct latch_state *s, const struct latch_cfg *c);
unsigned latch_need_distinct(const struct latch_state *s, const struct latch_cfg *c);

#ifdef __cplusplus
}
#endif

#endif
