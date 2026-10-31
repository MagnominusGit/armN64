/*
 * arMLEt - gate.h: the command-kind gate of a derived build (hle.h: struct hle_gate). Written by the validator thread (validate.c) and the emulation thread (probe.c), read by the
 * front-ends' loops (hle_cmd) on the emulation thread: atomics, each word on its own (a kind that becomes allowed a task later than it could have is harmless).
 *
 * Copyright (C) 2026 arMLEt contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ARMLET_GATE_H
#define ARMLET_GATE_H

#include <stdatomic.h>
#include <stdint.h>

#include "latch.h"

struct hle_gate {
    _Atomic uint64_t kinds[LATCH_KWORDS];   /* the kinds that were in a validated task (latch_state.kinds) */
    _Atomic int latched;                    /* the build is latched (a G_LOAD_UCODE may take it over) */
    _Atomic int open;                       /* every kind (ARMLET_PROBE_LATCH=1, the test override) */
    _Atomic int dead;                       /* unlatched for good: not even a replay takes it over */
};

#endif
