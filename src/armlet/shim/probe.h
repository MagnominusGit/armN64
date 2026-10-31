// armN64 has no probe of unknown builds (arMLEt's probe.c): only the registered builds run.
#ifndef ARMLET_PROBE_H
#define ARMLET_PROBE_H

#include <stdint.h>

static inline void armlet_probe_note_load(uint32_t text, uint32_t data, uint32_t dsize) {
	(void) text, (void) data, (void) dsize;
}

#endif
