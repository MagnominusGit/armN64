// armN64's stand-in for arMLEt's microcode identification (src/armlet.c).
#ifndef ARMLET_GFX_UCODE_H
#define ARMLET_GFX_UCODE_H

#include "armlet.h"

struct armlet_ucode_id {
	uint64_t id;
	unsigned family;
	char name[160];
};

// FNV-1a-64 over the 32-bit words of RDRAM at addr (armN64's RDRAM is big-endian bytes).
uint64_t armlet_ucode_hash(const unsigned char* rdram, uint32_t addr, uint32_t bytes);
bool armlet_gfx_allowed(const struct armlet_config* cfg, uint64_t id);

#endif
