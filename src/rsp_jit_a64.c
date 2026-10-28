#include "n64.h"

// The RSP runs in its interpreter on AArch64 for now.

void* rspJitLookup(struct N64* n, uint32_t pc) {
	return NULL;
}

void rspJitFree(struct N64* n) {
}
