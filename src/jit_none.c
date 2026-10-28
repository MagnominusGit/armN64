#include "n64.h"

// No code generator for this host: everything runs in the interpreters.

void* jitCompile(struct N64* n, const uint32_t* ops, uint32_t count, uint32_t vaddr, uint32_t paddr) {
	return NULL;
}

void jitReset(struct N64* n) {
	++n->blockGen;
}

void jitFree(struct N64* n) {
}

void* rspJitLookup(struct N64* n, uint32_t pc) {
	return NULL;
}

void rspJitFree(struct N64* n) {
}
