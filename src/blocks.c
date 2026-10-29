#include "n64.h"

#include <stdlib.h>
#include <string.h>

// Blocks of instructions fetched once from cached RDRAM code, run by the interpreter's own step
// (cpuStepOp) without fetching each instruction again. A write to a page throws away its blocks.

enum { BLOCK_MAX = 64, PAGE_WORDS = 1024 };

struct Block {
	void* code;
	// The virtual address the code was compiled for (TLB-mapped code can move).
	uint32_t vaddr;
	uint32_t count;
	uint32_t ops[BLOCK_MAX];
};

// Jumps and branches, whose delay slot ends a block.
static bool isBranch(uint32_t op) {
	switch (op >> 26) {
	case 0x00:
		return (op & 0x3E) == 0x08;
	case 0x01:
		return ((op >> 16) & 0x0C) == 0;
	case 0x02: case 0x03: case 0x04: case 0x05: case 0x06: case 0x07:
	case 0x14: case 0x15: case 0x16: case 0x17:
		return true;
	case 0x11:
		return ((op >> 21) & 31) == 0x08;
	}
	return false;
}

// The idle loop "b self; nop" (or "j self; nop"): nothing changes but the cycle count, two cycles a
// pass once its I-cache lines are in.
bool blockIsIdleLoop(const uint32_t* ops, uint32_t count, uint32_t vaddr) {
	uint32_t jumpSelf = 0x08000000 | ((vaddr >> 2) & 0x03FFFFFF);
	return count == 2 && (ops[0] == 0x1000FFFF || ops[0] == jumpSelf) && ops[1] == 0;
}

void blocksInvalidatePage(struct N64* n, uint32_t page) {
	struct Block** t = n->blockPage[page];
	if (t) {
		for (int i = 0; i < PAGE_WORDS; ++i) {
			free(t[i]);
		}
		free(t);
		n->blockPage[page] = NULL;
	}
	n->pageFlags[page] &= ~PAGE_CODE;
	memset(&n->codeWords[page << 7], 0, 128);
	++n->blockGen;
}

void blocksFree(struct N64* n) {
	for (uint32_t p = 0; p < (RDRAM_SIZE >> 12); ++p) {
		if (n->blockPage[p]) {
			blocksInvalidatePage(n, p);
		}
	}
	jitReset(n);
}

// From paddr to a branch's delay slot, the end of the page or BLOCK_MAX instructions.
static struct Block* compile(struct N64* n, uint32_t vaddr, uint32_t paddr) {
	struct Block* b = malloc(sizeof(*b));
	if (!b) {
		return NULL;
	}
	uint32_t count = 0, a = paddr;
	bool delay = false;
	while (count < BLOCK_MAX) {
		uint32_t op = be32(&n->rdram[a]);
		b->ops[count++] = op;
		a += 4;
		if (delay || !(a & 0xFFF)) {
			break;
		}
		delay = isBranch(op);
	}
	b->count = count;
	b->vaddr = vaddr;
	b->code = NULL;
	if (n->useJit && !(b->code = jitCompile(n, b->ops, count, vaddr, paddr))) {
		// The code arena is full: start it again, and run this instruction on its own.
		free(b);
		blocksFree(n);
		return NULL;
	}
	return b;
}

static void jitRun(struct N64* n, void* code) {
	if (cpuJitReady(n)) {
		n->jitEnter(n, code);
	} else {
		cpuStep(n);
		n->cycles += 1 + n->cpu.stall;
		n->cpu.stall = 0;
	}
}

void blocksRun(struct N64* n) {
	struct CPU* c = &n->cpu;
	do {
		uint32_t v = (uint32_t) c->pc;
		struct JitEntry* je = &n->jitTable[(v >> 2) & (JIT_TABLE_SIZE - 1)];
		if (je->vaddr == v && je->gen == n->blockGen && c->pc == (uint64_t) (int64_t) (int32_t) v && !c->nextInDelay) {
			jitRun(n, je->code);
			continue;
		}
		uint32_t paddr = v & 0x1FFFFFFF;
		// Cached RDRAM code (kseg0, or mapped through the TLB) runs from blocks; the rest, and a
		// delay slot reached by leaving a block early, takes the plain step.
		bool mapped = (v & 0xC0000000) != 0x80000000;
		if ((v & 0xE0000000) == 0xA0000000 || (v & 3) || c->pc != (uint64_t) (int64_t) (int32_t) v ||
		    c->nextInDelay || (mapped && !cpuPeekAddress(n, v, &paddr)) || paddr >= RDRAM_SIZE) {
			cpuStep(n);
			n->cycles += 1 + c->stall;
			c->stall = 0;
			continue;
		}
		uint32_t page = paddr >> 12;
		struct Block** t = n->blockPage[page];
		if (!t && !(t = n->blockPage[page] = calloc(PAGE_WORDS, sizeof(*t)))) {
			cpuStep(n);
			n->cycles += 1 + c->stall;
			c->stall = 0;
			continue;
		}
		struct Block* b = t[(paddr >> 2) & (PAGE_WORDS - 1)];
		if (b && b->vaddr != v) {
			free(b);
			b = t[(paddr >> 2) & (PAGE_WORDS - 1)] = NULL;
		}
		if (!b) {
			// A full code arena throws every block away, this page's table with them.
			if (!(b = compile(n, v, paddr))) {
				cpuStep(n);
				n->cycles += 1 + c->stall;
				c->stall = 0;
				continue;
			}
			t[(paddr >> 2) & (PAGE_WORDS - 1)] = b;
			n->pageFlags[page] |= PAGE_CODE;
			for (uint32_t w = paddr >> 2; w < (paddr >> 2) + b->count; ++w) {
				n->codeWords[w >> 3] |= (uint8_t) (1 << (w & 7));
			}
		}
		if (b->code) {
			*je = (struct JitEntry) { v, n->blockGen, b->code };
			jitRun(n, b->code);
			continue;
		}
		uint64_t pc = c->pc;
		uint32_t gen = n->blockGen;
		for (uint32_t i = 0; i < b->count; ++i) {
			cpuStepOp(n, b->ops[i], paddr + i * 4);
			n->cycles += 1 + c->stall;
			c->stall = 0;
			pc += 4;
			// Stop at the next event, where the program didn't go on to the next instruction,
			// or when a write threw this block away.
			if (n->cycles >= n->eventAt || c->pc != pc || n->blockGen != gen) {
				break;
			}
		}
	} while (n->cycles < n->eventAt);
}
