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

// The registers an instruction of a poll loop reads and writes (bit 32 for none), or false for one
// a poll loop can't have: anything but loads, ALU instructions that can't trap, and NOP.
static bool pollOp(uint32_t op, uint64_t* reads, uint64_t* writes) {
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, rd = (op >> 11) & 31;
	switch (op >> 26) {
	case 0x00:
		switch (op & 0x3F) {
		case 0x00: case 0x02: case 0x03: case 0x38: case 0x3A: case 0x3B: case 0x3C: case 0x3E: case 0x3F:
			*reads = 1ull << rt;
			*writes = 1ull << rd;
			return true;
		case 0x04: case 0x06: case 0x07: case 0x14: case 0x16: case 0x17:
		case 0x21: case 0x23: case 0x24: case 0x25: case 0x26: case 0x27: case 0x2A: case 0x2B: case 0x2D: case 0x2F:
			*reads = 1ull << rs | 1ull << rt;
			*writes = 1ull << rd;
			return true;
		}
		return false;
	case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x0E: case 0x19:
	case 0x20: case 0x21: case 0x23: case 0x24: case 0x25: case 0x27: case 0x37:
		*reads = 1ull << rs;
		*writes = 1ull << rt;
		return true;
	case 0x0F:
		*reads = 0;
		*writes = 1ull << rt;
		return true;
	}
	return false;
}

uint32_t blockPollPass(const uint32_t* ops, uint32_t count, uint32_t vaddr) {
	if (count < 3 || count > 16) {
		return 0;
	}
	// A conditional branch back to the start, then its delay slot.
	uint32_t branch = ops[count - 2];
	int code = branch >> 26;
	bool conditional = (code >= 0x04 && code <= 0x07) || (code == 0x01 && !(((branch >> 16) & 31) & ~1u));
	if (!conditional || vaddr + 4 * (count - 1) + ((uint32_t) (int16_t) branch << 2) != vaddr) {
		return 0;
	}
	uint64_t written = 0, readFirst = 0;
	bool load = false;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k];
		uint64_t reads, writes;
		if (k == count - 2) {
			reads = 1ull << ((op >> 21) & 31) | (code == 0x04 || code == 0x05 ? 1ull << ((op >> 16) & 31) : 0);
			writes = 0;
		} else if (!pollOp(op, &reads, &writes)) {
			return 0;
		}
		load |= op >> 26 >= 0x20;
		readFirst |= reads & ~written;
		written |= writes;
	}
	// Each pass reads only what no pass changes, so all of them do the same.
	return load && !((readFirst & written) & ~1ull) ? count : 0;
}

// Clears table entries: all, those for mapped code, or those for one RDRAM page.
void jitTableClear(struct N64* n, bool mappedOnly, int page) {
	for (int i = 0; i < JIT_TABLE_SIZE; ++i) {
		struct JitEntry* je = &n->jitTable[i];
		if ((mappedOnly && je->genAt == &n->steadyGen) || (page >= 0 && je->page != (uint32_t) page)) {
			continue;
		}
		// No pc matches an odd address.
		je->vaddr = 1;
	}
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
	jitTableClear(n, false, (int) page);
	++n->blockGen;
}

void blocksFree(struct N64* n) {
	for (uint32_t p = 0; p < (RDRAM_SIZE >> 12); ++p) {
		if (n->blockPage[p]) {
			blocksInvalidatePage(n, p);
		}
	}
	jitReset(n);
	jitTableClear(n, false, -1);
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
		if (je->vaddr == v && *je->genAt == je->gen && c->pc == (uint64_t) (int64_t) (int32_t) v && !c->nextInDelay) {
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
			const uint32_t* genAt = mapped ? &n->vpageGen[v >> 12] : &n->steadyGen;
			*je = (struct JitEntry) { v, *genAt, genAt, b->code, page, 0 };
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
