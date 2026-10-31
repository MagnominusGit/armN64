#include "n64.h"
#include <stdio.h>
#include <unistd.h>

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

uint32_t blockPollPass(const uint32_t* ops, uint32_t count, uint32_t vaddr, int* reload) {
	*reload = 0;
	if (count < 2 || count > 16) {
		return 0;
	}
	// A conditional branch back to the start, then its delay slot.
	uint32_t branch = ops[count - 2];
	int code = branch >> 26;
	bool conditional = (code >= 0x04 && code <= 0x07) || (code >= 0x14 && code <= 0x17) || (code == 0x01 && !(((branch >> 16) & 31) & ~1u));
	if (!conditional || vaddr + 4 * (count - 1) + ((uint32_t) (int16_t) branch << 2) != vaddr) {
		return 0;
	}
	// Or a branch on a register against zero with the load of that register in its delay slot: each
	// pass then reads what the pass before loaded, the same while the branch is taken on it.
	int rs = (branch >> 21) & 31, rt = (branch >> 16) & 31;
	uint32_t delay = ops[count - 1];
	if ((code & ~0x10) >= 0x04 && (code & ~0x10) <= 0x05 && (!rs || !rt) && (rs | rt) &&
	    ((delay >> 26) == 0x23 || (delay >> 26) == 0x27) && (int) ((delay >> 16) & 31) == (rs | rt) && (int) ((delay >> 21) & 31) != (rs | rt)) {
		*reload = (rs | rt) | ((code & ~0x10) == 0x04 ? 32 : 0);
	}
	uint64_t written = 0, readFirst = 0;
	bool load = false;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k];
		uint64_t reads, writes;
		if (k == count - 2) {
			reads = *reload ? 0 : 1ull << rs | ((code & ~0x10) == 0x04 || (code & ~0x10) == 0x05 ? 1ull << rt : 0);
			writes = 0;
		} else if (!pollOp(op, &reads, &writes) || (*reload && k < count - 2 && (writes & 1ull << (*reload & 31)))) {
			*reload = 0;
			return 0;
		}
		load |= op >> 26 >= 0x20;
		readFirst |= reads & ~written;
		written |= writes;
	}
	// Each pass reads only what no pass changes, so all of them do the same.
	if (!load || ((readFirst & written) & ~1ull)) {
		*reload = 0;
		return 0;
	}
	return count;
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
// N64_PERFMAP=1: compiled blocks named by guest address in /tmp/perf-<pid>.map for perf.
void perfMapAdd(const void* code, size_t size, const char* kind, uint32_t address) {
	static FILE* perfMap;
	static int perfMapState;
	if (perfMapState == 0) {
		perfMapState = getenv("N64_PERFMAP") ? 1 : -1;
		if (perfMapState == 1) {
			char path[64];
			snprintf(path, sizeof(path), "/tmp/perf-%d.map", (int) getpid());
			perfMap = fopen(path, "w");
		}
	}
	if (perfMap) {
		fprintf(perfMap, "%lx %lx %s_%08x\n", (unsigned long) (uintptr_t) code, (unsigned long) size, kind, address);
		fflush(perfMap);
	}
}

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
	size_t before = n->jitUsed;
	if (n->useJit && !(b->code = jitCompile(n, b->ops, count, vaddr, paddr))) {
		// The code arena is full: start it again, and run this instruction on its own.
		free(b);
		blocksFree(n);
		return NULL;
	}
	if (b->code) {
		perfMapAdd(b->code, n->jitUsed - before, "n64", vaddr);
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

void idleLoad(struct N64* n, uint32_t paddr, int size, uint64_t value) {
	struct Idle* d = &n->idle;
	if (paddr >= RDRAM_SIZE || d->loads == (int) (sizeof(d->load) / sizeof(d->load[0]))) {
		d->fatal = true;
		return;
	}
	d->load[d->loads++] = (struct IdleLoad) { paddr, size, value };
}

static uint64_t rdramValue(const struct N64* n, uint32_t paddr, int size) {
	uint64_t v = 0;
	for (int i = 0; i < size; ++i) {
		v = v << 8 | n->rdram[(paddr + (uint32_t) i) & (RDRAM_SIZE - 1)];
	}
	return v;
}

void idleStore(struct N64* n, uint32_t paddr, int size, uint64_t value) {
	if (paddr >= RDRAM_SIZE) {
		n->idle.fatal = true;
	} else if (rdramValue(n, paddr, size) != (size == 8 ? value : value & ((1ull << (size * 8)) - 1))) {
		n->idle.changed = true;
	}
}

// Interpreter steps until pc (not in a delay slot) after at least one, within limit steps and the
// slice; false when it doesn't get there or something fatal happened.
static bool idleStepTo(struct N64* n, uint32_t pc, int limit) {
	struct CPU* c = &n->cpu;
	for (int k = 0; k < limit && n->cycles < n->eventAt; ++k) {
		cpuStep(n);
		n->cycles += 1 + c->stall;
		c->stall = 0;
		if (n->idle.fatal) {
			return false;
		}
		if ((uint32_t) c->pc == pc && !c->nextInDelay) {
			return true;
		}
	}
	return false;
}

struct IdleRegs {
	uint64_t gpr[32], hi, lo, fpr[32];
	uint32_t fcr31;
};

static void idleRegs(const struct CPU* c, struct IdleRegs* out) {
	memset(out, 0, sizeof(*out));
	memcpy(out->gpr, c->gpr, sizeof(out->gpr));
	memcpy(out->fpr, c->fpr, sizeof(out->fpr));
	out->hi = c->hi, out->lo = c->lo, out->fcr31 = c->fcr31;
}

// Interpreter steps until pc again with the registers as given (a loop inside the pass goes round
// as it needs), within limit steps and the slice; false as idleStepTo.
static bool idleStepBack(struct N64* n, uint32_t pc, const struct IdleRegs* regs, int limit) {
	struct CPU* c = &n->cpu;
	struct IdleRegs now;
	for (int k = 0; k < limit && n->cycles < n->eventAt; ++k) {
		cpuStep(n);
		n->cycles += 1 + c->stall;
		c->stall = 0;
		if (n->idle.fatal) {
			return false;
		}
		if ((uint32_t) c->pc == pc && !c->nextInDelay) {
			idleRegs(c, &now);
			if (!memcmp(&now, regs, sizeof(now))) {
				return true;
			}
		}
	}
	return false;
}

// A loop that only waits: two passes from here back to here, the second leaving every register as
// it found it, storing nothing new and touching no register of the RCP. It then goes round the
// same until something it loads changes, so the rest of each slice is skipped (interrupts and other
// events still come at their times; hle only, the passes skipped aren't timed exactly).
static bool idleRun(struct N64* n) {
	struct Idle* d = &n->idle;
	struct CPU* c = &n->cpu;
	enum { PASS_LIMIT = 256 };
	if (d->on) {
		// The pass goes the same while no interrupt is due and its loads read what they did: on to
		// the slice's end or the timer's interrupt, whichever comes first.
		if ((uint32_t) c->pc != d->pc || c->nextInDelay) {
			// Back round to where the loop was traced (a slice ran the loop as it is).
			d->tracing = true, d->fatal = false, d->loads = d->passLoads;
			bool back = idleStepTo(n, d->pc, PASS_LIMIT);
			d->tracing = false;
			if (!back) {
				d->on = false;
				return false;
			}
		}
		// An interrupt raised and unmasked stops it too, enabled or not here (the pass may enable
		// interrupts on its way round).
		bool same = cpuJitReady(n) && n->jitLimit > n->cycles && !(c->cop0[CP0_CAUSE] & c->cop0[CP0_STATUS] & 0xFF00);
		for (int i = 0; same && i < d->passLoads; ++i) {
			same = rdramValue(n, d->load[i].addr, d->load[i].size) == d->load[i].value;
		}
		if (!same) {
			d->on = false;
			return false;
		}
		++n->stats.idleSlices;
		n->stats.idleCycles += n->jitLimit - n->cycles;
		n->cycles = n->jitLimit;
		return true;
	}
	if (d->near < 4 || c->nextInDelay) {
		return false;
	}
	d->near = 0;
	uint32_t pc = (uint32_t) c->pc;
	struct IdleRegs before, after;
	idleRegs(c, &before);
	d->tracing = true, d->fatal = false, d->changed = false;
	bool once = idleStepBack(n, pc, &before, PASS_LIMIT);
	d->changed = false, d->loads = 0;
	bool twice = once && idleStepBack(n, pc, &before, PASS_LIMIT);
	idleRegs(c, &after);
	d->tracing = false;
	if (!twice || d->changed || d->loads > (int) (sizeof(d->load) / sizeof(d->load[0])) / 2 ||
	    memcmp(&before, &after, sizeof(before))) {
		d->backoff = 64;
		return false;
	}
	d->passLoads = d->loads;
	d->on = true;
	d->pc = pc;
	return false;
}

// At the end of a slice: where the CPU is, to see it staying in a small loop (with a call, two
// places): the last slices ended in at most two 512-byte regions.
static void idleWatch(struct N64* n) {
	struct Idle* d = &n->idle;
	enum { RECENT = sizeof(d->recent) / sizeof(d->recent[0]) };
	memmove(d->recent + 1, d->recent, sizeof(d->recent) - sizeof(d->recent[0]));
	d->recent[0] = (uint32_t) n->cpu.pc >> 9;
	if (d->backoff) {
		--d->backoff;
		d->near = 0;
		return;
	}
	uint32_t a = d->recent[0], b = a;
	bool few = true;
	for (int i = 1; i < RECENT && few; ++i) {
		if (d->recent[i] != a) {
			few = b == a || d->recent[i] == b;
			b = d->recent[i];
		}
	}
	d->near = few ? 4 : 0;
}

void blocksRun(struct N64* n) {
	struct CPU* c = &n->cpu;
	if (n->idle.enabled) {
		if (idleRun(n)) {
			return;
		}
	}
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
	if (n->idle.enabled && !n->idle.on) {
		idleWatch(n);
	}
}
