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

// With HLE accuracy, a block the JIT compiles goes on past a conditional branch that can go either
// way and doesn't go back to the block's start (a loop keeps its own block), when its delay slot isn't
// another branch: the branch taken leaves the block.
static bool throughBranch(const struct N64* n, uint32_t op, uint32_t delayOp, uint32_t vaddr, uint32_t at) {
	int code = op >> 26, rs = (op >> 21) & 31, rt = (op >> 16) & 31;
	bool either = (code == 0x04 && rs != rt) || code == 0x05 || (code == 0x06 && rs) || code == 0x07 || (code == 0x01 && (rt == 0 || (rt == 1 && rs)));
	return jitBranchesInside && n->fast && n->useJit && either && !isBranch(delayOp) && at + 4 + ((uint32_t) (int16_t) op << 2) != vaddr;
}

// The idle loop "b self; nop" (or "j self; nop"): nothing changes but the cycle count, two cycles a
// pass once its I-cache lines are in.
bool blockIsIdleLoop(const uint32_t* ops, uint32_t count, uint32_t vaddr) {
	uint32_t jumpSelf = 0x08000000 | ((vaddr >> 2) & 0x03FFFFFF);
	return count == 2 && (ops[0] == 0x1000FFFF || ops[0] == jumpSelf) && ops[1] == 0;
}

// Half of an idle loop over two blocks, "j there; nop" with "j here; nop" there: where it jumps to
// (0 for none). The other half is checked as it runs.
uint32_t blockIdlePair(const uint32_t* ops, uint32_t count, uint32_t vaddr) {
	if (count != 2 || ops[0] >> 26 != 0x02 || ops[1]) {
		return 0;
	}
	uint32_t const target = (vaddr & 0xF0000000) | (ops[0] & 0x03FFFFFF) << 2;
	return target != vaddr && (target & 0xFF800000) == 0x80000000 && !(target & 7) ? target : 0;
}

// A loop counting while it waits, "b self; addiu r, r, imm" (or daddiu, or j self): only r changes,
// by imm each time round. Its register (0 for none), and whether the add is a doubleword one.
int blockCountLoop(const uint32_t* ops, uint32_t count, uint32_t vaddr, int32_t* imm, bool* dword) {
	uint32_t jumpSelf = 0x08000000 | ((vaddr >> 2) & 0x03FFFFFF);
	if (count != 2 || (ops[0] != 0x1000FFFF && ops[0] != jumpSelf) || (ops[1] >> 26 != 0x09 && ops[1] >> 26 != 0x19)) {
		return 0;
	}
	int const rs = (ops[1] >> 21) & 31, rt = (ops[1] >> 16) & 31;
	*imm = (int16_t) ops[1];
	*dword = ops[1] >> 26 == 0x19;
	return rs == rt ? rt : 0;
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
	// Each pass reads only what no pass changes, so all of them do the same (without loads, a branch on
	// registers nothing changes: going round once, it goes round until an interrupt).
	(void) load;
	if ((readFirst & written) & ~1ull) {
		*reload = 0;
		return 0;
	}
	return count;
}

// A block thrown away alone: whatever still jumps to its code finds the pc's block again.
static void dropBlock(struct N64* n, struct Block* b) {
	if (b && b->code) {
		struct JitEntry* je = &n->jitTable[(b->vaddr >> 2) & (JIT_TABLE_SIZE - 1)];
		if (je->code == b->code) {
			je->vaddr = 1;
		}
		jitKill(n, b->code);
	}
	free(b);
}

void blocksInvalidatePage(struct N64* n, uint32_t page) {
	struct Block** t = n->blockPage[page];
	if (t) {
		for (int i = 0; i < PAGE_WORDS; ++i) {
			dropBlock(n, t[i]);
		}
		free(t);
		n->blockPage[page] = NULL;
	}
	n->pageFlags[page] &= ~PAGE_CODE;
	memset(&n->codeWords[page << 7], 0, 128);
	++n->codeGen;
}

void blocksFree(struct N64* n) {
	for (uint32_t p = 0; p < (RDRAM_SIZE >> 12); ++p) {
		if (n->blockPage[p]) {
			blocksInvalidatePage(n, p);
		}
	}
	jitReset(n);
	for (uint32_t i = 0; i < JIT_TABLE_SIZE; ++i) {
		n->jitTable[i].vaddr = 1;
	}
}

// From paddr to a branch's delay slot, the end of the page or BLOCK_MAX instructions.
// N64_PERFMAP=1: compiled blocks named by guest address in /tmp/perf-<pid>.map for perf; =2 also
// their code, one after another, in /tmp/perf-<pid>.code.
void perfMapAdd(const void* code, size_t size, const char* kind, uint32_t address) {
	static FILE *perfMap, *perfCode;
	static int perfMapState;
	if (perfMapState == 0) {
		const char* map = getenv("N64_PERFMAP");
		perfMapState = map ? 1 : -1;
		if (perfMapState == 1) {
			char path[64];
			snprintf(path, sizeof(path), "/tmp/perf-%d.map", (int) getpid());
			perfMap = fopen(path, "w");
			if (map[0] == '2') {
				snprintf(path, sizeof(path), "/tmp/perf-%d.code", (int) getpid());
				perfCode = fopen(path, "wb");
			}
		}
	}
	if (perfMap) {
		fprintf(perfMap, "%lx %lx %s_%08x\n", (unsigned long) (uintptr_t) code, (unsigned long) size, kind, address);
		fflush(perfMap);
	}
	if (perfCode) {
		fwrite(code, 1, size, perfCode);
		fflush(perfCode);
	}
}

static struct Block* compile(struct N64* n, uint32_t vaddr, uint32_t paddr) {
	struct Block* b = malloc(sizeof(*b));
	if (!b) {
		return NULL;
	}
	uint32_t count = 0, a = paddr;
	// The end of the last branch gone past: a block that stops at BLOCK_MAX after one ends there
	// instead, where the code past it would start its own block.
	uint32_t through = 0;
	bool delay = false;
	while (count < BLOCK_MAX) {
		uint32_t op = be32(&n->rdram[a]);
		b->ops[count++] = op;
		a += 4;
		if (!(a & 0xFFF)) {
			break;
		}
		if (delay) {
			if (!throughBranch(n, b->ops[count - 2], op, vaddr, vaddr + 4 * (count - 2))) {
				break;
			}
			through = count;
		}
		delay = isBranch(op);
		if (count == BLOCK_MAX && through) {
			count = through;
			break;
		}
	}
	b->count = count;
	b->vaddr = vaddr;
	b->code = NULL;
	size_t before = n->jitUsed;
	// Code the I-cache holds otherwise than memory is run by the interpreter, from the I-cache.
	if (n->useJit && cpuCodeCoherent(n, vaddr, paddr, count)) {
		// Code too long for one block is compiled as a shorter block; when even a short one doesn't
		// fit, the code arena is full: start it again, and run this instruction on its own.
		while (!(b->code = jitCompile(n, b->ops, count, vaddr, paddr))) {
			if (count <= 8) {
				free(b);
				blocksFree(n);
				return NULL;
			}
			count /= 2;
			count += isBranch(b->ops[count - 1]);
		}
		b->count = count;
	}
	if (b->code) {
		perfMapAdd(b->code, n->jitUsed - before, "n64", vaddr);
	}
	return b;
}

static void jitRun(struct N64* n, void* code) {
	if (n->linkGen != n->blockGen) {
		jitUnlink(n);
	}
	if (cpuJitReady(n)) {
		n->jitEnter(n, code);
	} else {
		cpuStep(n);
		n->cycles += n->opCycles + n->cpu.stall;
		n->cpu.stall = 0;
	}
}

// The bytes a pass stored to, with what they held before it: the pass stores nothing new when they
// hold the same at its end (a thread's state made runnable and running again, say).
static struct {
	uint32_t addr;
	uint8_t was;
} passBytes[256];
static int passByteCount;

static bool passStoredNew(const struct N64* n) {
	for (int i = 0; i < passByteCount; ++i) {
		if (n->rdram[passBytes[i].addr] != passBytes[i].was) {
			return true;
		}
	}
	return false;
}

// A load in a pass: from RDRAM, or MI_INTR_MASK (a thread's switch saves it); the passes skipped
// read the same as long as these do. Bytes the pass stored first don't count.
void idleLoad(struct N64* n, uint32_t paddr, int size, uint64_t value) {
	struct Idle* d = &n->idle;
	if ((paddr >= RDRAM_SIZE && !(paddr == 0x0430000C && size == 4)) || d->loads == (int) (sizeof(d->load) / sizeof(d->load[0]))) {
		d->fatal = true;
		return;
	}
	// What the pass stored itself it reads back as it left it.
	for (int i = 0; i < passByteCount && paddr < RDRAM_SIZE; ++i) {
		if (passBytes[i].addr - paddr < (uint32_t) size) {
			return;
		}
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

// A store in a pass: to RDRAM, or to MI_INTR_MASK leaving the mask as it is (a thread dispatch
// sets it to the thread's own mask each time).
void idleStore(struct N64* n, uint32_t paddr, int size, uint64_t value) {
	if (paddr == 0x0430000C && size == 4) {
		uint32_t mask = n->miMask;
		for (int line = 0; line < 6; ++line) {
			mask = value & (1u << (line * 2)) ? mask & ~(1u << line) : mask;
			mask = value & (2u << (line * 2)) ? mask | 1u << line : mask;
		}
		n->idle.fatal |= mask != n->miMask;
	} else if (paddr >= RDRAM_SIZE) {
		n->idle.fatal = true;
	} else {
		(void) value;
		for (int k = 0; k < size; ++k) {
			uint32_t const a = paddr + (uint32_t) k;
			int i = 0;
			while (i < passByteCount && passBytes[i].addr != a) {
				++i;
			}
			if (i == passByteCount) {
				if (passByteCount == (int) (sizeof(passBytes) / sizeof(passBytes[0]))) {
					n->idle.changed = true;
					return;
				}
				passBytes[passByteCount].addr = a;
				passBytes[passByteCount++].was = n->rdram[a];
			}
		}
	}
}

// Interpreter steps until pc (not in a delay slot) after at least one, within limit steps and the
// slice; false when it doesn't get there or something fatal happened.
static bool idleStepTo(struct N64* n, uint32_t pc, int limit) {
	struct CPU* c = &n->cpu;
	for (int k = 0; k < limit && n->cycles < n->eventAt; ++k) {
		cpuStep(n);
		n->cycles += n->opCycles + c->stall;
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

// The registers a pass is compared by: all but k0 and k1, the exception handlers' own (a pass that
// yields leaves them as the dispatch did).
static void idleRegs(const struct CPU* c, struct IdleRegs* out) {
	memset(out, 0, sizeof(*out));
	memcpy(out->gpr, c->gpr, sizeof(out->gpr));
	out->gpr[26] = out->gpr[27] = 0;
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
		n->cycles += n->opCycles + c->stall;
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
	enum { PASS_LIMIT = 512 };
	if (d->on) {
		// The pass goes the same while no interrupt is due and its loads read what they did: on to
		// the slice's end or the timer's interrupt, whichever comes first.
		if ((uint32_t) c->pc != d->pc || c->nextInDelay) {
			// Back round to where the loop was traced (a slice ran the loop as it is).
			d->tracing = true, d->fatal = false, d->loads = d->passLoads, passByteCount = 0;
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
			same = (d->load[i].addr == 0x0430000C ? n->miMask : rdramValue(n, d->load[i].addr, d->load[i].size)) == d->load[i].value;
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
	d->tracing = true, d->fatal = false, d->changed = false, passByteCount = 0;
	bool once = idleStepBack(n, pc, &before, PASS_LIMIT);
	d->changed = false, d->loads = 0, passByteCount = 0;
	bool twice = once && idleStepBack(n, pc, &before, PASS_LIMIT);
	d->changed |= passStoredNew(n);
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

// At the end of a slice: where the CPU is, to see it staying in a small loop (with calls, a few
// places; a thread yielding to itself goes through the OS's switch): the last slices ended in at
// most four 512-byte regions.
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
	uint32_t seen[4];
	int count = 0;
	bool few = true;
	for (int i = 0; i < RECENT && few; ++i) {
		int k = 0;
		while (k < count && seen[k] != d->recent[i]) {
			++k;
		}
		if (k == count) {
			few = count < 4;
			seen[count < 4 ? count++ : 3] = d->recent[i];
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
		if (je->vaddr == v && je->gen == jitTag(n, v) && c->pc == (uint64_t) (int64_t) (int32_t) v && !c->nextInDelay) {
			jitRun(n, je->code);
			continue;
		}
		uint32_t paddr = v & 0x1FFFFFFF;
		// Cached RDRAM code (kseg0, or mapped through the TLB) runs from blocks; the rest, and a
		// delay slot reached by leaving a block early, takes the plain step.
		bool mapped = (v & 0xC0000000) != 0x80000000;
		if ((v & 0xE0000000) == 0xA0000000 || (v & 3) || c->pc != (uint64_t) (int64_t) (int32_t) v ||
		    c->nextInDelay || (mapped && !cpuCodeAddress(n, c->pc, &paddr)) || paddr >= RDRAM_SIZE) {
			cpuStep(n);
			n->cycles += n->opCycles + c->stall;
			c->stall = 0;
			continue;
		}
		uint32_t page = paddr >> 12;
		struct Block** t = n->blockPage[page];
		if (!t && !(t = n->blockPage[page] = calloc(PAGE_WORDS, sizeof(*t)))) {
			cpuStep(n);
			n->cycles += n->opCycles + c->stall;
			c->stall = 0;
			continue;
		}
		struct Block* b = t[(paddr >> 2) & (PAGE_WORDS - 1)];
		if (b && b->vaddr != v) {
			dropBlock(n, b);
			b = t[(paddr >> 2) & (PAGE_WORDS - 1)] = NULL;
		}
		if (!b) {
			// A full code arena throws every block away, this page's table with them.
			if (!(b = compile(n, v, paddr))) {
				cpuStep(n);
				n->cycles += n->opCycles + c->stall;
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
			// Kept only with the page known: an unknown one's tag is the same for every page.
			cpuCodePage(n, v);
			if (jitTag(n, v) < RDRAM_SIZE) {
				*je = (struct JitEntry) { v, jitTag(n, v), b->code };
			}
			jitRun(n, b->code);
			continue;
		}
		uint64_t pc = c->pc;
		uint32_t gen = n->blockGen, code = n->codeGen;
		for (uint32_t i = 0; i < b->count; ++i) {
			cpuStepOp(n, b->ops[i], paddr + i * 4);
			n->cycles += n->opCycles + c->stall;
			c->stall = 0;
			pc += 4;
			// Stop at the next event, where the program didn't go on to the next instruction,
			// or when a write threw this block away.
			if (n->cycles >= n->eventAt || c->pc != pc || n->blockGen != gen || n->codeGen != code) {
				break;
			}
		}
	} while (n->cycles < n->eventAt);
	if (n->idle.enabled && !n->idle.on) {
		idleWatch(n);
	}
}
