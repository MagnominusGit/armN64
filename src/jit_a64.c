#include "n64.h"
#include "a64/emit.h"

#include <stddef.h>
#include <sys/mman.h>

// AArch64 code for a block of cached RDRAM instructions, as jit_x64.c does it: the ALU, branches and
// RDRAM loads and stores inline, the rest through cpuJitStep, stopping after any instruction that
// takes the cycle count to the limit.
//
// Registers: x19 = n, x20 = &n->cpu, x21 = cycles left to the limit (the limit less n->cycles),
// x22 = the limit, x23 = RDRAM,
// x25 = vpages, x26 = pageFlags, x27 = cpu.dcacheTag, x28 = cpu.dcacheDirty, w29 = whether a branch
// inside the block is taken; x24 and x9-x15 hold
// guest registers within a block, a new value written back to memory when the register is given up,
// before a call, and on the way out; x0-x8, x16 and x17 are scratch. Each block starts with a branch to the way out, for the conditional branches that
// can't reach it.

// The arena's first half has the shared code and the blocks' hot code (n->jitUsed), its second half
// their cold code: slow paths, misses and the exits a pass seldom takes. How much of the second half
// is used is kept at the end of the shared code's space.
enum { ARENA_SIZE = 128 << 20, BLOCK_CODE_MAX = 32 << 10, SHARED_SIZE = 512, CHAIN_REST = 384 };

static uint64_t* coldUsed(struct N64* n) {
	return (uint64_t*) (n->jitArena + SHARED_SIZE - 8);
}

const bool jitBranchesInside = true;

enum { N = 19, CPU = 20, LEFT = 21, LIMIT = 22, RAM = 23, TLB = 25, PAGES = 26, DTAG = 27, DIRTY = 28, IP = 16, IP1 = 17 };

// The host registers that hold guest registers; the last seven don't survive calls.
enum { POOL = 8, POOL_SAVED = 1 };
static const uint8_t pool[POOL] = { 24, 9, 10, 11, 12, 13, 14, 15 };

// FLE: sp and v0 live in x27 and x28 (the D-cache's registers, which HLE doesn't use) from the way
// in to the way out; memory has them only around calls that read or write guest registers.
enum { PINNED = 2 };
static const uint8_t pinnedGuest[PINNED] = { 29, 2 }, pinnedHost[PINNED] = { DTAG, DIRTY };

static int pinnedReg(bool fle, int g) {
	for (int i = 0; fle && i < PINNED; ++i) {
		if (pinnedGuest[i] == g) {
			return pinnedHost[i];
		}
	}
	return -1;
}

static void storePinned(struct Emitter* e) {
	for (int i = 0; i < PINNED; ++i) {
		aStrX(e, pinnedHost[i], CPU, (unsigned) offsetof(struct CPU, gpr) + 8u * pinnedGuest[i]);
	}
}

static void loadPinned(struct Emitter* e) {
	for (int i = 0; i < PINNED; ++i) {
		aLdrX(e, pinnedHost[i], CPU, (unsigned) offsetof(struct CPU, gpr) + 8u * pinnedGuest[i]);
	}
}

// Which guest register each pool register holds, when it was last used, and which ones memory
// doesn't have yet (a bit each).
struct Cache {
	int8_t guest[POOL];
	int8_t host[32];
	uint32_t used[POOL];
	uint32_t clock;
	uint8_t dirty;
};

#define CPU_OFF(f) ((unsigned) offsetof(struct CPU, f))
#define N64_OFF(f) ((unsigned) offsetof(struct N64, f))

static unsigned gprOff(int i) {
	return CPU_OFF(gpr) + 8 * i;
}

// Loads and stores of n's fields, which may be too far for an offset.
static int fieldBase(struct Emitter* e, unsigned off, unsigned size) {
	if (off % size == 0 && off / size < 4096) {
		return N;
	}
	aAddImmX(e, IP, N, off);
	return IP;
}

static void loadN(struct Emitter* e, int rt, unsigned off, unsigned size) {
	int base = fieldBase(e, off, size);
	unsigned at = base == N ? off : 0;
	if (size == 8) {
		aLdrX(e, rt, base, at);
	} else if (size == 4) {
		aLdr(e, rt, base, at);
	} else {
		aLdrb(e, rt, base, at);
	}
}

static void storeN(struct Emitter* e, int rt, unsigned off, unsigned size) {
	int base = fieldBase(e, off, size);
	unsigned at = base == N ? off : 0;
	if (size == 8) {
		aStrX(e, rt, base, at);
	} else {
		aStr(e, rt, base, at);
	}
}

// A 64-bit constant in as few instructions as it takes.
static void movConst(struct Emitter* e, int rd, uint64_t v) {
	if (v == sext32((uint32_t) v)) {
		aMovImm(e, rd, (uint32_t) v);
		if ((int32_t) v < 0) {
			aSxtw(e, rd, rd);
		}
	} else {
		aMovImm64(e, rd, v);
	}
}

static void storeConst(struct Emitter* e, unsigned cpuOff, uint64_t v) {
	movConst(e, IP1, v);
	aStrX(e, IP1, CPU, cpuOff);
}

static void storeByte(struct Emitter* e, unsigned cpuOff, unsigned v) {
	if (v) {
		aMovImm(e, IP1, v);
	}
	aStrb(e, v ? IP1 : ZR, CPU, cpuOff);
}

// A 64-bit register copy that also takes the zero register.
static void movReg(struct Emitter* e, int rd, int rm) {
	aDp(e, A_ORR | W64, rd, ZR, rm, S_LSL, 0);
}

enum Kind { PLAIN, BRANCH, LIKELY };

struct Exit {
	uint8_t* site;
	uint64_t pc;
	enum Kind kind;
	bool delay;
	bool skip;
	bool last;
	bool jumped;
	// The load mark the CPU leaves with.
	uint8_t mark;
	// A likely branch not taken: its load cycle (cpuLoadStall), or -1 to check the CPU's mark.
	int8_t stall;
	struct Cache cache;
	// After the delay slot of a branch inside the block, taken: on to its target.
	bool side;
	uint64_t to;
};

struct Slow {
	uint8_t* site[8];
	// The registers at each site, whose new values go to memory before the interpreter.
	struct Cache siteCache[8];
	int sites;
	uint8_t* resume;
	uint32_t op, paddr;
	uint64_t pc;
	bool delay, lineStart, last;
	// Kept even with no site yet: a D-cache miss path may add one.
	bool keep;
	// The guest registers held where the fast path resumes, to load again after the interpreter.
	struct Cache cache;
	// The load mark before it (cpuLoadMark), or -1 when it's the block's first.
	int prevMark;
	// Whether the next instruction in the block counts a load cycle after it.
	bool nextStall;
};

struct Miss {
	uint8_t* site;
	uint8_t* resume;
	// The access's slow path, for a misaligned address or one past RDRAM, and its size.
	struct Slow* slow;
	int size;
	// A load that, with the GL renderer, waits for the GPU's pages when the line comes in.
	bool glLoad;
	bool store;
	// The guest registers held where the fast path resumes, to load again after a call.
	struct Cache cache;
};

struct Jit {
	struct Emitter e;
	// The instruction being emitted: where its slow path (the interpreter) is entered from, the
	// cycles it holds the pipeline for, and whether it wrote Status (w2 is then nonzero if that
	// leaves an interrupt due).
	struct Slow* slow;
	int extraCycles;
	// The load cycle of the branch being compiled, for its exit when a likely one isn't taken.
	int8_t branchStall;
	// The load mark of the instruction being compiled (cpuLoadMark).
	uint8_t mark;
	bool interruptCheck;
	// The instruction set pc and nextPc itself (ERET): the block leaves after it.
	bool jumped;
	// The branch being compiled is inside the block (w29 = taken).
	bool inside;
	struct Exit exits[3 * 64];
	int exitCount;
	struct Slow slows[64];
	int slowCount;
	struct Miss misses[64];
	int missCount;
	// Stores to pages with compiled code, which check that they miss its words.
	struct Miss codeChecks[64];
	int codeCheckCount;
	// Loads from pages the RDP may still be drawing to, which wait for it.
	struct Miss rdpWaits[64];
	int rdpWaitCount;
	// With HLE accuracy, accesses not to kseg0's RDRAM or not aligned: through vpages in the cold code.
	struct Miss mapped[64];
	int mappedCount;
	uint8_t* leave;
	bool rdpThreads, rdpGl;
	// Outside mle the D-cache holds the data cached accesses use; with HLE accuracy (fast)
	// there are no caches.
	bool cacheData, fast, plainCount;
	// FLE (Fastest): cycles counted at the block's exits, and no load interlocks, address checks or
	// FPU result fix-ups.
	bool fle;
	int opCycles;
	// Status.FR as the block was compiled: with it clear, odd 32-bit FPU registers are the high
	// halves of even ones.
	bool fr;
	// Status already checked on the way here for CU1, and for FR: until a call or a join.
	bool cu1Seen, frSeen;
	// While hot exits are emitted: the cold code, where their calls to jitLink go.
	struct Emitter* cold;
	uint8_t* arena;
	struct Cache cache;
	// The guest registers known to hold a constant here (a bit each), a 32-bit one sign-extended.
	uint32_t known;
	int32_t value[32];
};

static void cacheClear(struct Cache* c) {
	for (int i = 0; i < POOL; ++i) {
		c->guest[i] = -1;
	}
	for (int g = 0; g < 32; ++g) {
		c->host[g] = -1;
	}
	c->dirty = 0;
}

static void forget(struct Cache* c, int g) {
	if (c->host[g] >= 0) {
		c->dirty &= (uint8_t) ~(1u << c->host[g]);
		c->guest[c->host[g]] = -1;
		c->host[g] = -1;
	}
}

// The new values a cache holds, to memory.
static void flushDirty(struct Emitter* e, const struct Cache* c) {
	for (int k = 0; k < POOL; ++k) {
		if (c->dirty >> k & 1) {
			aStrX(e, pool[k], CPU, gprOff(c->guest[k]));
		}
	}
}

static void writeBack(struct Jit* j) {
	flushDirty(&j->e, &j->cache);
	j->cache.dirty = 0;
}

static void slowSite(struct Slow* s, uint8_t* site, const struct Cache* c) {
	s->siteCache[s->sites] = *c;
	s->site[s->sites++] = site;
}

// Calls clobber the pool registers that aren't callee-saved.
static void forgetCallerSaved(struct Cache* c) {
	for (int i = POOL_SAVED; i < POOL; ++i) {
		if (c->guest[i] >= 0) {
			forget(c, c->guest[i]);
		}
	}
}

// A pool register for guest g: its own, a free one, or the one least recently used.
static int claim(struct Jit* j, int g) {
	struct Cache* c = &j->cache;
	int i = c->host[g];
	if (i < 0) {
		i = 0;
		for (int k = 0; k < POOL; ++k) {
			if (c->guest[k] < 0) {
				i = k;
				break;
			}
			if (c->used[k] < c->used[i]) {
				i = k;
			}
		}
		if (c->guest[i] >= 0) {
			if (c->dirty >> i & 1) {
				aStrX(&j->e, pool[i], CPU, gprOff(c->guest[i]));
			}
			c->host[c->guest[i]] = -1;
		}
		c->dirty &= (uint8_t) ~(1u << i);
		c->guest[i] = (int8_t) g;
		c->host[g] = (int8_t) i;
	}
	c->used[i] = ++c->clock;
	return i;
}

// The host register holding guest g, loaded if need be (the zero register for r0).
static int readReg(struct Jit* j, int g) {
	if (!g) {
		return ZR;
	}
	if (pinnedReg(j->fle, g) >= 0) {
		return pinnedReg(j->fle, g);
	}
	bool held = j->cache.host[g] >= 0;
	int host = pool[claim(j, g)];
	if (!held) {
		aLdrX(&j->e, host, CPU, gprOff(g));
	}
	return host;
}

// The host register to compute guest g's new value in (scratch x8 for r0, which keeps 0); commit()
// then marks it new.
static int writeReg(struct Jit* j, int g) {
	return !g ? 8 : pinnedReg(j->fle, g) >= 0 ? pinnedReg(j->fle, g) : pool[claim(j, g)];
}

static void commit(struct Jit* j, int g, int host) {
	(void) host;
	if (g) {
		if (j->cache.host[g] >= 0) {
			j->cache.dirty |= (uint8_t) (1u << j->cache.host[g]);
		}
		j->known &= ~(1u << g);
	}
}

// Guest g's constant value, when it is known.
static bool constant(const struct Jit* j, int g, int32_t* v) {
	*v = g ? j->value[g] : 0;
	return !g || (j->known >> g & 1);
}

static void setKnown(struct Jit* j, int g, uint64_t v) {
	if (g && v == sext32((uint32_t) v)) {
		j->known |= 1u << g;
		j->value[g] = (int32_t) v;
	}
}

// Guest g = v, through the cache.
static void setConst(struct Jit* j, int g, uint64_t v) {
	if (g) {
		int d = writeReg(j, g);
		movConst(&j->e, d, v);
		commit(j, g, d);
		setKnown(j, g, v);
	}
}

// Guest g = host register r (a scratch one).
static void setReg(struct Jit* j, int g, int r) {
	if (g) {
		int d = writeReg(j, g);
		movReg(&j->e, d, r);
		commit(j, g, d);
	}
}

static void exitAt(struct Jit* j, uint8_t* site, uint64_t pc, enum Kind kind, bool delay, bool skip) {
	j->exits[j->exitCount++] = (struct Exit) { site, pc, kind, delay, skip, false, false, j->mark, skip ? j->branchStall : 0, j->cache, false, 0 };
}

// Sets the flags for a branch condition and returns the condition under which it is taken.
static int branchCondition(struct Jit* j, uint32_t op, int rs, int rt) {
	struct Emitter* e = &j->e;
	int s = readReg(j, rs);
	switch (op >> 26) {
	case 0x04: case 0x14:
	case 0x05: case 0x15:
		aDp(e, A_SUBS | W64, ZR, s, readReg(j, rt), S_LSL, 0);
		return (op >> 26) & 1 ? C_NE : C_EQ;
	case 0x06: case 0x16:
		aDp(e, A_SUBS | W64, ZR, s, ZR, S_LSL, 0);
		return C_LE;
	case 0x07: case 0x17:
		aDp(e, A_SUBS | W64, ZR, s, ZR, S_LSL, 0);
		return C_GT;
	}
	aDp(e, A_SUBS | W64, ZR, s, ZR, S_LSL, 0);
	return rt & 1 ? C_GE : C_LT;
}

static int memSize(uint32_t op);
static void emitMemory(struct Jit* j, struct Slow* slow);

// Leaves the instruction for the interpreter, before it has changed anything, when cond holds.
static void slowIf(struct Jit* j, int cond) {
	slowSite(j->slow, aBcond(&j->e, cond, NULL), &j->cache);
}

// The interpreter raises "coprocessor unusable" when Status.CU1 is off; with fr, the instruction
// needs the Status.FR the block was compiled with.
static void needCop1(struct Jit* j, bool fr) {
	if (j->cu1Seen && (j->frSeen || !fr)) {
		return;
	}
	struct Emitter* e = &j->e;
	aLdr(e, 16, CPU, CPU_OFF(cop0) + 8 * CP0_STATUS);
	if (!j->cu1Seen) {
		slowSite(j->slow, aTbz(e, 16, 29, NULL), &j->cache);
	}
	if (fr && !j->frSeen) {
		slowSite(j->slow, j->fr ? aTbz(e, 16, 26, NULL) : aTbnz(e, 16, 26, NULL), &j->cache);
	}
	j->cu1Seen = true;
	j->frSeen |= fr;
}

// Whether fpuExecute runs op without raising an exception: the moves (MFC1, DMFC1, CFC1, MTC1,
// DMTC1).
static bool fpuOpValid(uint32_t op) {
	int fmt = (op >> 21) & 31;
	return fmt <= 5 && fmt != 3;
}

// What fpuProbe runs: CTC1 and the S, D, W and L operations. One that would raise an exception leaves
// for the interpreter, which takes it.
static bool fpuProbed(uint32_t op) {
	int fmt = (op >> 21) & 31;
	return fmt == 6 || fmt == 16 || fmt == 17 || fmt == 20 || fmt == 21;
}

// Single-precision instructions on s registers.
enum { F_MUL = 0x1E200800, F_DIV = 0x1E201800, F_ADD = 0x1E202800, F_SUB = 0x1E203800, F_ABS = 0x1E20C000, F_NEG = 0x1E214000 };

static void aFop(struct Emitter* e, uint32_t op, int d, int n, int m) {
	emit32(e, op | m << 16 | n << 5 | d);
}

static void aFldr(struct Emitter* e, int t, int n, unsigned offset) {
	emit32(e, 0xBD400000 | (offset / 4) << 10 | n << 5 | t);
}

static void aFstr(struct Emitter* e, int t, int n, unsigned offset) {
	emit32(e, 0xBD000000 | (offset / 4) << 10 | n << 5 | t);
}

// A 32-bit FPU register, as get32 and set32 find it.
static unsigned fprOff(const struct Jit* j, int r) {
	return j->fr ? CPU_OFF(fpr) + 8 * r : CPU_OFF(fpr) + 8 * (r & ~1) + 4 * (r & 1);
}

// A single-precision result in w0 as setS stores it: flushed to zero when subnormal with FCR31.FS
// set, and the host's default NaN made the VR4300's.
static void storeSingle(struct Jit* j, int fd) {
	struct Emitter* e = &j->e;
	if (j->fle) {
		aStr(e, 0, CPU, fprOff(j, fd));
		return;
	}
	aLdr(e, 17, CPU, CPU_OFF(fcr31));
	uint8_t* keep = aTbz(e, 17, 24, NULL);
	aLogicImm(e, A_ANDS_IMM, ZR, 0, 0x7F800000);
	uint8_t* normal = aBcond(e, C_NE, NULL);
	aLogicImm(e, A_AND_IMM, 0, 0, 0x80000000);
	patchHere(e, keep);
	patchHere(e, normal);
	aLogicImm(e, A_AND_IMM, 1, 0, 0x7FFFFFFF);
	aMovImm(e, 2, 0x7FC00000);
	aDp(e, A_SUBS, ZR, 1, 2, S_LSL, 0);
	aMovImm(e, 2, 0x7FBFFFFF);
	aCsel(e, 0, 2, 0, C_EQ);
	aStr(e, 0, CPU, fprOff(j, fd));
}

// A 64-bit FPU register, as get64 and set64 find it.
static unsigned fprOff64(const struct Jit* j, int r) {
	return CPU_OFF(fpr) + 8 * (unsigned) (j->fr ? r : r & ~1);
}

// A double-precision result in x0 as setD stores it: flushed to zero when subnormal with FCR31.FS
// set, and the host's default NaN made the VR4300's.
static void storeDouble(struct Jit* j, int fd) {
	struct Emitter* e = &j->e;
	if (j->fle) {
		aStrX(e, 0, CPU, fprOff64(j, fd));
		return;
	}
	aLdr(e, 17, CPU, CPU_OFF(fcr31));
	uint8_t* keep = aTbz(e, 17, 24, NULL);
	emit32(e, 0xD374F801); // ubfx x1, x0, #52, #11: the exponent
	uint8_t* normal = aCbnzX(e, 1, NULL);
	emit32(e, 0x92410000); // and x0, x0, #0x8000000000000000
	patchHere(e, keep);
	patchHere(e, normal);
	emit32(e, 0x9240F801); // and x1, x0, #0x7FFFFFFFFFFFFFFF
	emit32(e, 0xD2EFFF02); // movz x2, #0x7FF8, lsl #48
	emit32(e, 0xEB02003F); // cmp x1, x2
	emit32(e, 0x92F00102); // movn x2, #0x8008, lsl #48: 0x7FF7FFFFFFFFFFFF
	emit32(e, 0x9A800040); // csel x0, x2, x0, eq
	aStrX(e, 0, CPU, fprOff64(j, fd));
}

// Double-precision arithmetic, moves, compares and the conversions between S and D, as
// fpuExecuteFast does them.
static void emitFpuDouble(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int fmt = (op >> 21) & 31, func = op & 0x3F, ft = (op >> 16) & 31, fs = (op >> 11) & 31, fd = (op >> 6) & 31;
	j->extraCycles = (int) fpuCycles(fmt, func);
	if (fmt == 16) {
		// cvt.d.s
		aFldr(e, 0, CPU, fprOff(j, fs));
		emit32(e, 0x1E22C000); // fcvt d0, s0
		emit32(e, 0x9E660000); // fmov x0, d0
		storeDouble(j, fd);
		return;
	}
	emit32(e, 0xFD400000 | (fprOff64(j, fs) / 8) << 10 | (unsigned) CPU << 5 | 0); // ldr d0
	if (func >= 0x30) {
		emit32(e, 0xFD400000 | (fprOff64(j, ft) / 8) << 10 | (unsigned) CPU << 5 | 1); // ldr d1
		emit32(e, 0x1E602000 | 1 << 16 | 0 << 5); // fcmp d0, d1
		aMovImm(e, 0, 0);
		static const int conds[3] = { C_VS, C_EQ, C_MI };
		for (int k = 0; k < 3; ++k) {
			if (func & (1 << k)) {
				aCset(e, 1, conds[k]);
				aDp(e, A_ORR, 0, 0, 1, S_LSL, 0);
			}
		}
		aLdr(e, 1, CPU, CPU_OFF(fcr31));
		aBfi(e, 1, 0, 23, 1);
		aStr(e, 1, CPU, CPU_OFF(fcr31));
		return;
	}
	if (func == 0x20) {
		// cvt.s.d
		emit32(e, 0x1E624000); // fcvt s0, d0
		emit32(e, 0x1E260000); // fmov w0, s0
		storeSingle(j, fd);
		return;
	}
	if (func == 6) {
		emit32(e, 0xFD000000 | (fprOff64(j, fd) / 8) << 10 | (unsigned) CPU << 5 | 0); // str d0
		return;
	}
	if (func == 5 || func == 7) {
		aFop(e, (func == 5 ? F_ABS : F_NEG) | 0x00400000, 0, 0, 0);
	} else {
		emit32(e, 0xFD400000 | (fprOff64(j, ft) / 8) << 10 | (unsigned) CPU << 5 | 1); // ldr d1
		aFop(e, (const uint32_t[]) { F_ADD, F_SUB, F_MUL, F_DIV }[func] | 0x00400000, 0, 0, 1);
	}
	emit32(e, 0x9E660000); // fmov x0, d0
	storeDouble(j, fd);
}

// With HLE accuracy, the FPU instructions most code runs, inline for the Status.FR the block was
// compiled with: the host's arithmetic, as fpuExecuteFast.
static bool emitFpuFast(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int fmt = (op >> 21) & 31, func = op & 0x3F, ft = (op >> 16) & 31, fs = (op >> 11) & 31, fd = (op >> 6) & 31;
	bool single = fmt == 16 && (func <= 3 || (func >= 5 && func <= 7) || func == 13 || func >= 0x30);
	bool dbl = (fmt == 17 && (func <= 3 || (func >= 5 && func <= 7) || func == 0x20 || func >= 0x30)) || (fmt == 16 && func == 0x21);
	if (!(fmt == 0 || fmt == 2 || fmt == 4 || single || dbl || (fmt == 20 && func == 32))) {
		return false;
	}
	needCop1(j, true);
	switch (fmt) {
	case 0:
		if (ft) {
			int d = writeReg(j, ft);
			aLdr(e, d, CPU, fprOff(j, fs));
			aSxtw(e, d, d);
			commit(j, ft, d);
		}
		return true;
	case 2:
		if (fs == 31) {
			int d = writeReg(j, ft);
			aLdr(e, d, CPU, CPU_OFF(fcr31));
			aSxtw(e, d, d);
			commit(j, ft, d);
		} else {
			setConst(j, ft, fs == 0 ? 0xA00 : 0);
		}
		return true;
	case 4:
		aStr(e, readReg(j, ft), CPU, fprOff(j, fs));
		return true;
	case 20:
		aLdr(e, 0, CPU, fprOff(j, fs));
		emit32(e, 0x1E220000 | 0 << 5 | 0);
		aFstr(e, 0, CPU, fprOff(j, fd));
		j->extraCycles = (int) fpuCycles(fmt, func);
		return true;
	}
	if (dbl) {
		emitFpuDouble(j, op);
		return true;
	}
	j->extraCycles = (int) fpuCycles(fmt, func);
	aFldr(e, 0, CPU, fprOff(j, fs));
	if (func >= 0x30) {
		// c.cond.s: FCMP gives N for less, Z for equal and V for unordered.
		aFldr(e, 1, CPU, fprOff(j, ft));
		emit32(e, 0x1E202000 | 1 << 16 | 0 << 5);
		aMovImm(e, 0, 0);
		static const int conds[3] = { C_VS, C_EQ, C_MI };
		for (int k = 0; k < 3; ++k) {
			if (func & (1 << k)) {
				aCset(e, 1, conds[k]);
				aDp(e, A_ORR, 0, 0, 1, S_LSL, 0);
			}
		}
		aLdr(e, 1, CPU, CPU_OFF(fcr31));
		aBfi(e, 1, 0, 23, 1);
		aStr(e, 1, CPU, CPU_OFF(fcr31));
		return true;
	}
	switch (func) {
	case 6:
		aFstr(e, 0, CPU, fprOff(j, fd));
		return true;
	case 13:
		// trunc.w.s: out of range or NaN gives the most negative word, as toWord.
		emit32(e, 0x9E380000 | 0 << 5 | 0);
		aSxtw(e, 1, 0);
		aMovImm(e, 2, 0x80000000);
		aDp(e, A_SUBS | W64, ZR, 1, 0, S_LSL, 0);
		aCsel(e, 0, 0, 2, C_EQ);
		emit32(e, 0x1E202000 | 0 << 16 | 0 << 5);
		aCsel(e, 0, 0, 2, C_VC);
		aStr(e, 0, CPU, fprOff(j, fd));
		return true;
	case 5:
	case 7:
		aFop(e, func == 5 ? F_ABS : F_NEG, 0, 0, 0);
		break;
	default:
		aFldr(e, 1, CPU, fprOff(j, ft));
		aFop(e, (const uint32_t[]) { F_ADD, F_SUB, F_MUL, F_DIV }[func], 0, 0, 1);
		break;
	}
	emit32(e, 0x1E260000 | 0 << 5 | 0);
	storeSingle(j, fd);
	return true;
}

// The FPU's moves inline for the Status.FR the block was compiled with.
static bool emitFpu(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int fmt = (op >> 21) & 31, ft = (op >> 16) & 31, fs = (op >> 11) & 31;
	if (fmt != 0 && fmt != 2 && fmt != 4) {
		return false;
	}
	needCop1(j, true);
	switch (fmt) {
	case 0:
		if (ft) {
			int d = writeReg(j, ft);
			aLdr(e, d, CPU, fprOff(j, fs));
			aSxtw(e, d, d);
			commit(j, ft, d);
		}
		return true;
	case 2:
		if (fs == 31) {
			int d = writeReg(j, ft);
			aLdr(e, d, CPU, CPU_OFF(fcr31));
			aSxtw(e, d, d);
			commit(j, ft, d);
		} else {
			setConst(j, ft, fs == 0 ? 0xA00 : 0);
		}
		return true;
	default:
		aStr(e, readReg(j, ft), CPU, fprOff(j, fs));
		return true;
	}
}

// Leaves for the interpreter when the operand in w0 (x0 with dbl) is denormal, infinite or NaN; x1
// keeps it shifted left once (no sign).
static void fpuNormal(struct Jit* j, bool dbl) {
	struct Emitter* e = &j->e;
	if (dbl) {
		emit32(e, 0xD37FF801); // lsl x1, x0, #1
		aMovImm64(e, 2, 0x0020000000000000ull);
		aDp(e, A_SUB | W64, 3, 1, 2, S_LSL, 0);
		aMovImm64(e, 2, 0xFFC0000000000000ull);
		aDp(e, A_SUBS | W64, ZR, 3, 2, S_LSL, 0);
	} else {
		aLslImm(e, 1, 0, 1);
		aMovImm(e, 2, 0x01000000);
		aDp(e, A_SUB, 3, 1, 2, S_LSL, 0);
		aMovImm(e, 2, 0xFE000000);
		aDp(e, A_SUBS, ZR, 3, 2, S_LSL, 0);
	}
	uint8_t* normal = aBcond(e, C_LO, NULL);
	slowSite(j->slow, dbl ? aCbnzX(e, 1, NULL) : aCbnz(e, 1, NULL), &j->cache);
	patchHere(e, normal);
}

// With LLE and MLE accuracy, MOV, NEG, ABS, the four operations and C.cond in S and D, for the
// Status.FR the block was compiled with, where fpu.c takes no exception path: operands normal or zero,
// the host raising no more than inexact, inexact not enabled, the result not denormal. Anything else
// the interpreter runs.
static bool emitFpuExact(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int const fmt = (op >> 21) & 31, func = op & 0x3F, ft = (op >> 16) & 31, fs = (op >> 11) & 31, fd = (op >> 6) & 31;
	bool const dbl = fmt == 17, arith = func <= 3, compare = func >= 0x30, sign = func == 5 || func == 7;
	if ((fmt != 16 && fmt != 17) || !(arith || compare || sign || func == 6)) {
		return false;
	}
	needCop1(j, true);
	// The first operand as fpu.c's sOf and dOf find it, the second and the result as named.
	unsigned const a = CPU_OFF(fpr) + 8 * (unsigned) (j->fr ? fs : fs & ~1), b = CPU_OFF(fpr) + 8 * (unsigned) ft;
	unsigned const d = CPU_OFF(fpr) + 8 * (unsigned) fd;
	if (func == 6) {
		aLdrX(e, 0, CPU, a);
		aStrX(e, 0, CPU, d);
		return true;
	}
	if (compare) {
		for (int k = 0; k < 2; ++k) {
			if (dbl) {
				aLdrX(e, 0, CPU, k ? b : a);
				emit32(e, 0xD37FF801); // lsl x1, x0, #1
				aMovImm64(e, 2, 0xFFE0000000000000ull);
				aDp(e, A_SUBS | W64, ZR, 1, 2, S_LSL, 0);
			} else {
				aLdr(e, 0, CPU, k ? b : a);
				aLslImm(e, 1, 0, 1);
				aMovImm(e, 2, 0xFF000000);
				aDp(e, A_SUBS, ZR, 1, 2, S_LSL, 0);
			}
			slowIf(j, C_HI);
		}
		if (dbl) {
			emit32(e, 0xFD400000 | (a / 8) << 10 | (unsigned) CPU << 5 | 0); // ldr d0
			emit32(e, 0xFD400000 | (b / 8) << 10 | (unsigned) CPU << 5 | 1); // ldr d1
			emit32(e, 0x1E602000 | 1 << 16 | 0 << 5); // fcmp d0, d1
		} else {
			aFldr(e, 0, CPU, a);
			aFldr(e, 1, CPU, b);
			emit32(e, 0x1E202000 | 1 << 16 | 0 << 5); // fcmp s0, s1
		}
		aMovImm(e, 0, 0);
		static const int conds[3] = { C_AL, C_EQ, C_MI };
		for (int k = 1; k < 3; ++k) {
			if (func & (1 << k)) {
				aCset(e, 1, conds[k]);
				aDp(e, A_ORR, 0, 0, 1, S_LSL, 0);
			}
		}
		aLdr(e, 1, CPU, CPU_OFF(fcr31));
		aLogicImm(e, A_AND_IMM, 1, 1, ~0x3F000u);
		aBfi(e, 1, 0, 23, 1);
		aStr(e, 1, CPU, CPU_OFF(fcr31));
		return true;
	}
	if (sign) {
		if (dbl) {
			aLdrX(e, 0, CPU, a);
		} else {
			aLdr(e, 0, CPU, a);
		}
		fpuNormal(j, dbl);
		if (dbl) {
			emit32(e, func == 7 ? 0xD2410000 : 0x9240F800); // eor x0, x0, #1 << 63 / and x0, x0, #~(1 << 63)
		} else {
			aLogicImm(e, func == 7 ? A_EOR_IMM : A_AND_IMM, 0, 0, func == 7 ? 0x80000000u : 0x7FFFFFFFu);
		}
		aStrX(e, 0, CPU, d);
		aLdr(e, 1, CPU, CPU_OFF(fcr31));
		aLogicImm(e, A_AND_IMM, 1, 1, ~0x3F000u);
		aStr(e, 1, CPU, CPU_OFF(fcr31));
		return true;
	}
	// The stall in x5: the operation's cycles, or 2 in all (1 more) when an operand is zero or, for
	// MUL, a power of two (fpu.c early).
	uint32_t const cycles = fpuCycles(fmt, func);
	aMovImm(e, 5, cycles);
	aMovImm(e, 6, cycles > 1 ? 1 : cycles);
	for (int k = 0; k < 2; ++k) {
		if (dbl) {
			aLdrX(e, 0, CPU, k ? b : a);
		} else {
			aLdr(e, 0, CPU, k ? b : a);
		}
		fpuNormal(j, dbl);
		aDp(e, A_SUBS | W64, ZR, 1, ZR, S_LSL, 0);
		aCselX(e, 5, 6, 5, C_EQ);
		if (func == 2) {
			if (dbl) {
				emit32(e, 0xD374CC02); // lsl x2, x0, #12
			} else {
				aLslImm(e, 2, 0, 9);
			}
			aDp(e, (dbl ? A_SUBS | W64 : A_SUBS), ZR, 2, ZR, S_LSL, 0);
			aCselX(e, 5, 6, 5, C_EQ);
		}
	}
	// The host's flags cleared, the operation, its flags: anything but inexact goes the other way.
	emit32(e, 0xD51B443F); // msr fpsr, xzr
	if (dbl) {
		emit32(e, 0xFD400000 | (a / 8) << 10 | (unsigned) CPU << 5 | 0); // ldr d0
		emit32(e, 0xFD400000 | (b / 8) << 10 | (unsigned) CPU << 5 | 1); // ldr d1
		aFop(e, (const uint32_t[]) { F_ADD, F_SUB, F_MUL, F_DIV }[func] | 0x00400000, 0, 0, 1);
		emit32(e, 0x9E660000); // fmov x0, d0
	} else {
		aFldr(e, 0, CPU, a);
		aFldr(e, 1, CPU, b);
		aFop(e, (const uint32_t[]) { F_ADD, F_SUB, F_MUL, F_DIV }[func], 0, 0, 1);
		emit32(e, 0x1E260000); // fmov w0, s0
	}
	emit32(e, 0xD53B4424); // mrs x4, fpsr
	aLogicImm(e, A_ANDS_IMM, ZR, 4, 0xF);
	slowIf(j, C_NE);
	// A denormal result (exact, so no underflow flag).
	fpuNormal(j, dbl);
	// Inexact: its cause, and its flag unless enabled (then the exception, the other way).
	aLdr(e, 2, CPU, CPU_OFF(fcr31));
	aLogicImm(e, A_AND_IMM, 2, 2, ~0x3F000u);
	uint8_t* exact = aTbz(e, 4, 4, NULL);
	slowSite(j->slow, aTbnz(e, 2, 7, NULL), &j->cache);
	aLogicImm(e, A_ORR_IMM, 2, 2, 0x1000);
	aLogicImm(e, A_ORR_IMM, 2, 2, 0x4);
	patchHere(e, exact);
	aStr(e, 2, CPU, CPU_OFF(fcr31));
	aStrX(e, 0, CPU, d);
	aDp(e, A_SUB | W64, LEFT, LEFT, 5, S_LSL, 0);
	return true;
}

// MFC0 and DMFC0 of anything but Random and the registers that read the latch; MTC0 and DMTC0 of EPC
// and ErrorEPC.
static bool emitCop0(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, rd = (op >> 11) & 31;
	unsigned reg = CPU_OFF(cop0) + 8 * rd;
	if (op == 0x42000018 && !j->slow->delay) {
		// ERET: to ErrorEPC with ERL cleared when it is set, else to EPC with EXL cleared.
		aLdrX(e, 0, CPU, CPU_OFF(cop0) + 8 * CP0_STATUS);
		aLdrX(e, 1, CPU, CPU_OFF(cop0) + 8 * CP0_EPC);
		aMovImm(e, 2, 2);
		uint8_t* exl = aTbz(e, 0, 2, NULL);
		aLdrX(e, 1, CPU, CPU_OFF(cop0) + 8 * CP0_ERROREPC);
		aMovImm(e, 2, 4);
		patchHere(e, exl);
		aDp(e, A_BIC | W64, 0, 0, 2, S_LSL, 0);
		aStrX(e, 0, CPU, CPU_OFF(cop0) + 8 * CP0_STATUS);
		aStrX(e, 1, CPU, CPU_OFF(pc));
		aDpImm(e, A_ADD_IMM | W64, 1, 1, 4);
		aStrX(e, 1, CPU, CPU_OFF(nextPc));
		aStrb(e, ZR, CPU, CPU_OFF(llBit));
		j->extraCycles = 2;
		j->jumped = true;
		return true;
	}
	if (rs == 0x00 || rs == 0x01) {
		// Random, and the unused registers that read the write latch, in C.
		if (rd == CP0_RANDOM || (rd >= 21 && rd <= 25) || rd == 31) {
			return false;
		}
		if (rt) {
			int d = writeReg(j, rt);
			if (rd == CP0_COUNT && !j->plainCount) {
				return false;
			}
			if (rd == CP0_COUNT) {
				// Count: half the cycles before this instruction, from countBase.
				aDp(e, A_SUB | W64, 0, LIMIT, LEFT, S_LSL, 0);
				aLdrX(e, 1, CPU, CPU_OFF(countBase));
				aDp(e, A_SUB | W64, 0, 0, 1, S_LSL, 0);
				aLsrImmX(e, 0, 0, 1);
				if (rs == 0x00) {
					aSxtw(e, d, 0);
				} else {
					aMov(e, d, 0);
				}
			} else {
				aLdrX(e, d, CPU, reg);
				if (rs == 0x00) {
					aSxtw(e, d, d);
				}
			}
			commit(j, rt, d);
		}
		return true;
	}
	// Status in C: a write may leave the mode compiled code runs in.
	if ((rs != 0x04 && rs != 0x05) || (rd != CP0_EPC && rd != CP0_ERROREPC)) {
		return false;
	}
	int t = readReg(j, rt);
	// Each write also goes to the latch.
	aStrX(e, t, CPU, CPU_OFF(cop0) + 8 * CP0_LATCH);
	aStrX(e, t, CPU, reg);
	return true;
}

static bool emitInline(struct Jit* j, uint32_t op, uint64_t pc, enum Kind* kind) {
	struct Emitter* e = &j->e;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, rd = (op >> 11) & 31, sa = (op >> 6) & 31;
	int32_t imm = (int16_t) op;
	uint64_t target = pc + 4 + ((uint64_t) (int64_t) imm << 2);
	*kind = PLAIN;
	switch (op >> 26) {
	case 0x00: {
		int f = op & 0x3F;
		switch (f) {
		case 0x00: case 0x02: case 0x03:
			if (rd) {
				int t = readReg(j, rt), d = writeReg(j, rd);
				if (f == 0x03) {
					aAsrImmX(e, d, t, sa);
				} else if (f == 0x02) {
					aLsrImm(e, d, t, sa);
				} else {
					aLslImm(e, d, t, sa);
				}
				aSxtw(e, d, d);
				commit(j, rd, d);
			}
			return true;
		case 0x04: case 0x06: case 0x07:
			if (rd) {
				int s = readReg(j, rs), t = readReg(j, rt), d = writeReg(j, rd);
				if (f == 0x07) {
					aAndBits(e, 1, s, 0, 5);
					aDp(e, A_ASRV | W64, d, t, 1, S_LSL, 0);
				} else {
					aDp(e, f == 0x04 ? A_LSLV : A_LSRV, d, t, s, S_LSL, 0);
				}
				aSxtw(e, d, d);
				commit(j, rd, d);
			}
			return true;
		case 0x08: case 0x09:
			aStrX(e, readReg(j, rs), CPU, CPU_OFF(nextPc));
			if (f == 0x09) {
				setConst(j, rd, pc + 8);
			}
			*kind = BRANCH;
			return true;
		case 0x0F:
			return true;
		case 0x10: case 0x12:
			if (rd) {
				int d = writeReg(j, rd);
				aLdrX(e, d, CPU, f == 0x10 ? CPU_OFF(hi) : CPU_OFF(lo));
				commit(j, rd, d);
			}
			return true;
		case 0x11: case 0x13:
			aStrX(e, readReg(j, rs), CPU, f == 0x11 ? CPU_OFF(hi) : CPU_OFF(lo));
			return true;
		case 0x14: case 0x16: case 0x17:
			if (rd) {
				int s = readReg(j, rs), t = readReg(j, rt), d = writeReg(j, rd);
				aDp(e, (f == 0x14 ? A_LSLV : f == 0x16 ? A_LSRV : A_ASRV) | W64, d, t, s, S_LSL, 0);
				commit(j, rd, d);
			}
			return true;
		case 0x18: case 0x19: {
			// The products' low and high words, sign-extended; the multiplier holds the pipeline. MULT
			// takes rs whole and rt sign-extended from 35 bits, as the interpreter.
			int s = readReg(j, rs), t = readReg(j, rt);
			if (f == 0x18) {
				aLslImmX(e, 1, t, 29);
				aAsrImmX(e, 1, 1, 29);
				emit32(e, 0x9B000000 | (1 << 16) | (ZR << 10) | (s << 5) | 0);  // MUL x0, s, x1
			} else {
				aUmull(e, 0, s, t);
			}
			aSxtw(e, 1, 0);
			aStrX(e, 1, CPU, CPU_OFF(lo));
			aAsrImmX(e, 0, 0, 32);
			aSxtw(e, 0, 0);
			aStrX(e, 0, CPU, CPU_OFF(hi));
			j->extraCycles = 4;
			return true;
		}
		case 0x1A: case 0x1B: {
			// LO the quotient and HI the remainder, sign-extended (INT32_MIN / -1 wraps, as SDIV does).
			// By zero HI is the dividend and LO all ones, or 1 for a negative dividend in DIV.
			// DIV by a divisor that isn't a 32-bit one sign-extended divides as 64 bits, in C.
			int s = readReg(j, rs), t = readReg(j, rt);
			if (f == 0x1A && t != ZR) {
				aSxtw(e, 2, t);
				aDp(e, A_SUBS | W64, ZR, t, 2, S_LSL, 0);
				slowIf(j, C_NE);
			}
			emit32(e, (f == 0x1A ? 0x1AC00C00 : 0x1AC00800) | (t << 16) | (s << 5) | 0);
			emit32(e, 0x1B008000 | (t << 16) | (s << 10) | (0 << 5) | 1);
			aSxtw(e, 1, 1);
			aStrX(e, 1, CPU, CPU_OFF(hi));
			aSxtw(e, 0, 0);
			uint8_t* nonzero = aCbnz(e, t, NULL);
			if (f == 0x1A) {
				aLsrImm(e, 2, s, 31);
				aLslImmX(e, 2, 2, 1);
				aDpImm(e, A_SUB_IMM | W64, 0, 2, 1);
			} else {
				movConst(e, 0, ~0ull);
			}
			patchHere(e, nonzero);
			aStrX(e, 0, CPU, CPU_OFF(lo));
			j->extraCycles = 36;
			return true;
		}
		case 0x20: case 0x22: case 0x2C: case 0x2E: {
			// Overflow raises an exception, through the interpreter, before rd changes.
			uint32_t wide = f >= 0x2C ? W64 : 0;
			aDp(e, ((f & 2) ? A_SUBS : A_ADDS) | wide, 0, readReg(j, rs), readReg(j, rt), S_LSL, 0);
			slowIf(j, C_VS);
			if (!wide) {
				aSxtw(e, 0, 0);
			}
			setReg(j, rd, 0);
			return true;
		}
		case 0x21: case 0x23:
			if (rd) {
				int s = readReg(j, rs), t = readReg(j, rt), d = writeReg(j, rd);
				aDp(e, f == 0x21 ? A_ADD : A_SUB, d, s, t, S_LSL, 0);
				aSxtw(e, d, d);
				commit(j, rd, d);
			}
			return true;
		case 0x24: case 0x25: case 0x26: case 0x27: case 0x2D: case 0x2F: {
			static const uint32_t alu[] = { A_AND, A_ORR, A_EOR, A_ORR };
			if (rd) {
				int s = readReg(j, rs), t = readReg(j, rt), d = writeReg(j, rd);
				aDp(e, (f == 0x2D ? A_ADD : f == 0x2F ? A_SUB : alu[f - 0x24]) | W64, d, s, t, S_LSL, 0);
				if (f == 0x27) {
					aDp(e, A_ORN | W64, d, ZR, d, S_LSL, 0);
				}
				commit(j, rd, d);
			}
			return true;
		}
		case 0x2A: case 0x2B:
			if (rd) {
				int s = readReg(j, rs), t = readReg(j, rt), d = writeReg(j, rd);
				aDp(e, A_SUBS | W64, ZR, s, t, S_LSL, 0);
				aCset(e, d, f == 0x2A ? C_LT : C_LO);
				commit(j, rd, d);
			}
			return true;
		case 0x38: case 0x3A: case 0x3B: case 0x3C: case 0x3E: case 0x3F:
			if (rd) {
				int amount = sa + (f >= 0x3C ? 32 : 0);
				int t = readReg(j, rt), d = writeReg(j, rd);
				if ((f & 3) == 0) {
					aLslImmX(e, d, t, amount);
				} else if ((f & 3) == 2) {
					aLsrImmX(e, d, t, amount);
				} else {
					aAsrImmX(e, d, t, amount);
				}
				commit(j, rd, d);
			}
			return true;
		}
		return false;
	}
	case 0x01:
		if (((op >> 16) & 0x0C) != 0) {
			return false;
		}
		break;
	case 0x02: case 0x03:
		storeConst(e, CPU_OFF(nextPc), ((pc + 4) & ~0x0FFFFFFFull) | (uint64_t) (op & 0x03FFFFFF) << 2);
		if (op >> 26 == 0x03) {
			setConst(j, 31, pc + 8);
		}
		*kind = BRANCH;
		return true;
	case 0x04: case 0x05: case 0x06: case 0x07:
	case 0x14: case 0x15: case 0x16: case 0x17:
		break;
	case 0x08: case 0x18: {
		uint32_t wide = op >> 26 == 0x18 ? W64 : 0;
		int s = readReg(j, rs);
		movConst(e, 1, (uint64_t) (int64_t) imm);
		aDp(e, A_ADDS | wide, 0, s, 1, S_LSL, 0);
		slowIf(j, C_VS);
		if (!wide) {
			aSxtw(e, 0, 0);
		}
		setReg(j, rt, 0);
		return true;
	}
	case 0x09: case 0x19:
		if (rt) {
			int s = readReg(j, rs), d = writeReg(j, rt);
			uint32_t wide = op >> 26 == 0x19 ? W64 : 0;
			// Small immediates added as they are (register 31 is SP there, so r0 takes the long way).
			if (s != ZR && imm >= -4095 && imm <= 4095) {
				aDpImm(e, (imm < 0 ? A_SUB_IMM : A_ADD_IMM) | wide, d, s, (unsigned) (imm < 0 ? -imm : imm));
			} else {
				movConst(e, 1, (uint64_t) (int64_t) imm);
				aDp(e, A_ADD | wide, d, s, 1, S_LSL, 0);
			}
			if (op >> 26 == 0x09) {
				aSxtw(e, d, d);
			}
			int32_t v;
			bool was = constant(j, rs, &v);
			commit(j, rt, d);
			if (was) {
				uint64_t sum = (uint64_t) (int64_t) v + (uint64_t) (int64_t) imm;
				setKnown(j, rt, op >> 26 == 0x09 ? sext32((uint32_t) sum) : sum);
			}
		}
		return true;
	case 0x0A: case 0x0B:
		if (rt) {
			int s = readReg(j, rs), d = writeReg(j, rt);
			if (s != ZR && imm >= -4095 && imm <= 4095) {
				aDpImm(e, (imm < 0 ? A_ADDS_IMM : A_SUBS_IMM) | W64, ZR, s, (unsigned) (imm < 0 ? -imm : imm));
			} else {
				movConst(e, 1, (uint64_t) (int64_t) imm);
				aDp(e, A_SUBS | W64, ZR, s, 1, S_LSL, 0);
			}
			aCset(e, d, op >> 26 == 0x0A ? C_LT : C_LO);
			commit(j, rt, d);
		}
		return true;
	case 0x0C: case 0x0D: case 0x0E:
		if (rt) {
			static const uint32_t alu[] = { A_AND, A_ORR, A_EOR };
			int s = readReg(j, rs), d = writeReg(j, rt);
			// ANDI by a mask AArch64 encodes: the 32-bit form clears the upper half as ANDI does.
			if (op >> 26 == 0x0C && (uint16_t) op && aLogicImm(e, A_AND_IMM, d, s, (uint16_t) op)) {
				commit(j, rt, d);
				return true;
			}
			aMovImm(e, 1, (uint16_t) op);
			aDp(e, alu[(op >> 26) - 0x0C] | W64, d, s, 1, S_LSL, 0);
			int32_t v;
			bool was = constant(j, rs, &v) && op >> 26 == 0x0D;
			commit(j, rt, d);
			if (was) {
				setKnown(j, rt, (uint64_t) (int64_t) v | (uint16_t) op);
			}
		}
		return true;
	case 0x0F:
		setConst(j, rt, sext32((uint32_t) (uint16_t) op << 16));
		return true;
	case 0x10:
		return emitCop0(j, op);
	case 0x11:
		if (((op >> 21) & 31) == 0x08) {
			if (rt >= 4) {
				return false;
			}
			break;
		}
		if (!fpuOpValid(op) && !fpuProbed(op)) {
			return false;
		}
		if (j->fast ? emitFpuFast(j, op) : emitFpu(j, op) || emitFpuExact(j, op)) {
			return true;
		}
		// The FPU's own code, called directly. Its stall counts. With HLE accuracy it raises nothing.
		bool const probe = fpuProbed(op) && !j->fast;
		needCop1(j, false);
		writeBack(j);
		if (j->fle) {
			storePinned(e);
		}
		aMovX(e, 0, N);
		aMovImm(e, 1, op);
		aMovImm64(e, IP, (uint64_t) (uintptr_t) (j->fast ? (void*) fpuExecuteFast : probe ? (void*) fpuProbe : (void*) fpuExecute));
		aBlr(e, IP);
		if (j->fle) {
			loadPinned(e);
		}
		forgetCallerSaved(&j->cache);
		if (probe) {
			slowSite(j->slow, aTbz(e, 0, 0, NULL), &j->cache);
		}
		aLdr(e, 0, CPU, CPU_OFF(stall));
		aDp(e, A_SUB | W64, LEFT, LEFT, 0, S_LSL, 0);
		aStr(e, ZR, CPU, CPU_OFF(stall));
		// Moves to a general register: it has a new value; r0 is undone, as the interpreter zeroes it.
		if (((op >> 21) & 31) <= 2) {
			forget(&j->cache, rt);
			j->known &= ~(1u << rt);
			if (!rt) {
				aStrX(e, ZR, CPU, gprOff(0));
			}
		}
		return true;
	case 0x2F: {
		// CACHE on the D-cache (and the secondary cache, which isn't there) at an aligned kseg0
		// address, through cacheOp; I-cache ones, others and a write-back over code leave the block
		// through the interpreter.
		if (((op >> 16) & 3) == 0 && !j->fast) {
			return false;
		}
		int base = readReg(j, rs);
		if (base == ZR) {
			aMovImm(e, 0, (uint32_t) imm);
		} else {
			aMovImm(e, 1, (uint32_t) imm);
			aSxtw(e, 1, 1);
			aDp(e, A_ADD | W64, 0, base, 1, S_LSL, 0);
			aSxtw(e, 1, 0);
			aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
			slowIf(j, C_NE);
		}
		aMovImm(e, 1, 0xE0000003u);
		aDp(e, A_AND, 1, 0, 1, S_LSL, 0);
		aMovImm(e, 2, 0x80000000u);
		aDp(e, A_SUBS, ZR, 1, 2, S_LSL, 0);
		slowIf(j, C_NE);
		// Without caches (HLE accuracy) that's all.
		if (j->fast) {
			return true;
		}
		writeBack(j);
		aMov(e, 2, 0);
		aMovX(e, 0, N);
		aMovImm(e, 1, op);
		aMovImm64(e, IP, (uint64_t) (uintptr_t) cpuCacheOp);
		aBlr(e, IP);
		forgetCallerSaved(&j->cache);
		aDpImm(e, A_ADDS_IMM, ZR, 0, 1);
		slowIf(j, C_EQ);
		aDp(e, A_SUB | W64, LEFT, LEFT, 0, S_LSL, 0);
		return true;
	}
	case 0x20: case 0x21: case 0x23: case 0x24: case 0x25: case 0x27: case 0x37:
	case 0x28: case 0x29: case 0x2B: case 0x3F:
	case 0x31: case 0x35: case 0x39: case 0x3D:
		emitMemory(j, j->slow);
		return true;
	default:
		return false;
	}

	// Conditional branches. The link register is written whether or not the branch is taken, after
	// the condition has read the source.
	bool likely = (op >> 26) >= 0x14 || ((op >> 26) == 0x01 && (rt & 2)) || ((op >> 26) == 0x11 && (rt & 2));
	int cc;
	if (op >> 26 == 0x11) {
		// BC1F, BC1T and the likely ones: FCR31's condition bit, the causes cleared first.
		needCop1(j, false);
		aLdr(e, 0, CPU, CPU_OFF(fcr31));
		aLogicImm(e, A_AND_IMM, 0, 0, ~0x3F000u);
		aStr(e, 0, CPU, CPU_OFF(fcr31));
		aLogicImm(e, A_ANDS_IMM, ZR, 0, 1u << 23);
		cc = rt & 1 ? C_NE : C_EQ;
	} else {
		cc = branchCondition(j, op, rs, rt);
	}
	if ((op >> 26) == 0x01 && (rt & 0x10)) {
		setConst(j, 31, pc + 8);
	}
	if (likely) {
		exitAt(j, aBcond(e, cc ^ 1, NULL), pc, LIKELY, false, true);
		storeConst(e, CPU_OFF(nextPc), target);
		*kind = LIKELY;
	} else {
		movConst(e, 2, pc + 8);
		movConst(e, 3, target);
		aCselX(e, 2, 3, 2, cc);
		aStrX(e, 2, CPU, CPU_OFF(nextPc));
		if (j->inside) {
			aCset(e, 29, cc);
		}
		*kind = BRANCH;
	}
	return true;
}

// The delay slot's view: pc is where the branch goes and nextPc the instruction after that.
static void delayPrologue(struct Emitter* e) {
	aLdrX(e, 0, CPU, CPU_OFF(nextPc));
	aStrX(e, 0, CPU, CPU_OFF(pc));
	aDpImm(e, A_ADD_IMM | W64, 0, 0, 4);
	aStrX(e, 0, CPU, CPU_OFF(nextPc));
}

// The guest registers a cache holds, loaded again after a call.
static void reloadPool(struct Emitter* e, const struct Cache* c) {
	for (int k = 0; k < POOL; ++k) {
		if (c->guest[k] >= 0) {
			aLdrX(e, pool[k], CPU, gprOff(c->guest[k]));
		}
	}
}

// The I-cache line holding pc, as fetch checks it; a miss fills it in C (w2 kept for an interrupt
// check).
static void icacheCheck(struct Jit* j, uint32_t vaddr, uint32_t paddr, bool toStall) {
	struct Emitter* e = &j->e;
	aLdr(e, 0, CPU, CPU_OFF(icacheTag) + 4 * ((vaddr >> 5) & 511));
	aMovImm(e, 1, paddr & ~31u);
	aDp(e, A_SUBS, ZR, 0, 1, S_LSL, 0);
	uint8_t* hit = aBcond(e, C_EQ, NULL);
	flushDirty(e, &j->cache);
	aPush(e, 2, 3, 16);
	aMovX(e, 0, N);
	aMovImm(e, 1, vaddr);
	aMovImm(e, 2, paddr);
	aMovImm64(e, IP, (uint64_t) (uintptr_t) cpuIcacheMiss);
	aBlr(e, IP);
	if (toStall) {
		aLdr(e, 1, CPU, CPU_OFF(stall));
		aDp(e, A_ADD, 1, 1, 0, S_LSL, 0);
		aStr(e, 1, CPU, CPU_OFF(stall));
	} else {
		aDp(e, A_SUB | W64, LEFT, LEFT, 0, S_LSL, 0);
	}
	aPop(e, 2, 3, 16);
	reloadPool(e, &j->cache);
	patchHere(e, hit);
}

// The idle loop at vaddr, when its I-cache lines are in: whole passes of its two instructions up to a
// pass or less short of the limit, leaving the last instructions to run as usual.
static void skipIdle(struct Emitter* e, uint32_t vaddr, uint32_t paddr, bool fast, int opCycles) {
	uint8_t* skip[3];
	int sites = 0;
	int lines = fast ? 0 : (paddr + 4) & 31 ? 1 : 2;
	for (int i = 0; i < lines; ++i) {
		uint32_t v = vaddr + 4 * i, p = paddr + 4 * i;
		aLdr(e, 0, CPU, CPU_OFF(icacheTag) + 4 * ((v >> 5) & 511));
		aMovImm(e, 1, p & ~31u);
		aDp(e, A_SUBS, ZR, 0, 1, S_LSL, 0);
		skip[sites++] = aBcond(e, C_NE, NULL);
	}
	aDpImm(e, A_SUBS_IMM | W64, ZR, LEFT, 1);
	skip[sites++] = aBcond(e, C_LE, NULL);
	aDpImm(e, A_SUB_IMM | W64, 0, LEFT, 1);
	aLsrImmX(e, 0, 0, opCycles == 4 ? 3 : 1);
	aDp(e, A_SUB | W64, LEFT, LEFT, 0, S_LSL, opCycles == 4 ? 3 : 1);
	for (int i = 0; i < sites; ++i) {
		patchHere(e, skip[i]);
	}
}

static bool isBranchOp(uint32_t op) {
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

// Branches to where the full state is written before going back to C.
struct Sites {
	uint8_t* at[16];
	int count;
};

static void chain(struct Jit* j, struct Sites* miss);

// The instruction through cpuJitStep, which adds its cycles. Leaves if it asks to or at the limit;
// at the end of the block goes on to the next.
// A block's first instruction: the load cycle from the mark the CPU comes in with (cpuLoadStall).
static void loadStallCheck(struct Emitter* e, uint32_t op) {
	uint8_t marks[2];
	int const count = cpuStallMarks(op, marks);
	if (!count) {
		return;
	}
	aLdrb(e, IP, CPU, CPU_OFF(loadMark));
	uint8_t* stall[2];
	for (int i = 0; i < count; ++i) {
		aDpImm(e, A_SUBS_IMM, ZR, IP, marks[i]);
		stall[i] = aBcond(e, C_EQ, NULL);
	}
	uint8_t* none = aB(e, NULL);
	for (int i = 0; i < count; ++i) {
		patchHere(e, stall[i]);
	}
	aDpImm(e, A_SUB_IMM | W64, LEFT, LEFT, 1);
	patchHere(e, none);
}

static void emitHelper(struct Jit* j, uint32_t op, uint64_t pc, uint32_t paddr, bool delay, bool lineStart, bool last, int prevMark, bool nextStall) {
	struct Emitter* e = &j->e;
	// The interpreter takes the load cycle (cpuLoadStall) from the mark before: the CPU's own at the
	// block's start, else the previous instruction's.
	if (prevMark >= 0) {
		storeByte(e, CPU_OFF(loadMark), (unsigned) prevMark);
	}
	if (lineStart) {
		if (!j->fast) {
			icacheCheck(j, (uint32_t) pc, paddr, true);
		}
	}
	if (delay) {
		delayPrologue(e);
	} else {
		storeConst(e, CPU_OFF(pc), pc + 4);
		storeConst(e, CPU_OFF(nextPc), pc + 8);
	}
	storeConst(e, CPU_OFF(curPc), pc);
	storeByte(e, CPU_OFF(inDelay), delay);
	storeByte(e, CPU_OFF(nextInDelay), 0);
	aDp(e, A_SUB | W64, 1, LIMIT, LEFT, S_LSL, 0);
	storeN(e, 1, N64_OFF(cycles), 8);
	if (j->fle) {
		storePinned(e);
	}
	aMovX(e, 0, N);
	aMovImm(e, 1, op);
	aMovImm64(e, IP, (uint64_t) (uintptr_t) cpuJitStep);
	aBlr(e, IP);
	if (j->fle) {
		loadPinned(e);
	}
	loadN(e, 1, N64_OFF(cycles), 8);
	loadN(e, LIMIT, N64_OFF(jitLimit), 8);
	aDp(e, A_SUB | W64, LEFT, LIMIT, 1, S_LSL, 0);
	aCbnz(e, 0, j->leave);
	aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
	aBcond(e, C_LE, j->leave);
	if (nextStall) {
		// An uncached load (its mark cleared) leaves no load cycle for the next instruction, which
		// counts one.
		aLdrb(e, 0, CPU, CPU_OFF(loadMark));
		uint8_t* cached = aCbnz(e, 0, NULL);
		aDpImm(e, A_ADD_IMM | W64, LEFT, LEFT, 1);
		patchHere(e, cached);
	}
	if (last) {
		aLdrb(e, 0, CPU, CPU_OFF(nextInDelay));
		aCbnz(e, 0, j->leave);
		chain(j, NULL);
	}
}

static int memSize(uint32_t op) {
	switch (op >> 26) {
	case 0x20: case 0x24: case 0x28:
		return 1;
	case 0x21: case 0x25: case 0x29:
		return 2;
	case 0x23: case 0x27: case 0x2B:
		return 4;
	case 0x37: case 0x3F:
		return 8;
	case 0x31: case 0x39:
		return 4;
	case 0x35: case 0x3D:
		return 8;
	}
	return 0;
}

// x5 = the FPR an FPU load or store of size bytes uses, as get32/set32 and get64/set64 find it:
// with Status.FR clear, odd 32-bit registers are the high half of the even one below.
static void fprAddress(struct Emitter* e, int ft, int size) {
	unsigned own = CPU_OFF(fpr) + 8 * ft;
	aAddImmX(e, 5, CPU, own);
	if (!(ft & 1)) {
		return;
	}
	aAddImmX(e, 7, CPU, CPU_OFF(fpr) + 8 * (ft & ~1) + (size == 4 ? 4 : 0));
	aLdr(e, IP, CPU, CPU_OFF(cop0) + 8 * CP0_STATUS);
	aLogicImm(e, A_ANDS_IMM, ZR, IP, 1u << 26);
	aCselX(e, 5, 7, 5, C_EQ);
}

// A load or store to cached RDRAM, through kseg0 or a page the TLB mapped (vpages), with
// dataTiming's D-cache check; anything else jumps to the slow path before any change. Like the
// interpreter, it takes the low 32 bits of the address: w0 the virtual address, w6 the physical
// one and w1 the D-cache line. A page not looked up, and a misaligned address, fail the tag check
// and take the slow path from the miss path.
// The page in w1 written: its byte in dirtyPages, which follows pageFlags (x26), to 0.
static void markWritten(struct Emitter* e) {
	aDp(e, A_ADD | W64, 2, PAGES, 1, S_LSL, 0);
	aStrb(e, ZR, 2, N64_OFF(dirtyPages) - N64_OFF(pageFlags));
}

// cpuRdpWaitRead or cpuRdpWaitWrite for the physical address in w6 (w0 kept); the pool's
// caller-saved registers are lost.
static void rdpWaitCall(struct Emitter* e, const void* f, int size) {
	aPush(e, 0, 6, 16);
	aMovX(e, 0, N);
	aMov(e, 1, 6);
	aMovImm(e, 2, (uint32_t) size);
	aMovImm64(e, IP, (uint64_t) (uintptr_t) f);
	aBlr(e, IP);
	aPop(e, 0, 6, 16);
}

static void emitMemory(struct Jit* j, struct Slow* slow) {
	struct Emitter* e = &j->e;
	uint32_t op = slow->op;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, size = memSize(op);
	int code = op >> 26;
	bool fpu = code == 0x31 || code == 0x35 || code == 0x39 || code == 0x3D;
	bool isStore = code >= 0x28 && code != 0x37 && code != 0x31 && code != 0x35;
	int32_t offset = (int16_t) op;
	if (fpu) {
		needCop1(j, false);
	}

	// With HLE accuracy an address known now (a constant base) that is kseg0's RDRAM, aligned, needs
	// no checks: w6 is its physical address.
	int32_t baseValue = 0;
	bool const known = j->fast && constant(j, rs, &baseValue);
	int64_t const at = (int64_t) baseValue + offset;
	bool const direct = known && at == (int32_t) at && ((uint32_t) at ^ 0x80000000u) < RDRAM_SIZE && !((uint32_t) at & (uint32_t) (size - 1));
	if (direct) {
		aMovImm(e, 6, (uint32_t) at ^ 0x80000000u);
	} else {
		// The address in 64 bits, which the interpreter takes when it isn't a 32-bit one sign-extended (an
		// address error); the low 32 bits from here on.
		int base = readReg(j, rs);
		if (base == ZR) {
			aMovImm(e, 0, (uint32_t) offset);
		} else {
			if (offset >= 0 && offset < 4096) {
				aDpImm(e, A_ADD_IMM | W64, 0, base, (unsigned) offset);
			} else if (offset < 0 && offset > -4096) {
				aDpImm(e, A_SUB_IMM | W64, 0, base, (unsigned) -offset);
			} else {
				aMovImm(e, 1, (uint32_t) offset);
				aSxtw(e, 1, 1);
				aDp(e, A_ADD | W64, 0, base, 1, S_LSL, 0);
			}
			if (!j->fle) {
				emit32(e, 0xEB20C01F); // cmp x0, w0, sxtw
				slowIf(j, C_NE);
			}
		}
		aLogicImm(e, A_EOR_IMM, 6, 0, 0x80000000);
		if (j->fast) {
			// kseg0's RDRAM and aligned in one test; anything else through vpages in the cold code.
			aLogicImm(e, A_ANDS_IMM, ZR, 6, 0xFF800000u | (j->fle ? 0 : (uint32_t) (size - 1)));
			struct Miss* v = &j->mapped[j->mappedCount++];
			v->site = aBcond(e, C_NE, NULL);
			v->resume = e->p;
			v->slow = slow;
			v->size = size;
			v->store = isStore;
			v->cache = j->cache;
			slow->keep = true;
		} else {
			// kseg0's RDRAM directly; other pages from vpages, where one not looked up lands past RDRAM.
			aLsrImm(e, 2, 6, 23);
			uint8_t* kseg0 = aCbz(e, 2, NULL);
			aLsrImm(e, 1, 0, 12);
			if (isStore) {
				aDpImm(e, A_ADD_IMM | W64, 3, TLB, VPAGES * 4);
				aLdst(e, L_LDR, 5, 3, 1, L_UXTW | 0x1000);
			} else {
				aLdst(e, L_LDR, 5, TLB, 1, L_UXTW | 0x1000);
			}
			aDp(e, A_ADD, 6, 0, 5, S_LSL, 0);
			patchHere(e, kseg0);
		}
	}
	// Stores go slow on any flag (code, RDP) unless the page has only code that they miss; loads
	// on pages the RDP may still be drawing to. With the D-cache's data only its fills and
	// write-backs reach memory, in C.
	if (j->cacheData) {
	} else if (isStore) {
		// The page written: its byte in dirtyPages, after pageFlags.
		aLsrImm(e, 1, 6, 12);
		markWritten(e);
		aLdst(e, L_LDRB, 1, PAGES, 1, L_UXTW);
		struct Miss* k = &j->codeChecks[j->codeCheckCount++];
		k->site = aCbnz(e, 1, NULL);
		k->resume = e->p;
		k->slow = slow;
		k->size = size;
		k->cache = j->cache;
		slow->keep = true;
	} else if (j->rdpThreads) {
		aLsrImm(e, 1, 6, 12);
		aLdst(e, L_LDRB, 1, PAGES, 1, L_UXTW);
		struct Miss* w = &j->rdpWaits[j->rdpWaitCount++];
		w->site = aTbnz(e, 1, 1, NULL);
		w->resume = e->p;
		w->size = size;
		w->cache = j->cache;
	}

	if (!j->fast) {
		// The line index from the virtual address; a hit when its tag is this physical line and the
		// address is aligned.
		aUbfx(e, 1, 0, 4, 9);
		aLdst(e, L_LDR, 3, DTAG, 1, L_UXTW | 0x1000);
		aDp(e, A_EOR, 2, 3, 6, S_LSL, 0);
		aLogicImm(e, A_ANDS_IMM, ZR, 2, 0xFFFFFFF0u | (size - 1));
		struct Miss* m = &j->misses[j->missCount++];
		m->site = aBcond(e, C_NE, NULL);
		m->resume = e->p;
		m->slow = slow;
		m->size = size;
		m->glLoad = j->rdpGl && !isStore;
		m->cache = j->cache;
		slow->keep = true;
		if (isStore) {
			aMovImm(e, 3, 1);
			aLdst(e, L_STRB, 3, DIRTY, 1, L_UXTW);
		}
	}
	// The data: RDRAM at w6, or the line's at x2 + w3.
	int mb = RAM, mi = 6;
	if (j->cacheData) {
		aAddImmX(e, 2, CPU, CPU_OFF(dcacheData));
		aLogicImm(e, A_AND_IMM, 3, 0, 0x1FFF);
		mb = 2;
		mi = 3;
	}

	if (isStore && !fpu) {
		int t = readReg(j, rt);
		switch (op >> 26) {
		case 0x28:
			aLdst(e, L_STRB, t, mb, mi, L_UXTW);
			return;
		case 0x29:
			aRev16(e, 4, t);
			aLdst(e, L_STRH, 4, mb, mi, L_UXTW);
			return;
		case 0x2B:
			aRev(e, 4, t);
			aLdst(e, L_STR, 4, mb, mi, L_UXTW);
			return;
		default:
			aRevX(e, 4, t);
			aLdst(e, L_STRX, 4, mb, mi, L_UXTW);
			return;
		}
	}
	switch (op >> 26) {
	case 0x31: case 0x35:
		aLdst(e, size == 4 ? L_LDR : L_LDRX, 4, mb, mi, L_UXTW);
		if (size == 4) {
			aRev(e, 4, 4);
		} else {
			aRevX(e, 4, 4);
		}
		fprAddress(e, rt, size);
		if (size == 4) {
			aStr(e, 4, 5, 0);
		} else {
			aStrX(e, 4, 5, 0);
		}
		return;
	case 0x39: case 0x3D:
		fprAddress(e, rt, size);
		if (size == 4) {
			aLdr(e, 4, 5, 0);
			aRev(e, 4, 4);
		} else {
			aLdrX(e, 4, 5, 0);
			aRevX(e, 4, 4);
		}
		aLdst(e, size == 4 ? L_STR : L_STRX, 4, mb, mi, L_UXTW);
		return;
	}
	int d = writeReg(j, rt);
	switch (op >> 26) {
	case 0x20:
		aLdst(e, L_LDRB, d, mb, mi, L_UXTW);
		aSxtbX(e, d, d);
		break;
	case 0x24:
		aLdst(e, L_LDRB, d, mb, mi, L_UXTW);
		break;
	case 0x21: case 0x25:
		aLdst(e, L_LDRH, d, mb, mi, L_UXTW);
		aRev16(e, d, d);
		if (op >> 26 == 0x21) {
			aSxthX(e, d, d);
		}
		break;
	case 0x23: case 0x27:
		aLdst(e, L_LDR, d, mb, mi, L_UXTW);
		aRev(e, d, d);
		if (op >> 26 == 0x23) {
			aSxtw(e, d, d);
		}
		break;
	default:
		aLdst(e, L_LDRX, d, mb, mi, L_UXTW);
		aRevX(e, d, d);
		break;
	}
	commit(j, rt, d);
}

// Goes on to the block at the pc in memory if the jump table has it, or leaves.
// Runs of loads and stores through the same base register (context switches, prologues, copies):
// how many instructions from ops[k] make one, 0 for under three. The run stops before the block's
// last instruction, a load that changes the base, or an offset not a multiple of its size.
static uint32_t fusedRunLength(const uint32_t* ops, uint32_t k, uint32_t count) {
	int rs = (ops[k] >> 21) & 31;
	int32_t lo = (int16_t) ops[k], hi = lo;
	uint32_t n = 0;
	while (k + n + 1 < count) {
		uint32_t op = ops[k + n];
		int code = op >> 26;
		bool memory = (code >= 0x20 && code <= 0x2B && code != 0x22 && code != 0x26 && code != 0x2A) || code == 0x37 || code == 0x3F;
		int size = memory ? memSize(op) : 1;
		int32_t off = (int16_t) op;
		bool load = code < 0x28 || code == 0x37;
		if (!memory || (int) ((op >> 21) & 31) != rs || (off & (size - 1)) || (load && (int) ((op >> 16) & 31) == rs)) {
			break;
		}
		lo = off < lo ? off : lo;
		hi = off + size - 1 > hi ? off + size - 1 : hi;
		if (hi - lo >= 512) {
			break;
		}
		++n;
	}
	return n >= 3 ? n : 0;
}

// The run, when nothing in it can leave its fast path: the cycles can't reach the limit, its
// I-cache lines are in, the base is aligned for every access, all of them are in one RDRAM page with
// no flags, and every D-cache line they span is in. Every access then hits and takes no cycle more
// than its instruction, so the run is its loads and stores in order and its cycles at once. Anything else takes the
// instructions one by one (emitted after this), from the start of the run. Returns the branch to
// the end of the run.
static uint8_t* fuseRun(struct Jit* j, const uint32_t* ops, uint32_t len, uint32_t vaddr, uint32_t paddr, int prevMark) {
	struct Emitter* e = &j->e;
	uint8_t* out[96];
	int sites = 0;
	int rs = (ops[0] >> 21) & 31, align = 1;
	int32_t lo = (int16_t) ops[0], hi = lo;
	bool stores = false;
	for (uint32_t i = 0; i < len; ++i) {
		int32_t off = (int16_t) ops[i];
		int size = memSize(ops[i]);
		lo = off < lo ? off : lo;
		hi = off + size - 1 > hi ? off + size - 1 : hi;
		align = size > align ? size : align;
		stores |= ops[i] >> 26 >= 0x28 && ops[i] >> 26 != 0x37;
	}
	// Its load interlocks, as the instructions one by one take them; the first one's from the block
	// before is known only when running, so the limit allows for it.
	uint32_t cycles = len * (uint32_t) j->opCycles;
	bool markCheck = false;
	if (!j->fle) {
		cycles += prevMark >= 0 && cpuLoadStall(ops[0], (uint8_t) prevMark);
		for (uint32_t i = 1; i < len; ++i) {
			cycles += cpuLoadStall(ops[i], cpuLoadMark(ops[i - 1]));
		}
		markCheck = prevMark < 0;
	}
	// The base first: the instructions one by one hold it from here whichever way the checks go. The
	// run reads and writes the registers in memory.
	writeBack(j);
	int b = readReg(j, rs);
	aMovImm(e, 1, cycles + markCheck);
	aDp(e, A_SUBS | W64, ZR, LEFT, 1, S_LSL, 0);
	out[sites++] = aBcond(e, C_LE, NULL);
	for (uint32_t p = paddr & ~31u; p < paddr + 4 * len && !j->fast; p += 32) {
		uint32_t v = vaddr + (p - paddr);
		aLdr(e, 0, CPU, CPU_OFF(icacheTag) + 4 * ((v >> 5) & 511));
		aMovImm(e, 1, p & ~31u);
		aDp(e, A_SUBS, ZR, 0, 1, S_LSL, 0);
		out[sites++] = aBcond(e, C_NE, NULL);
	}
	if (align > 1) {
		aLogicImm(e, A_ANDS_IMM, ZR, b, align - 1);
		out[sites++] = aBcond(e, C_NE, NULL);
	}
	if (b == ZR) {
		aMovImm(e, 0, (uint32_t) lo);
		aMovImm(e, 8, (uint32_t) hi);
	} else {
		aAddImm(e, 0, b, lo, 1);
		aAddImm(e, 8, b, hi, 1);
	}
	aDp(e, A_EOR, 1, 0, 8, S_LSL, 0);
	aLsrImm(e, 1, 1, 12);
	out[sites++] = aCbnz(e, 1, NULL);
	aLsrImm(e, 1, 0, 12);
	if (stores) {
		aDpImm(e, A_ADD_IMM | W64, 3, TLB, VPAGES * 4);
		aLdst(e, L_LDR, 5, 3, 1, L_UXTW | 0x1000);
	} else {
		aLdst(e, L_LDR, 5, TLB, 1, L_UXTW | 0x1000);
	}
	aDp(e, A_ADD, 6, 0, 5, S_LSL, 0);
	aLsrImm(e, 1, 6, 12);
	if (stores && !j->cacheData) {
		markWritten(e);
	}
	aLdst(e, L_LDRB, 1, PAGES, 1, L_UXTW);
	if (stores) {
		out[sites++] = aCbnz(e, 1, NULL);
	} else {
		out[sites++] = aTbnz(e, 1, 7, NULL);
		if (j->rdpThreads) {
			out[sites++] = aTbnz(e, 1, 1, NULL);
		}
	}
	// The D-cache lines from the first address's to the last's; without caches, RDRAM.
	if (j->fast) {
		aLsrImm(e, 1, 6, 23);
		out[sites++] = aCbnz(e, 1, NULL);
	}
	for (int32_t step = 0; step <= hi - lo + 15 && !j->fast; step += 16) {
		int32_t at = step > hi - lo ? hi - lo : step;
		if (at) {
			aAddImm(e, 1, 0, at, 3);
			aUbfx(e, 1, 1, 4, 9);
		} else {
			aUbfx(e, 1, 0, 4, 9);
		}
		aLdst(e, L_LDR, 3, DTAG, 1, L_UXTW | 0x1000);
		aAddImm(e, 4, 6, at, 17);
		aDp(e, A_EOR, 3, 3, 4, S_LSL, 0);
		aLogicImm(e, A_ANDS_IMM, ZR, 3, 0xFFFFFFF0u);
		out[sites++] = aBcond(e, C_NE, NULL);
	}
	// x7 = the host address of the base's physical address less lo rounded down to 8, so every
	// access is at a positive offset that is a multiple of its size.
	int32_t floor8 = lo & ~7;
	if (j->cacheData) {
		// The lines' data, by virtual address, when the run doesn't wrap round the D-cache.
		aMov(e, 2, 0);
		if (lo - floor8) {
			aDpImm(e, A_SUB_IMM, 2, 2, (unsigned) (lo - floor8));
		}
		aLogicImm(e, A_AND_IMM, 2, 2, 0x1FFF);
		aAddImm(e, 4, 2, hi - floor8, 5);
		aLsrImm(e, 4, 4, 13);
		out[sites++] = aCbnz(e, 4, NULL);
		aAddImmX(e, 4, CPU, CPU_OFF(dcacheData));
		aDp(e, A_ADD | W64, 7, 4, 2, S_LSL, 0);
	} else {
		aMov(e, 7, 6);
		if (lo - floor8) {
			aDpImm(e, A_SUB_IMM | W64, 7, 7, (unsigned) (lo - floor8));
		}
		aDp(e, A_ADD | W64, 7, RAM, 7, S_LSL, 0);
	}
	if (stores) {
		aMovImm(e, 3, 1);
	}
	// The run moves registers through memory.
	if (j->fle) {
		storePinned(e);
	}
	for (uint32_t i = 0; i < len; ++i) {
		uint32_t op = ops[i];
		int rt = (op >> 16) & 31, code = op >> 26;
		unsigned d = (unsigned) ((int16_t) op - floor8);
		if (code < 0x28 || code == 0x37) {
			switch (code) {
			case 0x20: aLdrb(e, 4, 7, d); aSxtbX(e, 4, 4); break;
			case 0x24: aLdrb(e, 4, 7, d); break;
			case 0x21: aLdrh(e, 4, 7, d); aRev16(e, 4, 4); aSxthX(e, 4, 4); break;
			case 0x25: aLdrh(e, 4, 7, d); aRev16(e, 4, 4); break;
			case 0x23: aLdr(e, 4, 7, d); aRev(e, 4, 4); aSxtw(e, 4, 4); break;
			case 0x27: aLdr(e, 4, 7, d); aRev(e, 4, 4); break;
			default: aLdrX(e, 4, 7, d); aRevX(e, 4, 4);
			}
			if (rt) {
				aStrX(e, 4, CPU, gprOff(rt));
			}
			continue;
		}
		int v = ZR;
		if (rt) {
			aLdrX(e, 4, CPU, gprOff(rt));
			v = 4;
		}
		switch (code) {
		case 0x28: aStrb(e, v, 7, d); break;
		case 0x29: if (rt) { aRev16(e, 4, 4); } aStrh(e, v, 7, d); break;
		case 0x2B: if (rt) { aRev(e, 4, 4); } aStr(e, v, 7, d); break;
		default: if (rt) { aRevX(e, 4, 4); } aStrX(e, v, 7, d);
		}
		// The line it wrote is dirty (with caches).
		if (!j->fast) {
			aAddImm(e, 1, 0, (int16_t) op - lo, 2);
			aUbfx(e, 1, 1, 4, 9);
			aLdst(e, L_STRB, 3, DIRTY, 1, L_UXTW);
		}
	}
	if (j->fle) {
		loadPinned(e);
	}
	if (markCheck) {
		loadStallCheck(e, ops[0]);
	}
	aMovImm(e, 1, cycles);
	aDp(e, A_SUB | W64, LEFT, LEFT, 1, S_LSL, 0);
	uint8_t* done = aB(e, NULL);
	for (int i = 0; i < sites; ++i) {
		patchHere(e, out[i]);
	}
	return done;
}

// A poll loop (blockPollPass) about to go round again, pc back at its start (checked unless the
// caller knows it is), skips the whole passes of pass cycles that fit before the limit when all its
// I-cache and D-cache lines are in: every pass then reads the same memory, takes the same branch and
// costs the same.
static void pollSkip(struct Jit* j, const uint32_t* ops, uint32_t count, uint32_t vaddr, uint32_t paddr, uint32_t pass, int reload, bool checkPc) {
	struct Emitter* e = &j->e;
	uint8_t* out[64];
	int sites = 0;
	if (j->fle) {
		storePinned(e);
	}
	if (checkPc) {
		aLdrX(e, 0, CPU, CPU_OFF(pc));
		movConst(e, 1, sext32(vaddr));
		aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
		out[sites++] = aBcond(e, C_NE, NULL);
	}
	if (reload) {
		aLdrX(e, 0, CPU, gprOff(reload & 31));
		aDp(e, A_SUBS | W64, ZR, 0, ZR, S_LSL, 0);
		out[sites++] = aBcond(e, reload & 32 ? C_NE : C_EQ, NULL);
	}
	for (uint32_t p = paddr & ~31u; p < paddr + 4 * count && !j->fast; p += 32) {
		uint32_t v = vaddr + (p - paddr);
		aLdr(e, 0, CPU, CPU_OFF(icacheTag) + 4 * ((v >> 5) & 511));
		aMovImm(e, 1, p & ~31u);
		aDp(e, A_SUBS, ZR, 0, 1, S_LSL, 0);
		out[sites++] = aBcond(e, C_NE, NULL);
	}
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k];
		if (op >> 26 < 0x20) {
			continue;
		}
		int rs = (op >> 21) & 31, size = memSize(op);
		if (rs) {
			aLdr(e, 0, CPU, gprOff(rs));
			aAddImm(e, 0, 0, (int16_t) op, 1);
		} else {
			aMovImm(e, 0, (uint32_t) (int32_t) (int16_t) op);
		}
		aLsrImm(e, 1, 0, 12);
		aLdst(e, L_LDR, 5, TLB, 1, L_UXTW | 0x1000);
		aDp(e, A_ADD, 6, 0, 5, S_LSL, 0);
		if (j->rdpThreads) {
			aLsrImm(e, 1, 6, 12);
			aLdst(e, L_LDRB, 1, PAGES, 1, L_UXTW);
			out[sites++] = aTbnz(e, 1, 1, NULL);
		}
		if (j->fast) {
			aLsrImm(e, 1, 6, 23);
			out[sites++] = aCbnz(e, 1, NULL);
			if (size > 1) {
				aLogicImm(e, A_ANDS_IMM, ZR, 0, size - 1);
				out[sites++] = aBcond(e, C_NE, NULL);
			}
			continue;
		}
		aUbfx(e, 1, 0, 4, 9);
		aLdst(e, L_LDR, 3, DTAG, 1, L_UXTW | 0x1000);
		aDp(e, A_EOR, 2, 3, 6, S_LSL, 0);
		aLogicImm(e, A_ANDS_IMM, ZR, 2, 0xFFFFFFF0u | (size - 1));
		out[sites++] = aBcond(e, C_NE, NULL);
	}
	aDpImm(e, A_SUB_IMM, 0, LEFT, 1);
	aMovImm(e, 1, pass);
	aUdiv(e, 0, 0, 1);
	aMadd(e, 0, 0, 1, ZR);
	aDp(e, A_SUB | W64, LEFT, LEFT, 0, S_LSL, 0);
	for (int i = 0; i < sites; ++i) {
		patchHere(e, out[i]);
	}
}

void* jitLink(struct N64* n, uint32_t* site, uint32_t vaddr);

// The exits most passes take: the block's end, a branch inside it taken, a likely branch not taken.
static bool hotExit(const struct Exit* x) {
	return x->last || x->side || x->skip;
}

// A site's branch to cold code, through an unconditional one here.
static uint8_t* farSite(struct Emitter* e, uint8_t* site) {
	patchHere(e, site);
	return aB(e, NULL);
}

// To the block at target when the pc is there (known: it is): a jump that jitLink makes a direct one.
// Code the TLB maps keeps its page's vpages entry from then just after the jump, and goes on to the
// code after this while the page maps elsewhere.
static void linkTo(struct Jit* j, uint64_t target, struct Sites* miss, bool known) {
	struct Emitter* e = &j->e;
	uint8_t* other = NULL;
	if (!known) {
		aLdrX(e, 0, CPU, CPU_OFF(pc));
		movConst(e, 1, target);
		aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
		other = aBcond(e, C_NE, NULL);
	}
	uint8_t* moved = NULL;
	bool mapped = ((uint32_t) target & 0xC0000000u) != 0x80000000u;
	if (mapped) {
		aMovImm(e, 1, (uint32_t) target >> 12);
		aLdst(e, L_LDR, 4, TLB, 1, L_UXTW | 0x1000);
		emit32(e, 0x18000005 | 4 << 5);
		aDp(e, A_SUBS, ZR, 4, 5, S_LSL, 0);
		moved = aBcond(e, C_NE, NULL);
	}
	uint8_t* site = aB(e, NULL);
	if (mapped) {
		emit32(e, UINT32_MAX);
	}
	// The call, run once, goes with the cold code when there is some; a miss comes back here.
	struct Emitter* c = j->cold ? j->cold : e;
	uint8_t* missed = NULL;
	if (miss && c != e) {
		missed = e->p;
		miss->at[miss->count++] = aB(e, NULL);
	}
	patchBranch(site, c->p);
	aMovX(c, 0, N);
	aMovImm64(c, 1, (uint64_t) (uintptr_t) site);
	aMovImm(c, 2, (uint32_t) target);
	aMovImm64(c, IP, (uint64_t) (uintptr_t) jitLink);
	aBlr(c, IP);
	uint8_t* found = aCbnzX(c, 0, NULL);
	if (missed) {
		aB(c, missed);
	} else if (miss) {
		miss->at[miss->count++] = aB(c, NULL);
	} else {
		aB(c, j->leave);
	}
	patchHere(c, found);
	aBr(c, 0);
	if (other) {
		patchHere(e, other);
	}
	if (moved) {
		patchHere(e, moved);
	}
}

static void leaveIf(struct Jit* j, struct Sites* miss, int cond) {
	if (miss) {
		miss->at[miss->count++] = aBcond(&j->e, cond, NULL);
	} else {
		aBcond(&j->e, cond, j->leave);
	}
}

// To the block at the pc: through the two entries kept after this code ({vaddr, gen, code}, the
// latest first), else the jitTable (whose hit goes into the first entry), else the way out. The first
// entry is checked here, the rest in the shared code (chainRest), which comes back on a miss.
static void chain(struct Jit* j, struct Sites* miss) {
	struct Emitter* e = &j->e;
	aLdrX(e, 0, CPU, CPU_OFF(pc));
	aSxtw(e, 1, 0);
	aDp(e, A_SUBS | W64, ZR, 1, 0, S_LSL, 0);
	leaveIf(j, miss, C_NE);
	// Entries are good while their page maps where they were compiled (jitTag): w4, the address
	// through vpages.
	aLsrImm(e, 4, 0, 12);
	aLdst(e, L_LDR, 4, TLB, 4, L_UXTW | 0x1000);
	aDp(e, A_ADD, 4, 4, 0, S_LSL, 0);
	uint8_t* adr = e->p;
	emit32(e, 0);
	aLdpW(e, 3, 6, 2, 0);
	aDp(e, A_SUBS, ZR, 3, 0, S_LSL, 0);
	uint8_t* other = aBcond(e, C_NE, NULL);
	aDp(e, A_SUBS, ZR, 6, 4, S_LSL, 0);
	uint8_t* stale = aBcond(e, C_NE, NULL);
	aLdrX(e, 5, 2, 8);
	aBr(e, 5);
	patchHere(e, other);
	patchHere(e, stale);
	aBl(e, j->arena + CHAIN_REST);
	if (miss) {
		miss->at[miss->count++] = aB(e, NULL);
	} else {
		aB(e, j->leave);
	}
	if ((uintptr_t) e->p & 7) {
		emit32(e, 0xD503201F);
	}
	if (e->p + 32 <= e->end) {
		int32_t d = (int32_t) (e->p - adr);
		uint32_t op = 0x10000000 | (uint32_t) (d & 3) << 29 | (uint32_t) ((d >> 2) & 0x7FFFF) << 5 | 2;
		memcpy(adr, &op, 4);
	}
	for (int i = 0; i < 2; ++i) {
		emit32(e, 0);
		emit32(e, UINT32_MAX);
		emit32(e, 0);
		emit32(e, 0);
	}
}

// chain's second entry and the jitTable, from x0 = pc, w4 = its tag and x2 = the entries; back to
// the caller on a miss.
static void chainRest(struct Emitter* e) {
	aLdpW(e, 3, 6, 2, 16);
	aDp(e, A_SUBS, ZR, 3, 0, S_LSL, 0);
	uint8_t* other = aBcond(e, C_NE, NULL);
	aDp(e, A_SUBS, ZR, 6, 4, S_LSL, 0);
	uint8_t* stale = aBcond(e, C_NE, NULL);
	aLdrX(e, 5, 2, 24);
	aBr(e, 5);
	patchHere(e, other);
	patchHere(e, stale);
	aUbfx(e, 1, 0, 2, JIT_TABLE_BITS);
	aAddImmX(e, 7, N, N64_OFF(jitTable));
	aDp(e, A_ADD | W64, 7, 7, 1, S_LSL, 4);
	aLdpW(e, 3, 6, 7, (unsigned) offsetof(struct JitEntry, vaddr));
	aDp(e, A_SUBS, ZR, 3, 0, S_LSL, 0);
	uint8_t* missed = aBcond(e, C_NE, NULL);
	aDp(e, A_SUBS, ZR, 6, 4, S_LSL, 0);
	uint8_t* missed2 = aBcond(e, C_NE, NULL);
	aLdrX(e, 5, 7, (unsigned) offsetof(struct JitEntry, code));
	aLdrX(e, 3, 2, 0);
	aLdrX(e, 6, 2, 8);
	aStrX(e, 3, 2, 16);
	aStrX(e, 6, 2, 24);
	aStpW(e, 0, 4, 2, 0);
	aStrX(e, 5, 2, 8);
	aBr(e, 5);
	patchHere(e, missed);
	patchHere(e, missed2);
	aRet(e);
}

// To the block at the pc through the jitTable, else back to C.
static void lookup(struct Emitter* e, uint8_t* leave) {
	aLdrX(e, 0, CPU, CPU_OFF(pc));
	aSxtw(e, 1, 0);
	aDp(e, A_SUBS | W64, ZR, 1, 0, S_LSL, 0);
	aBcond(e, C_NE, leave);
	aLsrImm(e, 4, 0, 12);
	aLdst(e, L_LDR, 4, TLB, 4, L_UXTW | 0x1000);
	aDp(e, A_ADD, 4, 4, 0, S_LSL, 0);
	aUbfx(e, 1, 0, 2, JIT_TABLE_BITS);
	aAddImmX(e, 7, N, N64_OFF(jitTable));
	aDp(e, A_ADD | W64, 7, 7, 1, S_LSL, 4);
	aLdpW(e, 3, 6, 7, (unsigned) offsetof(struct JitEntry, vaddr));
	aDp(e, A_SUBS, ZR, 3, 0, S_LSL, 0);
	aBcond(e, C_NE, leave);
	aDp(e, A_SUBS, ZR, 6, 4, S_LSL, 0);
	aBcond(e, C_NE, leave);
	aLdrX(e, 5, 7, (unsigned) offsetof(struct JitEntry, code));
	aBr(e, 5);
}

// The arena starts with the way in, jitEnter(n, code), and the way out back to C.
static uint8_t* arenaGet(struct N64* n) {
	if (n->jitArena) {
		return n->jitArena;
	}
	void* p = mmap(NULL, ARENA_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (p == MAP_FAILED) {
		return NULL;
	}
	n->jitArena = p;
	struct Emitter e = { p, p, (uint8_t*) p + SHARED_SIZE - 8 };
	aPush(&e, 29, 30, 96);
	aStp(&e, 19, 20, 16);
	aStp(&e, 21, 22, 32);
	aStp(&e, 23, 24, 48);
	aStp(&e, 25, 26, 64);
	aStp(&e, 27, 28, 80);
	aMovX(&e, N, 0);
	aAddImmX(&e, CPU, N, N64_OFF(cpu));
	loadN(&e, 0, N64_OFF(cycles), 8);
	loadN(&e, LIMIT, N64_OFF(jitLimit), 8);
	aDp(&e, A_SUB | W64, LEFT, LIMIT, 0, S_LSL, 0);
	loadN(&e, RAM, N64_OFF(rdram), 8);
	loadN(&e, TLB, N64_OFF(vpages), 8);
	aAddImmX(&e, PAGES, N, N64_OFF(pageFlags));
	if (n->fle) {
		loadPinned(&e);
	} else {
		aAddImmX(&e, DTAG, CPU, CPU_OFF(dcacheTag));
		aAddImmX(&e, DIRTY, CPU, CPU_OFF(dcacheDirty));
	}
	aBr(&e, 1);
	n->jitLeave = e.p;
	if (n->fle) {
		storePinned(&e);
	}
	aDp(&e, A_SUB | W64, 0, LIMIT, LEFT, S_LSL, 0);
	storeN(&e, 0, N64_OFF(cycles), 8);
	aLdp(&e, 19, 20, 16);
	aLdp(&e, 21, 22, 32);
	aLdp(&e, 23, 24, 48);
	aLdp(&e, 25, 26, 64);
	aLdp(&e, 27, 28, 80);
	aPop(&e, 29, 30, 96);
	aRet(&e);
	n->jitRedispatch = e.p;
	lookup(&e, n->jitLeave);
	if (e.p > (uint8_t*) p + CHAIN_REST) {
		munmap(p, ARENA_SIZE);
		n->jitArena = NULL;
		return NULL;
	}
	e.p = (uint8_t*) p + CHAIN_REST;
	chainRest(&e);
	__builtin___clear_cache((char*) p, (char*) e.p);
	n->jitEnter = (void (*)(struct N64*, void*)) p;
	return p;
}

void jitReset(struct N64* n) {
	n->jitUsed = SHARED_SIZE;
	if (n->jitArena) {
		*coldUsed(n) = 0;
	}
	n->linkCount = 0;
	++n->blockGen;
}

// A thrown-away block's code starts again at jitRedispatch (any jump to it comes with the pc set).
void jitKill(struct N64* n, void* code) {
	uint32_t* p = code;
	*p = 0x14000000 | ((uint32_t) (((uint8_t*) n->jitRedispatch - (uint8_t*) p) >> 2) & 0x03FFFFFF);
	__builtin___clear_cache((char*) p, (char*) (p + 1));
}

void jitUnlink(struct N64* n) {
	for (uint32_t i = 0; i < n->linkCount; ++i) {
		*n->links[i].site = n->links[i].was;
		__builtin___clear_cache((char*) n->links[i].site, (char*) (n->links[i].site + 1));
	}
	n->linkCount = 0;
	n->linkGen = n->blockGen;
}

// From a block's jump to vaddr (see linkTo): its code, and the jump at site made a direct one to it;
// NULL to leave.
void* jitLink(struct N64* n, uint32_t* site, uint32_t vaddr) {
	struct JitEntry* je = &n->jitTable[(vaddr >> 2) & (JIT_TABLE_SIZE - 1)];
	if (je->vaddr != vaddr || je->gen != jitTag(n, vaddr) || n->linkGen != n->blockGen) {
		return NULL;
	}
	if ((vaddr & 0xC0000000u) != 0x80000000u) {
		site[1] = n->vpages[vaddr >> 12];
	}
	if (n->linkCount == n->linkCap) {
		uint32_t cap = n->linkCap ? 2 * n->linkCap : 1024;
		struct JitLink* links = realloc(n->links, cap * sizeof(*links));
		if (!links) {
			return je->code;
		}
		n->links = links;
		n->linkCap = cap;
	}
	n->links[n->linkCount++] = (struct JitLink) { site, *site };
	*site = 0x14000000 | ((uint32_t) (((uint8_t*) je->code - (uint8_t*) site) >> 2) & 0x03FFFFFF);
	__builtin___clear_cache((char*) site, (char*) (site + 1));
	return je->code;
}

void jitFree(struct N64* n) {
	free(n->links);
	n->links = NULL;
	n->linkCount = n->linkCap = 0;
	if (n->jitArena) {
		munmap(n->jitArena, ARENA_SIZE);
		n->jitArena = NULL;
	}
	n->jitUsed = SHARED_SIZE;
}

// Code for count instructions at vaddr (cached, at paddr in RDRAM), or NULL when the arena is full.
void* jitCompile(struct N64* n, const uint32_t* ops, uint32_t count, uint32_t vaddr, uint32_t paddr) {
	uint8_t* arena = arenaGet(n);
	if (n->jitUsed < SHARED_SIZE) {
		n->jitUsed = SHARED_SIZE;
	}
	if (!arena || n->jitUsed + BLOCK_CODE_MAX > ARENA_SIZE / 2 || *coldUsed(n) + BLOCK_CODE_MAX > ARENA_SIZE / 2) {
		return NULL;
	}
	static struct Jit jit;
	struct Jit* j = &jit;
	uint8_t* start = arena + n->jitUsed;
	j->e = (struct Emitter) { start, start, start + BLOCK_CODE_MAX };
	j->exitCount = j->slowCount = j->missCount = j->codeCheckCount = j->rdpWaitCount = j->mappedCount = 0;
	j->rdpThreads = n->rdpThreads > 0 || n->rdpGl;
	j->rdpGl = n->rdpGl;
	j->fast = n->fast;
	j->fle = n->fast && n->fle;
	j->plainCount = n->countCycles == 2;
	j->opCycles = (int) n->opCycles;
	j->cacheData = !n->compat && !n->fast;
	j->fr = n->cpu.cop0[CP0_STATUS] & (1 << 26);
	j->cu1Seen = j->frSeen = false;
	j->cold = NULL;
	j->arena = arena;
	struct Emitter* e = &j->e;
	j->leave = e->p;
	aB(e, n->jitLeave);
	uint8_t* entry = e->p;
	int reload;
	uint32_t pollPass = blockPollPass(ops, count, vaddr, &reload);
	// A poll loop going round in here tries its skip once a time it is entered: where it can't, the
	// tries would cost more than the passes.
	if (pollPass) {
		storeByte(e, CPU_OFF(pollTried), 0);
	}
	uint8_t* again = e->p;
	// FLE: the cycles of the instructions since the last count.
	int pending = 0;

	if (blockIsIdleLoop(ops, count, vaddr)) {
		skipIdle(e, vaddr, paddr, j->fast, j->opCycles);
	}
	// A conditional branch or jump at the end back to the start.
	bool loopsBack = false;
	if (count >= 2) {
		uint32_t b = ops[count - 2];
		uint32_t at = vaddr + 4 * (count - 2);
		int code = b >> 26;
		if ((code >= 0x04 && code <= 0x07) || (code >= 0x14 && code <= 0x17) || (code == 0x01 && !(b & 0x00100000))) {
			loopsBack = at + 4 + ((uint32_t) (int16_t) b << 2) == vaddr;
		} else if (code == 0x02) {
			loopsBack = (((at + 4) & 0xF0000000) | ((b & 0x03FFFFFF) << 2)) == vaddr;
		}
	}

	// Where the block goes on to at its end, for direct jumps: a branch's target and past its delay
	// slot (not known for JR and JALR).
	uint64_t ends[2];
	int endCount = 0;
	uint32_t endOp = count >= 2 ? ops[count - 2] : 0, endAt = vaddr + 4 * (count - 2);
	if (count < 2 || !isBranchOp(endOp)) {
		ends[endCount++] = sext32(vaddr + 4 * count);
	} else if (endOp >> 26 == 0x02 || endOp >> 26 == 0x03) {
		ends[endCount++] = sext32(((endAt + 4) & 0xF0000000) | ((endOp & 0x03FFFFFF) << 2));
	} else if (endOp >> 26) {
		ends[endCount++] = sext32(endAt + 4 + ((uint32_t) (int16_t) endOp << 2));
		ends[endCount++] = sext32(vaddr + 4 * count);
	}

	cacheClear(&j->cache);
	j->known = 0;
	bool delay = false;
	// A fused run's end, where its fast path joins the instructions one by one.
	uint8_t* fusedDone = NULL;
	uint32_t fusedEnd = 0;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k], p = paddr + k * 4;
		if (fusedDone && k == fusedEnd) {
			writeBack(j);
			patchHere(e, fusedDone);
			cacheClear(&j->cache);
			j->known = 0;
			j->cu1Seen = j->frSeen = false;
			fusedDone = NULL;
		}
		uint32_t run = delay ? 0 : fusedRunLength(ops, k, count);
		if (run && !fusedDone) {
			fusedDone = fuseRun(j, ops + k, run, vaddr + 4 * k, p, k ? cpuLoadMark(ops[k - 1]) : -1);
			fusedEnd = k + run;
		}
		uint64_t pc = sext32(vaddr + k * 4);
		bool last = k + 1 == count;
		bool lineStart = k == 0 || !(p & 31);
		enum Kind kind;
		uint8_t* mark = e->p;
		int exits = j->exitCount, misses = j->missCount, codeChecks = j->codeCheckCount;
		struct Cache cache = j->cache;
		uint32_t const known = j->known;
		int32_t values[32];
		memcpy(values, j->value, sizeof(values));
		bool branch = isBranchOp(op);
		// A branch inside the block (see throughBranch), and its delay slot, after which the block
		// leaves if the branch is taken; either through the interpreter leaves the block there.
		bool inside = branch && !delay && k + 2 < count, afterInside = delay && !last;
		struct Slow* slow = &j->slows[j->slowCount];
		int const prevMark = k ? cpuLoadMark(ops[k - 1]) : -1;
		j->mark = cpuLoadMark(op);
		j->branchStall = prevMark >= 0 ? (int8_t) cpuLoadStall(op, (uint8_t) prevMark) : -1;
		*slow = (struct Slow) { .op = op, .paddr = p, .pc = pc, .delay = delay, .lineStart = lineStart, .last = last || inside || afterInside, .prevMark = prevMark,
		                    .nextStall = k + 1 < count && cpuLoadStall(ops[k + 1], cpuLoadMark(op)) };
		j->slow = slow;
		j->extraCycles = 0;
		j->interruptCheck = false;
		j->jumped = false;
		j->inside = inside;
		// A branch's I-cache check comes first (a likely branch not taken leaves inside it); the rest
		// check after, so slow paths and Count reads see the cycles before this fetch.
		if (lineStart && branch && !delay && !j->fast) {
			icacheCheck(j, (uint32_t) pc, p, false);
		}
		if (!(delay && branch) && emitInline(j, op, pc, &kind)) {
			if (lineStart && !branch && !j->fast) {
				icacheCheck(j, (uint32_t) pc, p, false);
			}
			if (j->fle) {
				// No load interlocks; the cycles go at the exits.
				pending += j->opCycles + j->extraCycles;
				if (last || afterInside || j->interruptCheck || j->jumped || pending > 4000) {
					aDpImm(e, A_SUBS_IMM | W64, LEFT, LEFT, (unsigned) pending);
					pending = 0;
				}
			} else {
				if (prevMark >= 0) {
					j->extraCycles += cpuLoadStall(op, (uint8_t) prevMark);
				} else {
					loadStallCheck(e, op);
				}
				aDpImm(e, A_SUBS_IMM | W64, LEFT, LEFT, j->opCycles + j->extraCycles);
			}
			if (j->interruptCheck) {
				// An interrupt now due is taken before the next instruction.
				exitAt(j, aCbnz(e, 2, NULL), pc, kind, delay, false);
			}
			if (j->jumped) {
				exitAt(j, aB(e, NULL), pc, kind, delay, false);
				j->exits[j->exitCount - 1].jumped = true;
			} else if (last) {
				exitAt(j, aB(e, NULL), pc, kind, delay, false);
				j->exits[j->exitCount - 1].last = true;
			} else if (!j->fast) {
				// hle checks the limit at the block's end only.
				exitAt(j, aBcond(e, C_LE, NULL), pc, kind, delay, false);
			}
			if (afterInside) {
				// Taken, out to the target; not, on with the block if there are cycles left, as at a
				// block's end.
				uint32_t b = ops[k - 1], at = vaddr + 4 * (k - 1);
				exitAt(j, aCbnz(e, 29, NULL), pc, kind, delay, false);
				j->exits[j->exitCount - 1].side = true;
				j->exits[j->exitCount - 1].to = sext32(at + 4 + ((uint32_t) (int16_t) b << 2));
				aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
				exitAt(j, aBcond(e, C_LE, NULL), pc, kind, delay, false);
			}
			if (slow->sites || slow->keep) {
				slow->resume = e->p;
				slow->cache = j->cache;
				++j->slowCount;
			}
			delay = kind != PLAIN;
			continue;
		}
		e->p = mark;
		j->exitCount = exits;
		j->missCount = misses;
		j->codeCheckCount = codeChecks;
		j->cache = cache;
		j->known = known;
		memcpy(j->value, values, sizeof(values));
		writeBack(j);
		j->cu1Seen = j->frSeen = false;
		emitHelper(j, op, pc, p, delay, lineStart, last || inside || afterInside, prevMark, slow->nextStall);
		if (inside || afterInside) {
			break;
		}
		// The interpreter may have changed any register, and the call the scratch ones.
		cacheClear(&j->cache);
		j->known = 0;
		delay = branch;
	}

	// Leaving after an inline instruction: the state the interpreter would have after it. The hot
	// exits first, straight after the block, then the branches to the cold code, the cold code itself
	// and the cold exits.
	struct Emitter hot;
	uint8_t* const hotLeave = j->leave;
	uint8_t* const coldStart = arena + ARENA_SIZE / 2 + *coldUsed(n);
	struct Emitter cold = { coldStart, coldStart, coldStart + BLOCK_CODE_MAX };
	uint8_t* const coldLeave = cold.p;
	aB(&cold, n->jitLeave);
	j->cold = &cold;
	for (int pass = 1; pass >= 0; --pass) {
	if (pass == 0) {
		// To the cold code: a branch here each, which reaches it (a conditional one may not).
		for (int i = 0; i < j->codeCheckCount; ++i) {
			j->codeChecks[i].site = farSite(e, j->codeChecks[i].site);
		}
		for (int i = 0; i < j->rdpWaitCount; ++i) {
			j->rdpWaits[i].site = farSite(e, j->rdpWaits[i].site);
		}
		for (int i = 0; i < j->missCount; ++i) {
			j->misses[i].site = farSite(e, j->misses[i].site);
		}
		for (int i = 0; i < j->mappedCount; ++i) {
			j->mapped[i].site = farSite(e, j->mapped[i].site);
		}
		for (int i = 0; i < j->slowCount; ++i) {
			for (int k = 0; k < j->slows[i].sites; ++k) {
				j->slows[i].site[k] = farSite(e, j->slows[i].site[k]);
			}
		}
		for (int i = 0; i < j->exitCount; ++i) {
			if (!hotExit(&j->exits[i])) {
				j->exits[i].site = farSite(e, j->exits[i].site);
			}
		}
		hot = j->e;
		j->e = cold;
		j->leave = coldLeave;
		j->cold = NULL;
		// Stores to a page with code only: on unless they hit a word a block holds (n64HoldsCode), its
		// bit in codeWords. An aligned store touches one word, or two for SD.
		for (int i = 0; i < j->codeCheckCount; ++i) {
			struct Miss* k = &j->codeChecks[i];
			patchHere(e, k->site);
			flushDirty(e, &k->cache);
			struct Cache clean = k->cache;
			clean.dirty = 0;
			// Past RDRAM, the interpreter. On a page the RDP draws to or reads, it waits for the RDP
			// there in C first.
			aLsrImm(e, 2, 6, 23);
			slowSite(k->slow, aCbnz(e, 2, NULL), &clean);
			aLogicImm(e, A_ANDS_IMM, ZR, 1, PAGE_RDP);
			uint8_t* noRdp = aBcond(e, C_EQ, NULL);
			rdpWaitCall(e, (const void*) cpuRdpWaitWrite, k->size);
			reloadPool(e, &k->cache);
			aLsrImm(e, 1, 6, 12);
			aLdst(e, L_LDRB, 1, PAGES, 1, L_UXTW);
			patchHere(e, noRdp);
			uint8_t* coded = aTbnz(e, 1, 0, NULL);
			aB(e, k->resume);
			patchHere(e, coded);
			aAddImmX(e, IP, N, N64_OFF(codeWords));
			aLsrImm(e, 2, 6, 5);
			aLdst(e, L_LDRB, 2, IP, 2, L_UXTW);
			aUbfx(e, 3, 6, 2, 3);
			aDp(e, A_LSRV, 2, 2, 3, S_LSL, 0);
			aLogicImm(e, A_ANDS_IMM, ZR, 2, k->size == 8 ? 3 : 1);
			slowSite(k->slow, aBcond(e, C_NE, NULL), &clean);
			aB(e, k->resume);
		}

		// With HLE accuracy, a page the TLB maps through vpages (one not looked up lands past RDRAM), and
		// an aligned address; else the interpreter.
		for (int i = 0; i < j->mappedCount; ++i) {
			struct Miss* v = &j->mapped[i];
			patchHere(e, v->site);
			aLsrImm(e, 1, 0, 12);
			if (v->store) {
				aDpImm(e, A_ADD_IMM | W64, 3, TLB, VPAGES * 4);
				aLdst(e, L_LDR, 5, 3, 1, L_UXTW | 0x1000);
			} else {
				aLdst(e, L_LDR, 5, TLB, 1, L_UXTW | 0x1000);
			}
			aDp(e, A_ADD, 6, 0, 5, S_LSL, 0);
			aLsrImm(e, 2, 6, 23);
			slowSite(v->slow, aCbnz(e, 2, NULL), &v->cache);
			if (v->size > 1) {
				aLogicImm(e, A_ANDS_IMM, ZR, 0, v->size - 1);
				slowSite(v->slow, aBcond(e, C_NE, NULL), &v->cache);
			}
			aB(e, v->resume);
		}

		// Loads from pages the RDP may still be drawing to wait for it in C, then go on inline: the same
		// path (and time) whichever renderer draws and however long it keeps the pages.
		for (int i = 0; i < j->rdpWaitCount; ++i) {
			struct Miss* w = &j->rdpWaits[i];
			patchHere(e, w->site);
			flushDirty(e, &w->cache);
			rdpWaitCall(e, (const void*) cpuRdpWaitRead, w->size);
			reloadPool(e, &w->cache);
			aB(e, w->resume);
		}

		// D-cache misses, filled (and a dirty line written back) in C. A misaligned address, one past
		// RDRAM, or a write-back over code, takes the slow path instead.
		for (int i = 0; i < j->missCount; ++i) {
			struct Miss* m = &j->misses[i];
			patchHere(e, m->site);
			flushDirty(e, &m->cache);
			struct Cache clean = m->cache;
			clean.dirty = 0;
			if (m->size > 1) {
				aLogicImm(e, A_ANDS_IMM, ZR, 0, m->size - 1);
				slowSite(m->slow, aBcond(e, C_NE, NULL), &clean);
			}
			aLsrImm(e, 2, 6, 23);
			slowSite(m->slow, aCbnz(e, 2, NULL), &clean);
			if (m->glLoad) {
				// A line coming in from a page the GPU draws to (a game invalidates its cache before
				// reading what the RDP drew) waits for the GPU first.
				aLsrImm(e, 3, 6, 12);
				aLdst(e, L_LDRB, 3, PAGES, 3, L_UXTW);
				uint8_t* ready = aTbz(e, 3, 1, NULL);
				rdpWaitCall(e, (const void*) cpuRdpWaitRead, 16);
				patchHere(e, ready);
			}
			aPush(e, 0, 6, 16);
			aMov(e, 1, 0);
			aMov(e, 2, 6);
			aMovX(e, 0, N);
			aDp(e, A_SUB | W64, 3, LIMIT, LEFT, S_LSL, 0);
			aMovImm(e, 4, (m->slow->op >> 26) >= 0x28);
			aMovImm64(e, IP, (uint64_t) (uintptr_t) cpuDcacheMiss);
			aBlr(e, IP);
			aMov(e, 4, 0);
			aPop(e, 0, 6, 16);
			aDpImm(e, A_ADDS_IMM, ZR, 4, 1);
			slowSite(m->slow, aBcond(e, C_EQ, NULL), &clean);
			aDp(e, A_SUB | W64, LEFT, LEFT, 4, S_LSL, 0);
			reloadPool(e, &m->cache);
			aUbfx(e, 1, 0, 4, 9);
			aB(e, m->resume);
		}
		for (int i = 0; i < j->slowCount; ++i) {
			struct Slow* sl = &j->slows[i];
			// Sites with new register values store them first.
			uint8_t* stored[8];
			int storedCount = 0;
			for (int k = 0; k < sl->sites; ++k) {
				if (sl->siteCache[k].dirty) {
					patchHere(e, sl->site[k]);
					flushDirty(e, &sl->siteCache[k]);
					stored[storedCount++] = aB(e, NULL);
				}
			}
			for (int k = 0; k < sl->sites; ++k) {
				if (!sl->siteCache[k].dirty) {
					patchHere(e, sl->site[k]);
				}
			}
			for (int k = 0; k < storedCount; ++k) {
				patchHere(e, stored[k]);
			}
			// Everything is in memory now: nothing for the helper's I-cache check to write back from
			// registers the rest of the block assigned.
			cacheClear(&j->cache);
			emitHelper(j, sl->op, sl->pc, sl->paddr, sl->delay, sl->lineStart, sl->last, sl->prevMark, sl->nextStall);
			if (!sl->last) {
				// The registers the fast path holds from here, as the interpreter left them.
				for (int k = 0; k < POOL; ++k) {
					if (sl->cache.guest[k] >= 0) {
						aLdrX(e, pool[k], CPU, gprOff(sl->cache.guest[k]));
					}
				}
				aB(e, sl->resume);
			}
		}
	}
	for (int i = 0; i < j->exitCount; ++i) {
		struct Exit* x = &j->exits[i];
		if (hotExit(x) != (pass == 1)) {
			continue;
		}
		patchHere(e, x->site);
		flushDirty(e, &x->cache);
		if (!j->fle) {
			if (x->stall < 0) {
				loadStallCheck(e, ops[((uint32_t) x->pc - vaddr) >> 2]);
			} else if (x->stall) {
				aDpImm(e, A_SUB_IMM | W64, LEFT, LEFT, x->stall);
			}
			storeByte(e, CPU_OFF(loadMark), x->mark);
		}
		if (loopsBack && x->last && x->delay && !x->skip) {
			// A loop going round again with cycles left: straight back to the start, before the
			// state for leaving, which the next pass writes anew (a poll loop skipping what passes
			// it can first).
			aLdrX(e, 0, CPU, CPU_OFF(nextPc));
			movConst(e, 1, sext32(vaddr));
			aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
			uint8_t* away = aBcond(e, C_NE, NULL);
			aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
			uint8_t* spent = aBcond(e, C_LE, NULL);
			if (pollPass) {
				aLdrb(e, 0, CPU, CPU_OFF(pollTried));
				uint8_t* tried = aCbnz(e, 0, NULL);
				storeByte(e, CPU_OFF(pollTried), 1);
				pollSkip(j, ops, count, vaddr, paddr, pollPass * (uint32_t) j->opCycles, reload, false);
				patchHere(e, tried);
			}
			aB(e, again);
			patchHere(e, away);
			patchHere(e, spent);
		}
		bool moved = false;
		// With HLE accuracy, on to the next block with only pc (and the load mark) written: it writes the
		// rest before anything reads it. The full state below where it goes back to C instead.
		if (j->fast && !x->jumped && !x->skip && (x->side || (x->last && x->kind == PLAIN))) {
			struct Sites full = { .count = 0 };
			aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
			full.at[full.count++] = aBcond(e, C_LE, NULL);
			// A block's one end (after a jump or no branch at all) is where its last exit goes.
			bool known = x->side || (endCount == 1 && !pollPass);
			if (x->side) {
				movConst(e, 0, x->to);
			} else if (known) {
				movConst(e, 0, ends[0]);
			} else if (x->delay) {
				aLdrX(e, 0, CPU, CPU_OFF(nextPc));
			} else {
				movConst(e, 0, x->pc + 4);
			}
			aStrX(e, 0, CPU, CPU_OFF(pc));
			if (x->side) {
				linkTo(j, x->to, &full, true);
			} else {
				if (pollPass) {
					pollSkip(j, ops, count, vaddr, paddr, pollPass * (uint32_t) j->opCycles, reload, true);
				}
				for (int k = 0; k < endCount; ++k) {
					linkTo(j, ends[k], &full, known);
				}
			}
			// A known kseg0 target is always linked to; mapped ones go on here when their page moved.
			if (!known || ((uint32_t) (x->side ? x->to : ends[0]) & 0xC0000000u) != 0x80000000u) {
				chain(j, &full);
			}
			// The full state is cold.
			for (int k = 0; k < full.count; ++k) {
				patchHere(e, full.at[k]);
			}
			uint8_t* far = aB(e, NULL);
			hot = j->e;
			j->e = cold;
			j->leave = coldLeave;
			j->cold = NULL;
			patchHere(e, far);
			moved = true;
		}
		storeConst(e, CPU_OFF(curPc), x->pc);
		storeByte(e, CPU_OFF(inDelay), x->delay);
		storeByte(e, CPU_OFF(nextInDelay), x->kind != PLAIN && !x->skip);
		if (x->jumped) {
			aB(e, j->leave);
			continue;
		}
		if (x->skip) {
			aDpImm(e, A_SUB_IMM | W64, LEFT, LEFT, 2 * j->opCycles);
			storeConst(e, CPU_OFF(pc), x->pc + 8);
			storeConst(e, CPU_OFF(nextPc), x->pc + 12);
		} else if (x->delay) {
			delayPrologue(e);
		} else {
			storeConst(e, CPU_OFF(pc), x->pc + 4);
			if (x->kind == PLAIN) {
				storeConst(e, CPU_OFF(nextPc), x->pc + 8);
			}
		}
		if (x->side) {
			aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
			aBcond(e, C_LE, j->leave);
			linkTo(j, x->to, NULL, false);
			chain(j, NULL);
		} else if (x->last && !(x->kind != PLAIN && !x->skip)) {
			aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
			aBcond(e, C_LE, j->leave);
			if (pollPass) {
				pollSkip(j, ops, count, vaddr, paddr, pollPass * (uint32_t) j->opCycles, reload, true);
			}
			for (int k = 0; k < endCount; ++k) {
				linkTo(j, ends[k], NULL, false);
			}
			chain(j, NULL);
		} else if (x->skip) {
			aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
			aBcond(e, C_LE, j->leave);
			linkTo(j, x->pc + 8, NULL, false);
			chain(j, NULL);
		} else {
			aB(e, j->leave);
		}
		if (moved) {
			cold = j->e;
			j->e = hot;
			j->leave = hotLeave;
			j->cold = &cold;
		}
	}
	}

	cold = j->e;
	j->e = hot;
	j->cold = NULL;
	if (e->p > e->end || cold.p > cold.end) {
		return NULL;
	}
	__builtin___clear_cache((char*) start, (char*) e->p);
	__builtin___clear_cache((char*) coldStart, (char*) cold.p);
	n->jitUsed += (size_t) (e->p - start + 15) & ~(size_t) 15;
	*coldUsed(n) += (size_t) (cold.p - coldStart + 15) & ~(size_t) 15;
	return entry;
}
