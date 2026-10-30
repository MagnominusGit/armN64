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
// x25 = vpages, x26 = pageFlags, x27 = cpu.dcacheTag, x28 = cpu.dcacheDirty; x24 and x9-x15 hold
// guest registers within a
// block (each written through to memory at once, so memory is always up to date); x0-x8, x16 and
// x17 are scratch. Each block starts with a branch to the way out, for the conditional branches that
// can't reach it.

enum { ARENA_SIZE = 64 << 20, BLOCK_CODE_MAX = 32 << 10, SHARED_SIZE = 256 };

enum { N = 19, CPU = 20, LEFT = 21, LIMIT = 22, RAM = 23, TLB = 25, PAGES = 26, DTAG = 27, DIRTY = 28, IP = 16, IP1 = 17 };

// The host registers that hold guest registers; the last seven don't survive calls.
enum { POOL = 8, POOL_SAVED = 1 };
static const uint8_t pool[POOL] = { 24, 9, 10, 11, 12, 13, 14, 15 };

// Which guest register each pool register holds, and when it was last used.
struct Cache {
	int8_t guest[POOL];
	int8_t host[32];
	uint32_t used[POOL];
	uint32_t clock;
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
};

struct Slow {
	uint8_t* site[8];
	int sites;
	uint8_t* resume;
	uint32_t op, paddr;
	uint64_t pc;
	bool delay, lineStart, last;
	// Kept even with no site yet: a D-cache miss path may add one.
	bool keep;
	// The guest registers held where the fast path resumes, to load again after the interpreter.
	struct Cache cache;
};

struct Miss {
	uint8_t* site;
	uint8_t* resume;
	// The access's slow path, for a misaligned address or one past RDRAM, and its size.
	struct Slow* slow;
	int size;
	// A load that, with the GL renderer, waits for the GPU's pages when the line comes in.
	bool glLoad;
};

struct Jit {
	struct Emitter e;
	// The instruction being emitted: where its slow path (the interpreter) is entered from, the
	// cycles it holds the pipeline for, and whether it wrote Status (w2 is then nonzero if that
	// leaves an interrupt due).
	struct Slow* slow;
	int extraCycles;
	bool interruptCheck;
	// The instruction set pc and nextPc itself (ERET): the block leaves after it.
	bool jumped;
	struct Exit exits[3 * 64];
	int exitCount;
	struct Slow slows[64];
	int slowCount;
	struct Miss misses[64];
	int missCount;
	// Stores to pages with compiled code, which check that they miss its words.
	struct Miss codeChecks[64];
	int codeCheckCount;
	uint8_t* leave;
	bool rdpThreads, rdpGl;
	// Status.FR as the block was compiled: with it clear, odd 32-bit FPU registers are the high
	// halves of even ones.
	bool fr;
	struct Cache cache;
};

static void cacheClear(struct Cache* c) {
	for (int i = 0; i < POOL; ++i) {
		c->guest[i] = -1;
	}
	for (int g = 0; g < 32; ++g) {
		c->host[g] = -1;
	}
}

static void forget(struct Cache* c, int g) {
	if (c->host[g] >= 0) {
		c->guest[c->host[g]] = -1;
		c->host[g] = -1;
	}
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
			c->host[c->guest[i]] = -1;
		}
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
	bool held = j->cache.host[g] >= 0;
	int host = pool[claim(j, g)];
	if (!held) {
		aLdrX(&j->e, host, CPU, gprOff(g));
	}
	return host;
}

// The host register to compute guest g's new value in (scratch x8 for r0, which keeps 0); commit()
// then writes it through.
static int writeReg(struct Jit* j, int g) {
	return g ? pool[claim(j, g)] : 8;
}

static void commit(struct Jit* j, int g, int host) {
	if (g) {
		aStrX(&j->e, host, CPU, gprOff(g));
	}
}

// Guest g = v, through the cache.
static void setConst(struct Jit* j, int g, uint64_t v) {
	if (g) {
		int d = writeReg(j, g);
		movConst(&j->e, d, v);
		commit(j, g, d);
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
	j->exits[j->exitCount++] = (struct Exit) { site, pc, kind, delay, skip, false, false };
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

// rd = rn & value, through a scratch register if AArch64 can't encode the constant.
static void andConst(struct Emitter* e, int rd, int rn, uint32_t value) {
	if (!aLogicImm(e, A_AND_IMM, rd, rn, value)) {
		aMovImm(e, 17, value);
		aDp(e, A_AND, rd, rn, 17, S_LSL, 0);
	}
}

static int memSize(uint32_t op);
static void emitMemory(struct Jit* j, struct Slow* slow);

// Leaves the instruction for the interpreter, before it has changed anything, when cond holds.
static void slowIf(struct Jit* j, int cond) {
	j->slow->site[j->slow->sites++] = aBcond(&j->e, cond, NULL);
}

// The interpreter raises "coprocessor unusable" when Status.CU1 is off.
static void needCop1(struct Jit* j) {
	aLdr(&j->e, 16, CPU, CPU_OFF(cop0) + 8 * CP0_STATUS);
	j->slow->site[j->slow->sites++] = aTbz(&j->e, 16, 29, NULL);
}

// Whether fpuExecute runs op without raising an exception (it only raises RI).
static bool fpuOpValid(uint32_t op) {
	int fmt = (op >> 21) & 31, func = op & 0x3F;
	if (fmt <= 6) {
		return fmt != 3;
	}
	if (fmt != 16 && fmt != 17 && fmt != 20 && fmt != 21) {
		return false;
	}
	return func <= 15 || func == 32 || func == 33 || func == 36 || func == 37 || func >= 0x30;
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

// The FPU instructions most code runs, inline for the Status.FR the block was compiled with.
static bool emitFpu(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int fmt = (op >> 21) & 31, func = op & 0x3F, ft = (op >> 16) & 31, fs = (op >> 11) & 31, fd = (op >> 6) & 31;
	bool single = fmt == 16 && (func <= 3 || (func >= 5 && func <= 7) || func == 13 || func >= 0x30);
	if (!(fmt == 0 || fmt == 2 || fmt == 4 || single || (fmt == 20 && func == 32))) {
		return false;
	}
	needCop1(j);
	j->slow->site[j->slow->sites++] = j->fr ? aTbz(e, 16, 26, NULL) : aTbnz(e, 16, 26, NULL);
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

// MFC0 and DMFC0 of anything but Random; MTC0 and DMTC0 of Status, EPC and ErrorEPC.
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
		j->jumped = true;
		return true;
	}
	if (rs == 0x00 || rs == 0x01) {
		if (rd == CP0_RANDOM) {
			return false;
		}
		if (rt) {
			int d = writeReg(j, rt);
			if (rd == CP0_COUNT) {
				// Count: half the cycles before this instruction, from countBase.
				aDp(e, A_SUB | W64, 0, LIMIT, LEFT, S_LSL, 0);
				aLsrImmX(e, 0, 0, 1);
				aLdrX(e, 1, CPU, CPU_OFF(countBase));
				aDp(e, A_SUB | W64, 0, 0, 1, S_LSL, 0);
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
	if (rs != 0x04 && rs != 0x05) {
		return false;
	}
	int t = readReg(j, rt);
	if (rd == CP0_STATUS) {
		andConst(e, 0, t, 0xFFF7FFFF);
		aStrX(e, 0, CPU, reg);
		// Interrupts on (IE, not EXL or ERL) and one of them pending.
		aAndBits(e, 1, 0, 0, 3);
		aDpImm(e, A_SUBS_IMM, ZR, 1, 1);
		aMovImm(e, 2, 0);
		uint8_t* off = aBcond(e, C_NE, NULL);
		aLdr(e, 2, CPU, CPU_OFF(cop0) + 8 * CP0_CAUSE);
		aDp(e, A_AND, 2, 2, 0, S_LSL, 0);
		aAndBits(e, 2, 2, 8, 8);
		patchHere(e, off);
		j->interruptCheck = true;
		return true;
	}
	if (rd == CP0_EPC || rd == CP0_ERROREPC) {
		if (rs == 0x04) {
			aSxtw(e, 0, t);
			aStrX(e, 0, CPU, reg);
		} else {
			aStrX(e, t, CPU, reg);
		}
		return true;
	}
	return false;
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
			// The products' low and high words, sign-extended; the multiplier holds the pipeline.
			int s = readReg(j, rs), t = readReg(j, rt);
			if (f == 0x18) {
				aSmull(e, 0, s, t);
			} else {
				aUmull(e, 0, s, t);
			}
			aSxtw(e, 1, 0);
			aStrX(e, 1, CPU, CPU_OFF(lo));
			aAsrImmX(e, 0, 0, 32);
			aSxtw(e, 0, 0);
			aStrX(e, 0, CPU, CPU_OFF(hi));
			j->extraCycles = 3;
			return true;
		}
		case 0x1A: case 0x1B: {
			// LO the quotient and HI the remainder, sign-extended (INT32_MIN / -1 wraps, as SDIV does).
			// By zero HI is the dividend and LO all ones, or 1 for a negative dividend in DIV.
			int s = readReg(j, rs), t = readReg(j, rt);
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
			movConst(e, 1, (uint64_t) (int64_t) imm);
			aDp(e, A_ADD | (op >> 26 == 0x19 ? W64 : 0), d, s, 1, S_LSL, 0);
			if (op >> 26 == 0x09) {
				aSxtw(e, d, d);
			}
			commit(j, rt, d);
		}
		return true;
	case 0x0A: case 0x0B:
		if (rt) {
			int s = readReg(j, rs), d = writeReg(j, rt);
			movConst(e, 1, (uint64_t) (int64_t) imm);
			aDp(e, A_SUBS | W64, ZR, s, 1, S_LSL, 0);
			aCset(e, d, op >> 26 == 0x0A ? C_LT : C_LO);
			commit(j, rt, d);
		}
		return true;
	case 0x0C: case 0x0D: case 0x0E:
		if (rt) {
			static const uint32_t alu[] = { A_AND, A_ORR, A_EOR };
			int s = readReg(j, rs), d = writeReg(j, rt);
			aMovImm(e, 1, (uint16_t) op);
			aDp(e, alu[(op >> 26) - 0x0C] | W64, d, s, 1, S_LSL, 0);
			commit(j, rt, d);
		}
		return true;
	case 0x0F:
		setConst(j, rt, sext32((uint32_t) (uint16_t) op << 16));
		return true;
	case 0x10:
		return emitCop0(j, op);
	case 0x11:
		if (((op >> 21) & 31) == 0x08) {
			break;
		}
		if (!fpuOpValid(op)) {
			return false;
		}
		if (emitFpu(j, op)) {
			return true;
		}
		// The FPU's own code, called directly: it raises nothing once CU1 is on. Its stall counts.
		needCop1(j);
		aMovX(e, 0, N);
		aMovImm(e, 1, op);
		aMovImm64(e, IP, (uint64_t) (uintptr_t) fpuExecute);
		aBlr(e, IP);
		forgetCallerSaved(&j->cache);
		aLdr(e, 0, CPU, CPU_OFF(stall));
		aDp(e, A_SUB | W64, LEFT, LEFT, 0, S_LSL, 0);
		aStr(e, ZR, CPU, CPU_OFF(stall));
		// Moves to a general register: it has a new value; r0 is undone, as the interpreter zeroes it.
		if (((op >> 21) & 31) <= 2) {
			forget(&j->cache, rt);
			if (!rt) {
				aStrX(e, ZR, CPU, gprOff(0));
			}
		}
		return true;
	case 0x2F: {
		// CACHE on the D-cache, as cacheOp: the line the address indexes is written back if dirty
		// (not for action 3) and emptied by actions 0, 4 and 5. I-cache ones leave the block through
		// the interpreter; the secondary cache isn't there.
		int which = (op >> 16) & 3, action = (op >> 18) & 7;
		if (which == 0) {
			return false;
		}
		if (which != 1) {
			return true;
		}
		int base = readReg(j, rs);
		if (base == ZR) {
			aMovImm(e, 0, (uint32_t) imm);
		} else {
			aAddImm(e, 0, base, imm, 1);
		}
		aUbfx(e, 1, 0, 4, 9);
		if (action != 3) {
			aLdst(e, L_LDRB, 3, DIRTY, 1, L_UXTW);
			uint8_t* clean = aCbz(e, 3, NULL);
			aLdst(e, L_STRB, ZR, DIRTY, 1, L_UXTW);
			aDpImm(e, A_SUB_IMM | W64, LEFT, LEFT, DCACHE_WRITEBACK);
			patchHere(e, clean);
		}
		if (action == 0 || action == 4 || action == 5) {
			aMovImm(e, 2, DCACHE_NONE);
			aLdst(e, L_STR, 2, DTAG, 1, L_UXTW | 0x1000);
		}
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
		// BC1F, BC1T and the likely ones: FCR31's condition bit.
		needCop1(j);
		aLdr(e, 0, CPU, CPU_OFF(fcr31));
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

// The I-cache line holding pc, as fetchTiming checks it.
static void icacheCheck(struct Emitter* e, uint32_t vaddr, uint32_t paddr, bool toStall) {
	unsigned tag = CPU_OFF(icacheTag) + 4 * ((vaddr >> 5) & 511);
	aLdr(e, 0, CPU, tag);
	aMovImm(e, 1, paddr >> 5 | 0x80000000u);
	aDp(e, A_SUBS, ZR, 0, 1, S_LSL, 0);
	uint8_t* hit = aBcond(e, C_EQ, NULL);
	aStr(e, 1, CPU, tag);
	if (toStall) {
		aLdr(e, 0, CPU, CPU_OFF(stall));
		aDpImm(e, A_ADD_IMM, 0, 0, ICACHE_FILL);
		aStr(e, 0, CPU, CPU_OFF(stall));
	} else {
		aDpImm(e, A_SUB_IMM | W64, LEFT, LEFT, ICACHE_FILL);
	}
	patchHere(e, hit);
}

// The idle loop at vaddr, when its I-cache lines are in: whole passes of two cycles up to one or two
// cycles short of the limit, leaving the last instructions to run as usual.
static void skipIdle(struct Emitter* e, uint32_t vaddr, uint32_t paddr) {
	uint8_t* skip[3];
	int sites = 0;
	int lines = (paddr + 4) & 31 ? 1 : 2;
	for (int i = 0; i < lines; ++i) {
		uint32_t v = vaddr + 4 * i, p = paddr + 4 * i;
		aLdr(e, 0, CPU, CPU_OFF(icacheTag) + 4 * ((v >> 5) & 511));
		aMovImm(e, 1, p >> 5 | 0x80000000u);
		aDp(e, A_SUBS, ZR, 0, 1, S_LSL, 0);
		skip[sites++] = aBcond(e, C_NE, NULL);
	}
	aDpImm(e, A_SUBS_IMM | W64, ZR, LEFT, 1);
	skip[sites++] = aBcond(e, C_LE, NULL);
	aDpImm(e, A_SUB_IMM | W64, 0, LEFT, 1);
	aLsrImmX(e, 0, 0, 1);
	aDp(e, A_SUB | W64, LEFT, LEFT, 0, S_LSL, 1);
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

static void chain(struct Jit* j);

// The instruction through cpuJitStep, which adds its cycles. Leaves if it asks to or at the limit;
// at the end of the block goes on to the next.
static void emitHelper(struct Jit* j, uint32_t op, uint64_t pc, uint32_t paddr, bool delay, bool lineStart, bool last) {
	struct Emitter* e = &j->e;
	if (lineStart) {
		icacheCheck(e, (uint32_t) pc, paddr, true);
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
	aMovX(e, 0, N);
	aMovImm(e, 1, op);
	aMovImm64(e, IP, (uint64_t) (uintptr_t) cpuJitStep);
	aBlr(e, IP);
	loadN(e, 1, N64_OFF(cycles), 8);
	loadN(e, LIMIT, N64_OFF(jitLimit), 8);
	aDp(e, A_SUB | W64, LEFT, LIMIT, 1, S_LSL, 0);
	aCbnz(e, 0, j->leave);
	aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
	aBcond(e, C_LE, j->leave);
	if (last) {
		aLdrb(e, 0, CPU, CPU_OFF(nextInDelay));
		aCbnz(e, 0, j->leave);
		chain(j);
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
static void emitMemory(struct Jit* j, struct Slow* slow) {
	struct Emitter* e = &j->e;
	uint32_t op = slow->op;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, size = memSize(op);
	int code = op >> 26;
	bool fpu = code == 0x31 || code == 0x35 || code == 0x39 || code == 0x3D;
	bool isStore = code >= 0x28 && code != 0x37 && code != 0x31 && code != 0x35;
	int32_t offset = (int16_t) op;
	if (fpu) {
		needCop1(j);
	}

	int base = readReg(j, rs);
	if (base == ZR) {
		aMovImm(e, 0, (uint32_t) offset);
	} else if (offset >= 0 && offset < 4096) {
		aDpImm(e, A_ADD_IMM, 0, base, (unsigned) offset);
	} else if (offset < 0 && offset > -4096) {
		aDpImm(e, A_SUB_IMM, 0, base, (unsigned) -offset);
	} else {
		aMovImm(e, 1, (uint32_t) offset);
		aDp(e, A_ADD, 0, base, 1, S_LSL, 0);
	}
	// The physical page from vpages: one not looked up lands past RDRAM.
	aLsrImm(e, 1, 0, 12);
	if (isStore) {
		aDpImm(e, A_ADD_IMM | W64, 3, TLB, VPAGES * 4);
		aLdst(e, L_LDR, 5, 3, 1, L_UXTW | 0x1000);
	} else {
		aLdst(e, L_LDR, 5, TLB, 1, L_UXTW | 0x1000);
	}
	aDp(e, A_ADD, 6, 0, 5, S_LSL, 0);
	// Stores go slow on any flag (code, RDP) unless the page has only code that they miss; loads
	// on pages the RDP may still be drawing to.
	if (isStore) {
		aLsrImm(e, 1, 6, 12);
		aLdst(e, L_LDRB, 1, PAGES, 1, L_UXTW);
		struct Miss* k = &j->codeChecks[j->codeCheckCount++];
		k->site = aCbnz(e, 1, NULL);
		k->resume = e->p;
		k->slow = slow;
		k->size = size;
		slow->keep = true;
	} else if (j->rdpThreads) {
		aLsrImm(e, 1, 6, 12);
		aLdst(e, L_LDRB, 1, PAGES, 1, L_UXTW);
		slow->site[slow->sites++] = aTbnz(e, 1, 1, NULL);
	}

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
	slow->keep = true;
	if (isStore) {
		aMovImm(e, 3, 1);
		aLdst(e, L_STRB, 3, DIRTY, 1, L_UXTW);
	}

	if (isStore && !fpu) {
		int t = readReg(j, rt);
		switch (op >> 26) {
		case 0x28:
			aLdst(e, L_STRB, t, RAM, 6, L_UXTW);
			return;
		case 0x29:
			aRev16(e, 4, t);
			aLdst(e, L_STRH, 4, RAM, 6, L_UXTW);
			return;
		case 0x2B:
			aRev(e, 4, t);
			aLdst(e, L_STR, 4, RAM, 6, L_UXTW);
			return;
		default:
			aRevX(e, 4, t);
			aLdst(e, L_STRX, 4, RAM, 6, L_UXTW);
			return;
		}
	}
	switch (op >> 26) {
	case 0x31: case 0x35:
		aLdst(e, size == 4 ? L_LDR : L_LDRX, 4, RAM, 6, L_UXTW);
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
		aLdst(e, size == 4 ? L_STR : L_STRX, 4, RAM, 6, L_UXTW);
		return;
	}
	int d = writeReg(j, rt);
	switch (op >> 26) {
	case 0x20:
		aLdst(e, L_LDRB, d, RAM, 6, L_UXTW);
		aSxtbX(e, d, d);
		break;
	case 0x24:
		aLdst(e, L_LDRB, d, RAM, 6, L_UXTW);
		break;
	case 0x21: case 0x25:
		aLdst(e, L_LDRH, d, RAM, 6, L_UXTW);
		aRev16(e, d, d);
		if (op >> 26 == 0x21) {
			aSxthX(e, d, d);
		}
		break;
	case 0x23: case 0x27:
		aLdst(e, L_LDR, d, RAM, 6, L_UXTW);
		aRev(e, d, d);
		if (op >> 26 == 0x23) {
			aSxtw(e, d, d);
		}
		break;
	default:
		aLdst(e, L_LDRX, d, RAM, 6, L_UXTW);
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
// no flags, and every D-cache line they span is in. Every access then hits and takes one cycle, so
// the run is its loads and stores in order and its cycles at once. Anything else takes the
// instructions one by one (emitted after this), from the start of the run. Returns the branch to
// the end of the run.
static uint8_t* fuseRun(struct Jit* j, const uint32_t* ops, uint32_t len, uint32_t vaddr, uint32_t paddr) {
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
	// The base first: the instructions one by one hold it from here whichever way the checks go.
	int b = readReg(j, rs);
	aMovImm(e, 1, len);
	aDp(e, A_SUBS | W64, ZR, LEFT, 1, S_LSL, 0);
	out[sites++] = aBcond(e, C_LE, NULL);
	for (uint32_t p = paddr & ~31u; p < paddr + 4 * len; p += 32) {
		uint32_t v = vaddr + (p - paddr);
		aLdr(e, 0, CPU, CPU_OFF(icacheTag) + 4 * ((v >> 5) & 511));
		aMovImm(e, 1, p >> 5 | 0x80000000u);
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
	aLdst(e, L_LDRB, 1, PAGES, 1, L_UXTW);
	if (stores) {
		out[sites++] = aCbnz(e, 1, NULL);
	} else {
		out[sites++] = aTbnz(e, 1, 7, NULL);
		if (j->rdpThreads) {
			out[sites++] = aTbnz(e, 1, 1, NULL);
		}
	}
	// The D-cache lines from the first address's to the last's.
	for (int32_t step = 0; step <= hi - lo + 15; step += 16) {
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
	aMov(e, 7, 6);
	if (lo - floor8) {
		aDpImm(e, A_SUB_IMM | W64, 7, 7, (unsigned) (lo - floor8));
	}
	aDp(e, A_ADD | W64, 7, RAM, 7, S_LSL, 0);
	if (stores) {
		aMovImm(e, 3, 1);
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
		// The line it wrote is dirty.
		aAddImm(e, 1, 0, (int16_t) op - lo, 2);
		aUbfx(e, 1, 1, 4, 9);
		aLdst(e, L_STRB, 3, DIRTY, 1, L_UXTW);
	}
	aMovImm(e, 1, len);
	aDp(e, A_SUB | W64, LEFT, LEFT, 1, S_LSL, 0);
	uint8_t* done = aB(e, NULL);
	for (int i = 0; i < sites; ++i) {
		patchHere(e, out[i]);
	}
	return done;
}

// A poll loop (blockPollPass) about to go round again, pc back at its start, skips the whole passes
// of pass cycles that fit before the limit when all its I-cache and D-cache lines are in: every pass
// then reads the same memory, takes the same branch and costs the same.
static void pollSkip(struct Jit* j, const uint32_t* ops, uint32_t count, uint32_t vaddr, uint32_t paddr, uint32_t pass) {
	struct Emitter* e = &j->e;
	uint8_t* out[64];
	int sites = 0;
	aLdrX(e, 0, CPU, CPU_OFF(pc));
	movConst(e, 1, sext32(vaddr));
	aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
	out[sites++] = aBcond(e, C_NE, NULL);
	for (uint32_t p = paddr & ~31u; p < paddr + 4 * count; p += 32) {
		uint32_t v = vaddr + (p - paddr);
		aLdr(e, 0, CPU, CPU_OFF(icacheTag) + 4 * ((v >> 5) & 511));
		aMovImm(e, 1, p >> 5 | 0x80000000u);
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

static void chain(struct Jit* j) {
	struct Emitter* e = &j->e;
	aLdrX(e, 0, CPU, CPU_OFF(pc));
	aSxtw(e, 1, 0);
	aDp(e, A_SUBS | W64, ZR, 1, 0, S_LSL, 0);
	aBcond(e, C_NE, j->leave);
	aUbfx(e, 1, 0, 2, 14);
	aAddImmX(e, 2, N, N64_OFF(jitTable));
	aDp(e, A_ADD | W64, 2, 2, 1, S_LSL, 4);
	aLdr(e, 3, 2, (unsigned) offsetof(struct JitEntry, vaddr));
	aDp(e, A_SUBS, ZR, 3, 0, S_LSL, 0);
	aBcond(e, C_NE, j->leave);
	aLdr(e, 3, 2, (unsigned) offsetof(struct JitEntry, gen));
	loadN(e, 4, N64_OFF(blockGen), 4);
	aDp(e, A_SUBS, ZR, 3, 4, S_LSL, 0);
	aBcond(e, C_NE, j->leave);
	aLdrX(e, 5, 2, (unsigned) offsetof(struct JitEntry, code));
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
	struct Emitter e = { p, p, (uint8_t*) p + SHARED_SIZE };
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
	aAddImmX(&e, DTAG, CPU, CPU_OFF(dcacheTag));
	aAddImmX(&e, DIRTY, CPU, CPU_OFF(dcacheDirty));
	aBr(&e, 1);
	n->jitLeave = e.p;
	aDp(&e, A_SUB | W64, 0, LIMIT, LEFT, S_LSL, 0);
	storeN(&e, 0, N64_OFF(cycles), 8);
	aLdp(&e, 19, 20, 16);
	aLdp(&e, 21, 22, 32);
	aLdp(&e, 23, 24, 48);
	aLdp(&e, 25, 26, 64);
	aLdp(&e, 27, 28, 80);
	aPop(&e, 29, 30, 96);
	aRet(&e);
	__builtin___clear_cache((char*) p, (char*) e.p);
	n->jitEnter = (void (*)(struct N64*, void*)) p;
	return p;
}

void jitReset(struct N64* n) {
	n->jitUsed = SHARED_SIZE;
	++n->blockGen;
}

void jitFree(struct N64* n) {
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
	if (!arena || n->jitUsed + BLOCK_CODE_MAX > ARENA_SIZE) {
		return NULL;
	}
	static struct Jit jit;
	struct Jit* j = &jit;
	uint8_t* start = arena + n->jitUsed;
	j->e = (struct Emitter) { start, start, start + BLOCK_CODE_MAX };
	j->exitCount = j->slowCount = j->missCount = j->codeCheckCount = 0;
	j->rdpThreads = n->rdpThreads > 0;
	j->rdpGl = n->rdpGl;
	j->fr = n->cpu.cop0[CP0_STATUS] & (1 << 26);
	struct Emitter* e = &j->e;
	j->leave = e->p;
	aB(e, n->jitLeave);
	uint8_t* entry = e->p;

	if (blockIsIdleLoop(ops, count, vaddr)) {
		skipIdle(e, vaddr, paddr);
	}
	uint32_t pollPass = blockPollPass(ops, count, vaddr);
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

	cacheClear(&j->cache);
	bool delay = false;
	// A fused run's end, where its fast path joins the instructions one by one.
	uint8_t* fusedDone = NULL;
	uint32_t fusedEnd = 0;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k], p = paddr + k * 4;
		if (fusedDone && k == fusedEnd) {
			patchHere(e, fusedDone);
			cacheClear(&j->cache);
			fusedDone = NULL;
		}
		uint32_t run = delay ? 0 : fusedRunLength(ops, k, count);
		if (run && !fusedDone) {
			fusedDone = fuseRun(j, ops + k, run, vaddr + 4 * k, p);
			fusedEnd = k + run;
		}
		uint64_t pc = sext32(vaddr + k * 4);
		bool last = k + 1 == count;
		bool lineStart = k == 0 || !(p & 31);
		enum Kind kind;
		uint8_t* mark = e->p;
		int exits = j->exitCount, misses = j->missCount, codeChecks = j->codeCheckCount;
		struct Cache cache = j->cache;
		bool branch = isBranchOp(op);
		struct Slow* slow = &j->slows[j->slowCount];
		*slow = (struct Slow) { .op = op, .paddr = p, .pc = pc, .delay = delay, .lineStart = lineStart, .last = last };
		j->slow = slow;
		j->extraCycles = 0;
		j->interruptCheck = false;
		j->jumped = false;
		// A branch's I-cache check comes first (a likely branch not taken leaves inside it); the rest
		// check after, so slow paths and Count reads see the cycles before this fetch.
		if (lineStart && branch && !delay) {
			icacheCheck(e, (uint32_t) pc, p, false);
		}
		if (!(delay && branch) && emitInline(j, op, pc, &kind)) {
			if (lineStart && !branch) {
				icacheCheck(e, (uint32_t) pc, p, false);
			}
			aDpImm(e, A_SUBS_IMM | W64, LEFT, LEFT, 1 + j->extraCycles);
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
			} else {
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
		emitHelper(j, op, pc, p, delay, lineStart, last);
		// The interpreter may have changed any register, and the call the scratch ones.
		cacheClear(&j->cache);
		delay = branch;
	}

	// Stores to a page with code only: on unless they hit a word a block holds (n64HoldsCode), its
	// bit in codeWords. An aligned store touches one word, or two for SD.
	for (int i = 0; i < j->codeCheckCount; ++i) {
		struct Miss* k = &j->codeChecks[i];
		patchHere(e, k->site);
		aDpImm(e, A_SUBS_IMM, ZR, 1, PAGE_CODE);
		k->slow->site[k->slow->sites++] = aBcond(e, C_NE, NULL);
		aAddImmX(e, IP, N, N64_OFF(codeWords));
		aLsrImm(e, 2, 6, 5);
		aLdst(e, L_LDRB, 2, IP, 2, L_UXTW);
		aUbfx(e, 3, 6, 2, 3);
		aDp(e, A_LSRV, 2, 2, 3, S_LSL, 0);
		aLogicImm(e, A_ANDS_IMM, ZR, 2, k->size == 8 ? 3 : 1);
		k->slow->site[k->slow->sites++] = aBcond(e, C_NE, NULL);
		aB(e, k->resume);
	}

	// D-cache line fills, as dataTiming: the fill, the write-back of a dirty line, the new tag. A
	// misaligned address, or one past RDRAM, takes the slow path instead.
	for (int i = 0; i < j->missCount; ++i) {
		struct Miss* m = &j->misses[i];
		patchHere(e, m->site);
		if (m->size > 1) {
			aLogicImm(e, A_ANDS_IMM, ZR, 0, m->size - 1);
			m->slow->site[m->slow->sites++] = aBcond(e, C_NE, NULL);
		}
		aLsrImm(e, 2, 6, 23);
		m->slow->site[m->slow->sites++] = aCbnz(e, 2, NULL);
		if (m->glLoad) {
			// A line coming in from a page the GPU draws to (a game invalidates its cache before
			// reading what the RDP drew): the C path waits for the GPU.
			aLsrImm(e, 3, 6, 12);
			aLdst(e, L_LDRB, 3, PAGES, 3, L_UXTW);
			m->slow->site[m->slow->sites++] = aTbnz(e, 3, 1, NULL);
		}
		aLogicImm(e, A_AND_IMM, 2, 6, ~0xFu);
		aDpImm(e, A_SUB_IMM | W64, LEFT, LEFT, DCACHE_FILL);
		aLdst(e, L_LDRB, 3, DIRTY, 1, L_UXTW);
		uint8_t* clean = aCbz(e, 3, NULL);
		aDpImm(e, A_SUB_IMM | W64, LEFT, LEFT, DCACHE_WRITEBACK);
		patchHere(e, clean);
		aLdst(e, L_STR, 2, DTAG, 1, L_UXTW | 0x1000);
		aLdst(e, L_STRB, ZR, DIRTY, 1, L_UXTW);
		aB(e, m->resume);
	}
	for (int i = 0; i < j->slowCount; ++i) {
		struct Slow* sl = &j->slows[i];
		for (int k = 0; k < sl->sites; ++k) {
			patchHere(e, sl->site[k]);
		}
		emitHelper(j, sl->op, sl->pc, sl->paddr, sl->delay, sl->lineStart, sl->last);
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

	// Leaving after an inline instruction: the state the interpreter would have after it.
	for (int i = 0; i < j->exitCount; ++i) {
		struct Exit* x = &j->exits[i];
		patchHere(e, x->site);
		if (loopsBack && x->last && x->delay && !x->skip) {
			// A loop going round again with cycles left: straight back to the start, before the
			// state for leaving, which the next pass writes anew.
			aLdrX(e, 0, CPU, CPU_OFF(nextPc));
			movConst(e, 1, sext32(vaddr));
			aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
			uint8_t* away = aBcond(e, C_NE, NULL);
			aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
			aBcond(e, C_GT, entry);
			patchHere(e, away);
		}
		storeConst(e, CPU_OFF(curPc), x->pc);
		storeByte(e, CPU_OFF(inDelay), x->delay);
		storeByte(e, CPU_OFF(nextInDelay), x->kind != PLAIN && !x->skip);
		if (x->jumped) {
			aB(e, j->leave);
			continue;
		}
		if (x->skip) {
			aDpImm(e, A_SUB_IMM | W64, LEFT, LEFT, 1);
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
		if (x->last && !(x->kind != PLAIN && !x->skip)) {
			aDp(e, A_SUBS | W64, ZR, LEFT, ZR, S_LSL, 0);
			aBcond(e, C_LE, j->leave);
			if (pollPass) {
				pollSkip(j, ops, count, vaddr, paddr, pollPass);
			}
			chain(j);
		} else {
			aB(e, j->leave);
		}
	}

	if (e->p > e->end) {
		return NULL;
	}
	__builtin___clear_cache((char*) start, (char*) e->p);
	n->jitUsed += (size_t) (e->p - start + 15) & ~(size_t) 15;
	return entry;
}
