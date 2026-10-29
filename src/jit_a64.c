#include "n64.h"
#include "a64/emit.h"

#include <stddef.h>
#include <sys/mman.h>

// AArch64 code for a block of cached RDRAM instructions, as jit_x64.c does it: the ALU, branches and
// RDRAM loads and stores inline, the rest through cpuJitStep, stopping after any instruction that
// takes the cycle count to the limit.
//
// Registers: x19 = n, x20 = &n->cpu, x21 = n->cycles, x22 = the limit, x23 = RDRAM, x24 = jitTable,
// x25 = tlbCache, x26 = pageFlags, x27 = cpu.dcacheTag, x28 = cpu.dcacheDirty. Each block starts with
// a branch to the way out, for the conditional branches that can't reach it.

enum { ARENA_SIZE = 32 << 20, BLOCK_CODE_MAX = 32 << 10, SHARED_SIZE = 256 };

enum { N = 19, CPU = 20, CYCLES = 21, LIMIT = 22, RAM = 23, TABLE = 24, TLB = 25, CODE = 26, DTAG = 27, DIRTY = 28, IP = 16 };

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
	movConst(e, 9, v);
	aStrX(e, 9, CPU, cpuOff);
}

static void storeByte(struct Emitter* e, unsigned cpuOff, unsigned v) {
	if (v) {
		aMovImm(e, 9, v);
	}
	aStrb(e, v ? 9 : ZR, CPU, cpuOff);
}

static void loadGpr(struct Emitter* e, int rt, int i) {
	aLdrX(e, rt, CPU, gprOff(i));
}

static void loadGpr32(struct Emitter* e, int rt, int i) {
	aLdr(e, rt, CPU, gprOff(i));
}

static void storeGpr(struct Emitter* e, int i, int rt) {
	if (i) {
		aStrX(e, rt, CPU, gprOff(i));
	}
}

static void storeGprConst(struct Emitter* e, int i, uint64_t v) {
	if (i) {
		storeConst(e, gprOff(i), v);
	}
}

enum Kind { PLAIN, BRANCH, LIKELY };

struct Exit {
	uint8_t* site;
	uint64_t pc;
	enum Kind kind;
	bool delay;
	bool skip;
	bool last;
};

struct Slow {
	uint8_t* site[8];
	int sites;
	uint8_t* resume;
	uint32_t op, paddr;
	uint64_t pc;
	bool delay, lineStart, last;
};

struct Miss {
	uint8_t* site;
	uint8_t* resume;
};

struct Jit {
	struct Emitter e;
	// The instruction being emitted: where its slow path (the interpreter) is entered from, the
	// cycles it holds the pipeline for, and whether it wrote Status (w2 is then nonzero if that
	// leaves an interrupt due).
	struct Slow* slow;
	int extraCycles;
	bool interruptCheck;
	struct Exit exits[3 * 64];
	int exitCount;
	struct Slow slows[64];
	int slowCount;
	struct Miss misses[64];
	int missCount;
	uint8_t* leave;
	bool rdpThreads;
};

static void exitAt(struct Jit* j, uint8_t* site, uint64_t pc, enum Kind kind, bool delay, bool skip) {
	j->exits[j->exitCount++] = (struct Exit) { site, pc, kind, delay, skip, false };
}

// Sets the flags for a branch condition and returns the condition under which it is taken.
static int branchCondition(struct Emitter* e, uint32_t op, int rs, int rt) {
	loadGpr(e, 0, rs);
	switch (op >> 26) {
	case 0x04: case 0x14:
	case 0x05: case 0x15:
		loadGpr(e, 1, rt);
		aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
		return (op >> 26) & 1 ? C_NE : C_EQ;
	case 0x06: case 0x16:
		aDpImm(e, A_SUBS_IMM | W64, ZR, 0, 0);
		return C_LE;
	case 0x07: case 0x17:
		aDpImm(e, A_SUBS_IMM | W64, ZR, 0, 0);
		return C_GT;
	}
	aDpImm(e, A_SUBS_IMM | W64, ZR, 0, 0);
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

// MFC0 and DMFC0 of anything but Random; MTC0 and DMTC0 of Status, EPC and ErrorEPC.
static bool emitCop0(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, rd = (op >> 11) & 31;
	unsigned reg = CPU_OFF(cop0) + 8 * rd;
	if (rs == 0x00 || rs == 0x01) {
		if (rd == CP0_RANDOM) {
			return false;
		}
		if (rt) {
			if (rd == CP0_COUNT) {
				// Count: half the cycles before this instruction, from countBase.
				aLsrImmX(e, 0, CYCLES, 1);
				aLdrX(e, 1, CPU, CPU_OFF(countBase));
				aDp(e, A_SUB | W64, 0, 0, 1, S_LSL, 0);
				if (rs == 0x00) {
					aSxtw(e, 0, 0);
				} else {
					aMov(e, 0, 0);
				}
			} else {
				aLdrX(e, 0, CPU, reg);
				if (rs == 0x00) {
					aSxtw(e, 0, 0);
				}
			}
			storeGpr(e, rt, 0);
		}
		return true;
	}
	if (rs != 0x04 && rs != 0x05) {
		return false;
	}
	if (rd == CP0_STATUS) {
		loadGpr32(e, 0, rt);
		andConst(e, 0, 0, 0xFFF7FFFF);
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
		loadGpr(e, 0, rt);
		if (rs == 0x04) {
			aSxtw(e, 0, 0);
		}
		aStrX(e, 0, CPU, reg);
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
				if (f == 0x03) {
					loadGpr(e, 0, rt);
					aAsrImmX(e, 0, 0, sa);
				} else {
					loadGpr32(e, 0, rt);
					if (sa) {
						f ? aLsrImm(e, 0, 0, sa) : aLslImm(e, 0, 0, sa);
					}
				}
				aSxtw(e, 0, 0);
				storeGpr(e, rd, 0);
			}
			return true;
		case 0x04: case 0x06: case 0x07:
			if (rd) {
				loadGpr32(e, 1, rs);
				if (f == 0x07) {
					aAndBits(e, 1, 1, 0, 5);
					loadGpr(e, 0, rt);
					aDp(e, A_ASRV | W64, 0, 0, 1, S_LSL, 0);
				} else {
					loadGpr32(e, 0, rt);
					aDp(e, f == 0x04 ? A_LSLV : A_LSRV, 0, 0, 1, S_LSL, 0);
				}
				aSxtw(e, 0, 0);
				storeGpr(e, rd, 0);
			}
			return true;
		case 0x08: case 0x09:
			loadGpr(e, 0, rs);
			aStrX(e, 0, CPU, CPU_OFF(nextPc));
			if (f == 0x09) {
				storeGprConst(e, rd, pc + 8);
			}
			*kind = BRANCH;
			return true;
		case 0x0F:
			return true;
		case 0x10: case 0x12:
			if (rd) {
				aLdrX(e, 0, CPU, f == 0x10 ? CPU_OFF(hi) : CPU_OFF(lo));
				storeGpr(e, rd, 0);
			}
			return true;
		case 0x11: case 0x13:
			loadGpr(e, 0, rs);
			aStrX(e, 0, CPU, f == 0x11 ? CPU_OFF(hi) : CPU_OFF(lo));
			return true;
		case 0x14: case 0x16: case 0x17:
			if (rd) {
				loadGpr(e, 1, rs);
				loadGpr(e, 0, rt);
				aDp(e, (f == 0x14 ? A_LSLV : f == 0x16 ? A_LSRV : A_ASRV) | W64, 0, 0, 1, S_LSL, 0);
				storeGpr(e, rd, 0);
			}
			return true;
		case 0x18: case 0x19:
			// The products' low and high words, sign-extended; the multiplier holds the pipeline.
			loadGpr32(e, 0, rs);
			loadGpr32(e, 1, rt);
			if (f == 0x18) {
				aSmull(e, 0, 0, 1);
			} else {
				aUmull(e, 0, 0, 1);
			}
			aSxtw(e, 1, 0);
			aStrX(e, 1, CPU, CPU_OFF(lo));
			aAsrImmX(e, 0, 0, 32);
			aSxtw(e, 0, 0);
			aStrX(e, 0, CPU, CPU_OFF(hi));
			j->extraCycles = 3;
			return true;
		case 0x20: case 0x22: case 0x2C: case 0x2E: {
			// Overflow raises an exception, through the interpreter, before rd changes.
			uint32_t wide = f >= 0x2C ? W64 : 0;
			loadGpr(e, 0, rs);
			loadGpr(e, 1, rt);
			aDp(e, ((f & 2) ? A_SUBS : A_ADDS) | wide, 0, 0, 1, S_LSL, 0);
			slowIf(j, C_VS);
			if (!wide) {
				aSxtw(e, 0, 0);
			}
			storeGpr(e, rd, 0);
			return true;
		}
		case 0x21: case 0x23:
			if (rd) {
				loadGpr32(e, 0, rs);
				loadGpr32(e, 1, rt);
				aDp(e, f == 0x21 ? A_ADD : A_SUB, 0, 0, 1, S_LSL, 0);
				aSxtw(e, 0, 0);
				storeGpr(e, rd, 0);
			}
			return true;
		case 0x24: case 0x25: case 0x26: case 0x27: case 0x2D: case 0x2F: {
			static const uint32_t alu[] = { A_AND, A_ORR, A_EOR, A_ORR };
			if (rd) {
				loadGpr(e, 0, rs);
				loadGpr(e, 1, rt);
				aDp(e, (f == 0x2D ? A_ADD : f == 0x2F ? A_SUB : alu[f - 0x24]) | W64, 0, 0, 1, S_LSL, 0);
				if (f == 0x27) {
					aDp(e, A_ORN | W64, 0, ZR, 0, S_LSL, 0);
				}
				storeGpr(e, rd, 0);
			}
			return true;
		}
		case 0x2A: case 0x2B:
			if (rd) {
				loadGpr(e, 0, rs);
				loadGpr(e, 1, rt);
				aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
				aCset(e, 0, f == 0x2A ? C_LT : C_LO);
				storeGpr(e, rd, 0);
			}
			return true;
		case 0x38: case 0x3A: case 0x3B: case 0x3C: case 0x3E: case 0x3F:
			if (rd) {
				int amount = sa + (f >= 0x3C ? 32 : 0);
				loadGpr(e, 0, rt);
				if (amount) {
					if ((f & 3) == 0) {
						aLslImmX(e, 0, 0, amount);
					} else if ((f & 3) == 2) {
						aLsrImmX(e, 0, 0, amount);
					} else {
						aAsrImmX(e, 0, 0, amount);
					}
				}
				storeGpr(e, rd, 0);
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
			storeGprConst(e, 31, pc + 8);
		}
		*kind = BRANCH;
		return true;
	case 0x04: case 0x05: case 0x06: case 0x07:
	case 0x14: case 0x15: case 0x16: case 0x17:
		break;
	case 0x08: case 0x18: {
		uint32_t wide = op >> 26 == 0x18 ? W64 : 0;
		loadGpr(e, 0, rs);
		movConst(e, 1, (uint64_t) (int64_t) imm);
		aDp(e, A_ADDS | wide, 0, 0, 1, S_LSL, 0);
		slowIf(j, C_VS);
		if (!wide) {
			aSxtw(e, 0, 0);
		}
		storeGpr(e, rt, 0);
		return true;
	}
	case 0x09:
		if (rt) {
			loadGpr32(e, 0, rs);
			aAddImm(e, 0, 0, imm, 1);
			aSxtw(e, 0, 0);
			storeGpr(e, rt, 0);
		}
		return true;
	case 0x0A: case 0x0B:
		if (rt) {
			loadGpr(e, 0, rs);
			movConst(e, 1, (uint64_t) (int64_t) imm);
			aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
			aCset(e, 0, op >> 26 == 0x0A ? C_LT : C_LO);
			storeGpr(e, rt, 0);
		}
		return true;
	case 0x0C: case 0x0D: case 0x0E:
		if (rt) {
			static const uint32_t alu[] = { A_AND, A_ORR, A_EOR };
			loadGpr(e, 0, rs);
			aMovImm(e, 1, (uint16_t) op);
			aDp(e, alu[(op >> 26) - 0x0C] | W64, 0, 0, 1, S_LSL, 0);
			storeGpr(e, rt, 0);
		}
		return true;
	case 0x0F:
		storeGprConst(e, rt, sext32((uint32_t) (uint16_t) op << 16));
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
		// The FPU's own code, called directly: it raises nothing once CU1 is on. Its stall counts.
		needCop1(j);
		aMovX(e, 0, N);
		aMovImm(e, 1, op);
		aMovImm64(e, IP, (uint64_t) (uintptr_t) fpuExecute);
		aBlr(e, IP);
		aLdr(e, 0, CPU, CPU_OFF(stall));
		aDp(e, A_ADD | W64, CYCLES, CYCLES, 0, S_LSL, 0);
		aStr(e, ZR, CPU, CPU_OFF(stall));
		// A move to r0 is undone, as the interpreter zeroes r0 after every instruction.
		if (((op >> 21) & 31) <= 2 && !rt) {
			aStrX(e, ZR, CPU, gprOff(0));
		}
		return true;
	case 0x20: case 0x21: case 0x23: case 0x24: case 0x25: case 0x27: case 0x37:
	case 0x28: case 0x29: case 0x2B: case 0x3F:
	case 0x31: case 0x35: case 0x39: case 0x3D:
		emitMemory(j, j->slow);
		return true;
	case 0x19:
		if (rt) {
			loadGpr(e, 0, rs);
			movConst(e, 1, (uint64_t) (int64_t) imm);
			aDp(e, A_ADD | W64, 0, 0, 1, S_LSL, 0);
			storeGpr(e, rt, 0);
		}
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
		cc = branchCondition(e, op, rs, rt);
	}
	if ((op >> 26) == 0x01 && (rt & 0x10)) {
		storeGprConst(e, 31, pc + 8);
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
		aDpImm(e, A_ADD_IMM, 0, 0, 48);
		aStr(e, 0, CPU, CPU_OFF(stall));
	} else {
		aDpImm(e, A_ADD_IMM | W64, CYCLES, CYCLES, 48);
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
	aDp(e, A_SUBS | W64, 0, LIMIT, CYCLES, S_LSL, 0);
	aDpImm(e, A_SUBS_IMM | W64, ZR, 0, 1);
	skip[sites++] = aBcond(e, C_LE, NULL);
	aDpImm(e, A_SUB_IMM | W64, 0, 0, 1);
	aLsrImmX(e, 0, 0, 1);
	aDp(e, A_ADD | W64, CYCLES, CYCLES, 0, S_LSL, 1);
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
	storeN(e, CYCLES, N64_OFF(cycles), 8);
	aMovX(e, 0, N);
	aMovImm(e, 1, op);
	aMovImm64(e, IP, (uint64_t) (uintptr_t) cpuJitStep);
	aBlr(e, IP);
	loadN(e, CYCLES, N64_OFF(cycles), 8);
	loadN(e, LIMIT, N64_OFF(jitLimit), 8);
	aCbnz(e, 0, j->leave);
	aDp(e, A_SUBS | W64, ZR, CYCLES, LIMIT, S_LSL, 0);
	aBcond(e, C_HS, j->leave);
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
	aAddImmX(e, 6, CPU, CPU_OFF(fpr) + 8 * (ft & ~1) + (size == 4 ? 4 : 0));
	aLdr(e, 7, CPU, CPU_OFF(cop0) + 8 * CP0_STATUS);
	aLogicImm(e, A_ANDS_IMM, ZR, 7, 1u << 26);
	aCselX(e, 5, 6, 5, C_EQ);
}

// A load or store to cached RDRAM, through kseg0 or a page in the TLB cache, with dataTiming's
// D-cache check; anything else jumps to the slow path before any change. w0 ends up the physical
// address, w8 the virtual one, w1 the D-cache line and w2 its tag.
static void emitMemory(struct Jit* j, struct Slow* slow) {
	struct Emitter* e = &j->e;
	uint32_t op = slow->op;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, size = memSize(op);
	int code = op >> 26;
	bool fpu = code == 0x31 || code == 0x35 || code == 0x39 || code == 0x3D;
	bool isStore = code >= 0x28 && code != 0x37 && code != 0x31 && code != 0x35;
	if (fpu) {
		needCop1(j);
	}

	loadGpr(e, 0, rs);
	if ((int16_t) op) {
		movConst(e, 1, (uint64_t) (int64_t) (int16_t) op);
		aDp(e, A_ADD | W64, 0, 0, 1, S_LSL, 0);
	}
	aSxtw(e, 1, 0);
	aDp(e, A_SUBS | W64, ZR, 1, 0, S_LSL, 0);
	slow->site[slow->sites++] = aBcond(e, C_NE, NULL);
	aMov(e, 8, 0);
	andConst(e, 1, 0, 0xFF800000u | (size - 1));
	aMovImm(e, 2, 0x80000000u);
	aDp(e, A_SUBS, ZR, 1, 2, S_LSL, 0);
	uint8_t* kseg0 = aBcond(e, C_EQ, NULL);

	// Mapped: the physical page from the TLB cache.
	if (size > 1) {
		aLogicImm(e, A_ANDS_IMM, ZR, 0, size - 1);
		slow->site[slow->sites++] = aBcond(e, C_NE, NULL);
	}
	aLsrImm(e, 1, 0, 12);
	aAndBits(e, 2, 1, 0, 8);
	aLogicImm(e, A_ORR_IMM, 1, 1, 0x80000000u);
	aDp(e, A_ADD | W64, 3, TLB, 2, S_LSL, 3);
	aLdpW(e, 4, 5, 3, 0);
	aDp(e, A_SUBS, ZR, 4, 1, S_LSL, 0);
	slow->site[slow->sites++] = aBcond(e, C_NE, NULL);
	if (isStore) {
		slow->site[slow->sites++] = aTbz(e, 5, 0, NULL);
	}
	andConst(e, 5, 5, ~0xFFFu);
	aAndBits(e, 0, 0, 0, 12);
	aDp(e, A_ORR, 0, 0, 5, S_LSL, 0);
	aDpImm(e, A_SUBS_IMM, ZR, 0, RDRAM_SIZE);
	slow->site[slow->sites++] = aBcond(e, C_HS, NULL);
	uint8_t* mapped = aB(e, NULL);

	patchHere(e, kseg0);
	aAndBits(e, 0, 0, 0, 23);
	patchHere(e, mapped);
	// Stores go slow on any flag (code, RDP), loads on pages the RDP may still be drawing to.
	if (isStore || j->rdpThreads) {
		aLsrImm(e, 1, 0, 12);
		aLdst(e, L_LDRB, 1, CODE, 1, L_UXTW);
		slow->site[slow->sites++] = isStore ? aCbnz(e, 1, NULL) : aTbnz(e, 1, 1, NULL);
	}

	// The line index from the virtual address, the tag from the physical one.
	aUbfx(e, 1, 8, 4, 9);
	aLsrImm(e, 2, 0, 4);
	aLogicImm(e, A_ORR_IMM, 2, 2, 0x80000000u);
	aLdst(e, L_LDR, 3, DTAG, 1, L_UXTW | 0x1000);
	aDp(e, A_SUBS, ZR, 3, 2, S_LSL, 0);
	struct Miss* m = &j->misses[j->missCount++];
	m->site = aBcond(e, C_NE, NULL);
	m->resume = e->p;
	if (isStore) {
		aMovImm(e, 3, 1);
		aLdst(e, L_STRB, 3, DIRTY, 1, L_UXTW);
	}

	switch (op >> 26) {
	case 0x20:
		aLdst(e, L_LDRB, 4, RAM, 0, L_UXTW);
		aSxtbX(e, 4, 4);
		break;
	case 0x24:
		aLdst(e, L_LDRB, 4, RAM, 0, L_UXTW);
		break;
	case 0x21: case 0x25:
		aLdst(e, L_LDRH, 4, RAM, 0, L_UXTW);
		aRev16(e, 4, 4);
		if (op >> 26 == 0x21) {
			aSxthX(e, 4, 4);
		}
		break;
	case 0x23: case 0x27:
		aLdst(e, L_LDR, 4, RAM, 0, L_UXTW);
		aRev(e, 4, 4);
		if (op >> 26 == 0x23) {
			aSxtw(e, 4, 4);
		}
		break;
	case 0x37:
		aLdst(e, L_LDRX, 4, RAM, 0, L_UXTW);
		aRevX(e, 4, 4);
		break;
	case 0x28:
		loadGpr32(e, 4, rt);
		aLdst(e, L_STRB, 4, RAM, 0, L_UXTW);
		break;
	case 0x29:
		loadGpr32(e, 4, rt);
		aRev16(e, 4, 4);
		aLdst(e, L_STRH, 4, RAM, 0, L_UXTW);
		break;
	case 0x2B:
		loadGpr32(e, 4, rt);
		aRev(e, 4, 4);
		aLdst(e, L_STR, 4, RAM, 0, L_UXTW);
		break;
	case 0x3F:
		loadGpr(e, 4, rt);
		aRevX(e, 4, 4);
		aLdst(e, L_STRX, 4, RAM, 0, L_UXTW);
		break;
	case 0x31: case 0x35:
		aLdst(e, size == 4 ? L_LDR : L_LDRX, 4, RAM, 0, L_UXTW);
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
		aLdst(e, size == 4 ? L_STR : L_STRX, 4, RAM, 0, L_UXTW);
		return;
	}
	if (!isStore) {
		storeGpr(e, rt, 4);
	}
}

// Goes on to the block at the pc in memory if the jump table has it, or leaves.
static void chain(struct Jit* j) {
	struct Emitter* e = &j->e;
	aLdrX(e, 0, CPU, CPU_OFF(pc));
	aSxtw(e, 1, 0);
	aDp(e, A_SUBS | W64, ZR, 1, 0, S_LSL, 0);
	aBcond(e, C_NE, j->leave);
	aUbfx(e, 1, 0, 2, 14);
	aDp(e, A_ADD | W64, 2, TABLE, 1, S_LSL, 4);
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
	loadN(&e, CYCLES, N64_OFF(cycles), 8);
	loadN(&e, LIMIT, N64_OFF(jitLimit), 8);
	loadN(&e, RAM, N64_OFF(rdram), 8);
	aAddImmX(&e, TABLE, N, N64_OFF(jitTable));
	aAddImmX(&e, TLB, N, N64_OFF(tlbCache));
	aAddImmX(&e, CODE, N, N64_OFF(pageFlags));
	aAddImmX(&e, DTAG, CPU, CPU_OFF(dcacheTag));
	aAddImmX(&e, DIRTY, CPU, CPU_OFF(dcacheDirty));
	aBr(&e, 1);
	n->jitLeave = e.p;
	storeN(&e, CYCLES, N64_OFF(cycles), 8);
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
	j->exitCount = j->slowCount = j->missCount = 0;
	j->rdpThreads = n->rdpThreads > 0;
	struct Emitter* e = &j->e;
	j->leave = e->p;
	aB(e, n->jitLeave);
	uint8_t* entry = e->p;

	if (blockIsIdleLoop(ops, count, vaddr)) {
		skipIdle(e, vaddr, paddr);
	}

	bool delay = false;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k], p = paddr + k * 4;
		uint64_t pc = sext32(vaddr + k * 4);
		bool last = k + 1 == count;
		bool lineStart = k == 0 || !(p & 31);
		enum Kind kind;
		uint8_t* mark = e->p;
		int exits = j->exitCount, misses = j->missCount;
		bool branch = isBranchOp(op);
		struct Slow* slow = &j->slows[j->slowCount];
		*slow = (struct Slow) { .op = op, .paddr = p, .pc = pc, .delay = delay, .lineStart = lineStart, .last = last };
		j->slow = slow;
		j->extraCycles = 0;
		j->interruptCheck = false;
		// A branch's I-cache check comes first (a likely branch not taken leaves inside it); the rest
		// check after, so slow paths and Count reads see the cycles before this fetch.
		if (lineStart && branch && !delay) {
			icacheCheck(e, (uint32_t) pc, p, false);
		}
		if (!(delay && branch) && emitInline(j, op, pc, &kind)) {
			if (lineStart && !branch) {
				icacheCheck(e, (uint32_t) pc, p, false);
			}
			aDpImm(e, A_ADD_IMM | W64, CYCLES, CYCLES, 1 + j->extraCycles);
			if (j->interruptCheck) {
				// An interrupt now due is taken before the next instruction.
				exitAt(j, aCbnz(e, 2, NULL), pc, kind, delay, false);
			}
			if (last) {
				exitAt(j, aB(e, NULL), pc, kind, delay, false);
				j->exits[j->exitCount - 1].last = true;
			} else {
				aDp(e, A_SUBS | W64, ZR, CYCLES, LIMIT, S_LSL, 0);
				exitAt(j, aBcond(e, C_HS, NULL), pc, kind, delay, false);
			}
			if (slow->sites) {
				slow->resume = e->p;
				++j->slowCount;
			}
			delay = kind != PLAIN;
			continue;
		}
		e->p = mark;
		j->exitCount = exits;
		j->missCount = misses;
		emitHelper(j, op, pc, p, delay, lineStart, last);
		delay = branch;
	}

	// D-cache line fills, as dataTiming: the fill, the write-back of a dirty line, the new tag.
	for (int i = 0; i < j->missCount; ++i) {
		struct Miss* m = &j->misses[i];
		patchHere(e, m->site);
		aDpImm(e, A_ADD_IMM | W64, CYCLES, CYCLES, 40);
		aLdst(e, L_LDRB, 3, DIRTY, 1, L_UXTW);
		uint8_t* clean = aCbz(e, 3, NULL);
		aDpImm(e, A_ADD_IMM | W64, CYCLES, CYCLES, 40);
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
			aB(e, sl->resume);
		}
	}

	// Leaving after an inline instruction: the state the interpreter would have after it.
	for (int i = 0; i < j->exitCount; ++i) {
		struct Exit* x = &j->exits[i];
		patchHere(e, x->site);
		storeConst(e, CPU_OFF(curPc), x->pc);
		storeByte(e, CPU_OFF(inDelay), x->delay);
		storeByte(e, CPU_OFF(nextInDelay), x->kind != PLAIN && !x->skip);
		if (x->skip) {
			aDpImm(e, A_ADD_IMM | W64, CYCLES, CYCLES, 1);
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
			aDp(e, A_SUBS | W64, ZR, CYCLES, LIMIT, S_LSL, 0);
			aBcond(e, C_HS, j->leave);
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
