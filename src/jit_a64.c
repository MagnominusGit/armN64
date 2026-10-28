#include "n64.h"
#include "a64/emit.h"

#include <stddef.h>
#include <sys/mman.h>

// AArch64 code for a block of cached RDRAM instructions, as jit_x64.c does it: the ALU, branches and
// RDRAM loads and stores inline, the rest through cpuJitStep, stopping after any instruction that
// takes the cycle count to the limit.
//
// Registers: x19 = n, x20 = &n->cpu, x21 = n->cycles, x22 = the limit, x23 = RDRAM, x24 = jitTable,
// x25 = tlbCache, x26 = codePage, x27 = cpu.dcacheTag, x28 = cpu.dcacheDirty. Each block starts with
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
	struct Exit exits[2 * 64];
	int exitCount;
	struct Slow slows[64];
	int slowCount;
	struct Miss misses[64];
	int missCount;
	uint8_t* leave;
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
	bool likely = (op >> 26) >= 0x14 || ((op >> 26) == 0x01 && (rt & 2));
	int cc = branchCondition(e, op, rs, rt);
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
	}
	return 0;
}

// rd = rn & value, through a scratch register if AArch64 can't encode the constant.
static void andConst(struct Emitter* e, int rd, int rn, uint32_t value) {
	if (!aLogicImm(e, A_AND_IMM, rd, rn, value)) {
		aMovImm(e, 17, value);
		aDp(e, A_AND, rd, rn, 17, S_LSL, 0);
	}
}

// A load or store to cached RDRAM, through kseg0 or a page in the TLB cache, with dataTiming's
// D-cache check; anything else jumps to the slow path before any change. w0 ends up the physical
// address, w8 the virtual one, w1 the D-cache line and w2 its tag.
static void emitMemory(struct Jit* j, struct Slow* slow) {
	struct Emitter* e = &j->e;
	uint32_t op = slow->op;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, size = memSize(op);
	bool isStore = (op >> 26) >= 0x28 && (op >> 26) != 0x37;
	slow->sites = 0;

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
	if (isStore) {
		aLsrImm(e, 1, 0, 12);
		aLdst(e, L_LDRB, 1, CODE, 1, L_UXTW);
		slow->site[slow->sites++] = aCbnz(e, 1, NULL);
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
	aAddImmX(&e, CODE, N, N64_OFF(codePage));
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
	struct Emitter* e = &j->e;
	j->leave = e->p;
	aB(e, n->jitLeave);
	uint8_t* entry = e->p;

	bool delay = false;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k], p = paddr + k * 4;
		uint64_t pc = sext32(vaddr + k * 4);
		bool last = k + 1 == count;
		bool lineStart = k == 0 || !(p & 31);
		enum Kind kind;
		uint8_t* mark = e->p;
		if (memSize(op)) {
			struct Slow* slow = &j->slows[j->slowCount++];
			*slow = (struct Slow) { .op = op, .paddr = p, .pc = pc, .delay = delay, .lineStart = lineStart, .last = last };
			emitMemory(j, slow);
			if (lineStart) {
				icacheCheck(e, (uint32_t) pc, p, false);
			}
			aDpImm(e, A_ADD_IMM | W64, CYCLES, CYCLES, 1);
			if (last) {
				exitAt(j, aB(e, NULL), pc, PLAIN, delay, false);
				j->exits[j->exitCount - 1].last = true;
			} else {
				aDp(e, A_SUBS | W64, ZR, CYCLES, LIMIT, S_LSL, 0);
				exitAt(j, aBcond(e, C_HS, NULL), pc, PLAIN, delay, false);
			}
			slow->resume = e->p;
			delay = false;
			continue;
		}
		bool inl = !(delay && isBranchOp(op));
		if (inl) {
			if (lineStart) {
				icacheCheck(e, (uint32_t) pc, p, false);
			}
			inl = emitInline(j, op, pc, &kind);
			if (!inl) {
				e->p = mark;
				if (j->exitCount && j->exits[j->exitCount - 1].site >= mark) {
					--j->exitCount;
				}
			}
		}
		if (inl) {
			aDpImm(e, A_ADD_IMM | W64, CYCLES, CYCLES, 1);
			if (last) {
				exitAt(j, aB(e, NULL), pc, kind, delay, false);
				j->exits[j->exitCount - 1].last = true;
			} else {
				aDp(e, A_SUBS | W64, ZR, CYCLES, LIMIT, S_LSL, 0);
				exitAt(j, aBcond(e, C_HS, NULL), pc, kind, delay, false);
			}
			delay = kind != PLAIN;
			continue;
		}
		emitHelper(j, op, pc, p, delay, lineStart, last);
		delay = isBranchOp(op);
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
