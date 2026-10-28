#include "n64.h"
#include "x64/emit.h"

#include <stddef.h>
#include <sys/mman.h>

// x86-64 code for a block of cached RDRAM instructions. The ALU and branches run inline; the rest
// goes through cpuJitStep. Code stops after any instruction that takes the cycle count to the
// limit (the next event, or Count reaching Compare), leaving the CPU as the interpreter would.
//
// Registers: rbx = n, r12 = &n->cpu, r13 = n->cycles, r14 = the limit. A block that ends below the
// limit goes on to the next through n->jitTable without returning to C.

enum { ARENA_SIZE = 32 << 20, BLOCK_CODE_MAX = 16 << 10, SHARED_SIZE = 256 };

#define CPU_AT(f) M(R12, (int32_t) offsetof(struct CPU, f))
#define N64_AT(f) M(RBX, (int32_t) offsetof(struct N64, f))

static struct Opnd gpr(int i) {
	return M(R12, (int32_t) (offsetof(struct CPU, gpr) + 8 * i));
}

static bool fits32(uint64_t v) {
	return (uint64_t) (int64_t) (int32_t) v == v;
}

// Loads a 64-bit constant without touching the flags.
static void movConst(struct Emitter* e, int reg, uint64_t v) {
	if (fits32(v)) {
		emitOp1(e, 8, 0xC7, 0, R(reg));
		e32(e, (uint32_t) v);
	} else {
		xMovI64(e, reg, v);
	}
}

static void storeConst(struct Emitter* e, struct Opnd dst, uint64_t v) {
	if (fits32(v)) {
		xStoreI(e, 8, dst, (uint32_t) v);
	} else {
		xMovI64(e, RAX, v);
		xStore(e, 8, dst, RAX);
	}
}

static void storeGpr(struct Emitter* e, int i, int reg) {
	if (i) {
		xStore(e, 8, gpr(i), reg);
	}
}

static void storeGprConst(struct Emitter* e, int i, uint64_t v) {
	if (i) {
		storeConst(e, gpr(i), v);
	}
}

enum Kind { PLAIN, BRANCH, LIKELY };

// Where an inline instruction leaves the block: after it, or, for a branch-likely not taken,
// past its delay slot.
struct Exit {
	uint8_t* site;
	uint64_t pc;
	enum Kind kind;
	bool delay;
	bool skip;
	bool last;
};

struct Jit {
	struct Emitter e;
	struct Exit exits[2 * 64];
	int exitCount;
	uint8_t* leave;
};

static void exitAt(struct Jit* j, uint8_t* site, uint64_t pc, enum Kind kind, bool delay, bool skip) {
	j->exits[j->exitCount++] = (struct Exit) { site, pc, kind, delay, skip, false };
}

// Sets the flags for a branch condition and returns the condition code under which it is taken.
static int branchCondition(struct Emitter* e, uint32_t op, int rs, int rt) {
	switch (op >> 26) {
	case 0x04: case 0x14:
		xLoad(e, 8, RAX, gpr(rs));
		xAluRM(e, X_CMP, 8, RAX, gpr(rt));
		return CC_E;
	case 0x05: case 0x15:
		xLoad(e, 8, RAX, gpr(rs));
		xAluRM(e, X_CMP, 8, RAX, gpr(rt));
		return CC_NE;
	case 0x06: case 0x16:
		xAluI(e, X_CMP, 8, gpr(rs), 0);
		return CC_LE;
	case 0x07: case 0x17:
		xAluI(e, X_CMP, 8, gpr(rs), 0);
		return CC_G;
	}
	xAluI(e, X_CMP, 8, gpr(rs), 0);
	return rt & 1 ? CC_GE : CC_L;
}

// Emits the instruction at pc inline if it can be, and says what kind it is.
static bool emitInline(struct Jit* j, uint32_t op, uint64_t pc, enum Kind* kind) {
	struct Emitter* e = &j->e;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, rd = (op >> 11) & 31, sa = (op >> 6) & 31;
	int32_t imm = (int16_t) op;
	uint64_t target = pc + 4 + ((uint64_t) (int64_t) imm << 2);
	*kind = PLAIN;
	switch (op >> 26) {
	case 0x00:
		switch (op & 0x3F) {
		case 0x00: case 0x02: case 0x03:
			if (rd) {
				if ((op & 0x3F) == 0x03) {
					xLoad(e, 8, RAX, gpr(rt));
					xShiftI(e, X_SAR, 8, R(RAX), sa);
				} else {
					xLoad(e, 4, RAX, gpr(rt));
					if (sa) {
						xShiftI(e, (op & 0x3F) ? X_SHR : X_SHL, 4, R(RAX), sa);
					}
				}
				xMovsxd(e, RAX, R(RAX));
				storeGpr(e, rd, RAX);
			}
			return true;
		case 0x04: case 0x06: case 0x07:
			if (rd) {
				xLoad(e, 4, RCX, gpr(rs));
				if ((op & 0x3F) == 0x07) {
					xAluI(e, X_AND, 4, R(RCX), 31);
					xLoad(e, 8, RAX, gpr(rt));
					xShiftCl(e, X_SAR, 8, R(RAX));
				} else {
					xLoad(e, 4, RAX, gpr(rt));
					xShiftCl(e, (op & 0x3F) == 0x04 ? X_SHL : X_SHR, 4, R(RAX));
				}
				xMovsxd(e, RAX, R(RAX));
				storeGpr(e, rd, RAX);
			}
			return true;
		case 0x08: case 0x09:
			xLoad(e, 8, RAX, gpr(rs));
			xStore(e, 8, CPU_AT(nextPc), RAX);
			if ((op & 0x3F) == 0x09) {
				storeGprConst(e, rd, pc + 8);
			}
			*kind = BRANCH;
			return true;
		case 0x0F:
			return true;
		case 0x10: case 0x12:
			if (rd) {
				xLoad(e, 8, RAX, (op & 0x3F) == 0x10 ? CPU_AT(hi) : CPU_AT(lo));
				storeGpr(e, rd, RAX);
			}
			return true;
		case 0x11: case 0x13:
			xLoad(e, 8, RAX, gpr(rs));
			xStore(e, 8, (op & 0x3F) == 0x11 ? CPU_AT(hi) : CPU_AT(lo), RAX);
			return true;
		case 0x14: case 0x16: case 0x17:
			if (rd) {
				xLoad(e, 4, RCX, gpr(rs));
				xLoad(e, 8, RAX, gpr(rt));
				xShiftCl(e, (op & 0x3F) == 0x14 ? X_SHL : (op & 0x3F) == 0x16 ? X_SHR : X_SAR, 8, R(RAX));
				storeGpr(e, rd, RAX);
			}
			return true;
		case 0x21: case 0x23:
			if (rd) {
				xLoad(e, 4, RAX, gpr(rs));
				xAluRM(e, (op & 0x3F) == 0x21 ? X_ADD : X_SUB, 4, RAX, gpr(rt));
				xMovsxd(e, RAX, R(RAX));
				storeGpr(e, rd, RAX);
			}
			return true;
		case 0x24: case 0x25: case 0x26: case 0x27: case 0x2D: case 0x2F: {
			static const int alu[] = { X_AND, X_OR, X_XOR, X_OR };
			if (rd) {
				int f = op & 0x3F;
				xLoad(e, 8, RAX, gpr(rs));
				xAluRM(e, f == 0x2D ? X_ADD : f == 0x2F ? X_SUB : alu[f - 0x24], 8, RAX, gpr(rt));
				if (f == 0x27) {
					xNot(e, 8, R(RAX));
				}
				storeGpr(e, rd, RAX);
			}
			return true;
		}
		case 0x2A: case 0x2B:
			if (rd) {
				xLoad(e, 8, RAX, gpr(rs));
				xAluRM(e, X_CMP, 8, RAX, gpr(rt));
				xSetcc(e, (op & 0x3F) == 0x2A ? CC_L : CC_B, R(RAX));
				xMovzx(e, RAX, 1, R(RAX));
				storeGpr(e, rd, RAX);
			}
			return true;
		case 0x38: case 0x3A: case 0x3B: case 0x3C: case 0x3E: case 0x3F: {
			if (rd) {
				int f = op & 0x3F, amount = sa + (f >= 0x3C ? 32 : 0);
				xLoad(e, 8, RAX, gpr(rt));
				if (amount) {
					xShiftI(e, (f & 3) == 0 ? X_SHL : (f & 3) == 2 ? X_SHR : X_SAR, 8, R(RAX), amount);
				}
				storeGpr(e, rd, RAX);
			}
			return true;
		}
		}
		return false;
	case 0x01:
		if (((op >> 16) & 0x0C) != 0) {
			return false;
		}
		break;
	case 0x02: case 0x03:
		storeConst(e, CPU_AT(nextPc), ((pc + 4) & ~0x0FFFFFFFull) | (uint64_t) (op & 0x03FFFFFF) << 2);
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
			xLoad(e, 4, RAX, gpr(rs));
			xAluI(e, X_ADD, 4, R(RAX), imm);
			xMovsxd(e, RAX, R(RAX));
			storeGpr(e, rt, RAX);
		}
		return true;
	case 0x0A: case 0x0B:
		if (rt) {
			xAluI(e, X_CMP, 8, gpr(rs), imm);
			xSetcc(e, op >> 26 == 0x0A ? CC_L : CC_B, R(RAX));
			xMovzx(e, RAX, 1, R(RAX));
			storeGpr(e, rt, RAX);
		}
		return true;
	case 0x0C: case 0x0D: case 0x0E:
		if (rt) {
			static const int alu[] = { X_AND, X_OR, X_XOR };
			xLoad(e, 8, RAX, gpr(rs));
			xAluI(e, alu[(op >> 26) - 0x0C], 8, R(RAX), (uint16_t) op);
			storeGpr(e, rt, RAX);
		}
		return true;
	case 0x0F:
		storeGprConst(e, rt, sext32((uint32_t) (uint16_t) op << 16));
		return true;
	case 0x19:
		if (rt) {
			xLoad(e, 8, RAX, gpr(rs));
			xAluI(e, X_ADD, 8, R(RAX), imm);
			storeGpr(e, rt, RAX);
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
		exitAt(j, xJcc(e, cc ^ 1, NULL), pc, LIKELY, false, true);
		storeConst(e, CPU_AT(nextPc), target);
		*kind = LIKELY;
	} else {
		movConst(e, RCX, pc + 8);
		movConst(e, RDX, target);
		xCmov(e, cc, 8, RCX, R(RDX));
		xStore(e, 8, CPU_AT(nextPc), RCX);
		*kind = BRANCH;
	}
	return true;
}

// The delay slot's view: pc is where the branch goes and nextPc the instruction after that.
static void delayPrologue(struct Emitter* e) {
	xLoad(e, 8, RAX, CPU_AT(nextPc));
	xStore(e, 8, CPU_AT(pc), RAX);
	xAluI(e, X_ADD, 8, R(RAX), 4);
	xStore(e, 8, CPU_AT(nextPc), RAX);
}

// The I-cache line holding pc, as fetchTiming checks it.
static void icacheCheck(struct Emitter* e, uint32_t vaddr, uint32_t paddr, bool toStall) {
	struct Opnd tag = M(R12, (int32_t) (offsetof(struct CPU, icacheTag) + 4 * ((vaddr >> 5) & 511)));
	uint32_t want = paddr >> 5 | 0x80000000u;
	xAluI(e, X_CMP, 4, tag, (int32_t) want);
	uint8_t* hit = xJcc(e, CC_E, NULL);
	xStoreI(e, 4, tag, want);
	if (toStall) {
		xAluI(e, X_ADD, 4, CPU_AT(stall), 48);
	} else {
		xAluI(e, X_ADD, 8, R(R13), 48);
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

// Goes on to the block at the pc in memory if the jump table has it, or leaves.
static void chain(struct Jit* j) {
	struct Emitter* e = &j->e;
	xLoad(e, 8, RAX, CPU_AT(pc));
	xMovsxd(e, RCX, R(RAX));
	xAluRR(e, X_CMP, 8, R(RCX), RAX);
	xJcc(e, CC_NE, j->leave);
	xMovRR(e, 4, RCX, RAX);
	xShiftI(e, X_SHR, 4, R(RCX), 2);
	xAluI(e, X_AND, 4, R(RCX), JIT_TABLE_SIZE - 1);
	xShiftI(e, X_SHL, 4, R(RCX), 4);
	xLea(e, 8, RDX, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, jitTable)));
	xAluRM(e, X_CMP, 4, RAX, M(RDX, (int32_t) offsetof(struct JitEntry, vaddr)));
	xJcc(e, CC_NE, j->leave);
	xLoad(e, 4, RAX, N64_AT(blockGen));
	xAluRM(e, X_CMP, 4, RAX, M(RDX, (int32_t) offsetof(struct JitEntry, gen)));
	xJcc(e, CC_NE, j->leave);
	xJmpM(e, M(RDX, (int32_t) offsetof(struct JitEntry, code)));
}

// The arena starts with the way in, jitEnter(n, code), and the way out back to C.
static uint8_t* arenaGet(struct N64* n) {
	if (!n->jitArena) {
		void* p = mmap(NULL, ARENA_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (p == MAP_FAILED) {
			return NULL;
		}
		n->jitArena = p;
		struct Emitter e = { p, p, (uint8_t*) p + SHARED_SIZE };
		xPush(&e, RBX);
		xPush(&e, R12);
		xPush(&e, R13);
		xPush(&e, R14);
		xPush(&e, R15);
		xMovRR(&e, 8, RBX, RDI);
		xLea(&e, 8, R12, M(RDI, (int32_t) offsetof(struct N64, cpu)));
		xLoad(&e, 8, R13, M(RBX, (int32_t) offsetof(struct N64, cycles)));
		xLoad(&e, 8, R14, M(RBX, (int32_t) offsetof(struct N64, jitLimit)));
		xJmpR(&e, RSI);
		n->jitLeave = e.p;
		xStore(&e, 8, M(RBX, (int32_t) offsetof(struct N64, cycles)), R13);
		xPop(&e, R15);
		xPop(&e, R14);
		xPop(&e, R13);
		xPop(&e, R12);
		xPop(&e, RBX);
		xRet(&e);
		n->jitEnter = (void (*)(struct N64*, void*)) p;
	}
	return n->jitArena;
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
	j->e = (struct Emitter) { arena + n->jitUsed, arena + n->jitUsed, arena + n->jitUsed + BLOCK_CODE_MAX };
	j->exitCount = 0;
	j->leave = n->jitLeave;
	struct Emitter* e = &j->e;

	bool delay = false;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k], p = paddr + k * 4;
		uint64_t pc = sext32(vaddr + k * 4);
		bool last = k + 1 == count;
		bool lineStart = k == 0 || !(p & 31);
		enum Kind kind;
		uint8_t* mark = e->p;
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
			xAluI(e, X_ADD, 8, R(R13), 1);
			if (last) {
				exitAt(j, xJmp(e, NULL), pc, kind, delay, false);
				j->exits[j->exitCount - 1].last = true;
			} else {
				xAluRR(e, X_CMP, 8, R(R13), R14);
				exitAt(j, xJcc(e, CC_AE, NULL), pc, kind, delay, false);
			}
			delay = kind != PLAIN;
			continue;
		}
		if (lineStart) {
			icacheCheck(e, (uint32_t) pc, p, true);
		}
		if (delay) {
			delayPrologue(e);
		} else {
			storeConst(e, CPU_AT(pc), pc + 4);
			storeConst(e, CPU_AT(nextPc), pc + 8);
		}
		storeConst(e, CPU_AT(curPc), pc);
		xStoreI(e, 1, CPU_AT(inDelay), delay);
		xStoreI(e, 1, CPU_AT(nextInDelay), 0);
		xStore(e, 8, N64_AT(cycles), R13);
		xMovRR(e, 8, RDI, RBX);
		xMovIKeepFlags(e, RSI, op);
		xMovI64(e, RAX, (uint64_t) (uintptr_t) cpuJitStep);
		xCallR(e, RAX);
		xLoad(e, 8, R13, N64_AT(cycles));
		xLoad(e, 8, R14, N64_AT(jitLimit));
		xTest(e, 1, R(RAX), RAX);
		xJcc(e, CC_NE, j->leave);
		xAluRR(e, X_CMP, 8, R(R13), R14);
		xJcc(e, CC_AE, j->leave);
		if (last) {
			xAluI(e, X_CMP, 1, CPU_AT(nextInDelay), 0);
			xJcc(e, CC_NE, j->leave);
			chain(j);
		}
		delay = isBranchOp(op);
	}

	// Leaving after an inline instruction: the state the interpreter would have after it.
	for (int i = 0; i < j->exitCount; ++i) {
		struct Exit* x = &j->exits[i];
		patchHere(e, x->site);
		storeConst(e, CPU_AT(curPc), x->pc);
		xStoreI(e, 1, CPU_AT(inDelay), x->delay);
		xStoreI(e, 1, CPU_AT(nextInDelay), x->kind != PLAIN && !x->skip);
		if (x->skip) {
			xAluI(e, X_ADD, 8, R(R13), 1);
			storeConst(e, CPU_AT(pc), x->pc + 8);
			storeConst(e, CPU_AT(nextPc), x->pc + 12);
		} else if (x->delay) {
			delayPrologue(e);
		} else {
			storeConst(e, CPU_AT(pc), x->pc + 4);
			if (x->kind == PLAIN) {
				storeConst(e, CPU_AT(nextPc), x->pc + 8);
			}
		}
		if (x->last && !(x->kind != PLAIN && !x->skip)) {
			xAluRR(e, X_CMP, 8, R(R13), R14);
			xJcc(e, CC_AE, j->leave);
			chain(j);
		} else {
			xJmp(e, j->leave);
		}
	}

	if (e->p > e->end) {
		return NULL;
	}
	uint8_t* code = arena + n->jitUsed;
	n->jitUsed += (size_t) (e->p - code + 15) & ~(size_t) 15;
	return code;
}
