#include "n64.h"
#include "x64/emit.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

// x86-64 code for blocks of RSP instructions. Each block's first instruction takes its timing from
// rspTimingStep (the instruction before it isn't known); the rest have the timing compiled in.
// Scalar instructions and DMEM loads and stores run inline, the vector unit and COP0 through C.
// Code stops after any instruction that takes the clock to the limit, as rspCatchUp does.
//
// Registers: rbx = n, r12 = &n->rsp, r13 = clock, r14 = the limit, r15 = lastStart,
// ebp = lastAlone.

enum { ARENA_SIZE = 4 << 20, BLOCK_CODE_MAX = 32 << 10, SHARED_SIZE = 256, BLOCK_MAX = 64 };

#define RSP_AT(f) M(R12, (int32_t) offsetof(struct RSP, f))
#define N64_AT(f) M(RBX, (int32_t) offsetof(struct N64, f))

struct RspBlock {
	struct RspBlock* next;
	void* code;
	uint32_t count;
	uint8_t ops[BLOCK_MAX * 4];
};

static struct Opnd gpr(int i) {
	return M(R12, (int32_t) (offsetof(struct RSP, r) + 4 * i));
}

static struct Opnd scalarReady(int i) {
	return M(R12, (int32_t) (offsetof(struct RSP, scalarReady) + 8 * i));
}

static struct Opnd vectorReady(int i) {
	return M(R12, (int32_t) (offsetof(struct RSP, vectorReady) + 8 * i));
}

static struct Opnd dmem(int index) {
	return MI(RBX, index, 1, (int32_t) offsetof(struct N64, sp.mem));
}

enum Kind { PLAIN, BRANCH };

struct Exit {
	uint8_t* site;
	uint32_t op, pc;
	enum Kind kind;
	bool delay, last, halt, pcsStored;
};

struct Jit {
	struct Emitter e;
	struct Exit exits[2 * BLOCK_MAX];
	int exitCount;
	uint8_t* leave;
};

static void exitAt(struct Jit* j, uint8_t* site, uint32_t op, uint32_t pc, enum Kind kind, bool delay) {
	j->exits[j->exitCount++] = (struct Exit) { site, op, pc, kind, delay, false, false, false };
}

static void callC(struct Emitter* e, const void* f) {
	xMovI64(e, RAX, (uint64_t) (uintptr_t) f);
	xCallR(e, RAX);
}

static void storeTiming(struct Emitter* e) {
	xStore(e, 8, RSP_AT(clock), R13);
	xStore(e, 8, RSP_AT(lastStart), R15);
	xStore(e, 1, RSP_AT(lastAlone), RBP);
}

static void loadTiming(struct Emitter* e) {
	xLoad(e, 8, R13, RSP_AT(clock));
	xLoad(e, 8, R15, RSP_AT(lastStart));
	xMovzx(e, RBP, 1, RSP_AT(lastAlone));
}

// rspTimingStep for an instruction whose predecessor (prev, at pc - 4) is in the same block, which
// issued alone or paired as ebp says. Paired when it may and all it reads was ready at lastStart.
static void emitTiming(struct Emitter* e, uint32_t op, uint32_t prev, uint32_t pc) {
	int scalar[8], vector[16], nv, written[16], latency[16], nw;
	int ns = rspTimingRegs(op, scalar, vector, &nv, written, latency, &nw);
	bool pairable = rspPairable(prev, (pc - 4) & 0xFFC, op);
	if (pairable) {
		xAluRR(e, X_XOR, 4, R(RAX), RAX);
	} else {
		xMovRR(e, 8, RAX, R13);
	}
	for (int i = 0; i < ns + nv; ++i) {
		struct Opnd m = i < ns ? scalarReady(scalar[i]) : vectorReady(vector[i - ns]);
		if (i < ns && scalar[i] == 0) {
			continue;
		}
		xAluRM(e, X_CMP, 8, RAX, m);
		xCmov(e, CC_B, 8, RAX, m);
	}
	uint8_t* alone1 = NULL;
	uint8_t* alone2 = NULL;
	uint8_t* done = NULL;
	if (pairable) {
		xTest(e, 4, R(RBP), RBP);
		alone1 = xJcc(e, CC_E, NULL);
		xAluRR(e, X_CMP, 8, R(RAX), R15);
		alone2 = xJcc(e, CC_A, NULL);
		xAluRR(e, X_XOR, 4, R(RBP), RBP);
		done = xJmp(e, NULL);
		patchHere(e, alone1);
		patchHere(e, alone2);
		xAluRR(e, X_CMP, 8, R(RAX), R13);
		xCmov(e, CC_B, 8, RAX, R(R13));
	}
	xMovRR(e, 8, R15, RAX);
	xLea(e, 8, R13, M(RAX, 1 + rspIsBranch(op)));
	xMovI(e, RBP, 1);
	if (done) {
		patchHere(e, done);
	}
	for (int i = 0; i < nw; ++i) {
		if (written[i] == 0) {
			continue;
		}
		xLea(e, 8, RAX, M(R15, latency[i]));
		xStore(e, 8, written[i] < 32 ? scalarReady(written[i]) : vectorReady(written[i] - 32), RAX);
	}
}

// The DMEM address of rs + offset in ecx, and whether the access of size bytes wraps past the end.
static uint8_t* dmemAddress(struct Emitter* e, int rs, int32_t offset, int size) {
	xLoad(e, 4, RCX, gpr(rs));
	if (offset) {
		xAluI(e, X_ADD, 4, R(RCX), offset);
	}
	xAluI(e, X_AND, 4, R(RCX), 0xFFF);
	if (size == 1) {
		return NULL;
	}
	xAluI(e, X_CMP, 4, R(RCX), 0x1000 - size);
	return xJcc(e, CC_A, NULL);
}

// One DMEM byte at a time, wrapping, as the interpreter does it.
static void byteLoop(struct Emitter* e, bool store, int size, bool sign) {
	if (store) {
		for (int i = size - 1; i >= 0; --i) {
			xLea(e, 4, RDX, M(RCX, i));
			xAluI(e, X_AND, 4, R(RDX), 0xFFF);
			xStore(e, 1, dmem(RDX), RAX);
			xShiftI(e, X_SHR, 4, R(RAX), 8);
		}
		return;
	}
	xAluRR(e, X_XOR, 4, R(RAX), RAX);
	for (int i = 0; i < size; ++i) {
		xLea(e, 4, RDX, M(RCX, i));
		xAluI(e, X_AND, 4, R(RDX), 0xFFF);
		xShiftI(e, X_SHL, 4, R(RAX), 8);
		xMovzx(e, RDX, 1, dmem(RDX));
		xAluRR(e, X_OR, 4, R(RAX), RDX);
	}
	if (sign && size < 4) {
		xShiftI(e, X_SHL, 4, R(RAX), 32 - 8 * size);
		xShiftI(e, X_SAR, 4, R(RAX), 32 - 8 * size);
	}
}

static void emitMemory(struct Emitter* e, uint32_t op) {
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31;
	int code = op >> 26;
	bool store = code >= 0x28;
	int size = (code & 3) == 0 ? 1 : (code & 3) == 1 ? 2 : 4;
	bool sign = code == 0x20 || code == 0x21;
	uint8_t* wraps = dmemAddress(e, rs, (int16_t) op, size);
	if (store) {
		xLoad(e, 4, RAX, gpr(rt));
	}
	if (size == 1) {
		if (store) {
			xStore(e, 1, dmem(RCX), RAX);
		} else if (sign) {
			xMovsx(e, RAX, 1, dmem(RCX));
		} else {
			xMovzx(e, RAX, 1, dmem(RCX));
		}
	} else {
		if (size == 2) {
			if (store) {
				xShiftI(e, X_ROL, 2, R(RAX), 8);
				xStore(e, 2, dmem(RCX), RAX);
			} else {
				xMovzx(e, RAX, 2, dmem(RCX));
				xBswap(e, 4, RAX);
				xShiftI(e, sign ? X_SAR : X_SHR, 4, R(RAX), 16);
			}
		} else if (store) {
			xBswap(e, 4, RAX);
			xStore(e, 4, dmem(RCX), RAX);
		} else {
			xLoad(e, 4, RAX, dmem(RCX));
			xBswap(e, 4, RAX);
		}
		uint8_t* done = xJmp(e, NULL);
		patchHere(e, wraps);
		if (store) {
			xLoad(e, 4, RAX, gpr(rt));
		}
		byteLoop(e, store, size, sign);
		patchHere(e, done);
	}
	if (!store && rt) {
		xStore(e, 4, gpr(rt), RAX);
	}
}

// nextPc = taken (cc) ? target : pc + 8, as branch() leaves it.
static void branchTo(struct Emitter* e, int cc, uint32_t pc, uint32_t target) {
	xMovIKeepFlags(e, RCX, (pc + 8) & 0xFFC);
	xMovIKeepFlags(e, RDX, target & 0xFFC);
	xCmov(e, cc, 4, RCX, R(RDX));
	xStore(e, 4, RSP_AT(nextPc), RCX);
}

static void storeConst(struct Emitter* e, int i, uint32_t v) {
	if (i) {
		xStoreI(e, 4, gpr(i), v);
	}
}

// Stores sp.pc and nextPc as the interpreter has them while it runs the instruction at pc.
static void storePcs(struct Emitter* e, uint32_t pc, bool delay) {
	if (delay) {
		xLoad(e, 4, RAX, RSP_AT(nextPc));
		xStore(e, 4, N64_AT(sp.pc), RAX);
		xAluI(e, X_ADD, 4, R(RAX), 4);
		xAluI(e, X_AND, 4, R(RAX), 0xFFC);
		xStore(e, 4, RSP_AT(nextPc), RAX);
	} else {
		xStoreI(e, 4, N64_AT(sp.pc), (pc + 4) & 0xFFC);
		xStoreI(e, 4, RSP_AT(nextPc), (pc + 8) & 0xFFC);
	}
}

// IMEM's generation, in the frame's spare slot, to see whether a call rewrote it.
static void saveGen(struct Emitter* e) {
	xLoad(e, 4, RAX, N64_AT(rspCodeGen));
	xStore(e, 4, M(RSP, 0), RAX);
}

// Runs the instruction and says whether it was timed as a branch. Sets *check when the code must
// see whether to stop after it (BREAK, and COP0 writes, which can halt the RSP or rewrite IMEM).
static enum Kind emitInstruction(struct Jit* j, uint32_t op, uint32_t pc, bool delay, bool* check, bool* pcsStored) {
	struct Emitter* e = &j->e;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, rd = (op >> 11) & 31, sa = (op >> 6) & 31;
	uint32_t imm = (uint32_t) (int32_t) (int16_t) op;
	uint32_t target = pc + 4 + (imm << 2);
	*check = false;
	*pcsStored = false;
	switch (op >> 26) {
	case 0x00: {
		int f = op & 0x3F;
		switch (f) {
		case 0x00: case 0x02: case 0x03:
			if (rd) {
				xLoad(e, 4, RAX, gpr(rt));
				if (sa) {
					xShiftI(e, f == 0 ? X_SHL : f == 2 ? X_SHR : X_SAR, 4, R(RAX), sa);
				}
				xStore(e, 4, gpr(rd), RAX);
			}
			return PLAIN;
		case 0x04: case 0x06: case 0x07:
			if (rd) {
				xLoad(e, 4, RCX, gpr(rs));
				xLoad(e, 4, RAX, gpr(rt));
				xShiftCl(e, f == 4 ? X_SHL : f == 6 ? X_SHR : X_SAR, 4, R(RAX));
				xStore(e, 4, gpr(rd), RAX);
			}
			return PLAIN;
		case 0x08: case 0x09:
			xLoad(e, 4, RAX, gpr(rs));
			xAluI(e, X_AND, 4, R(RAX), 0xFFC);
			xStore(e, 4, RSP_AT(nextPc), RAX);
			if (f == 0x09) {
				storeConst(e, rd, pc + 8);
			}
			return BRANCH;
		case 0x0D:
			storePcs(e, pc, delay);
			*pcsStored = true;
			saveGen(e);
			xMovRR(e, 8, RDI, RBX);
			callC(e, rspBreak);
			*check = true;
			return PLAIN;
		case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25: case 0x26: case 0x27: {
			static const int alu[] = { X_ADD, X_ADD, X_SUB, X_SUB, X_AND, X_OR, X_XOR, X_OR };
			if (rd) {
				xLoad(e, 4, RAX, gpr(rs));
				xAluRM(e, alu[f - 0x20], 4, RAX, gpr(rt));
				if (f == 0x27) {
					xNot(e, 4, R(RAX));
				}
				xStore(e, 4, gpr(rd), RAX);
			}
			return PLAIN;
		}
		case 0x2A: case 0x2B:
			if (rd) {
				xLoad(e, 4, RCX, gpr(rs));
				xAluRR(e, X_XOR, 4, R(RAX), RAX);
				xAluRM(e, X_CMP, 4, RCX, gpr(rt));
				xSetcc(e, f == 0x2A ? CC_L : CC_B, R(RAX));
				xStore(e, 4, gpr(rd), RAX);
			}
			return PLAIN;
		}
		return PLAIN;
	}
	case 0x01:
		switch (rt) {
		case 0x00: case 0x01: case 0x10: case 0x11:
			xAluI(e, X_CMP, 4, gpr(rs), 0);
			if (rt & 0x10) {
				storeConst(e, 31, pc + 8);
			}
			branchTo(e, rt & 1 ? CC_GE : CC_L, pc, target);
			return BRANCH;
		}
		// Not a branch, but timed as one.
		xStoreI(e, 4, RSP_AT(nextPc), (pc + 8) & 0xFFC);
		return BRANCH;
	case 0x02: case 0x03:
		xStoreI(e, 4, RSP_AT(nextPc), ((op & 0x3FF) << 2) & 0xFFC);
		if (op >> 26 == 0x03) {
			storeConst(e, 31, pc + 8);
		}
		return BRANCH;
	case 0x04: case 0x05:
		xLoad(e, 4, RAX, gpr(rs));
		xAluRM(e, X_CMP, 4, RAX, gpr(rt));
		branchTo(e, op >> 26 == 0x04 ? CC_E : CC_NE, pc, target);
		return BRANCH;
	case 0x06: case 0x07:
		xAluI(e, X_CMP, 4, gpr(rs), 0);
		branchTo(e, op >> 26 == 0x06 ? CC_LE : CC_G, pc, target);
		return BRANCH;
	case 0x08: case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x0E: {
		static const int alu[] = { X_ADD, X_ADD, 0, 0, X_AND, X_OR, X_XOR };
		if (!rt) {
			return PLAIN;
		}
		int code = op >> 26;
		if (code == 0x0A || code == 0x0B) {
			xLoad(e, 4, RCX, gpr(rs));
			xAluRR(e, X_XOR, 4, R(RAX), RAX);
			xAluI(e, X_CMP, 4, R(RCX), (int32_t) imm);
			xSetcc(e, code == 0x0A ? CC_L : CC_B, R(RAX));
		} else {
			xLoad(e, 4, RAX, gpr(rs));
			xAluI(e, alu[code - 0x08], 4, R(RAX), code >= 0x0C ? (int32_t) (uint16_t) op : (int32_t) imm);
		}
		xStore(e, 4, gpr(rt), RAX);
		return PLAIN;
	}
	case 0x0F:
		storeConst(e, rt, (uint32_t) (uint16_t) op << 16);
		return PLAIN;
	case 0x10:
		if (rs == 0x00 || rs == 0x04) {
			storePcs(e, pc, delay);
			*pcsStored = true;
			storeTiming(e);
			saveGen(e);
			xMovRR(e, 8, RDI, RBX);
			xMovI(e, RSI, op);
			callC(e, rspCop0);
			loadTiming(e);
			*check = rs == 0x04;
		}
		return PLAIN;
	case 0x12:
		xMovRR(e, 8, RDI, RBX);
		xMovI(e, RSI, op);
		callC(e, rspCop2);
		return PLAIN;
	case 0x20: case 0x21: case 0x23: case 0x24: case 0x25: case 0x27:
	case 0x28: case 0x29: case 0x2B:
		emitMemory(e, op);
		return PLAIN;
	case 0x32: case 0x3A:
		xMovRR(e, 8, RDI, RBX);
		xMovI(e, RSI, op);
		callC(e, op >> 26 == 0x32 ? (const void*) rspVectorLoad : (const void*) rspVectorStore);
		return PLAIN;
	}
	return PLAIN;
}

static uint8_t* arenaGet(struct N64* n) {
	if (n->rspArena) {
		return n->rspArena;
	}
	void* p = mmap(NULL, ARENA_SIZE, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (p == MAP_FAILED) {
		return NULL;
	}
	n->rspArena = p;
	n->rspUsed = SHARED_SIZE;
	// rspJitEnter(n, code, limit), and the way back out.
	struct Emitter e = { p, p, (uint8_t*) p + SHARED_SIZE };
	xPush(&e, RBX);
	xPush(&e, RBP);
	xPush(&e, R12);
	xPush(&e, R13);
	xPush(&e, R14);
	xPush(&e, R15);
	xAluI(&e, X_SUB, 8, R(RSP), 8);
	xMovRR(&e, 8, RBX, RDI);
	xLea(&e, 8, R12, M(RDI, (int32_t) offsetof(struct N64, rsp)));
	xMovRR(&e, 8, R14, RDX);
	xJmpR(&e, RSI);
	n->rspLeave = e.p;
	xAluI(&e, X_ADD, 8, R(RSP), 8);
	xPop(&e, R15);
	xPop(&e, R14);
	xPop(&e, R13);
	xPop(&e, R12);
	xPop(&e, RBP);
	xPop(&e, RBX);
	xRet(&e);
	n->rspEnter = (void (*)(struct N64*, void*, uint64_t)) p;
	return p;
}

void rspJitFree(struct N64* n) {
	for (int i = 0; i < 1024; ++i) {
		struct RspBlock* b = n->rspVariants[i];
		while (b) {
			struct RspBlock* next = b->next;
			free(b);
			b = next;
		}
		n->rspVariants[i] = NULL;
		n->rspTable[i].code = NULL;
	}
	if (n->rspArena) {
		munmap(n->rspArena, ARENA_SIZE);
		n->rspArena = NULL;
	}
}

// Goes on to the block at sp.pc if the table has it for the current IMEM, or leaves.
static void chain(struct Jit* j) {
	struct Emitter* e = &j->e;
	xAluRR(e, X_CMP, 8, R(R13), R14);
	xJcc(e, CC_AE, j->leave);
	xLoad(e, 4, RCX, N64_AT(sp.pc));
	xLea(e, 8, RDX, MI(RBX, RCX, 4, (int32_t) offsetof(struct N64, rspTable)));
	xLoad(e, 4, RAX, N64_AT(rspCodeGen));
	xAluRM(e, X_CMP, 4, RAX, M(RDX, (int32_t) offsetof(struct RspJitEntry, gen)));
	xJcc(e, CC_NE, j->leave);
	xLoad(e, 8, RAX, M(RDX, (int32_t) offsetof(struct RspJitEntry, code)));
	xTest(e, 8, R(RAX), RAX);
	xJcc(e, CC_E, j->leave);
	xJmpR(e, RAX);
}

static void* compile(struct N64* n, struct RspBlock* b, uint32_t start) {
	uint8_t* arena = arenaGet(n);
	if (!arena || n->rspUsed + BLOCK_CODE_MAX > ARENA_SIZE) {
		return NULL;
	}
	static struct Jit jit;
	struct Jit* j = &jit;
	j->e = (struct Emitter) { arena + n->rspUsed, arena + n->rspUsed, arena + n->rspUsed + BLOCK_CODE_MAX };
	j->exitCount = 0;
	j->leave = n->rspLeave;
	struct Emitter* e = &j->e;

	bool delay = false;
	uint32_t prev = 0;
	for (uint32_t k = 0; k < b->count; ++k) {
		uint32_t op = be32(&b->ops[k * 4]), pc = start + k * 4;
		bool last = k + 1 == b->count;
		if (k == 0) {
			xMovRR(e, 8, RDI, RBX);
			xMovI(e, RSI, op);
			xMovI(e, RDX, pc);
			callC(e, rspTimingStep);
			loadTiming(e);
		} else {
			emitTiming(e, op, prev, pc);
		}
		bool check, pcsStored;
		enum Kind kind = emitInstruction(j, op, pc, delay, &check, &pcsStored);
		uint8_t* halted = NULL;
		uint8_t* gen = NULL;
		if (check) {
			xTestI(e, 4, N64_AT(sp.status), 1);
			halted = xJcc(e, CC_NE, NULL);
			xLoad(e, 4, RAX, N64_AT(rspCodeGen));
			xAluRM(e, X_CMP, 4, RAX, M(RSP, 0));
			gen = xJcc(e, CC_NE, NULL);
		}
		if (last) {
			exitAt(j, xJmp(e, NULL), op, pc, kind, delay);
			j->exits[j->exitCount - 1].last = true;
		} else {
			xAluRR(e, X_CMP, 8, R(R13), R14);
			exitAt(j, xJcc(e, CC_AE, NULL), op, pc, kind, delay);
		}
		j->exits[j->exitCount - 1].pcsStored = pcsStored;
		if (check) {
			// Halted or IMEM rewritten: leave whatever the clock says.
			uint8_t* over = xJmp(e, NULL);
			patchHere(e, halted);
			patchHere(e, gen);
			exitAt(j, xJmp(e, NULL), op, pc, kind, delay);
			j->exits[j->exitCount - 1].halt = true;
			j->exits[j->exitCount - 1].pcsStored = pcsStored;
			patchHere(e, over);
		}
		delay = kind == BRANCH;
		prev = op;
	}

	// Leaving after an instruction: the state rspStep would leave.
	for (int i = 0; i < j->exitCount; ++i) {
		struct Exit* x = &j->exits[i];
		patchHere(e, x->site);
		// A COP0 or BREAK instruction stored the pcs before its call.
		if (x->delay && !x->pcsStored) {
			storePcs(e, x->pc, true);
		} else if (!x->delay) {
			xStoreI(e, 4, N64_AT(sp.pc), (x->pc + 4) & 0xFFC);
			if (x->kind == PLAIN) {
				xStoreI(e, 4, RSP_AT(nextPc), (x->pc + 8) & 0xFFC);
			}
		}
		xStoreI(e, 4, RSP_AT(lastOp), x->op);
		xStoreI(e, 4, RSP_AT(lastPc), x->pc);
		xStoreI(e, 1, RSP_AT(inDelay), x->kind == BRANCH);
		storeTiming(e);
		if (x->last && !x->halt && x->kind == PLAIN) {
			chain(j);
		} else {
			xJmp(e, j->leave);
		}
	}
	if (e->p > e->end) {
		return NULL;
	}
	uint8_t* code = arena + n->rspUsed;
	n->rspUsed += (size_t) (e->p - code + 15) & ~(size_t) 15;
	return code;
}

// From pc to a branch's delay slot, the end of IMEM or BLOCK_MAX instructions.
static uint32_t blockLength(const uint8_t* imem, uint32_t pc) {
	uint32_t count = 0;
	bool delay = false;
	while (count < BLOCK_MAX && pc + count * 4 < 0x1000) {
		uint32_t op = be32(&imem[pc + count * 4]);
		// A branch in a delay slot is left to the interpreter.
		if (delay && rspIsBranch(op)) {
			break;
		}
		++count;
		if (delay) {
			break;
		}
		delay = rspIsBranch(op);
	}
	return count;
}

// The code for the block at pc: from the table, a block compiled before for the same instructions,
// or a new one. NULL when the arena is full (it starts again).
void* rspJitLookup(struct N64* n, uint32_t pc) {
	struct RspJitEntry* t = &n->rspTable[pc >> 2];
	if (t->gen == n->rspCodeGen && t->code) {
		return t->code;
	}
	const uint8_t* imem = &n->sp.mem[0x1000];
	uint32_t count = blockLength(imem, pc);
	for (struct RspBlock* b = n->rspVariants[pc >> 2]; b; b = b->next) {
		if (b->count == count && !memcmp(b->ops, &imem[pc], count * 4)) {
			*t = (struct RspJitEntry) { n->rspCodeGen, b->code };
			return b->code;
		}
	}
	struct RspBlock* b = malloc(sizeof(*b));
	if (!b) {
		return NULL;
	}
	b->count = count;
	memcpy(b->ops, &imem[pc], count * 4);
	if (!(b->code = compile(n, b, pc))) {
		free(b);
		rspJitFree(n);
		return NULL;
	}
	b->next = n->rspVariants[pc >> 2];
	n->rspVariants[pc >> 2] = b;
	*t = (struct RspJitEntry) { n->rspCodeGen, b->code };
	return b->code;
}
