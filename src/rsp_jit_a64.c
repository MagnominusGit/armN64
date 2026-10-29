#include "n64.h"
#include "a64/emit.h"

#include <stddef.h>
#include <sys/mman.h>

// AArch64 code for blocks of RSP instructions, as rsp_jit_x64.c does it: the first instruction's
// timing from rspTimingStep, the rest compiled in; scalar instructions and DMEM loads and stores
// inline, the vector unit and COP0 through C.
//
// Registers: x19 = n, x20 = &n->rsp, x21 = clock, x22 = the limit, x23 = lastStart,
// w24 = lastAlone, x25 = DMEM, w26 = IMEM's generation before a call.

enum { ARENA_SIZE = 4 << 20, BLOCK_CODE_MAX = 32 << 10, SHARED_SIZE = 256, BLOCK_MAX = 64 };

enum { N = 19, RSP = 20, CLOCK = 21, LIMIT = 22, START = 23, ALONE = 24, DMEM = 25, GEN = 26, IP = 16 };

#define RSP_OFF(f) ((unsigned) offsetof(struct RSP, f))
#define N64_OFF(f) ((unsigned) offsetof(struct N64, f))

static unsigned gprOff(int i) {
	return RSP_OFF(r) + 4 * i;
}

// The base register for n's field at off: n itself, or x16 = its address when it's too far for an
// offset (*at is what to add).
static int fieldBase(struct Emitter* e, unsigned off, unsigned size, unsigned* at) {
	if (off % size == 0 && off / size < 4096) {
		*at = off;
		return N;
	}
	aAddImmX(e, IP, N, off);
	*at = 0;
	return IP;
}

static void loadN32(struct Emitter* e, int rt, unsigned off) {
	unsigned at;
	int base = fieldBase(e, off, 4, &at);
	aLdr(e, rt, base, at);
}

static void storeN32(struct Emitter* e, int rt, unsigned off) {
	unsigned at;
	int base = fieldBase(e, off, 4, &at);
	aStr(e, rt, base, at);
}

static void callC(struct Emitter* e, const void* f) {
	aMovImm64(e, IP, (uint64_t) (uintptr_t) f);
	aBlr(e, IP);
}

static void storeTiming(struct Emitter* e) {
	aStrX(e, CLOCK, RSP, RSP_OFF(clock));
	aStrX(e, START, RSP, RSP_OFF(lastStart));
	aStrb(e, ALONE, RSP, RSP_OFF(lastAlone));
}

static void loadTiming(struct Emitter* e) {
	aLdrX(e, CLOCK, RSP, RSP_OFF(clock));
	aLdrX(e, START, RSP, RSP_OFF(lastStart));
	aLdrb(e, ALONE, RSP, RSP_OFF(lastAlone));
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

// rspTimingStep for an instruction whose predecessor (prev, at pc - 4) is in the same block, which
// issued alone or paired as w24 says. Paired when it may and all it reads was ready at lastStart.
static void emitTiming(struct Emitter* e, uint32_t op, uint32_t prev, uint32_t pc) {
	int scalar[8], vector[16], nv, written[16], latency[16], nw;
	int ns = rspTimingRegs(op, scalar, vector, &nv, written, latency, &nw);
	bool pairable = rspPairable(prev, (pc - 4) & 0xFFC, op);
	if (pairable) {
		aMovImm(e, 0, 0);
	} else {
		aMovX(e, 0, CLOCK);
	}
	for (int i = 0; i < ns + nv; ++i) {
		if (i < ns && scalar[i] == 0) {
			continue;
		}
		aLdrX(e, 1, RSP, i < ns ? RSP_OFF(scalarReady) + 8 * scalar[i] : RSP_OFF(vectorReady) + 8 * vector[i - ns]);
		aDp(e, A_SUBS | W64, ZR, 0, 1, S_LSL, 0);
		aCselX(e, 0, 1, 0, C_LO);
	}
	uint8_t* done = NULL;
	if (pairable) {
		uint8_t* alone1 = aCbz(e, ALONE, NULL);
		aDp(e, A_SUBS | W64, ZR, 0, START, S_LSL, 0);
		uint8_t* alone2 = aBcond(e, C_HI, NULL);
		aMov(e, ALONE, ZR);
		done = aB(e, NULL);
		patchHere(e, alone1);
		patchHere(e, alone2);
		aDp(e, A_SUBS | W64, ZR, 0, CLOCK, S_LSL, 0);
		aCselX(e, 0, CLOCK, 0, C_LO);
	}
	aMovX(e, START, 0);
	aDpImm(e, A_ADD_IMM | W64, CLOCK, 0, 1 + rspIsBranch(op));
	aMovImm(e, ALONE, 1);
	if (done) {
		patchHere(e, done);
	}
	for (int i = 0; i < nw; ++i) {
		if (written[i] == 0) {
			continue;
		}
		aDpImm(e, A_ADD_IMM | W64, 1, START, latency[i]);
		aStrX(e, 1, RSP, written[i] < 32 ? RSP_OFF(scalarReady) + 8 * written[i] : RSP_OFF(vectorReady) + 8 * (written[i] - 32));
	}
}

// The DMEM address of rs + offset in w1, and whether the access of size bytes wraps past the end.
static uint8_t* dmemAddress(struct Emitter* e, int rs, int32_t offset, int size) {
	aLdr(e, 1, RSP, gprOff(rs));
	if (offset) {
		aAddImm(e, 1, 1, offset, 17);
	}
	aAndBits(e, 1, 1, 0, 12);
	if (size == 1) {
		return NULL;
	}
	aDpImm(e, A_SUBS_IMM, ZR, 1, 0x1000 - size);
	return aBcond(e, C_HI, NULL);
}

// One DMEM byte at a time, wrapping, as the interpreter does it.
static void byteLoop(struct Emitter* e, bool store, int size, bool sign) {
	if (store) {
		for (int i = size - 1; i >= 0; --i) {
			aDpImm(e, A_ADD_IMM, 2, 1, i);
			aAndBits(e, 2, 2, 0, 12);
			aLdst(e, L_STRB, 0, DMEM, 2, L_UXTW);
			aLsrImm(e, 0, 0, 8);
		}
		return;
	}
	aMov(e, 0, ZR);
	for (int i = 0; i < size; ++i) {
		aDpImm(e, A_ADD_IMM, 2, 1, i);
		aAndBits(e, 2, 2, 0, 12);
		aLdst(e, L_LDRB, 3, DMEM, 2, L_UXTW);
		aDp(e, A_ORR, 0, 3, 0, S_LSL, 8);
	}
	if (sign && size < 4) {
		aLslImm(e, 0, 0, 32 - 8 * size);
		aAsrImm(e, 0, 0, 32 - 8 * size);
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
		aLdr(e, 0, RSP, gprOff(rt));
	}
	if (size == 1) {
		if (store) {
			aLdst(e, L_STRB, 0, DMEM, 1, L_UXTW);
		} else {
			aLdst(e, sign ? L_LDRSB : L_LDRB, 0, DMEM, 1, L_UXTW);
		}
	} else {
		if (size == 2) {
			if (store) {
				aRev16(e, 0, 0);
				aLdst(e, L_STRH, 0, DMEM, 1, L_UXTW);
			} else {
				aLdst(e, L_LDRH, 0, DMEM, 1, L_UXTW);
				aRev16(e, 0, 0);
				if (sign) {
					aSxth(e, 0, 0);
				}
			}
		} else if (store) {
			aRev(e, 0, 0);
			aLdst(e, L_STR, 0, DMEM, 1, L_UXTW);
		} else {
			aLdst(e, L_LDR, 0, DMEM, 1, L_UXTW);
			aRev(e, 0, 0);
		}
		uint8_t* done = aB(e, NULL);
		patchHere(e, wraps);
		if (store) {
			aLdr(e, 0, RSP, gprOff(rt));
		}
		byteLoop(e, store, size, sign);
		patchHere(e, done);
	}
	if (!store && rt) {
		aStr(e, 0, RSP, gprOff(rt));
	}
}

// nextPc = taken (cond) ? target : pc + 8, as branch() leaves it.
static void branchTo(struct Emitter* e, int cond, uint32_t pc, uint32_t target) {
	aMovImm(e, 2, (pc + 8) & 0xFFC);
	aMovImm(e, 3, target & 0xFFC);
	aCsel(e, 2, 3, 2, cond);
	aStr(e, 2, RSP, RSP_OFF(nextPc));
}

static void storeConst(struct Emitter* e, int i, uint32_t v) {
	if (i) {
		aMovImm(e, 9, v);
		aStr(e, 9, RSP, gprOff(i));
	}
}

// Stores sp.pc and nextPc as the interpreter has them while it runs the instruction at pc.
static void storePcs(struct Emitter* e, uint32_t pc, bool delay) {
	if (delay) {
		aLdr(e, 0, RSP, RSP_OFF(nextPc));
		storeN32(e, 0, N64_OFF(sp.pc));
		aDpImm(e, A_ADD_IMM, 0, 0, 4);
		aAndBits(e, 0, 0, 2, 10);
	} else {
		aMovImm(e, 0, (pc + 4) & 0xFFC);
		storeN32(e, 0, N64_OFF(sp.pc));
		aMovImm(e, 0, (pc + 8) & 0xFFC);
	}
	aStr(e, 0, RSP, RSP_OFF(nextPc));
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
				aLdr(e, 0, RSP, gprOff(rt));
				if (sa) {
					if (f == 0) {
						aLslImm(e, 0, 0, sa);
					} else if (f == 2) {
						aLsrImm(e, 0, 0, sa);
					} else {
						aAsrImm(e, 0, 0, sa);
					}
				}
				aStr(e, 0, RSP, gprOff(rd));
			}
			return PLAIN;
		case 0x04: case 0x06: case 0x07:
			if (rd) {
				aLdr(e, 1, RSP, gprOff(rs));
				aLdr(e, 0, RSP, gprOff(rt));
				aDp(e, f == 4 ? A_LSLV : f == 6 ? A_LSRV : A_ASRV, 0, 0, 1, S_LSL, 0);
				aStr(e, 0, RSP, gprOff(rd));
			}
			return PLAIN;
		case 0x08: case 0x09:
			aLdr(e, 0, RSP, gprOff(rs));
			aAndBits(e, 0, 0, 2, 10);
			aStr(e, 0, RSP, RSP_OFF(nextPc));
			if (f == 0x09) {
				storeConst(e, rd, pc + 8);
			}
			return BRANCH;
		case 0x0D:
			storePcs(e, pc, delay);
			*pcsStored = true;
			loadN32(e, GEN, N64_OFF(rspCodeGen));
			aMovX(e, 0, N);
			callC(e, rspBreak);
			*check = true;
			return PLAIN;
		case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25: case 0x26: case 0x27: {
			static const uint32_t alu[] = { A_ADD, A_ADD, A_SUB, A_SUB, A_AND, A_ORR, A_EOR, A_ORR };
			if (rd) {
				aLdr(e, 0, RSP, gprOff(rs));
				aLdr(e, 1, RSP, gprOff(rt));
				aDp(e, alu[f - 0x20], 0, 0, 1, S_LSL, 0);
				if (f == 0x27) {
					aMvn(e, 0, 0);
				}
				aStr(e, 0, RSP, gprOff(rd));
			}
			return PLAIN;
		}
		case 0x2A: case 0x2B:
			if (rd) {
				aLdr(e, 0, RSP, gprOff(rs));
				aLdr(e, 1, RSP, gprOff(rt));
				aDp(e, A_SUBS, ZR, 0, 1, S_LSL, 0);
				aCset(e, 0, f == 0x2A ? C_LT : C_LO);
				aStr(e, 0, RSP, gprOff(rd));
			}
			return PLAIN;
		}
		return PLAIN;
	}
	case 0x01:
		switch (rt) {
		case 0x00: case 0x01: case 0x10: case 0x11:
			aLdr(e, 0, RSP, gprOff(rs));
			aDpImm(e, A_SUBS_IMM, ZR, 0, 0);
			if (rt & 0x10) {
				storeConst(e, 31, pc + 8);
			}
			branchTo(e, rt & 1 ? C_GE : C_LT, pc, target);
			return BRANCH;
		}
		// Not a branch, but timed as one.
		aMovImm(e, 0, (pc + 8) & 0xFFC);
		aStr(e, 0, RSP, RSP_OFF(nextPc));
		return BRANCH;
	case 0x02: case 0x03:
		aMovImm(e, 0, ((op & 0x3FF) << 2) & 0xFFC);
		aStr(e, 0, RSP, RSP_OFF(nextPc));
		if (op >> 26 == 0x03) {
			storeConst(e, 31, pc + 8);
		}
		return BRANCH;
	case 0x04: case 0x05:
		aLdr(e, 0, RSP, gprOff(rs));
		aLdr(e, 1, RSP, gprOff(rt));
		aDp(e, A_SUBS, ZR, 0, 1, S_LSL, 0);
		branchTo(e, op >> 26 == 0x04 ? C_EQ : C_NE, pc, target);
		return BRANCH;
	case 0x06: case 0x07:
		aLdr(e, 0, RSP, gprOff(rs));
		aDpImm(e, A_SUBS_IMM, ZR, 0, 0);
		branchTo(e, op >> 26 == 0x06 ? C_LE : C_GT, pc, target);
		return BRANCH;
	case 0x08: case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x0E: {
		static const uint32_t alu[] = { A_ADD, A_ADD, 0, 0, A_AND, A_ORR, A_EOR };
		if (!rt) {
			return PLAIN;
		}
		int code = op >> 26;
		aLdr(e, 0, RSP, gprOff(rs));
		aMovImm(e, 1, code >= 0x0C ? (uint16_t) op : imm);
		if (code == 0x0A || code == 0x0B) {
			aDp(e, A_SUBS, ZR, 0, 1, S_LSL, 0);
			aCset(e, 0, code == 0x0A ? C_LT : C_LO);
		} else {
			aDp(e, alu[code - 0x08], 0, 0, 1, S_LSL, 0);
		}
		aStr(e, 0, RSP, gprOff(rt));
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
			loadN32(e, GEN, N64_OFF(rspCodeGen));
			aMovX(e, 0, N);
			aMovImm(e, 1, op);
			callC(e, rspCop0);
			loadTiming(e);
			*check = rs == 0x04;
		}
		return PLAIN;
	case 0x12:
		if ((op & (1 << 25)) && rspVectorFn(op)) {
			aMovX(e, 0, RSP);
			aAddImmX(e, 1, RSP, RSP_OFF(vr[(op >> 6) & 31]));
			aAddImmX(e, 2, RSP, RSP_OFF(vr[(op >> 11) & 31]));
			aAddImmX(e, 3, RSP, RSP_OFF(vr[(op >> 16) & 31]));
			aMovImm64(e, 4, (uint64_t) (uintptr_t) rspElementMask(op));
			callC(e, rspVectorFn(op));
			return PLAIN;
		}
		aMovX(e, 0, N);
		aMovImm(e, 1, op);
		callC(e, rspCop2);
		return PLAIN;
	case 0x20: case 0x21: case 0x23: case 0x24: case 0x25: case 0x27:
	case 0x28: case 0x29: case 0x2B:
		emitMemory(e, op);
		return PLAIN;
	case 0x32: case 0x3A:
		aMovX(e, 0, N);
		aMovImm(e, 1, op);
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
	aPush(&e, 29, 30, 80);
	aStp(&e, 19, 20, 16);
	aStp(&e, 21, 22, 32);
	aStp(&e, 23, 24, 48);
	aStp(&e, 25, 26, 64);
	aMovX(&e, N, 0);
	aAddImmX(&e, RSP, N, N64_OFF(rsp));
	aMovX(&e, LIMIT, 2);
	aAddImmX(&e, DMEM, N, N64_OFF(sp.mem));
	aBr(&e, 1);
	n->rspLeave = e.p;
	aLdp(&e, 19, 20, 16);
	aLdp(&e, 21, 22, 32);
	aLdp(&e, 23, 24, 48);
	aLdp(&e, 25, 26, 64);
	aPop(&e, 29, 30, 80);
	aRet(&e);
	__builtin___clear_cache((char*) p, (char*) e.p);
	n->rspEnter = (void (*)(struct N64*, void*, uint64_t)) p;
	return p;
}

void rspArenaFree(struct N64* n) {
	if (n->rspArena) {
		munmap(n->rspArena, ARENA_SIZE);
		n->rspArena = NULL;
	}
}

// Goes on to the block at sp.pc if the table has it for the current IMEM, or leaves.
static void chain(struct Jit* j) {
	struct Emitter* e = &j->e;
	aDp(e, A_SUBS | W64, ZR, CLOCK, LIMIT, S_LSL, 0);
	aBcond(e, C_HS, j->leave);
	loadN32(e, 0, N64_OFF(sp.pc));
	aAddImmX(e, 1, N, N64_OFF(rspTable));
	aDp(e, A_ADD | W64, 1, 1, 0, S_LSL, 2);
	aLdr(e, 2, 1, (unsigned) offsetof(struct RspJitEntry, gen));
	loadN32(e, 3, N64_OFF(rspCodeGen));
	aDp(e, A_SUBS, ZR, 2, 3, S_LSL, 0);
	aBcond(e, C_NE, j->leave);
	aLdrX(e, 4, 1, (unsigned) offsetof(struct RspJitEntry, code));
	aCbnzX(e, 4, e->p + 8);
	aB(e, j->leave);
	aBr(e, 4);
}

// Code for count instructions (as IMEM holds them) from start, or NULL when the arena is full.
void* rspCompile(struct N64* n, const uint8_t* ops, uint32_t count, uint32_t start) {
	uint8_t* arena = arenaGet(n);
	if (!arena || n->rspUsed + BLOCK_CODE_MAX > ARENA_SIZE) {
		return NULL;
	}
	static struct Jit jit;
	struct Jit* j = &jit;
	uint8_t* begin = arena + n->rspUsed;
	j->e = (struct Emitter) { begin, begin, begin + BLOCK_CODE_MAX };
	j->exitCount = 0;
	struct Emitter* e = &j->e;
	j->leave = e->p;
	aB(e, n->rspLeave);
	uint8_t* entry = e->p;

	bool delay = false;
	uint32_t prev = 0;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = be32(&ops[k * 4]), pc = start + k * 4;
		bool last = k + 1 == count;
		if (k == 0) {
			aMovX(e, 0, N);
			aMovImm(e, 1, op);
			aMovImm(e, 2, pc);
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
			loadN32(e, 0, N64_OFF(sp.status));
			halted = aTbnz(e, 0, 0, NULL);
			loadN32(e, 0, N64_OFF(rspCodeGen));
			aDp(e, A_SUBS, ZR, 0, GEN, S_LSL, 0);
			gen = aBcond(e, C_NE, NULL);
		}
		if (last) {
			exitAt(j, aB(e, NULL), op, pc, kind, delay);
			j->exits[j->exitCount - 1].last = true;
		} else {
			aDp(e, A_SUBS | W64, ZR, CLOCK, LIMIT, S_LSL, 0);
			exitAt(j, aBcond(e, C_HS, NULL), op, pc, kind, delay);
		}
		j->exits[j->exitCount - 1].pcsStored = pcsStored;
		if (check) {
			// Halted or IMEM rewritten: leave whatever the clock says.
			uint8_t* over = aB(e, NULL);
			patchHere(e, halted);
			patchHere(e, gen);
			exitAt(j, aB(e, NULL), op, pc, kind, delay);
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
			aMovImm(e, 0, (x->pc + 4) & 0xFFC);
			storeN32(e, 0, N64_OFF(sp.pc));
			if (x->kind == PLAIN) {
				aMovImm(e, 0, (x->pc + 8) & 0xFFC);
				aStr(e, 0, RSP, RSP_OFF(nextPc));
			}
		}
		aMovImm(e, 0, x->op);
		aStr(e, 0, RSP, RSP_OFF(lastOp));
		aMovImm(e, 0, x->pc);
		aStr(e, 0, RSP, RSP_OFF(lastPc));
		aMovImm(e, 0, x->kind == BRANCH);
		aStrb(e, 0, RSP, RSP_OFF(inDelay));
		storeTiming(e);
		if (x->last && !x->halt && x->kind == PLAIN) {
			chain(j);
		} else {
			aB(e, j->leave);
		}
	}
	if (e->p > e->end) {
		return NULL;
	}
	__builtin___clear_cache((char*) begin, (char*) e->p);
	n->rspUsed += (size_t) (e->p - begin + 15) & ~(size_t) 15;
	return entry;
}
