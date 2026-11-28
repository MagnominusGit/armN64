/* Copyright (c) 2026 Magnominus
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Licensed under the GNU General Public License, version 2 or (at your option) any later version: see LICENSE. */
#include "n64.h"
#include "x64/emit.h"

#include <stddef.h>
#include <string.h>
#include <sys/mman.h>

// x86-64 code for a block of cached RDRAM instructions. The ALU and branches run inline; the rest
// goes through cpuJitStep. Code stops after any instruction that takes the cycle count to the
// limit (the next event, or Count reaching Compare), leaving the CPU as the interpreter would.
//
// Registers: rbx = n, r12 = &n->cpu, r13 = n->cycles, r14 = the limit, r15 = RDRAM. A block that ends below the
// limit goes on to the next through n->jitTable without returning to C.

// The arena's first half has the shared code and the blocks' hot code (n->jitUsed), its second half
// their cold code: slow paths, misses and the exits but the block's end. How much of the second half
// is used is kept at the end of the shared code's space.
enum { ARENA_SIZE = 256 << 20, BLOCK_CODE_MAX = 32 << 10, SHARED_SIZE = 256 };

static uint64_t* coldUsed(struct N64* n) {
	return (uint64_t*) (n->jitArena + SHARED_SIZE - 8);
}

#define CPU_AT(f) M(R12, (int32_t) offsetof(struct CPU, f))
#define N64_AT(f) M(RBX, (int32_t) offsetof(struct N64, f))

// FLE keeps $sp in RBP; the C calls that can use it store it first and load it after.
static bool pinSp;

static struct Opnd gprMem(int i) {
	return M(R12, (int32_t) (offsetof(struct CPU, gpr) + 8 * i));
}

static struct Opnd gpr(int i) {
	return pinSp && i == 29 ? R(RBP) : gprMem(i);
}

static void storePinned(struct Emitter* e) {
	if (pinSp) {
		xStore(e, 8, gprMem(29), RBP);
	}
}

static void loadPinned(struct Emitter* e) {
	if (pinSp) {
		xLoad(e, 8, RBP, gprMem(29));
	}
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
	bool jumped;
	// The load mark the CPU leaves with.
	uint8_t mark;
	// A likely branch not taken: its load cycle (cpuLoadStall), or -1 to check the CPU's mark.
	int8_t stall;
};

// A load or store that left its inline path: for the helper, or for a D-cache line fill.
struct Slow {
	uint8_t* site[8];
	int sites;
	uint8_t* resume;
	uint32_t op, paddr;
	uint64_t pc;
	bool delay, lineStart, last;
	// Kept even with no site yet: a D-cache miss path may add one.
	bool keep;
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
	// A store (its vpages entry is the one for writes).
	bool store;
};

struct Jit {
	struct Emitter e;
	// The instruction being emitted: where its slow path (the interpreter) is entered from, the
	// cycles it holds the pipeline for, and whether it wrote Status (ecx is then nonzero if that
	// leaves an interrupt due).
	struct Slow* slow;
	int extraCycles;
	// The load cycle of the branch being compiled, for its exit when a likely one isn't taken.
	int8_t branchStall;
	// The load mark of the instruction being compiled (cpuLoadMark).
	uint8_t mark;
	bool interruptCheck;
	struct Exit exits[3 * 64];
	int exitCount;
	struct Slow slows[64];
	int slowCount;
	// A poll loop's block (pollSkip): its loads' addresses kept in CPU.pollAt from its start.
	bool poll;
	uint32_t pollBase;
	// Guest registers holding a 32-bit value known as the block runs (bit per register; r0 always),
	// and their values: what lui, addiu and ori made of known ones.
	uint32_t known;
	int32_t value[32];
	struct Miss misses[64];
	int missCount;
	// Fast CPU accesses outside kseg0's RDRAM, through vpages in the cold code.
	struct Miss mapped[64];
	int mappedCount;

	// Stores to pages with compiled code, which check that they miss its words.
	struct Miss codeChecks[64];
	int codeCheckCount;
	// Loads from pages the RDP may still be drawing to, which wait for it.
	struct Miss rdpWaits[64];
	int rdpWaitCount;
	// The instruction set pc and nextPc itself (ERET): the block leaves after it.
	bool jumped;
	uint8_t* leave;
	bool rdpThreads, rdpGl;
	// Outside mle the D-cache holds the data cached accesses use; with HLE accuracy (fast)
	// there are no caches.
	bool cacheData, fast, plainCount;
	// FLE (Fastest): cycles counted in runs, and no load interlocks, address checks or FPU result
	// fix-ups.
	bool fle;
	// The RDRAM the console has (ramSize, 4 MiB without the Expansion Pak): past it, the bus's.
	uint32_t ramSize;
	// Status.FR as the block was compiled.
	bool fr;
	int opCycles;
};

static void exitAt(struct Jit* j, uint8_t* site, uint64_t pc, enum Kind kind, bool delay, bool skip) {
	j->exits[j->exitCount++] = (struct Exit) { site, pc, kind, delay, skip, false, false, j->mark, skip ? j->branchStall : 0 };
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

static int memSize(uint32_t op);
static void emitMemory(struct Jit* j, struct Slow* slow);

// Leaves the instruction for the interpreter, before it has changed anything, when cc holds.
static void slowIf(struct Jit* j, int cc) {
	j->slow->site[j->slow->sites++] = xJcc(&j->e, cc, NULL);
}

// The interpreter raises "coprocessor unusable" when Status.CU1 is off.
static void needCop1(struct Jit* j) {
	xTestI(&j->e, 4, M(R12, (int32_t) (offsetof(struct CPU, cop0) + 8 * CP0_STATUS)), 1u << 29);
	slowIf(j, CC_E);
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

// The FPU's own code, called directly. Its stall counts. With HLE accuracy it raises nothing.
static void fpuCall(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int const rt = (op >> 16) & 31;
	bool const probe = fpuProbed(op) && !j->fast;
	storePinned(e);
	xMovRR(e, 8, RDI, RBX);
	xMovIKeepFlags(e, RSI, op);
	xMovI64(e, RAX, (uint64_t) (uintptr_t) (j->fast ? (void*) fpuExecuteFast : probe ? (void*) fpuProbe : (void*) fpuExecute));
	xCallR(e, RAX);
	loadPinned(e);
	if (probe) {
		xTest(e, 1, R(RAX), RAX);
		slowIf(j, CC_E);
	}
	xLoad(e, 4, RAX, CPU_AT(stall));
	xAluRR(e, X_ADD, 8, R(R13), RAX);
	xStoreI(e, 4, CPU_AT(stall), 0);
	// A move to r0 is undone, as the interpreter zeroes r0 after every instruction.
	if (((op >> 21) & 31) <= 2 && !rt) {
		xStoreI(e, 8, gpr(0), 0);
	}
}

// An SSE instruction on xmm registers 0-7: prefix (F3 single, F2 double, 66 or none), 0F, op.
static void xSse(struct Emitter* e, uint8_t prefix, uint8_t op, int xmm, struct Opnd rm, int width) {
	if (prefix) {
		e8(e, prefix);
	}
	uint8_t const code[2] = { 0x0F, op };
	emitOp(e, width, code, 2, xmm, rm);
}

static struct Opnd fprAt(int r) {
	return CPU_AT(fpr[r]);
}

// An FPU register as the block's Status.FR has it: with FR clear, an odd 32-bit register is the high
// half of the even one below, and a 64-bit one is the even one.
static struct Opnd fprS(const struct Jit* j, int r) {
	return j->fr || !(r & 1) ? fprAt(r) : M(R12, (int32_t) (offsetof(struct CPU, fpr) + 8 * (r & ~1) + 4));
}

static struct Opnd fprD(const struct Jit* j, int r) {
	return fprAt(j->fr ? r : r & ~1);
}

static struct Opnd fpr(const struct Jit* j, bool dbl, int r) {
	return dbl ? fprD(j, r) : fprS(j, r);
}

// To bad when the value in RAX (single, or double with dbl) is denormal, infinite or NaN; ECX/RCX
// keeps its exponent bits.
static void fpuNormal(struct Emitter* e, bool dbl, uint8_t** bad) {
	if (dbl) {
		xMovI64(e, RDX, 0x7FF0000000000000ull);
		xMovRR(e, 8, RCX, RAX);
		xAluRR(e, X_AND, 8, R(RCX), RDX);
		xAluRR(e, X_CMP, 8, R(RCX), RDX);
		bad[0] = xJcc(e, CC_E, NULL);
		xTest(e, 8, R(RCX), RCX);
		uint8_t* normal = xJcc(e, CC_NE, NULL);
		xMovRR(e, 8, RDX, RAX);
		xShiftI(e, X_SHL, 8, R(RDX), 1);
		bad[1] = xJcc(e, CC_NE, NULL);
		patchHere(e, normal);
	} else {
		xMovRR(e, 4, RCX, RAX);
		xAluI(e, X_AND, 4, R(RCX), 0x7F800000);
		xAluI(e, X_CMP, 4, R(RCX), 0x7F800000);
		bad[0] = xJcc(e, CC_E, NULL);
		xTest(e, 4, R(RCX), RCX);
		uint8_t* normal = xJcc(e, CC_NE, NULL);
		xTestI(e, 4, R(RAX), 0x7FFFFF);
		bad[1] = xJcc(e, CC_NE, NULL);
		patchHere(e, normal);
	}
}

// To early when the operand in RAX (exponent bits in RCX) makes the operation take 2 cycles
// (fpu.c early, trivialS/trivialD): zero, or for MUL a power of two.
static void fpuTrivial(struct Emitter* e, bool dbl, bool mul, uint8_t** early, int* earlyCount) {
	xTest(e, dbl ? 8 : 4, R(RCX), RCX);
	early[(*earlyCount)++] = xJcc(e, CC_E, NULL);
	if (mul) {
		if (dbl) {
			xMovRR(e, 8, RDX, RAX);
			xShiftI(e, X_SHL, 8, R(RDX), 12);
		} else {
			xMovRR(e, 4, RDX, RAX);
			xAluI(e, X_AND, 4, R(RDX), 0x7FFFFF);
		}
		early[(*earlyCount)++] = xJcc(e, CC_E, NULL);
	}
}

// A result in RAX (single, or double with dbl) stored as fpu_fast.c's setS and setD store it:
// flushed to zero when subnormal with FCR31.FS set, and the host's default NaN made the VR4300's
// (FLE: as it is).
static void fastResult(struct Emitter* e, bool dbl, struct Opnd dst, bool fle) {
	if (fle) {
		xStore(e, dbl ? 8 : 4, dst, RAX);
		return;
	}
	xTestI(e, 4, CPU_AT(fcr31), 1u << 24);
	uint8_t* keep = xJcc(e, CC_E, NULL);
	if (dbl) {
		xMovI64(e, RDX, 0x7FF0000000000000ull);
		xTest(e, 8, R(RAX), RDX);
		uint8_t* normal = xJcc(e, CC_NE, NULL);
		xMovI64(e, RDX, 0x8000000000000000ull);
		xAluRR(e, X_AND, 8, R(RAX), RDX);
		patchHere(e, normal);
		patchHere(e, keep);
		xMovI64(e, RSI, 0x7FF7FFFFFFFFFFFFull);
		xMovRR(e, 8, RCX, RAX);
		xShiftI(e, X_SHL, 8, R(RCX), 1);
		xMovI64(e, RDX, 0xFFF0000000000000ull);
		xAluRR(e, X_CMP, 8, R(RCX), RDX);
		xCmov(e, CC_E, 8, RAX, R(RSI));
		xStore(e, 8, dst, RAX);
	} else {
		xTestI(e, 4, R(RAX), 0x7F800000);
		uint8_t* normal = xJcc(e, CC_NE, NULL);
		xAluI(e, X_AND, 4, R(RAX), (int32_t) 0x80000000u);
		patchHere(e, normal);
		patchHere(e, keep);
		xMovI(e, RSI, 0x7FBFFFFF);
		xMovRR(e, 4, RCX, RAX);
		xAluI(e, X_AND, 4, R(RCX), 0x7FFFFFFF);
		xAluI(e, X_CMP, 4, R(RCX), 0x7FC00000);
		xCmov(e, CC_E, 4, RAX, R(RSI));
		xStore(e, 4, dst, RAX);
	}
}

// With HLE accuracy, the FPU instructions most code runs inline, as fpuExecuteFast runs them: the
// moves, the four operations, square root, ABS, MOV, NEG and C.cond in S and D, and the conversions
// between S, D and W, for the Status.FR the block was compiled with; anything else goes through
// fpuCall.
static void fpuFast(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int const fmt = (op >> 21) & 31, ft = (op >> 16) & 31, fs = (op >> 11) & 31, fd = (op >> 6) & 31, func = op & 0x3F;
	bool const dbl = fmt == 17;
	// An odd register means what Status.FR says: the block leaves when it isn't the one compiled for.
	if ((fs | fd | (fmt >= 16 ? ft : 0)) & 1) {
		xTestI(e, 4, M(R12, (int32_t) (offsetof(struct CPU, cop0) + 8 * CP0_STATUS)), 1u << 26);
		slowIf(j, j->fr ? CC_E : CC_NE);
	}
	switch (fmt) {
	case 0:
	case 1:
		if (ft) {
			if (fmt == 0) {
				xLoad(e, 4, RAX, fprS(j, fs));
				xMovsxd(e, RAX, R(RAX));
			} else {
				xLoad(e, 8, RAX, fprD(j, fs));
			}
			xStore(e, 8, gpr(ft), RAX);
		}
		return;
	case 2:
		if (ft) {
			if (fs == 31) {
				xLoad(e, 4, RAX, CPU_AT(fcr31));
				xMovsxd(e, RAX, R(RAX));
				xStore(e, 8, gpr(ft), RAX);
			} else {
				xStoreI(e, 8, gpr(ft), fs == 0 ? 0xA00 : 0);
			}
		}
		return;
	case 4:
	case 5:
		xLoad(e, fmt == 4 ? 4 : 8, RAX, gpr(ft));
		xStore(e, fmt == 4 ? 4 : 8, fmt == 4 ? fprS(j, fs) : fprD(j, fs), RAX);
		return;
	case 6:
		// CTC1 to FCR31 (to the others it does nothing), and the host's rounding mode when it changes, as
		// fpuHostRound sets it: RC in MXCSR is 4 minus the mode, in two bits.
		if (fs == 31) {
			xLoad(e, 4, RAX, gpr(ft));
			xAluI(e, X_AND, 4, R(RAX), 0x0183FFFF);
			xStore(e, 4, CPU_AT(fcr31), RAX);
			xAluI(e, X_AND, 4, R(RAX), 3);
			xAluRM(e, X_CMP, 4, RAX, CPU_AT(hostRound));
			uint8_t* same = xJcc(e, CC_E, NULL);
			xStore(e, 4, CPU_AT(hostRound), RAX);
			xMovI(e, RCX, 4);
			xAluRR(e, X_SUB, 4, R(RCX), RAX);
			xAluI(e, X_AND, 4, R(RCX), 3);
			xShiftI(e, X_SHL, 4, R(RCX), 13);
			xSse(e, 0, 0xAE, 3, CPU_AT(mxcsr), 4);
			xLoad(e, 4, RDX, CPU_AT(mxcsr));
			xAluI(e, X_AND, 4, R(RDX), (int32_t) ~0x6000u);
			xAluRR(e, X_OR, 4, R(RDX), RCX);
			xStore(e, 4, CPU_AT(mxcsr), RDX);
			xSse(e, 0, 0xAE, 2, CPU_AT(mxcsr), 4);
			patchHere(e, same);
		}
		return;
	}
	bool const known = ((fmt == 16 || fmt == 17) && (func <= 7 || func == 13 || func == (dbl ? 0x20 : 0x21) || func == 0x24 || func >= 0x30)) ||
	                   (fmt == 20 && (func == 32 || func == 33));
	if (!known) {
		fpuCall(j, op);
		return;
	}
	uint32_t const cycles = fpuCycles(fmt, func);
	if (cycles) {
		xAluI(e, X_ADD, 8, R(R13), (int32_t) cycles);
	}
	uint8_t const prefix = dbl ? 0xF2 : 0xF3;
	int const width = dbl ? 8 : 4;
	struct Opnd const out = M(RSP, -16);
	if (func >= 0x30) {
		// C.cond: less, equal and unordered as asked, into bit 23.
		xAluRR(e, X_XOR, 4, R(RAX), RAX);
		xAluRR(e, X_XOR, 4, R(RCX), RCX);
		xAluRR(e, X_XOR, 4, R(RDX), RDX);
		xSse(e, prefix, 0x10, 0, fpr(j, dbl, fs), 4);
		xSse(e, dbl ? 0x66 : 0, 0x2E, 0, fpr(j, dbl, ft), 4);
		xSetcc(e, CC_B, R(RAX));
		xSetcc(e, CC_NP, R(RCX));
		xSetcc(e, CC_E, R(RDX));
		xAluRR(e, X_XOR, 4, R(RSI), RSI);
		if (func & 4) {
			xAluRR(e, X_AND, 4, R(RAX), RCX);
			xAluRR(e, X_OR, 4, R(RSI), RAX);
		}
		if (func & 2) {
			xAluRR(e, X_AND, 4, R(RDX), RCX);
			xAluRR(e, X_OR, 4, R(RSI), RDX);
		}
		if (func & 1) {
			xAluI(e, X_XOR, 4, R(RCX), 1);
			xAluRR(e, X_OR, 4, R(RSI), RCX);
		}
		xShiftI(e, X_SHL, 4, R(RSI), 23);
		xLoad(e, 4, RAX, CPU_AT(fcr31));
		xAluI(e, X_AND, 4, R(RAX), (int32_t) ~(1u << 23));
		xAluRR(e, X_OR, 4, R(RAX), RSI);
		xStore(e, 4, CPU_AT(fcr31), RAX);
		return;
	}
	if (fmt == 20) {
		// CVT.S.W and CVT.D.W.
		xSse(e, func == 32 ? 0xF3 : 0xF2, 0x2A, 0, fprS(j, fs), 4);
		xSse(e, func == 32 ? 0xF3 : 0xF2, 0x11, 0, out, 4);
		xLoad(e, func == 32 ? 4 : 8, RAX, out);
		fastResult(e, func == 33, fpr(j, func == 33, fd), j->fle);
		return;
	}
	switch (func) {
	case 5:
	case 7:
		xLoad(e, width, RAX, fpr(j, dbl, fs));
		if (dbl) {
			xMovI64(e, RDX, func == 7 ? 0x8000000000000000ull : 0x7FFFFFFFFFFFFFFFull);
			xAluRR(e, func == 7 ? X_XOR : X_AND, 8, R(RAX), RDX);
		} else {
			xAluI(e, func == 7 ? X_XOR : X_AND, 4, R(RAX), func == 7 ? (int32_t) 0x80000000u : 0x7FFFFFFF);
		}
		fastResult(e, dbl, fpr(j, dbl, fd), j->fle);
		return;
	case 6:
		xLoad(e, width, RAX, fpr(j, dbl, fs));
		xStore(e, width, fpr(j, dbl, fd), RAX);
		return;
	case 13:
		// TRUNC.W: out of range or NaN gives the most negative word, as toWord and the host do.
		xSse(e, prefix, 0x2C, RAX, fpr(j, dbl, fs), 4);
		xStore(e, 4, fprS(j, fd), RAX);
		return;
	case 0x24:
		// CVT.W: rounded as FCR31 says, the host's rounding mode kept the same by CTC1.
		xSse(e, prefix, 0x2D, RAX, fpr(j, dbl, fs), 4);
		xStore(e, 4, fprS(j, fd), RAX);
		return;
	case 0x20:
	case 0x21:
		// CVT.S.D and CVT.D.S.
		xSse(e, prefix, 0x5A, 0, fpr(j, dbl, fs), 4);
		xSse(e, dbl ? 0xF3 : 0xF2, 0x11, 0, out, 4);
		xLoad(e, dbl ? 4 : 8, RAX, out);
		fastResult(e, !dbl, fpr(j, !dbl, fd), j->fle);
		return;
	}
	static const uint8_t ops[5] = { 0x58, 0x5C, 0x59, 0x5E, 0x51 };
	if (func == 4) {
		xSse(e, prefix, 0x51, 0, fpr(j, dbl, fs), 4);
	} else {
		xSse(e, prefix, 0x10, 0, fpr(j, dbl, fs), 4);
		xSse(e, prefix, ops[func], 0, fpr(j, dbl, ft), 4);
	}
	xSse(e, prefix, 0x11, 0, out, 4);
	xLoad(e, width, RAX, out);
	fastResult(e, dbl, fpr(j, dbl, fd), j->fle);
}

// MFC1, DMFC1, MTC1, DMTC1, MOV, NEG and ABS, the four operations and C.cond in S and D compiled
// for operands and results fpu.c takes no exception path on (normal or zero, the host raising no
// more than inexact, with inexact not enabled), and with the first operand's register the same in
// both FR modes; anything else goes through fpuCall.
static void fpuInline(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int const fmt = (op >> 21) & 31, ft = (op >> 16) & 31, fs = (op >> 11) & 31, fd = (op >> 6) & 31, func = op & 0x3F;
	bool const dbl = fmt == 17;
	if (fs & 1) {
		fpuCall(j, op);
		return;
	}
	if (fmt == 0 || fmt == 1) {
		if (ft) {
			if (fmt == 0) {
				xLoad(e, 4, RAX, fprAt(fs));
				xMovsxd(e, RAX, R(RAX));
			} else {
				xLoad(e, 8, RAX, fprAt(fs));
			}
			xStore(e, 8, gpr(ft), RAX);
		}
		return;
	}
	if (fmt == 4 || fmt == 5) {
		xLoad(e, fmt == 4 ? 4 : 8, RAX, gpr(ft));
		xStore(e, fmt == 4 ? 4 : 8, fprAt(fs), RAX);
		return;
	}
	bool const arith = func <= 3, compare = func >= 0x30, sign = func == 5 || func == 7;
	if ((fmt != 16 && fmt != 17) || !(arith || compare || sign || func == 6)) {
		fpuCall(j, op);
		return;
	}
	if (func == 6) {
		xLoad(e, 8, RAX, fprAt(fs));
		xStore(e, 8, fprAt(fd), RAX);
		return;
	}
	uint8_t* bad[12];
	int badCount = 0;
	uint8_t* early[4];
	int earlyCount = 0;
	int const width = dbl ? 8 : 4;
	if (compare) {
		// Only NaNs take another path; C.cond leaves no flags.
		for (int k = 0; k < 2; ++k) {
			xLoad(e, width, RAX, fprAt(k ? ft : fs));
			if (dbl) {
				xShiftI(e, X_SHL, 8, R(RAX), 1);
				xMovI64(e, RDX, 0xFFE0000000000000ull);
				xAluRR(e, X_CMP, 8, R(RAX), RDX);
			} else {
				xAluI(e, X_AND, 4, R(RAX), 0x7FFFFFFF);
				xAluI(e, X_CMP, 4, R(RAX), 0x7F800000);
			}
			bad[badCount++] = xJcc(e, CC_A, NULL);
		}
		xAluRR(e, X_XOR, 4, R(RDX), RDX);
		xSse(e, dbl ? 0xF2 : 0xF3, 0x10, 0, fprAt(fs), 4);
		xSse(e, dbl ? 0x66 : 0, 0x2E, 0, fprAt(ft), 4);
		// Less (CF) and equal (ZF) as asked, into bit 23.
		if (func & 4) {
			xSetcc(e, CC_B, R(RDX));
		}
		if (func & 2) {
			xSetcc(e, CC_E, R(RCX));
			xAluRR(e, X_OR, 1, R(RDX), RCX);
		}
		xShiftI(e, X_SHL, 4, R(RDX), 23);
		xLoad(e, 4, RAX, CPU_AT(fcr31));
		xAluI(e, X_AND, 4, R(RAX), (int32_t) ~(0x3F000u | 1u << 23));
		xAluRR(e, X_OR, 4, R(RAX), RDX);
		xStore(e, 4, CPU_AT(fcr31), RAX);
	} else if (sign) {
		xLoad(e, width, RAX, fprAt(fs));
		fpuNormal(e, dbl, &bad[badCount]);
		badCount += 2;
		if (dbl) {
			xMovI64(e, RDX, 0x8000000000000000ull);
			if (func == 7) {
				xAluRR(e, X_XOR, 8, R(RAX), RDX);
			} else {
				xNot(e, 8, R(RDX));
				xAluRR(e, X_AND, 8, R(RAX), RDX);
			}
		} else if (func == 7) {
			xAluI(e, X_XOR, 4, R(RAX), (int32_t) 0x80000000u);
		} else {
			xAluI(e, X_AND, 4, R(RAX), 0x7FFFFFFF);
		}
		xStore(e, 8, fprAt(fd), RAX);
		xAluI(e, X_AND, 4, CPU_AT(fcr31), (int32_t) ~0x3F000u);
	} else {
		for (int k = 0; k < 2; ++k) {
			xLoad(e, width, RAX, fprAt(k ? ft : fs));
			fpuNormal(e, dbl, &bad[badCount]);
			badCount += 2;
			fpuTrivial(e, dbl, func == 2, early, &earlyCount);
		}
		// Not early: the operation's cycles.
		uint32_t const cycles = fpuCycles(fmt, func);
		xMovI(e, RSI, cycles);
		uint8_t* stalled = xJmp(e, NULL);
		for (int i = 0; i < earlyCount; ++i) {
			patchHere(e, early[i]);
		}
		xMovI(e, RSI, cycles > 1 ? 1 : cycles);
		patchHere(e, stalled);
		// The host's flags cleared, the operation, its flags: anything but inexact goes the other way.
		struct Opnd const csr = M(RSP, -8), out = M(RSP, -16);
		xSse(e, 0, 0xAE, 3, csr, 4);
		xAluI(e, X_AND, 4, csr, ~0x3F);
		xSse(e, 0, 0xAE, 2, csr, 4);
		static const uint8_t ops[4] = { 0x58, 0x5C, 0x59, 0x5E };
		uint8_t const prefix = dbl ? 0xF2 : 0xF3;
		xSse(e, prefix, 0x10, 0, fprAt(fs), 4);
		xSse(e, prefix, ops[func], 0, fprAt(ft), 4);
		xSse(e, 0, 0xAE, 3, csr, 4);
		xLoad(e, 4, RDI, csr);
		xTestI(e, 4, R(RDI), 0x1F);
		bad[badCount++] = xJcc(e, CC_NE, NULL);
		// A denormal result (exact, so no underflow flag).
		xSse(e, prefix, 0x11, 0, out, 4);
		xLoad(e, width, RAX, out);
		fpuNormal(e, dbl, &bad[badCount]);
		badCount += 2;
		// Inexact: its cause, and its flag unless enabled (then the exception, the other way).
		xLoad(e, 4, RDX, CPU_AT(fcr31));
		xAluI(e, X_AND, 4, R(RDX), (int32_t) ~0x3F000u);
		xTestI(e, 4, R(RDI), 0x20);
		uint8_t* exact = xJcc(e, CC_E, NULL);
		xTestI(e, 4, R(RDX), 0x80);
		bad[badCount++] = xJcc(e, CC_NE, NULL);
		xAluI(e, X_OR, 4, R(RDX), 0x1004);
		patchHere(e, exact);
		xStore(e, 4, CPU_AT(fcr31), RDX);
		if (!dbl) {
			xMovRR(e, 4, RAX, RAX);
		}
		xStore(e, 8, fprAt(fd), RAX);
		xAluRR(e, X_ADD, 8, R(R13), RSI);
	}
	uint8_t* done = xJmp(e, NULL);
	for (int i = 0; i < badCount; ++i) {
		patchHere(e, bad[i]);
	}
	fpuCall(j, op);
	patchHere(e, done);
}

// MFC0 and DMFC0 of anything but Random and the registers that read the latch; MTC0 and DMTC0 of EPC
// and ErrorEPC.
// ecx = the interrupts due with Status in eax: none unless IE is set and EXL and ERL are clear.
static void pendingInterrupts(struct Emitter* e) {
	xLoad(e, 4, RCX, CPU_AT(cop0[CP0_CAUSE]));
	xAluRR(e, X_AND, 4, R(RCX), RAX);
	xAluI(e, X_AND, 4, R(RCX), 0xFF00);
	xAluRR(e, X_XOR, 4, R(RDX), RDX);
	xAluI(e, X_AND, 4, R(RAX), 7);
	xAluI(e, X_CMP, 4, R(RAX), 1);
	xCmov(e, CC_NE, 4, RCX, R(RDX));
}

static bool emitCop0(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, rd = (op >> 11) & 31;
	struct Opnd reg = M(R12, (int32_t) (offsetof(struct CPU, cop0) + 8 * rd));
	if (op == 0x42000018 && !j->slow->delay) {
		// ERET: to ErrorEPC with ERL cleared when it is set, else to EPC with EXL cleared.
		xLoad(e, 8, RAX, CPU_AT(cop0[CP0_STATUS]));
		xLoad(e, 8, RCX, CPU_AT(cop0[CP0_EPC]));
		xLoad(e, 8, RDX, CPU_AT(cop0[CP0_ERROREPC]));
		xMovI(e, RSI, ~2u);
		xMovI(e, RDI, ~4u);
		xTestI(e, 4, R(RAX), 4);
		xCmov(e, CC_NE, 8, RCX, R(RDX));
		xCmov(e, CC_NE, 8, RSI, R(RDI));
		xMovsxd(e, RSI, R(RSI));
		xAluRR(e, X_AND, 8, R(RAX), RSI);
		xStore(e, 8, CPU_AT(cop0[CP0_STATUS]), RAX);
		xStore(e, 8, CPU_AT(pc), RCX);
		xAluI(e, X_ADD, 8, R(RCX), 4);
		xStore(e, 8, CPU_AT(nextPc), RCX);
		xStoreI(e, 1, CPU_AT(llBit), 0);
		j->extraCycles = 2;
		j->jumped = true;
		return true;
	}
	if (rs == 0x00 || rs == 0x01) {
		// Random, and the unused registers that read the write latch, in C.
		if (rd == CP0_RANDOM || (rd >= 21 && rd <= 25) || rd == 31) {
			return false;
		}
		if (rt && rd == CP0_COUNT && !j->plainCount) {
			return false;
		}
		if (rt) {
			if (rd == CP0_COUNT) {
				// Count: half the cycles before this instruction, from countBase.
				xMovRR(e, 8, RAX, R13);
				xAluRM(e, X_SUB, 8, RAX, CPU_AT(countBase));
				xShiftI(e, X_SHR, 8, R(RAX), 1);
				if (rs == 0x00) {
					xMovsxd(e, RAX, R(RAX));
				} else {
					xMovRR(e, 4, RAX, RAX);
				}
			} else {
				xLoad(e, 8, RAX, reg);
				if (rs == 0x00) {
					xMovsxd(e, RAX, R(RAX));
				}
			}
			storeGpr(e, rt, RAX);
			// FLE: a loop polling Count lets time pass faster (cpu.c); a read long after the last
			// only notes when it was.
			if (rd == CP0_COUNT && j->fle) {
				xMovRR(e, 8, RAX, R13);
				xAluRM(e, X_SUB, 8, RAX, CPU_AT(countReadEnd));
				xStore(e, 8, CPU_AT(countReadEnd), R13);
				xAluI(e, X_CMP, 8, R(RAX), COUNT_POLL_WITHIN);
				uint8_t* late = xJcc(e, CC_AE, NULL);
				storePinned(e);
				xMovRR(e, 8, RDI, RBX);
				xMovRR(e, 8, RSI, R13);
				xMovI64(e, RAX, (uint64_t) (uintptr_t) cpuCountPolled);
				xCallR(e, RAX);
				loadPinned(e);
				xLoad(e, 4, RAX, CPU_AT(stall));
				xAluRR(e, X_ADD, 8, R(R13), RAX);
				xStoreI(e, 4, CPU_AT(stall), 0);
				patchHere(e, late);
			}
		}
		return true;
	}
	if (rs != 0x04 && rs != 0x05) {
		return false;
	}
	// Status: a write that changes the mode compiled code runs in (CU1, FR, RE, the address widths,
	// KSU, ERL or EXL) goes to the interpreter; after one that doesn't, ecx holds the interrupts it
	// leaves due.
	if (rd == CP0_STATUS) {
		xLoad(e, 4, RAX, gpr(rt));
		xAluI(e, X_AND, 4, R(RAX), (int32_t) 0xFFF7FFFFu);
		xLoad(e, 4, RCX, reg);
		xAluRR(e, X_XOR, 4, R(RCX), RAX);
		xTestI(e, 4, R(RCX), 1u << 29 | 1u << 26 | 1u << 25 | 0xFE);
		slowIf(j, CC_NE);
		xLoad(e, 8, RDX, gpr(rt));
		xStore(e, 8, M(R12, (int32_t) (offsetof(struct CPU, cop0) + 8 * CP0_LATCH)), RDX);
		xStore(e, 8, reg, RAX);
		pendingInterrupts(e);
		// One it leaves due is taken after the next instruction, as the interpreter has it.
		xTest(e, 4, R(RCX), RCX);
		xSetcc(e, CC_NE, CPU_AT(intHold));
		j->interruptCheck = true;
		return true;
	}
	// The others in C but EPC and ErrorEPC.
	if (rd != CP0_EPC && rd != CP0_ERROREPC) {
		return false;
	}
	// Each write also goes to the latch.
	xLoad(e, 8, RAX, gpr(rt));
	xStore(e, 8, M(R12, (int32_t) (offsetof(struct CPU, cop0) + 8 * CP0_LATCH)), RAX);
	xStore(e, 8, reg, RAX);
	return true;

	return false;
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
		case 0x18: case 0x19:
			// The products' low and high words, sign-extended; the multiplier holds the pipeline. MULT
			// takes rs whole and rt sign-extended from 35 bits, as the interpreter.
			if ((op & 0x3F) == 0x18) {
				xLoad(e, 8, RAX, gpr(rs));
				xLoad(e, 8, RCX, gpr(rt));
				xShiftI(e, X_SHL, 8, R(RCX), 29);
				xShiftI(e, X_SAR, 8, R(RCX), 29);
			} else {
				xLoad(e, 4, RAX, gpr(rs));
				xLoad(e, 4, RCX, gpr(rt));
			}
			xImul(e, 8, RAX, R(RCX));
			xMovsxd(e, RCX, R(RAX));
			xStore(e, 8, CPU_AT(lo), RCX);
			xShiftI(e, X_SAR, 8, R(RAX), 32);
			xMovsxd(e, RAX, R(RAX));
			xStore(e, 8, CPU_AT(hi), RAX);
			j->extraCycles = 4;
			return true;
		case 0x1A: case 0x1B:
			// The quotient and remainder, sign-extended; a divisor of 0 goes to the interpreter. DIV
			// takes the dividend's low word, and the divisor as 64 bits when its bits 63 and 31 agree
			// (as the interpreter), else its low word.
			if ((op & 0x3F) == 0x1A) {
				xLoad(e, 4, RAX, gpr(rs));
				xMovsxd(e, RAX, R(RAX));
				xLoad(e, 8, RCX, gpr(rt));
				xMovsxd(e, RDX, R(RCX));
				xMovRR(e, 8, RSI, RCX);
				xShiftI(e, X_SHL, 8, R(RSI), 32);
				xAluRR(e, X_XOR, 8, R(RSI), RCX);
				xCmov(e, CC_S, 8, RCX, R(RDX));
				xTest(e, 8, R(RCX), RCX);
				slowIf(j, CC_E);
				xMovRR(e, 8, RDX, RAX);
				xShiftI(e, X_SAR, 8, R(RDX), 63);
				emitOp1(e, 8, 0xF7, 7, R(RCX));
			} else {
				xLoad(e, 4, RAX, gpr(rs));
				xLoad(e, 4, RCX, gpr(rt));
				xTest(e, 4, R(RCX), RCX);
				slowIf(j, CC_E);
				xAluRR(e, X_XOR, 4, R(RDX), RDX);
				xDiv(e, 4, R(RCX));
			}
			xMovsxd(e, RAX, R(RAX));
			xStore(e, 8, CPU_AT(lo), RAX);
			xMovsxd(e, RDX, R(RDX));
			xStore(e, 8, CPU_AT(hi), RDX);
			j->extraCycles = 36;
			return true;
		case 0x20: case 0x22: case 0x2C: case 0x2E: {
			// Overflow raises an exception, through the interpreter, before rd changes.
			int width = (op & 0x3F) >= 0x2C ? 8 : 4;
			xLoad(e, width, RAX, gpr(rs));
			xAluRM(e, (op & 2) ? X_SUB : X_ADD, width, RAX, gpr(rt));
			slowIf(j, CC_O);
			if (width == 4) {
				xMovsxd(e, RAX, R(RAX));
			}
			storeGpr(e, rd, RAX);
			return true;
		}
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
	case 0x08: case 0x18: {
		int width = op >> 26 == 0x18 ? 8 : 4;
		xLoad(e, width, RAX, gpr(rs));
		xAluI(e, X_ADD, width, R(RAX), imm);
		slowIf(j, CC_O);
		if (width == 4) {
			xMovsxd(e, RAX, R(RAX));
		}
		storeGpr(e, rt, RAX);
		return true;
	}
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
		needCop1(j);
		if (!j->fast) {
			fpuInline(j, op);
		} else {
			fpuFast(j, op);
		}
		return true;
	case 0x20: case 0x21: case 0x23: case 0x24: case 0x25: case 0x27: case 0x37:
	case 0x28: case 0x29: case 0x2B: case 0x3F:
	case 0x31: case 0x35: case 0x39: case 0x3D:
		emitMemory(j, j->slow);
		return true;
	case 0x19:
		if (rt) {
			xLoad(e, 8, RAX, gpr(rs));
			xAluI(e, X_ADD, 8, R(RAX), imm);
			storeGpr(e, rt, RAX);
		}
		return true;
	case 0x2F:
		// CACHE on the D-cache: with HLE accuracy there are no caches, and an address in kseg0 or
		// kseg1 raises nothing; another goes to the interpreter for its TLB exceptions. As after any
		// instruction in C, an interrupt due is taken next.
		if (!j->fast || ((op >> 16) & 3) != 1) {
			return false;
		}
		xLoad(e, 8, RAX, gpr(rs));
		xAluI(e, X_ADD, 8, R(RAX), imm);
		xAluI(e, X_SUB, 8, R(RAX), INT32_MIN);
		xAluI(e, X_CMP, 8, R(RAX), 0x40000000);
		slowIf(j, CC_AE);
		xLoad(e, 4, RAX, CPU_AT(cop0[CP0_STATUS]));
		pendingInterrupts(e);
		j->interruptCheck = true;
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
		needCop1(j);
		xAluI(e, X_AND, 4, CPU_AT(fcr31), (int32_t) ~0x3F000u);
		xTestI(e, 4, CPU_AT(fcr31), 1u << 23);
		cc = rt & 1 ? CC_NE : CC_E;
	} else {
		cc = branchCondition(e, op, rs, rt);
	}
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

// The I-cache line holding pc, as fetch checks it; a miss fills it in C (ecx kept for an interrupt
// check).
static void icacheCheck(struct Emitter* e, uint32_t vaddr, uint32_t paddr, bool toStall) {
	struct Opnd tag = M(R12, (int32_t) (offsetof(struct CPU, icacheTag) + 4 * ((vaddr >> 5) & 511)));
	xAluI(e, X_CMP, 4, tag, (int32_t) (paddr & ~31u));
	uint8_t* hit = xJcc(e, CC_E, NULL);
	xPush(e, RAX);
	xPush(e, RCX);
	xMovRR(e, 8, RDI, RBX);
	xMovI(e, RSI, vaddr);
	xMovI(e, RDX, paddr);
	xMovI64(e, RAX, (uint64_t) (uintptr_t) cpuIcacheMiss);
	xCallR(e, RAX);
	if (toStall) {
		xAluRR(e, X_ADD, 4, CPU_AT(stall), RAX);
	} else {
		xAluRR(e, X_ADD, 8, R(R13), RAX);
	}
	xPop(e, RCX);
	xPop(e, RAX);
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
		struct Opnd tag = M(R12, (int32_t) (offsetof(struct CPU, icacheTag) + 4 * ((v >> 5) & 511)));
		xAluI(e, X_CMP, 4, tag, (int32_t) (p & ~31u));
		skip[sites++] = xJcc(e, CC_NE, NULL);
	}
	xMovRR(e, 8, RAX, R14);
	xAluRR(e, X_SUB, 8, R(RAX), R13);
	xAluI(e, X_CMP, 8, R(RAX), 1);
	skip[sites++] = xJcc(e, CC_LE, NULL);
	xAluI(e, X_SUB, 8, R(RAX), 1);
	xShiftI(e, X_SHR, 8, R(RAX), opCycles == 4 ? 3 : 1);
	xLea(e, 8, R13, MI(R13, RAX, opCycles == 4 ? 8 : 2, 0));
	for (int i = 0; i < sites; ++i) {
		patchHere(e, skip[i]);
	}
}

// An idle loop over two blocks (blockIdlePair), with the fast CPU: while the other half still jumps
// back, whole times round both until the next event.
static void skipPair(struct Emitter* e, uint32_t vaddr, uint32_t target, int opCycles) {
	uint8_t* skip[3];
	int sites = 0;
	uint32_t const back = 0x08000000 | ((vaddr >> 2) & 0x03FFFFFF);
	xAluI(e, X_CMP, 4, M(R15, (int32_t) (target & 0x7FFFFF)), (int32_t) __builtin_bswap32(back));
	skip[sites++] = xJcc(e, CC_NE, NULL);
	xAluI(e, X_CMP, 4, M(R15, (int32_t) ((target & 0x7FFFFF) + 4)), 0);
	skip[sites++] = xJcc(e, CC_NE, NULL);
	int const shift = opCycles == 4 ? 4 : opCycles == 2 ? 3 : 2;
	xMovRR(e, 8, RAX, R14);
	xAluRR(e, X_SUB, 8, R(RAX), R13);
	xAluI(e, X_CMP, 8, R(RAX), 1 << shift);
	skip[sites++] = xJcc(e, CC_LE, NULL);
	xAluI(e, X_SUB, 8, R(RAX), 1);
	xShiftI(e, X_SHR, 8, R(RAX), shift);
	xShiftI(e, X_SHL, 8, R(RAX), shift);
	xAluRR(e, X_ADD, 8, R(R13), RAX);
	for (int i = 0; i < sites; ++i) {
		patchHere(e, skip[i]);
	}
}

// A count loop (blockCountLoop) run whole times round until the next event, as skipIdle runs the idle
// loop: the cycles of each two instructions, and the register's add times as many.
static void skipCount(struct Emitter* e, uint32_t vaddr, uint32_t paddr, bool fast, int opCycles, int reg, int32_t imm, bool dword) {
	uint8_t* skip[3];
	int sites = 0;
	int lines = fast ? 0 : (paddr + 4) & 31 ? 1 : 2;
	for (int i = 0; i < lines; ++i) {
		uint32_t v = vaddr + 4 * i, p = paddr + 4 * i;
		struct Opnd tag = M(R12, (int32_t) (offsetof(struct CPU, icacheTag) + 4 * ((v >> 5) & 511)));
		xAluI(e, X_CMP, 4, tag, (int32_t) (p & ~31u));
		skip[sites++] = xJcc(e, CC_NE, NULL);
	}
	int const shift = opCycles == 4 ? 3 : opCycles == 2 ? 2 : 1;
	xMovRR(e, 8, RAX, R14);
	xAluRR(e, X_SUB, 8, R(RAX), R13);
	xAluI(e, X_CMP, 8, R(RAX), 1 << shift);
	skip[sites++] = xJcc(e, CC_LE, NULL);
	xAluI(e, X_SUB, 8, R(RAX), 1);
	xShiftI(e, X_SHR, 8, R(RAX), shift);
	xMovRR(e, 8, RCX, RAX);
	xShiftI(e, X_SHL, 8, R(RCX), shift);
	xAluRR(e, X_ADD, 8, R(R13), RCX);
	movConst(e, RDX, (uint64_t) (int64_t) imm);
	xImul(e, 8, RAX, R(RDX));
	xLoad(e, 8, RCX, gpr(reg));
	xAluRR(e, X_ADD, 8, R(RCX), RAX);
	if (!dword) {
		xMovsxd(e, RCX, R(RCX));
	}
	xStore(e, 8, gpr(reg), RCX);
	for (int i = 0; i < sites; ++i) {
		patchHere(e, skip[i]);
	}
}

// The known registers after op: what it makes of known ones, and none it writes otherwise (all of
// them after an instruction not decoded here).
static void trackConst(struct Jit* j, uint32_t op) {
	int const rs = (op >> 21) & 31, rt = (op >> 16) & 31, rd = (op >> 11) & 31, code = op >> 26;
	uint32_t const imm = op & 0xFFFF;
	uint32_t written = 0;
	switch (code) {
	case 0x00:
		written = 1u << rd;
		break;
	case 0x01:
	case 0x03:
		written = 1u << 31;
		break;
	case 0x02: case 0x04: case 0x05: case 0x06: case 0x07: case 0x14: case 0x15: case 0x16: case 0x17:
	case 0x28: case 0x29: case 0x2A: case 0x2B: case 0x2C: case 0x2D: case 0x2E: case 0x2F: case 0x3F:
	case 0x31: case 0x35: case 0x39: case 0x3D:
		break;
	case 0x09: case 0x0D: case 0x0F: {
		bool const from = code == 0x0F || (j->known >> rs & 1);
		int32_t const v = code == 0x0F ? (int32_t) (imm << 16) : code == 0x09 ? (int32_t) ((uint32_t) j->value[rs] + (uint32_t) (int16_t) imm) : (int32_t) ((uint32_t) j->value[rs] | imm);
		if (rt) {
			j->known = from ? j->known | 1u << rt : j->known & ~(1u << rt);
			j->value[rt] = v;
		}
		return;
	}
	case 0x08: case 0x0A: case 0x0B: case 0x0C: case 0x0E: case 0x10: case 0x11: case 0x12: case 0x18: case 0x19:
	case 0x1A: case 0x1B: case 0x20: case 0x21: case 0x22: case 0x23: case 0x24: case 0x25: case 0x26: case 0x27:
	case 0x30: case 0x34: case 0x37: case 0x38: case 0x3C:
		written = 1u << rt;
		break;
	default:
		written = ~1u;
		break;
	}
	j->known &= ~written | 1u;
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
// A block's first instruction: the load cycle from the mark the CPU comes in with (cpuLoadStall).
static void loadStallCheck(struct Emitter* e, uint32_t op) {
	uint8_t marks[2];
	int const count = cpuStallMarks(op, marks);
	if (!count) {
		return;
	}
	uint8_t* stall[2];
	for (int i = 0; i < count; ++i) {
		xAluI(e, X_CMP, 1, CPU_AT(loadMark), (int8_t) marks[i]);
		stall[i] = xJcc(e, CC_E, NULL);
	}
	uint8_t* none = xJmp(e, NULL);
	for (int i = 0; i < count; ++i) {
		patchHere(e, stall[i]);
	}
	xAluI(e, X_ADD, 8, R(R13), 1);
	patchHere(e, none);
}

static void emitHelper(struct Jit* j, uint32_t op, uint64_t pc, uint32_t paddr, bool delay, bool lineStart, bool last, int prevMark, bool nextStall) {
	struct Emitter* e = &j->e;
	// The interpreter takes the load cycle (cpuLoadStall) from the mark before: the CPU's own at the
	// block's start, else the previous instruction's.
	if (prevMark >= 0) {
		xStoreI(e, 1, CPU_AT(loadMark), prevMark);
	}
	if (lineStart) {
		if (!j->fast) {
			icacheCheck(e, (uint32_t) pc, paddr, true);
		}
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
	storePinned(e);
	xMovRR(e, 8, RDI, RBX);
	xMovIKeepFlags(e, RSI, op);
	xMovI64(e, RAX, (uint64_t) (uintptr_t) cpuJitStep);
	xCallR(e, RAX);
	loadPinned(e);
	xLoad(e, 8, R13, N64_AT(cycles));
	xLoad(e, 8, R14, N64_AT(jitLimit));
	xTest(e, 1, R(RAX), RAX);
	xJcc(e, CC_NE, j->leave);
	xAluRR(e, X_CMP, 8, R(R13), R14);
	xJcc(e, CC_AE, j->leave);
	if (nextStall) {
		// An uncached load (its mark cleared) leaves no load cycle for the next instruction, which
		// counts one.
		xAluI(e, X_CMP, 1, CPU_AT(loadMark), 0);
		uint8_t* cached = xJcc(e, CC_NE, NULL);
		xAluI(e, X_SUB, 8, R(R13), 1);
		patchHere(e, cached);
	}
	if (last) {
		xAluI(e, X_CMP, 1, CPU_AT(nextInDelay), 0);
		xJcc(e, CC_NE, j->leave);
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

// rsi = the FPR an FPU load or store of size bytes uses, as get32/set32 and get64/set64 find it:
// with Status.FR clear, odd 32-bit registers are the high half of the even one below.
static void fprAddress(struct Emitter* e, int ft, int size) {
	int32_t own = (int32_t) (offsetof(struct CPU, fpr) + 8 * ft);
	if (!(ft & 1)) {
		xLea(e, 8, RSI, M(R12, own));
		return;
	}
	int32_t pair = (int32_t) (offsetof(struct CPU, fpr) + 8 * (ft & ~1)) + (size == 4 ? 4 : 0);
	xLea(e, 8, RSI, M(R12, own));
	xLea(e, 8, RDI, M(R12, pair));
	xTestI(e, 4, M(R12, (int32_t) (offsetof(struct CPU, cop0) + 8 * CP0_STATUS)), 1u << 26);
	xCmov(e, CC_E, 8, RSI, R(RDI));
}

// A load or store to cached RDRAM, through kseg0 or a page the TLB mapped (vpages), with
// dataTiming's D-cache check. Anything else jumps to the slow path before any change: a page not
// looked up, and a misaligned address, fail the tag check and leave from the miss path. eax is the
// physical address and r8d the virtual one.
// cpuRdpWaitRead or cpuRdpWaitWrite for the physical address in eax (the virtual one in r8 kept), at
// the cycle the code has reached (the RSP may catch up to it).
static void rdpWaitCall(struct Emitter* e, const void* f, int size) {
	xStore(e, 8, N64_AT(cycles), R13);
	xPush(e, RAX);
	xPush(e, R8);
	xMovRR(e, 8, RDI, RBX);
	xMovRR(e, 4, RSI, RAX);
	xMovI(e, RDX, (uint32_t) size);
	xMovI64(e, RAX, (uint64_t) (uintptr_t) f);
	xCallR(e, RAX);
	xPop(e, R8);
	xPop(e, RAX);
}

static void emitMemory(struct Jit* j, struct Slow* slow) {
	struct Emitter* e = &j->e;
	uint32_t op = slow->op;
	int rs = (op >> 21) & 31, rt = (op >> 16) & 31, size = memSize(op);
	int code = op >> 26;
	bool fpu = code == 0x31 || code == 0x35 || code == 0x39 || code == 0x3D;
	bool isStore = code >= 0x28 && code != 0x37 && code != 0x31 && code != 0x35;
	struct Opnd ram = j->cacheData ? MI(R12, RCX, 1, (int32_t) offsetof(struct CPU, dcacheData)) : MI(R15, RAX, 1, 0);
	if (fpu) {
		needCop1(j);
	}

	// With the fast CPU, an address known now (a constant base) that is kseg0's RDRAM, aligned, needs
	// no checks.
	uint32_t const at = (uint32_t) j->value[rs] + (uint32_t) (int16_t) op;
	bool const direct = j->fast && (j->known >> rs & 1) && (at ^ 0x80000000u) < j->ramSize && !(at & (uint32_t) (size - 1));
	if (direct) {
		xMovI(e, R8, at);
		xMovI(e, RAX, at ^ 0x80000000u);
	} else {
	// The address in 64 bits, which the interpreter takes when it isn't a 32-bit one sign-extended (an
	// address error); the low 32 bits from here on.
	xLoad(e, 8, RAX, gpr(rs));
	if ((int16_t) op) {
		xAluI(e, X_ADD, 8, R(RAX), (int16_t) op);
	}
	if (!j->fle) {
		xMovsxd(e, RCX, R(RAX));
		xAluRR(e, X_CMP, 8, R(RCX), RAX);
		slow->site[slow->sites++] = xJcc(e, CC_NE, NULL);
	}
	xMovRR(e, 4, R8, RAX);
	if (j->fast) {
		// kseg0's RDRAM (and aligned, but with FLE) in one test; anything else through vpages in the
		// cold code.
		xAluI(e, X_XOR, 4, R(RAX), (int32_t) 0x80000000u);
		xTestI(e, 4, R(RAX), ~(j->ramSize - 1) | (j->fle ? 0 : (uint32_t) (size - 1)));
		struct Miss* v = &j->mapped[j->mappedCount++];
		v->site = xJcc(e, CC_NE, NULL);
		v->resume = e->p;
		v->slow = slow;
		v->size = size;
		v->store = isStore;
		slow->keep = true;
	} else {
		// The physical page from vpages: one not looked up lands past RDRAM.
		xMovRR(e, 4, RCX, RAX);
		xShiftI(e, X_SHR, 4, R(RCX), 12);
		xLoad(e, 8, RDX, N64_AT(vpages));
		xAluRM(e, X_ADD, 4, RAX, MI(RDX, RCX, 4, isStore ? VPAGES * 4 : 0));
	}
	}
	if (j->poll && !isStore && (uint32_t) (slow->pc - j->pollBase) < 64) {
		uint32_t const k = (uint32_t) (slow->pc - j->pollBase) / 4;
		xStore(e, 4, CPU_AT(pollAt[k]), RAX);
		xStore(e, 4, CPU_AT(pollVirt[k]), R8);
	}
	// Stores go slow on any flag (code, RDP) unless the page has only code that they miss; loads
	// on pages the RDP may still be drawing to. With the D-cache's data only its fills and
	// write-backs reach memory, in C.
	if (!j->cacheData && (isStore || j->rdpThreads)) {
		xMovRR(e, 4, RCX, RAX);
		xShiftI(e, X_SHR, 4, R(RCX), 12);
		if (isStore) {
			xStoreI(e, 1, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, dirtyPages)), 0);
			xAluI(e, X_CMP, 1, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, pageFlags)), 0);
			struct Miss* k = &j->codeChecks[j->codeCheckCount++];
			k->site = xJcc(e, CC_NE, NULL);
			k->resume = e->p;
			k->slow = slow;
			k->size = size;
			slow->keep = true;
		} else {
			xTestI(e, 1, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, pageFlags)), PAGE_RDP_WRITES);
			struct Miss* w = &j->rdpWaits[j->rdpWaitCount++];
			w->site = xJcc(e, CC_NE, NULL);
			w->resume = e->p;
			w->size = size;
		}
	}

	if (j->fast) {
		// Aligned and in RDRAM: tested on the way here.
	} else {
	// The line index from the virtual address; a hit when its tag is this physical line and the
	// address is aligned.
	xMovRR(e, 4, RCX, R8);
	xShiftI(e, X_SHR, 4, R(RCX), 4);
	xAluI(e, X_AND, 4, R(RCX), 511);
	xLoad(e, 4, RDX, MI(R12, RCX, 4, (int32_t) offsetof(struct CPU, dcacheTag)));
	xAluRR(e, X_XOR, 4, R(RDX), RAX);
	xTestI(e, 4, R(RDX), 0xFFFFFFF0u | (uint32_t) (size - 1));
	struct Miss* m = &j->misses[j->missCount++];
	m->site = xJcc(e, CC_NE, NULL);
	m->resume = e->p;
	m->slow = slow;
	m->size = size;
	m->glLoad = j->rdpGl && !isStore;
	slow->keep = true;
	if (isStore) {
		xStoreI(e, 1, MI(R12, RCX, 1, (int32_t) offsetof(struct CPU, dcacheDirty)), 1);
	}
	}
	if (j->cacheData) {
		xMovRR(e, 4, RCX, R8);
		xAluI(e, X_AND, 4, R(RCX), 0x1FFF);
	}

	switch (op >> 26) {
	case 0x20:
		xMovsx(e, RDX, 1, ram);
		xMovsxd(e, RDX, R(RDX));
		break;
	case 0x24:
		xMovzx(e, RDX, 1, ram);
		break;
	case 0x21: case 0x25:
		xMovzx(e, RDX, 2, ram);
		xBswap(e, 4, RDX);
		xShiftI(e, op >> 26 == 0x21 ? X_SAR : X_SHR, 4, R(RDX), 16);
		if (op >> 26 == 0x21) {
			xMovsxd(e, RDX, R(RDX));
		}
		break;
	case 0x23: case 0x27:
		xLoad(e, 4, RDX, ram);
		xBswap(e, 4, RDX);
		if (op >> 26 == 0x23) {
			xMovsxd(e, RDX, R(RDX));
		}
		break;
	case 0x37:
		xLoad(e, 8, RDX, ram);
		xBswap(e, 8, RDX);
		break;
	case 0x28:
		xLoad(e, 1, RDX, gpr(rt));
		xStore(e, 1, ram, RDX);
		break;
	case 0x29:
		xMovzx(e, RDX, 2, gpr(rt));
		xShiftI(e, X_ROL, 2, R(RDX), 8);
		xStore(e, 2, ram, RDX);
		break;
	case 0x2B:
		xLoad(e, 4, RDX, gpr(rt));
		xBswap(e, 4, RDX);
		xStore(e, 4, ram, RDX);
		break;
	case 0x3F:
		xLoad(e, 8, RDX, gpr(rt));
		xBswap(e, 8, RDX);
		xStore(e, 8, ram, RDX);
		break;
	case 0x31: case 0x35:
		xLoad(e, size, RDX, ram);
		xBswap(e, size, RDX);
		fprAddress(e, rt, size);
		xStore(e, size, M(RSI, 0), RDX);
		return;
	case 0x39: case 0x3D:
		fprAddress(e, rt, size);
		xLoad(e, size, RDX, M(RSI, 0));
		xBswap(e, size, RDX);
		xStore(e, size, ram, RDX);
		return;
	}
	if (!isStore) {
		storeGpr(e, rt, RDX);
	}
}

// Goes on to the block at the pc in memory if the jump table has it, or leaves.
// A poll loop (blockPollPass) about to go round again, pc back at its start, skips the whole passes
// of pass cycles that fit before the limit when all its I-cache and D-cache lines are in: every pass
// then reads the same memory, takes the same branch and costs the same.
static void pollSkip(struct Jit* j, const uint32_t* ops, uint32_t count, uint32_t vaddr, uint32_t paddr, uint32_t pass, int reload) {
	struct Emitter* e = &j->e;
	uint8_t* out[64];
	int sites = 0;
	xLoad(e, 8, RAX, CPU_AT(pc));
	movConst(e, RCX, sext32(vaddr));
	xAluRR(e, X_CMP, 8, R(RAX), RCX);
	out[sites++] = xJcc(e, CC_NE, NULL);
	if (reload) {
		xLoad(e, 8, RDX, gpr(reload & 31));
		xAluI(e, X_CMP, 8, R(RDX), 0);
		out[sites++] = xJcc(e, reload & 32 ? CC_NE : CC_E, NULL);
	}
	for (uint32_t p = paddr & ~31u; p < paddr + 4 * count && !j->fast; p += 32) {
		uint32_t v = vaddr + (p - paddr);
		struct Opnd tag = M(R12, (int32_t) (offsetof(struct CPU, icacheTag) + 4 * ((v >> 5) & 511)));
		xAluI(e, X_CMP, 4, tag, (int32_t) (p & ~31u));
		out[sites++] = xJcc(e, CC_NE, NULL);
	}
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k];
		if (op >> 26 < 0x20) {
			continue;
		}
		int size = memSize(op);
		if (j->poll) {
			// Where the pass just made took it from (its base may be what an earlier load in it read).
			xLoad(e, 4, RAX, CPU_AT(pollAt[k]));
			xLoad(e, 4, R8, CPU_AT(pollVirt[k]));
		} else {
			xLoad(e, 4, RAX, gpr((op >> 21) & 31));
			xAluI(e, X_ADD, 4, R(RAX), (int16_t) op);
			xMovRR(e, 4, R8, RAX);
			xMovRR(e, 4, RCX, RAX);
			xShiftI(e, X_SHR, 4, R(RCX), 12);
			xLoad(e, 8, RDX, N64_AT(vpages));
			xAluRM(e, X_ADD, 4, RAX, MI(RDX, RCX, 4, 0));
		}
		if (j->rdpThreads) {
			xMovRR(e, 4, RCX, RAX);
			xShiftI(e, X_SHR, 4, R(RCX), 12);
			xTestI(e, 1, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, pageFlags)), PAGE_RDP_WRITES);
			out[sites++] = xJcc(e, CC_NE, NULL);
		}
		if (j->fast) {
			xTestI(e, 4, R(RAX), ~(j->ramSize - 1) | (uint32_t) (size - 1));
			out[sites++] = xJcc(e, CC_NE, NULL);
			continue;
		}
		xMovRR(e, 4, RCX, R8);
		xShiftI(e, X_SHR, 4, R(RCX), 4);
		xAluI(e, X_AND, 4, R(RCX), 511);
		xLoad(e, 4, RDX, MI(R12, RCX, 4, (int32_t) offsetof(struct CPU, dcacheTag)));
		xAluRR(e, X_XOR, 4, R(RDX), RAX);
		xTestI(e, 4, R(RDX), 0xFFFFFFF0u | (uint32_t) (size - 1));
		out[sites++] = xJcc(e, CC_NE, NULL);
	}
	xMovRR(e, 8, RAX, R14);
	xAluRR(e, X_SUB, 8, R(RAX), R13);
	xAluI(e, X_SUB, 4, R(RAX), 1);
	xAluRR(e, X_XOR, 4, R(RDX), RDX);
	xMovI(e, RCX, pass);
	xDiv(e, 4, R(RCX));
	xImul(e, 4, RAX, R(RCX));
	xAluRR(e, X_ADD, 8, R(R13), RAX);
	for (int i = 0; i < sites; ++i) {
		patchHere(e, out[i]);
	}
}

// To the block at the pc through the jitTable, else to leave.
static void lookup(struct Emitter* e, uint8_t* leave) {
	xLoad(e, 8, RAX, CPU_AT(pc));
	xMovsxd(e, RCX, R(RAX));
	xAluRR(e, X_CMP, 8, R(RCX), RAX);
	xJcc(e, CC_NE, leave);
	xMovRR(e, 4, RCX, RAX);
	xShiftI(e, X_SHR, 4, R(RCX), 2);
	xAluI(e, X_AND, 4, R(RCX), JIT_TABLE_SIZE - 1);
	xShiftI(e, X_SHL, 4, R(RCX), 4);
	xLea(e, 8, RDX, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, jitTable)));
	xAluRM(e, X_CMP, 4, RAX, M(RDX, (int32_t) offsetof(struct JitEntry, vaddr)));
	xJcc(e, CC_NE, leave);
	// Good while its page maps where it was compiled (jitTag): the address through vpages.
	xLoad(e, 8, R8, N64_AT(vpages));
	xMovRR(e, 4, RCX, RAX);
	xShiftI(e, X_SHR, 4, R(RCX), 12);
	xAluRM(e, X_ADD, 4, RAX, MI(R8, RCX, 4, 0));
	xAluRM(e, X_CMP, 4, RAX, M(RDX, (int32_t) offsetof(struct JitEntry, gen)));
	xJcc(e, CC_NE, leave);
	xJmpM(e, M(RDX, (int32_t) offsetof(struct JitEntry, code)));
}

static void chain(struct Jit* j) {
	lookup(&j->e, j->leave);
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
		xLoad(&e, 8, R15, M(RBX, (int32_t) offsetof(struct N64, rdram)));
		pinSp = n->fast && n->fle;
		if (pinSp) {
			xPush(&e, RBP);
			xAluI(&e, X_SUB, 8, R(RSP), 8);
			loadPinned(&e);
		}
		xJmpR(&e, RSI);
		n->jitLeave = e.p;
		if (pinSp) {
			storePinned(&e);
			xAluI(&e, X_ADD, 8, R(RSP), 8);
			xPop(&e, RBP);
		}
		xStore(&e, 8, M(RBX, (int32_t) offsetof(struct N64, cycles)), R13);
		xPop(&e, R15);
		xPop(&e, R14);
		xPop(&e, R13);
		xPop(&e, R12);
		xPop(&e, RBX);
		xRet(&e);
		n->jitRedispatch = e.p;
		lookup(&e, n->jitLeave);
		n->jitEnter = (void (*)(struct N64*, void*)) p;
	}
	return n->jitArena;
}

void jitReset(struct N64* n) {
	n->jitUsed = SHARED_SIZE;
	if (n->jitArena) {
		*coldUsed(n) = 0;
	}
	++n->blockGen;
}

void jitUnlink(struct N64* n) {
	n->linkGen = n->blockGen;
}

// A thrown-away block's code starts again at jitRedispatch (any jump to it comes with the pc set).
void jitKill(struct N64* n, void* code) {
	uint8_t* p = code;
	int32_t d = (int32_t) ((uint8_t*) n->jitRedispatch - (p + 5));
	p[0] = 0xE9;
	memcpy(p + 1, &d, 4);
}

void jitFree(struct N64* n) {
	if (n->jitArena) {
		munmap(n->jitArena, ARENA_SIZE);
		n->jitArena = NULL;
	}
	n->jitUsed = SHARED_SIZE;
}

const bool jitBranchesInside = false;

// Leaving after an inline instruction: the state the interpreter would have after it.
static void emitExit(struct Jit* j, const struct Exit* x, const uint32_t* ops, uint32_t count, uint32_t vaddr, uint32_t paddr,
                     uint32_t pollPass, int reload) {
	struct Emitter* e = &j->e;
	if (x->site) {
		patchHere(e, x->site);
	}
	if (!j->fle) {
		if (x->stall < 0) {
			loadStallCheck(e, ops[((uint32_t) x->pc - vaddr) >> 2]);
		} else if (x->stall) {
			xAluI(e, X_ADD, 8, R(R13), x->stall);
		}
		xStoreI(e, 1, CPU_AT(loadMark), x->mark);
	}
	storeConst(e, CPU_AT(curPc), x->pc);
	xStoreI(e, 1, CPU_AT(inDelay), x->delay);
	xStoreI(e, 1, CPU_AT(nextInDelay), x->kind != PLAIN && !x->skip);
	if (x->jumped) {
		xJmp(e, j->leave);
		return;
	}
	if (x->skip) {
		xAluI(e, X_ADD, 8, R(R13), 2 * j->opCycles);
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
		if (pollPass) {
			pollSkip(j, ops, count, vaddr, paddr, pollPass * (uint32_t) j->opCycles, reload);
		}
		chain(j);
	} else {
		xJmp(e, j->leave);
	}
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
	j->e = (struct Emitter) { arena + n->jitUsed, arena + n->jitUsed, arena + n->jitUsed + BLOCK_CODE_MAX };
	j->exitCount = j->slowCount = j->missCount = j->codeCheckCount = j->rdpWaitCount = j->mappedCount = 0;
	j->leave = n->jitLeave;
	j->rdpThreads = n->rdpThreads > 0 || n->rdpGl;
	j->rdpGl = n->rdpGl;
	j->fast = n->fast;
	j->fle = n->fast && n->fle;
	j->ramSize = n->ramSize;
	pinSp = j->fle;
	j->fr = n->cpu.cop0[CP0_STATUS] >> 26 & 1;
	j->plainCount = n->countCycles == 2;
	j->opCycles = (int) n->opCycles;
	j->cacheData = !n->compat && !n->fast;
	struct Emitter* e = &j->e;

	if (blockIsIdleLoop(ops, count, vaddr)) {
		skipIdle(e, vaddr, paddr, j->fast, j->opCycles);
	}
	int32_t countImm;
	bool countDword;
	int const countReg = blockCountLoop(ops, count, vaddr, &countImm, &countDword);
	if (countReg) {
		skipCount(e, vaddr, paddr, j->fast, j->opCycles, countReg, countImm, countDword);
	}
	uint32_t const pairTarget = j->fast ? blockIdlePair(ops, count, vaddr) : 0;
	if (pairTarget) {
		skipPair(e, vaddr, pairTarget, j->opCycles);
	}
	int reload;
	uint32_t pollPass = blockPollPass(ops, count, vaddr, &reload);
	// FLE: its loads' addresses as each pass makes them (it counts no stalls, so a skipped pass costs
	// what a pass made does whatever the loads depend on).
	j->poll = pollPass != 0 && j->fle, j->pollBase = vaddr;
	for (uint32_t k = 0; j->poll && k < count; ++k) {
		if (ops[k] >> 26 >= 0x20) {
			xStoreI(e, 4, CPU_AT(pollAt[k]), RDRAM_SIZE);
		}
	}

	bool delay = false;
	int pending = 0;
	j->known = 1, j->value[0] = 0;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k], p = paddr + k * 4;
		uint64_t pc = sext32(vaddr + k * 4);
		if (k) {
			trackConst(j, ops[k - 1]);
		}
		bool last = k + 1 == count;
		bool lineStart = k == 0 || !(p & 31);
		enum Kind kind;
		uint8_t* mark = e->p;
		int exits = j->exitCount, misses = j->missCount, codeChecks = j->codeCheckCount, mapped = j->mappedCount;
		bool branch = isBranchOp(op);
		struct Slow* slow = &j->slows[j->slowCount];
		int const prevMark = k ? cpuLoadMark(ops[k - 1]) : -1;
		j->mark = cpuLoadMark(op);
		j->branchStall = prevMark >= 0 ? (int8_t) cpuLoadStall(op, (uint8_t) prevMark) : -1;
		*slow = (struct Slow) { .op = op, .paddr = p, .pc = pc, .delay = delay, .lineStart = lineStart, .last = last, .prevMark = prevMark,
		                    .nextStall = k + 1 < count && cpuLoadStall(ops[k + 1], cpuLoadMark(op)) };
		j->slow = slow;
		j->extraCycles = 0;
		j->interruptCheck = false;
		j->jumped = false;
		// A branch's I-cache check comes first (a likely branch not taken leaves inside it); the rest
		// check after, so slow paths and Count reads see the cycles before this fetch.
		if (lineStart && branch && !delay && !j->fast) {
			icacheCheck(e, (uint32_t) pc, p, false);
		}
		if (!(delay && branch) && emitInline(j, op, pc, &kind)) {
			if (lineStart && !branch && !j->fast) {
				icacheCheck(e, (uint32_t) pc, p, false);
			}
			if (j->fle) {
				// No load interlocks; the cycles go in at branches and exits.
				pending += j->opCycles + j->extraCycles;
				if (last || delay || kind != PLAIN || j->interruptCheck || j->jumped || pending > 4000) {
					xAluI(e, X_ADD, 8, R(R13), pending);
					pending = 0;
				}
			} else {
				if (prevMark >= 0) {
					j->extraCycles += cpuLoadStall(op, (uint8_t) prevMark);
				} else {
					loadStallCheck(e, op);
				}
				xAluI(e, X_ADD, 8, R(R13), j->opCycles + j->extraCycles);
			}
			if (j->interruptCheck) {
				// An interrupt now due is taken before the next instruction.
				xTest(e, 4, R(RCX), RCX);
				exitAt(j, xJcc(e, CC_NE, NULL), pc, kind, delay, false);
			}
			if (j->jumped) {
				exitAt(j, xJmp(e, NULL), pc, kind, delay, false);
				j->exits[j->exitCount - 1].jumped = true;
			} else if (last) {
				exitAt(j, xJmp(e, NULL), pc, kind, delay, false);
				j->exits[j->exitCount - 1].last = true;
			} else if (!j->fast) {
				// hle checks the limit at the block's end only.
				xAluRR(e, X_CMP, 8, R(R13), R14);
				exitAt(j, xJcc(e, CC_AE, NULL), pc, kind, delay, false);
			}
			if (slow->sites || slow->keep) {
				slow->resume = e->p;
				++j->slowCount;
			}
			delay = kind != PLAIN;
			continue;
		}
		e->p = mark;
		j->exitCount = exits;
		j->missCount = misses;
		j->codeCheckCount = codeChecks;
		j->mappedCount = mapped;
		emitHelper(j, op, pc, p, delay, lineStart, last, prevMark, slow->nextStall);
		delay = branch;
	}

	// The block's end goes on straight after it (its jump there dropped); the rest is cold.
	for (int i = 0; i < j->exitCount; ++i) {
		struct Exit* x = &j->exits[i];
		if (x->last) {
			if (x->site + 4 == e->p) {
				e->p -= 5;
				x->site = NULL;
			}
			emitExit(j, x, ops, count, vaddr, paddr, pollPass, reload);
		}
	}
	if (e->p > e->end) {
		return NULL;
	}
	struct Emitter const hot = j->e;
	uint8_t* const coldStart = arena + ARENA_SIZE / 2 + *coldUsed(n);
	j->e = (struct Emitter) { coldStart, coldStart, coldStart + BLOCK_CODE_MAX };

	// Stores to a page with code only: on unless they hit a word a block holds (n64HoldsCode), its
	// bit in codeWords. An aligned store touches one word, or two for SD.
	for (int i = 0; i < j->codeCheckCount; ++i) {
		struct Miss* k = &j->codeChecks[i];
		patchHere(e, k->site);
		// Past RDRAM, the interpreter. On a page the RDP draws to or reads, or the running RSP's DMA read,
		// it waits for them there in C first.
		xTestI(e, 4, R(RAX), ~(j->ramSize - 1));
		k->slow->site[k->slow->sites++] = xJcc(e, CC_NE, NULL);
		xTestI(e, 1, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, pageFlags)), PAGE_RDP | PAGE_RSP_READS);
		uint8_t* noRdp = xJcc(e, CC_E, NULL);
		rdpWaitCall(e, (const void*) cpuRdpWaitWrite, k->size);
		xMovRR(e, 4, RCX, RAX);
		xShiftI(e, X_SHR, 4, R(RCX), 12);
		patchHere(e, noRdp);
		xTestI(e, 1, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, pageFlags)), PAGE_CODE);
		xJcc(e, CC_E, k->resume);
		xMovRR(e, 4, RDX, RAX);
		xShiftI(e, X_SHR, 4, R(RDX), 5);
		xMovzx(e, RDX, 1, MI(RBX, RDX, 1, (int32_t) offsetof(struct N64, codeWords)));
		xMovRR(e, 4, RCX, RAX);
		xShiftI(e, X_SHR, 4, R(RCX), 2);
		xAluI(e, X_AND, 4, R(RCX), 7);
		xShiftCl(e, X_SHR, 4, R(RDX));
		xTestI(e, 4, R(RDX), k->size == 8 ? 3 : 1);
		k->slow->site[k->slow->sites++] = xJcc(e, CC_NE, NULL);
		xJmp(e, k->resume);
	}

	// Loads from pages the RDP may still be drawing to wait for it in C, then go on inline: the same
	// path (and time) whichever renderer draws and however long it keeps the pages.
	for (int i = 0; i < j->rdpWaitCount; ++i) {
		struct Miss* w = &j->rdpWaits[i];
		patchHere(e, w->site);
		rdpWaitCall(e, (const void*) cpuRdpWaitRead, w->size);
		xJmp(e, w->resume);
	}

	// Fast CPU accesses outside kseg0's RDRAM: the physical address from vpages, which must be RDRAM
	// (and, but with FLE, aligned).
	for (int i = 0; i < j->mappedCount; ++i) {
		struct Miss* v = &j->mapped[i];
		patchHere(e, v->site);
		xMovRR(e, 4, RAX, R8);
		xMovRR(e, 4, RCX, RAX);
		xShiftI(e, X_SHR, 4, R(RCX), 12);
		xLoad(e, 8, RDX, N64_AT(vpages));
		xAluRM(e, X_ADD, 4, RAX, MI(RDX, RCX, 4, v->store ? VPAGES * 4 : 0));
		xTestI(e, 4, R(RAX), ~(j->ramSize - 1));
		v->slow->site[v->slow->sites++] = xJcc(e, CC_NE, NULL);
		if (v->size > 1 && !j->fle) {
			xTestI(e, 4, R(R8), (uint32_t) (v->size - 1));
			v->slow->site[v->slow->sites++] = xJcc(e, CC_NE, NULL);
		}
		xJmp(e, v->resume);
	}

	// D-cache misses, filled (and a dirty line written back) in C. A misaligned address, one past
	// RDRAM, or a write-back over code, takes the slow path instead.
	for (int i = 0; i < j->missCount; ++i) {
		struct Miss* m = &j->misses[i];
		patchHere(e, m->site);
		if (m->size > 1) {
			xTestI(e, 4, R(R8), (uint32_t) (m->size - 1));
			m->slow->site[m->slow->sites++] = xJcc(e, CC_NE, NULL);
		}
		xTestI(e, 4, R(RAX), ~(j->ramSize - 1));
		m->slow->site[m->slow->sites++] = xJcc(e, CC_NE, NULL);
		if (m->glLoad) {
			// A line coming in from a page the GPU draws to (a game invalidates its cache before
			// reading what the RDP drew) waits for the GPU first.
			xMovRR(e, 4, RDX, RAX);
			xShiftI(e, X_SHR, 4, R(RDX), 12);
			xTestI(e, 1, MI(RBX, RDX, 1, (int32_t) offsetof(struct N64, pageFlags)), PAGE_RDP_WRITES);
			uint8_t* ready = xJcc(e, CC_E, NULL);
			rdpWaitCall(e, (const void*) cpuRdpWaitRead, 16);
			patchHere(e, ready);
		}
		xPush(e, RAX);
		xPush(e, R8);
		xMovRR(e, 8, RDI, RBX);
		xMovRR(e, 4, RSI, R8);
		xMovRR(e, 4, RDX, RAX);
		xMovRR(e, 8, RCX, R13);
		xMovI(e, R8, (m->slow->op >> 26) >= 0x28);
		xMovI64(e, RAX, (uint64_t) (uintptr_t) cpuDcacheMiss);
		xCallR(e, RAX);
		xMovRR(e, 4, RDX, RAX);
		xPop(e, R8);
		xPop(e, RAX);
		xAluI(e, X_CMP, 4, R(RDX), -1);
		m->slow->site[m->slow->sites++] = xJcc(e, CC_E, NULL);
		xAluRR(e, X_ADD, 8, R(R13), RDX);
		xMovRR(e, 4, RCX, R8);
		xShiftI(e, X_SHR, 4, R(RCX), 4);
		xAluI(e, X_AND, 4, R(RCX), 511);
		xJmp(e, m->resume);
	}
	for (int i = 0; i < j->slowCount; ++i) {
		struct Slow* sl = &j->slows[i];
		for (int k = 0; k < sl->sites; ++k) {
			patchHere(e, sl->site[k]);
		}
		emitHelper(j, sl->op, sl->pc, sl->paddr, sl->delay, sl->lineStart, sl->last, sl->prevMark, sl->nextStall);
		if (!sl->last) {
			xJmp(e, sl->resume);
		}
	}

	for (int i = 0; i < j->exitCount; ++i) {
		if (!j->exits[i].last) {
			emitExit(j, &j->exits[i], ops, count, vaddr, paddr, pollPass, reload);
		}
	}

	if (e->p > e->end) {
		return NULL;
	}
	*coldUsed(n) += (size_t) (e->p - coldStart + 15) & ~(size_t) 15;
	j->e = hot;
	uint8_t* code = arena + n->jitUsed;
	n->jitUsed += (size_t) (e->p - code + 15) & ~(size_t) 15;
	return code;
}
