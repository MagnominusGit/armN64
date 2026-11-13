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

enum { ARENA_SIZE = 256 << 20, BLOCK_CODE_MAX = 32 << 10, SHARED_SIZE = 256 };

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
	struct Miss misses[64];
	int missCount;
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
	xMovRR(e, 8, RDI, RBX);
	xMovIKeepFlags(e, RSI, op);
	xMovI64(e, RAX, (uint64_t) (uintptr_t) (j->fast ? (void*) fpuExecuteFast : probe ? (void*) fpuProbe : (void*) fpuExecute));
	xCallR(e, RAX);
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
// flushed to zero when subnormal with FCR31.FS set, and the host's default NaN made the VR4300's.
static void fastResult(struct Emitter* e, bool dbl, int fd) {
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
		xStore(e, 8, fprAt(fd), RAX);
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
		xStore(e, 4, fprAt(fd), RAX);
	}
}

// With HLE accuracy, the FPU instructions most code runs inline, as fpuExecuteFast runs them: the
// moves, the four operations, square root, ABS, MOV, NEG and C.cond in S and D, and the conversions
// between S, D and W. Only with even registers, the same in both FR modes; anything else goes
// through fpuCall.
static void fpuFast(struct Jit* j, uint32_t op) {
	struct Emitter* e = &j->e;
	int const fmt = (op >> 21) & 31, ft = (op >> 16) & 31, fs = (op >> 11) & 31, fd = (op >> 6) & 31, func = op & 0x3F;
	bool const dbl = fmt == 17;
	switch (fmt) {
	case 0:
	case 1:
		if (fs & 1) {
			break;
		}
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
		if (fs & 1) {
			break;
		}
		xLoad(e, fmt == 4 ? 4 : 8, RAX, gpr(ft));
		xStore(e, fmt == 4 ? 4 : 8, fprAt(fs), RAX);
		return;
	}
	bool const known = ((fmt == 16 || fmt == 17) && (func <= 7 || func == 13 || func == (dbl ? 0x20 : 0x21) || func >= 0x30)) ||
	                   (fmt == 20 && (func == 32 || func == 33));
	if (!known || ((fs | ft | fd) & 1)) {
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
		xSse(e, prefix, 0x10, 0, fprAt(fs), 4);
		xSse(e, dbl ? 0x66 : 0, 0x2E, 0, fprAt(ft), 4);
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
		xSse(e, func == 32 ? 0xF3 : 0xF2, 0x2A, 0, fprAt(fs), 4);
		xSse(e, func == 32 ? 0xF3 : 0xF2, 0x11, 0, out, 4);
		xLoad(e, func == 32 ? 4 : 8, RAX, out);
		fastResult(e, func == 33, fd);
		return;
	}
	switch (func) {
	case 5:
	case 7:
		xLoad(e, width, RAX, fprAt(fs));
		if (dbl) {
			xMovI64(e, RDX, func == 7 ? 0x8000000000000000ull : 0x7FFFFFFFFFFFFFFFull);
			xAluRR(e, func == 7 ? X_XOR : X_AND, 8, R(RAX), RDX);
		} else {
			xAluI(e, func == 7 ? X_XOR : X_AND, 4, R(RAX), func == 7 ? (int32_t) 0x80000000u : 0x7FFFFFFF);
		}
		fastResult(e, dbl, fd);
		return;
	case 6:
		xLoad(e, width, RAX, fprAt(fs));
		xStore(e, width, fprAt(fd), RAX);
		return;
	case 13:
		// TRUNC.W: out of range or NaN gives the most negative word, as toWord and the host do.
		xSse(e, prefix, 0x2C, RAX, fprAt(fs), 4);
		xStore(e, 4, fprAt(fd), RAX);
		return;
	case 0x20:
	case 0x21:
		// CVT.S.D and CVT.D.S.
		xSse(e, prefix, 0x5A, 0, fprAt(fs), 4);
		xSse(e, dbl ? 0xF3 : 0xF2, 0x11, 0, out, 4);
		xLoad(e, dbl ? 4 : 8, RAX, out);
		fastResult(e, !dbl, fd);
		return;
	}
	static const uint8_t ops[5] = { 0x58, 0x5C, 0x59, 0x5E, 0x51 };
	if (func == 4) {
		xSse(e, prefix, 0x51, 0, fprAt(fs), 4);
	} else {
		xSse(e, prefix, 0x10, 0, fprAt(fs), 4);
		xSse(e, prefix, ops[func], 0, fprAt(ft), 4);
	}
	xSse(e, prefix, 0x11, 0, out, 4);
	xLoad(e, width, RAX, out);
	fastResult(e, dbl, fd);
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
		}
		return true;
	}
	if (rs != 0x04 && rs != 0x05) {
		return false;
	}
	// Status in C: a write may leave the mode compiled code runs in.
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
// cpuRdpWaitRead or cpuRdpWaitWrite for the physical address in eax (the virtual one in r8 kept).
static void rdpWaitCall(struct Emitter* e, const void* f, int size) {
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

	// The address in 64 bits, which the interpreter takes when it isn't a 32-bit one sign-extended (an
	// address error); the low 32 bits from here on.
	xLoad(e, 8, RAX, gpr(rs));
	if ((int16_t) op) {
		xAluI(e, X_ADD, 8, R(RAX), (int16_t) op);
	}
	xMovsxd(e, RCX, R(RAX));
	xAluRR(e, X_CMP, 8, R(RCX), RAX);
	slow->site[slow->sites++] = xJcc(e, CC_NE, NULL);
	xMovRR(e, 4, R8, RAX);
	// The physical page from vpages: one not looked up lands past RDRAM.
	xMovRR(e, 4, RCX, RAX);
	xShiftI(e, X_SHR, 4, R(RCX), 12);
	xLoad(e, 8, RDX, N64_AT(vpages));
	xAluRM(e, X_ADD, 4, RAX, MI(RDX, RCX, 4, isStore ? VPAGES * 4 : 0));
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
		// Aligned and in RDRAM.
		xTestI(e, 4, R(RAX), ~(uint32_t) (RDRAM_SIZE - 1));
		slow->site[slow->sites++] = xJcc(e, CC_NE, NULL);
		if (size > 1) {
			xTestI(e, 4, R(R8), (uint32_t) (size - 1));
			slow->site[slow->sites++] = xJcc(e, CC_NE, NULL);
		}
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
		xLoad(e, 4, RAX, gpr((op >> 21) & 31));
		xAluI(e, X_ADD, 4, R(RAX), (int16_t) op);
		xMovRR(e, 4, R8, RAX);
		xMovRR(e, 4, RCX, RAX);
		xShiftI(e, X_SHR, 4, R(RCX), 12);
		xLoad(e, 8, RDX, N64_AT(vpages));
		xAluRM(e, X_ADD, 4, RAX, MI(RDX, RCX, 4, 0));
		if (j->rdpThreads) {
			xMovRR(e, 4, RCX, RAX);
			xShiftI(e, X_SHR, 4, R(RCX), 12);
			xTestI(e, 1, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, pageFlags)), PAGE_RDP_WRITES);
			out[sites++] = xJcc(e, CC_NE, NULL);
		}
		if (j->fast) {
			xTestI(e, 4, R(RAX), ~(uint32_t) (RDRAM_SIZE - 1) | (uint32_t) (size - 1));
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
		xJmpR(&e, RSI);
		n->jitLeave = e.p;
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
	j->exitCount = j->slowCount = j->missCount = j->codeCheckCount = j->rdpWaitCount = 0;
	j->leave = n->jitLeave;
	j->rdpThreads = n->rdpThreads > 0 || n->rdpGl;
	j->rdpGl = n->rdpGl;
	j->fast = n->fast;
	j->plainCount = n->countCycles == 2;
	j->opCycles = (int) n->opCycles;
	j->cacheData = !n->compat && !n->fast;
	struct Emitter* e = &j->e;

	if (blockIsIdleLoop(ops, count, vaddr)) {
		skipIdle(e, vaddr, paddr, j->fast, j->opCycles);
	}
	int reload;
	uint32_t pollPass = blockPollPass(ops, count, vaddr, &reload);

	bool delay = false;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = ops[k], p = paddr + k * 4;
		uint64_t pc = sext32(vaddr + k * 4);
		bool last = k + 1 == count;
		bool lineStart = k == 0 || !(p & 31);
		enum Kind kind;
		uint8_t* mark = e->p;
		int exits = j->exitCount, misses = j->missCount, codeChecks = j->codeCheckCount;
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
			if (prevMark >= 0) {
				j->extraCycles += cpuLoadStall(op, (uint8_t) prevMark);
			} else {
				loadStallCheck(e, op);
			}
			xAluI(e, X_ADD, 8, R(R13), j->opCycles + j->extraCycles);
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
		emitHelper(j, op, pc, p, delay, lineStart, last, prevMark, slow->nextStall);
		delay = branch;
	}

	// Stores to a page with code only: on unless they hit a word a block holds (n64HoldsCode), its
	// bit in codeWords. An aligned store touches one word, or two for SD.
	for (int i = 0; i < j->codeCheckCount; ++i) {
		struct Miss* k = &j->codeChecks[i];
		patchHere(e, k->site);
		// Past RDRAM, the interpreter. On a page the RDP draws to or reads, it waits for the RDP
		// there in C first.
		xTestI(e, 4, R(RAX), ~(uint32_t) (RDRAM_SIZE - 1));
		k->slow->site[k->slow->sites++] = xJcc(e, CC_NE, NULL);
		xTestI(e, 1, MI(RBX, RCX, 1, (int32_t) offsetof(struct N64, pageFlags)), PAGE_RDP);
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

	// D-cache misses, filled (and a dirty line written back) in C. A misaligned address, one past
	// RDRAM, or a write-back over code, takes the slow path instead.
	for (int i = 0; i < j->missCount; ++i) {
		struct Miss* m = &j->misses[i];
		patchHere(e, m->site);
		if (m->size > 1) {
			xTestI(e, 4, R(R8), (uint32_t) (m->size - 1));
			m->slow->site[m->slow->sites++] = xJcc(e, CC_NE, NULL);
		}
		xTestI(e, 4, R(RAX), ~(uint32_t) (RDRAM_SIZE - 1));
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

	// Leaving after an inline instruction: the state the interpreter would have after it.
	for (int i = 0; i < j->exitCount; ++i) {
		struct Exit* x = &j->exits[i];
		patchHere(e, x->site);
		if (x->stall < 0) {
			loadStallCheck(e, ops[((uint32_t) x->pc - vaddr) >> 2]);
		} else if (x->stall) {
			xAluI(e, X_ADD, 8, R(R13), x->stall);
		}
		xStoreI(e, 1, CPU_AT(loadMark), x->mark);
		storeConst(e, CPU_AT(curPc), x->pc);
		xStoreI(e, 1, CPU_AT(inDelay), x->delay);
		xStoreI(e, 1, CPU_AT(nextInDelay), x->kind != PLAIN && !x->skip);
		if (x->jumped) {
			xJmp(e, j->leave);
			continue;
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

	if (e->p > e->end) {
		return NULL;
	}
	uint8_t* code = arena + n->jitUsed;
	n->jitUsed += (size_t) (e->p - code + 15) & ~(size_t) 15;
	return code;
}
