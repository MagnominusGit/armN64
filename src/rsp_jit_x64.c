#include "n64.h"
#include "x64/emit.h"

#include <stddef.h>
#include <sys/mman.h>

// x86-64 code for blocks of RSP instructions. Each block's first instruction takes its timing from
// rspTimingStep (the instruction before it isn't known) unless the timing is plain; the rest have the
// timing compiled in. Scalar instructions, DMEM loads and stores and most of the vector unit run
// inline, the rest of the vector unit and COP0 through C. In lle, code stops after any instruction that takes the clock to the limit, as
// rspCatchUp does; in mle and hle at the block's end.
//
// Registers: rbx = n, r12 = &n->rsp, r13 = clock, r14 = the limit, r15 = lastStart,
// ebp = lastAlone.

enum { ARENA_SIZE = 32 << 20, BLOCK_CODE_MAX = 32 << 10, SHARED_SIZE = 256, BLOCK_MAX = 64 };

#define RSP_AT(f) M(R12, (int32_t) offsetof(struct RSP, f))
#define N64_AT(f) M(RBX, (int32_t) offsetof(struct N64, f))

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
	bool delay, last, halt, pcsStored, accDirty;
};

struct Jit {
	struct Emitter e;
	struct Exit exits[2 * BLOCK_MAX];
	int exitCount;
	uint8_t* leave;
};

// The accumulator stays in xmm3-xmm5 between the block's vector instructions: loaded when one first
// wants it, written back where something else may read it (a C call, the block's exits), and not at
// all where it is dead: the instructions after overwrite all of it before any reads it.
static struct {
	bool cached, dirty;
	// For each instruction of the block: whether it may read the accumulator, and whether the
	// accumulator may be read after it (before something overwrites all of it).
	bool reads[64], liveAfter[64];
	int k;
} acc;

static void exitAt(struct Jit* j, uint8_t* site, uint32_t op, uint32_t pc, enum Kind kind, bool delay) {
	j->exits[j->exitCount++] = (struct Exit) { site, op, pc, kind, delay, false, false, false, acc.dirty };
}

static void vStoreAcc(struct Emitter* e);

// Before a C call: the accumulator back in memory if anything may read it there; the call takes
// every xmm register.
static void accBeforeCall(struct Emitter* e) {
	if (acc.dirty && (acc.reads[acc.k] || acc.liveAfter[acc.k])) {
		vStoreAcc(e);
	}
	acc.cached = acc.dirty = false;
}

static void callRaw(struct Emitter* e, const void* f) {
	xMovI64(e, RAX, (uint64_t) (uintptr_t) f);
	xCallR(e, RAX);
}

static void callC(struct Emitter* e, const void* f) {
	accBeforeCall(e);
	callRaw(e, f);
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
// rspPlainTiming, for the block being compiled.
static bool plainTiming;

static void emitTiming(struct Emitter* e, uint32_t op, uint32_t prev, uint32_t pc) {
	if (plainTiming) {
		xLea(e, 8, R13, M(R13, 1 + rspIsBranch(op)));
		return;
	}
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

// SSE instructions: [prefix] [REX] 0F op (0F 38 op or 0F 3A op for op > 0xFF) with a ModRM.
enum { XMM0, XMM1, XMM2 };

static void sse(struct Emitter* e, uint8_t prefix, uint32_t op, int reg, struct Opnd rm) {
	if (prefix) {
		e8(e, prefix);
	}
	emitRex(e, 4, reg, rm, false);
	e8(e, 0x0F);
	if (op > 0xFF) {
		e8(e, op >> 8);
	}
	e8(e, op);
	emitModrm(e, reg, rm);
}

// The bytes of every 16-bit lane of x swapped (t is scratch).
static void sseSwapBytes(struct Emitter* e, int x, int t) {
	sse(e, 0x66, 0x6F, t, R(x));
	sse(e, 0x66, 0x71, 6, R(x));
	e8(e, 8);
	sse(e, 0x66, 0x71, 2, R(t));
	e8(e, 8);
	sse(e, 0x66, 0xEB, x, R(t));
}

// MOVD, MOVQ or MOVDQU between xmm x and memory, 4, 8 or 16 bytes.
static void sseLoad(struct Emitter* e, int size, int x, struct Opnd m) {
	sse(e, size == 4 ? 0x66 : 0xF3, size == 4 ? 0x6E : size == 8 ? 0x7E : 0x6F, x, m);
}

static void sseStore(struct Emitter* e, int size, struct Opnd m, int x) {
	sse(e, size == 16 ? 0xF3 : 0x66, size == 4 ? 0x7E : size == 8 ? 0xD6 : 0x7F, x, m);
}

// The vector unit in SSE2, as rsp_vu.c computes it, the same way rsp_jit_a64.c does in NEON. Each
// instruction loads what it reads and stores what it writes; xmm0 = vs, xmm1 = the element of vt,
// xmm2 = the result, xmm3-xmm5 = the accumulator's high, middle and low slices, xmm6-xmm15 scratch.
enum { ACC_H = 3, ACC_M = 4, ACC_L = 5 };

enum {
	P_ADDW = 0xFD, P_SUBW = 0xF9, P_MULLW = 0xD5, P_MULHW = 0xE5, P_MULHUW = 0xE4, P_ADDD = 0xFE, P_SUBD = 0xFA,
	P_AND = 0xDB, P_ANDN = 0xDF, P_OR = 0xEB, P_XOR = 0xEF, P_CMPEQW = 0x75, P_CMPGTW = 0x65,
	P_PACKSSDW = 0x6B, P_PACKSSWB = 0x63, P_UNPCKLWD = 0x61, P_UNPCKHWD = 0x69,
};

// The lanes' bits (lane i is bit i), RIP-relative at the end of the arena's shared code.
static const uint16_t laneBits[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
enum { LANE_BITS_AT = SHARED_SIZE - 16 };
static uint8_t* laneBitsAt;

static void p2(struct Emitter* e, uint32_t op, int d, int s) {
	sse(e, 0x66, op, d, R(s));
}

static void pMov(struct Emitter* e, int d, int s) {
	if (d != s) {
		p2(e, 0x6F, d, s);
	}
}

// PSRAW, PSRLW, PSLLW, PSRAD by an immediate.
static void pShift(struct Emitter* e, int group, int kind, int d, int amount) {
	sse(e, 0x66, group, kind, R(d));
	e8(e, amount);
}

static void pSra(struct Emitter* e, int d, int amount) {
	pShift(e, 0x71, 4, d, amount);
}

static void pSrl(struct Emitter* e, int d, int amount) {
	pShift(e, 0x71, 2, d, amount);
}

static void pSll(struct Emitter* e, int d, int amount) {
	pShift(e, 0x71, 6, d, amount);
}

// PSHUFD, PSHUFLW (F2) and PSHUFHW (F3) of d in place.
static void pShuf(struct Emitter* e, uint8_t prefix, int d, int imm) {
	sse(e, prefix, 0x70, d, R(d));
	e8(e, imm);
}

static void pZero(struct Emitter* e, int d) {
	p2(e, P_XOR, d, d);
}

static void pOnes(struct Emitter* e, int d) {
	p2(e, P_CMPEQW, d, d);
}

// Every lane 0x8000.
static void pMin(struct Emitter* e, int d) {
	pOnes(e, d);
	pSll(e, d, 15);
}

static void pNot(struct Emitter* e, int d, int t) {
	pOnes(e, t);
	p2(e, P_XOR, d, t);
}

// d = a >u b; d isn't b, t is scratch.
static void pCmpHi(struct Emitter* e, int d, int a, int b, int t) {
	pMin(e, t);
	pMov(e, d, a);
	p2(e, P_XOR, d, t);
	p2(e, P_XOR, t, b);
	p2(e, P_CMPGTW, d, t);
}

// d = m ? a : b; d isn't m, t is scratch.
static void pSelect(struct Emitter* e, int d, int m, int a, int b, int t) {
	pMov(e, t, m);
	p2(e, P_ANDN, t, b);
	pMov(e, d, a);
	p2(e, P_AND, d, m);
	p2(e, P_OR, d, t);
}

// d = the mask of the bits in eax (maskOf); and eax = the lanes of mask x as bits.
static void pMaskOf(struct Emitter* e, int d, int t) {
	sse(e, 0x66, 0x6E, d, R(RAX));
	pShuf(e, 0xF2, d, 0);
	pShuf(e, 0x66, d, 0);
	if (t & 8) {
		e8(e, 0x66);
		e8(e, 0x44);
	} else {
		e8(e, 0x66);
	}
	e8(e, 0x0F);
	e8(e, 0x6F);
	e8(e, ((t & 7) << 3) | 5);
	e32(e, (uint32_t) (laneBitsAt - (e->p + 4)));
	p2(e, P_AND, d, t);
	p2(e, P_CMPEQW, d, t);
}

static void pBits(struct Emitter* e, int x, int t) {
	pMov(e, t, x);
	p2(e, P_PACKSSWB, t, t);
	sse(e, 0x66, 0xD7, RAX, R(t));
	xMovzx(e, RAX, 1, R(RAX));
}

// ax = bits of lo | bits of hi << 8.
static void pBitsPair(struct Emitter* e, int lo, int hi, int t) {
	pBits(e, hi, t);
	xMovRR(e, 4, RDX, RAX);
	pBits(e, lo, t);
	xShiftI(e, X_SHL, 4, R(RDX), 8);
	xAluRR(e, X_OR, 4, R(RAX), RDX);
}

// dlo, dhi = the lanes of x sign-extended to 32 bits, lanes 0-3 and 4-7.
static void pWiden(struct Emitter* e, int dlo, int dhi, int x) {
	pMov(e, dlo, x);
	p2(e, P_UNPCKLWD, dlo, x);
	pShift(e, 0x72, 4, dlo, 16);
	pMov(e, dhi, x);
	p2(e, P_UNPCKHWD, dhi, x);
	pShift(e, 0x72, 4, dhi, 16);
}

// d = the high halves of the products a * b, signed or unsigned each (mulhiSS, mulhiUU, mulhiSU,
// mulhiUS); d isn't b, t is scratch.
static void pMulhi(struct Emitter* e, int d, int a, int b, bool signedA, bool signedB, int t) {
	pMov(e, d, a);
	p2(e, signedA || signedB ? P_MULHW : P_MULHUW, d, b);
	if (signedA != signedB) {
		// The unsigned operand's sign bit counted as 0x10000: add the signed one where it is set.
		pMov(e, t, signedA ? b : a);
		pSra(e, t, 15);
		p2(e, P_AND, t, signedA ? a : b);
		p2(e, P_ADDW, d, t);
	}
}

static void pMulLo(struct Emitter* e, int d, int a, int b) {
	pMov(e, d, a);
	p2(e, P_MULLW, d, b);
}

// The accumulator += (h, m, l), wrapping at 48 bits, as accAdd: carries from the low and middle
// slices (masks, so subtracting one adds 1). h, m and l must be xmm12-xmm15 (or xmm0-xmm2).
static void pAccAdd(struct Emitter* e, int h, int m, int l) {
	pMov(e, 6, ACC_L);
	p2(e, P_ADDW, 6, l);
	pCmpHi(e, 7, ACC_L, 6, 10);
	pMov(e, 8, ACC_M);
	p2(e, P_ADDW, 8, m);
	pCmpHi(e, 9, ACC_M, 8, 10);
	pMov(e, ACC_M, 8);
	p2(e, P_SUBW, ACC_M, 7);
	pZero(e, 11);
	p2(e, P_CMPEQW, 11, ACC_M);
	p2(e, P_AND, 7, 11);
	p2(e, P_ADDW, ACC_H, h);
	p2(e, P_SUBW, ACC_H, 9);
	p2(e, P_SUBW, ACC_H, 7);
	pMov(e, ACC_L, 6);
}

// d = bits 16-47 of the accumulator saturated to 16 signed bits (clampSigned), the low slice when
// they fit (clampLow), or saturated to 16 unsigned bits (clampUnsigned).
static void pClampSigned(struct Emitter* e, int d) {
	pMov(e, 6, ACC_M);
	p2(e, P_UNPCKLWD, 6, ACC_H);
	pMov(e, 7, ACC_M);
	p2(e, P_UNPCKHWD, 7, ACC_H);
	p2(e, P_PACKSSDW, 6, 7);
	pMov(e, d, 6);
}

static void pClampLow(struct Emitter* e, int d) {
	pMov(e, 6, ACC_M);
	pSra(e, 6, 15);
	p2(e, P_CMPEQW, 6, ACC_H);
	pMov(e, 7, ACC_H);
	pSra(e, 7, 15);
	pNot(e, 7, 8);
	pMov(e, d, 6);
	p2(e, P_AND, d, ACC_L);
	p2(e, P_ANDN, 6, 7);
	p2(e, P_OR, d, 6);
}

static void pClampUnsigned(struct Emitter* e, int d) {
	pMov(e, 6, ACC_H);
	pSra(e, 6, 15);
	pZero(e, 8);
	pMov(e, 7, ACC_H);
	p2(e, P_CMPGTW, 7, 8);
	pMov(e, 9, ACC_M);
	pSra(e, 9, 15);
	p2(e, P_OR, 7, 9);
	pMov(e, d, ACC_M);
	p2(e, P_OR, d, 7);
	p2(e, P_ANDN, 6, d);
	pMov(e, d, 6);
}

static void vLoadAcc(struct Emitter* e) {
	sseLoad(e, 16, ACC_H, RSP_AT(accH));
	sseLoad(e, 16, ACC_M, RSP_AT(accM));
	sseLoad(e, 16, ACC_L, RSP_AT(accL));
}

static void vStoreAcc(struct Emitter* e) {
	sseStore(e, 16, RSP_AT(accH), ACC_H);
	sseStore(e, 16, RSP_AT(accM), ACC_M);
	sseStore(e, 16, RSP_AT(accL), ACC_L);
}

// The accumulator in xmm3-xmm5 for an instruction that reads it.
static void accWant(struct Emitter* e) {
	if (!acc.cached) {
		vLoadAcc(e);
		acc.cached = true;
	}
}

// The accumulator's low slice set from x.
static void accSetLow(struct Emitter* e, int x) {
	if (acc.cached) {
		pMov(e, ACC_L, x);
		acc.dirty = true;
	} else {
		sseStore(e, 16, RSP_AT(accL), x);
	}
}

// The vector instructions compiled to SSE; false for the rest, which call their C functions.
static bool emitVectorOp(struct Emitter* e, uint32_t op) {
	int el = (op >> 21) & 15, vt = (op >> 16) & 31, vs = (op >> 11) & 31, vd = (op >> 6) & 31;
	int func = op & 0x3F;
	bool known = func <= 0x01 || (func >= 0x04 && func <= 0x09) || (func >= 0x0C && func <= 0x11) || func == 0x14 ||
	             func == 0x15 || func == 0x1D || (func >= 0x20 && func <= 0x25) || (func >= 0x27 && func <= 0x2D);
	if (!known) {
		return false;
	}
	if (func == 0x1D) {
		if (el >= 8 && el <= 10 && acc.cached) {
			pMov(e, XMM2, ACC_H + el - 8);
		} else if (el >= 8 && el <= 10) {
			sseLoad(e, 16, XMM2, el == 8 ? RSP_AT(accH) : el == 9 ? RSP_AT(accM) : RSP_AT(accL));
		} else {
			pZero(e, XMM2);
		}
		sseStore(e, 16, RSP_AT(vr[vd]), XMM2);
		return true;
	}
	sseLoad(e, 16, XMM0, RSP_AT(vr[vs]));
	sseLoad(e, 16, XMM1, RSP_AT(vr[vt]));
	if (el == 2 || el == 3) {
		pShuf(e, 0xF2, XMM1, el == 2 ? 0xA0 : 0xF5);
		pShuf(e, 0xF3, XMM1, el == 2 ? 0xA0 : 0xF5);
	} else if (el >= 4 && el <= 7) {
		pShuf(e, 0xF2, XMM1, (el - 4) * 0x55);
		pShuf(e, 0xF3, XMM1, (el - 4) * 0x55);
	} else if (el >= 8) {
		int lane = el & 7;
		pShuf(e, lane < 4 ? 0xF2 : 0xF3, XMM1, (lane & 3) * 0x55);
		pShuf(e, 0x66, XMM1, lane < 4 ? 0x00 : 0xAA);
	}
	switch (func) {
	case 0x00: case 0x01: case 0x08: case 0x09:
		// VMULF, VMULU, VMACF, VMACU: acc (+)= s * t * 2, rounded when not adding.
		pMulhi(e, 13, XMM0, XMM1, true, true, 6);
		pMulLo(e, 14, XMM0, XMM1);
		pMov(e, 15, 14);
		pSll(e, 15, 1);
		pSll(e, 13, 1);
		pSrl(e, 14, 15);
		p2(e, P_OR, 13, 14);
		pMin(e, 6);
		pMov(e, 12, XMM0);
		p2(e, P_CMPEQW, 12, 6);
		pMov(e, 7, XMM1);
		p2(e, P_CMPEQW, 7, 6);
		p2(e, P_AND, 12, 7);
		if (func < 0x08) {
			pMov(e, ACC_L, 15);
			p2(e, P_XOR, ACC_L, 6);
			pMov(e, 7, 15);
			pSrl(e, 7, 15);
			pMov(e, ACC_M, 13);
			p2(e, P_ADDW, ACC_M, 7);
			pMov(e, ACC_H, ACC_M);
			pSra(e, ACC_H, 15);
			p2(e, P_ANDN, 12, ACC_H);
			pMov(e, ACC_H, 12);
		} else {
			accWant(e);
			pMov(e, 14, 13);
			pSra(e, 14, 15);
			p2(e, P_ANDN, 12, 14);
			pAccAdd(e, 12, 13, 15);
		}
		acc.cached = acc.dirty = true;
		if (func & 1) {
			pClampUnsigned(e, XMM2);
		} else {
			pClampSigned(e, XMM2);
		}
		break;
	case 0x04:
		pMulhi(e, ACC_L, XMM0, XMM1, false, false, 6);
		pZero(e, ACC_H);
		pZero(e, ACC_M);
		acc.cached = acc.dirty = true;
		pMov(e, XMM2, ACC_L);
		break;
	case 0x05: case 0x06:
		// VMUDM, VMUDN: the middle slice from a mixed product, the low from the plain one.
		pMulhi(e, ACC_M, XMM0, XMM1, func == 0x05, func == 0x06, 6);
		pMulLo(e, ACC_L, XMM0, XMM1);
		pMov(e, ACC_H, ACC_M);
		pSra(e, ACC_H, 15);
		acc.cached = acc.dirty = true;
		pMov(e, XMM2, func == 0x05 ? ACC_M : ACC_L);
		break;
	case 0x07:
		pMulhi(e, ACC_H, XMM0, XMM1, true, true, 6);
		pMulLo(e, ACC_M, XMM0, XMM1);
		pZero(e, ACC_L);
		acc.cached = acc.dirty = true;
		pClampSigned(e, XMM2);
		break;
	case 0x0C:
		accWant(e);
		pMulhi(e, 12, XMM0, XMM1, false, false, 6);
		pZero(e, 13);
		pAccAdd(e, 13, 13, 12);
		acc.cached = acc.dirty = true;
		pClampLow(e, XMM2);
		break;
	case 0x0D: case 0x0E:
		accWant(e);
		pMulhi(e, 12, XMM0, XMM1, func == 0x0D, func == 0x0E, 6);
		pMulLo(e, 13, XMM0, XMM1);
		pMov(e, 14, 12);
		pSra(e, 14, 15);
		pAccAdd(e, 14, 12, 13);
		acc.cached = acc.dirty = true;
		if (func == 0x0D) {
			pClampSigned(e, XMM2);
		} else {
			pClampLow(e, XMM2);
		}
		break;
	case 0x0F:
		accWant(e);
		pMulhi(e, 12, XMM0, XMM1, true, true, 6);
		pMulLo(e, 13, XMM0, XMM1);
		pZero(e, 14);
		pAccAdd(e, 12, 13, 14);
		acc.cached = acc.dirty = true;
		pClampSigned(e, XMM2);
		break;
	case 0x10: case 0x11: {
		// VADD, VSUB: with VCO's carries, the sum wrapping into the accumulator and saturated into vd.
		int add = func == 0x11 ? P_SUBW : P_ADDW, addd = func == 0x11 ? P_SUBD : P_ADDD;
		xMovzx(e, RAX, 2, RSP_AT(vco));
		pMaskOf(e, 12, 15);
		pSrl(e, 12, 15);
		pMov(e, 13, XMM0);
		p2(e, add, 13, XMM1);
		p2(e, add, 13, 12);
		accSetLow(e, 13);
		pWiden(e, 6, 7, XMM0);
		pWiden(e, 8, 9, XMM1);
		pWiden(e, 10, 11, 12);
		p2(e, addd, 6, 8);
		p2(e, addd, 6, 10);
		p2(e, addd, 7, 9);
		p2(e, addd, 7, 11);
		p2(e, P_PACKSSDW, 6, 7);
		pMov(e, XMM2, 6);
		xStoreI(e, 2, RSP_AT(vco), 0);
		break;
	}
	case 0x14:
		pMov(e, XMM2, XMM0);
		p2(e, P_ADDW, XMM2, XMM1);
		pCmpHi(e, 6, XMM0, XMM2, 7);
		pBits(e, 6, 7);
		xStore(e, 2, RSP_AT(vco), RAX);
		accSetLow(e, XMM2);
		break;
	case 0x15:
		pMov(e, XMM2, XMM0);
		p2(e, P_SUBW, XMM2, XMM1);
		pCmpHi(e, 6, XMM1, XMM0, 7);
		pZero(e, 8);
		p2(e, P_CMPEQW, 8, XMM2);
		pNot(e, 8, 9);
		pBitsPair(e, 6, 8, 7);
		xStore(e, 2, RSP_AT(vco), RAX);
		accSetLow(e, XMM2);
		break;
	case 0x20: case 0x21: case 0x22: case 0x23: {
		// VLT, VEQ, VNE, VGE: VCC from the comparison, with VCO's carries and not-equals.
		xMovzx(e, RAX, 2, RSP_AT(vco));
		pMaskOf(e, 6, 15);
		xShiftI(e, X_SHR, 4, R(RAX), 8);
		pMaskOf(e, 7, 15);
		pMov(e, 8, XMM0);
		p2(e, P_CMPEQW, 8, XMM1);
		switch (func) {
		case 0x20:
			pMov(e, 9, XMM1);
			p2(e, P_CMPGTW, 9, XMM0);
			pMov(e, 10, 8);
			p2(e, P_AND, 10, 7);
			p2(e, P_AND, 10, 6);
			p2(e, P_OR, 9, 10);
			break;
		case 0x21:
			pMov(e, 9, 7);
			p2(e, P_ANDN, 9, 8);
			break;
		case 0x22:
			pMov(e, 9, 8);
			pNot(e, 9, 10);
			p2(e, P_OR, 9, 7);
			break;
		default:
			pMov(e, 9, XMM0);
			p2(e, P_CMPGTW, 9, XMM1);
			pMov(e, 10, 7);
			p2(e, P_AND, 10, 6);
			p2(e, P_ANDN, 10, 8);
			p2(e, P_OR, 9, 10);
		}
		if (func == 0x21) {
			pMov(e, XMM2, XMM1);
		} else {
			pSelect(e, XMM2, 9, XMM0, XMM1, 10);
		}
		pBits(e, 9, 10);
		xStore(e, 2, RSP_AT(vcc), RAX);
		xStoreI(e, 2, RSP_AT(vco), 0);
		accSetLow(e, XMM2);
		break;
	}
	case 0x24:
		// VCL.
		xMovzx(e, RAX, 2, RSP_AT(vcc));
		pMaskOf(e, 6, 15);
		xShiftI(e, X_SHR, 4, R(RAX), 8);
		pMaskOf(e, 7, 15);
		xMovzx(e, RAX, 2, RSP_AT(vco));
		pMaskOf(e, 8, 15);
		xShiftI(e, X_SHR, 4, R(RAX), 8);
		pMaskOf(e, 9, 15);
		xMovzx(e, RAX, 1, RSP_AT(vce));
		pMaskOf(e, 10, 15);
		pMov(e, 11, XMM0);
		p2(e, P_ADDW, 11, XMM1);
		pCmpHi(e, 12, XMM0, 11, 13);
		pZero(e, 13);
		p2(e, P_CMPEQW, 13, 11);
		pMov(e, 14, 12);
		pNot(e, 14, 15);
		p2(e, P_OR, 14, 13);
		pMov(e, 15, 12);
		p2(e, P_ANDN, 15, 13);
		pSelect(e, 11, 10, 14, 15, 12);
		pMov(e, 12, 9);
		p2(e, P_ANDN, 12, 8);
		pSelect(e, 13, 12, 11, 6, 14);
		pMov(e, 12, 8);
		p2(e, P_OR, 12, 9);
		pCmpHi(e, 10, XMM1, XMM0, 14);
		pNot(e, 10, 14);
		pSelect(e, 11, 12, 7, 10, 14);
		pZero(e, 6);
		p2(e, P_SUBW, 6, XMM1);
		pSelect(e, 7, 13, 6, XMM0, 9);
		pSelect(e, 10, 11, XMM1, XMM0, 9);
		pSelect(e, XMM2, 8, 7, 10, 9);
		pBitsPair(e, 13, 11, 9);
		xStore(e, 2, RSP_AT(vcc), RAX);
		xStoreI(e, 2, RSP_AT(vco), 0);
		xStoreI(e, 1, RSP_AT(vce), 0);
		accSetLow(e, XMM2);
		break;
	case 0x25:
		// VCH.
		pMov(e, 6, XMM0);
		p2(e, P_XOR, 6, XMM1);
		pSra(e, 6, 15);
		pMov(e, 7, XMM0);
		p2(e, P_ADDW, 7, XMM1);
		pMov(e, 8, XMM0);
		p2(e, P_SUBW, 8, XMM1);
		pMov(e, 9, XMM1);
		pSra(e, 9, 15);
		pZero(e, 15);
		pMov(e, 10, 7);
		p2(e, P_CMPGTW, 10, 15);
		pNot(e, 10, 14);
		pSelect(e, 11, 6, 10, 9, 14);
		pMov(e, 12, 8);
		pSra(e, 12, 15);
		pNot(e, 12, 14);
		pSelect(e, 13, 6, 9, 12, 14);
		pZero(e, 9);
		p2(e, P_SUBW, 9, XMM1);
		pSelect(e, 10, 11, 9, XMM0, 12);
		pSelect(e, 14, 13, XMM1, XMM0, 12);
		pSelect(e, XMM2, 6, 10, 14, 12);
		pBitsPair(e, 11, 13, 12);
		xStore(e, 2, RSP_AT(vcc), RAX);
		// VCO: the sign lanes, and the lanes not equal: sum not 0 or -1 where signs differ, diff not 0.
		pZero(e, 9);
		p2(e, P_CMPEQW, 9, 7);
		pOnes(e, 10);
		p2(e, P_CMPEQW, 10, 7);
		pMov(e, 11, 9);
		p2(e, P_OR, 11, 10);
		pNot(e, 11, 12);
		pZero(e, 12);
		p2(e, P_CMPEQW, 12, 8);
		pNot(e, 12, 13);
		pSelect(e, 13, 6, 11, 12, 14);
		pBitsPair(e, 6, 13, 14);
		xStore(e, 2, RSP_AT(vco), RAX);
		p2(e, P_AND, 10, 6);
		pBits(e, 10, 14);
		xStore(e, 1, RSP_AT(vce), RAX);
		accSetLow(e, XMM2);
		break;
	case 0x27:
		xMovzx(e, RAX, 2, RSP_AT(vcc));
		pMaskOf(e, 6, 15);
		pSelect(e, XMM2, 6, XMM0, XMM1, 7);
		xStoreI(e, 2, RSP_AT(vco), 0);
		accSetLow(e, XMM2);
		break;
	default:
		// VAND, VNAND, VOR, VNOR, VXOR, VNXOR.
		pMov(e, XMM2, XMM0);
		p2(e, func <= 0x29 ? P_AND : func <= 0x2B ? P_OR : P_XOR, XMM2, XMM1);
		if (func & 1) {
			pNot(e, XMM2, 6);
		}
		accSetLow(e, XMM2);
	}
	sseStore(e, 16, RSP_AT(vr[vd]), XMM2);
	return true;
}

// LSV, LLV, LDV and LQV (SSV to SQV) at an even element that fits: the lanes move whole,
// byte-swapped. Addresses past the end of DMEM, and LQV/SQV off a 16-byte boundary, take the
// general path in C.
static bool emitVectorMemory(struct Emitter* e, uint32_t op, bool store) {
	int kind = (op >> 11) & 31, el = (op >> 7) & 15, vt = (op >> 16) & 31;
	if (kind < 1 || kind > 4 || (el & 1) || el + (1 << kind) > 16) {
		return false;
	}
	int size = 1 << kind;
	int32_t offset = ((int32_t) ((op & 0x7F) << 25) >> 25) * size;
	// The C path below takes every xmm register: the accumulator goes back to memory first where it
	// is live.
	if (acc.dirty && acc.liveAfter[acc.k]) {
		vStoreAcc(e);
	}
	acc.dirty = false;
	uint8_t* wraps = dmemAddress(e, (op >> 21) & 31, offset, size);
	uint8_t* unaligned = NULL;
	if (size == 16) {
		xTestI(e, 4, R(RCX), 15);
		unaligned = xJcc(e, CC_NE, NULL);
	}
	struct Opnd lanes = RSP_AT(vr[vt][el / 2]);
	struct Opnd from = store ? lanes : dmem(RCX), to = store ? dmem(RCX) : lanes;
	if (size == 2) {
		xMovzx(e, RAX, 2, from);
		xShiftI(e, X_ROL, 2, R(RAX), 8);
		xStore(e, 2, to, RAX);
	} else if (size == 4) {
		xLoad(e, 4, RAX, from);
		xBswap(e, 4, RAX);
		xShiftI(e, X_ROL, 4, R(RAX), 16);
		xStore(e, 4, to, RAX);
	} else {
		sseLoad(e, size, XMM0, from);
		sseSwapBytes(e, XMM0, XMM1);
		sseStore(e, size, to, XMM0);
	}
	uint8_t* done = xJmp(e, NULL);
	patchHere(e, wraps);
	if (unaligned) {
		patchHere(e, unaligned);
	}
	xMovRR(e, 8, RDI, RBX);
	xMovI(e, RSI, op);
	callRaw(e, store ? (const void*) rspVectorStore : (const void*) rspVectorLoad);
	if (acc.cached) {
		vLoadAcc(e);
	}
	patchHere(e, done);
	return true;
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
				storeConst(e, rd, (pc + 8) & 0xFFF);
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
		// The rest act as SRLV rd, rs, rs.
		if (rd) {
			xLoad(e, 4, RCX, gpr(rs));
			xMovRR(e, 4, RAX, RCX);
			xShiftCl(e, X_SHR, 4, R(RAX));
			xStore(e, 4, gpr(rd), RAX);
		}
		return PLAIN;
	}
	case 0x01:
		switch (rt) {
		case 0x00: case 0x01: case 0x10: case 0x11:
			xAluI(e, X_CMP, 4, gpr(rs), 0);
			if (rt & 0x10) {
				storeConst(e, 31, (pc + 8) & 0xFFF);
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
			storeConst(e, 31, (pc + 8) & 0xFFF);
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
		if ((op & (1 << 25)) && emitVectorOp(e, op)) {
			return PLAIN;
		}
		if ((op & (1 << 25)) && rspVectorFn(op)) {
			xMovRR(e, 8, RDI, R12);
			xLea(e, 8, RSI, RSP_AT(vr[(op >> 6) & 31]));
			xLea(e, 8, RDX, RSP_AT(vr[(op >> 11) & 31]));
			xLea(e, 8, RCX, RSP_AT(vr[(op >> 16) & 31]));
			xMovI64(e, R8, (uint64_t) (uintptr_t) rspElementMask(op));
			callC(e, rspVectorFn(op));
			return PLAIN;
		}
		xMovRR(e, 8, RDI, RBX);
		xMovI(e, RSI, op);
		callC(e, rspCop2);
		return PLAIN;
	case 0x20: case 0x21: case 0x23: case 0x24: case 0x25: case 0x27:
	case 0x28: case 0x29: case 0x2B:
		emitMemory(e, op);
		return PLAIN;
	case 0x32: case 0x3A:
		if (emitVectorMemory(e, op, op >> 26 == 0x3A)) {
			return PLAIN;
		}
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
	memcpy((uint8_t*) p + LANE_BITS_AT, laneBits, sizeof(laneBits));
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

void rspArenaFree(struct N64* n) {
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

// What an instruction does to the accumulator: 1 it may read it (C calls count as reading it), 2 it
// overwrites all of it first.
static int accUse(uint32_t op) {
	if ((op >> 26) != 0x12 || !(op & (1 << 25))) {
		return 0;
	}
	int f = op & 0x3F;
	if (f == 0x00 || f == 0x01 || (f >= 0x04 && f <= 0x07)) {
		return 2;
	}
	bool lowOnly = f == 0x10 || f == 0x11 || f == 0x14 || f == 0x15 || (f >= 0x20 && f <= 0x25) || (f >= 0x27 && f <= 0x2D);
	return lowOnly ? 0 : 1;
}

// Whether the block may leave after the instruction: the last, a halt or a rewritten IMEM after a
// COP0 write or BREAK, and in lle every instruction (the clock's limit).
static bool exitsAfter(uint32_t op, bool last) {
	return last || !plainTiming || ((op >> 26) == 0x10 && ((op >> 21) & 31) == 0x04) || ((op >> 26) == 0x00 && (op & 0x3F) == 0x0D);
}

static void accLiveness(const uint8_t* ops, uint32_t count) {
	bool live = true;
	for (uint32_t k = count; k-- > 0;) {
		uint32_t op = be32(&ops[k * 4]);
		acc.liveAfter[k] = exitsAfter(op, k + 1 == count) || live;
		int use = accUse(op);
		acc.reads[k] = use & 1;
		live = (use & 1) || (!(use & 2) && acc.liveAfter[k]);
	}
}

// Code for count instructions (as IMEM holds them) from start, or NULL when the arena is full.
void* rspCompile(struct N64* n, const uint8_t* ops, uint32_t count, uint32_t start) {
	plainTiming = n->rspPlainTiming;
	uint8_t* arena = arenaGet(n);
	if (!arena || n->rspUsed + BLOCK_CODE_MAX > ARENA_SIZE) {
		return NULL;
	}
	laneBitsAt = arena + LANE_BITS_AT;
	static struct Jit jit;
	struct Jit* j = &jit;
	j->e = (struct Emitter) { arena + n->rspUsed, arena + n->rspUsed, arena + n->rspUsed + BLOCK_CODE_MAX };
	j->exitCount = 0;
	j->leave = n->rspLeave;
	struct Emitter* e = &j->e;

	accLiveness(ops, count);
	acc.cached = acc.dirty = false;
	bool delay = false;
	uint32_t prev = 0;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = be32(&ops[k * 4]), pc = start + k * 4;
		bool last = k + 1 == count;
		acc.k = (int) k;
		if (k == 0 && plainTiming) {
			loadTiming(e);
			emitTiming(e, op, 0, pc);
		} else if (k == 0) {
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
			j->exits[j->exitCount - 1].pcsStored = pcsStored;
		} else if (!plainTiming) {
			// lle stops at the instruction that reaches the limit; mle and hle at the block's end.
			xAluRR(e, X_CMP, 8, R(R13), R14);
			exitAt(j, xJcc(e, CC_AE, NULL), op, pc, kind, delay);
			j->exits[j->exitCount - 1].pcsStored = pcsStored;
		}
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
		if (x->accDirty) {
			vStoreAcc(e);
		}
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
