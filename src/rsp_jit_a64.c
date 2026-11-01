#include "n64.h"
#include "a64/emit.h"

#include <stddef.h>
#include <sys/mman.h>

// AArch64 code for blocks of RSP instructions, as rsp_jit_x64.c does it, but with every
// instruction's timing compiled in (the first one's pairing checked against lastDesc); scalar instructions and DMEM loads and stores
// inline, the vector unit and COP0 through C but COP0's SP status, DMA and semaphore reads.
//
// Registers: x19 = n, x20 = &n->rsp, x21 = clock, x22 = the limit, x23 = lastStart,
// w24 = lastAlone, x25 = DMEM, w26 = IMEM's generation before a call, x27 = the vector registers
// (vr, then the accumulator, VCO, VCC and VCE), x28 = vectorConstants.

enum { ARENA_SIZE = 32 << 20, BLOCK_CODE_MAX = 32 << 10, SHARED_SIZE = 256, BLOCK_MAX = 64 };

enum { N = 19, RSP = 20, CLOCK = 21, LIMIT = 22, START = 23, ALONE = 24, DMEM = 25, GEN = 26, VR = 27, VCONST = 28, IP = 16 };

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

// The accumulator stays in v3-v5 between the block's vector instructions: loaded when one first
// wants it, written back where something else may read it (a C call, the block's exits), and not at
// all where it is dead: the instructions after overwrite all of it before any reads it.
static struct {
	bool cached, dirty;
	// For each instruction of the block: whether it may read the accumulator, and whether the
	// accumulator may be read after it (before something overwrites all of it).
	bool reads[64], liveAfter[64];
	int k;
} acc;

static void vStoreAcc(struct Emitter* e);

// Before a C call: the accumulator back in memory if anything may read it there; the call takes
// v3-v5.
static void accBeforeCall(struct Emitter* e) {
	if (acc.dirty && (acc.reads[acc.k] || acc.liveAfter[acc.k])) {
		vStoreAcc(e);
	}
	acc.cached = acc.dirty = false;
}

static void callRaw(struct Emitter* e, const void* f) {
	aMovImm64(e, IP, (uint64_t) (uintptr_t) f);
	aBlr(e, IP);
}

static void callC(struct Emitter* e, const void* f) {
	accBeforeCall(e);
	callRaw(e, f);
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
	bool delay, last, halt, pcsStored, accDirty;
};

struct Jit {
	struct Emitter e;
	struct Exit exits[2 * BLOCK_MAX];
	int exitCount;
	uint8_t* leave;
};

// MFC0 of SP_STATUS, DMA_FULL, DMA_BUSY or the semaphore, as busRead32 gives them at the clock.
static void emitSpRead(struct Emitter* e, int reg, int rt) {
	unsigned at;
	if (reg == 7) {
		int const base = fieldBase(e, N64_OFF(sp.semaphore), 4, &at);
		aLdr(e, 0, base, at);
		aMovImm(e, 1, 1);
		aStr(e, 1, base, at);
	} else {
		if (reg == 4) {
			loadN32(e, 0, N64_OFF(sp.status));
		}
		for (int flag = 5; flag <= 6; ++flag) {
			if (reg != 4 && reg != flag) {
				continue;
			}
			int const base = fieldBase(e, flag == 5 ? N64_OFF(sp.dmaFullUntil) : N64_OFF(sp.dmaUntil), 8, &at);
			aLdrX(e, 1, base, at);
			aDp(e, A_SUBS | W64, ZR, CLOCK, 1, S_LSL, 0);
			aCset(e, reg == 4 ? 1 : 0, C_LO);
			if (reg == 4) {
				aDp(e, A_ORR, 0, 0, 1, S_LSL, flag == 5 ? 3 : 2);
			}
		}
	}
	if (rt) {
		aStr(e, 0, RSP, gprOff(rt));
	}
}

static void exitAt(struct Jit* j, uint8_t* site, uint32_t op, uint32_t pc, enum Kind kind, bool delay) {
	j->exits[j->exitCount++] = (struct Exit) { site, op, pc, kind, delay, false, false, false, acc.dirty };
}

// rspTimingStep for an instruction whose predecessor (prev, at pc - 4) is in the same block, which
// issued alone or paired as w24 says. Paired when it may and all it reads was ready at lastStart.
// rspPlainTiming, for the block being compiled.
static bool plainTiming;

static void emitTimingAs(struct Emitter* e, uint32_t op, bool pairable) {
	if (plainTiming) {
		aDpImm(e, A_ADD_IMM | W64, CLOCK, CLOCK, 1 + rspIsBranch(op));
		return;
	}
	int scalar[8], vector[16], nv, written[16], latency[16], nw;
	int ns = rspTimingRegs(op, scalar, vector, &nv, written, latency, &nw);
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

static void emitTiming(struct Emitter* e, uint32_t op, uint32_t prev, uint32_t pc) {
	emitTimingAs(e, op, rspPairable(prev, (pc - 4) & 0xFFC, op));
}

// The first instruction's timing: whether it may pair with the instruction before, which another
// block or the interpreter left in lastAlone, lastPc, inDelay and lastDesc (as rspTimingStep's
// pairs() decides it), then as emitTiming.
static void emitEntryTiming(struct Emitter* e, uint32_t op) {
	loadTiming(e);
	uint16_t self = rspTimingDesc(op);
	if (self & 4) {
		emitTimingAs(e, op, false);
		return;
	}
	int rt = (op >> 16) & 31, rd = (op >> 11) & 31, sa = (op >> 6) & 31, code = op >> 26;
	int forbidden[3], count = 0;
	if (self & 2) {
		forbidden[count++] = rt, forbidden[count++] = rd, forbidden[count++] = sa;
	} else {
		if (code == 0x3A || code == 0x32) {
			forbidden[count++] = rt;
		}
		if (code == 0x12) {
			forbidden[count++] = rd;
		}
	}
	uint8_t* alone[8];
	int sites = 0;
	alone[sites++] = aCbz(e, ALONE, NULL);
	aLdr(e, 0, RSP, RSP_OFF(lastPc));
	alone[sites++] = aTbnz(e, 0, 2, NULL);
	aLdrb(e, 0, RSP, RSP_OFF(inDelay));
	alone[sites++] = aCbnz(e, 0, NULL);
	// Not a branch, not CFC2 or CTC2, and the other unit (vector or scalar) than op's.
	aLdrh(e, 0, RSP, RSP_OFF(lastDesc));
	aLogicImm(e, A_AND_IMM, 1, 0, 7);
	aDpImm(e, A_SUBS_IMM, ZR, 1, self & 2 ? 0 : 2);
	alone[sites++] = aBcond(e, C_NE, NULL);
	// Nor reading a vector register it writes.
	aLsrImm(e, 1, 0, 8);
	for (int i = 0; i < count; ++i) {
		aDpImm(e, A_SUBS_IMM, ZR, 1, (unsigned) forbidden[i] + 1);
		alone[sites++] = aBcond(e, C_EQ, NULL);
	}
	emitTimingAs(e, op, true);
	uint8_t* done = aB(e, NULL);
	for (int i = 0; i < sites; ++i) {
		patchHere(e, alone[i]);
	}
	emitTimingAs(e, op, false);
	patchHere(e, done);
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

// The vector unit in NEON, as rsp_vu.c computes it: lanes are .8H, x27 points at vr. Each
// instruction loads what it reads and stores what it writes; v0 = vs, v1 = the element of vt,
// v2 = the result, v3-v5 = the accumulator's high, middle and low slices, v16-v31 scratch.

// Byte indices for TBL for the quarter elements (4-7), then each lane's bit.
static const uint16_t vectorConstants[5 * 8] __attribute__((aligned(16))) = {
	0x0100, 0x0100, 0x0100, 0x0100, 0x0908, 0x0908, 0x0908, 0x0908,
	0x0302, 0x0302, 0x0302, 0x0302, 0x0B0A, 0x0B0A, 0x0B0A, 0x0B0A,
	0x0504, 0x0504, 0x0504, 0x0504, 0x0D0C, 0x0D0C, 0x0D0C, 0x0D0C,
	0x0706, 0x0706, 0x0706, 0x0706, 0x0F0E, 0x0F0E, 0x0F0E, 0x0F0E,
	1, 2, 4, 8, 16, 32, 64, 128,
};

enum { VR_ACCH = 512, VR_ACCM = 528, VR_ACCL = 544, VR_VCO = 560, VR_VCC = 562, VR_VCE = 564, CONST_LANES = 64 };

// Three-register operations on .8H (or .16B for the bitwise ones), and .4S/.4H widening forms.
enum {
	V_ADD = 0x4E608400, V_SUB = 0x6E608400, V_MUL = 0x4E609C00,
	V_SMULL = 0x0E60C000, V_SMULL2 = 0x4E60C000, V_UMULL = 0x2E60C000, V_UMULL2 = 0x6E60C000,
	V_UZP2 = 0x4E405800, V_ZIP1 = 0x4E403800, V_ZIP2 = 0x4E407800, V_TRN1 = 0x4E402800, V_TRN2 = 0x4E406800,
	V_AND = 0x4E201C00, V_ORR = 0x4EA01C00, V_EOR = 0x6E201C00, V_BIC = 0x4E601C00, V_ORN = 0x4EE01C00,
	V_BIT = 0x6EA01C00, V_BIF = 0x6EE01C00, V_BSL = 0x6E601C00,
	V_CMEQ = 0x6E608C00, V_CMGT = 0x4E603400, V_CMGE = 0x4E603C00, V_CMHI = 0x6E603400, V_CMTST = 0x4E608C00,
	V_SADDL = 0x0E600000, V_SADDL2 = 0x4E600000, V_SSUBL = 0x0E602000, V_SSUBL2 = 0x4E602000,
	V_SADDW = 0x0E601000, V_SADDW2 = 0x4E601000, V_SSUBW = 0x0E603000, V_SSUBW2 = 0x4E603000,
	V_TBL = 0x4E000000,
};

// Two-register operations: compares with zero, NOT, NEG, the narrowing SQXTN and the lane sum.
enum {
	V_CMEQ0 = 0x4E609800, V_CMLT0 = 0x4E60A800, V_CMGT0 = 0x4E608800, V_CMGE0 = 0x6E608800, V_CMLE0 = 0x6E609800,
	V_NOT = 0x6E205800, V_NEG = 0x6E60B800, V_SQXTN = 0x0E614800, V_SQXTN2 = 0x4E614800, V_ADDV = 0x4E71B800,
};

static void v3(struct Emitter* e, uint32_t op, int d, int n, int m) {
	emit32(e, op | (m << 16) | (n << 5) | d);
}

static void v2(struct Emitter* e, uint32_t op, int d, int n) {
	emit32(e, op | (n << 5) | d);
}

// Shifts of every lane by an immediate.
static void vSshr(struct Emitter* e, int d, int n, int shift) {
	emit32(e, 0x4F000400 | ((32 - shift) << 16) | (n << 5) | d);
}

static void vUshr(struct Emitter* e, int d, int n, int shift) {
	emit32(e, 0x6F000400 | ((32 - shift) << 16) | (n << 5) | d);
}

static void vShl(struct Emitter* e, int d, int n, int shift) {
	emit32(e, 0x4F005400 | ((16 + shift) << 16) | (n << 5) | d);
}

// Every lane to 0x8000; all bits clear; lane i of n everywhere; w to every lane.
static void vMin(struct Emitter* e, int d) {
	emit32(e, 0x4F04A400 | d);
}

static void vZero(struct Emitter* e, int d) {
	emit32(e, 0x4F00E400 | d);
}

static void vDupLane(struct Emitter* e, int d, int n, int lane) {
	emit32(e, 0x4E000400 | ((lane << 2 | 2) << 16) | (n << 5) | d);
}

static void vDupW(struct Emitter* e, int d, int w) {
	emit32(e, 0x4E020C00 | (w << 5) | d);
}

static void vLoad(struct Emitter* e, int v, int base, unsigned off) {
	emit32(e, 0x3DC00000 | (off / 16) << 10 | (base << 5) | v);
}

static void vStore(struct Emitter* e, int v, int base, unsigned off) {
	emit32(e, 0x3D800000 | (off / 16) << 10 | (base << 5) | v);
}

// w = the lanes of mask v as bits (bits in rsp_vu.c), and v = the mask of bits w (maskOf); t is scratch.
static void vBits(struct Emitter* e, int w, int v, int t) {
	vLoad(e, t, VCONST, CONST_LANES);
	v3(e, V_AND, t, v, t);
	v2(e, V_ADDV, t, t);
	emit32(e, 0x0E023C00 | (t << 5) | w);
}

static void vMaskOf(struct Emitter* e, int v, int w, int t) {
	vDupW(e, v, w);
	vLoad(e, t, VCONST, CONST_LANES);
	v3(e, V_CMTST, v, v, t);
}

// d = the high halves of the products a * b: signed by signed, unsigned by unsigned, and signed by
// unsigned (or the other way round), as mulhiSS, mulhiUU, mulhiSU and mulhiUS.
static void vMulhi(struct Emitter* e, int d, int a, int b, bool signedA, bool signedB) {
	bool s = signedA || signedB;
	v3(e, s ? V_SMULL : V_UMULL, 16, a, b);
	v3(e, s ? V_SMULL2 : V_UMULL2, 17, a, b);
	v3(e, V_UZP2, d, 16, 17);
	if (signedA != signedB) {
		// The unsigned operand's sign bit counted as 0x10000: add the signed one where it is set.
		vSshr(e, 18, signedA ? b : a, 15);
		v3(e, V_AND, 18, 18, signedA ? a : b);
		v3(e, V_ADD, d, d, 18);
	}
}

// acc (v3-v5) += (h, m, l), wrapping at 48 bits, as accAdd: carries from the low and middle
// slices (masks, so subtracting one adds 1). h, m and l must not be v24-v27 or v31.
static void vAccAdd(struct Emitter* e, int h, int m, int l) {
	v3(e, V_ADD, 24, 5, l);
	v3(e, V_CMHI, 25, 5, 24);
	v3(e, V_ADD, 26, 4, m);
	v3(e, V_CMHI, 27, 4, 26);
	v3(e, V_SUB, 4, 26, 25);
	v2(e, V_CMEQ0, 31, 4);
	v3(e, V_AND, 25, 25, 31);
	v3(e, V_ADD, 3, 3, h);
	v3(e, V_SUB, 3, 3, 27);
	v3(e, V_SUB, 3, 3, 25);
	v3(e, V_ORR, 5, 24, 24);
}

// d = bits 16-47 of the accumulator saturated to 16 signed bits (clampSigned), the low slice when
// they fit (clampLow), or saturated to 16 unsigned bits (clampUnsigned).
static void vClampSigned(struct Emitter* e, int d) {
	v3(e, V_ZIP1, 24, 4, 3);
	v3(e, V_ZIP2, 25, 4, 3);
	v2(e, V_SQXTN, d, 24);
	v2(e, V_SQXTN2, d, 25);
}

static void vClampLow(struct Emitter* e, int d) {
	vSshr(e, 24, 4, 15);
	v3(e, V_CMEQ, 24, 3, 24);
	v2(e, V_CMGE0, d, 3);
	v3(e, V_BIT, d, 5, 24);
}

static void vClampUnsigned(struct Emitter* e, int d) {
	v2(e, V_CMLT0, 24, 3);
	v2(e, V_CMGT0, 25, 3);
	v2(e, V_CMLT0, 26, 4);
	v3(e, V_ORR, 25, 25, 26);
	v3(e, V_ORR, d, 4, 25);
	v3(e, V_BIC, d, d, 24);
}

static void vLoadAcc(struct Emitter* e) {
	vLoad(e, 3, VR, VR_ACCH);
	vLoad(e, 4, VR, VR_ACCM);
	vLoad(e, 5, VR, VR_ACCL);
}

static void vStoreAcc(struct Emitter* e) {
	vStore(e, 3, VR, VR_ACCH);
	vStore(e, 4, VR, VR_ACCM);
	vStore(e, 5, VR, VR_ACCL);
}

// The accumulator in v3-v5 for an instruction that reads it.
static void accWant(struct Emitter* e) {
	if (!acc.cached) {
		vLoadAcc(e);
		acc.cached = true;
	}
}

// The accumulator's low slice set from vector register v.
static void accSetLow(struct Emitter* e, int v) {
	if (acc.cached) {
		if (v != 5) {
			v3(e, V_ORR, 5, v, v);
		}
		acc.dirty = true;
	} else {
		vStore(e, v, VR, VR_ACCL);
	}
}

// The vector instructions compiled to NEON; false for the rest, which call their C functions.
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
			v3(e, V_ORR, 2, 3 + el - 8, 3 + el - 8);
		} else if (el >= 8 && el <= 10) {
			vLoad(e, 2, VR, VR_ACCH + 16 * (el - 8));
		} else {
			vZero(e, 2);
		}
		vStore(e, 2, VR, 16 * vd);
		return true;
	}
	vLoad(e, 0, VR, 16 * vs);
	vLoad(e, 1, VR, 16 * vt);
	if (el == 2 || el == 3) {
		v3(e, el == 2 ? V_TRN1 : V_TRN2, 1, 1, 1);
	} else if (el >= 4 && el <= 7) {
		vLoad(e, 16, VCONST, 16 * (el - 4));
		v3(e, V_TBL, 1, 1, 16);
	} else if (el >= 8) {
		vDupLane(e, 1, 1, el & 7);
	}
	switch (func) {
	case 0x00: case 0x01: case 0x08: case 0x09:
		// VMULF, VMULU, VMACF, VMACU: acc (+)= s * t * 2, rounded when not adding.
		vMulhi(e, 17, 0, 1, true, true);
		v3(e, V_MUL, 16, 0, 1);
		vShl(e, 18, 16, 1);
		vShl(e, 19, 17, 1);
		vUshr(e, 20, 16, 15);
		v3(e, V_ORR, 19, 19, 20);
		vMin(e, 20);
		v3(e, V_CMEQ, 21, 0, 20);
		v3(e, V_CMEQ, 22, 1, 20);
		v3(e, V_AND, 21, 21, 22);
		if (func < 0x08) {
			v3(e, V_EOR, 5, 18, 20);
			vUshr(e, 22, 18, 15);
			v3(e, V_ADD, 4, 19, 22);
			vSshr(e, 3, 4, 15);
			v3(e, V_BIC, 3, 3, 21);
		} else {
			accWant(e);
			vSshr(e, 28, 19, 15);
			v3(e, V_BIC, 28, 28, 21);
			vAccAdd(e, 28, 19, 18);
		}
		acc.cached = acc.dirty = true;
		if (func & 1) {
			vClampUnsigned(e, 2);
		} else {
			vClampSigned(e, 2);
		}
		break;
	case 0x04:
		vMulhi(e, 5, 0, 1, false, false);
		vZero(e, 3);
		vZero(e, 4);
		acc.cached = acc.dirty = true;
		v3(e, V_ORR, 2, 5, 5);
		break;
	case 0x05: case 0x06:
		// VMUDM, VMUDN: the middle slice from a mixed product, the low from the plain one.
		vMulhi(e, 4, 0, 1, func == 0x05, func == 0x06);
		v3(e, V_MUL, 5, 0, 1);
		vSshr(e, 3, 4, 15);
		acc.cached = acc.dirty = true;
		v3(e, V_ORR, 2, func == 0x05 ? 4 : 5, func == 0x05 ? 4 : 5);
		break;
	case 0x07:
		vMulhi(e, 3, 0, 1, true, true);
		v3(e, V_MUL, 4, 0, 1);
		vZero(e, 5);
		acc.cached = acc.dirty = true;
		vClampSigned(e, 2);
		break;
	case 0x0C:
		accWant(e);
		vMulhi(e, 28, 0, 1, false, false);
		vZero(e, 29);
		vAccAdd(e, 29, 29, 28);
		acc.cached = acc.dirty = true;
		vClampLow(e, 2);
		break;
	case 0x0D: case 0x0E:
		accWant(e);
		vMulhi(e, 28, 0, 1, func == 0x0D, func == 0x0E);
		v3(e, V_MUL, 29, 0, 1);
		vSshr(e, 30, 28, 15);
		vAccAdd(e, 30, 28, 29);
		acc.cached = acc.dirty = true;
		if (func == 0x0D) {
			vClampSigned(e, 2);
		} else {
			vClampLow(e, 2);
		}
		break;
	case 0x0F:
		accWant(e);
		vMulhi(e, 28, 0, 1, true, true);
		v3(e, V_MUL, 29, 0, 1);
		vZero(e, 30);
		vAccAdd(e, 28, 29, 30);
		acc.cached = acc.dirty = true;
		vClampSigned(e, 2);
		break;
	case 0x10: case 0x11: {
		// VADD, VSUB: with VCO's carries, the sum wrapping into the accumulator and saturated into vd.
		bool sub = func == 0x11;
		aLdrh(e, 0, VR, VR_VCO);
		vMaskOf(e, 16, 0, 17);
		vUshr(e, 16, 16, 15);
		v3(e, sub ? V_SUB : V_ADD, 5, 0, 1);
		v3(e, sub ? V_SUB : V_ADD, 5, 5, 16);
		accSetLow(e, 5);
		v3(e, sub ? V_SSUBL : V_SADDL, 17, 0, 1);
		v3(e, sub ? V_SSUBL2 : V_SADDL2, 18, 0, 1);
		v3(e, sub ? V_SSUBW : V_SADDW, 17, 17, 16);
		v3(e, sub ? V_SSUBW2 : V_SADDW2, 18, 18, 16);
		v2(e, V_SQXTN, 2, 17);
		v2(e, V_SQXTN2, 2, 18);
		aStrh(e, ZR, VR, VR_VCO);
		break;
	}
	case 0x14:
		v3(e, V_ADD, 2, 0, 1);
		v3(e, V_CMHI, 16, 0, 2);
		vBits(e, 0, 16, 17);
		aStrh(e, 0, VR, VR_VCO);
		accSetLow(e, 2);
		break;
	case 0x15:
		v3(e, V_SUB, 2, 0, 1);
		v3(e, V_CMHI, 16, 1, 0);
		vBits(e, 0, 16, 17);
		v2(e, V_CMEQ0, 16, 2);
		v2(e, V_NOT, 16, 16);
		vBits(e, 1, 16, 17);
		aDp(e, A_ORR, 0, 0, 1, S_LSL, 8);
		aStrh(e, 0, VR, VR_VCO);
		accSetLow(e, 2);
		break;
	case 0x20: case 0x21: case 0x22: case 0x23: {
		// VLT, VEQ, VNE, VGE: VCC from the comparison, with VCO's carries and not-equals.
		aLdrh(e, 0, VR, VR_VCO);
		vMaskOf(e, 16, 0, 17);
		aLsrImm(e, 0, 0, 8);
		vMaskOf(e, 18, 0, 17);
		v3(e, V_CMEQ, 19, 0, 1);
		switch (func) {
		case 0x20:
			v3(e, V_CMGT, 20, 1, 0);
			v3(e, V_AND, 21, 19, 18);
			v3(e, V_AND, 21, 21, 16);
			v3(e, V_ORR, 20, 20, 21);
			break;
		case 0x21:
			v3(e, V_BIC, 20, 19, 18);
			break;
		case 0x22:
			v3(e, V_ORN, 20, 18, 19);
			break;
		default:
			v3(e, V_CMGT, 20, 0, 1);
			v3(e, V_AND, 21, 18, 16);
			v3(e, V_BIC, 21, 19, 21);
			v3(e, V_ORR, 20, 20, 21);
		}
		if (func == 0x21) {
			v3(e, V_ORR, 2, 1, 1);
		} else {
			v3(e, V_ORR, 2, 1, 1);
			v3(e, V_BIT, 2, 0, 20);
		}
		vBits(e, 0, 20, 17);
		aStrh(e, 0, VR, VR_VCC);
		aStrh(e, ZR, VR, VR_VCO);
		accSetLow(e, 2);
		break;
	}
	case 0x24: {
		// VCL.
		aLdrh(e, 0, VR, VR_VCC);
		vMaskOf(e, 20, 0, 17);
		aLsrImm(e, 0, 0, 8);
		vMaskOf(e, 21, 0, 17);
		aLdrh(e, 0, VR, VR_VCO);
		vMaskOf(e, 22, 0, 17);
		aLsrImm(e, 0, 0, 8);
		vMaskOf(e, 23, 0, 17);
		aLdrb(e, 0, VR, VR_VCE);
		vMaskOf(e, 24, 0, 17);
		v3(e, V_ADD, 25, 0, 1);
		v3(e, V_CMHI, 26, 0, 25);
		v2(e, V_CMEQ0, 27, 25);
		v3(e, V_ORN, 28, 27, 26);
		v3(e, V_BIC, 29, 27, 26);
		v3(e, V_BSL, 24, 28, 29);
		v3(e, V_BIC, 28, 22, 23);
		v3(e, V_BIT, 20, 24, 28);
		v3(e, V_ORR, 28, 22, 23);
		v3(e, V_CMHI, 29, 1, 0);
		v2(e, V_NOT, 29, 29);
		v3(e, V_BIF, 21, 29, 28);
		v2(e, V_NEG, 30, 1);
		v3(e, V_ORR, 31, 0, 0);
		v3(e, V_BIT, 31, 30, 20);
		v3(e, V_ORR, 2, 0, 0);
		v3(e, V_BIT, 2, 1, 21);
		v3(e, V_BIT, 2, 31, 22);
		vBits(e, 0, 20, 17);
		vBits(e, 1, 21, 17);
		aDp(e, A_ORR, 0, 0, 1, S_LSL, 8);
		aStrh(e, 0, VR, VR_VCC);
		aStrh(e, ZR, VR, VR_VCO);
		aStrb(e, ZR, VR, VR_VCE);
		accSetLow(e, 2);
		break;
	}
	case 0x25: {
		// VCH.
		v3(e, V_EOR, 20, 0, 1);
		v2(e, V_CMLT0, 20, 20);
		v3(e, V_ADD, 21, 0, 1);
		v3(e, V_SUB, 22, 0, 1);
		v2(e, V_CMLT0, 23, 1);
		v2(e, V_CMLE0, 24, 21);
		v3(e, V_ORR, 25, 23, 23);
		v3(e, V_BIT, 25, 24, 20);
		v2(e, V_CMGE0, 26, 22);
		v3(e, V_ORR, 27, 26, 26);
		v3(e, V_BIT, 27, 23, 20);
		v2(e, V_NEG, 28, 1);
		v3(e, V_ORR, 29, 0, 0);
		v3(e, V_BIT, 29, 28, 25);
		v3(e, V_ORR, 2, 0, 0);
		v3(e, V_BIT, 2, 1, 27);
		v3(e, V_BIT, 2, 29, 20);
		vBits(e, 0, 25, 17);
		vBits(e, 1, 27, 17);
		aDp(e, A_ORR, 0, 0, 1, S_LSL, 8);
		aStrh(e, 0, VR, VR_VCC);
		// VCO: the sign lanes, and the lanes not equal: sum not 0 or -1 where signs differ, diff not 0.
		v2(e, V_CMEQ0, 28, 21);
		v2(e, V_NOT, 29, 21);
		v2(e, V_CMEQ0, 29, 29);
		v3(e, V_ORR, 30, 28, 29);
		v2(e, V_NOT, 30, 30);
		v2(e, V_CMEQ0, 31, 22);
		v2(e, V_NOT, 31, 31);
		v3(e, V_BIT, 31, 30, 20);
		vBits(e, 0, 20, 17);
		vBits(e, 1, 31, 17);
		aDp(e, A_ORR, 0, 0, 1, S_LSL, 8);
		aStrh(e, 0, VR, VR_VCO);
		v3(e, V_AND, 29, 29, 20);
		vBits(e, 0, 29, 17);
		aStrb(e, 0, VR, VR_VCE);
		accSetLow(e, 2);
		break;
	}
	case 0x27:
		aLdrh(e, 0, VR, VR_VCC);
		vMaskOf(e, 16, 0, 17);
		v3(e, V_ORR, 2, 1, 1);
		v3(e, V_BIT, 2, 0, 16);
		aStrh(e, ZR, VR, VR_VCO);
		accSetLow(e, 2);
		break;
	default:
		// VAND, VNAND, VOR, VNOR, VXOR, VNXOR.
		v3(e, func <= 0x29 ? V_AND : func <= 0x2B ? V_ORR : V_EOR, 2, 0, 1);
		if (func & 1) {
			v2(e, V_NOT, 2, 2);
		}
		accSetLow(e, 2);
	}
	vStore(e, 2, VR, 16 * vd);
	return true;
}

// SIMD loads and stores of 2, 4, 8 or 16 bytes (H, S, D, Q): at base + wm (uxtw), or at [base].
static uint32_t vectorSize(int size) {
	return size == 2 ? 0x40000000 : size == 4 ? 0x80000000 : size == 8 ? 0xC0000000 : 0x00800000;
}

static void aLdrV(struct Emitter* e, int vt, int base, int wm, int size) {
	emit32(e, 0x3C604800 | vectorSize(size) | (wm << 16) | (base << 5) | vt);
}

static void aStrV(struct Emitter* e, int vt, int base, int wm, int size) {
	emit32(e, 0x3C204800 | vectorSize(size) | (wm << 16) | (base << 5) | vt);
}

static void aLdrVAt(struct Emitter* e, int vt, int base, int size) {
	emit32(e, 0x3D400000 | vectorSize(size) | (base << 5) | vt);
}

static void aStrVAt(struct Emitter* e, int vt, int base, int size) {
	emit32(e, 0x3D000000 | vectorSize(size) | (base << 5) | vt);
}

// REV16 of every byte pair in vt.
static void aRev16V(struct Emitter* e, int vd, int vn) {
	emit32(e, 0x4E201800 | (vn << 5) | vd);
}

// LSV, LLV, LDV and LQV (SSV to SQV) at an even element that fits: the lanes move whole through v0,
// byte-swapped. Addresses past the end of DMEM, and LQV/SQV off a 16-byte boundary, take the
// general path in C.
static bool emitVectorMemory(struct Emitter* e, uint32_t op, bool store) {
	int kind = (op >> 11) & 31, el = (op >> 7) & 15, vt = (op >> 16) & 31;
	if (kind < 1 || kind > 4 || (el & 1) || el + (1 << kind) > 16) {
		return false;
	}
	int size = 1 << kind;
	int32_t offset = ((int32_t) ((op & 0x7F) << 25) >> 25) * size;
	// The C path below takes v3-v5: the accumulator goes back to memory first where it is live.
	if (acc.dirty && acc.liveAfter[acc.k]) {
		vStoreAcc(e);
	}
	acc.dirty = false;
	uint8_t* wraps = dmemAddress(e, (op >> 21) & 31, offset, size);
	uint8_t* unaligned = NULL;
	if (size == 16) {
		aLslImm(e, 0, 1, 28);
		aDpImm(e, A_SUBS_IMM, ZR, 0, 0);
		unaligned = aBcond(e, C_NE, NULL);
	}
	aAddImmX(e, 2, RSP, RSP_OFF(vr[vt][el / 2]));
	if (store) {
		aLdrVAt(e, 0, 2, size);
		aRev16V(e, 0, 0);
		aStrV(e, 0, DMEM, 1, size);
	} else {
		aLdrV(e, 0, DMEM, 1, size);
		aRev16V(e, 0, 0);
		aStrVAt(e, 0, 2, size);
	}
	uint8_t* done = aB(e, NULL);
	patchHere(e, wraps);
	if (unaligned) {
		patchHere(e, unaligned);
	}
	aMovX(e, 0, N);
	aMovImm(e, 1, op);
	callRaw(e, store ? (const void*) rspVectorStore : (const void*) rspVectorLoad);
	if (acc.cached) {
		vLoadAcc(e);
	}
	patchHere(e, done);
	return true;
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
				storeConst(e, rd, (pc + 8) & 0xFFF);
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
		// The rest act as SRLV rd, rs, rs.
		if (rd) {
			aLdr(e, 0, RSP, gprOff(rs));
			aDp(e, A_LSRV, 0, 0, 0, S_LSL, 0);
			aStr(e, 0, RSP, gprOff(rd));
		}
		return PLAIN;
	}
	case 0x01:
		switch (rt) {
		case 0x00: case 0x01: case 0x10: case 0x11:
			aLdr(e, 0, RSP, gprOff(rs));
			aDpImm(e, A_SUBS_IMM, ZR, 0, 0);
			if (rt & 0x10) {
				storeConst(e, 31, (pc + 8) & 0xFFF);
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
			storeConst(e, 31, (pc + 8) & 0xFFF);
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
		if (rs == 0x00 && (rd & 15) >= 4 && (rd & 15) <= 7) {
			emitSpRead(e, rd & 15, rt);
		} else if (rs == 0x00 || rs == 0x04) {
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
		if ((op & (1 << 25)) && emitVectorOp(e, op)) {
			return PLAIN;
		}
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
		if (emitVectorMemory(e, op, op >> 26 == 0x3A)) {
			return PLAIN;
		}
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
	aPush(&e, 29, 30, 96);
	aStp(&e, 19, 20, 16);
	aStp(&e, 21, 22, 32);
	aStp(&e, 23, 24, 48);
	aStp(&e, 25, 26, 64);
	aStp(&e, 27, 28, 80);
	aMovX(&e, N, 0);
	aAddImmX(&e, RSP, N, N64_OFF(rsp));
	aMovX(&e, LIMIT, 2);
	aAddImmX(&e, DMEM, N, N64_OFF(sp.mem));
	aAddImmX(&e, VR, RSP, RSP_OFF(vr));
	aMovImm64(&e, VCONST, (uint64_t) (uintptr_t) vectorConstants);
	aBr(&e, 1);
	n->rspLeave = e.p;
	aLdp(&e, 19, 20, 16);
	aLdp(&e, 21, 22, 32);
	aLdp(&e, 23, 24, 48);
	aLdp(&e, 25, 26, 64);
	aLdp(&e, 27, 28, 80);
	aPop(&e, 29, 30, 96);
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

void* rspCompile(struct N64* n, const uint8_t* ops, uint32_t count, uint32_t start) {
	plainTiming = n->rspPlainTiming;
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

	accLiveness(ops, count);
	acc.cached = acc.dirty = false;
	uint32_t const wait = plainTiming ? rspDmaWait(ops, count) : 0;
	if (wait) {
		aMovX(e, 0, N);
		aMovImm(e, 1, wait);
		aMovX(e, 2, LIMIT);
		callC(e, rspDmaWaitSkip);
	}
	bool delay = false;
	uint32_t prev = 0;
	for (uint32_t k = 0; k < count; ++k) {
		uint32_t op = be32(&ops[k * 4]), pc = start + k * 4;
		bool last = k + 1 == count;
		acc.k = (int) k;
		if (k == 0) {
			emitEntryTiming(e, op);
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
			j->exits[j->exitCount - 1].pcsStored = pcsStored;
		} else if (!plainTiming) {
			// lle stops at the instruction that reaches the limit; mle and hle at the block's end.
			aDp(e, A_SUBS | W64, ZR, CLOCK, LIMIT, S_LSL, 0);
			exitAt(j, aBcond(e, C_HS, NULL), op, pc, kind, delay);
			j->exits[j->exitCount - 1].pcsStored = pcsStored;
		}
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
		if (x->accDirty) {
			vStoreAcc(e);
		}
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
		aMovImm(e, 0, rspTimingDesc(x->op));
		aStrh(e, 0, RSP, RSP_OFF(lastDesc));
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
