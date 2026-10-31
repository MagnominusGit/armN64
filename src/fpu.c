#include "n64.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <string.h>

// The VR4300's FPU as ares has it from hardware tests (nemu64-test): each operation's cause bits and
// the sticky flags in FCR31, an exception instead of the result when one is enabled, the
// unimplemented operation exception for denormal and quiet NaN operands, denormal results (unless FS
// flushes them) and out-of-range conversions, and the register views of both FR modes.

#define FT ((op >> 16) & 31)
#define FS ((op >> 11) & 31)
#define FD ((op >> 6) & 31)

// FCR31: flags from bit 2, enables from 7 and causes from 12 (inexact, underflow, overflow, division
// by zero, invalid), the unimplemented operation cause at 17, the condition at 23, FS at 24.
enum { F_INEXACT = 1, F_UNDERFLOW = 2, F_OVERFLOW = 4, F_DIVZERO = 8, F_INVALID = 16 };
enum { CAUSES = 0x3F000u, CAUSE_E = 1u << 17, FLUSH = 1u << 24 };

static bool fr(struct N64* n) {
	return n->cpu.cop0[CP0_STATUS] & (1 << 26);
}

// The register views: "t" for the moves, loads and stores (with FR clear a 32-bit odd register is the
// upper half of the pair), "s" for an operation's first operand (with FR clear the even register);
// the second operand of float operations and the result as named, a 32-bit result clearing the upper
// half.
static uint32_t getT32(struct N64* n, int r) {
	return fr(n) || !(r & 1) ? (uint32_t) n->cpu.fpr[r] : (uint32_t) (n->cpu.fpr[r & ~1] >> 32);
}

static void setT32(struct N64* n, int r, uint32_t v) {
	uint64_t* f = n->cpu.fpr;
	if (fr(n) || !(r & 1)) {
		f[r] = (f[r] & ~0xFFFFFFFFull) | v;
	} else {
		f[r & ~1] = (f[r & ~1] & 0xFFFFFFFFull) | (uint64_t) v << 32;
	}
}

static uint64_t getT64(struct N64* n, int r) {
	return n->cpu.fpr[fr(n) ? r : r & ~1];
}

static void setT64(struct N64* n, int r, uint64_t v) {
	n->cpu.fpr[fr(n) ? r : r & ~1] = v;
}

static float sOf(struct N64* n, int r) {
	uint32_t v = (uint32_t) n->cpu.fpr[fr(n) ? r : r & ~1];
	float f;
	memcpy(&f, &v, 4);
	return f;
}

static double dOf(struct N64* n, int r) {
	uint64_t v = n->cpu.fpr[fr(n) ? r : r & ~1];
	double d;
	memcpy(&d, &v, 8);
	return d;
}

static float tOf32(struct N64* n, int r) {
	uint32_t v = (uint32_t) n->cpu.fpr[r];
	float f;
	memcpy(&f, &v, 4);
	return f;
}

static double tOf64(struct N64* n, int r) {
	double d;
	memcpy(&d, &n->cpu.fpr[r], 8);
	return d;
}

static void putS(struct N64* n, int r, float f) {
	uint32_t v;
	memcpy(&v, &f, 4);
	n->cpu.fpr[r] = v;
}

static void putD(struct N64* n, int r, double d) {
	memcpy(&n->cpu.fpr[r], &d, 8);
}

// A cause: its bit set; true when its exception is enabled, else its flag set too.
static bool cause(struct N64* n, int bit) {
	uint32_t* f = &n->cpu.fcr31;
	*f |= (uint32_t) bit << 12;
	if (*f & (uint32_t) bit << 7) {
		return true;
	}
	*f |= (uint32_t) bit << 2;
	return false;
}

// While fpuProbe runs an instruction, an exception is noted rather than taken.
static _Thread_local bool probing, raised;

static void raise(struct N64* n) {
	if (probing) {
		raised = true;
	} else {
		cpuException(n, EXC_FPE, 0);
	}
}

// The unimplemented operation exception, which can't be masked.
static void unimplemented(struct N64* n) {
	n->cpu.fcr31 |= CAUSE_E;
	raise(n);
}

// What the host's exception flags (fenv) say of an operation, its result kept unless this returns
// true (the exception taken). Conversions to integers report invalid as unimplemented.
static bool hostExceptions(struct N64* n, int exc, bool conversion) {
	if (!exc) {
		return false;
	}
	uint32_t const f = n->cpu.fcr31;
	if (conversion && (exc & FE_INVALID)) {
		unimplemented(n);
		return true;
	}
	if ((exc & FE_UNDERFLOW) && (!(f & FLUSH) || (f & (F_UNDERFLOW << 7)) || (f & (F_INEXACT << 7)))) {
		unimplemented(n);
		return true;
	}
	bool enabled = false;
	if (exc & FE_DIVBYZERO) {
		enabled |= cause(n, F_DIVZERO);
	}
	if (exc & FE_INEXACT) {
		enabled |= cause(n, F_INEXACT);
	}
	if (exc & FE_UNDERFLOW) {
		enabled |= cause(n, F_UNDERFLOW);
	}
	if (exc & FE_OVERFLOW) {
		enabled |= cause(n, F_OVERFLOW);
	}
	if (exc & FE_INVALID) {
		enabled |= cause(n, F_INVALID);
	}
	if (enabled) {
		raise(n);
	}
	return enabled;
}

// The host's exception flags: MXCSR or FPSR directly, as fenv also goes through the x87's.
#if defined(__x86_64__)
static void hostClear(void) {
	uint32_t m;
	__asm__ volatile("stmxcsr %0" : "=m"(m) : : "memory");
	m &= ~0x3Fu;
	__asm__ volatile("ldmxcsr %0" : : "m"(m) : "memory");
}

static int hostFlags(void) {
	uint32_t m;
	__asm__ volatile("stmxcsr %0" : "=m"(m) : : "memory");
	return (m & 1 ? FE_INVALID : 0) | (m & 4 ? FE_DIVBYZERO : 0) | (m & 8 ? FE_OVERFLOW : 0) | (m & 16 ? FE_UNDERFLOW : 0) |
	       (m & 32 ? FE_INEXACT : 0);
}
#elif defined(__aarch64__)
static void hostClear(void) {
	__asm__ volatile("msr fpsr, xzr" : : : "memory");
}

static int hostFlags(void) {
	uint64_t m;
	__asm__ volatile("mrs %0, fpsr" : "=r"(m) : : "memory");
	return (m & 1 ? FE_INVALID : 0) | (m & 2 ? FE_DIVBYZERO : 0) | (m & 4 ? FE_OVERFLOW : 0) | (m & 8 ? FE_UNDERFLOW : 0) |
	       (m & 16 ? FE_INEXACT : 0);
}
#else
static void hostClear(void) {
	feclearexcept(FE_ALL_EXCEPT);
}

static int hostFlags(void) {
	return fetestexcept(FE_DIVBYZERO | FE_INEXACT | FE_UNDERFLOW | FE_OVERFLOW | FE_INVALID);
}
#endif

// The VR4300's NaNs are the other way round from IEEE's: bit 22 (51) set is signaling.
static bool signalingS(float f) {
	uint32_t v;
	memcpy(&v, &f, 4);
	return isnan(f) && (v >> 22 & 1);
}

static bool signalingD(double d) {
	uint64_t v;
	memcpy(&v, &d, 8);
	return isnan(d) && (v >> 51 & 1);
}

// Operands: a denormal or a quiet NaN is unimplemented, a signaling NaN invalid. False after the
// exception.
static bool inputs(struct N64* n, int ka, int kb, bool snanA, bool snanB) {
	if ((ka == FP_NAN && !snanA) || (kb == FP_NAN && !snanB) || ka == FP_SUBNORMAL || kb == FP_SUBNORMAL) {
		unimplemented(n);
		return false;
	}
	if ((ka == FP_NAN || kb == FP_NAN) && cause(n, F_INVALID)) {
		raise(n);
		return false;
	}
	return true;
}

static bool inputS(struct N64* n, float a) {
	return inputs(n, fpclassify(a), FP_ZERO, signalingS(a), false);
}

static bool inputD(struct N64* n, double a) {
	return inputs(n, fpclassify(a), FP_ZERO, signalingD(a), false);
}

// A denormal result: unimplemented unless FS flushes it (to zero, or the smallest normal number
// toward the rounding) with underflow and inexact. False after the exception.
static bool flushed(struct N64* n) {
	uint32_t const fcr = n->cpu.fcr31;
	if (!(fcr & FLUSH) || (fcr & (F_UNDERFLOW << 7)) || (fcr & (F_INEXACT << 7))) {
		unimplemented(n);
		return false;
	}
	cause(n, F_UNDERFLOW);
	cause(n, F_INEXACT);
	return true;
}

// A result as written: a denormal flushed, any NaN the default NaN.
static bool outputS(struct N64* n, float* f) {
	int const k = fpclassify(*f);
	if (k == FP_SUBNORMAL) {
		if (!flushed(n)) {
			return false;
		}
		bool const negative = signbit(*f);
		int const mode = (int) (n->cpu.fcr31 & 3);
		*f = mode == 2 ? (negative ? -0.0f : FLT_MIN) : mode == 3 ? (negative ? -FLT_MIN : 0.0f) : (negative ? -0.0f : 0.0f);
	} else if (k == FP_NAN) {
		uint32_t const v = 0x7FBFFFFF;
		memcpy(f, &v, 4);
	}
	return true;
}

static bool outputD(struct N64* n, double* d) {
	int const k = fpclassify(*d);
	if (k == FP_SUBNORMAL) {
		if (!flushed(n)) {
			return false;
		}
		bool const negative = signbit(*d);
		int const mode = (int) (n->cpu.fcr31 & 3);
		*d = mode == 2 ? (negative ? -0.0 : DBL_MIN) : mode == 3 ? (negative ? -DBL_MIN : 0.0) : (negative ? -0.0 : 0.0);
	} else if (k == FP_NAN) {
		uint64_t const v = 0x7FF7FFFFFFFFFFFFull;
		memcpy(d, &v, 8);
	}
	return true;
}

static const int roundModes[4] = { FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD };

// To an integral value: nearest (ties to even), toward zero, up, down.
static double roundWith(double v, int mode) {
	switch (mode) {
	case 0: {
		double r = round(v);
		return fabs(v - trunc(v)) == 0.5 ? 2.0 * round(v / 2.0) : r;
	}
	case 1:
		return trunc(v);
	case 2:
		return ceil(v);
	default:
		return floor(v);
	}
}

// Cycles an operation holds the pipeline beyond the first (VR4300 manual, as ares has them).
uint32_t fpuCycles(int fmt, int func) {
	bool dbl = fmt == 17;
	switch (func) {
	case 0:
	case 1:
		return 2;
	case 2:
		return dbl ? 7 : 4;
	case 3:
	case 4:
		return dbl ? 57 : 28;
	case 32:
		return fmt == 17 ? 1 : 4;
	case 33:
		return fmt == 16 ? 0 : 4;
	default:
		return (func >= 8 && func <= 15) || func == 36 || func == 37 ? 4 : 0;
	}
}

// C.cond: with a NaN, invalid when the compare signals (functions 8-15) or the NaN does, and the
// condition the unordered bit; else less and equal as asked.
static void compare(struct N64* n, double a, double b, bool snanA, bool snanB, int func) {
	struct CPU* c = &n->cpu;
	bool cond;
	if (isnan(a) || isnan(b)) {
		bool const signal = func & 8;
		if (((isnan(a) && (signal || snanA)) || (isnan(b) && (signal || snanB))) && cause(n, F_INVALID)) {
			raise(n);
			return;
		}
		cond = func & 1;
	} else {
		cond = ((func & 4) && a < b) || ((func & 2) && a == b);
	}
	c->fcr31 = (c->fcr31 & ~(1u << 23)) | (uint32_t) cond << 23;
}

// ROUND, TRUNC, CEIL, FLOOR and CVT to a word or long: denormal, infinite, NaN or past the integer's
// range (2^31, 2^53) is unimplemented; inexact when rounding changed the value.
static void toInteger(struct N64* n, int fd, double v, int k, int mode, bool word) {
	if (k == FP_SUBNORMAL || k == FP_INFINITE || k == FP_NAN || (word ? v >= 0x1p31 || v < -0x1p31 : v >= 0x1p53 || v <= -0x1p53)) {
		unimplemented(n);
		return;
	}
	double const r = roundWith(v, mode);
	int const exc = (word && (r >= 0x1p31 || r < -0x1p31) ? FE_INVALID : 0) | (r != v ? FE_INEXACT : 0);
	if (hostExceptions(n, exc, word)) {
		return;
	}
	n->cpu.fpr[fd] = word ? (uint64_t) (uint32_t) (int32_t) r : (uint64_t) (int64_t) r;
}

static void single(struct N64* n, uint32_t op, int func) {
	float const a = sOf(n, FS);
	if (func <= 3) {
		float const b = tOf32(n, FT);
		if (!inputs(n, fpclassify(a), fpclassify(b), signalingS(a), signalingS(b))) {
			return;
		}
		hostClear();
		volatile float va = a, vb = b, vr;
		switch (func) {
		case 0:
			vr = va + vb;
			break;
		case 1:
			vr = va - vb;
			break;
		case 2:
			vr = va * vb;
			break;
		default:
			vr = va / vb;
		}
		float r = vr;
		if (!hostExceptions(n, hostFlags(), false) && outputS(n, &r)) {
			putS(n, FD, r);
		}
		return;
	}
	if (func >= 0x30) {
		float const b = tOf32(n, FT);
		compare(n, a, b, signalingS(a), signalingS(b), func);
		return;
	}
	if (func >= 8 && func <= 15) {
		toInteger(n, FD, a, fpclassify(a), func & 3, func >= 12);
		return;
	}
	if (func == 0x24 || func == 0x25) {
		toInteger(n, FD, a, fpclassify(a), (int) (n->cpu.fcr31 & 3), func == 0x24);
		return;
	}
	if (func != 4 && func != 5 && func != 7 && func != 0x21) {
		unimplemented(n);
		return;
	}
	if (!inputS(n, a)) {
		return;
	}
	if (func == 0x21) {
		double r = a;
		if (outputD(n, &r)) {
			putD(n, FD, r);
		}
		return;
	}
	hostClear();
	volatile float va = a, vr = func == 4 ? sqrtf(va) : func == 5 ? fabsf(va) : -va;
	float r = vr;
	if (!hostExceptions(n, hostFlags(), false) && outputS(n, &r)) {
		putS(n, FD, r);
	}
}

static void dbl(struct N64* n, uint32_t op, int func) {
	double const a = dOf(n, FS);
	if (func <= 3) {
		double const b = tOf64(n, FT);
		if (!inputs(n, fpclassify(a), fpclassify(b), signalingD(a), signalingD(b))) {
			return;
		}
		hostClear();
		volatile double va = a, vb = b, vr;
		switch (func) {
		case 0:
			vr = va + vb;
			break;
		case 1:
			vr = va - vb;
			break;
		case 2:
			vr = va * vb;
			break;
		default:
			vr = va / vb;
		}
		double r = vr;
		if (!hostExceptions(n, hostFlags(), false) && outputD(n, &r)) {
			putD(n, FD, r);
		}
		return;
	}
	if (func >= 0x30) {
		double const b = tOf64(n, FT);
		compare(n, a, b, signalingD(a), signalingD(b), func);
		return;
	}
	if (func >= 8 && func <= 15) {
		toInteger(n, FD, a, fpclassify(a), func & 3, func >= 12);
		return;
	}
	if (func == 0x24 || func == 0x25) {
		toInteger(n, FD, a, fpclassify(a), (int) (n->cpu.fcr31 & 3), func == 0x24);
		return;
	}
	if (func != 4 && func != 5 && func != 7 && func != 0x20) {
		unimplemented(n);
		return;
	}
	if (!inputD(n, a)) {
		return;
	}
	hostClear();
	if (func == 0x20) {
		volatile double va = a;
		volatile float vr = (float) va;
		float r = vr;
		if (!hostExceptions(n, hostFlags(), false) && outputS(n, &r)) {
			putS(n, FD, r);
		}
		return;
	}
	volatile double va = a, vr = func == 4 ? sqrt(va) : func == 5 ? fabs(va) : -va;
	double r = vr;
	if (!hostExceptions(n, hostFlags(), false) && outputD(n, &r)) {
		putD(n, FD, r);
	}
}

// CVT.S and CVT.D of a word or long (a long past 2^55 is unimplemented); the rest of W and L is.
static void integer(struct N64* n, uint32_t op, int fmt, int func) {
	int64_t const v = fmt == 20 ? (int32_t) (uint32_t) n->cpu.fpr[fr(n) ? FS : FS & ~1] : (int64_t) getT64(n, FS);
	if ((func != 0x20 && func != 0x21) || (fmt == 21 && (v >= 0x0080000000000000ll || v < -0x0080000000000000ll))) {
		unimplemented(n);
		return;
	}
	hostClear();
	volatile int64_t vv = v;
	if (func == 0x20) {
		volatile float vr = (float) vv;
		float r = vr;
		if (!hostExceptions(n, hostFlags(), false) && outputS(n, &r)) {
			putS(n, FD, r);
		}
	} else {
		volatile double vr = (double) vv;
		double r = vr;
		if (!hostExceptions(n, hostFlags(), false) && outputD(n, &r)) {
			putD(n, FD, r);
		}
	}
}

// CTC1 FCR31: the host's rounding mode follows; a cause written with its exception enabled, or the
// unimplemented one, raises it.
static void writeFcr31(struct N64* n, uint32_t v) {
	struct CPU* c = &n->cpu;
	static _Thread_local int hostMode = -1;
	c->fcr31 = v & 0x0183FFFF;
	if ((int) (c->fcr31 & 3) != hostMode) {
		hostMode = (int) (c->fcr31 & 3);
		fesetround(roundModes[hostMode]);
	}
	if ((c->fcr31 >> 12 & c->fcr31 >> 7 & 0x1F) || (c->fcr31 & CAUSE_E)) {
		raise(n);
	}
}

// BC1 starts with the causes clear, as every operation (but MOV) does.
void fpuClearCauses(struct N64* n) {
	n->cpu.fcr31 &= ~CAUSES;
}

void fpuExecute(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	int fmt = (op >> 21) & 31;
	switch (fmt) {
	case 0:
		c->gpr[FT] = sext32(getT32(n, FS));
		return;
	case 1:
		c->gpr[FT] = getT64(n, FS);
		return;
	case 2:
		c->gpr[FT] = FS == 31 ? sext32(c->fcr31) : FS == 0 ? 0xA00 : 0;
		return;
	case 4:
		setT32(n, FS, (uint32_t) c->gpr[FT]);
		return;
	case 5:
		setT64(n, FS, c->gpr[FT]);
		return;
	case 6:
		if (FS == 31) {
			writeFcr31(n, (uint32_t) c->gpr[FT]);
		}
		return;
	}
	int func = op & 0x3F;
	if ((fmt == 16 || fmt == 17) && func == 6) {
		c->fpr[FD] = getT64(n, FS);
		return;
	}
	c->fcr31 &= ~CAUSES;
	if (fmt == 16 || fmt == 17 || fmt == 20 || fmt == 21) {
		c->stall += fpuCycles(fmt, func);
	}
	if (fmt == 16) {
		single(n, op, func);
	} else if (fmt == 17) {
		dbl(n, op, func);
	} else if (fmt == 20 || fmt == 21) {
		integer(n, op, fmt, func);
	} else {
		// DCFC1, DCTC1 and the formats there are none of.
		unimplemented(n);
	}
}

// fpuExecute for compiled code: false, with nothing changed, when the instruction raises an exception
// (the interpreter then runs it).
bool fpuProbe(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	uint64_t const fd = c->fpr[FD];
	uint32_t const fcr31 = c->fcr31, stall = c->stall;
	probing = true;
	raised = false;
	fpuExecute(n, op);
	probing = false;
	if (raised) {
		c->fpr[FD] = fd;
		c->fcr31 = fcr31;
		c->stall = stall;
		return false;
	}
	return true;
}
