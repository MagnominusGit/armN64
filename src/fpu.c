#include "n64.h"

#include <fenv.h>
#include <math.h>
#include <string.h>

#define FT ((op >> 16) & 31)
#define FS ((op >> 11) & 31)
#define FD ((op >> 6) & 31)

static bool fr(struct N64* n) {
	return n->cpu.cop0[CP0_STATUS] & (1 << 26);
}

// With Status.FR clear, a 32-bit register n is half of the 64-bit register n & ~1.
static uint32_t get32(struct N64* n, int r) {
	if (fr(n) || !(r & 1)) {
		return (uint32_t) n->cpu.fpr[r];
	}
	return n->cpu.fpr[r & ~1] >> 32;
}

static void set32(struct N64* n, int r, uint32_t v) {
	uint64_t* f = n->cpu.fpr;
	if (fr(n) || !(r & 1)) {
		f[r] = (f[r] & ~0xFFFFFFFFull) | v;
	} else {
		f[r & ~1] = (f[r & ~1] & 0xFFFFFFFFull) | (uint64_t) v << 32;
	}
}

static uint64_t get64(struct N64* n, int r) {
	return n->cpu.fpr[fr(n) ? r : r & ~1];
}

static void set64(struct N64* n, int r, uint64_t v) {
	n->cpu.fpr[fr(n) ? r : r & ~1] = v;
}

static float getS(struct N64* n, int r) {
	uint32_t v = get32(n, r);
	float f;
	memcpy(&f, &v, 4);
	return f;
}

static double getD(struct N64* n, int r) {
	uint64_t v = get64(n, r);
	double d;
	memcpy(&d, &v, 8);
	return d;
}

static void setS(struct N64* n, int r, float f) {
	uint32_t v;
	memcpy(&v, &f, 4);
	set32(n, r, v);
}

static void setD(struct N64* n, int r, double d) {
	uint64_t v;
	memcpy(&v, &d, 8);
	set64(n, r, v);
}

static const int roundModes[4] = { FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD, FE_DOWNWARD };

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

static double source(struct N64* n, int fmt, int r) {
	switch (fmt) {
	case 16:
		return getS(n, r);
	case 17:
		return getD(n, r);
	case 20:
		return (int32_t) get32(n, r);
	default:
		return (double) (int64_t) get64(n, r);
	}
}

static void result(struct N64* n, int fmt, int r, double v) {
	if (fmt == 16) {
		setS(n, r, (float) v);
	} else {
		setD(n, r, v);
	}
}

void fpuExecute(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	int fmt = (op >> 21) & 31;
	switch (fmt) {
	case 0:
		c->gpr[FT] = sext32(get32(n, FS));
		return;
	case 1:
		c->gpr[FT] = get64(n, FS);
		return;
	case 2:
		c->gpr[FT] = FS == 31 ? sext32(c->fcr31) : FS == 0 ? 0xA00 : 0;
		return;
	case 4:
		set32(n, FS, (uint32_t) c->gpr[FT]);
		return;
	case 5:
		set64(n, FS, c->gpr[FT]);
		return;
	case 6:
		if (FS == 31) {
			c->fcr31 = (uint32_t) c->gpr[FT] & 0x0183FFFF;
			fesetround(roundModes[c->fcr31 & 3]);
		}
		return;
	}
	if (fmt != 16 && fmt != 17 && fmt != 20 && fmt != 21) {
		cpuException(n, EXC_RI, 0);
		return;
	}
	int func = op & 0x3F;
	if (func >= 0x30) {
		double a = source(n, fmt, FS), b = source(n, fmt, FT);
		bool unordered = isnan(a) || isnan(b);
		bool cond = ((func & 4) && !unordered && a < b) || ((func & 2) && !unordered && a == b) ||
		            ((func & 1) && unordered);
		c->fcr31 = (c->fcr31 & ~(1u << 23)) | (uint32_t) cond << 23;
		return;
	}
	if (fmt == 16 && func <= 7) {
		float a = getS(n, FS), b = getS(n, FT);
		switch (func) {
		case 0:
			setS(n, FD, a + b);
			return;
		case 1:
			setS(n, FD, a - b);
			return;
		case 2:
			setS(n, FD, a * b);
			return;
		case 3:
			setS(n, FD, a / b);
			return;
		case 4:
			setS(n, FD, sqrtf(a));
			return;
		case 5:
			setS(n, FD, fabsf(a));
			return;
		case 6:
			set32(n, FD, get32(n, FS));
			return;
		case 7:
			setS(n, FD, -a);
			return;
		}
	}
	double a = source(n, fmt, FS);
	switch (func) {
	case 0:
		result(n, fmt, FD, a + source(n, fmt, FT));
		return;
	case 1:
		result(n, fmt, FD, a - source(n, fmt, FT));
		return;
	case 2:
		result(n, fmt, FD, a * source(n, fmt, FT));
		return;
	case 3:
		result(n, fmt, FD, a / source(n, fmt, FT));
		return;
	case 4:
		result(n, fmt, FD, sqrt(a));
		return;
	case 5:
		result(n, fmt, FD, fabs(a));
		return;
	case 6:
		set64(n, FD, get64(n, FS));
		return;
	case 7:
		result(n, fmt, FD, -a);
		return;
	case 8:
	case 9:
	case 10:
	case 11:
		set64(n, FD, (uint64_t) (int64_t) roundWith(a, func - 8));
		return;
	case 12:
	case 13:
	case 14:
	case 15:
		set32(n, FD, (uint32_t) (int32_t) roundWith(a, func - 12));
		return;
	case 32:
		setS(n, FD, (float) a);
		return;
	case 33:
		setD(n, FD, a);
		return;
	case 36:
		set32(n, FD, (uint32_t) (int32_t) roundWith(a, c->fcr31 & 3));
		return;
	case 37:
		set64(n, FD, (uint64_t) (int64_t) roundWith(a, c->fcr31 & 3));
		return;
	}
	cpuException(n, EXC_RI, 0);
}
