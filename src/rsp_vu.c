#include "n64.h"

#include <string.h>
#if defined(__SSSE3__)
#include <tmmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

// The RSP vector unit, eight lanes at a time. The 48-bit accumulator is kept as three 16-bit
// slices (high, middle, low), so every operation works on 16-bit lanes.

typedef int16_t v8 __attribute__((vector_size(16)));
typedef uint16_t u8v __attribute__((vector_size(16)));
typedef int32_t w8 __attribute__((vector_size(32)));

static uint16_t reciprocals[512];
static uint16_t inverseRoots[512];

void rspInitTables(void) {
	for (uint32_t i = 0; i < 512; ++i) {
		uint64_t a = i + 512;
		uint64_t b = (1ull << 34) / a;
		// The first entry would be 0x20000; the table holds 0xFFFF there.
		reciprocals[i] = i == 0 ? 0xFFFF : (uint16_t) ((b + 1) >> 8);
		a = (i + 512) >> (i & 1);
		b = 1 << 17;
		while (a * (b + 1) * (b + 1) < (1ull << 44)) {
			++b;
		}
		inverseRoots[i] = (uint16_t) (b >> 1);
	}
}

static uint32_t rcp(int32_t input) {
	int32_t mask = input >> 31;
	int32_t data = input ^ mask;
	if (input > -32768) {
		data -= mask;
	}
	if (data == 0) {
		return 0x7FFFFFFF;
	}
	if (input == -32768) {
		return 0xFFFF0000;
	}
	int shift = __builtin_clz((uint32_t) data);
	uint32_t index = (uint32_t) (((uint64_t) (uint32_t) data << shift & 0x7FC00000) >> 22);
	uint32_t result = (0x10000u | reciprocals[index]) << 14;
	return (result >> (31 - shift)) ^ (uint32_t) mask;
}

static uint32_t rsq(int32_t input) {
	int32_t mask = input >> 31;
	int32_t data = input ^ mask;
	if (input > -32768) {
		data -= mask;
	}
	if (data == 0) {
		return 0x7FFFFFFF;
	}
	if (input == -32768) {
		return 0xFFFF0000;
	}
	int shift = __builtin_clz((uint32_t) data);
	uint32_t index = (uint32_t) (((uint64_t) (uint32_t) data << shift & 0x7FC00000) >> 22);
	index = ((index | 0x200) & 0x3FE) | (shift & 1);
	uint32_t result = (0x10000u | inverseRoots[index & 0x1FF]) << 14;
	return (result >> ((31 - shift) >> 1)) ^ (uint32_t) mask;
}

static v8 splat(int16_t x) {
	return (v8) { x, x, x, x, x, x, x, x };
}

static v8 load(const int16_t* p) {
	v8 v;
	memcpy(&v, p, sizeof(v));
	return v;
}

static void store(int16_t* p, v8 v) {
	memcpy(p, &v, sizeof(v));
}

static v8 sel(v8 mask, v8 a, v8 b) {
	return (mask & a) | (~mask & b);
}

static v8 shr(v8 v, int n) {
	return (v8) ((u8v) v >> n);
}

static v8 ltu(v8 a, v8 b) {
	return (v8) ((u8v) a < (u8v) b);
}

// The high halves of 16x16-bit products: signed by signed, unsigned by unsigned, signed by
// unsigned and unsigned by signed.
static v8 mulhiSS(v8 a, v8 b) {
#if defined(__SSE2__)
	return (v8) _mm_mulhi_epi16((__m128i) a, (__m128i) b);
#else
	return __builtin_convertvector((__builtin_convertvector(a, w8) * __builtin_convertvector(b, w8)) >> 16, v8);
#endif
}

static v8 mulhiUU(v8 a, v8 b) {
#if defined(__SSE2__)
	return (v8) _mm_mulhi_epu16((__m128i) a, (__m128i) b);
#else
	typedef uint32_t uw8 __attribute__((vector_size(32)));
	return (v8) __builtin_convertvector((__builtin_convertvector((u8v) a, uw8) * __builtin_convertvector((u8v) b, uw8)) >> 16, u8v);
#endif
}

static v8 mulhiSU(v8 a, v8 b) {
	return mulhiSS(a, b) + (a & (b >> 15));
}

static v8 mulhiUS(v8 a, v8 b) {
	return mulhiSS(a, b) + (b & (a >> 15));
}

// Lane i of a mask as bit i, and bits back to a mask.
static unsigned bits(v8 mask) {
#if defined(__SSE2__)
	return (unsigned) _mm_movemask_epi8(_mm_packs_epi16((__m128i) mask, _mm_setzero_si128())) & 0xFF;
#else
	unsigned b = 0;
	for (int i = 0; i < 8; ++i) {
		b |= (unsigned) (mask[i] & 1) << i;
	}
	return b;
#endif
}

static v8 maskOf(unsigned b) {
	v8 lane = { 1, 2, 4, 8, 16, 32, 64, 128 };
	return (splat((int16_t) b) & lane) != splat(0);
}

// The element a vector instruction takes from vt: all lanes, pairs, quads or one lane.
static v8 element(v8 t, int e) {
#if defined(__SSSE3__)
	// Byte indices for PSHUFB.
	static const uint8_t masks[16][16] __attribute__((aligned(16))) = {
		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
		{ 0, 1, 0, 1, 4, 5, 4, 5, 8, 9, 8, 9, 12, 13, 12, 13 },
		{ 2, 3, 2, 3, 6, 7, 6, 7, 10, 11, 10, 11, 14, 15, 14, 15 },
		{ 0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 8, 9, 8, 9, 8, 9 },
		{ 2, 3, 2, 3, 2, 3, 2, 3, 10, 11, 10, 11, 10, 11, 10, 11 },
		{ 4, 5, 4, 5, 4, 5, 4, 5, 12, 13, 12, 13, 12, 13, 12, 13 },
		{ 6, 7, 6, 7, 6, 7, 6, 7, 14, 15, 14, 15, 14, 15, 14, 15 },
		{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 },
		{ 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3 },
		{ 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5 },
		{ 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7 },
		{ 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9 },
		{ 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11 },
		{ 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13 },
		{ 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15 },
	};
	return (v8) _mm_shuffle_epi8((__m128i) t, *(const __m128i*) masks[e]);
#else
	switch (e) {
	case 2: return __builtin_shuffle(t, (v8) { 0, 0, 2, 2, 4, 4, 6, 6 });
	case 3: return __builtin_shuffle(t, (v8) { 1, 1, 3, 3, 5, 5, 7, 7 });
	case 4: return __builtin_shuffle(t, (v8) { 0, 0, 0, 0, 4, 4, 4, 4 });
	case 5: return __builtin_shuffle(t, (v8) { 1, 1, 1, 1, 5, 5, 5, 5 });
	case 6: return __builtin_shuffle(t, (v8) { 2, 2, 2, 2, 6, 6, 6, 6 });
	case 7: return __builtin_shuffle(t, (v8) { 3, 3, 3, 3, 7, 7, 7, 7 });
	case 8: case 9: case 10: case 11: case 12: case 13: case 14: case 15:
		return splat(t[e & 7]);
	}
	return t;
#endif
}

// s + t + c or s - t - c (c is 0 or 1 a lane), saturated to 16 signed bits.
static v8 addSaturated(v8 s, v8 t, v8 c, bool sub) {
#if defined(__SSE2__)
	__m128i parts[2];
	for (int half = 0; half < 2; ++half) {
		__m128i a = half ? _mm_unpackhi_epi16((__m128i) s, (__m128i) s) : _mm_unpacklo_epi16((__m128i) s, (__m128i) s);
		__m128i b = half ? _mm_unpackhi_epi16((__m128i) t, (__m128i) t) : _mm_unpacklo_epi16((__m128i) t, (__m128i) t);
		__m128i k = half ? _mm_unpackhi_epi16((__m128i) c, (__m128i) c) : _mm_unpacklo_epi16((__m128i) c, (__m128i) c);
		a = _mm_srai_epi32(a, 16), b = _mm_srai_epi32(b, 16), k = _mm_srai_epi32(k, 16);
		parts[half] = sub ? _mm_sub_epi32(_mm_sub_epi32(a, b), k) : _mm_add_epi32(_mm_add_epi32(a, b), k);
	}
	return (v8) _mm_packs_epi32(parts[0], parts[1]);
#else
	w8 v = __builtin_convertvector(s, w8);
	w8 wt = __builtin_convertvector(t, w8), wc = __builtin_convertvector(c, w8);
	v = sub ? v - wt - wc : v + wt + wc;
	w8 lo = { -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768 };
	w8 hi = { 32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767 };
	w8 under = v < lo, over = v > hi;
	return __builtin_convertvector((under & lo) | (over & hi) | (~(under | over) & v), v8);
#endif
}

struct Acc {
	v8 h, m, l;
};

// acc += (h, m, l), wrapping at 48 bits.
static void accAdd(struct Acc* a, v8 h, v8 m, v8 l) {
	v8 low = a->l + l;
	v8 c0 = ltu(low, a->l);
	v8 mid = a->m + m;
	v8 c1 = ltu(mid, a->m);
	v8 mid2 = mid - c0;
	v8 c2 = c0 & (mid2 == splat(0));
	a->h = a->h + h - c1 - c2;
	a->m = mid2;
	a->l = low;
}

// Bits 16-31 of the accumulator saturated to 16 signed bits; the low slice saturated when the
// accumulator doesn't fit in 32 signed bits; bits 16-31 saturated to 16 unsigned bits.
static v8 clampSigned(v8 h, v8 m) {
	return sel(h == (m >> 15), m, sel(h < splat(0), splat(-32768), splat(32767)));
}

static v8 clampLow(v8 h, v8 m, v8 l) {
	return sel(h == (m >> 15), l, sel(h < splat(0), splat(0), splat(-1)));
}

static v8 clampUnsigned(v8 h, v8 m) {
	return sel(h < splat(0), splat(0), sel((h > splat(0)) | (m < splat(0)), splat(-1), m));
}

// a >> 1 as 32 bits (h:m), saturated and with the low 4 bits cleared, for VMULQ and VMACQ.
static v8 quantize(v8 h, v8 m) {
	return clampSigned(h >> 1, shr(m, 1) | (h << 15)) & splat(~0xF);
}

void rspVectorOp(struct RSP* r, uint32_t op) {
	int e = (op >> 21) & 15, vt = (op >> 16) & 31, vs = (op >> 11) & 31, vd = (op >> 6) & 31;
	int func = op & 0x3F;
	v8 s = load(r->vr[vs]);
	v8 t = element(load(r->vr[vt]), e);
	struct Acc a = { load(r->accH), load(r->accM), load(r->accL) };
	v8 zero = splat(0), d;
	switch (func) {
	case 0x00:
	case 0x01:
	case 0x08:
	case 0x09: {
		v8 lo = s * t, hi = mulhiSS(s, t);
		v8 l2 = lo << 1, h2 = (hi << 1) | shr(lo, 15);
		v8 overflow = (s == splat(-32768)) & (t == splat(-32768));
		if (func < 0x08) {
			a.l = l2 ^ splat(-32768);
			a.m = h2 + shr(l2, 15);
			a.h = (a.m >> 15) & ~overflow;
		} else {
			accAdd(&a, (h2 >> 15) & ~overflow, h2, l2);
		}
		d = func & 1 ? clampUnsigned(a.h, a.m) : clampSigned(a.h, a.m);
		break;
	}
	case 0x02:
	case 0x0A: {
		v8 cond = func == 0x02 ? a.h >= zero : a.h < zero;
		v8 sign = t >> 15;
		if (vs & 1) {
			accAdd(&a, sign & cond, t & cond, zero);
		} else {
			accAdd(&a, sign & cond, sign & cond, t & cond);
		}
		d = clampSigned(a.h, a.m);
		break;
	}
	case 0x03: {
		v8 lo = s * t, hi = mulhiSS(s, t);
		v8 add = splat(31) & (hi >> 15);
		v8 lo2 = lo + add;
		a.h = hi - ltu(lo2, lo);
		a.m = lo2;
		a.l = zero;
		d = quantize(a.h, a.m);
		break;
	}
	case 0x04:
		a.l = mulhiUU(s, t);
		a.m = a.h = zero;
		d = a.l;
		break;
	case 0x05:
		a.l = s * t;
		a.m = mulhiSU(s, t);
		a.h = a.m >> 15;
		d = a.m;
		break;
	case 0x06:
		a.l = s * t;
		a.m = mulhiUS(s, t);
		a.h = a.m >> 15;
		d = a.l;
		break;
	case 0x07:
		a.h = mulhiSS(s, t);
		a.m = s * t;
		a.l = zero;
		d = clampSigned(a.h, a.m);
		break;
	case 0x0B: {
		v8 bit5 = (a.m & splat(0x20)) != zero;
		v8 negative = a.h < zero;
		v8 add = negative & ~bit5;
		v8 sub = ~negative & ~bit5 & ((a.h > zero) | ((a.h == zero) & ~ltu(a.m, splat(32))));
		v8 up = a.m + splat(32), down = a.m - splat(32);
		a.h = a.h - (add & ltu(up, a.m)) + (sub & ltu(a.m, splat(32)));
		a.m = sel(add, up, sel(sub, down, a.m));
		d = quantize(a.h, a.m);
		break;
	}
	case 0x0C:
		accAdd(&a, zero, zero, mulhiUU(s, t));
		d = clampLow(a.h, a.m, a.l);
		break;
	case 0x0D: {
		v8 m = mulhiSU(s, t);
		accAdd(&a, m >> 15, m, s * t);
		d = clampSigned(a.h, a.m);
		break;
	}
	case 0x0E: {
		v8 m = mulhiUS(s, t);
		accAdd(&a, m >> 15, m, s * t);
		d = clampLow(a.h, a.m, a.l);
		break;
	}
	case 0x0F:
		accAdd(&a, mulhiSS(s, t), s * t, zero);
		d = clampSigned(a.h, a.m);
		break;
	case 0x10:
	case 0x11: {
		v8 carry = maskOf(r->vco) & splat(1);
		a.l = func == 0x10 ? s + t + carry : s - t - carry;
		d = addSaturated(s, t, carry, func == 0x11);
		r->vco = 0;
		break;
	}
	case 0x13: {
		v8 negative = s < zero;
		v8 positive = sel(s == zero, zero, t);
		v8 minus = sel(t == splat(-32768), splat(32767), -t);
		d = sel(negative, minus, positive);
		a.l = sel(negative, -t, positive);
		break;
	}
	case 0x14:
		d = s + t;
		r->vco = (uint16_t) bits(ltu(d, s));
		a.l = d;
		break;
	case 0x15:
		d = s - t;
		r->vco = (uint16_t) (bits(ltu(s, t)) | bits(d != zero) << 8);
		a.l = d;
		break;
	case 0x1D:
		d = e == 8 ? a.h : e == 9 ? a.m : e == 10 ? a.l : zero;
		break;
	case 0x20:
	case 0x21:
	case 0x22:
	case 0x23: {
		v8 ne = maskOf(r->vco >> 8), c = maskOf(r->vco), eq = s == t, cond;
		switch (func) {
		case 0x20:
			cond = (s < t) | (eq & ne & c);
			break;
		case 0x21:
			cond = eq & ~ne;
			break;
		case 0x22:
			cond = ~eq | ne;
			break;
		default:
			cond = (s > t) | (eq & ~(ne & c));
		}
		d = func == 0x21 ? t : sel(cond, s, t);
		r->vcc = (uint16_t) bits(cond);
		r->vco = 0;
		a.l = d;
		break;
	}
	case 0x24: {
		v8 le = maskOf(r->vcc), ge = maskOf(r->vcc >> 8);
		v8 c = maskOf(r->vco), ne = maskOf(r->vco >> 8), vce = maskOf(r->vce);
		v8 sum = s + t;
		v8 carry = ltu(sum, s), zeroSum = sum == zero;
		v8 leNew = sel(vce, zeroSum | ~carry, zeroSum & ~carry);
		le = sel(c & ~ne, leNew, le);
		ge = sel(~c & ~ne, ~ltu(s, t), ge);
		d = sel(c, sel(le, -t, s), sel(ge, t, s));
		r->vcc = (uint16_t) (bits(le) | bits(ge) << 8);
		r->vco = 0;
		r->vce = 0;
		a.l = d;
		break;
	}
	case 0x25:
	case 0x26: {
		v8 sign = (s ^ t) < zero;
		v8 sum = s + t, diff = s - t;
		v8 tNeg = t < zero;
		v8 le = sel(sign, func == 0x25 ? sum <= zero : sum < zero, tNeg);
		v8 ge = sel(sign, tNeg, diff >= zero);
		d = sel(sign, sel(le, func == 0x25 ? -t : ~t, s), sel(ge, t, s));
		r->vcc = (uint16_t) (bits(le) | bits(ge) << 8);
		if (func == 0x25) {
			v8 ne = sel(sign, (sum != zero) & (sum != splat(-1)), diff != zero);
			r->vco = (uint16_t) (bits(sign) | bits(ne) << 8);
			r->vce = (uint8_t) bits(sign & (sum == splat(-1)));
		} else {
			r->vco = 0;
			r->vce = 0;
		}
		a.l = d;
		break;
	}
	case 0x27:
		d = sel(maskOf(r->vcc), s, t);
		r->vco = 0;
		a.l = d;
		break;
	case 0x28:
		d = s & t;
		a.l = d;
		break;
	case 0x29:
		d = ~(s & t);
		a.l = d;
		break;
	case 0x2A:
		d = s | t;
		a.l = d;
		break;
	case 0x2B:
		d = ~(s | t);
		a.l = d;
		break;
	case 0x2C:
		d = s ^ t;
		a.l = d;
		break;
	case 0x2D:
		d = ~(s ^ t);
		a.l = d;
		break;
	case 0x30:
	case 0x31:
	case 0x32:
	case 0x33:
	case 0x34:
	case 0x35:
	case 0x36: {
		int de = vs & 7;
		int16_t in = r->vr[vt][e & 7];
		d = load(r->vr[vd]);
		a.l = t;
		if (func == 0x33) {
			d[de] = t[de];
			break;
		}
		if (func == 0x32 || func == 0x36) {
			r->divIn = in;
			r->divDp = true;
			d[de] = (int16_t) (r->divOut >> 16);
			break;
		}
		int32_t input = (func == 0x31 || func == 0x35) && r->divDp ? (int32_t) ((uint32_t) (uint16_t) r->divIn << 16 | (uint16_t) in)
		                                                         : in;
		uint32_t result = func <= 0x32 ? rcp(input) : rsq(input);
		r->divOut = result;
		r->divDp = false;
		d[de] = (int16_t) result;
		break;
	}
	case 0x37:
	case 0x3F:
		return;
	default:
		d = zero;
	}
	store(r->accH, a.h);
	store(r->accM, a.m);
	store(r->accL, a.l);
	store(r->vr[vd], d);
}
