#include "n64.h"

#include <string.h>
#if defined(__SSSE3__)
#include <tmmintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
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
#elif defined(__ARM_NEON)
	int32x4_t lo = vmull_s16(vget_low_s16((int16x8_t) a), vget_low_s16((int16x8_t) b));
	int32x4_t hi = vmull_high_s16((int16x8_t) a, (int16x8_t) b);
	return (v8) vuzp2q_s16(vreinterpretq_s16_s32(lo), vreinterpretq_s16_s32(hi));
#else
	return __builtin_convertvector((__builtin_convertvector(a, w8) * __builtin_convertvector(b, w8)) >> 16, v8);
#endif
}

static v8 mulhiUU(v8 a, v8 b) {
#if defined(__SSE2__)
	return (v8) _mm_mulhi_epu16((__m128i) a, (__m128i) b);
#elif defined(__ARM_NEON)
	uint32x4_t lo = vmull_u16(vget_low_u16((uint16x8_t) a), vget_low_u16((uint16x8_t) b));
	uint32x4_t hi = vmull_high_u16((uint16x8_t) a, (uint16x8_t) b);
	return (v8) vuzp2q_u16(vreinterpretq_u16_u32(lo), vreinterpretq_u16_u32(hi));
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
#elif defined(__ARM_NEON)
	static const uint16_t lane[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
	return vaddvq_u16(vandq_u16((uint16x8_t) mask, vld1q_u16(lane)));
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

// The element a vector instruction takes from vt (all lanes, pairs, quads or one lane), as byte
// indices for PSHUFB and TBL.
static const uint8_t elementMasks[16][16] __attribute__((aligned(16))) = {
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

static v8 pick(const int16_t* t, const uint8_t* mask) {
#if defined(__SSSE3__)
	return (v8) _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*) t), _mm_load_si128((const __m128i*) mask));
#elif defined(__ARM_NEON)
	return (v8) vqtbl1q_u8(vld1q_u8((const uint8_t*) t), vld1q_u8(mask));
#else
	typedef uint8_t b16 __attribute__((vector_size(16)));
	b16 bytes, indices;
	memcpy(&bytes, t, sizeof(bytes));
	memcpy(&indices, mask, sizeof(indices));
	return (v8) __builtin_shuffle(bytes, indices);
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
#elif defined(__ARM_NEON)
	int16x8_t a = (int16x8_t) s, b = (int16x8_t) t, k = (int16x8_t) c;
	int32x4_t lo = sub ? vsubw_s16(vsubl_s16(vget_low_s16(a), vget_low_s16(b)), vget_low_s16(k))
	                   : vaddw_s16(vaddl_s16(vget_low_s16(a), vget_low_s16(b)), vget_low_s16(k));
	int32x4_t hi = sub ? vsubw_high_s16(vsubl_high_s16(a, b), k) : vaddw_high_s16(vaddl_high_s16(a, b), k);
	return (v8) vqmovn_high_s32(vqmovn_s32(lo), hi);
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

static struct Acc loadAcc(const struct RSP* r) {
	return (struct Acc) { load(r->accH), load(r->accM), load(r->accL) };
}

static void storeAcc(struct RSP* r, struct Acc a) {
	store(r->accH, a.h);
	store(r->accM, a.m);
	store(r->accL, a.l);
}

// One function an instruction, given vd, vs and vt and the element's byte indices. Each reads its
// operands before writing vd, which may be one of them.
#define VECTOR_OP(name) static void name(struct RSP* r, int16_t* dp, const int16_t* sp, const int16_t* tp, const uint8_t* ep)
#define OPERANDS v8 s = load(sp), t = pick(tp, ep), zero = splat(0); (void) zero

static inline __attribute__((always_inline)) void multiplyFraction(struct RSP* r, int16_t* dp, v8 s, v8 t, bool add, bool clampU) {
	v8 lo = s * t, hi = mulhiSS(s, t);
	v8 l2 = lo << 1, h2 = (hi << 1) | shr(lo, 15);
	v8 overflow = (s == splat(-32768)) & (t == splat(-32768));
	struct Acc a;
	if (add) {
		a = loadAcc(r);
		accAdd(&a, (h2 >> 15) & ~overflow, h2, l2);
	} else {
		a.l = l2 ^ splat(-32768);
		a.m = h2 + shr(l2, 15);
		a.h = (a.m >> 15) & ~overflow;
	}
	storeAcc(r, a);
	store(dp, clampU ? clampUnsigned(a.h, a.m) : clampSigned(a.h, a.m));
}

VECTOR_OP(vmulf) {
	OPERANDS;
	multiplyFraction(r, dp, s, t, false, false);
}

VECTOR_OP(vmulu) {
	OPERANDS;
	multiplyFraction(r, dp, s, t, false, true);
}

VECTOR_OP(vmacf) {
	OPERANDS;
	multiplyFraction(r, dp, s, t, true, false);
}

VECTOR_OP(vmacu) {
	OPERANDS;
	multiplyFraction(r, dp, s, t, true, true);
}

VECTOR_OP(vmulq) {
	OPERANDS;
	v8 lo = s * t, hi = mulhiSS(s, t);
	v8 lo2 = lo + (splat(31) & (hi >> 15));
	struct Acc a = { hi - ltu(lo2, lo), lo2, zero };
	storeAcc(r, a);
	store(dp, quantize(a.h, a.m));
}

VECTOR_OP(vmudl) {
	OPERANDS;
	v8 l = mulhiUU(s, t);
	storeAcc(r, (struct Acc) { zero, zero, l });
	store(dp, l);
}

VECTOR_OP(vmudm) {
	OPERANDS;
	v8 m = mulhiSU(s, t);
	storeAcc(r, (struct Acc) { m >> 15, m, s * t });
	store(dp, m);
}

VECTOR_OP(vmudn) {
	OPERANDS;
	v8 l = s * t, m = mulhiUS(s, t);
	storeAcc(r, (struct Acc) { m >> 15, m, l });
	store(dp, l);
}

VECTOR_OP(vmudh) {
	OPERANDS;
	struct Acc a = { mulhiSS(s, t), s * t, zero };
	storeAcc(r, a);
	store(dp, clampSigned(a.h, a.m));
}

VECTOR_OP(vmacq) {
	v8 zero = splat(0);
	v8 h = load(r->accH), m = load(r->accM);
	v8 bit5 = (m & splat(0x20)) != zero;
	v8 negative = h < zero;
	v8 add = negative & ~bit5;
	v8 sub = ~negative & ~bit5 & ((h > zero) | ((h == zero) & ~ltu(m, splat(32))));
	v8 up = m + splat(32), down = m - splat(32);
	h = h - (add & ltu(up, m)) + (sub & ltu(m, splat(32)));
	m = sel(add, up, sel(sub, down, m));
	store(r->accH, h);
	store(r->accM, m);
	store(dp, quantize(h, m));
}

VECTOR_OP(vmadl) {
	OPERANDS;
	struct Acc a = loadAcc(r);
	accAdd(&a, zero, zero, mulhiUU(s, t));
	storeAcc(r, a);
	store(dp, clampLow(a.h, a.m, a.l));
}

VECTOR_OP(vmadm) {
	OPERANDS;
	v8 m = mulhiSU(s, t);
	struct Acc a = loadAcc(r);
	accAdd(&a, m >> 15, m, s * t);
	storeAcc(r, a);
	store(dp, clampSigned(a.h, a.m));
}

VECTOR_OP(vmadn) {
	OPERANDS;
	v8 m = mulhiUS(s, t);
	struct Acc a = loadAcc(r);
	accAdd(&a, m >> 15, m, s * t);
	storeAcc(r, a);
	store(dp, clampLow(a.h, a.m, a.l));
}

VECTOR_OP(vmadh) {
	OPERANDS;
	struct Acc a = loadAcc(r);
	accAdd(&a, mulhiSS(s, t), s * t, zero);
	storeAcc(r, a);
	store(dp, clampSigned(a.h, a.m));
}

VECTOR_OP(vadd) {
	OPERANDS;
	v8 carry = maskOf(r->vco) & splat(1);
	store(r->accL, s + t + carry);
	store(dp, addSaturated(s, t, carry, false));
	r->vco = 0;
}

VECTOR_OP(vsub) {
	OPERANDS;
	v8 carry = maskOf(r->vco) & splat(1);
	store(r->accL, s - t - carry);
	store(dp, addSaturated(s, t, carry, true));
	r->vco = 0;
}

VECTOR_OP(vabs) {
	OPERANDS;
	v8 negative = s < zero;
	v8 positive = sel(s == zero, zero, t);
	v8 minus = sel(t == splat(-32768), splat(32767), -t);
	store(r->accL, sel(negative, -t, positive));
	store(dp, sel(negative, minus, positive));
}

VECTOR_OP(vaddc) {
	OPERANDS;
	v8 d = s + t;
	r->vco = (uint16_t) bits(ltu(d, s));
	store(r->accL, d);
	store(dp, d);
}

VECTOR_OP(vsubc) {
	OPERANDS;
	v8 d = s - t;
	r->vco = (uint16_t) (bits(ltu(s, t)) | bits(d != zero) << 8);
	store(r->accL, d);
	store(dp, d);
}

VECTOR_OP(vsarH) {
	store(dp, load(r->accH));
}

VECTOR_OP(vsarM) {
	store(dp, load(r->accM));
}

VECTOR_OP(vsarL) {
	store(dp, load(r->accL));
}

VECTOR_OP(vzero) {
	store(dp, splat(0));
}

VECTOR_OP(vnop) {
}

static inline __attribute__((always_inline)) void compare(struct RSP* r, int16_t* dp, v8 s, v8 t, int func) {
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
	v8 d = func == 0x21 ? t : sel(cond, s, t);
	r->vcc = (uint16_t) bits(cond);
	r->vco = 0;
	store(r->accL, d);
	store(dp, d);
}

VECTOR_OP(vlt) {
	OPERANDS;
	compare(r, dp, s, t, 0x20);
}

VECTOR_OP(veq) {
	OPERANDS;
	compare(r, dp, s, t, 0x21);
}

VECTOR_OP(vne) {
	OPERANDS;
	compare(r, dp, s, t, 0x22);
}

VECTOR_OP(vge) {
	OPERANDS;
	compare(r, dp, s, t, 0x23);
}

VECTOR_OP(vcl) {
	OPERANDS;
	v8 le = maskOf(r->vcc), ge = maskOf(r->vcc >> 8);
	v8 c = maskOf(r->vco), ne = maskOf(r->vco >> 8), vce = maskOf(r->vce);
	v8 sum = s + t;
	v8 carry = ltu(sum, s), zeroSum = sum == zero;
	v8 leNew = sel(vce, zeroSum | ~carry, zeroSum & ~carry);
	le = sel(c & ~ne, leNew, le);
	ge = sel(~c & ~ne, ~ltu(s, t), ge);
	v8 d = sel(c, sel(le, -t, s), sel(ge, t, s));
	r->vcc = (uint16_t) (bits(le) | bits(ge) << 8);
	r->vco = 0;
	r->vce = 0;
	store(r->accL, d);
	store(dp, d);
}

static inline __attribute__((always_inline)) void clip(struct RSP* r, int16_t* dp, v8 s, v8 t, bool vch) {
	v8 zero = splat(0);
	v8 sign = (s ^ t) < zero;
	v8 sum = s + t, diff = s - t;
	v8 tNeg = t < zero;
	v8 le = sel(sign, vch ? sum <= zero : sum < zero, tNeg);
	v8 ge = sel(sign, tNeg, diff >= zero);
	v8 d = sel(sign, sel(le, vch ? -t : ~t, s), sel(ge, t, s));
	r->vcc = (uint16_t) (bits(le) | bits(ge) << 8);
	if (vch) {
		v8 ne = sel(sign, (sum != zero) & (sum != splat(-1)), diff != zero);
		r->vco = (uint16_t) (bits(sign) | bits(ne) << 8);
		r->vce = (uint8_t) bits(sign & (sum == splat(-1)));
	} else {
		r->vco = 0;
		r->vce = 0;
	}
	store(r->accL, d);
	store(dp, d);
}

VECTOR_OP(vch) {
	OPERANDS;
	clip(r, dp, s, t, true);
}

VECTOR_OP(vcr) {
	OPERANDS;
	clip(r, dp, s, t, false);
}

VECTOR_OP(vmrg) {
	OPERANDS;
	v8 d = sel(maskOf(r->vcc), s, t);
	r->vco = 0;
	store(r->accL, d);
	store(dp, d);
}

#define LOGIC_OP(name, expr) \
	VECTOR_OP(name) { \
		OPERANDS; \
		v8 d = expr; \
		store(r->accL, d); \
		store(dp, d); \
	}

LOGIC_OP(vand, s & t)
LOGIC_OP(vnand, ~(s & t))
LOGIC_OP(vor, s | t)
LOGIC_OP(vnor, ~(s | t))
LOGIC_OP(vxor, s ^ t)
LOGIC_OP(vnxor, ~(s ^ t))

static RspVectorFn* const vectorOps[64] = {
	[0x00] = vmulf, [0x01] = vmulu, [0x03] = vmulq, [0x04] = vmudl, [0x05] = vmudm, [0x06] = vmudn, [0x07] = vmudh,
	[0x08] = vmacf, [0x09] = vmacu, [0x0B] = vmacq, [0x0C] = vmadl, [0x0D] = vmadm, [0x0E] = vmadn, [0x0F] = vmadh,
	[0x10] = vadd, [0x11] = vsub, [0x12] = vzero, [0x13] = vabs, [0x14] = vaddc, [0x15] = vsubc,
	[0x16] = vzero, [0x17] = vzero, [0x18] = vzero, [0x19] = vzero, [0x1A] = vzero, [0x1B] = vzero, [0x1C] = vzero,
	[0x1E] = vzero, [0x1F] = vzero, [0x20] = vlt, [0x21] = veq, [0x22] = vne, [0x23] = vge, [0x24] = vcl, [0x25] = vch,
	[0x26] = vcr, [0x27] = vmrg, [0x28] = vand, [0x29] = vnand, [0x2A] = vor, [0x2B] = vnor, [0x2C] = vxor,
	[0x2D] = vnxor, [0x2E] = vzero, [0x2F] = vzero, [0x37] = vnop, [0x38] = vzero, [0x39] = vzero, [0x3A] = vzero,
	[0x3B] = vzero, [0x3C] = vzero, [0x3D] = vzero, [0x3E] = vzero, [0x3F] = vnop,
};

RspVectorFn* rspVectorFn(uint32_t op) {
	if ((op & 0x3F) == 0x1D) {
		int e = (op >> 21) & 15;
		return e == 8 ? vsarH : e == 9 ? vsarM : e == 10 ? vsarL : vzero;
	}
	return vectorOps[op & 0x3F];
}

const uint8_t* rspElementMask(uint32_t op) {
	return elementMasks[(op >> 21) & 15];
}

// VRNDP and VRNDN, which take vs's number, and the divides and VMOV, which take single lanes.
static void vectorOpByNumber(struct RSP* r, uint32_t op) {
	int e = (op >> 21) & 15, vt = (op >> 16) & 31, vs = (op >> 11) & 31, vd = (op >> 6) & 31;
	int func = op & 0x3F;
	v8 t = pick(r->vr[vt], elementMasks[e]);
	struct Acc a = loadAcc(r);
	v8 zero = splat(0), d;
	switch (func) {
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
	default: {
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
	}
	}
	storeAcc(r, a);
	store(r->vr[vd], d);
}

void rspVectorOp(struct RSP* r, uint32_t op) {
	RspVectorFn* f = rspVectorFn(op);
	if (f) {
		f(r, r->vr[(op >> 6) & 31], r->vr[(op >> 11) & 31], r->vr[(op >> 16) & 31], elementMasks[(op >> 21) & 15]);
	} else {
		vectorOpByNumber(r, op);
	}
}
