#include "n64.h"

#include <stdlib.h>
#include <string.h>

struct Tile {
	int format, size, line, tmem, palette;
	int cmS, cmT, maskS, maskT, shiftS, shiftT;
	int sl, tl, sh, th;
};

struct Color {
	int r, g, b, a;
};

struct Pixel {
	struct Color shade, tex0, tex1, combined;
	int z;
	int lodFrac;
};

// Colors as four lanes: red, green, blue and alpha.
typedef int32_t i4 __attribute__((vector_size(16)));

static i4 splat4(int v) {
	return (i4) { v, v, v, v };
}

static i4 lanes(struct Color c) {
	i4 v;
	memcpy(&v, &c, sizeof(v));
	return v;
}

static struct Color color(i4 v) {
	struct Color c;
	memcpy(&c, &v, sizeof(c));
	return c;
}

// A combiner input for one cycle: three channel values, or with step 0 one value for all three.
struct Input {
	const int* v;
	int step;
};

// One combiner cycle's inputs, (a - b) * m + d for color and alpha, and whether it uses noise.
struct Stage {
	struct Input a, b, m, d;
	const int* alpha[4];
	bool noise;
	// Every input is 0-255 (or 1.0 outside the multiplier), so the 9-bit sign rules change nothing.
	bool raw;
};

struct RDP {
	uint8_t tmem[0x1000];
	struct Tile tile[8];
	uint32_t colorAddr, colorWidth;
	int colorSize, colorFormat;
	uint32_t zAddr;
	uint32_t texAddr, texWidth;
	int texSize, texFormat;
	int scissorX0, scissorY0, scissorX1, scissorY1;
	uint64_t otherModes, combine;
	uint32_t fillColor;
	struct Color fog, blend, prim, env;
	int primLod, primZ, primDz;
	int minLevel, maxLevel;
	// The combiner's last output: what "combined" reads in one-cycle mode and in the first cycle.
	struct Color lastCombined;
	int k4, k5;
	// RDP clocks spent on the commands so far.
	uint64_t clocks;
	// The pixel being drawn, and where the combiner's inputs come from for the current combine mode.
	struct Pixel pix;
	struct Stage stage[2];
	// otherModes taken apart (setupModes): the cycle type, depth and alpha tests, and the blender's
	// P, A, M and B inputs for each cycle.
	int cycleType, zMode;
	bool zCompare, zUpdate, zPrim, alphaCompare, cvgAlpha, forceBlend;
	uint8_t blendSel[2][4];
};

static void setupCombiner(struct RDP* r);
static void setupModes(struct RDP* r);

static struct RDP* state(struct N64* n) {
	if (!n->rdp) {
		n->rdp = calloc(1, sizeof(struct RDP));
		if (n->rdp) {
			setupCombiner(n->rdp);
			setupModes(n->rdp);
		}
	}
	return n->rdp;
}

static uint64_t word(struct N64* n, uint32_t addr) {
	const uint8_t* p = n->dp.status & 1 ? &n->sp.mem[addr & 0xFF8] : &n->rdram[addr & (RDRAM_SIZE - 8)];
	return (uint64_t) be32(p) << 32 | be32(p + 4);
}

static inline int clamp8(int v) {
	return v < 0 ? 0 : v > 255 ? 255 : v;
}

static struct Color unpack(uint32_t v) {
	return (struct Color) { v >> 24, (v >> 16) & 255, (v >> 8) & 255, v & 255 };
}

// ---- Z buffer: 18-bit depth stored as 14 bits (3-bit exponent, 11-bit mantissa) and 2 dz bits.

static const int zShift[8] = { 6, 5, 4, 3, 2, 1, 0, 0 };
static const int zBase[8] = { 0, 0x20000, 0x30000, 0x38000, 0x3C000, 0x3E000, 0x3F000, 0x3F800 };

static uint16_t zCompress(int z) {
	int e = __builtin_clz(~((uint32_t) z << 14));
	e = e > 7 ? 7 : e;
	int m = ((z - zBase[e]) >> zShift[e]) & 0x7FF;
	return (uint16_t) ((e << 11 | m) << 2);
}

static int zDecompress(uint16_t v) {
	int e = (v >> 13) & 7, m = (v >> 2) & 0x7FF;
	return zBase[e] + (m << zShift[e]);
}

// ---- Framebuffer

static struct Color readPixel(struct N64* n, struct RDP* r, int x, int y) {
	uint32_t i = y * r->colorWidth + x;
	if (r->colorSize == 1) {
		int v = n->rdram[(r->colorAddr + i) & (RDRAM_SIZE - 1)];
		return (struct Color) { v, v, v, 0xE0 };
	}
	if (r->colorSize == 3) {
		uint32_t a = (r->colorAddr + i * 4) & (RDRAM_SIZE - 4);
		return unpack(be32(&n->rdram[a]));
	}
	uint32_t a = (r->colorAddr + i * 2) & (RDRAM_SIZE - 2);
	uint16_t p = (uint16_t) (n->rdram[a] << 8 | n->rdram[a + 1]);
	int cr = (p >> 11) & 31, cg = (p >> 6) & 31, cb = (p >> 1) & 31;
	return (struct Color) { cr << 3 | cr >> 2, cg << 3 | cg >> 2, cb << 3 | cb >> 2, p & 1 ? 255 : 0 };
}

static void writePixel(struct N64* n, struct RDP* r, int x, int y, struct Color c) {
	uint32_t i = y * r->colorWidth + x;
	// An 8-bit color image keeps one byte per pixel: the red channel.
	if (r->colorSize == 1) {
		n->rdram[(r->colorAddr + i) & (RDRAM_SIZE - 1)] = (uint8_t) c.r;
		return;
	}
	if (r->colorSize == 3) {
		uint32_t a = (r->colorAddr + i * 4) & (RDRAM_SIZE - 4);
		put32(&n->rdram[a], (uint32_t) c.r << 24 | (uint32_t) c.g << 16 | (uint32_t) c.b << 8 | (uint32_t) c.a);
		return;
	}
	uint32_t a = (r->colorAddr + i * 2) & (RDRAM_SIZE - 2);
	uint16_t p = (uint16_t) ((c.r >> 3) << 11 | (c.g >> 3) << 6 | (c.b >> 3) << 1 | (c.a >= 128 ? 1 : 0));
	n->rdram[a] = p >> 8;
	n->rdram[a + 1] = (uint8_t) p;
}

static uint16_t* zPointer(struct N64* n, struct RDP* r, int x, int y, uint8_t* tmp) {
	uint32_t a = (r->zAddr + (y * r->colorWidth + x) * 2) & (RDRAM_SIZE - 2);
	(void) tmp;
	return (uint16_t*) &n->rdram[a];
}

static uint16_t zRead(struct N64* n, struct RDP* r, int x, int y) {
	uint8_t* p = (uint8_t*) zPointer(n, r, x, y, NULL);
	return (uint16_t) (p[0] << 8 | p[1]);
}

static void zWrite(struct N64* n, struct RDP* r, int x, int y, uint16_t v) {
	uint8_t* p = (uint8_t*) zPointer(n, r, x, y, NULL);
	p[0] = v >> 8;
	p[1] = (uint8_t) v;
}

// ---- Textures

static uint8_t tmemByte(struct RDP* r, uint32_t addr) {
	return r->tmem[addr & 0xFFF];
}

static inline __attribute__((always_inline)) struct Color tlutColor(struct RDP* r, int index) {
	uint32_t a = 0x800 + index * 8;
	uint16_t v = (uint16_t) (tmemByte(r, a) << 8 | tmemByte(r, a + 1));
	if ((r->otherModes >> 46) & 1) {
		return (struct Color) { v >> 8, v >> 8, v >> 8, v & 255 };
	}
	int cr = (v >> 11) & 31, cg = (v >> 6) & 31, cb = (v >> 1) & 31;
	return (struct Color) { cr << 3 | cr >> 2, cg << 3 | cg >> 2, cb << 3 | cb >> 2, v & 1 ? 255 : 0 };
}

static inline __attribute__((always_inline)) struct Color texel(struct RDP* r, struct Tile* t, int s, int tt) {
	uint32_t base = t->tmem * 8 + tt * t->line * 8;
	uint32_t swap = tt & 1 ? 4 : 0;
	bool tlut = (r->otherModes >> 47) & 1;
	switch (t->size) {
	case 0: {
		uint32_t a = (base + (s >> 1)) ^ swap;
		int v = s & 1 ? tmemByte(r, a) & 15 : tmemByte(r, a) >> 4;
		if (t->format == 2 || tlut) {
			return tlutColor(r, t->palette << 4 | v);
		}
		if (t->format == 3) {
			int i = (v >> 1) * 255 / 7;
			return (struct Color) { i, i, i, v & 1 ? 255 : 0 };
		}
		int i = v * 17;
		return (struct Color) { i, i, i, i };
	}
	case 1: {
		int v = tmemByte(r, (base + s) ^ swap);
		if (t->format == 2 || tlut) {
			return tlutColor(r, v);
		}
		if (t->format == 3) {
			return (struct Color) { (v >> 4) * 17, (v >> 4) * 17, (v >> 4) * 17, (v & 15) * 17 };
		}
		return (struct Color) { v, v, v, v };
	}
	case 2: {
		uint32_t a = (base + s * 2) ^ swap;
		uint16_t v = (uint16_t) (tmemByte(r, a) << 8 | tmemByte(r, a + 1));
		if (t->format == 3) {
			return (struct Color) { v >> 8, v >> 8, v >> 8, v & 255 };
		}
		if (tlut) {
			return tlutColor(r, v >> 8);
		}
		int cr = (v >> 11) & 31, cg = (v >> 6) & 31, cb = (v >> 1) & 31;
		return (struct Color) { cr << 3 | cr >> 2, cg << 3 | cg >> 2, cb << 3 | cb >> 2, v & 1 ? 255 : 0 };
	}
	default: {
		uint32_t a = ((t->tmem * 8 + tt * t->line * 8) & 0x7FF) + s * 2;
		a ^= swap;
		return (struct Color) { tmemByte(r, a), tmemByte(r, a + 1), tmemByte(r, a + 0x800), tmemByte(r, a + 0x801) };
	}
	}
}

// Applies the tile's shift, offset, clamp, mirror and mask to a 10.5 coordinate.
static inline __attribute__((always_inline)) int wrapCoord(int c, int shift, int lo, int hi, int cm, int mask, int* frac) {
	if (shift < 11) {
		c >>= shift;
	} else {
		c <<= 16 - shift;
	}
	c -= lo << 3;
	bool clamp = (cm & 2) || !mask;
	if (clamp) {
		int max = (hi - lo) << 3;
		if (c < 0) {
			c = 0;
		} else if (c > max) {
			c = max;
		}
	}
	*frac = c & 31;
	int i = c >> 5;
	if (mask) {
		if ((cm & 1) && ((i >> mask) & 1)) {
			i = ~i;
		}
		i &= (1 << mask) - 1;
	}
	return i;
}

static struct Color sample(struct RDP* r, int tileIndex, int s, int t) {
	struct Tile* tile = &r->tile[tileIndex & 7];
	int fs, ft;
	int s0 = wrapCoord(s, tile->shiftS, tile->sl, tile->sh, tile->cmS, tile->maskS, &fs);
	int t0 = wrapCoord(t, tile->shiftT, tile->tl, tile->th, tile->cmT, tile->maskT, &ft);
	bool bilinear = ((r->otherModes >> 45) & 1) && ((r->otherModes >> 52) & 3) < 2;
	if (!bilinear) {
		return texel(r, tile, s0, t0);
	}
	int f;
	int s1 = wrapCoord(s + (32 << (tile->shiftS < 11 ? tile->shiftS : 0)), tile->shiftS, tile->sl, tile->sh, tile->cmS,
	                   tile->maskS, &f);
	int t1 = wrapCoord(t + (32 << (tile->shiftT < 11 ? tile->shiftT : 0)), tile->shiftT, tile->tl, tile->th, tile->cmT,
	                   tile->maskT, &f);
	i4 c0 = lanes(texel(r, tile, s0, t0)), c1 = lanes(texel(r, tile, s1, t0)), c2 = lanes(texel(r, tile, s0, t1));
	// Three-point filtering: the triangle of texels nearest the sample point.
	if (fs + ft < 32) {
		return color(c0 + ((c1 - c0) * fs + (c2 - c0) * ft) / 32);
	}
	i4 c3 = lanes(texel(r, tile, s1, t1));
	return color(c3 + ((c2 - c3) * (32 - fs) + (c1 - c3) * (32 - ft)) / 32);
}

// ---- Combiner and blender

static int rgbInput(struct RDP* r, struct Pixel* p, int sel, int channel, int which) {
	const int* c;
	switch (sel) {
	case 0:
		c = &p->combined.r;
		break;
	case 1:
		c = &p->tex0.r;
		break;
	case 2:
		c = &p->tex1.r;
		break;
	case 3:
		c = &r->prim.r;
		break;
	case 4:
		c = &p->shade.r;
		break;
	case 5:
		c = &r->env.r;
		break;
	default:
		if (which == 0 && sel == 6) {
			return 0x100;
		}
		if (which == 0 && sel == 7) {
			return rand() & 255;
		}
		if (which == 1 && sel == 7) {
			return r->k4;
		}
		if (which == 2) {
			switch (sel) {
			case 7:
				return p->combined.a;
			case 8:
				return p->tex0.a;
			case 9:
				return p->tex1.a;
			case 10:
				return r->prim.a;
			case 11:
				return p->shade.a;
			case 12:
				return r->env.a;
			case 13:
				return p->lodFrac;
			case 14:
				return r->primLod;
			case 15:
				return r->k5;
			}
		}
		if (which == 3 && sel == 6) {
			return 0x100;
		}
		return 0;
	}
	return c[channel];
}

static int alphaInput(struct RDP* r, struct Pixel* p, int sel, bool multiplier) {
	switch (sel) {
	case 0:
		return multiplier ? p->lodFrac : p->combined.a;
	case 1:
		return p->tex0.a;
	case 2:
		return p->tex1.a;
	case 3:
		return r->prim.a;
	case 4:
		return p->shade.a;
	case 5:
		return r->env.a;
	case 6:
		return multiplier ? r->primLod : 0x100;
	}
	return 0;
}

// The combiner works on 9-bit values: inputs from 0x180 up are negative, the multiplier is signed,
// and the result keeps 9 bits (Angrylion as the behaviour reference). clamp9 turns a result into the
// pixel's 8 bits: 0x100-0x17F saturate, 0x180-0x1FF are negative and give 0.
static int ext9(int v) {
	v &= 0x1FF;
	return (v & 0x180) == 0x180 ? v - 0x200 : v;
}

static int sign9(int v) {
	v &= 0x1FF;
	return v & 0x100 ? v - 0x200 : v;
}

static int clamp9(int v) {
	v &= 0x1FF;
	return v < 0x100 ? v : v < 0x180 ? 255 : 0;
}

static struct Color combine(struct RDP* r, struct Pixel* p, int cycle) {
	uint64_t c = r->combine;
	int subA = cycle ? (c >> 37) & 15 : (c >> 52) & 15;
	int mul = cycle ? (c >> 32) & 31 : (c >> 47) & 31;
	int subB = cycle ? (c >> 24) & 15 : (c >> 28) & 15;
	int add = cycle ? (c >> 6) & 7 : (c >> 15) & 7;
	int aA = cycle ? (c >> 21) & 7 : (c >> 44) & 7;
	int aMul = cycle ? (c >> 18) & 7 : (c >> 41) & 7;
	int aB = cycle ? (c >> 3) & 7 : (c >> 12) & 7;
	int aAdd = cycle ? c & 7 : (c >> 9) & 7;
	struct Color out;
	int* o = &out.r;
	for (int ch = 0; ch < 3; ++ch) {
		int a = ext9(rgbInput(r, p, subA, ch, 0)), b = ext9(rgbInput(r, p, subB, ch, 1));
		int m = sign9(rgbInput(r, p, mul, ch, 2)), d = ext9(rgbInput(r, p, add, ch, 3));
		o[ch] = (((a - b) * m + (d << 8) + 0x80) & 0x1FFFF) >> 8;
	}
	int a = ext9(alphaInput(r, p, aA, false)), b = ext9(alphaInput(r, p, aB, false));
	int m = sign9(alphaInput(r, p, aMul, true)), d = ext9(alphaInput(r, p, aAdd, false));
	out.a = (((a - b) * m + (d << 8) + 0x80) >> 8) & 0x1FF;
	return out;
}

static const int one = 0x100, zero = 0;

// Where rgbInput and alphaInput take each input from, as pointers into the RDP state.
static struct Input rgbSource(struct RDP* r, int sel, int which, bool* noise) {
	struct Pixel* p = &r->pix;
	switch (sel) {
	case 0:
		return (struct Input) { &p->combined.r, 1 };
	case 1:
		return (struct Input) { &p->tex0.r, 1 };
	case 2:
		return (struct Input) { &p->tex1.r, 1 };
	case 3:
		return (struct Input) { &r->prim.r, 1 };
	case 4:
		return (struct Input) { &p->shade.r, 1 };
	case 5:
		return (struct Input) { &r->env.r, 1 };
	}
	if ((which == 0 || which == 3) && sel == 6) {
		return (struct Input) { &one, 0 };
	}
	if (which == 0 && sel == 7) {
		*noise = true;
	}
	if (which == 1 && sel == 7) {
		return (struct Input) { &r->k4, 0 };
	}
	if (which == 2) {
		static const size_t pixelAlpha[] = { offsetof(struct Pixel, combined.a), offsetof(struct Pixel, tex0.a),
		                                     offsetof(struct Pixel, tex1.a) };
		switch (sel) {
		case 7: case 8: case 9:
			return (struct Input) { (const int*) ((const uint8_t*) p + pixelAlpha[sel - 7]), 0 };
		case 10:
			return (struct Input) { &r->prim.a, 0 };
		case 11:
			return (struct Input) { &p->shade.a, 0 };
		case 12:
			return (struct Input) { &r->env.a, 0 };
		case 13:
			return (struct Input) { &p->lodFrac, 0 };
		case 14:
			return (struct Input) { &r->primLod, 0 };
		case 15:
			return (struct Input) { &r->k5, 0 };
		}
	}
	return (struct Input) { &zero, 0 };
}

static const int* alphaSource(struct RDP* r, int sel, bool multiplier) {
	struct Pixel* p = &r->pix;
	switch (sel) {
	case 0:
		return multiplier ? &p->lodFrac : &p->combined.a;
	case 1:
		return &p->tex0.a;
	case 2:
		return &p->tex1.a;
	case 3:
		return &r->prim.a;
	case 4:
		return &p->shade.a;
	case 5:
		return &r->env.a;
	case 6:
		return multiplier ? &r->primLod : &one;
	}
	return &zero;
}

static bool smallInput(struct RDP* r, const int* v, bool multiplier) {
	struct Pixel* p = &r->pix;
	if (v == &one) {
		return !multiplier;
	}
	return !(v >= &p->combined.r && v <= &p->combined.a) && v != &r->k4 && v != &r->k5 && v != &p->lodFrac;
}

static void setupCombiner(struct RDP* r) {
	uint64_t c = r->combine;
	for (int cycle = 0; cycle < 2; ++cycle) {
		struct Stage* s = &r->stage[cycle];
		s->noise = false;
		s->a = rgbSource(r, cycle ? (c >> 37) & 15 : (c >> 52) & 15, 0, &s->noise);
		s->m = rgbSource(r, cycle ? (c >> 32) & 31 : (c >> 47) & 31, 2, &s->noise);
		s->b = rgbSource(r, cycle ? (c >> 24) & 15 : (c >> 28) & 15, 1, &s->noise);
		s->d = rgbSource(r, cycle ? (c >> 6) & 7 : (c >> 15) & 7, 3, &s->noise);
		s->alpha[0] = alphaSource(r, cycle ? (c >> 21) & 7 : (c >> 44) & 7, false);
		s->alpha[1] = alphaSource(r, cycle ? (c >> 3) & 7 : (c >> 12) & 7, false);
		s->alpha[2] = alphaSource(r, cycle ? (c >> 18) & 7 : (c >> 41) & 7, true);
		s->alpha[3] = alphaSource(r, cycle ? c & 7 : (c >> 9) & 7, false);
		s->raw = smallInput(r, s->a.v, false) && smallInput(r, s->b.v, false) && smallInput(r, s->m.v, true) &&
		         smallInput(r, s->d.v, false) && smallInput(r, s->alpha[0], false) && smallInput(r, s->alpha[1], false) &&
		         smallInput(r, s->alpha[2], true) && smallInput(r, s->alpha[3], false);
	}
}

static void setupModes(struct RDP* r) {
	uint64_t m = r->otherModes;
	r->cycleType = (m >> 52) & 3;
	r->zCompare = (m >> 4) & 1;
	r->zUpdate = (m >> 5) & 1;
	r->zPrim = (m >> 2) & 1;
	r->zMode = (m >> 10) & 3;
	r->alphaCompare = m & 1;
	r->cvgAlpha = (m >> 13) & 1;
	r->forceBlend = (m >> 14) & 1;
	for (int cycle = 0; cycle < 2; ++cycle) {
		for (int i = 0; i < 4; ++i) {
			r->blendSel[cycle][i] = (m >> (30 - cycle * 2 - i * 4)) & 3;
		}
	}
}

// An input's three channels (from a struct Color, or one value), with alpha in the last lane.
static i4 input4(const struct Input* in, const int* alpha) {
	i4 v;
	if (in->step) {
		memcpy(&v, in->v, sizeof(v));
	} else {
		v = splat4(*in->v);
	}
	v[3] = *alpha;
	return v;
}

static i4 ext9x4(i4 v) {
	v &= 0x1FF;
	return v - (((v & 0x180) == 0x180) & 0x200);
}

// combine() through the inputs setupCombiner found, for modes without noise; color and alpha
// keep the same 9 bits ((x & 0x1FFFF) >> 8 is (x >> 8) & 0x1FF).
static struct Color combineInputs(const struct Stage* s) {
	i4 a = input4(&s->a, s->alpha[0]), b = input4(&s->b, s->alpha[1]);
	i4 m = input4(&s->m, s->alpha[2]), d = input4(&s->d, s->alpha[3]);
	if (!s->raw) {
		a = ext9x4(a), b = ext9x4(b), d = ext9x4(d);
		m = (m << 23) >> 23;
	}
	return color((((a - b) * m + (d << 8) + 0x80) >> 8) & 0x1FF);
}

static struct Color combineCycle(struct RDP* r, struct Pixel* p, int cycle) {
	return r->stage[cycle].noise ? combine(r, p, cycle) : combineInputs(&r->stage[cycle]);
}

static struct Color blendColor(struct RDP* r, struct Color pixel, struct Color memory, struct Pixel* p, int cycle) {
	const uint8_t* sel = r->blendSel[cycle];
	int sp = sel[0], sa = sel[1], sm = sel[2], sb = sel[3];
	const struct Color* choices[4] = { &pixel, &memory, &r->blend, &r->fog };
	struct Color P = *choices[sp], M = *choices[sm];
	int a = sa == 0 ? pixel.a : sa == 1 ? r->fog.a : sa == 2 ? p->shade.a : 0;
	int b = sb == 0 ? 255 - a : sb == 1 ? memory.a : sb == 2 ? 255 : 0;
	// The blender works with 5-bit factors and doesn't normalise: (P * a + M * (b + 1)) / 32, wrapping
	// at 8 bits, so "one" for b adds.
	int fa = a >> 3, fb = (b >> 3) + 1;
	i4 out = ((lanes(P) * fa + lanes(M) * fb) >> 5) & 255;
	out[3] = pixel.a;
	return color(out);
}

// The blender's first input (P) for a cycle, without blending.
static struct Color blendInput(struct RDP* r, struct Color pixel, struct Color memory, int cycle) {
	int sp = r->blendSel[cycle][0];
	struct Color out = sp == 0 ? pixel : sp == 1 ? memory : sp == 2 ? r->blend : r->fog;
	out.a = pixel.a;
	return out;
}

static void shadePixel(struct N64* n, struct RDP* r, int x, int y, struct Pixel* p) {
	int cycleType = r->cycleType;
	r->clocks += cycleType == 1 ? 2 : 1;
	int z = r->zPrim ? r->primZ : p->z;
	if (z < 0) {
		z = 0;
	} else if (z > 0x3FFFF) {
		z = 0x3FFFF;
	}
	if (r->zAddr && r->zCompare) {
		int old = zDecompress(zRead(n, r, x, y));
		bool pass = r->zMode == 3 ? z <= old + 0x10 && z >= old - 0x10 : z <= old;
		if (!pass) {
			return;
		}
	}
	// One-cycle mode runs the combiner's second-cycle equation.
	p->combined = r->lastCombined;
	struct Color c = combineCycle(r, p, cycleType == 1 ? 0 : 1);
	if (cycleType == 1) {
		p->combined = c;
		c = combineCycle(r, p, 1);
	}
	r->lastCombined = c;
	c = (struct Color) { clamp9(c.r), clamp9(c.g), clamp9(c.b), clamp9(c.a) };
	if (r->alphaCompare && c.a < r->blend.a) {
		return;
	}
	// With alpha as coverage, alpha becomes the 3-bit coverage; a pixel with none isn't written.
	if (r->cvgAlpha && c.a < 32) {
		return;
	}
	struct Color memory = readPixel(n, r, x, y);
	// The blend formula applies only with force blend (or at partly covered edges, which aren't
	// modelled yet); otherwise a cycle passes its first input through. In 2-cycle mode the first
	// cycle always blends (fog), and its result is the second cycle's pixel.
	bool force = r->forceBlend;
	struct Color out = c;
	if (cycleType == 1) {
		struct Color first = blendColor(r, c, memory, p, 0);
		out = force ? blendColor(r, first, memory, p, 1) : blendInput(r, first, memory, 1);
	} else {
		out = force ? blendColor(r, c, memory, p, 0) : blendInput(r, c, memory, 0);
	}
	writePixel(n, r, x, y, out);
	if (r->zAddr && r->zUpdate) {
		zWrite(n, r, x, y, zCompress(z));
	}
}

// ---- Primitives

static void fillRect(struct N64* n, struct RDP* r, int x0, int y0, int x1, int y1) {
	if (x1 >= x0 && y1 >= y0) {
		r->clocks += (uint64_t) (x1 - x0 + 1) * (y1 - y0 + 1) / (r->colorSize == 3 ? 2 : 4);
	}
	int bytes = r->colorSize == 3 ? 4 : 2;
	for (int y = y0; y <= y1; ++y) {
		// A row that stays inside RDRAM, at an address the pixel size divides: big-endian stores.
		uint32_t first = r->colorAddr + (y * r->colorWidth + x0) * bytes;
		uint32_t last = r->colorAddr + (y * r->colorWidth + x1) * bytes + bytes - 1;
		if (r->colorSize >= 2 && x1 >= x0 && !(r->colorAddr & (bytes - 1)) && first <= last && last < RDRAM_SIZE) {
			uint8_t* row = &n->rdram[first];
			if (bytes == 4) {
				for (int x = x0; x <= x1; ++x) {
					put32(row + (x - x0) * 4, r->fillColor);
				}
			} else {
				uint16_t even = __builtin_bswap16((uint16_t) (r->fillColor >> 16)), odd = __builtin_bswap16((uint16_t) r->fillColor);
				for (int x = x0; x <= x1; ++x) {
					memcpy(row + (x - x0) * 2, x & 1 ? &odd : &even, 2);
				}
			}
			continue;
		}
		for (int x = x0; x <= x1; ++x) {
			uint32_t i = y * r->colorWidth + x;
			if (r->colorSize == 1) {
				uint32_t a = (r->colorAddr + i) & (RDRAM_SIZE - 1);
				n->rdram[a] = (uint8_t) (r->fillColor >> ((3 - (a & 3)) * 8));
			} else if (r->colorSize == 3) {
				put32(&n->rdram[(r->colorAddr + i * 4) & (RDRAM_SIZE - 4)], r->fillColor);
			} else {
				uint32_t a = (r->colorAddr + i * 2) & (RDRAM_SIZE - 2);
				uint16_t v = (uint16_t) (x & 1 ? r->fillColor : r->fillColor >> 16);
				n->rdram[a] = v >> 8;
				n->rdram[a + 1] = (uint8_t) v;
			}
		}
	}
}

static void clipRect(struct RDP* r, int* x0, int* y0, int* x1, int* y1) {
	if (*x0 < r->scissorX0 >> 2) {
		*x0 = r->scissorX0 >> 2;
	}
	if (*y0 < r->scissorY0 >> 2) {
		*y0 = r->scissorY0 >> 2;
	}
	if (*x1 > (r->scissorX1 >> 2) - 1) {
		*x1 = (r->scissorX1 >> 2) - 1;
	}
	if (*y1 > (r->scissorY1 >> 2) - 1) {
		*y1 = (r->scissorY1 >> 2) - 1;
	}
}

static int t0Coord(int t0, int d, int dtdy) {
	return ((t0 << 5) + d * dtdy) >> 5;
}

// Texture coordinates for the texture unit: S and T over W when perspective is on.
static void texCoords(int64_t sS, int64_t sT, int64_t sW, bool persp, int* s, int* t) {
	if (persp) {
		int wv = (int) ((sW >> 16) & 0x7FFF);
		if (wv == 0) {
			wv = 1;
		}
		*s = (int) (((sS >> 16) * 0x8000) / wv);
		*t = (int) (((sT >> 16) * 0x8000) / wv);
	} else {
		*s = (int) (sS >> 16);
		*t = (int) (sT >> 16);
	}
}

// Whether the combiner reads the LOD fraction in the cycles in use (then the RDP works out the LOD
// even with mipmapping off).
static bool lodFracUsed(struct RDP* r) {
	uint64_t c = r->combine;
	bool cycle1 = ((c >> 32) & 31) == 13 || ((c >> 18) & 7) == 0;
	bool cycle0 = ((c >> 47) & 31) == 13 || ((c >> 41) & 7) == 0;
	return cycle1 || (((r->otherModes >> 52) & 3) == 1 && cycle0);
}

// Level of detail from the texel step to the next pixel and the next line: picks the mipmap tiles
// and the fraction between them (the RDP's rules; Angrylion as the behaviour reference).
static int levelOfDetail(struct RDP* r, int primTile, int s, int t, int sx, int tx, int sy, int ty, int* t1, int* t2) {
	uint64_t m = r->otherModes;
	bool lodEn = (m >> 48) & 1, sharpen = (m >> 49) & 1, detail = (m >> 50) & 1;
	bool clamp = s < -0x10000 || s > 0xFFFF || t < -0x10000 || t > 0xFFFF || sx < -0x10000 || sx > 0xFFFF ||
	             tx < -0x10000 || tx > 0xFFFF || sy < -0x10000 || sy > 0xFFFF || ty < -0x10000 || ty > 0xFFFF;
	int lod = 0;
	if (!clamp) {
		int d[4] = { abs(sx - s), abs(tx - t), abs(sy - s), abs(ty - t) };
		for (int i = 0; i < 4; ++i) {
			lod = d[i] > lod ? d[i] : lod;
		}
		lod = lod >= 0x4000 ? (lod & 0x3FFF) | 0x4000 : lod;
	}
	int tile = 0, frac;
	bool magnify = false, distant;
	if ((lod & 0x4000) || clamp) {
		distant = true;
		frac = 0xFF;
	} else if (lod < 32) {
		magnify = true;
		distant = r->maxLevel == 0;
		int base = lod < r->minLevel ? r->minLevel : lod;
		frac = !sharpen && !detail ? (distant ? 0xFF : 0) : (base << 3) | (sharpen ? 0x100 : 0);
	} else {
		int v = (lod >> 5) & 0xFF;
		while (v > 1) {
			v >>= 1;
			++tile;
		}
		distant = !r->maxLevel || (lod & 0x6000) || tile >= r->maxLevel;
		frac = !sharpen && !detail && distant ? 0xFF : ((lod << 3) >> tile) & 0xFF;
	}
	*t1 = primTile;
	*t2 = primTile + 1;
	if (lodEn) {
		if (distant) {
			tile = r->maxLevel;
		}
		if (!detail) {
			*t1 = (primTile + tile) & 7;
			*t2 = distant || (!sharpen && magnify) ? *t1 : (*t1 + 1) & 7;
		} else {
			*t1 = (primTile + tile + (magnify ? 0 : 1)) & 7;
			*t2 = (primTile + tile + (!distant && !magnify ? 2 : 1)) & 7;
		}
	}
	return frac;
}

static void textureRect(struct N64* n, struct RDP* r, uint64_t w0, uint64_t w1, bool flip) {
	int xl = (w0 >> 44) & 0xFFF, yl = (w0 >> 32) & 0xFFF, tile = (w0 >> 24) & 7;
	int xh = (w0 >> 12) & 0xFFF, yh = w0 & 0xFFF;
	int s0 = (int16_t) (w1 >> 48), t0 = (int16_t) (w1 >> 32);
	int dsdx = (int16_t) (w1 >> 16), dtdy = (int16_t) w1;
	int cycleType = (r->otherModes >> 52) & 3;
	int x0 = xh >> 2, y0 = yh >> 2;
	int x1 = cycleType >= 2 ? xl >> 2 : ((xl + 3) >> 2) - 1;
	int y1 = cycleType >= 2 ? yl >> 2 : ((yl + 3) >> 2) - 1;
	if (cycleType == 2) {
		dsdx /= 4;
	}
	int cx0 = x0, cy0 = y0;
	clipRect(r, &x0, &y0, &x1, &y1);
	uint64_t copied = 0;
	r->maxLevel = 0;
	bool doLod = cycleType < 2 && (((r->otherModes >> 48) & 1) || lodFracUsed(r));
	for (int y = y0; y <= y1; ++y) {
		for (int x = x0; x <= x1; ++x) {
			int dx = x - cx0, dy = y - cy0;
			int s = ((s0 << 5) + (flip ? dy * dsdx : dx * dsdx)) >> 5;
			int t = ((t0 << 5) + (flip ? dx * dtdy : dy * dtdy)) >> 5;
			struct Pixel* p = &r->pix;
			*p = (struct Pixel) { 0 };
			int t1 = tile, t2 = tile + 1;
			if (doLod) {
				int sx = s + (flip ? 0 : dsdx >> 5), tx = t + (flip ? dtdy >> 5 : 0);
				int sy = s + (flip ? dsdx >> 5 : 0), ty = t + (flip ? 0 : dtdy >> 5);
				p->lodFrac = levelOfDetail(r, tile, s, t, sx, tx, sy, ty, &t1, &t2);
			}
			p->tex0 = sample(r, t1, s, t);
			p->tex1 = cycleType == 1 ? sample(r, t2, s, t) : p->tex0;
			if (cycleType == 2) {
				++copied;
				if ((r->otherModes & 1) && p->tex0.a == 0) {
					continue;
				}
				// Copying into an 8-bit image moves the texel bytes themselves (palette indices).
				if (r->colorSize == 1 && r->tile[tile & 7].size == 1) {
					struct Tile* t = &r->tile[tile & 7];
					int f;
					int ss = wrapCoord(s, t->shiftS, t->sl, t->sh, t->cmS, t->maskS, &f);
					int tt = wrapCoord(t0Coord(t0, flip ? dx : dy, dtdy), t->shiftT, t->tl, t->th, t->cmT, t->maskT, &f);
					uint32_t a = (t->tmem * 8 + tt * t->line * 8 + ss) ^ (tt & 1 ? 4 : 0);
					n->rdram[(r->colorAddr + y * r->colorWidth + x) & (RDRAM_SIZE - 1)] = tmemByte(r, a);
					continue;
				}
				writePixel(n, r, x, y, p->tex0);
				continue;
			}
			p->z = r->primZ;
			shadePixel(n, r, x, y, p);
		}
	}
	r->clocks += copied / 4;
}

static int32_t fix(uint64_t w, int shift) {
	return (int32_t) (uint32_t) (w >> shift);
}

static void triangle(struct N64* n, struct RDP* r, const uint64_t* w, int op) {
	bool shade = op & 4, tex = op & 2, zbuf = op & 1;
	int tile = (w[0] >> 48) & 7;
	int yl = ((int32_t) ((w[0] >> 32) & 0x3FFF) << 18) >> 18;
	int ym = ((int32_t) ((w[0] >> 16) & 0x3FFF) << 18) >> 18;
	int yh = ((int32_t) (w[0] & 0x3FFF) << 18) >> 18;
	int64_t xl = fix(w[1], 32), dxl = fix(w[1], 0);
	int64_t xh = fix(w[2], 32), dxh = fix(w[2], 0);
	int64_t xm = fix(w[3], 32), dxm = fix(w[3], 0);
	int idx = 4;
	int64_t sv[4] = { 0 }, sdx[4] = { 0 }, sde[4] = { 0 };
	int64_t tv[3] = { 0 }, tdx[3] = { 0 }, tde[3] = { 0 }, tdy[3] = { 0 };
	int64_t zv = 0, zdx = 0, zde = 0;
	if (shade) {
		for (int i = 0; i < 4; ++i) {
			int sh = 48 - i * 16;
			sv[i] = (int64_t) ((int32_t) ((uint32_t) ((w[idx] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 2] >> sh) & 0xFFFF)));
			sdx[i] = (int32_t) ((uint32_t) ((w[idx + 1] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 3] >> sh) & 0xFFFF));
			sde[i] = (int32_t) ((uint32_t) ((w[idx + 4] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 6] >> sh) & 0xFFFF));
		}
		idx += 8;
	}
	if (tex) {
		for (int i = 0; i < 3; ++i) {
			int sh = 48 - i * 16;
			tv[i] = (int32_t) ((uint32_t) ((w[idx] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 2] >> sh) & 0xFFFF));
			tdx[i] = (int32_t) ((uint32_t) ((w[idx + 1] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 3] >> sh) & 0xFFFF));
			tde[i] = (int32_t) ((uint32_t) ((w[idx + 4] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 6] >> sh) & 0xFFFF));
			tdy[i] = (int32_t) ((uint32_t) ((w[idx + 5] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 7] >> sh) & 0xFFFF));
		}
		idx += 8;
	}
	r->maxLevel = (w[0] >> 51) & 7;
	bool doLod = tex && (((r->otherModes >> 48) & 1) || lodFracUsed(r));
	if (zbuf) {
		zv = fix(w[idx], 32);
		zdx = fix(w[idx], 0);
		zde = fix(w[idx + 1], 32);
	}
	int cycleType = (r->otherModes >> 52) & 3;
	bool persp = (r->otherModes >> 51) & 1;
	int top = yh < r->scissorY0 ? r->scissorY0 : yh;
	int bottom = yl > r->scissorY1 ? r->scissorY1 : yl;
	for (int y = (top + 3) >> 2; y < (bottom + 3) >> 2; ++y) {
		// Edges and attributes are evaluated at the middle of the scanline, in quarter lines.
		int64_t q = (int64_t) y * 4 + 2;
		if (q < yh || q >= yl) {
			continue;
		}
		int64_t dy = q - (yh & ~3);
		int64_t major = xh + dxh * dy / 4;
		int64_t minor = q < ym ? xm + dxm * dy / 4 : xl + dxl * (q - ym) / 4;
		int64_t left = major < minor ? major : minor, right = major < minor ? minor : major;
		int x0 = (int) ((left + 0xFFFF) >> 16), x1 = (int) ((right + 0xFFFF) >> 16) - 1;
		if (x0 < r->scissorX0 >> 2) {
			x0 = r->scissorX0 >> 2;
		}
		if (x1 > (r->scissorX1 >> 2) - 1) {
			x1 = (r->scissorX1 >> 2) - 1;
		}
		int64_t ey = dy / 4;
		// Each attribute at x0, then plus its x gradient a pixel (x moves dx by whole 65536s, so the
		// shifted products step exactly).
		int64_t dx0 = (int64_t) x0 * 65536 + 32768 - xh - dxh * dy / 4;
		int64_t shadeAt[4], zAt = 0, texAt[3];
		for (int i = 0; i < 4; ++i) {
			shadeAt[i] = sv[i] + sde[i] * ey + (sdx[i] * dx0 >> 16);
		}
		for (int i = 0; i < 3; ++i) {
			texAt[i] = tv[i] + tde[i] * ey + (tdx[i] * dx0 >> 16);
		}
		if (zbuf) {
			zAt = zv + zde * ey + (zdx * dx0 >> 16);
		}
		for (int x = x0; x <= x1; ++x) {
			struct Pixel* p = &r->pix;
			*p = (struct Pixel) { 0 };
			if (shade) {
				p->shade = (struct Color) { clamp8((int) (shadeAt[0] >> 16)), clamp8((int) (shadeAt[1] >> 16)),
				                            clamp8((int) (shadeAt[2] >> 16)), clamp8((int) (shadeAt[3] >> 16)) };
				for (int i = 0; i < 4; ++i) {
					shadeAt[i] += sdx[i];
				}
			}
			if (zbuf) {
				p->z = (int) (zAt >> 13);
				zAt += zdx;
			}
			if (tex) {
				int64_t sS = texAt[0], sT = texAt[1], sW = texAt[2];
				for (int i = 0; i < 3; ++i) {
					texAt[i] += tdx[i];
				}
				int s, t, t1 = tile, t2 = tile + 1;
				texCoords(sS, sT, sW, persp, &s, &t);
				if (doLod) {
					int sx, tx, sy, ty;
					texCoords(sS + tdx[0], sT + tdx[1], sW + tdx[2], persp, &sx, &tx);
					texCoords(sS + tdy[0], sT + tdy[1], sW + tdy[2], persp, &sy, &ty);
					p->lodFrac = levelOfDetail(r, tile, s, t, sx, tx, sy, ty, &t1, &t2);
				}
				p->tex0 = sample(r, t1, s, t);
				p->tex1 = cycleType == 1 ? sample(r, t2, s, t) : p->tex0;
			}
			shadePixel(n, r, x, y, p);
		}
	}
}

// ---- Texture loads

static void loadTile(struct N64* n, struct RDP* r, uint64_t w, bool block) {
	struct Tile* t = &r->tile[(w >> 24) & 7];
	int sl = (w >> 44) & 0xFFF, tl = (w >> 32) & 0xFFF, sh = (w >> 12) & 0xFFF, th = w & 0xFFF;
	int bpp = r->texSize == 0 ? 0 : 1 << (r->texSize - 1);
	if (block) {
		int count = sh - sl + 1;
		uint32_t src = r->texAddr + (tl * r->texWidth + sl) * bpp;
		uint32_t bytes = (uint32_t) count * (bpp ? bpp : 1);
		if (r->texSize == 0) {
			bytes = (uint32_t) (count + 1) / 2;
		}
		int dxt = th;
		int lineT = 0;
		for (uint32_t i = 0; i < bytes; i += 8) {
			bool odd = (lineT >> 11) & 1;
			for (int b = 0; b < 8; ++b) {
				uint32_t dst = (t->tmem * 8 + i + b) ^ (odd ? 4 : 0);
				if (r->texSize == 3) {
					uint32_t texelIndex = (i + b) / 4, part = (i + b) & 3;
					uint32_t d = (t->tmem * 8 + texelIndex * 2 + (part & 1)) ^ (odd ? 4 : 0);
					r->tmem[(d + (part >= 2 ? 0x800 : 0)) & 0xFFF] = n->rdram[(src + i + b) & (RDRAM_SIZE - 1)];
				} else {
					r->tmem[dst & 0xFFF] = n->rdram[(src + i + b) & (RDRAM_SIZE - 1)];
				}
			}
			lineT += dxt;
		}
		t->sl = sl;
		t->tl = tl;
		t->sh = sh;
		t->th = tl;
		return;
	}
	int s0 = sl >> 2, t0 = tl >> 2, s1 = sh >> 2, t1 = th >> 2;
	for (int tt = t0; tt <= t1; ++tt) {
		uint32_t row = r->texAddr + (uint32_t) (tt * r->texWidth) * (bpp ? bpp : 1);
		uint32_t line = t->tmem * 8 + (tt - t0) * t->line * 8;
		uint32_t swap = (tt - t0) & 1 ? 4 : 0;
		if (r->texSize == 0) {
			for (int s = s0; s <= s1; s += 2) {
				r->tmem[((line + (s - s0) / 2) ^ swap) & 0xFFF] = n->rdram[(row + s / 2) & (RDRAM_SIZE - 1)];
			}
		} else if (r->texSize == 3) {
			for (int s = s0; s <= s1; ++s) {
				const uint8_t* src = &n->rdram[(row + s * 4) & (RDRAM_SIZE - 4)];
				uint32_t d = (line + (s - s0) * 2) ^ swap;
				r->tmem[d & 0x7FF] = src[0];
				r->tmem[(d + 1) & 0x7FF] = src[1];
				r->tmem[0x800 | (d & 0x7FF)] = src[2];
				r->tmem[0x800 | ((d + 1) & 0x7FF)] = src[3];
			}
		} else {
			for (int s = s0; s <= s1; ++s) {
				for (int b = 0; b < bpp; ++b) {
					uint32_t d = (line + (s - s0) * bpp + b) ^ swap;
					r->tmem[d & 0xFFF] = n->rdram[(row + s * bpp + b) & (RDRAM_SIZE - 1)];
				}
			}
		}
	}
	t->sl = sl;
	t->tl = tl;
	t->sh = sh;
	t->th = th;
}

static void loadTlut(struct N64* n, struct RDP* r, uint64_t w) {
	struct Tile* t = &r->tile[(w >> 24) & 7];
	int sl = ((w >> 44) & 0xFFF) >> 2, sh = ((w >> 12) & 0xFFF) >> 2;
	for (int i = sl; i <= sh; ++i) {
		uint32_t src = (r->texAddr + i * 2) & (RDRAM_SIZE - 2);
		uint32_t dst = t->tmem * 8 + (i - sl) * 8;
		for (int copy = 0; copy < 4; ++copy) {
			r->tmem[(dst + copy * 2) & 0xFFF] = n->rdram[src];
			r->tmem[(dst + copy * 2 + 1) & 0xFFF] = n->rdram[src + 1];
		}
	}
}

// ---- Command processing

static int commandWords(int op) {
	if (op >= 0x08 && op <= 0x0F) {
		return 4 + (op & 4 ? 8 : 0) + (op & 2 ? 8 : 0) + (op & 1 ? 2 : 0);
	}
	return op == 0x24 || op == 0x25 ? 2 : 1;
}

// Rough cost model: one or two clocks a pixel by cycle type, four pixels a clock to fill or copy,
// and a fixed setup per primitive. A full sync interrupts the CPU once that time has passed.
void rdpRun(struct N64* n) {
	if (n->dp.status & 2) {
		return;
	}
	struct RDP* r = state(n);
	uint64_t begin = n->dp.busyUntil > n->cycles ? n->dp.busyUntil : n->cycles;
	r->clocks = 0;
	while (n->dp.current < n->dp.end) {
		uint64_t w[22];
		w[0] = word(n, n->dp.current);
		int op = (w[0] >> 56) & 0x3F;
		int words = commandWords(op);
		if (n->dp.current + words * 8 > n->dp.end) {
			break;
		}
		for (int i = 1; i < words; ++i) {
			w[i] = word(n, n->dp.current + i * 8);
		}
		n->dp.current += words * 8;
		uint64_t c = w[0];
		switch (op) {
		case 0x08:
		case 0x09:
		case 0x0A:
		case 0x0B:
		case 0x0C:
		case 0x0D:
		case 0x0E:
		case 0x0F:
			triangle(n, r, w, op);
			r->clocks += 20;
			break;
		case 0x24:
		case 0x25:
			textureRect(n, r, w[0], w[1], op == 0x25);
			r->clocks += 10;
			break;
		case 0x29:
			++n->stats.dpSyncs;
			if (n->dp.syncAt) {
				n64Interrupt(n, MI_DP, true);
			}
			n->dp.syncAt = begin + r->clocks * 3 / 2;
			n64Event(n, n->dp.syncAt);
			break;
		case 0x2C:
			r->k4 = (c >> 9) & 0x1FF;
			r->k5 = c & 0x1FF;
			break;
		case 0x2D:
			r->scissorX0 = (c >> 44) & 0xFFF;
			r->scissorY0 = (c >> 32) & 0xFFF;
			r->scissorX1 = (c >> 12) & 0xFFF;
			r->scissorY1 = c & 0xFFF;
			break;
		case 0x2E:
			r->primZ = (int) ((c >> 16) & 0x7FFF) << 3;
			r->primDz = c & 0xFFFF;
			break;
		case 0x2F:
			r->otherModes = c;
			setupModes(r);
			break;
		case 0x30:
			loadTlut(n, r, c);
			break;
		case 0x32: {
			struct Tile* t = &r->tile[(c >> 24) & 7];
			t->sl = (c >> 44) & 0xFFF;
			t->tl = (c >> 32) & 0xFFF;
			t->sh = (c >> 12) & 0xFFF;
			t->th = c & 0xFFF;
			break;
		}
		case 0x33:
			loadTile(n, r, c, true);
			break;
		case 0x34:
			loadTile(n, r, c, false);
			break;
		case 0x35: {
			struct Tile* t = &r->tile[(c >> 24) & 7];
			t->format = (c >> 53) & 7;
			t->size = (c >> 51) & 3;
			t->line = (c >> 41) & 0x1FF;
			t->tmem = (c >> 32) & 0x1FF;
			t->palette = (c >> 20) & 15;
			t->cmT = (c >> 18) & 3;
			t->maskT = (c >> 14) & 15;
			t->shiftT = (c >> 10) & 15;
			t->cmS = (c >> 8) & 3;
			t->maskS = (c >> 4) & 15;
			t->shiftS = c & 15;
			break;
		}
		case 0x36: {
			int x1 = ((c >> 44) & 0xFFF) >> 2, y1 = ((c >> 32) & 0xFFF) >> 2;
			int x0 = ((c >> 12) & 0xFFF) >> 2, y0 = (c & 0xFFF) >> 2;
			int cycleType = (r->otherModes >> 52) & 3;
			clipRect(r, &x0, &y0, &x1, &y1);
			if (cycleType == 3) {
				fillRect(n, r, x0, y0, x1, y1);
			} else {
				for (int y = y0; y <= y1; ++y) {
					for (int x = x0; x <= x1; ++x) {
						struct Pixel* p = &r->pix;
						*p = (struct Pixel) { 0 };
						p->z = r->primZ;
						shadePixel(n, r, x, y, p);
					}
				}
			}
			break;
		}
		case 0x37:
			r->fillColor = (uint32_t) c;
			break;
		case 0x38:
			r->fog = unpack((uint32_t) c);
			break;
		case 0x39:
			r->blend = unpack((uint32_t) c);
			break;
		case 0x3A:
			r->prim = unpack((uint32_t) c);
			r->primLod = (c >> 32) & 0xFF;
			r->minLevel = (c >> 40) & 0x1F;
			break;
		case 0x3B:
			r->env = unpack((uint32_t) c);
			break;
		case 0x3C:
			r->combine = c;
			setupCombiner(r);
			break;
		case 0x3D:
			r->texFormat = (c >> 53) & 7;
			r->texSize = (c >> 51) & 3;
			r->texWidth = ((c >> 32) & 0x3FF) + 1;
			r->texAddr = c & 0x3FFFFFF;
			break;
		case 0x3E:
			r->zAddr = c & 0x3FFFFFF;
			break;
		case 0x3F:
			r->colorFormat = (c >> 53) & 7;
			r->colorSize = (c >> 51) & 3;
			r->colorWidth = ((c >> 32) & 0x3FF) + 1;
			r->colorAddr = c & 0x3FFFFFF;
			break;
		}
	}
	n->dp.busyUntil = begin + r->clocks * 3 / 2;
}

void rdpDestroy(struct N64* n) {
	free(n->rdp);
	n->rdp = NULL;
}
