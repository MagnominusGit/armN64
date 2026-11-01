#include "n64.h"
#include "gles.h"
#include "rdp.h"

#include <pthread.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

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

static bool ownsRow(const struct RDP* r, int y) {
	return r->rowCount == 1 || (r->rowCount && (uint32_t) y % (uint32_t) r->rowCount == (uint32_t) r->rowIndex);
}

static void setupCombiner(struct RDP* r);
static void setupModes(struct RDP* r);
static void gpuSync(struct N64* n, struct RDP* r);
static bool gpuDrew(struct Gpu* g, uint32_t addr, uint32_t len);
static bool gpuReadNeeds(struct Gpu* g, uint32_t addr, uint32_t len);
static bool gpuWrite(struct N64* n, struct RDP* r, uint32_t addr, uint32_t len);
static void gpuSyncRead(struct N64* n, struct RDP* r, uint32_t addr, uint32_t len);
static void gpuUpdate(struct N64* n, struct Gpu* g);
static void buildBlendDivide(void);

static struct RDP* state(struct N64* n) {
	if (!n->rdp) {
		n->rdp = calloc(1, sizeof(struct RDP));
		buildBlendDivide();
		if (n->rdp) {
			n->rdp->rowCount = 1;
			setupCombiner(n->rdp);
			setupModes(n->rdp);
		}
	}
	return n->rdp;
}

// Noise: a hash of the pixel and the primitive (the hardware's sequence isn't known), the same
// however many renderers draw and after a save state.
static uint32_t pixelNoise(const struct RDP* r, int x, int y, uint32_t tag) {
	uint32_t h = (uint32_t) x * 0x9E3779B1u ^ (uint32_t) y * 0x85EBCA77u ^ (uint32_t) r->seq * 0xC2B2AE3Du ^ tag;
	h ^= h >> 15;
	h *= 0x2C1B3C6Du;
	h ^= h >> 12;
	return h;
}

// The magic square and Bayer dither matrices (paraLLEl-RDP's dither.h, MIT).
static const uint8_t ditherMatrix[2][16] = {
	{ 0, 6, 1, 7, 4, 2, 5, 3, 3, 5, 2, 4, 7, 1, 6, 0 },
	{ 0, 4, 1, 5, 4, 0, 5, 1, 3, 7, 2, 6, 7, 3, 6, 2 },
};

// Whether the pixels take noise: from the combiner, a noise dither or the alpha test's dither.
static bool usesNoise(const struct RDP* r) {
	uint64_t m = r->otherModes;
	return r->stage[r->cycleType == 1 ? 0 : 1].noise || (r->cycleType == 1 && r->stage[1].noise) || ((m >> 38) & 3) == 2 || ((m >> 36) & 3) == 2 ||
	       ((m & 1) && ((m >> 1) & 1));
}

// A pixel's dither: three bits for each colour channel (red lowest) and alpha's.
static inline void ditherAt(const struct RDP* r, int x, int y, int* rgb, int* alpha) {
	int const modeRgb = r->ditherRgb, modeAlpha = r->ditherAlpha;
	if (modeRgb == 3 && modeAlpha == 3) {
		*rgb = *alpha = 0;
		return;
	}
	int i = (y & 3) * 4 + (x & 3);
	*rgb = modeRgb < 2 ? ditherMatrix[modeRgb][i] * 0x49 : modeRgb == 2 ? (int) (pixelNoise(r, x, y, 4) & 0x1FF) : 0;
	if (modeAlpha == 3) {
		*alpha = 0;
	} else if (modeAlpha == 2) {
		*alpha = (int) (pixelNoise(r, x, y, 5) & 7);
	} else {
		*alpha = modeRgb >= 2 ? ditherMatrix[modeRgb & 1][i] : *rgb & 7;
		*alpha = modeAlpha == 1 ? ~*alpha & 7 : *alpha;
	}
}

static uint64_t word(struct N64* n, uint32_t addr) {
	if (!(n->dp.status & 1) && (n->pageFlags[(addr & (RDRAM_SIZE - 8)) >> 12] & PAGE_RDP_WRITES)) {
		rdpWaitRead(n, addr & (RDRAM_SIZE - 8), 8);
	}
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
		struct Color c = unpack(be32(&n->rdram[a]));
		c.a &= 0xE0;
		return c;
	}
	uint32_t a = (r->colorAddr + i * 2) & (RDRAM_SIZE - 2);
	uint16_t p = (uint16_t) (n->rdram[a] << 8 | n->rdram[a + 1]);
	int cr = (p >> 11) & 31, cg = (p >> 6) & 31, cb = (p >> 1) & 31;
	// The coverage: the colour's low bit over the two hidden bits.
	return (struct Color) { cr << 3, cg << 3, cb << 3, ((p & 1) << 2 | n->hidden[a >> 1]) << 5 };
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
	n->hidden[a >> 1] = (c.a >> 5) & 3;
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

// The depth slope's low two bits are in the hidden bits.
static int zHidden(struct N64* n, struct RDP* r, int x, int y) {
	return n->hidden[((r->zAddr + (y * r->colorWidth + x) * 2) & (RDRAM_SIZE - 2)) >> 1];
}

static void zWrite(struct N64* n, struct RDP* r, int x, int y, uint16_t v, int hidden) {
	uint8_t* p = (uint8_t*) zPointer(n, r, x, y, NULL);
	p[0] = v >> 8;
	p[1] = (uint8_t) v;
	n->hidden[((r->zAddr + (y * r->colorWidth + x) * 2) & (RDRAM_SIZE - 2)) >> 1] = (uint8_t) hidden;
}

// A depth slope (|dz/dx| + |dz/dy| in whole steps) as the power of two above it, and that power's
// exponent (dz_compress).
static int normalizeDz(int sum) {
	if (sum & 0xC000) {
		return 0x8000;
	}
	if (!(sum & 0xFFFF)) {
		return 1;
	}
	if (sum == 1) {
		return 3;
	}
	for (int bit = 0x2000; bit > 0; bit >>= 1) {
		if (sum & bit) {
			return bit << 1;
		}
	}
	return 0;
}

static int dzEncode(int v) {
	return (v & 0xFF00 ? 8 : 0) | (v & 0xF0F0 ? 4 : 0) | (v & 0xCCCC ? 2 : 0) | (v & 0xAAAA ? 1 : 0);
}

static int highestBit(int v) {
	return v ? 1 << (31 - __builtin_clz((unsigned) v)) : 0;
}

// The blender's divider (for antialiased blends): the hardware's bit-serial division of a sum of
// products by the sum of its 4-bit factors.
static uint8_t blendDivide[0x8000];

static void buildBlendDivide(void) {
	static bool built;
	if (built) {
		return;
	}
	built = true;
	for (int i = 0; i < 0x8000; ++i) {
		int d = (i >> 11) & 15, n = i & 0x7FF, inv = ~d & 15, res = 0;
		int partial = (inv + (n >> 8) + 1) & 7;
		for (int k = 0; k < 8; ++k) {
			int bit = (n >> (7 - k)) & 1;
			int t = res & (0x100 >> k) ? inv + (partial << 1) + bit + 1 : d + (partial << 1) + bit;
			partial = t & 7;
			if (t & 0x10) {
				res |= 1 << (7 - k);
			}
		}
		blendDivide[i] = (uint8_t) res;
	}
}

// ---- Textures

static uint8_t tmemByte(struct RDP* r, uint32_t addr) {
	return r->tmem[addr & 0xFFF];
}

static inline __attribute__((always_inline)) struct Color tlutColor(struct RDP* r, int index, bool ia) {
	uint32_t a = 0x800 + index * 8;
	uint16_t v = (uint16_t) (tmemByte(r, a) << 8 | tmemByte(r, a + 1));
	if (ia) {
		return (struct Color) { v >> 8, v >> 8, v >> 8, v & 255 };
	}
	int cr = (v >> 11) & 31, cg = (v >> 6) & 31, cb = (v >> 1) & 31;
	return (struct Color) { cr << 3 | cr >> 2, cg << 3 | cg >> 2, cb << 3 | cb >> 2, v & 1 ? 255 : 0 };
}

// What a tile's texels are, from its size and format and the TLUT modes: each kind has its own
// sampler, so the fetches inside one don't test them again.
enum TexelKind {
	TK_I4, TK_IA4, TK_CI4, TK_CI4_IA, TK_I8, TK_IA8, TK_CI8, TK_CI8_IA,
	TK_YUV16, TK_IA16, TK_CI16, TK_CI16_IA, TK_RGBA16, TK_RGBA32,
};

static enum TexelKind texelKind(const struct RDP* r, const struct Tile* t) {
	bool tlut = (r->otherModes >> 47) & 1, ia = (r->otherModes >> 46) & 1;
	switch (t->size) {
	case 0:
		return t->format == 2 || tlut ? (ia ? TK_CI4_IA : TK_CI4) : t->format == 3 ? TK_IA4 : TK_I4;
	case 1:
		return t->format == 2 || tlut ? (ia ? TK_CI8_IA : TK_CI8) : t->format == 3 ? TK_IA8 : TK_I8;
	case 2:
		return t->format == 1 ? TK_YUV16 : t->format == 3 ? TK_IA16 : tlut ? (ia ? TK_CI16_IA : TK_CI16) : TK_RGBA16;
	}
	return TK_RGBA32;
}

static inline __attribute__((always_inline)) struct Color texel(struct RDP* r, const struct Tile* t, int s, int tt, enum TexelKind kind) {
	uint32_t base = t->tmem * 8 + tt * t->line * 8;
	uint32_t swap = tt & 1 ? 4 : 0;
	switch (kind) {
	case TK_I4: case TK_IA4: case TK_CI4: case TK_CI4_IA: {
		uint32_t a = (base + (s >> 1)) ^ swap;
		int v = s & 1 ? tmemByte(r, a) & 15 : tmemByte(r, a) >> 4;
		if (kind == TK_CI4 || kind == TK_CI4_IA) {
			return tlutColor(r, t->palette << 4 | v, kind == TK_CI4_IA);
		}
		if (kind == TK_IA4) {
			int i = (v >> 1) << 5 | (v >> 1) << 2 | (v >> 2);
			return (struct Color) { i, i, i, v & 1 ? 255 : 0 };
		}
		int i = v * 17;
		return (struct Color) { i, i, i, i };
	}
	case TK_I8: case TK_IA8: case TK_CI8: case TK_CI8_IA: {
		int v = tmemByte(r, (base + s) ^ swap);
		if (kind == TK_CI8 || kind == TK_CI8_IA) {
			return tlutColor(r, v, kind == TK_CI8_IA);
		}
		if (kind == TK_IA8) {
			return (struct Color) { (v >> 4) * 17, (v >> 4) * 17, (v >> 4) * 17, (v & 15) * 17 };
		}
		return (struct Color) { v, v, v, v };
	}
	case TK_YUV16: {
		// Y in the upper half of TMEM, U and V for each pair of texels in the lower half; U and V
		// come out signed (9 bits).
		uint32_t a = ((t->tmem * 8 + tt * t->line * 8) & 0x7FF) + s;
		int y = tmemByte(r, (a ^ swap) | 0x800);
		int u = tmemByte(r, ((a & ~1u) ^ swap) & 0x7FF), v = tmemByte(r, ((a | 1) ^ swap) & 0x7FF);
		return (struct Color) { u - 128, v - 128, y, y };
	}
	case TK_IA16: case TK_CI16: case TK_CI16_IA: case TK_RGBA16: {
		uint32_t a = (base + s * 2) ^ swap;
		uint16_t v = (uint16_t) (tmemByte(r, a) << 8 | tmemByte(r, a + 1));
		if (kind == TK_IA16) {
			return (struct Color) { v >> 8, v >> 8, v >> 8, v & 255 };
		}
		if (kind != TK_RGBA16) {
			return tlutColor(r, v >> 8, kind == TK_CI16_IA);
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

// Texture coordinates and filtering follow paraLLEl-RDP's texture.h (MIT): a 10.5 coordinate
// clamped to 16 bits, shifted, offset by the tile's start and, with clamping, held at its last whole
// texel; masked and mirrored per texel.
static inline __attribute__((always_inline)) int texClamp(int c, bool clamp, int lo, int hi, int shift) {
	c = c < -0x8000 ? -0x8000 : c > 0x7FFF ? 0x7FFF : c;
	c = shift < 11 ? c >> shift : (int32_t) ((uint32_t) c << (32 - shift)) >> 16;
	if (!clamp) {
		return c - (lo << 3);
	}
	if ((c >> 3) >= hi) {
		return (((hi >> 2) - (lo >> 2)) & 0x3FF) << 5;
	}
	c -= lo << 3;
	return c < 0 ? 0 : c;
}

static inline __attribute__((always_inline)) int texMask(int c, int mask, bool mirror) {
	if (mask) {
		int m = 1 << (mask > 10 ? 10 : mask);
		if (mirror && (c & m)) {
			c ^= m - 1;
		}
		c &= m - 1;
	}
	return c;
}

static inline __attribute__((always_inline)) struct Color sampleAs(struct RDP* r, const struct Tile* tile, int s, int t, bool bilerp,
                                                                    enum TexelKind kind) {
	uint64_t m = r->otherModes;
	bool quad = (m >> 45) & 1, tlut = (m >> 47) & 1, midMode = (m >> 40) & 1;
	int cs = texClamp(s, (tile->cmS & 2) || !tile->maskS, tile->sl, tile->sh, tile->shiftS);
	int ct = texClamp(t, (tile->cmT & 2) || !tile->maskT, tile->tl, tile->th, tile->shiftT);
	int fs = quad || tlut ? cs & 31 : 0, ft = quad || tlut ? ct & 31 : 0;
	int sum = fs + ft;
	cs >>= 5, ct >>= 5;
	int s0 = texMask(cs, tile->maskS, tile->cmS & 1), s1 = texMask(cs + 1, tile->maskS, tile->cmS & 1);
	int t0 = texMask(ct, tile->maskT, tile->cmT & 1), t1 = texMask(ct + 1, tile->maskT, tile->cmT & 1);
	int tdiff = t1 - t0 < -255 ? -255 : t1 - t0;
	t1 = (t0 & 0xFF) + tdiff, t0 &= 0xFF;
	bool mid = midMode && bilerp && fs == 16 && ft == 16;
	if (mid) {
		sum = 0;
	}
	int bs = sum >= 32 ? s1 : s0, bt = sum >= 32 ? t1 : t0;
	if (tlut && !quad) {
		bs = s0, bt = t0, s1 = s0, t1 = t0;
	}
	struct Color base = texel(r, tile, bs, bt, kind);
	if (mid) {
		i4 sum4 = lanes(base) + lanes(texel(r, tile, s1, t0, kind)) + lanes(texel(r, tile, s0, t1, kind)) + lanes(texel(r, tile, s1, t1, kind));
		return color((sum4 + 2) >> 2);
	}
	if (!bilerp || !(quad || tlut)) {
		return base;
	}
	i4 b = lanes(base), t10 = lanes(texel(r, tile, s1, t0, kind)), t01 = lanes(texel(r, tile, s0, t1, kind));
	int fx = sum >= 32 ? 32 - ft : fs, fy = sum >= 32 ? 32 - fs : ft;
	return color((((t10 - b) * fx + (t01 - b) * fy + 0x10) >> 5) + b);
}

static struct Color sample(struct RDP* r, int tileIndex, int s, int t, bool filter) {
	struct Tile* tile = &r->tile[tileIndex & 7];
	bool bilerp = filter && ((r->otherModes >> 52) & 3) < 2;
	switch (texelKind(r, tile)) {
#define SAMPLE_AS(kind) case kind: return sampleAs(r, tile, s, t, bilerp, kind);
	SAMPLE_AS(TK_I4) SAMPLE_AS(TK_IA4) SAMPLE_AS(TK_CI4) SAMPLE_AS(TK_CI4_IA) SAMPLE_AS(TK_I8) SAMPLE_AS(TK_IA8)
	SAMPLE_AS(TK_CI8) SAMPLE_AS(TK_CI8_IA) SAMPLE_AS(TK_YUV16) SAMPLE_AS(TK_IA16) SAMPLE_AS(TK_CI16) SAMPLE_AS(TK_CI16_IA)
	SAMPLE_AS(TK_RGBA16)
#undef SAMPLE_AS
	default:
		return sampleAs(r, tile, s, t, bilerp, TK_RGBA32);
	}
}

static struct Color convert(const struct RDP* r, struct Color c);

// One kind of texel at a time: the compiler makes a loop for each, without the kind's tests inside.
static inline __attribute__((always_inline)) void tileImageKind(struct RDP* r, const struct Tile* t, int w, int h, bool converted, uint32_t* out, enum TexelKind kind) {
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			struct Color c = texel(r, t, x, y, kind);
			if (converted) {
				c = convert(r, c);
			}
			out[y * w + x] = (uint32_t) clamp8(c.r) | (uint32_t) clamp8(c.g) << 8 | (uint32_t) clamp8(c.b) << 16 | (uint32_t) clamp8(c.a) << 24;
		}
	}
}

void rdpTileImage(struct RDP* r, int tileIndex, int w, int h, bool converted, uint32_t* out) {
	const struct Tile* t = &r->tile[tileIndex & 7];
	switch (texelKind(r, t)) {
	case TK_I4: tileImageKind(r, t, w, h, converted, out, TK_I4); break;
	case TK_IA4: tileImageKind(r, t, w, h, converted, out, TK_IA4); break;
	case TK_CI4: tileImageKind(r, t, w, h, converted, out, TK_CI4); break;
	case TK_CI4_IA: tileImageKind(r, t, w, h, converted, out, TK_CI4_IA); break;
	case TK_I8: tileImageKind(r, t, w, h, converted, out, TK_I8); break;
	case TK_IA8: tileImageKind(r, t, w, h, converted, out, TK_IA8); break;
	case TK_CI8: tileImageKind(r, t, w, h, converted, out, TK_CI8); break;
	case TK_CI8_IA: tileImageKind(r, t, w, h, converted, out, TK_CI8_IA); break;
	case TK_YUV16: tileImageKind(r, t, w, h, converted, out, TK_YUV16); break;
	case TK_IA16: tileImageKind(r, t, w, h, converted, out, TK_IA16); break;
	case TK_CI16: tileImageKind(r, t, w, h, converted, out, TK_CI16); break;
	case TK_CI16_IA: tileImageKind(r, t, w, h, converted, out, TK_CI16_IA); break;
	case TK_RGBA16: tileImageKind(r, t, w, h, converted, out, TK_RGBA16); break;
	case TK_RGBA32: tileImageKind(r, t, w, h, converted, out, TK_RGBA32); break;
	}
}

static int ext9(int v);

// The texture convert unit: YUV (U, V, Y in red, green, blue) to RGB with K0-K3.
static struct Color convert(const struct RDP* r, struct Color c) {
	int red = c.b + ((r->k0 * c.g + 0x80) >> 8);
	int green = c.b + ((r->k1 * c.r + r->k2 * c.g + 0x80) >> 8);
	int blue = c.b + ((r->k3 * c.r + 0x80) >> 8);
	return (struct Color) { ext9(red), ext9(green), ext9(blue), ext9(c.b) };
}

// Both texels of a pixel: each cycle filters (bilinear when the sample type says so) or converts
// from YUV; with convert-one the second cycle converts the first's texel.
static void textures(struct RDP* r, struct Pixel* p, int t1, int t2, int s, int t) {
	uint64_t m = r->otherModes;
	int cycleType = (m >> 52) & 3;
	bool filter0 = cycleType >= 2 || ((m >> 43) & 1), filter1 = (m >> 42) & 1, convertOne = (m >> 41) & 1;
	p->tex0 = sample(r, t1, s, t, filter0);
	struct Color first = p->tex0;
	if (!filter0) {
		p->tex0 = convert(r, p->tex0);
	}
	if (cycleType != 1) {
		p->tex1 = p->tex0;
	} else if (convertOne) {
		p->tex1 = convert(r, first);
	} else {
		p->tex1 = sample(r, t2, s, t, filter1);
		if (!filter1) {
			p->tex1 = convert(r, p->tex1);
		}
	}
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
			// Noise: a hash of the pixel and the primitive (the hardware's sequence isn't known), the
			// same however many renderers draw and after a save state.
			return (int) (pixelNoise(r, p->x, p->y, (uint32_t) channel) & 255);
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
		const int* combined = &r->pix.combined.r;
		const int* inputs[8] = { s->a.v, s->b.v, s->m.v, s->d.v, s->alpha[0], s->alpha[1], s->alpha[2], s->alpha[3] };
		s->readsCombined = false;
		for (int i = 0; i < 8; ++i) {
			s->readsCombined |= inputs[i] >= combined && inputs[i] <= combined + 3;
		}
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
	r->antialias = (m >> 3) & 1;
	r->imageRead = (m >> 6) & 1;
	r->colorOnCvg = (m >> 7) & 1;
	r->cvgDest = (m >> 8) & 3;
	r->cvgAlpha = (m >> 13) & 1;
	r->cvgTimesAlpha = (m >> 12) & 1;
	r->texConvert = r->cycleType < 2 && (!((m >> 43) & 1) || (r->cycleType == 1 && (!((m >> 42) & 1) || ((m >> 41) & 1))));
	r->forceBlend = (m >> 14) & 1;
	r->ditherRgb = (m >> 38) & 3, r->ditherAlpha = (m >> 36) & 3;
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
static struct Color combineInputs(const struct Stage* s, bool raw) {
	i4 a = input4(&s->a, s->alpha[0]), b = input4(&s->b, s->alpha[1]);
	i4 m = input4(&s->m, s->alpha[2]), d = input4(&s->d, s->alpha[3]);
	if (!raw) {
		a = ext9x4(a), b = ext9x4(b), d = ext9x4(d);
		m = (m << 23) >> 23;
	}
	return color((((a - b) * m + (d << 8) + 0x80) >> 8) & 0x1FF);
}

static struct Color combineCycle(struct RDP* r, struct Pixel* p, int cycle) {
	// Converted texels can be outside 0-255: then the 9-bit rules apply.
	return r->stage[cycle].noise ? combine(r, p, cycle) : combineInputs(&r->stage[cycle], r->stage[cycle].raw && !r->texConvert);
}

static struct Color blendColor(struct RDP* r, struct Color pixel, struct Color memory, struct Pixel* p, int cycle, bool divide) {
	const uint8_t* sel = r->blendSel[cycle];
	int sp = sel[0], sa = sel[1], sm = sel[2], sb = sel[3];
	const struct Color* choices[4] = { &pixel, &memory, &r->blend, &r->fog };
	struct Color P = *choices[sp], M = *choices[sm];
	int a = sa == 0 ? pixel.a : sa == 1 ? r->fog.a : sa == 2 ? p->shade.a : 0;
	int b = sb == 0 ? 255 - a : sb == 1 ? memory.a : sb == 2 ? 255 : 0;
	// 5-bit factors: (P * a + M * (b + 1)), over 32 (wrapping at 8 bits) with force blend or in the
	// first of two cycles, else through the divider by the factors' sum. With memory alpha the
	// factors are shifted by the depth slopes' difference.
	int fa = a >> 3, fb = b >> 3;
	if (sb == 1) {
		fa = (fa >> r->blShiftA) & 0x3C;
		fb = (fb >> r->blShiftB) | 3;
	}
	i4 sum = lanes(P) * fa + lanes(M) * (fb + 1);
	i4 out;
	if (divide) {
		int d = ((fa & ~3) + (fb & ~3) + 4) << 9;
		for (int i = 0; i < 3; ++i) {
			out[i] = blendDivide[d | ((sum[i] >> 2) & 0x7FF)];
		}
	} else {
		out = (sum >> 5) & 255;
	}
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

// The depth test (with the stored depth slope: coplanar and decal tests, and whether an
// antialiased edge blends), then the blender, and the coverage and depth written.
static void shadePixel(struct N64* n, struct RDP* r, int x, int y, struct Pixel* p) {
	p->x = x, p->y = y;
	int cycleType = r->cycleType;
	int z = r->zPrim ? r->primZ : p->z;
	z = z < 0 ? 0 : z > 0x3FFFF ? 0x3FFFF : z;
	// One-cycle mode runs the combiner's second-cycle equation.
	p->combined = r->lastCombined;
	struct Color c = combineCycle(r, p, cycleType == 1 ? 0 : 1);
	if (cycleType == 1) {
		// The second cycle sees the pipelined texels: texel 1 as texel 0 (and texel 0 as texel 1, as
		// paraLLEl-RDP has it for the next pixel's).
		struct Color t = p->tex0;
		p->tex0 = p->tex1, p->tex1 = t;
		p->combined = c;
		c = combineCycle(r, p, 1);
	}
	r->lastCombined = c;
	r->lastPos = r->seq << 24 | (uint64_t) y << 12 | (uint32_t) x;
	c = (struct Color) { clamp9(c.r), clamp9(c.g), clamp9(c.b), clamp9(c.a) };
	int rgbDither, alphaDither;
	ditherAt(r, x, y, &rgbDither, &alphaDither);
	// Alpha with 255 as 256; coverage times alpha scales the coverage, alpha as coverage replaces
	// alpha with that (else alpha is dithered). Two-cycle mode tests the first cycle's alpha.
	int reference = 0;
	if (cycleType == 1 && r->alphaCompare) {
		int a0 = clamp9(p->combined.a);
		a0 += (a0 + 1) >> 8;
		a0 = r->cvgAlpha ? (r->cvgTimesAlpha ? (a0 * p->cvg + 4) >> 3 : p->cvg << 5) : a0 + alphaDither;
		reference = a0 < 0 ? 0 : a0 > 255 ? 255 : a0;
	}
	int alpha = c.a + ((c.a + 1) >> 8);
	int cvg = p->cvg, scaled = cvg << 5;
	if (r->cvgTimesAlpha) {
		scaled = (alpha * cvg + 4) >> 3;
		cvg = scaled >> 5;
	}
	alpha = r->cvgAlpha ? scaled : alpha + alphaDither;
	c.a = alpha < 0 ? 0 : alpha > 255 ? 255 : alpha;
	if (cycleType != 1) {
		reference = c.a;
	}
	p->shade.a = p->shade.a + alphaDither > 255 ? 255 : p->shade.a + alphaDither;
	struct Color memory = readPixel(n, r, x, y);
	int memCvg = r->imageRead ? memory.a >> 5 : 7;
	if (!r->imageRead) {
		memory.a = 0xE0;
	}
	int dzpix = r->zPrim ? r->primDz : r->dzpix;
	int dzEnc = dzEncode(dzpix);
	bool overflow = (cvg + memCvg) & 8;
	bool blend;
	if (r->zAddr && r->zCompare) {
		uint16_t stored = zRead(n, r, x, y);
		int oz = zDecompress(stored);
		int rawDzMem = (stored & 3) << 2 | zHidden(n, r, x, y);
		int dzMem = 1 << rawDzMem;
		r->blShiftA = dzEnc - rawDzMem < 0 ? 0 : dzEnc - rawDzMem > 4 ? 4 : dzEnc - rawDzMem;
		r->blShiftB = rawDzMem - dzEnc < 0 ? 0 : rawDzMem - dzEnc > 4 ? 4 : rawDzMem - dzEnc;
		bool coplanar = false;
		int precision = (stored >> 13) & 15;
		if (precision < 3) {
			if (dzMem != 0x8000) {
				dzMem <<= 1;
				dzMem = dzMem < (16 >> precision) ? 16 >> precision : dzMem;
			} else {
				coplanar = true;
				dzMem = 0xFFFF;
			}
		}
		int dzNew = highestBit((dzpix | dzMem) & 0xFFFF);
		int dzShifted = dzNew << 3;
		bool farther = coplanar || z + dzShifted >= oz;
		bool infront = z < oz, nearer = coplanar || z - dzShifted <= oz, max = oz == 0x3FFFF;
		blend = r->forceBlend || (!overflow && r->antialias && farther);
		bool pass;
		switch (r->zMode) {
		case 0:
			pass = max || (overflow ? infront : nearer);
			break;
		case 1:
			if (!infront || !farther || !overflow) {
				pass = max || (overflow ? infront : nearer);
			} else {
				int enc = dzEncode(dzNew & 0xFFFF);
				cvg = ((((oz >> enc) - (z >> enc)) & 15) * cvg >> 3) & 15;
				pass = true;
			}
			break;
		case 2:
			pass = infront || max;
			break;
		default:
			pass = farther && nearer && !max;
		}
		if (!pass) {
			return;
		}
	} else {
		r->blShiftA = 0;
		r->blShiftB = dzEnc < 11 ? 4 : 15 - dzEnc;
		blend = r->forceBlend || (!overflow && r->antialias);
	}
	if (r->alphaCompare && reference < ((r->otherModes >> 1) & 1 ? (int) (pixelNoise(r, x, y, 6) & 255) : r->blend.a)) {
		return;
	}
	if (r->antialias ? !cvg : !p->cvbit) {
		return;
	}
	// The blend formula applies when blending, unless the pixel is opaque and the mode rejects
	// partial blends; without it a cycle passes its first input through. With colour on coverage
	// only a wrap of the coverage writes the colour. In 2-cycle mode the first cycle always blends
	// (fog), and its result is the second cycle's pixel.
	int last = cycleType == 1 ? 1 : 0;
	const uint8_t* sel = r->blendSel[last];
	bool partialReject = sel[1] == 0 && sel[3] == 0 && c.a >= 255;
	struct Color in = c;
	if (cycleType == 1) {
		in = blendColor(r, c, memory, p, 0, false);
	}
	struct Color out;
	if (r->colorOnCvg && !overflow) {
		const struct Color* choices[4] = { &in, &memory, &r->blend, &r->fog };
		out = *choices[sel[2]];
	} else if (!blend || partialReject) {
		out = blendInput(r, in, memory, last);
	} else {
		out = blendColor(r, in, memory, p, last, !r->forceBlend);
	}
	// The image keeps the coverage (the alpha bit of 16-bit pixels with two hidden bits, the top 3
	// alpha bits of 32-bit ones), as cvg_dest says.
	if (r->ditherRgb != 3) {
		int* o = &out.r;
		for (int ch = 0; ch < 3; ++ch) {
			if (((rgbDither >> (ch * 3)) & 7) < (o[ch] & 7)) {
				o[ch] = o[ch] > 247 ? 255 : (o[ch] & 0xF8) + 8;
			}
		}
	}
	int kept = r->cvgDest == 0 ? (blend ? cvg + memCvg : cvg - 1) : r->cvgDest == 1 ? (cvg + memCvg) & 7 : r->cvgDest == 2 ? 7 : memCvg;
	out.a = (kept & 8 ? 7 : kept & 7) << 5;
	writePixel(n, r, x, y, out);
	if (r->zAddr && r->zUpdate) {
		zWrite(n, r, x, y, (uint16_t) (zCompress(z) | dzEnc >> 2), dzEnc & 3);
	}
}

// ---- Primitives

static void fillRect(struct N64* n, struct RDP* r, int x0, int y0, int x1, int y1) {
	int bytes = r->colorSize == 3 ? 4 : 2;
	// 32 pixels of the fill as a row from x0 stores them, and their hidden bits.
	uint8_t pattern[128], hidden[32];
	for (int k = 0; k < 32; ++k) {
		if (bytes == 4) {
			put32(pattern + k * 4, r->fillColor);
		} else {
			uint16_t v = __builtin_bswap16((uint16_t) ((x0 + k) & 1 ? r->fillColor : r->fillColor >> 16));
			memcpy(pattern + k * 2, &v, 2);
			hidden[k] = ((x0 + k) & 1 ? r->fillColor : r->fillColor >> 16) & 1 ? 3 : 0;
		}
	}
	for (int y = y0; y <= y1; ++y) {
		if (!ownsRow(r, y)) {
			continue;
		}
		// A row that stays inside RDRAM, at an address the pixel size divides: big-endian stores.
		uint32_t first = r->colorAddr + (y * r->colorWidth + x0) * bytes;
		uint32_t last = r->colorAddr + (y * r->colorWidth + x1) * bytes + bytes - 1;
		if (r->colorSize >= 2 && x1 >= x0 && !(r->colorAddr & (bytes - 1)) && first <= last && last < RDRAM_SIZE) {
			uint8_t* row = &n->rdram[first];
			int count = x1 - x0 + 1;
			for (int k = 0; k < count; k += 32) {
				int run = count - k < 32 ? count - k : 32;
				memcpy(row + k * bytes, pattern, (size_t) run * bytes);
				if (bytes == 2) {
					memcpy(&n->hidden[(first >> 1) + (uint32_t) k], hidden, (size_t) run);
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
				n->hidden[a >> 1] = v & 1 ? 3 : 0;
			}
		}
	}
}

// The scissor's right edge takes in its own column in fill and copy modes (libdragon's rdpq, from
// hardware).
static void clipRect(struct RDP* r, int* x0, int* y0, int* x1, int* y1) {
	int const right = (r->scissorX1 >> 2) - (((r->otherModes >> 52) & 3) >= 2 ? 0 : 1);
	if (*x0 < r->scissorX0 >> 2) {
		*x0 = r->scissorX0 >> 2;
	}
	if (*y0 < r->scissorY0 >> 2) {
		*y0 = r->scissorY0 >> 2;
	}
	if (*x1 > right) {
		*x1 = right;
	}
	if (*y1 > (r->scissorY1 >> 2) - 1) {
		*y1 = (r->scissorY1 >> 2) - 1;
	}
}

static int t0Coord(int t0, int d, int dtdy) {
	return ((t0 << 5) + d * dtdy) >> 5;
}

// Texture coordinates for the texture unit: S and T over W when perspective is on.

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

// A texture or fill rectangle's first pixel, and its pixels after the scissor: the right and bottom
// edges are in only in fill and copy modes.
static void rectBounds(struct RDP* r, uint64_t w0, int* cx0, int* cy0, int* x0, int* y0, int* x1, int* y1) {
	int xl = (w0 >> 44) & 0xFFF, yl = (w0 >> 32) & 0xFFF;
	int xh = (w0 >> 12) & 0xFFF, yh = w0 & 0xFFF;
	int cycleType = (r->otherModes >> 52) & 3;
	*cx0 = *x0 = xh >> 2;
	*cy0 = *y0 = yh >> 2;
	*x1 = cycleType >= 2 ? xl >> 2 : ((xl + 3) >> 2) - 1;
	*y1 = cycleType >= 2 ? yl >> 2 : ((yl + 3) >> 2) - 1;
	clipRect(r, x0, y0, x1, y1);
}

void rdpRectBounds(struct RDP* r, uint64_t w0, int* cx0, int* cy0, int* x0, int* y0, int* x1, int* y1) {
	rectBounds(r, w0, cx0, cy0, x0, y0, x1, y1);
}

static void textureRect(struct N64* n, struct RDP* r, uint64_t w0, uint64_t w1, bool flip) {
	int tile = (w0 >> 24) & 7;
	int s0 = (int16_t) (w1 >> 48), t0 = (int16_t) (w1 >> 32);
	int dsdx = (int16_t) (w1 >> 16), dtdy = (int16_t) w1;
	int cycleType = (r->otherModes >> 52) & 3;
	if (cycleType == 2) {
		dsdx /= 4;
	}
	int cx0, cy0, x0, y0, x1, y1;
	rectBounds(r, w0, &cx0, &cy0, &x0, &y0, &x1, &y1);
	r->maxLevel = 0;
	bool doLod = cycleType < 2 && (((r->otherModes >> 48) & 1) || lodFracUsed(r));
	for (int y = y0; y <= y1; ++y) {
		if (!ownsRow(r, y)) {
			continue;
		}
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
			textures(r, p, t1, t2, s, t);
			if (cycleType == 2) {
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
			p->cvg = 8;
			p->cvbit = true;
			shadePixel(n, r, x, y, p);
		}
	}
}

static int32_t fix(uint64_t w, int shift) {
	return (int32_t) (uint32_t) (w >> shift);
}

// A triangle's edges: the y of its bottom, middle and top in quarter lines, and x and its slope
// along the low, high (major) and middle edges as the command has them (s15.16), and its direction.
struct Edges {
	int yl, ym, yh;
	int64_t xl, dxl, xh, dxh, xm, dxm;
	bool flip;
};

static struct Edges edges(const uint64_t* w) {
	return (struct Edges) {
		.yl = ((int32_t) ((w[0] >> 32) & 0x3FFF) << 18) >> 18,
		.ym = ((int32_t) ((w[0] >> 16) & 0x3FFF) << 18) >> 18,
		.yh = ((int32_t) (w[0] & 0x3FFF) << 18) >> 18,
		.xl = fix(w[1], 32), .dxl = fix(w[1], 0),
		.xh = fix(w[2], 32), .dxh = fix(w[2], 0),
		.xm = fix(w[3], 32), .dxm = fix(w[3], 0),
		.flip = (w[0] >> 55) & 1,
	};
}

// The depth slope a triangle's pixels carry (dzpix): |dz/dx| + |dz/dy| in whole steps, normalized.
static int triangleDz(const uint64_t* w, int op) {
	if (!(op & 1)) {
		return normalizeDz(0);
	}
	int idx = 4 + (op & 4 ? 8 : 0) + (op & 2 ? 8 : 0);
	int dx = (int) ((fix(w[idx], 0) >> 16) & 0xFFFF), dy = (int) ((fix(w[idx + 1], 0) >> 16) & 0xFFFF);
	int sum = (dy & 0x8000 ? ~dy & 0x7FFF : dy) + (dx & 0x8000 ? ~dx & 0x7FFF : dx);
	return normalizeDz(sum & 0xFFFF) & 0xFFFF;
}

// The first and past-the-last scanline with a sub-scanline inside the triangle and the scissor.
static void triangleRows(const struct RDP* r, const struct Edges* t, int* first, int* end) {
	int top = t->yh < r->scissorY0 ? r->scissorY0 : t->yh;
	int bottom = t->yl > r->scissorY1 ? r->scissorY1 : t->yl;
	*first = top >> 2;
	*end = ((bottom - 1) >> 2) + 1;
	if (*end < *first) {
		*end = *first;
	}
}

// Spans, coverage and attribute stepping follow paraLLEl-RDP (Copyright (c) 2020 Themaister, MIT
// licence: span_setup.comp, coverage.h, interpolation.h, perspective.h), itself checked against
// Angrylion-Plus.

static int32_t sext(uint32_t v, int bits) {
	return (int32_t) (v << (32 - bits)) >> (32 - bits);
}

// A triangle on scanline y: each of its four sub-scanlines' edges in eighths of a pixel inside the
// scissor (0xFFFF and 0 where it has none), and the pixels they reach.
struct Span {
	int x0, x1;
	int32_t left[4], right[4];
};

// An edge's x snapped to eighths of a pixel, the bits below kept as one.
static int32_t snapX(int32_t x) {
	return (x >> 12) | ((x & 0xFFF) != 0);
}

static bool span(const struct RDP* r, const struct Edges* t, int y, struct Span* s) {
	int32_t xh0 = sext((uint32_t) t->xh, 28) >> 1, xm0 = sext((uint32_t) t->xm, 28) >> 1, xl0 = sext((uint32_t) t->xl, 28) >> 1;
	int32_t dxh = sext((uint32_t) t->dxh >> 2, 28) >> 1, dxm = sext((uint32_t) t->dxm >> 2, 28) >> 1, dxl = sext((uint32_t) t->dxl >> 2, 28) >> 1;
	int ylo = t->yh > r->scissorY0 ? t->yh : r->scissorY0, yhi = t->yl < r->scissorY1 ? t->yl : r->scissorY1;
	int32_t loScissor = r->scissorX0 << 1, hiScissor = r->scissorX1 << 1;
	bool allInvalid = true, allOver = true, allUnder = true;
	int32_t start = INT32_MAX, end = INT32_MIN;
	for (int k = 0; k < 4; ++k) {
		int q = y * 4 + k;
		uint32_t dy = (uint32_t) (q - (t->yh & ~3));
		int32_t xh = sext((uint32_t) xh0 + dy * (uint32_t) dxh, 27);
		int32_t xl = q < t->ym ? sext((uint32_t) xm0 + dy * (uint32_t) dxm, 27) : sext((uint32_t) xl0 + (uint32_t) (q - t->ym) * (uint32_t) dxl, 27);
		int32_t left = t->flip ? snapX(xh) : snapX(xl), right = t->flip ? snapX(xl) : snapX(xh);
		bool invalid = (left >> 1) > (right >> 1);
		allOver &= (left < right ? left : right) >= hiScissor;
		allUnder &= (left > right ? left : right) < loScissor;
		left = left < loScissor ? loScissor : left > hiScissor ? hiScissor : left;
		right = right < loScissor ? loScissor : right > hiScissor ? hiScissor : right;
		if (invalid || q < ylo || q >= yhi) {
			left = 0xFFFF, right = 0;
		} else {
			allInvalid = false;
		}
		s->left[k] = left, s->right[k] = right;
		start = left < start ? left : start;
		end = right > end ? right : end;
	}
	s->x0 = start >> 3, s->x1 = end >> 3;
	return !allInvalid && !allOver && !allUnder;
}

// The samples of pixel x a span covers: two to a sub-scanline, bit 0 the first, at eighths 0 and 4
// of the pixel on even sub-scanlines and 2 and 6 on odd ones.
static int coverageMask(const struct Span* s, int x) {
	int mask = 0;
	for (int k = 0; k < 4; ++k) {
		for (int j = 0; j < 2; ++j) {
			int32_t e = x * 8 + (k & 1) * 2 + j * 4;
			if (e >= s->left[k] && e < s->right[k]) {
				mask |= 1 << (k * 2 + j);
			}
		}
	}
	return mask;
}

// The first covered sample's offset from the pixel's corner, in quarter pixels (for attributes taken
// there).
static void sampleOffset(int mask, int* xoff, int* yoff) {
	int first = __builtin_ctz((unsigned) mask);
	*yoff = first >> 1;
	*xoff = ((first & 1) << 1) + (*yoff & 1);
}

// Depth, 19 bits clamped to 18 the same way.
static int clampZ(int z) {
	z = sext((uint32_t) (z - (1 << 17)), 19) + (1 << 17);
	return z < 0 ? 0 : z > 0x3FFFF ? 0x3FFFF : z;
}

static const int16_t perspectiveTable[64][2] = {
	{ 0x4000, -252 * 4 }, { 0x3f04, -244 * 4 }, { 0x3e10, -238 * 4 }, { 0x3d22, -230 * 4 }, { 0x3c3c, -223 * 4 }, { 0x3b5d, -218 * 4 },
	{ 0x3a83, -210 * 4 }, { 0x39b1, -205 * 4 }, { 0x38e4, -200 * 4 }, { 0x381c, -194 * 4 }, { 0x375a, -189 * 4 }, { 0x369d, -184 * 4 },
	{ 0x35e5, -179 * 4 }, { 0x3532, -175 * 4 }, { 0x3483, -170 * 4 }, { 0x33d9, -166 * 4 }, { 0x3333, -162 * 4 }, { 0x3291, -157 * 4 },
	{ 0x31f4, -155 * 4 }, { 0x3159, -150 * 4 }, { 0x30c3, -147 * 4 }, { 0x3030, -143 * 4 }, { 0x2fa1, -140 * 4 }, { 0x2f15, -137 * 4 },
	{ 0x2e8c, -134 * 4 }, { 0x2e06, -131 * 4 }, { 0x2d83, -128 * 4 }, { 0x2d03, -125 * 4 }, { 0x2c86, -123 * 4 }, { 0x2c0b, -120 * 4 },
	{ 0x2b93, -117 * 4 }, { 0x2b1e, -115 * 4 }, { 0x2aab, -113 * 4 }, { 0x2a3a, -110 * 4 }, { 0x29cc, -108 * 4 }, { 0x2960, -106 * 4 },
	{ 0x28f6, -104 * 4 }, { 0x288e, -102 * 4 }, { 0x2828, -100 * 4 }, { 0x27c4, -98 * 4 },  { 0x2762, -96 * 4 },  { 0x2702, -94 * 4 },
	{ 0x26a4, -92 * 4 },  { 0x2648, -91 * 4 },  { 0x25ed, -89 * 4 },  { 0x2594, -87 * 4 },  { 0x253d, -86 * 4 },  { 0x24e7, -85 * 4 },
	{ 0x2492, -83 * 4 },  { 0x243f, -81 * 4 },  { 0x23ee, -80 * 4 },  { 0x239e, -79 * 4 },  { 0x234f, -77 * 4 },  { 0x2302, -76 * 4 },
	{ 0x22b6, -74 * 4 },  { 0x226c, -74 * 4 },  { 0x2222, -72 * 4 },  { 0x21da, -71 * 4 },  { 0x2193, -70 * 4 },  { 0x214d, -69 * 4 },
	{ 0x2108, -67 * 4 },  { 0x20c5, -67 * 4 },  { 0x2082, -65 * 4 },  { 0x2041, -65 * 4 },
};

// Texture coordinates from s, t and w (each the top 16 bits): with perspective, s and t times a
// table's reciprocal of w (the RDP's divide), saturating; 17 bits signed.
static inline __attribute__((always_inline)) void texCoords(int32_t sS, int32_t sT, int32_t sW, bool persp, int* s, int* t) {
	int32_t st[2] = { sS >> 16, sT >> 16 };
	if (persp) {
		int32_t w = sW >> 16;
		bool carry = w <= 0;
		w &= 0x7FFF;
		int msb = w ? 31 - __builtin_clz((unsigned) w) : -1;
		int shift = 14 - msb < 14 ? 14 - msb : 14;
		int norm = (w << shift) & 0x3FFF;
		int rcp = ((perspectiveTable[norm >> 8][1] * (norm & 0xFF)) >> 10) + perspectiveTable[norm >> 8][0];
		int32_t mask = ((1 << 30) - 1) & -((1 << 29) >> shift);
		for (int i = 0; i < 2; ++i) {
			int32_t prod = (int32_t) ((uint32_t) st[i] * (uint32_t) rcp);
			int32_t outside = prod & mask;
			int32_t v = shift != 14 ? prod >> (13 - shift) : (int32_t) ((uint32_t) prod << 1);
			if (outside != mask && outside != 0) {
				int32_t shifted = shift != 14 ? prod >> (13 - shift) : prod;
				v = (shifted & (1 << 29)) == 0 ? 0x7FFF : -0x8000;
			}
			st[i] = carry ? 0x7FFF : v;
		}
	}
	*s = st[0] < -0x10000 ? -0x10000 : st[0] > 0xFFFF ? 0xFFFF : st[0];
	*t = st[1] < -0x10000 ? -0x10000 : st[1] > 0xFFFF ? 0xFFFF : st[1];
}

// The attributes' 32-bit values: shade r g b a, then s t w, then z; at the top, along x, along the
// major edge and along y.
struct Attributes {
	int32_t v[8], dx[8], de[8], dy[8];
};

static struct Attributes attributes(const uint64_t* w, int op) {
	struct Attributes a;
	memset(&a, 0, sizeof(a));
	int idx = 4;
	for (int part = 0; part < 2; ++part) {
		bool on = part == 0 ? op & 4 : op & 2;
		if (!on) {
			continue;
		}
		for (int i = 0; i < (part == 0 ? 4 : 3); ++i) {
			int sh = 48 - i * 16, k = part * 4 + i;
			a.v[k] = (int32_t) ((uint32_t) ((w[idx] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 2] >> sh) & 0xFFFF));
			a.dx[k] = (int32_t) ((uint32_t) ((w[idx + 1] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 3] >> sh) & 0xFFFF));
			a.de[k] = (int32_t) ((uint32_t) ((w[idx + 4] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 6] >> sh) & 0xFFFF));
			a.dy[k] = (int32_t) ((uint32_t) ((w[idx + 5] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 7] >> sh) & 0xFFFF));
		}
		idx += 8;
	}
	if (op & 1) {
		a.v[7] = fix(w[idx], 32), a.dx[7] = fix(w[idx], 0), a.de[7] = fix(w[idx + 1], 32), a.dy[7] = fix(w[idx + 1], 0);
	}
	return a;
}

// The attributes on scanline y at pixel *baseX: stepped down the major edge a line at a time, then
// back along x by the edge's fraction of a pixel (latched on the last sub-scanline when the major
// edge leans away from the triangle).
static void rowAttributes(const struct Edges* t, const struct Attributes* a, int y, bool skipFrac, int32_t* at, int* baseX) {
	int32_t xh0 = sext((uint32_t) t->xh, 28) >> 1, dxh = sext((uint32_t) t->dxh >> 2, 28) >> 1;
	bool doOffset = t->flip == (((uint32_t) t->dxh >> 31) != 0);
	uint32_t dy = (uint32_t) (y - (t->yh >> 2));
	int32_t xh = (int32_t) ((uint32_t) xh0 + dy * ((uint32_t) dxh << 2));
	if (doOffset) {
		xh = (int32_t) ((uint32_t) xh + 3u * (uint32_t) dxh);
	}
	*baseX = xh >> 15;
	int32_t xfrac = skipFrac ? 0 : (xh >> 7) & 0xFF;
	for (int i = 0; i < 8; ++i) {
		int32_t diff = 0;
		if (doOffset) {
			int32_t deh = a->de[i] & ~0x1FF, dyh = a->dy[i] & ~0x1FF;
			diff = deh - (deh >> 2) - dyh + (dyh >> 2);
		}
		int32_t v = (int32_t) ((uint32_t) a->v[i] + dy * (uint32_t) a->de[i]);
		at[i] = (int32_t) (((uint32_t) (v & ~0x1FF) + (uint32_t) diff - (uint32_t) xfrac * (uint32_t) ((a->dx[i] >> 8) & ~1)) & ~0x3FFu);
	}
}

static void triangle(struct N64* n, struct RDP* r, const uint64_t* w, int op) {
	bool shade = op & 4, tex = op & 2, zbuf = op & 1;
	int tile = (w[0] >> 48) & 7;
	struct Edges e = edges(w);
	struct Attributes a = attributes(w, op);
	r->maxLevel = (w[0] >> 51) & 7;
	r->dzpix = triangleDz(w, op);
	bool doLod = tex && (((r->otherModes >> 48) & 1) || lodFracUsed(r));
	bool persp = (r->otherModes >> 51) & 1;
	int32_t dir = e.flip ? 1 : -1;
	int first, end;
	triangleRows(r, &e, &first, &end);
	for (int y = first; y < end; ++y) {
		struct Span sp;
		if (!ownsRow(r, y) || !span(r, &e, y, &sp)) {
			continue;
		}
		int32_t row[8];
		int baseX;
		rowAttributes(&e, &a, y, r->cycleType == 2, row, &baseX);
		// Pixels whose samples (eighths 0 to 6) are inside all four sub-scanlines are covered whole.
		int32_t inLeft = 0, inRight = INT32_MAX;
		for (int k = 0; k < 4; ++k) {
			inLeft = sp.left[k] > inLeft ? sp.left[k] : inLeft;
			inRight = sp.right[k] < inRight ? sp.right[k] : inRight;
		}
		for (int x = sp.x0; x <= sp.x1; ++x) {
			int mask = x * 8 >= inLeft && x * 8 + 6 < inRight ? 0xFF : coverageMask(&sp, x);
			// A pixel is drawn when any sample is covered with antialiasing, else when the first is.
			if (!mask || (!r->antialias && !(mask & 1))) {
				continue;
			}
			struct Pixel* p = &r->pix;
			*p = (struct Pixel) { 0 };
			p->cvg = __builtin_popcount((unsigned) mask);
			p->cvbit = mask & 1;
			uint32_t dx = (uint32_t) (x - baseX);
			int xoff, yoff;
			sampleOffset(mask, &xoff, &yoff);
			if (shade) {
				int c[4];
				for (int i = 0; i < 4; ++i) {
					int32_t v = (int32_t) ((uint32_t) row[i] + (uint32_t) (a.dx[i] & ~0x1F) * dx);
					int16_t snapped = (int16_t) (v >> 14);
					snapped = (int16_t) (snapped << 2);
					snapped = (int16_t) (snapped + (int16_t) (xoff * (int16_t) (a.dx[i] >> 14)) + (int16_t) (yoff * (int16_t) (a.dy[i] >> 14)));
					c[i] = clamp9(snapped >> 4);
				}
				p->shade = (struct Color) { c[0], c[1], c[2], c[3] };
			}
			if (zbuf) {
				int32_t z = (int32_t) ((uint32_t) row[7] + (uint32_t) a.dx[7] * dx);
				int32_t snapped = z >> 10;
				snapped = (int32_t) ((uint32_t) snapped << 2) + xoff * (a.dx[7] >> 10) + yoff * (a.dy[7] >> 10);
				p->z = clampZ(snapped >> 5);
			}
			if (tex) {
				int32_t stw[3], stwX[3], stwY[3];
				for (int i = 0; i < 3; ++i) {
					int32_t step = a.dx[4 + i] & ~0x1F;
					stw[i] = (int32_t) ((uint32_t) row[4 + i] + (uint32_t) step * dx);
					stwX[i] = (int32_t) ((uint32_t) stw[i] + (uint32_t) (dir * step));
					stwY[i] = (int32_t) ((uint32_t) stw[i] + (uint32_t) (a.dy[4 + i] & ~0x7FFF));
				}
				int s, t, t1 = tile, t2 = tile + 1;
				texCoords(stw[0], stw[1], stw[2], persp, &s, &t);
				if (doLod) {
					int sx, tx, sy, ty;
					texCoords(stwX[0], stwX[1], stwX[2], persp, &sx, &tx);
					texCoords(stwY[0], stwY[1], stwY[2], persp, &sy, &ty);
					p->lodFrac = levelOfDetail(r, tile, s, t, sx, tx, sy, ty, &t1, &t2);
				}
				textures(r, p, t1, t2, s, t);
			}
			shadePixel(n, r, x, y, p);
		}
	}
}

// ---- Texture loads

static void tmemPut(struct N64* n, struct RDP* r, uint32_t to, uint32_t from) {
	r->tmem[to] = n->rdram[from];
	if (r->tmemSource) {
		r->tmemSource[to] = from;
	}
}

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
		// Plain texels a TMEM word at a time (a whole word, the halves swapped on odd lines).
		bool plain = r->texSize < 3 && !(r->texSize == 2 && t->format == 1);
		uint32_t words = (bytes + 7) & ~7u;
		if (plain && !r->tmemSource && bytes <= 0x1000 && t->tmem * 8u + words <= 0x1000 && (src & (RDRAM_SIZE - 1)) + words <= RDRAM_SIZE) {
			const uint8_t* from = &n->rdram[src & (RDRAM_SIZE - 1)];
			uint8_t* to = &r->tmem[t->tmem * 8];
			for (uint32_t i = 0; i < words; i += 8, lineT += dxt) {
				if ((lineT >> 11) & 1) {
					memcpy(to + i, from + i + 4, 4);
					memcpy(to + i + 4, from + i, 4);
				} else {
					memcpy(to + i, from + i, 8);
				}
			}
			bytes = 0;
		}
		for (uint32_t i = 0; i < bytes; i += 8) {
			bool odd = (lineT >> 11) & 1;
			if (plain && !r->tmemSource && ((src + i) & (RDRAM_SIZE - 1)) + 8 <= RDRAM_SIZE) {
				const uint8_t* from = &n->rdram[(src + i) & (RDRAM_SIZE - 1)];
				uint8_t* to = &r->tmem[(t->tmem * 8 + i) & 0xFF8];
				if (odd) {
					memcpy(to, from + 4, 4);
					memcpy(to + 4, from, 4);
				} else {
					memcpy(to, from, 8);
				}
				lineT += dxt;
				continue;
			}
			for (int b = 0; b < 8; ++b) {
				uint32_t dst = (t->tmem * 8 + i + b) ^ (odd ? 4 : 0);
				if (r->texSize == 2 && t->format == 1) {
					uint32_t texel = (i + b) / 2, part = (i + b) & 3;
					uint32_t d = (t->tmem * 8 + (texel & ~1u) + (part >= 2 ? 1 : 0)) ^ (odd ? 4 : 0);
					tmemPut(n, r, (d & 0x7FF) | (part & 1 ? 0x800 : 0), (src + i + b) & (RDRAM_SIZE - 1));
				} else if (r->texSize == 3) {
					uint32_t texelIndex = (i + b) / 4, part = (i + b) & 3;
					uint32_t d = (t->tmem * 8 + texelIndex * 2 + (part & 1)) ^ (odd ? 4 : 0);
					tmemPut(n, r, (d + (part >= 2 ? 0x800 : 0)) & 0xFFF, (src + i + b) & (RDRAM_SIZE - 1));
				} else {
					tmemPut(n, r, dst & 0xFFF, (src + i + b) & (RDRAM_SIZE - 1));
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
		if (r->texSize == 3) {
			for (int s = s0; s <= s1; ++s) {
				uint32_t src = (row + s * 4) & (RDRAM_SIZE - 4);
				uint32_t d = (line + (s - s0) * 2) ^ swap;
				tmemPut(n, r, d & 0x7FF, src);
				tmemPut(n, r, (d + 1) & 0x7FF, src + 1);
				tmemPut(n, r, 0x800 | (d & 0x7FF), src + 2);
				tmemPut(n, r, 0x800 | ((d + 1) & 0x7FF), src + 3);
			}
		} else if (r->texSize == 2 && t->format == 1) {
			// YUV: U Y0 V Y1 in RDRAM; Y to the upper half, U and V to the lower.
			for (int s = s0; s <= s1; ++s) {
				uint32_t d = (line + (s - s0)) ^ swap;
				uint32_t src = (row + (s & ~1) * 2) & (RDRAM_SIZE - 4);
				tmemPut(n, r, d & 0x7FF, src + (s & 1 ? 2 : 0));
				tmemPut(n, r, 0x800 | (d & 0x7FF), src + (s & 1 ? 3 : 1));
			}
		} else if (s1 >= s0) {
			// The row's bytes in order (4-bit texels two to a byte), odd rows' word halves swapped.
			uint32_t from = r->texSize == 0 ? row + (uint32_t) s0 / 2 : row + (uint32_t) (s0 * bpp);
			uint32_t len = r->texSize == 0 ? (uint32_t) (s1 - s0) / 2 + 1 : (uint32_t) ((s1 - s0 + 1) * bpp);
			if (!r->tmemSource && line + len <= 0x1000 && (from & (RDRAM_SIZE - 1)) + len <= RDRAM_SIZE) {
				const uint8_t* src = &n->rdram[from & (RDRAM_SIZE - 1)];
				uint8_t* dst = &r->tmem[line];
				uint32_t k = 0;
				if (!swap) {
					memcpy(dst, src, len);
					k = len;
				}
				for (; k + 8 <= len; k += 8) {
					memcpy(dst + k, src + k + 4, 4);
					memcpy(dst + k + 4, src + k, 4);
				}
				for (; k < len; ++k) {
					dst[k ^ swap] = src[k];
				}
			} else {
				for (uint32_t k = 0; k < len; ++k) {
					tmemPut(n, r, ((line + k) ^ swap) & 0xFFF, (from + k) & (RDRAM_SIZE - 1));
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
			tmemPut(n, r, (dst + copy * 2) & 0xFFF, src);
			tmemPut(n, r, (dst + copy * 2 + 1) & 0xFFF, src + 1);
		}
	}
}

// ---- Command processing

static int commandWords(int op) {
	// The HLE's triangle for the GL renderer: a header and three vertices.
	if (op == 0x01) {
		return GLR_TRIANGLE_WORDS;
	}
	if (op >= 0x08 && op <= 0x0F) {
		return 4 + (op & 4 ? 8 : 0) + (op & 2 ? 8 : 0) + (op & 1 ? 2 : 0);
	}
	return op == 0x24 || op == 0x25 ? 2 : 1;
}

// The RDP's time for a primitive, from its shape alone: one or two clocks a pixel by cycle type,
// four pixels a clock to fill or copy, and a fixed setup per triangle and texture rectangle.
static uint64_t primitiveClocks(struct RDP* r, const uint64_t* w, int op) {
	int perPixel = r->cycleType == 1 ? 2 : 1;
	if (op >= 0x08 && op <= 0x0F) {
		struct Edges e = edges(w);
		uint64_t pixels = 0;
		int first, end;
		triangleRows(r, &e, &first, &end);
		for (int y = first; y < end; ++y) {
			struct Span sp;
			if (span(r, &e, y, &sp) && sp.x1 >= sp.x0) {
				pixels += (uint64_t) (sp.x1 - sp.x0 + 1);
			}
		}
		return pixels * perPixel + 20;
	}
	int x0, y0, x1, y1;
	if (op == 0x24 || op == 0x25) {
		int cx0, cy0;
		rectBounds(r, w[0], &cx0, &cy0, &x0, &y0, &x1, &y1);
	} else if (op == 0x36) {
		int cx0, cy0;
		rectBounds(r, w[0], &cx0, &cy0, &x0, &y0, &x1, &y1);
	} else {
		return 0;
	}
	uint64_t pixels = x1 >= x0 && y1 >= y0 ? (uint64_t) (x1 - x0 + 1) * (y1 - y0 + 1) : 0;
	if (op == 0x36) {
		return r->cycleType == 3 ? pixels / (r->colorSize == 3 ? 2 : 4) : pixels * perPixel;
	}
	return (r->cycleType == 2 ? pixels / 4 : pixels * perPixel) + 10;
}

// Every command but the full sync, drawing the rows r owns.
static void execute(struct N64* n, struct RDP* r, const uint64_t* w, int op) {
	uint64_t c = w[0];
	if (op == 0x30 || op == 0x32 || op == 0x33 || op == 0x34 || op == 0x35) {
		++r->tmemGen;
	}
	if ((op >= 0x08 && op <= 0x0F) || op == 0x24 || op == 0x25 || op == 0x36) {
		++r->seq;
	} else {
		++r->stateGen;
	}
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
		break;
	case 0x24:
	case 0x25:
		textureRect(n, r, w[0], w[1], op == 0x25);
		break;
	case 0x2C:
		r->k4 = (c >> 9) & 0x1FF;
		r->k5 = c & 0x1FF;
		r->k0 = (int) (((int64_t) (c << 10) >> 55) * 2 + 1);
		r->k1 = (int) (((int64_t) (c << 19) >> 55) * 2 + 1);
		r->k2 = (int) (((int64_t) (c << 28) >> 55) * 2 + 1);
		r->k3 = (int) (((int64_t) (c << 37) >> 55) * 2 + 1);
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
		int x0, y0, x1, y1, cx0, cy0;
		int cycleType = (r->otherModes >> 52) & 3;
		rectBounds(r, c, &cx0, &cy0, &x0, &y0, &x1, &y1);
		if (cycleType == 3) {
			fillRect(n, r, x0, y0, x1, y1);
		} else {
			for (int y = y0; y <= y1; ++y) {
				if (!ownsRow(r, y)) {
					continue;
				}
				for (int x = x0; x <= x1; ++x) {
					struct Pixel* p = &r->pix;
					*p = (struct Pixel) { .cvg = 8, .cvbit = true };
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

// ---- Drawing on other threads
//
// With N64_RDP_THREADS, renderer threads draw the commands, each the scanlines y % count == index,
// from a copy of the command stream, each with its own RDP state. The main thread keeps the RDP's
// time, and flags the RDRAM pages the commands draw to or read: anything else touching those
// pages first waits for the renderers (rdpWait), so the emulation sees what drawing in place would
// have left. In the stream, a barrier stops every renderer until all reach it: before a texture
// load from pages drawn to since the last one, before drawing to pages loaded from, and before the
// color or depth image changes. A primitive that needs the pixel drawn before it (the combined
// value in the first cycle, noise) is drawn by renderer 0 alone, between barriers.

enum { RING_SIZE = 1 << 18, RENDERERS_MAX = 8, ENTRY_ALONE = 1 << 16, ENTRY_BARRIER = 1 << 17 };

// RDP clocks of commands to gather before waking the renderers (waits wake them at once).
enum { WAKE_CLOCKS = 4096 };

struct Renderer {
	pthread_t thread;
	struct Renderers* all;
	struct RDP* rdp;
	int index;
	// How far through the stream it has drawn.
	_Atomic uint64_t done;
};

struct Renderers {
	struct N64* n;
	int count;
	struct Renderer renderer[RENDERERS_MAX];
	pthread_mutex_t lock;
	pthread_cond_t work, idle;
	pthread_barrier_t barrier;
	bool quit;
	// The stream: a header word (command, words, ENTRY_*) and the command's words, and the RDP
	// clocks of the commands put in since the renderers were last woken.
	uint64_t ring[RING_SIZE];
	_Atomic uint64_t written;
	uint64_t unannounced;
	// Pages flagged for the RDP, and pages drawn to and loaded from since the last barrier.
	uint16_t marked[RDRAM_SIZE >> 12];
	int markedCount;
	uint8_t since[RDRAM_SIZE >> 12];
	// For each page, where in the stream the last command drawing to it ends.
	uint64_t drawnUntil[RDRAM_SIZE >> 12];
	uint16_t sinceList[RDRAM_SIZE >> 12];
	int sinceCount;
};

enum { SINCE_DRAWN = 1, SINCE_LOADED = 2 };

static uint64_t oldestDone(struct Renderers* all) {
	uint64_t done = UINT64_MAX;
	for (int i = 0; i < all->count; ++i) {
		uint64_t d = atomic_load_explicit(&all->renderer[i].done, memory_order_acquire);
		done = d < done ? d : done;
	}
	return done;
}

static void wake(struct Renderers* all) {
	pthread_mutex_lock(&all->lock);
	pthread_cond_broadcast(&all->work);
	pthread_mutex_unlock(&all->lock);
	all->unannounced = 0;
}

// Wakes the renderers and waits until all have drawn up to pos.
static void waitFor(struct Renderers* all, uint64_t pos) {
	all->unannounced = 0;
	pthread_mutex_lock(&all->lock);
	pthread_cond_broadcast(&all->work);
	while (oldestDone(all) < pos) {
		pthread_cond_wait(&all->idle, &all->lock);
	}
	pthread_mutex_unlock(&all->lock);
}

// Draws a primitive that depends on the pixel before it: renderer 0 draws every row, after taking
// the latest combiner output from the others; the rest only follow its state changes.
static void drawAlone(struct Renderer* me, const uint64_t* w, int op) {
	struct Renderers* all = me->all;
	struct RDP* r = me->rdp;
	int rows = r->rowCount;
	pthread_barrier_wait(&all->barrier);
	if (me->index == 0) {
		for (int i = 1; i < all->count; ++i) {
			struct RDP* other = all->renderer[i].rdp;
			if (other->lastPos > r->lastPos) {
				r->lastPos = other->lastPos;
				r->lastCombined = other->lastCombined;
			}
		}
		r->rowCount = 1;
	} else {
		r->rowCount = 0;
	}
	execute(all->n, r, w, op);
	r->rowCount = rows;
	pthread_barrier_wait(&all->barrier);
}

static void* rendererMain(void* arg) {
	struct Renderer* me = arg;
	struct Renderers* all = me->all;
	uint64_t pos = 0;
	for (;;) {
		if (pos == atomic_load_explicit(&all->written, memory_order_acquire)) {
			pthread_mutex_lock(&all->lock);
			atomic_store_explicit(&me->done, pos, memory_order_release);
			pthread_cond_broadcast(&all->idle);
			while (!all->quit && pos == atomic_load_explicit(&all->written, memory_order_acquire)) {
				pthread_cond_wait(&all->work, &all->lock);
			}
			bool quit = all->quit && pos == atomic_load_explicit(&all->written, memory_order_acquire);
			pthread_mutex_unlock(&all->lock);
			if (quit) {
				return NULL;
			}
			continue;
		}
		uint64_t header = all->ring[pos % RING_SIZE];
		int words = (header >> 8) & 0xFF, op = header & 0x3F;
		uint64_t w[22];
		for (int i = 0; i < words; ++i) {
			w[i] = all->ring[(pos + 1 + i) % RING_SIZE];
		}
		if (header & ENTRY_BARRIER) {
			pthread_barrier_wait(&all->barrier);
		} else if (header & ENTRY_ALONE) {
			drawAlone(me, w, op);
		} else {
			execute(all->n, me->rdp, w, op);
		}
		pos += 1 + words;
		atomic_store_explicit(&me->done, pos, memory_order_release);
	}
}

static struct Renderers* startRenderers(struct N64* n, struct RDP* r) {
	struct Renderers* all = calloc(1, sizeof(*all));
	if (!all) {
		return NULL;
	}
	all->n = n;
	all->count = n->rdpThreads < RENDERERS_MAX ? n->rdpThreads : RENDERERS_MAX;
	pthread_mutex_init(&all->lock, NULL);
	pthread_cond_init(&all->work, NULL);
	pthread_cond_init(&all->idle, NULL);
	pthread_barrier_init(&all->barrier, NULL, all->count);
	for (int i = 0; i < all->count; ++i) {
		struct Renderer* me = &all->renderer[i];
		me->all = all;
		me->index = i;
		me->rdp = malloc(sizeof(struct RDP));
		memcpy(me->rdp, r, sizeof(struct RDP));
		me->rdp->renderers = NULL;
		me->rdp->rowIndex = i;
		me->rdp->rowCount = all->count;
		setupCombiner(me->rdp);
		pthread_create(&me->thread, NULL, rendererMain, me);
	}
	return all;
}

static void stopRenderers(struct Renderers* all) {
	pthread_mutex_lock(&all->lock);
	all->quit = true;
	pthread_cond_broadcast(&all->work);
	pthread_mutex_unlock(&all->lock);
	for (int i = 0; i < all->count; ++i) {
		pthread_join(all->renderer[i].thread, NULL);
		free(all->renderer[i].rdp);
	}
	pthread_barrier_destroy(&all->barrier);
	pthread_cond_destroy(&all->idle);
	pthread_cond_destroy(&all->work);
	pthread_mutex_destroy(&all->lock);
	free(all);
}

void rdpWait(struct N64* n) {
	if (!n->rdpPending) {
		return;
	}
	if (n->glr) {
		glrSync(n);
		return;
	}
	if (n->rdp->gpu) {
		gpuSync(n, n->rdp);
		return;
	}
	struct Renderers* all = n->rdp->renderers;
	waitFor(all, atomic_load_explicit(&all->written, memory_order_relaxed));
	for (int i = 0; i < all->markedCount; ++i) {
		n->pageFlags[all->marked[i]] &= ~PAGE_RDP;
	}
	all->markedCount = 0;
	for (int i = 0; i < all->sinceCount; ++i) {
		all->since[all->sinceList[i]] = 0;
	}
	all->sinceCount = 0;
	n->rdpPending = false;
}

static void put(struct Renderers* all, uint64_t header, const uint64_t* w, int words) {
	uint64_t pos = atomic_load_explicit(&all->written, memory_order_relaxed);
	if (pos + 1 + words - oldestDone(all) > RING_SIZE) {
		waitFor(all, pos + 1 + words - RING_SIZE);
	}
	all->ring[pos % RING_SIZE] = header | (uint64_t) words << 8;
	for (int i = 0; i < words; ++i) {
		all->ring[(pos + 1 + i) % RING_SIZE] = w[i];
	}
	atomic_store_explicit(&all->written, pos + 1 + words, memory_order_release);
}

static void barrier(struct Renderers* all) {
	put(all, ENTRY_BARRIER, NULL, 0);
	for (int i = 0; i < all->sinceCount; ++i) {
		all->since[all->sinceList[i]] = 0;
	}
	all->sinceCount = 0;
}

// The pages of RDRAM [addr, addr + len), wrapping.
#define FOR_PAGES(addr, len, p) \
	for (uint32_t p = ((addr) & (RDRAM_SIZE - 1)) >> 12, p##End = ((addr) + (len) - 1) & (RDRAM_SIZE - 1), p##Done = 0; !p##Done; \
	     p##Done = p == p##End >> 12, p = (p + 1) & ((RDRAM_SIZE >> 12) - 1))

// Whether any page in the range was drawn to (or loaded from) since the last barrier.
static bool touchedSince(struct Renderers* all, uint32_t addr, uint32_t len, int what) {
	FOR_PAGES(addr, len, p) {
		if (all->since[p] & what) {
			return true;
		}
	}
	return false;
}

static void mark(struct N64* n, struct Renderers* all, uint32_t addr, uint32_t len, uint8_t flag, int what, uint64_t until) {
	FOR_PAGES(addr, len, p) {
		if (!(n->pageFlags[p] & PAGE_RDP)) {
			all->marked[all->markedCount++] = (uint16_t) p;
		}
		n->pageFlags[p] |= flag;
		if (flag == PAGE_RDP_WRITES) {
			all->drawnUntil[p] = until;
		}
		if (!all->since[p]) {
			all->sinceList[all->sinceCount++] = (uint16_t) p;
		}
		all->since[p] |= what;
	}
	n->rdpPending = true;
}

// Waits until the RDP commands drawing to [addr, addr + len) are drawn, not the rest (the
// renderers keep going); the pages stay flagged for the next full wait.
// The CPU or a DMA is about to touch RDRAM [addr, addr + len) on a page the RDP draws to: with the
// GL renderer only the GPU's own rows matter, else everything waits.
void rdpWaitFor(struct N64* n, uint32_t addr, uint32_t len) {
	if (n->rdp && n->rdp->gpu && !n->glr) {
		if (!gpuWrite(n, n->rdp, addr, len)) {
			gpuSync(n, n->rdp);
			gpuWrite(n, n->rdp, addr, len);
		}
		return;
	}
	if (n->glr) {
		if (n->rdpPending && glrDrew(n, addr, len)) {
			glrSync(n);
		}
		return;
	}
	rdpWait(n);
}

// As rdpWaitFor, for a read: with the GPU drawing, only bytes it drew wait (a write waits for the
// whole page, which comes back from the GPU as one).
void rdpWaitRead(struct N64* n, uint32_t addr, uint32_t len) {
	if (n->rdp && n->rdp->gpu && !n->glr) {
		if (n->rdpPending && gpuReadNeeds(n->rdp->gpu, addr, len)) {
			gpuSyncRead(n, n->rdp, addr, len);
		}
		return;
	}
	rdpWaitFor(n, addr, len);
}

void rdpWaitDrawn(struct N64* n, uint32_t addr, uint32_t len) {
	if (!n->rdpPending) {
		return;
	}
	// The GL renderer gives back the pixels it drew there.
	if (n->glr) {
		glrSyncRange(n, addr, len);
		return;
	}
	struct Renderers* all = n->rdp->renderers;
	if (!all) {
		rdpWait(n);
		return;
	}
	uint64_t until = 0;
	FOR_PAGES(addr, len, p) {
		if ((n->pageFlags[p] & PAGE_RDP_WRITES) && all->drawnUntil[p] > until) {
			until = all->drawnUntil[p];
		}
	}
	if (until > oldestDone(all)) {
		waitFor(all, until);
	}
}

// The RDRAM a texture load reads, as loadTile and loadTlut address it (rounded up a little).
static void loadSource(struct RDP* r, const uint64_t* w, int op, uint32_t* addr, uint32_t* len) {
	uint64_t c = w[0];
	int sl = (c >> 44) & 0xFFF, tl = (c >> 32) & 0xFFF, sh = (c >> 12) & 0xFFF, th = c & 0xFFF;
	uint32_t bpp = r->texSize == 0 ? 0 : 1u << (r->texSize - 1);
	if (op == 0x30) {
		*addr = r->texAddr + (uint32_t) (sl >> 2) * 2;
		*len = (uint32_t) ((sh >> 2) - (sl >> 2) + 1) * 2;
	} else if (op == 0x33) {
		uint32_t count = (uint32_t) (sh - sl + 1);
		*addr = r->texAddr + (tl * r->texWidth + sl) * bpp;
		*len = (r->texSize == 0 ? (count + 1) / 2 : count * bpp) + 8;
	} else {
		uint32_t row = r->texWidth * (bpp ? bpp : 1);
		uint32_t t0 = (uint32_t) (tl >> 2), t1 = (uint32_t) (th >> 2);
		*addr = r->texAddr + t0 * row;
		*len = (t1 - t0) * row + ((uint32_t) (sh >> 2) + 1) * 4 + 8;
	}
	if (*len > RDRAM_SIZE) {
		*len = RDRAM_SIZE;
	}
}

// The scanlines a primitive may draw on, first to last.
static void primitiveRows(struct RDP* r, const uint64_t* w, int op, int* y0, int* y1) {
	int x0, x1;
	if (op >= 0x08 && op <= 0x0F) {
		struct Edges e = edges(w);
		int end;
		triangleRows(r, &e, y0, &end);
		*y1 = end - 1;
	} else if (op == 0x24 || op == 0x25) {
		int cx0, cy0;
		rectBounds(r, w[0], &cx0, &cy0, &x0, y0, &x1, y1);
	} else {
		int cx0, cy0;
		rectBounds(r, w[0], &cx0, &cy0, &x0, y0, &x1, y1);
	}
}

// Hands a command to the renderers, with the barriers and flags it needs.
static void submit(struct N64* n, struct RDP* r, const uint64_t* w, int op, int words) {
	struct Renderers* all = r->renderers;
	uint64_t header = (uint64_t) op;
	if ((op == 0x3E || op == 0x3F) && all->sinceCount) {
		barrier(all);
	} else if (op == 0x30 || op == 0x33 || op == 0x34) {
		uint32_t addr, len;
		loadSource(r, w, op, &addr, &len);
		if (touchedSince(all, addr, len, SINCE_DRAWN)) {
			barrier(all);
		}
		mark(n, all, addr, len, PAGE_RDP_READS, SINCE_LOADED, 0);
	} else if ((op >= 0x08 && op <= 0x0F) || op == 0x24 || op == 0x25 || op == 0x36) {
		int y0, y1;
		primitiveRows(r, w, op, &y0, &y1);
		uint32_t bpp = r->colorSize == 1 ? 1 : r->colorSize == 3 ? 4 : 2;
		uint32_t right = (uint32_t) ((r->scissorX1 >> 2) + 1);
		bool depth = r->zAddr && (r->zCompare || r->zUpdate);
		if (y1 >= y0) {
			uint32_t color = r->colorAddr + (uint32_t) y0 * r->colorWidth * bpp;
			uint32_t colorLen = (uint32_t) (y1 - y0) * r->colorWidth * bpp + right * bpp;
			uint32_t z = r->zAddr + (uint32_t) y0 * r->colorWidth * 2;
			uint32_t zLen = (uint32_t) (y1 - y0) * r->colorWidth * 2 + right * 2;
			if (touchedSince(all, color, colorLen, SINCE_LOADED) || (depth && touchedSince(all, z, zLen, SINCE_LOADED))) {
				barrier(all);
			}
			uint64_t until = atomic_load_explicit(&all->written, memory_order_relaxed) + 1 + words;
			mark(n, all, color, colorLen, PAGE_RDP_WRITES, SINCE_DRAWN, until);
			if (depth) {
				mark(n, all, z, zLen, PAGE_RDP_WRITES, SINCE_DRAWN, until);
			}
			// Rows that run into each other, or depth and color images that overlap row for
			// different rows, can't be split between renderers.
			bool overlap = depth && r->zAddr != r->colorAddr && z < color + colorLen && color < z + zLen;
			if (right - 1 > r->colorWidth || overlap) {
				header |= ENTRY_ALONE;
			}
		}
		bool shades = !(op == 0x36 && r->cycleType == 3) && !((op == 0x24 || op == 0x25) && r->cycleType == 2);
		const struct Stage* first = &r->stage[r->cycleType == 1 ? 0 : 1];
		if (shades && (first->readsCombined || usesNoise(r))) {
			header |= ENTRY_ALONE;
		}
	}
	put(all, header, w, words);
}

// ---- Drawing on the GPU
//
// With N64_RDP_GPU, a compute shader (rdp_gpu.glsl) draws the primitives. The main thread runs the
// other commands itself (texture loads included, into TMEM snapshots) and records each primitive
// with the state it draws with; triangles come with their spans and each row's starting attributes.
// A batch goes to the GPU at a full sync, before the color or depth image changes, or when
// full; each 16x16 tile of the picture draws its primitives in order. The pages the batch draws to
// are flagged like the renderer threads' (rdpWait brings them back). Primitives that need the pixel
// drawn before them are drawn here after everything before them is back.

#include "rdp_gpu_glsl.h"

enum {
	GPU_PRIM_SIZE = 25, GPU_TRI_SIZE = 35, GPU_TILE_FIELDS = 15, GPU_STATE_SIZE = 19 + 8 * GPU_TILE_FIELDS + 4,
	GPU_TILES = 64, GPU_PRIMS_MAX = 4000, GPU_LAST_SIZE = 1 << 20, GPU_BATCHES_MAX = 4096,
	// Tiles times primitives in one dispatch: a long dispatch trips the kernel's GPU hang check.
	GPU_ENTRIES_MAX = 1024,
	// TMEM bytes loaded from what the GPU drew, between full syncs.
	GPU_PATCHES_MAX = 1 << 20,
	GPU_KIND_TRIANGLE = 0, GPU_KIND_TEXRECT = 1, GPU_KIND_FILL = 2, GPU_KIND_RECT = 3,
	GPU_PAGE_BATCH = 1, GPU_PAGE_OWNED = 2, GPU_PAGE_MARKED = 4,
	GPU_VARIANTS = 48, GPU_RANGES = 256,
};

// What a batch's primitives use, for the shader variant that leaves out the rest (rdp_gpu.glsl's
// F_* defines, in this order; the color image's pixel size above them).
enum {
	F_TRI = 1 << 0, F_TEXRECT = 1 << 1, F_FILL = 1 << 2, F_RECT = 1 << 3, F_SHADE = 1 << 4, F_ZTRI = 1 << 5, F_TEX = 1 << 6,
	F_LOD = 1 << 7, F_PERSP = 1 << 8, F_TWO = 1 << 9, F_COPY = 1 << 10, F_BILERP = 1 << 11, F_CONV = 1 << 12, F_ZBUF = 1 << 13,
	F_TLUT = 1 << 14, F_FLAGS = 15,
};
static const char* const featureNames[F_FLAGS] = { "F_TRI", "F_TEXRECT", "F_FILL", "F_RECT", "F_SHADE", "F_ZTRI", "F_TEX", "F_LOD",
	                                                   "F_PERSP", "F_TWO", "F_COPY", "F_BILERP", "F_CONV", "F_ZBUF", "F_TLUT" };

struct Gpu {
	struct Gles* gl;
	int32_t prims[GPU_PRIMS_MAX * GPU_PRIM_SIZE];
	int primCount;
	int entries;
	int32_t* states;
	int stateCount, stateCap;
	int32_t* rows;
	size_t rowCount, rowCap;
	uint8_t* tmem;
	int tmemCount, tmemCap;
	bool stateDirty, tmemDirty;
	// A primitive in the batch reaches the image's last column or past it.
	bool edge;
	int32_t* upload;
	size_t uploadCap;
	uint8_t page[RDRAM_SIZE >> 12];
	uint16_t batchPages[RDRAM_SIZE >> 12], markedPages[RDRAM_SIZE >> 12];
	int batchPageCount, markedPageCount;
	// Each batch's part of the last-pixel buffer (3 words a tile), since the last collection.
	uint32_t lastUsed;
	uint32_t batchBase[GPU_BATCHES_MAX], batchTiles[GPU_BATCHES_MAX];
	int batchCount;
	// The bytes drawn since the last sync, for reads elsewhere on their pages to go ahead without
	// one; past GPU_RANGES, every read of a drawn page waits.
	uint32_t drawnFrom[GPU_RANGES], drawnTo[GPU_RANGES];
	int drawnCount;
	bool drawnAll;
	// Writes to pages it owns (over what it drew, after that drawing was submitted): copied to its
	// RDRAM once they are done (gpuUpdate), before anything later goes to the GPU.
	uint32_t updateFrom[GPU_RANGES], updateTo[GPU_RANGES];
	int updateCount;
	// Drawn bytes written over since the last sync and not drawn again: newest in RDRAM, so reads
	// of them go ahead.
	uint32_t overFrom[GPU_RANGES], overTo[GPU_RANGES];
	int overCount;
	// The batch's features, and the shader variants built so far (program 0 draws everything).
	int features;
	int variantKey[GPU_VARIANTS], variantProgram[GPU_VARIANTS], variantCount;
	// Loads of what the GPU drew: each such TMEM byte's value is gathered on the GPU into the patch
	// buffer when loaded (patch[d] its index, else -1), and put into the batch's TMEM copies (apply:
	// snapshot byte, index) before it draws. The CPU's TMEM has them after a full sync.
	uint32_t source[0x1000];
	int32_t patch[0x1000];
	int patched, gatherProgram, applyProgram;
	uint32_t patchUsed;
	uint32_t* apply;
	size_t applyCount, applyCap;
};

static const char gatherSource[] =
	"#version 310 es\n"
	"layout(local_size_x = 64) in;\n"
	"layout(std430, binding = 0) readonly buffer Rdram { uint rdram[]; };\n"
	"layout(std430, binding = 5) buffer Patch { uint gathered[]; };\n"
	"uniform int u[2];\n"
	"void main() {\n"
	"	int i = int(gl_GlobalInvocationID.x);\n"
	"	if (i < u[1]) {\n"
	"		uint a = gathered[u[0] + i];\n"
	"		gathered[u[0] + i] = (rdram[a >> 2] >> ((a & 3u) * 8u)) & 255u;\n"
	"	}\n"
	"}\n";

static const char applySource[] =
	"#version 310 es\n"
	"layout(local_size_x = 64) in;\n"
	"layout(std430, binding = 1) buffer Tmem { uint tmem[]; };\n"
	"layout(std430, binding = 2) readonly buffer Data { uint data[]; };\n"
	"layout(std430, binding = 5) readonly buffer Patch { uint gathered[]; };\n"
	"uniform int u[2];\n"
	"void main() {\n"
	"	int i = int(gl_GlobalInvocationID.x);\n"
	"	if (i < u[1]) {\n"
	"		uint at = data[u[0] + i * 2], s = (at & 3u) * 8u;\n"
	"		atomicAnd(tmem[at >> 2], ~(255u << s));\n"
	"		atomicOr(tmem[at >> 2], gathered[data[u[0] + i * 2 + 1]] << s);\n"
	"	}\n"
	"}\n";

static void* grow(void* p, size_t* cap, size_t need, size_t size) {
	if (need <= *cap) {
		return p;
	}
	size_t c = *cap ? *cap : 1024;
	while (c < need) {
		c *= 2;
	}
	void* q = realloc(p, c * size);
	if (!q) {
		abort();
	}
	*cap = c;
	return q;
}

static struct Gpu* gpuCreate(void) {
	struct Gpu* g = calloc(1, sizeof(*g));
	if (!g) {
		return NULL;
	}
	char error[2048] = "";
	g->gl = glesCreate(rdpGpuSource, error, sizeof(error));
	if (!g->gl) {
		fprintf(stderr, "GPU RDP unavailable (%s), drawing on the CPU\n", error);
		free(g);
		return NULL;
	}
	fprintf(stderr, "GPU RDP on %s\n", glesRenderer(g->gl));
	glesData(g->gl, 0, NULL, RDRAM_SIZE + RDRAM_SIZE / 2);
	glesData(g->gl, 3, NULL, GPU_LAST_SIZE * 4);
	glesData(g->gl, 5, NULL, GPU_PATCHES_MAX * 4);
	g->gatherProgram = glesAddProgram(g->gl, gatherSource, error, sizeof(error));
	g->applyProgram = g->gatherProgram < 0 ? -1 : glesAddProgram(g->gl, applySource, error, sizeof(error));
	if (g->applyProgram < 0) {
		fprintf(stderr, "GPU RDP loads of drawn texels unavailable (%s)\n", error);
	}
	memset(g->patch, 0xFF, sizeof(g->patch));
	g->stateDirty = g->tmemDirty = true;
	return g;
}

static void gpuDestroy(struct Gpu* g) {
	glesDestroy(g->gl);
	free(g->states);
	free(g->rows);
	free(g->tmem);
	free(g->upload);
	free(g->apply);
	free(g);
}

// The program for a batch's features: built the first time, the full one if that fails.
static int gpuVariant(struct Gpu* g, int features) {
	for (int i = 0; i < g->variantCount; ++i) {
		if (g->variantKey[i] == features) {
			return g->variantProgram[i];
		}
	}
	if (g->variantCount == GPU_VARIANTS) {
		return 0;
	}
	static char source[1 << 16];
	const char* body = strchr(rdpGpuSource, '\n') + 1;
	int at = snprintf(source, sizeof(source), "#version 310 es\n");
	for (int i = 0; i < F_FLAGS; ++i) {
		at += snprintf(source + at, sizeof(source) - (size_t) at, "#define %s %d\n", featureNames[i], (features >> i) & 1);
	}
	at += snprintf(source + at, sizeof(source) - (size_t) at, "#define F_CSIZE %d\n", features >> F_FLAGS);
	snprintf(source + at, sizeof(source) - (size_t) at, "%s", body);
	char error[2048] = "";
	int program = glesAddProgram(g->gl, source, error, sizeof(error));
	if (program < 0) {
		fprintf(stderr, "GPU RDP variant %x unavailable (%s)\n", features, error);
		program = 0;
	}
	g->variantKey[g->variantCount] = features;
	g->variantProgram[g->variantCount++] = program;
	return program;
}

// What a primitive uses, as gpuVariant's features (conservative: the shader's own tests decide).
static int gpuFeatures(struct RDP* r, int op) {
	uint64_t m = r->otherModes;
	int f = 0;
	bool tex;
	if (op >= 0x08 && op <= 0x0F) {
		f |= F_TRI | (op & 4 ? F_SHADE : 0) | (op & 1 ? F_ZTRI : 0) | ((op & 2) && ((m >> 51) & 1) ? F_PERSP : 0);
		tex = op & 2;
	} else if (op == 0x24 || op == 0x25) {
		f |= F_TEXRECT | (r->cycleType == 2 ? F_COPY : 0);
		tex = true;
	} else {
		f |= r->cycleType == 3 ? F_FILL : F_RECT;
		tex = false;
	}
	if (tex) {
		f |= F_TEX | (((m >> 48) & 1) || lodFracUsed(r) ? F_LOD : 0) | ((m >> 45) & 1 ? F_BILERP : 0) | ((m >> 47) & 1 ? F_TLUT : 0);
		f |= !((m >> 43) & 1) || !((m >> 42) & 1) || ((m >> 41) & 1) ? F_CONV : 0;
	}
	f |= r->cycleType == 1 ? F_TWO : 0;
	f |= r->zAddr && (((m >> 4) & 1) || ((m >> 5) & 1)) ? F_ZBUF : 0;
	return f;
}

static int32_t packColor(struct Color c) {
	return (int32_t) ((uint32_t) c.r << 24 | (uint32_t) c.g << 16 | (uint32_t) c.b << 8 | (uint32_t) c.a);
}

// The state a primitive draws with: a new record when anything changed since the last one.
static int gpuState(struct Gpu* g, struct RDP* r) {
	if (g->tmemDirty) {
		size_t cap = (size_t) g->tmemCap;
		g->tmem = grow(g->tmem, &cap, (size_t) (g->tmemCount + 1) * 0x1000, 1);
		g->tmemCap = (int) cap;
		memcpy(g->tmem + (size_t) g->tmemCount * 0x1000, r->tmem, 0x1000);
		for (int d = 0; g->patched && d < 0x1000; ++d) {
			if (g->patch[d] >= 0) {
				g->apply = grow(g->apply, &g->applyCap, g->applyCount * 2 + 2, sizeof(uint32_t));
				g->apply[g->applyCount * 2] = (uint32_t) g->tmemCount * 0x1000 + (uint32_t) d;
				g->apply[g->applyCount++ * 2 + 1] = (uint32_t) g->patch[d];
			}
		}
		++g->tmemCount;
		g->tmemDirty = false;
		g->stateDirty = true;
	}
	if (g->stateDirty) {
		size_t cap = (size_t) g->stateCap;
		g->states = grow(g->states, &cap, (size_t) (g->stateCount + 1) * GPU_STATE_SIZE, sizeof(int32_t));
		g->stateCap = (int) cap;
		int32_t* s = g->states + (size_t) g->stateCount * GPU_STATE_SIZE;
		s[0] = (int32_t) (uint32_t) r->otherModes;
		s[1] = (int32_t) (uint32_t) (r->otherModes >> 32);
		s[2] = (int32_t) (uint32_t) r->combine;
		s[3] = (int32_t) (uint32_t) (r->combine >> 32);
		s[4] = (int32_t) r->colorAddr;
		s[5] = (int32_t) r->colorWidth;
		s[6] = r->colorSize;
		s[7] = (int32_t) r->zAddr;
		s[8] = packColor(r->fog);
		s[9] = packColor(r->blend);
		s[10] = packColor(r->prim);
		s[11] = packColor(r->env);
		s[12] = r->primLod;
		s[13] = r->primZ;
		s[14] = r->minLevel;
		s[15] = r->k4;
		s[16] = r->k5;
		s[17] = (int32_t) r->fillColor;
		s[18] = g->tmemCount - 1;
		for (int i = 0; i < 8; ++i) {
			const struct Tile* t = &r->tile[i];
			int32_t* f = s + 19 + i * GPU_TILE_FIELDS;
			f[0] = t->format, f[1] = t->size, f[2] = t->line, f[3] = t->tmem, f[4] = t->palette;
			f[5] = t->cmS, f[6] = t->cmT, f[7] = t->maskS, f[8] = t->maskT, f[9] = t->shiftS, f[10] = t->shiftT;
			f[11] = t->sl, f[12] = t->tl, f[13] = t->sh, f[14] = t->th;
		}
		int32_t* k = s + 19 + 8 * GPU_TILE_FIELDS;
		k[0] = r->k0, k[1] = r->k1, k[2] = r->k2, k[3] = r->k3;
		++g->stateCount;
		g->stateDirty = false;
	}
	return (g->stateCount - 1) * GPU_STATE_SIZE;
}

// Adds [addr, addr + len) to a list of ranges, merged with one it touches (in two pieces when it
// wraps around RDRAM); false when the list is full.
static bool addRange(uint32_t* fromList, uint32_t* toList, int* count, uint32_t addr, uint32_t len) {
	uint32_t from = addr & (RDRAM_SIZE - 1), to = from + len;
	if (len >= RDRAM_SIZE) {
		return false;
	}
	if (to > RDRAM_SIZE) {
		return addRange(fromList, toList, count, from, RDRAM_SIZE - from) && addRange(fromList, toList, count, 0, to - RDRAM_SIZE);
	}
	for (int i = 0; i < *count; ++i) {
		if (from <= toList[i] && to >= fromList[i]) {
			fromList[i] = from < fromList[i] ? from : fromList[i];
			toList[i] = to > toList[i] ? to : toList[i];
			return true;
		}
	}
	if (*count == GPU_RANGES) {
		return false;
	}
	fromList[*count] = from;
	toList[(*count)++] = to;
	return true;
}

static bool inRanges(const uint32_t* fromList, const uint32_t* toList, int count, uint32_t addr, uint32_t len) {
	uint32_t from = addr & (RDRAM_SIZE - 1), to = from + len;
	if (len >= RDRAM_SIZE) {
		return true;
	}
	if (to > RDRAM_SIZE) {
		return inRanges(fromList, toList, count, from, RDRAM_SIZE - from) || inRanges(fromList, toList, count, 0, to - RDRAM_SIZE);
	}
	for (int i = 0; i < count; ++i) {
		if (from < toList[i] && to > fromList[i]) {
			return true;
		}
	}
	return false;
}

static void gpuMarkRange(struct Gpu* g, uint32_t addr, uint32_t len) {
	if (g->overCount && inRanges(g->overFrom, g->overTo, g->overCount, addr, len)) {
		g->overCount = 0;
	}
	if (!addRange(g->drawnFrom, g->drawnTo, &g->drawnCount, addr, len)) {
		g->drawnAll = true;
	}
}

// Whether the GPU may have drawn in [addr, addr + len) since the last sync.
static bool gpuDrew(struct Gpu* g, uint32_t addr, uint32_t len) {
	return g->drawnAll || inRanges(g->drawnFrom, g->drawnTo, g->drawnCount, addr, len);
}

// Whether a read of [addr, addr + len) needs what the GPU drew: not when all of it was written over
// since.
static bool gpuReadNeeds(struct Gpu* g, uint32_t addr, uint32_t len) {
	uint32_t from = addr & (RDRAM_SIZE - 1), to = from + len;
	for (int i = 0; i < g->overCount; ++i) {
		if (from >= g->overFrom[i] && to <= g->overTo[i]) {
			return false;
		}
	}
	return gpuDrew(g, addr, len);
}

static void gpuSubmit(struct N64* n, struct RDP* r);

// A write to RDRAM on pages the GPU owns goes ahead. Pages it has drawing to do on keep their copy:
// the written bytes follow once done, after that drawing (if over it) goes to the GPU. Other pages
// it lets go (loaded again when it needs them). False when it can't (a list is full): a sync, then.
static bool gpuWrite(struct N64* n, struct RDP* r, uint32_t addr, uint32_t len) {
	struct Gpu* g = r->gpu;
	bool marked = false;
	FOR_PAGES(addr, len, p) {
		marked |= (g->page[p] & GPU_PAGE_MARKED) != 0;
	}
	if (!marked) {
		FOR_PAGES(addr, len, p) {
			g->page[p] &= ~GPU_PAGE_OWNED;
			n->pageFlags[p] &= ~PAGE_RDP_READS;
		}
		return true;
	}
	bool over = n->rdpPending && gpuDrew(g, addr, len);
	if ((over && g->drawnAll) || g->updateCount == GPU_RANGES) {
		return false;
	}
	if (over) {
		gpuSubmit(n, r);
	}
	if (!addRange(g->updateFrom, g->updateTo, &g->updateCount, addr, len)) {
		return false;
	}
	if (over && !addRange(g->overFrom, g->overTo, &g->overCount, addr, len)) {
		g->overCount = 0;
	}
	return true;
}

// The writes since the last call, now done, into the GPU's RDRAM.
static void gpuUpdate(struct N64* n, struct Gpu* g) {
	for (int i = 0; i < g->updateCount; ++i) {
		glesSubData(g->gl, 0, g->updateFrom[i], &n->rdram[g->updateFrom[i]], g->updateTo[i] - g->updateFrom[i]);
	}
	g->updateCount = 0;
}

static void gpuMark(struct N64* n, struct Gpu* g, uint32_t addr, uint32_t len) {
	gpuMarkRange(g, addr, len);
	FOR_PAGES(addr, len, p) {
		if (!(g->page[p] & GPU_PAGE_MARKED)) {
			g->markedPages[g->markedPageCount++] = (uint16_t) p;
			g->page[p] |= GPU_PAGE_MARKED;
		}
		if (!(g->page[p] & GPU_PAGE_BATCH)) {
			g->batchPages[g->batchPageCount++] = (uint16_t) p;
			g->page[p] |= GPU_PAGE_BATCH;
		}
		n->pageFlags[p] |= PAGE_RDP_WRITES;
	}
	n->rdpPending = true;
}

// One edge's x at sub-scanline q (span's arithmetic), for a bounding box.
static int64_t edgeX(const struct Edges* e, int q, bool major) {
	int64_t dy = q - (e->yh & ~3);
	return major ? e->xh + e->dxh * dy / 4 : q < e->ym ? e->xm + e->dxm * dy / 4 : e->xl + e->dxl * (q - e->ym) / 4;
}

// triangle()'s setup, for the shader to work out spans and attributes from (GPU_TRI_SIZE words:
// edges, scissor, shade, texture and depth at the top), and a box around what it may draw. False if
// that is nothing.
static bool gpuTriangle(struct Gpu* g, struct RDP* r, const uint64_t* w, int op, int32_t* p) {
	bool shade = op & 4, tex = op & 2, zbuf = op & 1;
	struct Edges e = edges(w);
	int first, end;
	triangleRows(r, &e, &first, &end);
	if (end <= first) {
		return false;
	}
	// Each edge is a straight line between its sub-scanlines' ends (the minor one bends at ym).
	int qa = first * 4 > e.yh ? first * 4 : e.yh, qb = end * 4 - 1 < e.yl - 1 ? end * 4 - 1 : e.yl - 1;
	int64_t lo = INT64_MAX, hi = INT64_MIN;
	const int qs[4] = { qa, qb, e.ym - 1, e.ym };
	for (int i = 0; i < 4; ++i) {
		if (qs[i] < qa || qs[i] > qb) {
			continue;
		}
		for (int major = 0; major < 2; ++major) {
			int64_t x = edgeX(&e, qs[i], major);
			lo = x < lo ? x : lo;
			hi = x > hi ? x : hi;
		}
	}
	int64_t minX = lo >> 16, maxX = hi >> 16;
	minX = minX < r->scissorX0 >> 2 ? r->scissorX0 >> 2 : minX;
	maxX = maxX > (r->scissorX1 * 2 - 1) >> 3 ? (r->scissorX1 * 2 - 1) >> 3 : maxX;
	if (lo > hi || minX > maxX) {
		return false;
	}
	g->rows = grow(g->rows, &g->rowCap, g->rowCount + GPU_TRI_SIZE, sizeof(int32_t));
	int32_t* t = g->rows + g->rowCount;
	memset(t, 0, GPU_TRI_SIZE * sizeof(int32_t));
	t[0] = e.yl, t[1] = e.ym, t[2] = e.yh;
	t[3] = (int32_t) e.xl, t[4] = (int32_t) e.dxl, t[5] = (int32_t) e.xh, t[6] = (int32_t) e.dxh, t[7] = (int32_t) e.xm, t[8] = (int32_t) e.dxm;
	t[9] = r->scissorX0, t[10] = r->scissorY0, t[11] = r->scissorX1, t[12] = r->scissorY1;
	int idx = 4;
	if (shade) {
		for (int i = 0; i < 4; ++i) {
			int sh = 48 - i * 16;
			t[13 + i] = (int32_t) ((uint32_t) ((w[idx] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 2] >> sh) & 0xFFFF));
			p[11 + i] = (int32_t) ((uint32_t) ((w[idx + 1] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 3] >> sh) & 0xFFFF));
			t[17 + i] = (int32_t) ((uint32_t) ((w[idx + 4] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 6] >> sh) & 0xFFFF));
			t[29 + i] = (int32_t) ((uint32_t) ((w[idx + 5] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 7] >> sh) & 0xFFFF));
		}
		idx += 8;
	}
	if (tex) {
		for (int i = 0; i < 3; ++i) {
			int sh = 48 - i * 16;
			t[21 + i] = (int32_t) ((uint32_t) ((w[idx] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 2] >> sh) & 0xFFFF));
			p[15 + i] = (int32_t) ((uint32_t) ((w[idx + 1] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 3] >> sh) & 0xFFFF));
			t[24 + i] = (int32_t) ((uint32_t) ((w[idx + 4] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 6] >> sh) & 0xFFFF));
			p[18 + i] = (int32_t) ((uint32_t) ((w[idx + 5] >> sh) & 0xFFFF) << 16 | (uint32_t) ((w[idx + 7] >> sh) & 0xFFFF));
		}
		idx += 8;
	}
	if (zbuf) {
		t[27] = (int32_t) fix(w[idx], 32);
		p[21] = (int32_t) fix(w[idx], 0);
		t[28] = (int32_t) fix(w[idx + 1], 32);
		t[33] = (int32_t) fix(w[idx + 1], 0);
	}
	t[34] = e.flip;
	p[0] = GPU_KIND_TRIANGLE;
	p[2] = (int32_t) minX, p[3] = first, p[4] = (int32_t) maxX, p[5] = end - 1;
	p[6] = (w[0] >> 48) & 7;
	p[7] = (w[0] >> 51) & 7;
	p[8] = (int32_t) g->rowCount;
	p[10] = op & 7;
	g->rowCount += GPU_TRI_SIZE;
	return true;
}

// The tiles a primitive's box has on tile row ty; for a triangle, only those its edges reach on
// that row's sub-scanlines (a pixel wider each side).
static void rowTiles(const struct Gpu* g, const int32_t* p, int ty, int* tx0, int* tx1) {
	*tx0 = p[2] >> 4, *tx1 = p[4] >> 4;
	if (p[0] != GPU_KIND_TRIANGLE) {
		return;
	}
	const int32_t* t = g->rows + p[8];
	struct Edges e = { t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], false };
	int qa = p[3] * 4 > e.yh ? p[3] * 4 : e.yh, qb = p[5] * 4 + 3 < e.yl - 1 ? p[5] * 4 + 3 : e.yl - 1;
	qa = qa > ty * 64 ? qa : ty * 64;
	qb = qb < ty * 64 + 63 ? qb : ty * 64 + 63;
	int64_t lo = INT64_MAX, hi = INT64_MIN;
	const int qs[4] = { qa, qb, e.ym - 1, e.ym };
	for (int i = 0; i < 4; ++i) {
		if (qs[i] < qa || qs[i] > qb) {
			continue;
		}
		for (int major = 0; major < 2; ++major) {
			int64_t x = edgeX(&e, qs[i], major);
			lo = x < lo ? x : lo;
			hi = x > hi ? x : hi;
		}
	}
	if (lo > hi) {
		return;
	}
	int64_t a = ((lo >> 16) - 1) >> 4, b = ((hi >> 16) + 1) >> 4;
	*tx0 = a > *tx0 ? (int) a : *tx0;
	*tx1 = b < *tx1 ? (int) b : *tx1;
}

static void gpuCollectLast(struct Gpu* g, struct RDP* r) {
	if (!g->lastUsed) {
		return;
	}
	const uint32_t* last = glesMap(g->gl, 3, 0, g->lastUsed * 4);
	if (last) {
		for (int b = 0; b < g->batchCount; ++b) {
			uint32_t best = 0, at = 0;
			for (uint32_t t = 0; t < g->batchTiles[b]; ++t) {
				uint32_t i = g->batchBase[b] + t * 3;
				if (last[i] > best) {
					best = last[i];
					at = i;
				}
			}
			if (best) {
				uint32_t c = last[at + 1];
				r->lastCombined = (struct Color) { c & 0x1FF, (c >> 9) & 0x1FF, (c >> 18) & 0x1FF, last[at + 2] & 0x1FF };
			}
		}
		glesUnmap(g->gl, 3);
	}
	g->lastUsed = 0;
	g->batchCount = 0;
}

// Sends the batch to the GPU.
static void gpuSubmit(struct N64* n, struct RDP* r) {
	struct Gpu* g = r->gpu;
	if (!g->primCount) {
		return;
	}
	gpuUpdate(n, g);
	// The pages it draws to, unless the GPU's copy is already the newest.
	for (int i = 0; i < g->batchPageCount; ++i) {
		int p = g->batchPages[i];
		g->page[p] &= ~GPU_PAGE_BATCH;
		if (!(g->page[p] & GPU_PAGE_OWNED)) {
			int run = 1;
			while (i + run < g->batchPageCount && g->batchPages[i + run] == p + run && !(g->page[p + run] & GPU_PAGE_OWNED)) {
				g->page[p + run] &= ~GPU_PAGE_BATCH;
				++run;
			}
			glesSubData(g->gl, 0, (size_t) p << 12, &n->rdram[(size_t) p << 12], (size_t) run << 12);
			glesSubData(g->gl, 0, RDRAM_SIZE + ((size_t) p << 11), &n->hidden[(size_t) p << 11], (size_t) run << 11);
			for (int k = 0; k < run; ++k) {
				g->page[p + k] |= GPU_PAGE_OWNED;
			}
			i += run - 1;
		}
	}
	g->batchPageCount = 0;
	// The tiles each primitive's box covers, and the lists of primitives for each.
	int tx0 = GPU_TILES, ty0 = GPU_TILES, tx1 = -1, ty1 = -1;
	static int32_t counts[GPU_TILES * GPU_TILES];
	memset(counts, 0, sizeof(counts));
	size_t entries = 0;
	for (int i = 0; i < g->primCount; ++i) {
		const int32_t* p = g->prims + i * GPU_PRIM_SIZE;
		int ax = p[2] >> 4, ay = p[3] >> 4, bx = p[4] >> 4, by = p[5] >> 4;
		tx0 = ax < tx0 ? ax : tx0, ty0 = ay < ty0 ? ay : ty0;
		tx1 = bx > tx1 ? bx : tx1, ty1 = by > ty1 ? by : ty1;
		for (int ty = ay; ty <= by; ++ty) {
			int rx0, rx1;
			rowTiles(g, p, ty, &rx0, &rx1);
			for (int tx = rx0; tx <= rx1; ++tx) {
				++counts[ty * GPU_TILES + tx];
			}
			entries += (size_t) (rx1 >= rx0 ? rx1 - rx0 + 1 : 0);
		}
	}
	size_t statesAt = (size_t) g->primCount * GPU_PRIM_SIZE;
	size_t rowsAt = statesAt + (size_t) g->stateCount * GPU_STATE_SIZE;
	size_t binsAt = rowsAt + g->rowCount;
	size_t listsAt = binsAt + GPU_TILES * GPU_TILES * 2;
	size_t applyAt = listsAt + entries;
	size_t total = applyAt + g->applyCount * 2;
	g->upload = grow(g->upload, &g->uploadCap, total, sizeof(int32_t));
	int32_t* d = g->upload;
	memcpy(d, g->prims, statesAt * sizeof(int32_t));
	for (int i = 0; i < g->primCount; ++i) {
		int32_t* p = d + i * GPU_PRIM_SIZE;
		p[1] += (int32_t) statesAt;
		if (p[0] == GPU_KIND_TRIANGLE) {
			p[8] += (int32_t) rowsAt;
		}
	}
	memcpy(d + statesAt, g->states, (size_t) g->stateCount * GPU_STATE_SIZE * sizeof(int32_t));
	memcpy(d + rowsAt, g->rows, g->rowCount * sizeof(int32_t));
	int32_t* bins = d + binsAt;
	int32_t offset = 0;
	for (int t = 0; t < GPU_TILES * GPU_TILES; ++t) {
		bins[t * 2] = offset;
		bins[t * 2 + 1] = 0;
		offset += counts[t];
	}
	int32_t* lists = d + listsAt;
	for (int i = 0; i < g->primCount; ++i) {
		const int32_t* p = g->prims + i * GPU_PRIM_SIZE;
		for (int ty = p[3] >> 4; ty <= p[5] >> 4; ++ty) {
			int rx0, rx1;
			rowTiles(g, p, ty, &rx0, &rx1);
			for (int tx = rx0; tx <= rx1; ++tx) {
				int32_t* bin = bins + (ty * GPU_TILES + tx) * 2;
				lists[bin[0] + bin[1]++] = i;
			}
		}
	}
	unsigned across = (unsigned) (tx1 - tx0 + 1), down = (unsigned) (ty1 - ty0 + 1);
	if (g->lastUsed + across * down * 3 > GPU_LAST_SIZE || g->batchCount == GPU_BATCHES_MAX) {
		gpuCollectLast(g, r);
	}
	memcpy(d + applyAt, g->apply, g->applyCount * 2 * sizeof(int32_t));
	glesData(g->gl, 2, d, total * sizeof(int32_t));
	glesData(g->gl, 1, g->tmem, (size_t) g->tmemCount * 0x1000);
	if (g->applyCount) {
		int a[2] = { (int) applyAt, (int) g->applyCount };
		glesSelectProgram(g->gl, g->applyProgram);
		glesUniforms(g->gl, a, 2);
		glesDispatch(g->gl, (unsigned) (g->applyCount + 63) / 64, 1);
		g->applyCount = 0;
	}
	// 16-bit pixels go back a word at a time when no word holds pixels of two tiles.
	bool words = !g->edge && r->colorSize == 2 && !(r->colorAddr & 7) && !(r->colorWidth & 3) && !(r->zAddr & 7);
	g->edge = false;
	int u[12] = { (int) binsAt, (int) listsAt, tx0, ty0, (int) across, (int) g->lastUsed, (int) r->colorAddr, (int) r->colorWidth,
	              r->colorSize, (int) r->zAddr, words, 0 };
	glesSelectProgram(g->gl, gpuVariant(g, g->features | r->colorSize << F_FLAGS));
	g->features = 0;
	glesUniforms(g->gl, u, 12);
	glesDispatch(g->gl, across, down);
	g->batchBase[g->batchCount] = g->lastUsed;
	g->batchTiles[g->batchCount++] = across * down;
	g->lastUsed += across * down * 3;
	g->primCount = 0;
	g->entries = 0;
	g->stateCount = 0;
	g->rowCount = 0;
	g->tmemCount = 0;
	g->stateDirty = g->tmemDirty = true;
}

// Brings back everything the GPU drew.
static void gpuSync(struct N64* n, struct RDP* r) {
	struct Gpu* g = r->gpu;
	++n->stats.glSyncs;
	gpuSubmit(n, r);
	gpuUpdate(n, g);
	gpuCollectLast(g, r);
	// What the GPU drew with its hidden bits, copied out all at once (each read waits for the GPU):
	// the bytes drawn, else (too many ranges) every page drawn in, run by run.
	const int drawn = GPU_PAGE_OWNED | GPU_PAGE_MARKED;
	static int bindings[RDRAM_SIZE >> 11];
	static size_t offsets[RDRAM_SIZE >> 11], sizes[RDRAM_SIZE >> 11];
	static uint32_t firsts[RDRAM_SIZE >> 12], runs[RDRAM_SIZE >> 12];
	int count = 0;
	for (int i = 0; !g->drawnAll && i < g->drawnCount; ++i) {
		uint32_t from = g->drawnFrom[i] & ~1u, to = (g->drawnTo[i] + 1) & ~1u;
		firsts[count / 2] = from, runs[count / 2] = to - from;
		bindings[count] = 0, offsets[count] = from, sizes[count++] = to - from;
		bindings[count] = 0, offsets[count] = RDRAM_SIZE + (from >> 1), sizes[count++] = (to - from) >> 1;
	}
	for (int i = 0; g->drawnAll && i < g->markedPageCount; ++i) {
		int p = g->markedPages[i];
		if ((g->page[p] & drawn) != drawn) {
			continue;
		}
		while (p > 0 && (g->page[p - 1] & drawn) == drawn) {
			--p;
		}
		int run = 1;
		while (p + run < (RDRAM_SIZE >> 12) && (g->page[p + run] & drawn) == drawn) {
			++run;
		}
		for (int k = 0; k < run; ++k) {
			g->page[p + k] &= ~GPU_PAGE_OWNED;
		}
		firsts[count / 2] = (uint32_t) p << 12, runs[count / 2] = (uint32_t) run << 12;
		bindings[count] = 0, offsets[count] = (size_t) p << 12, sizes[count++] = (size_t) run << 12;
		bindings[count] = 0, offsets[count] = RDRAM_SIZE + ((size_t) p << 11), sizes[count++] = (size_t) run << 11;
	}
	const uint8_t* m = count ? glesMapRanges(g->gl, bindings, offsets, sizes, count) : NULL;
	if (m) {
		for (int i = 0; i < count / 2; ++i) {
			memcpy(&n->rdram[firsts[i]], m, runs[i]);
			m += runs[i];
			memcpy(&n->hidden[firsts[i] >> 1], m, runs[i] >> 1);
			m += runs[i] >> 1;
		}
		glesUnmap(g->gl, 0);
	}
	// Its copy of those pages stays as new as RDRAM: it keeps them, and writes there come to gpuWrite
	// (reads go ahead).
	for (int i = 0; i < g->markedPageCount; ++i) {
		int p = g->markedPages[i];
		g->page[p] = GPU_PAGE_OWNED;
		n->pageFlags[p] = (uint8_t) ((n->pageFlags[p] & ~PAGE_RDP_WRITES) | PAGE_RDP_READS);
	}
	g->markedPageCount = 0;
	g->drawnCount = 0;
	g->drawnAll = false;
	g->overCount = 0;
	n->rdpPending = false;
	if (g->patched) {
		const uint32_t* v = glesMap(g->gl, 5, 0, g->patchUsed * 4);
		for (int d = 0; v && d < 0x1000; ++d) {
			if (g->patch[d] >= 0) {
				r->tmem[d] = (uint8_t) v[g->patch[d]];
			}
		}
		glesUnmap(g->gl, 5);
		memset(g->patch, 0xFF, sizeof(g->patch));
		g->patched = 0;
	}
	g->patchUsed = 0;
}

// A load from bytes the GPU drew, done without waiting: the CPU loads (with RDRAM's older bytes)
// and keeps each TMEM byte's source, then the GPU gathers the drawn ones after the drawing before
// it. False when it can't: without the drawn bytes' ranges, or (after a full sync) with the patch
// buffer full.
static bool gpuLoadDrawn(struct N64* n, struct RDP* r, const uint64_t* w, int op) {
	struct Gpu* g = r->gpu;
	if (g->applyProgram < 0 || g->drawnAll) {
		return false;
	}
	gpuSubmit(n, r);
	gpuUpdate(n, g);
	memset(g->source, 0xFF, sizeof(g->source));
	r->tmemSource = g->source;
	execute(n, r, w, op);
	r->tmemSource = NULL;
	static uint32_t addrs[0x1000];
	uint32_t m = 0;
	for (int d = 0; d < 0x1000; ++d) {
		if (g->source[d] == UINT32_MAX) {
			continue;
		}
		g->patch[d] = -1;
		if (gpuReadNeeds(g, g->source[d], 1)) {
			g->patch[d] = (int32_t) (g->patchUsed + m);
			addrs[m++] = g->source[d];
		}
	}
	bool fits = g->patchUsed + m <= GPU_PATCHES_MAX;
	g->patched = 0;
	for (int d = 0; d < 0x1000; ++d) {
		if (!fits && g->source[d] != UINT32_MAX) {
			g->patch[d] = -1;
		}
		g->patched += g->patch[d] >= 0;
	}
	if (!fits) {
		gpuSync(n, r);
		return false;
	}
	if (!m) {
		return true;
	}
	glesSubData(g->gl, 5, g->patchUsed * 4, addrs, m * 4);
	int u[2] = { (int) g->patchUsed, (int) m };
	glesSelectProgram(g->gl, g->gatherProgram);
	glesUniforms(g->gl, u, 2);
	glesDispatch(g->gl, (m + 63) / 64, 1);
	g->patchUsed += m;
	return true;
}

// For a read of [addr, addr + len): what the GPU drew on those pages, back in RDRAM and off the drawn
// list (the rest stays on the GPU until wanted, or the next full sync).
static void gpuSyncRead(struct N64* n, struct RDP* r, uint32_t addr, uint32_t len) {
	struct Gpu* g = r->gpu;
	uint32_t from = addr & (RDRAM_SIZE - 1) & ~0xFFFu, to = ((addr & (RDRAM_SIZE - 1)) + len + 0xFFF) & ~0xFFFu;
	if (g->drawnAll || to > RDRAM_SIZE) {
		gpuSync(n, r);
		return;
	}
	// The drawn ranges it touches come back whole: a read goes on through what was drawn (a depth
	// buffer read here and there, NFL Quarterback Club 99).
	for (int i = 0; i < g->drawnCount; ++i) {
		uint32_t a = g->drawnFrom[i] & ~0xFFFu, b = (g->drawnTo[i] + 0xFFF) & ~0xFFFu;
		if (a < to && b > from) {
			from = a < from ? a : from;
			to = b > to ? b : to;
		}
	}
	++n->stats.glSyncs;
	gpuSubmit(n, r);
	gpuUpdate(n, g);
	static int bindings[GPU_RANGES * 2];
	static size_t offsets[GPU_RANGES * 2], sizes[GPU_RANGES * 2];
	uint32_t pieceFrom[GPU_RANGES], pieceTo[GPU_RANGES];
	int pieces = 0, count = 0;
	for (int i = 0; i < g->drawnCount; ++i) {
		uint32_t a = (g->drawnFrom[i] > from ? g->drawnFrom[i] : from) & ~1u, b = g->drawnTo[i] < to ? g->drawnTo[i] : to;
		b = (b + 1) & ~1u;
		if (a >= b) {
			continue;
		}
		pieceFrom[pieces] = a, pieceTo[pieces++] = b;
		bindings[count] = 0, offsets[count] = a, sizes[count++] = b - a;
		bindings[count] = 0, offsets[count] = RDRAM_SIZE + (a >> 1), sizes[count++] = (b - a) >> 1;
	}
	const uint8_t* m = count ? glesMapRanges(g->gl, bindings, offsets, sizes, count) : NULL;
	if (!m) {
		return;
	}
	for (int i = 0; i < pieces; ++i) {
		uint32_t a = pieceFrom[i], b = pieceTo[i];
		memcpy(&n->rdram[a], m, b - a);
		m += b - a;
		memcpy(&n->hidden[a >> 1], m, (b - a) >> 1);
		m += (b - a) >> 1;
	}
	glesUnmap(g->gl, 0);
	// Drawn ranges lose the part read back; one with it in its middle keeps it (back again later).
	for (int i = 0; i < g->drawnCount; ++i) {
		uint32_t a = g->drawnFrom[i], b = g->drawnTo[i];
		if (a >= from && b <= to) {
			g->drawnFrom[i] = g->drawnFrom[--g->drawnCount], g->drawnTo[i] = g->drawnTo[g->drawnCount];
			--i;
		} else if (a >= from && a < to) {
			g->drawnFrom[i] = to;
		} else if (b > from && b <= to) {
			g->drawnTo[i] = from;
		}
	}
}

// The GPU RDP's context, with its copy of RDRAM [addr, addr + len) and those bytes' hidden bits
// as new as RDRAM's, for work on the GPU that reads them after what it drew (the VI); NULL without
// the GPU RDP. Nothing waits: the GPU takes the work in order.
struct Gles* rdpGpuShare(struct N64* n, uint32_t addr, uint32_t len) {
	struct RDP* r = n->rdp;
	if (!r || !r->gpu) {
		return NULL;
	}
	struct Gpu* g = r->gpu;
	gpuSubmit(n, r);
	gpuUpdate(n, g);
	// Pages the GPU doesn't own are newer in RDRAM.
	FOR_PAGES(addr, len, p) {
		if (!(g->page[p] & GPU_PAGE_OWNED)) {
			glesSubData(g->gl, 0, (size_t) p << 12, &n->rdram[(size_t) p << 12], 0x1000);
			glesSubData(g->gl, 0, RDRAM_SIZE + ((size_t) p << 11), &n->hidden[(size_t) p << 11], 0x800);
		}
	}
	return g->gl;
}

// A command, with the GPU drawing.
static void gpuCommand(struct N64* n, struct RDP* r, const uint64_t* w, int op) {
	struct Gpu* g = r->gpu;
	gpuUpdate(n, g);
	bool draws = (op >= 0x08 && op <= 0x0F) || op == 0x24 || op == 0x25 || op == 0x36;
	if (!draws) {
		if (op == 0x30 || op == 0x33 || op == 0x34) {
			uint32_t addr, len;
			loadSource(r, w, op, &addr, &len);
			g->tmemDirty = true;
			if (n->rdpPending && gpuReadNeeds(g, addr, len)) {
				if (gpuLoadDrawn(n, r, w, op)) {
					g->stateDirty = true;
					return;
				}
				if (n->rdpPending) {
					gpuSyncRead(n, r, addr, len);
				}
			}
			if (g->patched) {
				// Bytes loaded over gathered ones are RDRAM's now.
				memset(g->source, 0xFF, sizeof(g->source));
				r->tmemSource = g->source;
				execute(n, r, w, op);
				r->tmemSource = NULL;
				for (int d = 0; d < 0x1000; ++d) {
					if (g->source[d] != UINT32_MAX && g->patch[d] >= 0) {
						g->patch[d] = -1;
						--g->patched;
					}
				}
				g->stateDirty = true;
				return;
			}
		} else if ((op == 0x3E || op == 0x3F) && g->primCount) {
			gpuSubmit(n, r);
		}
		execute(n, r, w, op);
		g->stateDirty = true;
		return;
	}
	++r->seq;
	if (op >= 0x08 && op <= 0x0F) {
		r->dzpix = triangleDz(w, op);
	}
	int y0, y1;
	primitiveRows(r, w, op, &y0, &y1);
	if (y1 < y0) {
		return;
	}
	uint32_t bpp = r->colorSize == 1 ? 1 : r->colorSize == 3 ? 4 : 2;
	uint32_t right = (uint32_t) ((r->scissorX1 >> 2) + 1);
	if (op == 0x24 || op == 0x25 || op == 0x36) {
		// A rectangle ends where it does, not at the scissor (Mario Tennis draws into a narrow image).
		int cx0, cy0, x0, ry0, x1, ry1;
		rectBounds(r, w[0], &cx0, &cy0, &x0, &ry0, &x1, &ry1);
		right = x1 >= x0 && (uint32_t) x1 + 1 < right ? (uint32_t) x1 + 1 : right;
	}
	bool depth = r->zAddr && (r->zCompare || r->zUpdate);
	uint32_t color = r->colorAddr + (uint32_t) y0 * r->colorWidth * bpp;
	uint32_t colorLen = (uint32_t) (y1 - y0) * r->colorWidth * bpp + right * bpp;
	uint32_t z = r->zAddr + (uint32_t) y0 * r->colorWidth * 2;
	uint32_t zLen = (uint32_t) (y1 - y0) * r->colorWidth * 2 + right * 2;
	bool overlap = depth && z < color + colorLen && color < z + zLen;
	bool shades = !(op == 0x36 && r->cycleType == 3) && !((op == 0x24 || op == 0x25) && r->cycleType == 2);
	const struct Stage* firstStage = &r->stage[r->cycleType == 1 ? 0 : 1];
	bool alone = right - 1 > r->colorWidth || overlap || (shades && firstStage->readsCombined);
	if (alone || y1 >= GPU_TILES * 16) {
		gpuSync(n, r);
		// execute counts the primitive itself (its noise from the same number as on the CPU).
		--r->seq;
		execute(n, r, w, op);
		// Drawn here, so newer than the GPU's copy.
		FOR_PAGES(color, colorLen, p) {
			g->page[p] &= ~GPU_PAGE_OWNED;
			n->pageFlags[p] &= ~PAGE_RDP_READS;
		}
		if (depth) {
			FOR_PAGES(z, zLen, p) {
				g->page[p] &= ~GPU_PAGE_OWNED;
				n->pageFlags[p] &= ~PAGE_RDP_READS;
			}
		}
		return;
	}
	// At most the scissor's width times the rows it may draw.
	int across = ((r->scissorX1 >> 2) >> 4) - ((r->scissorX0 >> 2) >> 4) + 1;
	if (g->primCount && g->entries + across * ((y1 >> 4) - (y0 >> 4) + 1) > GPU_ENTRIES_MAX) {
		gpuSubmit(n, r);
	}
	int32_t* p = g->prims + g->primCount * GPU_PRIM_SIZE;
	memset(p, 0, GPU_PRIM_SIZE * sizeof(int32_t));
	p[22] = r->dzpix;
	p[23] = r->primDz;
	p[24] = (int32_t) r->seq;
	if (op >= 0x08 && op <= 0x0F) {
		if (!gpuTriangle(g, r, w, op, p)) {
			return;
		}
	} else if (op == 0x24 || op == 0x25) {
		int cx0, cy0, x0, x1;
		rectBounds(r, w[0], &cx0, &cy0, &x0, &y0, &x1, &y1);
		if (x1 < x0 || y1 < y0) {
			return;
		}
		int dsdx = (int16_t) (w[1] >> 16);
		p[0] = GPU_KIND_TEXRECT;
		p[2] = x0, p[3] = y0, p[4] = x1, p[5] = y1;
		p[6] = (w[0] >> 24) & 7;
		p[8] = (int16_t) (w[1] >> 48);
		p[9] = (int16_t) (w[1] >> 32);
		p[10] = r->cycleType == 2 ? dsdx / 4 : dsdx;
		p[11] = (int16_t) w[1];
		p[12] = cx0, p[13] = cy0;
		p[14] = op == 0x25;
	} else {
		int x0, x1, cx0, cy0;
		rectBounds(r, w[0], &cx0, &cy0, &x0, &y0, &x1, &y1);
		if (x1 < x0 || y1 < y0) {
			return;
		}
		p[0] = r->cycleType == 3 ? GPU_KIND_FILL : GPU_KIND_RECT;
		p[2] = x0, p[3] = y0, p[4] = x1, p[5] = y1;
	}
	g->entries += ((p[4] >> 4) - (p[2] >> 4) + 1) * ((p[5] >> 4) - (p[3] >> 4) + 1);
	g->features |= gpuFeatures(r, op);
	g->edge |= p[4] >= (int) r->colorWidth;
	p[1] = gpuState(g, r);
	gpuMark(n, g, color, colorLen);
	if (depth) {
		gpuMark(n, g, z, zLen);
	}
	if (++g->primCount == GPU_PRIMS_MAX) {
		gpuSubmit(n, r);
	}
}

// N64_FRAMESKIP: whether a color image is one the VI shows (its origin within the first lines).
static int shownBuffer(const struct N64* n, uint32_t addr) {
	for (int i = 0; i < 4; ++i) {
		if (n->skip.shown[i] && n->skip.shown[i] >= addr && n->skip.shown[i] - addr < 0x10000) {
			return i;
		}
	}
	return -1;
}

// Follows the color images to count frames into shown buffers; true for a primitive to leave
// undrawn (the RDP's time for it still counts).
static bool skipCommand(struct N64* n, const uint64_t* w, int op) {
	struct FrameSkip* s = &n->skip;
	if (op == 0x3F) {
		uint32_t addr = w[0] & 0xFFFFFF;
		s->intoShown = shownBuffer(n, addr) >= 0;
		if (!s->intoShown) {
			return false;
		}
		// Images drawn in between (textures) don't start a frame; another shown buffer does.
		if (addr != s->drawing) {
			s->drawing = addr;
			s->skipping = ++s->frames & 1;
			int slot = 0;
			for (int i = 0; i < 4; ++i) {
				if (s->buffer[i] == addr || !s->buffer[i]) {
					slot = i;
					break;
				}
			}
			s->buffer[slot] = addr;
			s->skipped[slot] = s->skipping;
		}
		return false;
	}
	bool draws = (op >= 0x08 && op <= 0x0F) || op == 0x24 || op == 0x25 || op == 0x36;
	return draws && s->intoShown && s->skipping;
}

// Whether the frame in the buffer at the VI's origin was left undrawn.
bool frameSkipped(struct N64* n, uint32_t origin) {
	for (int i = 0; i < 4; ++i) {
		if (n->skip.buffer[i] && origin >= n->skip.buffer[i] && origin - n->skip.buffer[i] < 0x10000) {
			return n->skip.skipped[i];
		}
	}
	return false;
}

void rdpDrawSoftware(struct N64* n, struct RDP* r, const uint64_t* w, int op) {
	execute(n, r, w, op);
}

// Starts a run of commands: the renderers or the GPU on first use, and when the RDP can take them.
static struct RDP* beginCommands(struct N64* n, uint64_t* begin) {
	struct RDP* r = state(n);
	if (n->rdpGpu && !r->gpu) {
		r->gpu = gpuCreate();
		n->rdpGpu = r->gpu != NULL;
	}
	if (n->rdpGl && !n->glr) {
		n->glr = glrCreate(n);
	}
	if (n->rdpThreads && !r->renderers && !r->gpu && !n->glr) {
		r->renderers = startRenderers(n, r);
	}
	*begin = n->dp.busyUntil > n->cycles ? n->dp.busyUntil : n->cycles;
	r->clocks = 0;
	return r;
}

// The pages of rows from to to - 1 of an image at addr written.
static void rowsWritten(struct N64* n, uint32_t addr, int from, int to, uint32_t rowBytes) {
	if (to > from && rowBytes) {
		uint32_t first = addr + (uint32_t) from * rowBytes, last = addr + (uint32_t) to * rowBytes - 1;
		for (uint32_t a = first & ~0xFFFu; a <= last; a += 0x1000) {
			n->dirtyPages[(a & (RDRAM_SIZE - 1)) >> 12] = 0;
		}
	}
}

// One command: a full sync interrupts the CPU once the commands before it would be done.
static void command(struct N64* n, struct RDP* r, const uint64_t* w, int op, int words, uint64_t begin) {
	// A primitive writes the color image's rows within the scissor, and the depth image's.
	if ((op >= 0x08 && op <= 0x0F) || op == 0x24 || op == 0x25 || op == 0x36) {
		int const from = r->scissorY0 >> 2, to = (r->scissorY1 >> 2) + 1;
		uint32_t const width = r->colorWidth;
		rowsWritten(n, r->colorAddr, from, to, width * (r->colorSize == 3 ? 4 : r->colorSize == 2 ? 2 : 1));
		rowsWritten(n, r->zAddr, from, to, width * 2);
	}
	++n->checkCounts[0][op];
	if (n->checkLen[0] < 4096) {
		n->checkSeq[0][n->checkLen[0]++] = w[0];
	}
	if (op == 0x29) {
		n->dp.status &= ~DP_RUNNING;
		if (r->gpu) {
			gpuSubmit(n, r);
		}
		++n->stats.dpSyncs;
		uint64_t at = begin + r->clocks * 3 / 2;
		if (n->accuracy == ACCURACY_HLE && at < n->sp.hleDoneAt) {
			at = n->sp.hleDoneAt;
		}
		if (n->accuracy == ACCURACY_HLE && n->dp.syncAt && n->dp.syncQueued < 8) {
			n->dp.syncQueue[n->dp.syncQueued++] = at;
			return;
		}
		if (n->dp.syncAt) {
			n64Interrupt(n, MI_DP, true);
		}
		n->dp.syncAt = at;
		n64Event(n, n->dp.syncAt);
		return;
	}
	// A texture load from a color image the GPU drew waits for its pixels.
	if (n->glr && n->rdpPending && (op == 0x30 || op == 0x33 || op == 0x34)) {
		uint32_t addr, len;
		loadSource(r, w, op, &addr, &len);
		if (glrDrew(n, addr, len)) {
			glrSync(n);
		}
	}
	// The GL renderer draws; the RDP keeps the state (texture loads included) and the time.
	if (n->glr && (op == 0x01 || (op >= 0x08 && op <= 0x0F) || op == 0x24 || op == 0x25 || op == 0x36)) {
		if (!(n->skip.on && skipCommand(n, w, op))) {
			glrDraw(n, r, w, op);
		}
		uint64_t clocks = op == 0x01 ? primitiveClocks(r, w + 16, 0x08) : primitiveClocks(r, w, op);
		r->clocks += clocks;
		n->dp.bufBusy += (uint32_t) clocks;
		n->dp.pipeBusy += (uint32_t) clocks;
		return;
	}
	if (n->skip.on && skipCommand(n, w, op)) {
		uint64_t clocks = primitiveClocks(r, w, op);
		r->clocks += clocks;
		n->dp.bufBusy += (uint32_t) clocks;
		n->dp.pipeBusy += (uint32_t) clocks;
		return;
	}
	uint64_t clocks = primitiveClocks(r, w, op);
	r->clocks += clocks;
	n->dp.bufBusy += (uint32_t) clocks;
	n->dp.pipeBusy += (uint32_t) clocks;
	if (op == 0x30 || op == 0x33 || op == 0x34) {
		n->dp.tmemBusy += (uint32_t) clocks;
	}
	if (r->gpu) {
		gpuCommand(n, r, w, op);
		return;
	}
	if (!r->renderers) {
		execute(n, r, w, op);
		return;
	}
	submit(n, r, w, op, words);
	r->renderers->unannounced += clocks;
	// The main thread keeps the state the cost model and the flags need; it draws nothing.
	bool draws = (op >= 0x08 && op <= 0x0F) || op == 0x24 || op == 0x25 || op == 0x36;
	if (!draws && op != 0x30 && op != 0x33 && op != 0x34) {
		execute(n, r, w, op);
	}
}

static void endCommands(struct N64* n, struct RDP* r, uint64_t begin) {
	if (n->rdpTap) {
		n->rdpTap(n, NULL, 0);
	}
	if (r->renderers && r->renderers->unannounced >= WAKE_CLOCKS) {
		wake(r->renderers);
	}
	n->dp.busyUntil = begin + r->clocks * 3 / 2;
}

// Runs the commands from DP_CURRENT to DP_END, keeping the RDP's time.
void rdpRun(struct N64* n) {
	if (n->dp.status & 2) {
		return;
	}
	n->dp.status |= DP_RUNNING;
	uint64_t begin;
	struct RDP* r = beginCommands(n, &begin);
	struct DP* d = &n->dp;
	while (d->current < d->end) {
		uint64_t* w = d->partial;
		if (!d->partialWords) {
			w[d->partialWords++] = word(n, d->current);
			d->current += 8;
		}
		int op = (w[0] >> 56) & 0x3F;
		int words = commandWords(op);
		while (d->partialWords < words && d->current < d->end) {
			w[d->partialWords++] = word(n, d->current);
			d->current += 8;
		}
		if (d->partialWords < words) {
			break;
		}
		d->partialWords = 0;
		if (n->rdpTap) {
			n->rdpTap(n, w, words);
		}
		command(n, r, w, op, words, begin);
	}
	endCommands(n, r, begin);
}

// Runs a list of commands made outside RDRAM (hle.c).
void rdpCommands(struct N64* n, const uint64_t* list, size_t count) {
	n->dp.status |= DP_RUNNING;
	uint64_t begin;
	struct RDP* r = beginCommands(n, &begin);
	for (size_t i = 0; i < count;) {
		int op = (list[i] >> 56) & 0x3F;
		int words = commandWords(op);
		if (i + words > count) {
			break;
		}
		if (n->rdpTap) {
			n->rdpTap(n, &list[i], words);
		}
		command(n, r, &list[i], op, words, begin);
		i += words;
	}
	endCommands(n, r, begin);
}

// The RDP as a save state keeps it (once rdpWait has run). With renderer threads the main thread skips
// texture loads and draws, so the state is renderer 0's, with the latest combiner output of any.
void rdpStateGet(struct N64* n, struct RDP* out) {
	struct RDP* r = n->rdp;
	struct Renderers* all = r->renderers;
	if (!all) {
		memcpy(out, r, sizeof(*out));
		return;
	}
	memcpy(out, all->renderer[0].rdp, sizeof(*out));
	for (int i = 1; i < all->count; ++i) {
		const struct RDP* other = all->renderer[i].rdp;
		if (other->lastPos > out->lastPos) {
			out->lastPos = other->lastPos;
			out->lastCombined = other->lastCombined;
		}
	}
	out->clocks = r->clocks, out->tmemGen = r->tmemGen, out->stateGen = r->stateGen;
	out->rowIndex = r->rowIndex, out->rowCount = r->rowCount;
	out->renderers = r->renderers, out->gpu = r->gpu, out->tmemSource = r->tmemSource;
}

// A save state's RDP (NULL for none yet): the renderer threads, GPU RDP and GL renderer go, and
// start again from it when next drawn; with keepGl the GL renderer stays, forgetting what it knew of
// the RDP's state.
void rdpStateLoad(struct N64* n, const struct RDP* saved, bool keepGl) {
	struct Glr* glr = keepGl ? n->glr : NULL;
	if (glr) {
		n->glr = NULL;
		glrStateLoaded(glr);
	}
	rdpDestroy(n);
	n->glr = glr;
	if (saved && (n->rdp = malloc(sizeof(*n->rdp)))) {
		memcpy(n->rdp, saved, sizeof(*n->rdp));
		n->rdp->tmemSource = NULL;
		n->rdp->renderers = NULL;
		n->rdp->gpu = NULL;
		// The combiner's inputs point into the RDP's own fields.
		setupCombiner(n->rdp);
	}
}

void rdpDestroy(struct N64* n) {
	if (n->glr) {
		glrDestroy(n->glr);
		n->glr = NULL;
	}
	if (n->rdp && n->rdp->renderers) {
		rdpWait(n);
		stopRenderers(n->rdp->renderers);
	}
	if (n->rdp && n->rdp->gpu) {
		rdpWait(n);
		gpuDestroy(n->rdp->gpu);
	}
	free(n->rdp);
	n->rdp = NULL;
}
