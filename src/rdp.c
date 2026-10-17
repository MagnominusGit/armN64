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
	int k4, k5;
};

static struct RDP* state(struct N64* n) {
	if (!n->rdp) {
		n->rdp = calloc(1, sizeof(struct RDP));
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
	int e = 0;
	while (e < 7 && z >= zBase[e + 1]) {
		++e;
	}
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

static struct Color tlutColor(struct RDP* r, int index) {
	uint32_t a = 0x800 + index * 8;
	uint16_t v = (uint16_t) (tmemByte(r, a) << 8 | tmemByte(r, a + 1));
	if ((r->otherModes >> 46) & 1) {
		return (struct Color) { v >> 8, v >> 8, v >> 8, v & 255 };
	}
	int cr = (v >> 11) & 31, cg = (v >> 6) & 31, cb = (v >> 1) & 31;
	return (struct Color) { cr << 3 | cr >> 2, cg << 3 | cg >> 2, cb << 3 | cb >> 2, v & 1 ? 255 : 0 };
}

static struct Color texel(struct RDP* r, struct Tile* t, int s, int tt) {
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
static int wrapCoord(int c, int shift, int lo, int hi, int cm, int mask, int* frac) {
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
	struct Color c0 = texel(r, tile, s0, t0), c1 = texel(r, tile, s1, t0), c2 = texel(r, tile, s0, t1);
	struct Color out;
	// Three-point filtering: the triangle of texels nearest the sample point.
	if (fs + ft < 32) {
		out.r = c0.r + ((c1.r - c0.r) * fs + (c2.r - c0.r) * ft) / 32;
		out.g = c0.g + ((c1.g - c0.g) * fs + (c2.g - c0.g) * ft) / 32;
		out.b = c0.b + ((c1.b - c0.b) * fs + (c2.b - c0.b) * ft) / 32;
		out.a = c0.a + ((c1.a - c0.a) * fs + (c2.a - c0.a) * ft) / 32;
	} else {
		struct Color c3 = texel(r, tile, s1, t1);
		int gs = 32 - fs, gt = 32 - ft;
		out.r = c3.r + ((c2.r - c3.r) * gs + (c1.r - c3.r) * gt) / 32;
		out.g = c3.g + ((c2.g - c3.g) * gs + (c1.g - c3.g) * gt) / 32;
		out.b = c3.b + ((c2.b - c3.b) * gs + (c1.b - c3.b) * gt) / 32;
		out.a = c3.a + ((c2.a - c3.a) * gs + (c1.a - c3.a) * gt) / 32;
	}
	return out;
}

// ---- Combiner and blender

struct Pixel {
	struct Color shade, tex0, tex1, combined;
	int z;
	int lodFrac;
};

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
			return 255;
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
			return 255;
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
		return multiplier ? r->primLod : 255;
	}
	return 0;
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
		int a = rgbInput(r, p, subA, ch, 0), b = rgbInput(r, p, subB, ch, 1);
		int m = rgbInput(r, p, mul, ch, 2), d = rgbInput(r, p, add, ch, 3);
		o[ch] = clamp8(((a - b) * m + (d << 8) + 0x80) >> 8);
	}
	int a = alphaInput(r, p, aA, false), b = alphaInput(r, p, aB, false);
	int m = alphaInput(r, p, aMul, true), d = alphaInput(r, p, aAdd, false);
	out.a = clamp8(((a - b) * m + (d << 8) + 0x80) >> 8);
	return out;
}

static struct Color blendColor(struct RDP* r, struct Color pixel, struct Color memory, struct Pixel* p, int cycle) {
	uint64_t m = r->otherModes;
	int sp = (m >> (30 - cycle * 2)) & 3, sa = (m >> (26 - cycle * 2)) & 3;
	int sm = (m >> (22 - cycle * 2)) & 3, sb = (m >> (18 - cycle * 2)) & 3;
	const struct Color* choices[4] = { &pixel, &memory, &r->blend, &r->fog };
	struct Color P = *choices[sp], M = *choices[sm];
	int a = sa == 0 ? pixel.a : sa == 1 ? r->fog.a : sa == 2 ? p->shade.a : 0;
	int b = sb == 0 ? 255 - a : sb == 1 ? memory.a : sb == 2 ? 255 : 0;
	int sum = a + b;
	if (sum == 0) {
		return P;
	}
	struct Color out = { (P.r * a + M.r * b) / sum, (P.g * a + M.g * b) / sum, (P.b * a + M.b * b) / sum, pixel.a };
	return out;
}

// The blender's first input (P) for a cycle, without blending.
static struct Color blendInput(struct RDP* r, struct Color pixel, struct Color memory, int cycle) {
	int sp = (r->otherModes >> (30 - cycle * 2)) & 3;
	struct Color out = sp == 0 ? pixel : sp == 1 ? memory : sp == 2 ? r->blend : r->fog;
	out.a = pixel.a;
	return out;
}

static void shadePixel(struct N64* n, struct RDP* r, int x, int y, struct Pixel* p) {
	uint64_t m = r->otherModes;
	int cycleType = (m >> 52) & 3;
	bool zCompare = (m >> 4) & 1, zUpdate = (m >> 5) & 1;
	int z = (m >> 2) & 1 ? r->primZ : p->z;
	if (z < 0) {
		z = 0;
	} else if (z > 0x3FFFF) {
		z = 0x3FFFF;
	}
	if (r->zAddr && zCompare) {
		int old = zDecompress(zRead(n, r, x, y));
		int mode = (m >> 10) & 3;
		bool pass = mode == 3 ? z <= old + 0x10 && z >= old - 0x10 : z <= old;
		if (!pass) {
			return;
		}
	}
	struct Color c = combine(r, p, 0);
	if (cycleType == 1) {
		p->combined = c;
		c = combine(r, p, 1);
	}
	if ((m & 1) && c.a < r->blend.a) {
		return;
	}
	struct Color memory = readPixel(n, r, x, y);
	// The blend formula applies only with force blend (or at partly covered edges, which aren't
	// modelled yet); otherwise a cycle passes its first input through. In 2-cycle mode the first
	// cycle always blends (fog), and its result is the second cycle's pixel.
	bool force = (m >> 14) & 1;
	struct Color out = c;
	if (cycleType == 1) {
		struct Color first = blendColor(r, c, memory, p, 0);
		out = force ? blendColor(r, first, memory, p, 1) : blendInput(r, first, memory, 1);
	} else {
		out = force ? blendColor(r, c, memory, p, 0) : blendInput(r, c, memory, 0);
	}
	writePixel(n, r, x, y, out);
	if (r->zAddr && zUpdate) {
		zWrite(n, r, x, y, zCompress(z));
	}
}

// ---- Primitives

static void fillRect(struct N64* n, struct RDP* r, int x0, int y0, int x1, int y1) {
	for (int y = y0; y <= y1; ++y) {
		for (int x = x0; x <= x1; ++x) {
			uint32_t i = y * r->colorWidth + x;
			if (r->colorSize == 3) {
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
	for (int y = y0; y <= y1; ++y) {
		for (int x = x0; x <= x1; ++x) {
			int dx = x - cx0, dy = y - cy0;
			int s = ((s0 << 5) + (flip ? dy * dsdx : dx * dsdx)) >> 5;
			int t = ((t0 << 5) + (flip ? dx * dtdy : dy * dtdy)) >> 5;
			struct Pixel p = { 0 };
			p.tex0 = sample(r, tile, s, t);
			p.tex1 = cycleType == 1 ? sample(r, tile + 1, s, t) : p.tex0;
			if (cycleType == 2) {
				if ((r->otherModes & 1) && p.tex0.a == 0) {
					continue;
				}
				writePixel(n, r, x, y, p.tex0);
				continue;
			}
			p.z = r->primZ;
			shadePixel(n, r, x, y, &p);
		}
	}
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
	int64_t tv[3] = { 0 }, tdx[3] = { 0 }, tde[3] = { 0 };
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
		}
		idx += 8;
	}
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
		for (int x = x0; x <= x1; ++x) {
			int64_t dx = (int64_t) x * 65536 + 32768 - xh - dxh * dy / 4;
			struct Pixel p = { 0 };
			if (shade) {
				int ch[4];
				for (int i = 0; i < 4; ++i) {
					int64_t v = sv[i] + sde[i] * ey + (sdx[i] * dx >> 16);
					ch[i] = clamp8((int) (v >> 16));
				}
				p.shade = (struct Color) { ch[0], ch[1], ch[2], ch[3] };
			}
			if (zbuf) {
				int64_t v = zv + zde * ey + (zdx * dx >> 16);
				p.z = (int) (v >> 13);
			}
			if (tex) {
				int64_t sS = tv[0] + tde[0] * ey + (tdx[0] * dx >> 16);
				int64_t sT = tv[1] + tde[1] * ey + (tdx[1] * dx >> 16);
				int64_t sW = tv[2] + tde[2] * ey + (tdx[2] * dx >> 16);
				int s, t;
				if (persp) {
					int wv = (int) ((sW >> 16) & 0x7FFF);
					if (wv == 0) {
						wv = 1;
					}
					s = (int) (((sS >> 16) * 0x8000) / wv);
					t = (int) (((sT >> 16) * 0x8000) / wv);
				} else {
					s = (int) (sS >> 16);
					t = (int) (sT >> 16);
				}
				p.tex0 = sample(r, tile, s, t);
				p.tex1 = cycleType == 1 ? sample(r, tile + 1, s, t) : p.tex0;
			}
			shadePixel(n, r, x, y, &p);
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

void rdpRun(struct N64* n) {
	struct RDP* r = state(n);
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
			break;
		case 0x24:
		case 0x25:
			textureRect(n, r, w[0], w[1], op == 0x25);
			break;
		case 0x29:
			++n->stats.dpSyncs;
			n64Interrupt(n, MI_DP, true);
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
						struct Pixel p = { 0 };
						p.z = r->primZ;
						shadePixel(n, r, x, y, &p);
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
			break;
		case 0x3B:
			r->env = unpack((uint32_t) c);
			break;
		case 0x3C:
			r->combine = c;
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
}

void rdpDestroy(struct N64* n) {
	free(n->rdp);
	n->rdp = NULL;
}
