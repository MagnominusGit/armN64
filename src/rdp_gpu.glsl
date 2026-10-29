#version 310 es
// The RDP's pixel work (rdp.c) on the GPU: a workgroup draws a 16x16 tile of the picture, each
// invocation one pixel, through the tile's primitives in command order. Same integer arithmetic as
// rdp.c; the CPU works out triangle spans and each row's starting attributes (64-bit).
precision highp int;
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer Rdram { uint rdram[]; };
layout(std430, binding = 1) readonly buffer Tmem { uint tmem[]; };
layout(std430, binding = 2) readonly buffer Data { int data[]; };
layout(std430, binding = 3) writeonly buffer Last { uint lastOut[]; };

// 0 bins, 1 bin lists, 2 first tile x, 3 first tile y, 4 tiles across, 5 lastOut base, 6 color image,
// 7 its width, 8 its pixel size, 9 depth image
uniform int u[12];

const uint RDRAM_MASK = 0x7FFFFFu;
const int TILES_ACROSS = 64;

// Primitive kinds and record fields (rdp.c GPU_*).
const int KIND_TRIANGLE = 0, KIND_TEXRECT = 1, KIND_FILL = 2, KIND_RECT = 3;
const int P_KIND = 0, P_STATE = 1, P_X0 = 2, P_Y0 = 3, P_X1 = 4, P_Y1 = 5, P_TILE = 6, P_MAXLEVEL = 7;
const int P_ROWS = 8, P_FIRST = 9, P_FLAGS = 10, P_SDX = 11, P_TDX = 15, P_TDY = 18, P_ZDX = 21;
const int P_S0 = 8, P_T0 = 9, P_DSDX = 10, P_DTDY = 11, P_CX0 = 12, P_CY0 = 13, P_FLIP = 14;
const int PRIM_SIZE = 24, ROW_SIZE = 18;
const int S_OM_LO = 0, S_OM_HI = 1, S_CB_LO = 2, S_CB_HI = 3, S_CADDR = 4, S_CWIDTH = 5, S_CSIZE = 6, S_ZADDR = 7;
const int S_FOG = 8, S_BLEND = 9, S_PRIM = 10, S_ENV = 11, S_PRIMLOD = 12, S_PRIMZ = 13, S_MINLEVEL = 14;
const int S_K4 = 15, S_K5 = 16, S_FILL = 17, S_TMEM = 18, S_TILES = 19, TILE_SIZE = 15;

// ---- The primitive being drawn
int gS;
uint omLo, omHi, cbLo, cbHi;
ivec4 gFog, gBlend, gPrim, gEnv;
int gPrimLod, gK4, gK5, gTmem, gMaxLevel, gMinLevel;

// ---- The pixel
ivec4 pShade, pTex0, pTex1, pCombined;
int pZ, pLodFrac;
ivec4 lastCombined;
bool shaded;

ivec4 unpackColor(int v) {
	uint u = uint(v);
	return ivec4(int(u >> 24), int((u >> 16) & 255u), int((u >> 8) & 255u), int(u & 255u));
}

// ---- RDRAM, bytes in the host's order (little endian words)

uint rbyte(uint a) {
	a &= RDRAM_MASK;
	return (rdram[a >> 2] >> ((a & 3u) * 8u)) & 255u;
}

void wbyte(uint a, uint v) {
	a &= RDRAM_MASK;
	uint s = (a & 3u) * 8u;
	atomicAnd(rdram[a >> 2], ~(255u << s));
	atomicOr(rdram[a >> 2], (v & 255u) << s);
}

// Big-endian halfword at an even address.
uint rhalf(uint a) {
	return rbyte(a) << 8 | rbyte(a + 1u);
}

void whalf(uint a, uint v) {
	a &= RDRAM_MASK;
	uint s = (a & 3u) * 8u;
	uint bits = ((v >> 8) & 255u) | (v & 255u) << 8;
	atomicAnd(rdram[a >> 2], ~(0xFFFFu << s));
	atomicOr(rdram[a >> 2], bits << s);
}

// ---- 64-bit values as (low, high)

uvec2 add64(uvec2 a, uvec2 b) {
	uint carry;
	uint lo = uaddCarry(a.x, b.x, carry);
	return uvec2(lo, a.y + b.y + carry);
}

uvec2 ext64(int v) {
	return uvec2(uint(v), v < 0 ? 0xFFFFFFFFu : 0u);
}

uvec2 mul64(int a, int b) {
	int hi, lo;
	imulExtended(a, b, hi, lo);
	return uvec2(uint(lo), uint(hi));
}

uvec2 load64(int i) {
	return uvec2(uint(data[i]), uint(data[i + 1]));
}

// (int) (v >> sh) for 0 < sh < 32.
int bits64(uvec2 v, int sh) {
	return int((v.x >> uint(sh)) | (v.y << uint(32 - sh)));
}

// ---- Z buffer

// zShift and zBase in rdp.c, for exponents 0-7.
int zShift(int e) {
	return e < 6 ? 6 - e : 0;
}

int zBase(int e) {
	return e == 0 ? 0 : 0x40000 - (0x40000 >> e);
}

uint zCompress(int z) {
	uint x = ~(uint(z) << 14);
	int e = 31 - findMSB(x);
	e = e > 7 ? 7 : e;
	int m = ((z - zBase(e)) >> zShift(e)) & 0x7FF;
	return uint((e << 11 | m) << 2);
}

int zDecompress(uint v) {
	int e = int(v >> 13) & 7, m = int(v >> 2) & 0x7FF;
	return zBase(e) + (m << zShift(e));
}

// ---- Framebuffer: in a batch the color and depth images stay put, and every pixel of them belongs
// to one invocation (rdp.c draws on the CPU where rows or the two images run into each other), so
// each invocation keeps its pixel's color and depth (big-endian values) and writes them back at the
// end.

uint fbColor, fbZ;
bool colorDirty, zDirty;
uint colorAt, zAt;

uint rword(uint a) {
	return rbyte(a) << 24 | rbyte(a + 1u) << 16 | rbyte(a + 2u) << 8 | rbyte(a + 3u);
}

void loadPixel(int x, int y) {
	uint i = uint(y * u[7] + x);
	uint addr = uint(u[6]);
	int size = u[8];
	if (size == 1) {
		colorAt = (addr + i) & RDRAM_MASK;
		fbColor = rbyte(colorAt);
	} else if (size == 3) {
		colorAt = (addr + i * 4u) & (RDRAM_MASK & ~3u);
		fbColor = rword(colorAt);
	} else {
		colorAt = (addr + i * 2u) & (RDRAM_MASK & ~1u);
		fbColor = rhalf(colorAt);
	}
	zAt = (uint(u[9]) + i * 2u) & (RDRAM_MASK & ~1u);
	fbZ = u[9] != 0 ? rhalf(zAt) : 0u;
	colorDirty = zDirty = false;
}

void storePixel() {
	if (colorDirty) {
		int size = u[8];
		if (size == 1) {
			wbyte(colorAt, fbColor);
		} else if (size == 3) {
			uint v = fbColor;
			rdram[colorAt >> 2] = (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
		} else {
			whalf(colorAt, fbColor);
		}
	}
	if (zDirty) {
		whalf(zAt, fbZ);
	}
}

ivec4 readPixel(int x, int y) {
	int size = u[8];
	if (size == 1) {
		int v = int(fbColor);
		return ivec4(v, v, v, 0xE0);
	}
	if (size == 3) {
		return unpackColor(int(fbColor));
	}
	uint p = fbColor;
	int cr = int(p >> 11) & 31, cg = int(p >> 6) & 31, cb = int(p >> 1) & 31;
	return ivec4(cr << 3 | cr >> 2, cg << 3 | cg >> 2, cb << 3 | cb >> 2, (p & 1u) != 0u ? 255 : 0);
}

void writePixel(int x, int y, ivec4 c) {
	int size = u[8];
	colorDirty = true;
	if (size == 1) {
		fbColor = uint(c.r) & 255u;
	} else if (size == 3) {
		fbColor = uint(c.r & 255) << 24 | uint(c.g & 255) << 16 | uint(c.b & 255) << 8 | uint(c.a & 255);
	} else {
		fbColor = uint((c.r >> 3) << 11 | (c.g >> 3) << 6 | (c.b >> 3) << 1 | (c.a >= 128 ? 1 : 0));
	}
}

// ---- Textures

uint tmemByte(int addr) {
	int a = gTmem * 4096 + (addr & 0xFFF);
	return (tmem[a >> 2] >> (uint(a & 3) * 8u)) & 255u;
}

int tileField(int t, int f) {
	return data[gS + S_TILES + t * TILE_SIZE + f];
}

const int T_FORMAT = 0, T_SIZE = 1, T_LINE = 2, T_TMEM = 3, T_PALETTE = 4, T_CMS = 5, T_CMT = 6, T_MASKS = 7;
const int T_MASKT = 8, T_SHIFTS = 9, T_SHIFTT = 10, T_SL = 11, T_TL = 12, T_SH = 13, T_TH = 14;

ivec4 tlutColor(int index) {
	int a = 0x800 + index * 8;
	int v = int(tmemByte(a) << 8 | tmemByte(a + 1));
	if (((omHi >> 14) & 1u) != 0u) {
		return ivec4(v >> 8, v >> 8, v >> 8, v & 255);
	}
	int cr = (v >> 11) & 31, cg = (v >> 6) & 31, cb = (v >> 1) & 31;
	return ivec4(cr << 3 | cr >> 2, cg << 3 | cg >> 2, cb << 3 | cb >> 2, (v & 1) != 0 ? 255 : 0);
}

ivec4 texel(int t, int s, int tt) {
	int base = tileField(t, T_TMEM) * 8 + tt * tileField(t, T_LINE) * 8;
	int swap = (tt & 1) != 0 ? 4 : 0;
	bool tlut = ((omHi >> 15) & 1u) != 0u;
	int format = tileField(t, T_FORMAT);
	int size = tileField(t, T_SIZE);
	if (size == 0) {
		int a = (base + (s >> 1)) ^ swap;
		int v = int((s & 1) != 0 ? tmemByte(a) & 15u : tmemByte(a) >> 4);
		if (format == 2 || tlut) {
			return tlutColor(tileField(t, T_PALETTE) << 4 | v);
		}
		if (format == 3) {
			int i = (v >> 1) * 255 / 7;
			return ivec4(i, i, i, (v & 1) != 0 ? 255 : 0);
		}
		int i = v * 17;
		return ivec4(i, i, i, i);
	}
	if (size == 1) {
		int v = int(tmemByte((base + s) ^ swap));
		if (format == 2 || tlut) {
			return tlutColor(v);
		}
		if (format == 3) {
			return ivec4((v >> 4) * 17, (v >> 4) * 17, (v >> 4) * 17, (v & 15) * 17);
		}
		return ivec4(v, v, v, v);
	}
	if (size == 2) {
		int a = (base + s * 2) ^ swap;
		int v = int(tmemByte(a) << 8 | tmemByte(a + 1));
		if (format == 3) {
			return ivec4(v >> 8, v >> 8, v >> 8, v & 255);
		}
		if (tlut) {
			return tlutColor(v >> 8);
		}
		int cr = (v >> 11) & 31, cg = (v >> 6) & 31, cb = (v >> 1) & 31;
		return ivec4(cr << 3 | cr >> 2, cg << 3 | cg >> 2, cb << 3 | cb >> 2, (v & 1) != 0 ? 255 : 0);
	}
	int a = (base & 0x7FF) + s * 2;
	a ^= swap;
	return ivec4(int(tmemByte(a)), int(tmemByte(a + 1)), int(tmemByte(a + 0x800)), int(tmemByte(a + 0x801)));
}

int wrapCoord(int c, int shift, int lo, int hi, int cm, int mask, out int frac) {
	if (shift < 11) {
		c >>= shift;
	} else {
		c <<= 16 - shift;
	}
	c -= lo << 3;
	bool clampIt = (cm & 2) != 0 || mask == 0;
	if (clampIt) {
		int maxC = (hi - lo) << 3;
		if (c < 0) {
			c = 0;
		} else if (c > maxC) {
			c = maxC;
		}
	}
	frac = c & 31;
	int i = c >> 5;
	if (mask != 0) {
		if ((cm & 1) != 0 && ((i >> mask) & 1) != 0) {
			i = ~i;
		}
		i &= (1 << mask) - 1;
	}
	return i;
}

int wrapS(int t, int s, out int frac) {
	return wrapCoord(s, tileField(t, T_SHIFTS), tileField(t, T_SL), tileField(t, T_SH), tileField(t, T_CMS), tileField(t, T_MASKS), frac);
}

int wrapT(int t, int v, out int frac) {
	return wrapCoord(v, tileField(t, T_SHIFTT), tileField(t, T_TL), tileField(t, T_TH), tileField(t, T_CMT), tileField(t, T_MASKT), frac);
}

// C's division by 32 (toward zero).
ivec4 div32(ivec4 v) {
	return (v + ((v >> 31) & 31)) >> 5;
}

// The texels go through one texel() call (a loop the compiler leaves alone): the shader stays small.
ivec4 sampleTile(int t, int s, int tt) {
	t &= 7;
	int fs, ft, f;
	int s0 = wrapS(t, s, fs);
	int t0 = wrapT(t, tt, ft);
	bool bilinear = ((omHi >> 13) & 1u) != 0u && ((omHi >> 20) & 3u) < 2u;
	int s1 = s0, t1 = t0;
	if (bilinear) {
		int shiftS = tileField(t, T_SHIFTS), shiftT = tileField(t, T_SHIFTT);
		s1 = wrapS(t, s + (32 << (shiftS < 11 ? shiftS : 0)), f);
		t1 = wrapT(t, tt + (32 << (shiftT < 11 ? shiftT : 0)), f);
	}
	bool upper = bilinear && fs + ft >= 32;
	int first = upper ? 1 : 0, end = bilinear ? (upper ? 4 : 3) : 1;
	ivec4 c0 = ivec4(0), c1 = ivec4(0), c2 = ivec4(0), c3 = ivec4(0);
	for (int k = first; k < end; ++k) {
		ivec4 c = texel(t, k == 1 || k == 3 ? s1 : s0, k >= 2 ? t1 : t0);
		if (k == 0) {
			c0 = c;
		} else if (k == 1) {
			c1 = c;
		} else if (k == 2) {
			c2 = c;
		} else {
			c3 = c;
		}
	}
	if (!bilinear) {
		return c0;
	}
	if (!upper) {
		return c0 + div32((c1 - c0) * fs + (c2 - c0) * ft);
	}
	return c3 + div32((c2 - c3) * (32 - fs) + (c1 - c3) * (32 - ft));
}

// ---- Combiner and blender

// The inputs that aren't colors: one value for all three channels.
int rgbScalar(int sel, int which) {
	if (which == 0 && sel == 6) {
		return 0x100;
	}
	if (which == 1 && sel == 7) {
		return gK4;
	}
	if (which == 2) {
		switch (sel) {
		case 7: return pCombined.a;
		case 8: return pTex0.a;
		case 9: return pTex1.a;
		case 10: return gPrim.a;
		case 11: return pShade.a;
		case 12: return gEnv.a;
		case 13: return pLodFrac;
		case 14: return gPrimLod;
		case 15: return gK5;
		}
	}
	if (which == 3 && sel == 6) {
		return 0x100;
	}
	return 0;
}

ivec3 rgbIn(int sel, int which) {
	switch (sel) {
	case 0: return pCombined.rgb;
	case 1: return pTex0.rgb;
	case 2: return pTex1.rgb;
	case 3: return gPrim.rgb;
	case 4: return pShade.rgb;
	case 5: return gEnv.rgb;
	}
	return ivec3(rgbScalar(sel, which));
}

int alphaIn(int sel, bool multiplier) {
	switch (sel) {
	case 0: return multiplier ? pLodFrac : pCombined.a;
	case 1: return pTex0.a;
	case 2: return pTex1.a;
	case 3: return gPrim.a;
	case 4: return pShade.a;
	case 5: return gEnv.a;
	case 6: return multiplier ? gPrimLod : 0x100;
	}
	return 0;
}

int ext9(int v) {
	v &= 0x1FF;
	return (v & 0x180) == 0x180 ? v - 0x200 : v;
}

int sign9(int v) {
	v &= 0x1FF;
	return v - ((v & 0x100) << 1);
}

ivec3 ext9v(ivec3 v) {
	v &= 0x1FF;
	return v - ivec3(equal(v & 0x180, ivec3(0x180))) * 0x200;
}

ivec3 sign9v(ivec3 v) {
	v &= 0x1FF;
	return v - ((v & 0x100) << 1);
}

int clamp9(int v) {
	v &= 0x1FF;
	return v < 0x100 ? v : v < 0x180 ? 255 : 0;
}

ivec4 combine(int cycle) {
	uint hi = cbHi, lo = cbLo;
	int subA = cycle != 0 ? int(hi >> 5) & 15 : int(hi >> 20) & 15;
	int mul = cycle != 0 ? int(hi) & 31 : int(hi >> 15) & 31;
	int subB = cycle != 0 ? int(lo >> 24) & 15 : int(lo >> 28) & 15;
	int add = cycle != 0 ? int(lo >> 6) & 7 : int(lo >> 15) & 7;
	int aA = cycle != 0 ? int(lo >> 21) & 7 : int(hi >> 12) & 7;
	int aMul = cycle != 0 ? int(lo >> 18) & 7 : int(hi >> 9) & 7;
	int aB = cycle != 0 ? int(lo >> 3) & 7 : int(lo >> 12) & 7;
	int aAdd = cycle != 0 ? int(lo) & 7 : int(lo >> 9) & 7;
	ivec3 a = ext9v(rgbIn(subA, 0)), b = ext9v(rgbIn(subB, 1));
	ivec3 m = sign9v(rgbIn(mul, 2)), d = ext9v(rgbIn(add, 3));
	ivec3 rgb = (((a - b) * m + (d << 8) + 0x80) & 0x1FFFF) >> 8;
	int aa = ext9(alphaIn(aA, false)), ab = ext9(alphaIn(aB, false));
	int am = sign9(alphaIn(aMul, true)), ad = ext9(alphaIn(aAdd, false));
	return ivec4(rgb, (((aa - ab) * am + (ad << 8) + 0x80) >> 8) & 0x1FF);
}

int blendSel(int cycle, int i) {
	return int(omLo >> uint(30 - cycle * 2 - i * 4)) & 3;
}

ivec4 blendColor(ivec4 pixel, ivec4 memory, int cycle) {
	int sp = blendSel(cycle, 0), sa = blendSel(cycle, 1), sm = blendSel(cycle, 2), sb = blendSel(cycle, 3);
	ivec4 P = sp == 0 ? pixel : sp == 1 ? memory : sp == 2 ? gBlend : gFog;
	ivec4 M = sm == 0 ? pixel : sm == 1 ? memory : sm == 2 ? gBlend : gFog;
	int a = sa == 0 ? pixel.a : sa == 1 ? gFog.a : sa == 2 ? pShade.a : 0;
	int b = sb == 0 ? 255 - a : sb == 1 ? memory.a : sb == 2 ? 255 : 0;
	int fa = a >> 3, fb = (b >> 3) + 1;
	ivec4 o = ((P * fa + M * fb) >> 5) & 255;
	o.a = pixel.a;
	return o;
}

ivec4 blendInput(ivec4 pixel, ivec4 memory, int cycle) {
	int sp = blendSel(cycle, 0);
	ivec4 o = sp == 0 ? pixel : sp == 1 ? memory : sp == 2 ? gBlend : gFog;
	o.a = pixel.a;
	return o;
}

void shadePixel(int x, int y) {
	int cycleType = int(omHi >> 20) & 3;
	bool zPrim = ((omLo >> 2) & 1u) != 0u;
	int z = zPrim ? data[gS + S_PRIMZ] : pZ;
	z = clamp(z, 0, 0x3FFFF);
	bool zCompare = ((omLo >> 4) & 1u) != 0u, zUpdate = ((omLo >> 5) & 1u) != 0u;
	bool zbuf = data[gS + S_ZADDR] != 0;
	if (zbuf && zCompare) {
		int old = zDecompress(fbZ);
		int zMode = int(omLo >> 10) & 3;
		bool pass = zMode == 3 ? z <= old + 0x10 && z >= old - 0x10 : z <= old;
		if (!pass) {
			return;
		}
	}
	// One-cycle mode runs the combiner's second-cycle equation.
	pCombined = lastCombined;
	ivec4 c = ivec4(0);
	for (int cycle = cycleType == 1 ? 0 : 1; cycle < 2; ++cycle) {
		c = combine(cycle);
		pCombined = c;
	}
	lastCombined = c;
	shaded = true;
	c = ivec4(clamp9(c.r), clamp9(c.g), clamp9(c.b), clamp9(c.a));
	if (((omLo >> 13) & 1u) != 0u) {
		if (((omLo >> 12) & 1u) == 0u) {
			c.a = 255;
		} else if (c.a < 32) {
			return;
		}
	}
	if ((omLo & 1u) != 0u && c.a < gBlend.a) {
		return;
	}
	ivec4 memory = readPixel(x, y);
	bool force = ((omLo >> 14) & 1u) != 0u;
	// In 2-cycle mode the first cycle always blends; the last blends only with force blend.
	int cycles = cycleType == 1 ? 2 : 1;
	for (int cycle = 0; cycle < cycles; ++cycle) {
		c = cycle < cycles - 1 || force ? blendColor(c, memory, cycle) : blendInput(c, memory, cycle);
	}
	writePixel(x, y, c);
	if (zbuf && zUpdate) {
		fbZ = zCompress(z);
		zDirty = true;
	}
}

// ---- Texture coordinates and level of detail

uvec2 neg64(uvec2 v) {
	return add64(uvec2(~v.x, ~v.y), uvec2(1u, 0u));
}

// (int) (((s >> 16) * 0x8000) / w), C's 64-bit division, for 0 < w < 0x8000.
int perspDivide(uvec2 s, int w) {
	uvec2 t = uvec2((s.x >> 16) | (s.y << 16), uint(int(s.y) >> 16));
	uvec2 n = uvec2(t.x << 15, (t.y << 15) | (t.x >> 17));
	bool negative = int(n.y) < 0;
	if (negative) {
		n = neg64(n);
	}
	uint d = uint(w);
	uint r = n.y - (n.y / d) * d;
	uint t1 = (r << 16) | (n.x >> 16);
	uint q1 = t1 / d, r1 = t1 - q1 * d;
	uint t2 = (r1 << 16) | (n.x & 0xFFFFu);
	uint q = (q1 << 16) + t2 / d;
	return int(negative ? 0u - q : q);
}

ivec2 texCoords(uvec2 sS, uvec2 sT, uvec2 sW, bool persp) {
	if (!persp) {
		return ivec2(bits64(sS, 16), bits64(sT, 16));
	}
	int wv = int(sW.x >> 16) & 0x7FFF;
	if (wv == 0) {
		wv = 1;
	}
	ivec2 o = ivec2(0);
	for (int k = 0; k < 2; ++k) {
		int v = perspDivide(k == 0 ? sS : sT, wv);
		if (k == 0) {
			o.x = v;
		} else {
			o.y = v;
		}
	}
	return o;
}

bool lodFracUsed() {
	bool cycle1 = (cbHi & 31u) == 13u || ((cbLo >> 18) & 7u) == 0u;
	bool cycle0 = ((cbHi >> 15) & 31u) == 13u || ((cbHi >> 9) & 7u) == 0u;
	return cycle1 || (((omHi >> 20) & 3u) == 1u && cycle0);
}

int levelOfDetail(int primTile, int s, int t, int sx, int tx, int sy, int ty, out int t1, out int t2) {
	bool lodEn = ((omHi >> 16) & 1u) != 0u, sharpen = ((omHi >> 17) & 1u) != 0u, detail = ((omHi >> 18) & 1u) != 0u;
	bool clampIt = s < -0x10000 || s > 0xFFFF || t < -0x10000 || t > 0xFFFF || sx < -0x10000 || sx > 0xFFFF ||
	               tx < -0x10000 || tx > 0xFFFF || sy < -0x10000 || sy > 0xFFFF || ty < -0x10000 || ty > 0xFFFF;
	int lod = 0;
	if (!clampIt) {
		lod = max(max(abs(sx - s), abs(tx - t)), max(abs(sy - s), abs(ty - t)));
		lod = lod >= 0x4000 ? (lod & 0x3FFF) | 0x4000 : lod;
	}
	int tile = 0, frac;
	bool magnify = false, distant;
	if ((lod & 0x4000) != 0 || clampIt) {
		distant = true;
		frac = 0xFF;
	} else if (lod < 32) {
		magnify = true;
		distant = gMaxLevel == 0;
		int base = lod < gMinLevel ? gMinLevel : lod;
		frac = !sharpen && !detail ? (distant ? 0xFF : 0) : (base << 3) | (sharpen ? 0x100 : 0);
	} else {
		int v = (lod >> 5) & 0xFF;
		while (v > 1) {
			v >>= 1;
			++tile;
		}
		distant = gMaxLevel == 0 || (lod & 0x6000) != 0 || tile >= gMaxLevel;
		frac = !sharpen && !detail && distant ? 0xFF : ((lod << 3) >> tile) & 0xFF;
	}
	t1 = primTile;
	t2 = primTile + 1;
	if (lodEn) {
		if (distant) {
			tile = gMaxLevel;
		}
		if (!detail) {
			t1 = (primTile + tile) & 7;
			t2 = distant || (!sharpen && magnify) ? t1 : (t1 + 1) & 7;
		} else {
			t1 = (primTile + tile + (magnify ? 0 : 1)) & 7;
			t2 = (primTile + tile + (!distant && !magnify ? 2 : 1)) & 7;
		}
	}
	return frac;
}

// ---- Primitives

void loadState(int p) {
	gS = data[p + P_STATE];
	omLo = uint(data[gS + S_OM_LO]);
	omHi = uint(data[gS + S_OM_HI]);
	cbLo = uint(data[gS + S_CB_LO]);
	cbHi = uint(data[gS + S_CB_HI]);
	gFog = unpackColor(data[gS + S_FOG]);
	gBlend = unpackColor(data[gS + S_BLEND]);
	gPrim = unpackColor(data[gS + S_PRIM]);
	gEnv = unpackColor(data[gS + S_ENV]);
	gPrimLod = data[gS + S_PRIMLOD];
	gMinLevel = data[gS + S_MINLEVEL];
	gK4 = data[gS + S_K4];
	gK5 = data[gS + S_K5];
	gTmem = data[gS + S_TMEM];
	gMaxLevel = data[p + P_MAXLEVEL];
}

void clearPixel() {
	pShade = ivec4(0);
	pTex0 = ivec4(0);
	pTex1 = ivec4(0);
	pCombined = ivec4(0);
	pZ = 0;
	pLodFrac = 0;
}

void fillPixel(int x, int y) {
	uint fill = uint(data[gS + S_FILL]);
	int size = u[8];
	colorDirty = true;
	if (size == 1) {
		fbColor = (fill >> ((3u - (colorAt & 3u)) * 8u)) & 255u;
	} else if (size == 3) {
		fbColor = fill;
	} else {
		fbColor = (x & 1) != 0 ? fill & 0xFFFFu : fill >> 16;
	}
}

int shadeAt(int start, int step, int k) {
	return bits64(add64(load64(start), mul64(k, data[step])), 16);
}

// One primitive at the pixel: the attributes by kind, then one path for texturing, the level of
// detail, and shading.
void drawPrim(int p, int x, int y) {
	int kind = data[p + P_KIND];
	clearPixel();
	int cycleType = int(omHi >> 20) & 3;
	int tile = data[p + P_TILE];
	bool tex = false, lod = false, copy = false;
	ivec2 st = ivec2(0), sx = ivec2(0), sy = ivec2(0);
	if (kind == KIND_FILL) {
		fillPixel(x, y);
		return;
	}
	if (kind == KIND_TRIANGLE) {
		int row = data[p + P_ROWS] + (y - data[p + P_FIRST]) * ROW_SIZE;
		int x0 = data[row], x1 = data[row + 1];
		if (x < x0 || x > x1) {
			return;
		}
		int k = x - x0;
		int flags = data[p + P_FLAGS];
		if ((flags & 4) != 0) {
			pShade = clamp(ivec4(shadeAt(row + 2, p + P_SDX, k), shadeAt(row + 4, p + P_SDX + 1, k), shadeAt(row + 6, p + P_SDX + 2, k),
			                     shadeAt(row + 8, p + P_SDX + 3, k)), 0, 255);
		}
		if ((flags & 1) != 0) {
			pZ = bits64(add64(load64(row + 16), mul64(k, data[p + P_ZDX])), 13);
		}
		if ((flags & 2) != 0) {
			tex = true;
			lod = ((omHi >> 16) & 1u) != 0u || lodFracUsed();
			bool persp = ((omHi >> 19) & 1u) != 0u;
			uvec2 sS = add64(load64(row + 10), mul64(k, data[p + P_TDX]));
			uvec2 sT = add64(load64(row + 12), mul64(k, data[p + P_TDX + 1]));
			uvec2 sW = add64(load64(row + 14), mul64(k, data[p + P_TDX + 2]));
			int count = lod ? 3 : 1;
			for (int j = 0; j < count; ++j) {
				uvec2 a = sS, b = sT, c = sW;
				if (j > 0) {
					int d = j == 1 ? P_TDX : P_TDY;
					a = add64(a, ext64(data[p + d]));
					b = add64(b, ext64(data[p + d + 1]));
					c = add64(c, ext64(data[p + d + 2]));
				}
				ivec2 v = texCoords(a, b, c, persp);
				if (j == 0) {
					st = v;
				} else if (j == 1) {
					sx = v;
				} else {
					sy = v;
				}
			}
		}
	} else if (kind == KIND_TEXRECT) {
		int s0 = data[p + P_S0], t0 = data[p + P_T0], dsdx = data[p + P_DSDX], dtdy = data[p + P_DTDY];
		bool flip = data[p + P_FLIP] != 0;
		int dx = x - data[p + P_CX0], dy = y - data[p + P_CY0];
		int s = ((s0 << 5) + (flip ? dy * dsdx : dx * dsdx)) >> 5;
		int t = ((t0 << 5) + (flip ? dx * dtdy : dy * dtdy)) >> 5;
		st = ivec2(s, t);
		tex = true;
		lod = cycleType < 2 && (((omHi >> 16) & 1u) != 0u || lodFracUsed());
		sx = ivec2(s + (flip ? 0 : dsdx >> 5), t + (flip ? dtdy >> 5 : 0));
		sy = ivec2(s + (flip ? dsdx >> 5 : 0), t + (flip ? 0 : dtdy >> 5));
		copy = cycleType == 2;
		pZ = data[gS + S_PRIMZ];
	} else {
		pZ = data[gS + S_PRIMZ];
	}
	if (tex) {
		int t1 = tile, t2 = tile + 1;
		if (lod) {
			pLodFrac = levelOfDetail(tile, st.x, st.y, sx.x, sx.y, sy.x, sy.y, t1, t2);
		}
		int samples = cycleType == 1 ? 2 : 1;
		for (int i = 0; i < samples; ++i) {
			ivec4 c = sampleTile(i == 0 ? t1 : t2, st.x, st.y);
			if (i == 0) {
				pTex0 = c;
			} else {
				pTex1 = c;
			}
		}
		if (samples == 1) {
			pTex1 = pTex0;
		}
	}
	if (copy) {
		if ((omLo & 1u) != 0u && pTex0.a == 0) {
			return;
		}
		int ti = tile & 7;
		if (data[gS + S_CSIZE] == 1 && tileField(ti, T_SIZE) == 1) {
			// Copying into an 8-bit image moves the texel bytes themselves (palette indices).
			bool flip = data[p + P_FLIP] != 0;
			int dx = x - data[p + P_CX0], dy = y - data[p + P_CY0];
			int f;
			int ss = wrapS(ti, st.x, f);
			int tt = wrapT(ti, ((data[p + P_T0] << 5) + (flip ? dx : dy) * data[p + P_DTDY]) >> 5, f);
			int a = (tileField(ti, T_TMEM) * 8 + tt * tileField(ti, T_LINE) * 8 + ss) ^ ((tt & 1) != 0 ? 4 : 0);
			fbColor = tmemByte(a);
			colorDirty = true;
			return;
		}
		writePixel(x, y, pTex0);
		return;
	}
	shadePixel(x, y);
}

shared uint bestKey;

void main() {
	ivec2 tileXY = ivec2(gl_WorkGroupID.xy) + ivec2(u[2], u[3]);
	int x = tileXY.x * 16 + int(gl_LocalInvocationID.x);
	int y = tileXY.y * 16 + int(gl_LocalInvocationID.y);
	int bin = u[0] + (tileXY.y * TILES_ACROSS + tileXY.x) * 2;
	int first = data[bin], count = data[bin + 1];
	uint key = 0u;
	ivec4 keyColor = ivec4(0);
	if (count > 0) {
		loadPixel(x, y);
	}
	for (int i = 0; i < count; ++i) {
		int index = data[u[1] + first + i];
		int p = index * PRIM_SIZE;
		if (x < data[p + P_X0] || x > data[p + P_X1] || y < data[p + P_Y0] || y > data[p + P_Y1]) {
			continue;
		}
		loadState(p);
		shaded = false;
		drawPrim(p, x, y);
		if (shaded) {
			key = uint(index + 1) << 20 | uint(y) << 10 | uint(x);
			keyColor = lastCombined;
		}
	}
	if (count > 0) {
		storePixel();
	}
	// The last pixel through the combiner, in drawing order, for the CPU's next lastCombined.
	if (gl_LocalInvocationIndex == 0u) {
		bestKey = 0u;
	}
	barrier();
	atomicMax(bestKey, key);
	barrier();
	uint out0 = uint(u[5]) + (gl_WorkGroupID.y * uint(u[4]) + gl_WorkGroupID.x) * 3u;
	if (gl_LocalInvocationIndex == 0u && bestKey == 0u) {
		lastOut[out0] = 0u;
	}
	if (key != 0u && key == bestKey) {
		lastOut[out0] = key;
		lastOut[out0 + 1u] = uint(keyColor.r & 0x1FF) | uint(keyColor.g & 0x1FF) << 9 | uint(keyColor.b & 0x1FF) << 18;
		lastOut[out0 + 2u] = uint(keyColor.a & 0x1FF);
	}
}
