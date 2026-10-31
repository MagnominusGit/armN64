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

// What the batch's primitives use (rdp.c gpuFeatures): the compiler leaves out the rest. Without
// these defines the shader draws everything.
#ifndef F_TRI
#define F_TRI 1
#define F_TEXRECT 1
#define F_FILL 1
#define F_RECT 1
#define F_SHADE 1
#define F_ZTRI 1
#define F_TEX 1
#define F_LOD 1
#define F_PERSP 1
#define F_TWO 1
#define F_COPY 1
#define F_BILERP 1
#define F_CONV 1
#define F_ZBUF 1
#define F_TLUT 1
#define F_CSIZE 0
#endif
const bool HAS_TRI = F_TRI != 0, HAS_TEXRECT = F_TEXRECT != 0, HAS_FILL = F_FILL != 0, HAS_RECT = F_RECT != 0;
const bool HAS_SHADE = F_SHADE != 0, HAS_ZTRI = F_ZTRI != 0, HAS_TEX = F_TEX != 0, HAS_LOD = F_LOD != 0;
const bool HAS_PERSP = F_PERSP != 0, HAS_TWO = F_TWO != 0, HAS_COPY = F_COPY != 0, HAS_BILERP = F_BILERP != 0;
const bool HAS_CONV = F_CONV != 0, HAS_ZBUF = F_ZBUF != 0, HAS_TLUT = F_TLUT != 0;

// The color image's pixel size: fixed for the variant, else the batch's.
int colorSize() {
	return F_CSIZE != 0 ? F_CSIZE : u[8];
}

const uint RDRAM_MASK = 0x7FFFFFu;
const int TILES_ACROSS = 64;

// Primitive kinds and record fields (rdp.c GPU_*).
const int KIND_TRIANGLE = 0, KIND_TEXRECT = 1, KIND_FILL = 2, KIND_RECT = 3;
const int P_KIND = 0, P_STATE = 1, P_X0 = 2, P_Y0 = 3, P_X1 = 4, P_Y1 = 5, P_TILE = 6, P_MAXLEVEL = 7;
const int P_TRI = 8, P_FLAGS = 10, P_SDX = 11, P_TDX = 15, P_TDY = 18, P_ZDX = 21;
const int P_S0 = 8, P_T0 = 9, P_DSDX = 10, P_DTDY = 11, P_CX0 = 12, P_CY0 = 13, P_FLIP = 14;
const int PRIM_SIZE = 24;
// A triangle's record (rdp.c gpuTriangle): its edges, the scissor, and the shade, texture and depth
// values at its top.
const int T_YL = 0, T_YM = 1, T_YH = 2, T_XL = 3, T_DXL = 4, T_XH = 5, T_DXH = 6, T_XM = 7, T_DXM = 8, T_SCISSOR = 9;
const int T_SV = 13, T_SDE = 17, T_TV = 21, T_TDE = 24, T_ZV = 27, T_ZDE = 28, T_SDY = 29, T_ZDY = 33, T_FLIP = 34;
const int S_OM_LO = 0, S_OM_HI = 1, S_CB_LO = 2, S_CB_HI = 3, S_CADDR = 4, S_CWIDTH = 5, S_CSIZE = 6, S_ZADDR = 7;
const int S_FOG = 8, S_BLEND = 9, S_PRIM = 10, S_ENV = 11, S_PRIMLOD = 12, S_PRIMZ = 13, S_MINLEVEL = 14;
const int S_K4 = 15, S_K5 = 16, S_FILL = 17, S_TMEM = 18, S_TILES = 19, TILE_SIZE = 15, S_KCONV = 19 + 8 * 15;


// ---- The primitive being drawn
int gS;
uint omLo, omHi, cbLo, cbHi;
ivec4 gFog, gBlend, gPrim, gEnv;
int gPrimLod, gK4, gK5, gTmem, gMaxLevel, gMinLevel;
ivec4 gKConv;

// ---- The pixel
ivec4 pShade, pTex0, pTex1, pCombined;
int pZ, pLodFrac, pCvg, pDz, pPrimDz;
bool pCvbit;
int blShiftA, blShiftB;
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

bool less64(uvec2 a, uvec2 b) {
	return int(a.y) < int(b.y) || (a.y == b.y && a.x < b.x);
}

// v >> sh with sign, for 0 < sh < 32.
uvec2 shift64(uvec2 v, int sh) {
	return uvec2((v.x >> uint(sh)) | (v.y << uint(32 - sh)), uint(int(v.y) >> sh));
}

// C's v / 4 (toward zero).
uvec2 quarter64(uvec2 v) {
	return shift64(int(v.y) < 0 ? add64(v, uvec2(3u, 0u)) : v, 2);
}

// a * b, the low 64 bits.
uvec2 mulWide(int a, uvec2 b) {
	uint hi, lo;
	umulExtended(uint(a), b.x, hi, lo);
	if (a < 0) {
		hi -= b.x;
	}
	return uvec2(lo, hi + uint(a) * b.y);
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

uint fbColor, fbZ, fbHidden, fbZHidden;
bool colorDirty, zDirty;
uint colorAt, zAt;

// RDRAM's hidden bits follow it in the buffer, a byte for each halfword.
uint hiddenAt() {
	return RDRAM_MASK + 1u + (colorAt >> 1);
}

uint rhiddenAt(uint a) {
	return (rdram[a >> 2] >> ((a & 3u) * 8u)) & 3u;
}

void whiddenAt(uint a, uint v) {
	uint s = (a & 3u) * 8u;
	atomicAnd(rdram[a >> 2], ~(255u << s));
	atomicOr(rdram[a >> 2], (v & 3u) << s);
}

uint rhidden() {
	return rhiddenAt(hiddenAt());
}

void whidden(uint v) {
	whiddenAt(hiddenAt(), v);
}

uint rword(uint a) {
	return rbyte(a) << 24 | rbyte(a + 1u) << 16 | rbyte(a + 2u) << 8 | rbyte(a + 3u);
}

void loadPixel(int x, int y) {
	uint i = uint(y * u[7] + x);
	uint addr = uint(u[6]);
	int size = colorSize();
	if (size == 1) {
		colorAt = (addr + i) & RDRAM_MASK;
		fbColor = rbyte(colorAt);
	} else if (size == 3) {
		colorAt = (addr + i * 4u) & (RDRAM_MASK & ~3u);
		fbColor = rword(colorAt);
	} else {
		colorAt = (addr + i * 2u) & (RDRAM_MASK & ~1u);
		fbColor = rhalf(colorAt);
		fbHidden = rhidden();
	}
	zAt = (uint(u[9]) + i * 2u) & (RDRAM_MASK & ~1u);
	fbZ = HAS_ZBUF && u[9] != 0 ? rhalf(zAt) : 0u;
	fbZHidden = HAS_ZBUF && u[9] != 0 ? rhiddenAt(RDRAM_MASK + 1u + (zAt >> 1)) : 0u;
	colorDirty = zDirty = false;
}

void storePixel() {
	if (colorDirty) {
		int size = colorSize();
		if (size == 1) {
			wbyte(colorAt, fbColor);
		} else if (size == 3) {
			uint v = fbColor;
			rdram[colorAt >> 2] = (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
		} else {
			whalf(colorAt, fbColor);
			whidden(fbHidden);
		}
	}
	if (HAS_ZBUF && zDirty) {
		whalf(zAt, fbZ);
		whiddenAt(RDRAM_MASK + 1u + (zAt >> 1), fbZHidden);
	}
}

ivec4 readPixel(int x, int y) {
	int size = colorSize();
	if (size == 1) {
		int v = int(fbColor);
		return ivec4(v, v, v, 0xE0);
	}
	if (size == 3) {
		ivec4 c = unpackColor(int(fbColor));
		c.a &= 0xE0;
		return c;
	}
	uint p = fbColor;
	int cr = int(p >> 11) & 31, cg = int(p >> 6) & 31, cb = int(p >> 1) & 31;
	return ivec4(cr << 3, cg << 3, cb << 3, int((p & 1u) << 2 | fbHidden) << 5);
}

void writePixel(int x, int y, ivec4 c) {
	int size = colorSize();
	colorDirty = true;
	if (size == 1) {
		fbColor = uint(c.r) & 255u;
	} else if (size == 3) {
		fbColor = uint(c.r & 255) << 24 | uint(c.g & 255) << 16 | uint(c.b & 255) << 8 | uint(c.a & 255);
	} else {
		fbColor = uint((c.r >> 3) << 11 | (c.g >> 3) << 6 | (c.b >> 3) << 1 | (c.a >= 128 ? 1 : 0));
		fbHidden = uint(c.a >> 5) & 3u;
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
	bool tlut = HAS_TLUT && ((omHi >> 15) & 1u) != 0u;
	int format = tileField(t, T_FORMAT);
	int size = tileField(t, T_SIZE);
	if (size == 0) {
		int a = (base + (s >> 1)) ^ swap;
		int v = int((s & 1) != 0 ? tmemByte(a) & 15u : tmemByte(a) >> 4);
		if (format == 2 || tlut) {
			return tlutColor(tileField(t, T_PALETTE) << 4 | v);
		}
		if (format == 3) {
			int i = (v >> 1) << 5 | (v >> 1) << 2 | (v >> 2);
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
		if (format == 1) {
			int a = (base & 0x7FF) + s;
			int y = int(tmemByte((a ^ swap) | 0x800));
			int u = int(tmemByte(((a & ~1) ^ swap) & 0x7FF)), v = int(tmemByte(((a | 1) ^ swap) & 0x7FF));
			return ivec4(u - 128, v - 128, y, y);
		}
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

// rdp.c texClamp, texMask and sampleAs (after paraLLEl-RDP's texture.h, MIT).
int texClamp(int c, bool clampIt, int lo, int hi, int shift) {
	c = clamp(c, -0x8000, 0x7FFF);
	c = shift < 11 ? c >> shift : (c << (32 - shift)) >> 16;
	if (!clampIt) {
		return c - (lo << 3);
	}
	if ((c >> 3) >= hi) {
		return (((hi >> 2) - (lo >> 2)) & 0x3FF) << 5;
	}
	return max(c - (lo << 3), 0);
}

int texMask(int c, int mask, bool mirror) {
	if (mask != 0) {
		int m = 1 << min(mask, 10);
		if (mirror && (c & m) != 0) {
			c ^= m - 1;
		}
		c &= m - 1;
	}
	return c;
}

// The texels go through few texel() calls (a loop the compiler leaves alone): the shader stays small.
ivec4 sampleTile(int t, int s, int tt, bool bilerp) {
	t &= 7;
	bool quad = ((omHi >> 13) & 1u) != 0u, tlut = ((omHi >> 15) & 1u) != 0u, midMode = ((omHi >> 8) & 1u) != 0u;
	bool filter = HAS_BILERP && bilerp && ((omHi >> 20) & 3u) < 2u;
	int cmS = tileField(t, T_CMS), cmT = tileField(t, T_CMT), maskS = tileField(t, T_MASKS), maskT = tileField(t, T_MASKT);
	int cs = texClamp(s, (cmS & 2) != 0 || maskS == 0, tileField(t, T_SL), tileField(t, T_SH), tileField(t, T_SHIFTS));
	int ct = texClamp(tt, (cmT & 2) != 0 || maskT == 0, tileField(t, T_TL), tileField(t, T_TH), tileField(t, T_SHIFTT));
	int fs = quad || tlut ? cs & 31 : 0, ft = quad || tlut ? ct & 31 : 0;
	int sum = fs + ft;
	cs >>= 5;
	ct >>= 5;
	int s0 = texMask(cs, maskS, (cmS & 1) != 0), s1 = texMask(cs + 1, maskS, (cmS & 1) != 0);
	int t0 = texMask(ct, maskT, (cmT & 1) != 0), t1 = texMask(ct + 1, maskT, (cmT & 1) != 0);
	int tdiff = max(t1 - t0, -255);
	t1 = (t0 & 0xFF) + tdiff;
	t0 &= 0xFF;
	bool mid = midMode && filter && fs == 16 && ft == 16;
	if (mid) {
		sum = 0;
	}
	if (tlut && !quad) {
		s1 = s0, t1 = t0;
	}
	bool upper = sum >= 32 && quad;
	bool lerp = filter && quad;
	int count = mid ? 4 : lerp ? 3 : 1;
	ivec4 b = ivec4(0), t10 = ivec4(0), t01 = ivec4(0), t11 = ivec4(0);
	for (int k = 0; k < count; ++k) {
		int ks = k == 0 ? (upper ? s1 : s0) : k == 1 || k == 3 ? s1 : s0;
		int kt = k == 0 ? (upper ? t1 : t0) : k >= 2 ? t1 : t0;
		ivec4 c = texel(t, ks, kt);
		if (k == 0) {
			b = c;
		} else if (k == 1) {
			t10 = c;
		} else if (k == 2) {
			t01 = c;
		} else {
			t11 = c;
		}
	}
	if (mid) {
		return (b + t10 + t01 + t11 + 2) >> 2;
	}
	if (!lerp) {
		return b;
	}
	int fx = upper ? 32 - ft : fs, fy = upper ? 32 - fs : ft;
	return (((t10 - b) * fx + (t01 - b) * fy + 0x10) >> 5) + b;
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

// The texture convert unit (rdp.c convert): U, V, Y in red, green, blue to RGB.
ivec4 convertYuv(ivec4 c) {
	int red = c.b + ((gKConv.x * c.g + 0x80) >> 8);
	int green = c.b + ((gKConv.y * c.r + gKConv.z * c.g + 0x80) >> 8);
	int blue = c.b + ((gKConv.w * c.r + 0x80) >> 8);
	return ivec4(ext9(red), ext9(green), ext9(blue), ext9(c.b));
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

// The blender's divider: the hardware's bit-serial division of a sum of products by the sum of its
// 4-bit factors.
int blendDivide(int i) {
	int d = (i >> 11) & 15, n = i & 0x7FF, inv = ~d & 15, res = 0;
	int partial = (inv + (n >> 8) + 1) & 7;
	for (int k = 0; k < 8; ++k) {
		int bit = (n >> (7 - k)) & 1;
		int t = (res & (0x100 >> k)) != 0 ? inv + (partial << 1) + bit + 1 : d + (partial << 1) + bit;
		partial = t & 7;
		if ((t & 0x10) != 0) {
			res |= 1 << (7 - k);
		}
	}
	return res;
}

ivec4 blendColor(ivec4 pixel, ivec4 memory, int cycle, bool divide) {
	int sp = blendSel(cycle, 0), sa = blendSel(cycle, 1), sm = blendSel(cycle, 2), sb = blendSel(cycle, 3);
	ivec4 P = sp == 0 ? pixel : sp == 1 ? memory : sp == 2 ? gBlend : gFog;
	ivec4 M = sm == 0 ? pixel : sm == 1 ? memory : sm == 2 ? gBlend : gFog;
	int a = sa == 0 ? pixel.a : sa == 1 ? gFog.a : sa == 2 ? pShade.a : 0;
	int b = sb == 0 ? 255 - a : sb == 1 ? memory.a : sb == 2 ? 255 : 0;
	int fa = a >> 3, fb = b >> 3;
	if (sb == 1) {
		fa = (fa >> blShiftA) & 0x3C;
		fb = (fb >> blShiftB) | 3;
	}
	ivec4 sum = P * fa + M * (fb + 1);
	ivec4 o;
	if (divide) {
		int d = ((fa & ~3) + (fb & ~3) + 4) << 9;
		o = ivec4(blendDivide(d | ((sum.r >> 2) & 0x7FF)), blendDivide(d | ((sum.g >> 2) & 0x7FF)), blendDivide(d | ((sum.b >> 2) & 0x7FF)), 0);
	} else {
		o = (sum >> 5) & 255;
	}
	o.a = pixel.a;
	return o;
}

ivec4 blendInput(ivec4 pixel, ivec4 memory, int cycle) {
	int sp = blendSel(cycle, 0);
	ivec4 o = sp == 0 ? pixel : sp == 1 ? memory : sp == 2 ? gBlend : gFog;
	o.a = pixel.a;
	return o;
}

int normalizeHighest(int v) {
	return v != 0 ? 1 << findMSB(v) : 0;
}

int dzEncode(int v) {
	return ((v & 0xFF00) != 0 ? 8 : 0) | ((v & 0xF0F0) != 0 ? 4 : 0) | ((v & 0xCCCC) != 0 ? 2 : 0) | ((v & 0xAAAA) != 0 ? 1 : 0);
}

// The magic square and Bayer dither matrices (paraLLEl-RDP's dither.h, MIT). Noise dither is drawn
// on the CPU (rdp.c usesNoise).
const int ditherMatrix[32] = int[32](0, 6, 1, 7, 4, 2, 5, 3, 3, 5, 2, 4, 7, 1, 6, 0, 0, 4, 1, 5, 4, 0, 5, 1, 3, 7, 2, 6, 7, 3, 6, 2);

void ditherAt(int x, int y, out int rgb, out int alpha) {
	int modeRgb = int(omHi >> 6) & 3, modeAlpha = int(omHi >> 4) & 3;
	int i = (y & 3) * 4 + (x & 3);
	rgb = modeRgb < 2 ? ditherMatrix[modeRgb * 16 + i] * 0x49 : 0;
	if (modeAlpha >= 2) {
		alpha = 0;
	} else {
		alpha = modeRgb >= 2 ? ditherMatrix[(modeRgb & 1) * 16 + i] : rgb & 7;
		alpha = modeAlpha == 1 ? ~alpha & 7 : alpha;
	}
}

void shadePixel(int x, int y) {
	int cycleType = int(omHi >> 20) & 3;
	bool zPrim = ((omLo >> 2) & 1u) != 0u;
	int z = zPrim ? data[gS + S_PRIMZ] : pZ;
	z = clamp(z, 0, 0x3FFFF);
	bool zCompare = ((omLo >> 4) & 1u) != 0u, zUpdate = ((omLo >> 5) & 1u) != 0u;
	bool zbuf = data[gS + S_ZADDR] != 0;
	// One-cycle mode runs the combiner's second-cycle equation.
	pCombined = lastCombined;
	ivec4 c = ivec4(0);
	int firstAlpha = 0;
	for (int cycle = HAS_TWO && cycleType == 1 ? 0 : 1; cycle < 2; ++cycle) {
		c = combine(cycle);
		pCombined = c;
		if (cycle == 0) {
			firstAlpha = c.a;
			// rdp.c: the second cycle sees the pipelined texels.
			ivec4 swap = pTex0;
			pTex0 = pTex1;
			pTex1 = swap;
		}
	}
	lastCombined = c;
	shaded = true;
	c = ivec4(clamp9(c.r), clamp9(c.g), clamp9(c.b), clamp9(c.a));
	bool antialias = ((omLo >> 3) & 1u) != 0u, cvgTimesAlpha = ((omLo >> 12) & 1u) != 0u;
	bool imageRead = ((omLo >> 6) & 1u) != 0u, colorOnCvg = ((omLo >> 7) & 1u) != 0u, force = ((omLo >> 14) & 1u) != 0u;
	bool cvgAlpha = ((omLo >> 13) & 1u) != 0u;
	int rgbDither, alphaDither;
	ditherAt(x, y, rgbDither, alphaDither);
	int reference = 0;
	if (HAS_TWO && cycleType == 1 && (omLo & 1u) != 0u) {
		int a0 = clamp9(firstAlpha);
		a0 += (a0 + 1) >> 8;
		a0 = cvgAlpha ? (cvgTimesAlpha ? (a0 * pCvg + 4) >> 3 : pCvg << 5) : a0 + alphaDither;
		reference = clamp(a0, 0, 255);
	}
	int alpha = c.a + ((c.a + 1) >> 8);
	int cvg = pCvg, scaled = cvg << 5;
	if (cvgTimesAlpha) {
		scaled = (alpha * cvg + 4) >> 3;
		cvg = scaled >> 5;
	}
	alpha = cvgAlpha ? scaled : alpha + alphaDither;
	c.a = clamp(alpha, 0, 255);
	if (!(HAS_TWO && cycleType == 1)) {
		reference = c.a;
	}
	pShade.a = min(pShade.a + alphaDither, 255);
	ivec4 memory = readPixel(x, y);
	int memCvg = imageRead ? memory.a >> 5 : 7;
	if (!imageRead) {
		memory.a = 0xE0;
	}
	int dzpix = zPrim ? pPrimDz : pDz;
	int dzEnc = dzEncode(dzpix);
	bool overflow = ((cvg + memCvg) & 8) != 0;
	bool blend;
	if (HAS_ZBUF && zbuf && zCompare) {
		int oz = zDecompress(fbZ);
		int rawDzMem = int((fbZ & 3u) << 2 | fbZHidden);
		int dzMem = 1 << rawDzMem;
		blShiftA = clamp(dzEnc - rawDzMem, 0, 4);
		blShiftB = clamp(rawDzMem - dzEnc, 0, 4);
		bool coplanar = false;
		int zPrecision = int(fbZ >> 13) & 15;
		if (zPrecision < 3) {
			if (dzMem != 0x8000) {
				dzMem = max(dzMem << 1, 16 >> zPrecision);
			} else {
				coplanar = true;
				dzMem = 0xFFFF;
			}
		}
		int dzNew = normalizeHighest((dzpix | dzMem) & 0xFFFF);
		int dzShifted = dzNew << 3;
		bool farther = coplanar || z + dzShifted >= oz;
		bool infront = z < oz, nearer = coplanar || z - dzShifted <= oz, isMax = oz == 0x3FFFF;
		blend = force || (!overflow && antialias && farther);
		int zMode = int(omLo >> 10) & 3;
		bool pass;
		if (zMode == 0) {
			pass = isMax || (overflow ? infront : nearer);
		} else if (zMode == 1) {
			if (!infront || !farther || !overflow) {
				pass = isMax || (overflow ? infront : nearer);
			} else {
				int enc = dzEncode(dzNew & 0xFFFF);
				cvg = ((((oz >> enc) - (z >> enc)) & 15) * cvg >> 3) & 15;
				pass = true;
			}
		} else if (zMode == 2) {
			pass = infront || isMax;
		} else {
			pass = farther && nearer && !isMax;
		}
		if (!pass) {
			return;
		}
	} else {
		blShiftA = 0;
		blShiftB = dzEnc < 11 ? 4 : 15 - dzEnc;
		blend = force || (!overflow && antialias);
	}
	if ((omLo & 1u) != 0u && reference < gBlend.a) {
		return;
	}
	if (antialias ? cvg == 0 : !pCvbit) {
		return;
	}
	int last = HAS_TWO && cycleType == 1 ? 1 : 0;
	bool partialReject = blendSel(last, 1) == 0 && blendSel(last, 3) == 0 && c.a >= 255;
	ivec4 inColor = c;
	if (HAS_TWO && cycleType == 1) {
		inColor = blendColor(c, memory, 0, false);
	}
	if (colorOnCvg && !overflow) {
		int sm = blendSel(last, 2);
		c = sm == 0 ? inColor : sm == 1 ? memory : sm == 2 ? gBlend : gFog;
	} else if (!blend || partialReject) {
		c = blendInput(inColor, memory, last);
	} else {
		c = blendColor(inColor, memory, last, !force);
	}
	if (((omHi >> 6) & 3u) != 3u) {
		for (int ch = 0; ch < 3; ++ch) {
			int v = c[ch];
			if (((rgbDither >> (ch * 3)) & 7) < (v & 7)) {
				c[ch] = v > 247 ? 255 : (v & 0xF8) + 8;
			}
		}
	}
	int cvgDest = int(omLo >> 8) & 3;
	int kept = cvgDest == 0 ? (blend ? cvg + memCvg : cvg - 1) : cvgDest == 1 ? (cvg + memCvg) & 7 : cvgDest == 2 ? 7 : memCvg;
	c.a = ((kept & 8) != 0 ? 7 : kept & 7) << 5;
	writePixel(x, y, c);
	if (HAS_ZBUF && zbuf && zUpdate) {
		fbZ = zCompress(z) | uint(dzEnc >> 2);
		fbZHidden = uint(dzEnc & 3);
		zDirty = true;
	}
}

// ---- Texture coordinates and level of detail

uvec2 neg64(uvec2 v) {
	return add64(uvec2(~v.x, ~v.y), uvec2(1u, 0u));
}

const ivec2 perspectiveTable[64] = ivec2[64](
	ivec2(0x4000, -252 * 4), ivec2(0x3f04, -244 * 4), ivec2(0x3e10, -238 * 4), ivec2(0x3d22, -230 * 4), ivec2(0x3c3c, -223 * 4), ivec2(0x3b5d, -218 * 4),
	ivec2(0x3a83, -210 * 4), ivec2(0x39b1, -205 * 4), ivec2(0x38e4, -200 * 4), ivec2(0x381c, -194 * 4), ivec2(0x375a, -189 * 4), ivec2(0x369d, -184 * 4),
	ivec2(0x35e5, -179 * 4), ivec2(0x3532, -175 * 4), ivec2(0x3483, -170 * 4), ivec2(0x33d9, -166 * 4), ivec2(0x3333, -162 * 4), ivec2(0x3291, -157 * 4),
	ivec2(0x31f4, -155 * 4), ivec2(0x3159, -150 * 4), ivec2(0x30c3, -147 * 4), ivec2(0x3030, -143 * 4), ivec2(0x2fa1, -140 * 4), ivec2(0x2f15, -137 * 4),
	ivec2(0x2e8c, -134 * 4), ivec2(0x2e06, -131 * 4), ivec2(0x2d83, -128 * 4), ivec2(0x2d03, -125 * 4), ivec2(0x2c86, -123 * 4), ivec2(0x2c0b, -120 * 4),
	ivec2(0x2b93, -117 * 4), ivec2(0x2b1e, -115 * 4), ivec2(0x2aab, -113 * 4), ivec2(0x2a3a, -110 * 4), ivec2(0x29cc, -108 * 4), ivec2(0x2960, -106 * 4),
	ivec2(0x28f6, -104 * 4), ivec2(0x288e, -102 * 4), ivec2(0x2828, -100 * 4), ivec2(0x27c4, -98 * 4),  ivec2(0x2762, -96 * 4),  ivec2(0x2702, -94 * 4),
	ivec2(0x26a4, -92 * 4),  ivec2(0x2648, -91 * 4),  ivec2(0x25ed, -89 * 4),  ivec2(0x2594, -87 * 4),  ivec2(0x253d, -86 * 4),  ivec2(0x24e7, -85 * 4),
	ivec2(0x2492, -83 * 4),  ivec2(0x243f, -81 * 4),  ivec2(0x23ee, -80 * 4),  ivec2(0x239e, -79 * 4),  ivec2(0x234f, -77 * 4),  ivec2(0x2302, -76 * 4),
	ivec2(0x22b6, -74 * 4),  ivec2(0x226c, -74 * 4),  ivec2(0x2222, -72 * 4),  ivec2(0x21da, -71 * 4),  ivec2(0x2193, -70 * 4),  ivec2(0x214d, -69 * 4),
	ivec2(0x2108, -67 * 4),  ivec2(0x20c5, -67 * 4),  ivec2(0x2082, -65 * 4),  ivec2(0x2041, -65 * 4));

// rdp.c texCoords: s and t (the top 16 bits), with perspective times a table's reciprocal of w.
ivec2 texCoords(int sS, int sT, int sW, bool persp) {
	ivec2 st = ivec2(sS >> 16, sT >> 16);
	if (HAS_PERSP && persp) {
		int w = sW >> 16;
		bool carry = w <= 0;
		w &= 0x7FFF;
		int shift = min(14 - findMSB(w), 14);
		int norm = (w << shift) & 0x3FFF;
		ivec2 e = perspectiveTable[norm >> 8];
		int rcp = ((e.y * (norm & 0xFF)) >> 10) + e.x;
		int mask = ((1 << 30) - 1) & -((1 << 29) >> shift);
		for (int i = 0; i < 2; ++i) {
			int prod = st[i] * rcp;
			int outside = prod & mask;
			int v = shift != 14 ? prod >> (13 - shift) : prod << 1;
			if (outside != mask && outside != 0) {
				int shifted = shift != 14 ? prod >> (13 - shift) : prod;
				v = (shifted & (1 << 29)) == 0 ? 0x7FFF : -0x8000;
			}
			st[i] = carry ? 0x7FFF : v;
		}
	}
	return clamp(st, ivec2(-0x10000), ivec2(0xFFFF));
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
	gKConv = ivec4(data[gS + S_KCONV], data[gS + S_KCONV + 1], data[gS + S_KCONV + 2], data[gS + S_KCONV + 3]);
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
	pCvg = 8;
	pCvbit = true;
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
		fbHidden = (fbColor & 1u) != 0u ? 3u : 0u;
	}
}

// rdp.c's span, coverage and attribute stepping (after paraLLEl-RDP, MIT).
int snapX(int x) {
	return (x >> 12) | ((x & 0xFFF) != 0 ? 1 : 0);
}

int edgeX(int t, int f) {
	return bitfieldExtract(data[t + f], 0, 28) >> 1;
}

int edgeSlope(int t, int f) {
	return bitfieldExtract(int(uint(data[t + f]) >> 2), 0, 28) >> 1;
}

// A triangle's span on row y: its four sub-scanlines' left | right << 16 in eighths of a pixel
// (0xFFFF where it has none).
ivec4 rowSpan(int p, int y) {
	int t = data[p + P_TRI];
	int yl = data[t + T_YL], ym = data[t + T_YM], yh = data[t + T_YH];
	int xh0 = edgeX(t, T_XH), xm0 = edgeX(t, T_XM), xl0 = edgeX(t, T_XL);
	int dxh = edgeSlope(t, T_DXH), dxm = edgeSlope(t, T_DXM), dxl = edgeSlope(t, T_DXL);
	bool flip = data[t + T_FLIP] != 0;
	ivec4 sc = ivec4(data[t + T_SCISSOR], data[t + T_SCISSOR + 1], data[t + T_SCISSOR + 2], data[t + T_SCISSOR + 3]);
	int ylo = max(yh, sc.y), yhi = min(yl, sc.w), loScissor = sc.x << 1, hiScissor = sc.z << 1;
	bool allInvalid = true, allOver = true, allUnder = true;
	ivec4 span;
	for (int k = 0; k < 4; ++k) {
		int q = y * 4 + k, dy = q - (yh & ~3);
		int xh = bitfieldExtract(xh0 + dy * dxh, 0, 27);
		int xl = q < ym ? bitfieldExtract(xm0 + dy * dxm, 0, 27) : bitfieldExtract(xl0 + (q - ym) * dxl, 0, 27);
		int left = flip ? snapX(xh) : snapX(xl), right = flip ? snapX(xl) : snapX(xh);
		bool invalid = (left >> 1) > (right >> 1);
		allOver = allOver && min(left, right) >= hiScissor;
		allUnder = allUnder && max(left, right) < loScissor;
		left = clamp(left, loScissor, hiScissor);
		right = clamp(right, loScissor, hiScissor);
		if (invalid || q < ylo || q >= yhi) {
			left = 0xFFFF, right = 0;
		} else {
			allInvalid = false;
		}
		span[k] = left | right << 16;
	}
	return allInvalid || allOver || allUnder ? ivec4(0xFFFF) : span;
}

// An attribute on the row: down the major edge a line at a time, back along x by its fraction.
int rowAttribute(int v, int de, int dyv, int dx, int dy, int xfrac, bool doOffset) {
	int diff = 0;
	if (doOffset) {
		int deh = de & ~0x1FF, dyh = dyv & ~0x1FF;
		diff = deh - (deh >> 2) - dyh + (dyh >> 2);
	}
	v += dy * de;
	return ((v & ~0x1FF) + diff - xfrac * ((dx >> 8) & ~1)) & ~0x3FF;
}

int sext16(int v) {
	return bitfieldExtract(v, 0, 16);
}

// One primitive at the pixel: the attributes by kind, then one path for texturing, the level of
// detail, and shading.
void drawPrim(int p, int x, int y, ivec4 span) {
	int kind = data[p + P_KIND];
	if (!(HAS_TRI && kind == KIND_TRIANGLE)) {
		loadState(p);
	}
	clearPixel();
	pDz = data[p + 22];
	pPrimDz = data[p + 23];
	int tile = data[p + P_TILE];
	bool tex = false, lod = false, copy = false;
	ivec2 st = ivec2(0), sx = ivec2(0), sy = ivec2(0);
	if (HAS_FILL && kind == KIND_FILL) {
		fillPixel(x, y);
		return;
	}
	if (HAS_TRI && kind == KIND_TRIANGLE) {
		int t = data[p + P_TRI];
		int yh = data[t + T_YH];
		int mask = 0;
		for (int k = 0; k < 4; ++k) {
			int l8 = span[k] & 0xFFFF, r8 = span[k] >> 16;
			for (int j = 0; j < 2; ++j) {
				int e = x * 8 + (k & 1) * 2 + j * 4;
				if (e >= l8 && e < r8) {
					mask |= 1 << (k * 2 + j);
				}
			}
		}
		// A pixel is drawn when any sample is covered with antialiasing, else when the first is.
		if (mask == 0 || ((data[data[p + P_STATE] + S_OM_LO] & 8) == 0 && (mask & 1) == 0)) {
			return;
		}
		loadState(p);
		pCvg = bitCount(mask);
		pCvbit = (mask & 1) != 0;
		int flags = data[p + P_FLAGS];
		bool flip = data[t + T_FLIP] != 0;
		int dxh = edgeSlope(t, T_DXH);
		bool doOffset = flip == (data[t + T_DXH] < 0);
		int dy = y - (yh >> 2);
		int xh = edgeX(t, T_XH) + dy * (dxh << 2) + (doOffset ? 3 * dxh : 0);
		int xfrac = (int(omHi >> 20) & 3) == 2 ? 0 : (xh >> 7) & 0xFF;
		int dxp = x - (xh >> 15);
		int firstBit = findLSB(mask);
		int yoff = firstBit >> 1, xoff = ((firstBit & 1) << 1) + (yoff & 1);
		if (HAS_SHADE && (flags & 4) != 0) {
			ivec4 c;
			for (int i = 0; i < 4; ++i) {
				int dx = data[p + P_SDX + i], dyv = data[t + T_SDY + i];
				int v = rowAttribute(data[t + T_SV + i], data[t + T_SDE + i], dyv, dx, dy, xfrac, doOffset) + (dx & ~0x1F) * dxp;
				int snapped = sext16(sext16(v >> 14) << 2);
				snapped = sext16(snapped + sext16(xoff * sext16(dx >> 14)) + sext16(yoff * sext16(dyv >> 14)));
				c[i] = clamp9(snapped >> 4);
			}
			pShade = c;
		}
		if (HAS_ZTRI && (flags & 1) != 0) {
			int dz = data[p + P_ZDX], dzy = data[t + T_ZDY];
			int v = rowAttribute(data[t + T_ZV], data[t + T_ZDE], dzy, dz, dy, xfrac, doOffset) + dz * dxp;
			int snapped = ((v >> 10) << 2) + xoff * (dz >> 10) + yoff * (dzy >> 10);
			int z = bitfieldExtract((snapped >> 5) - 0x20000, 0, 19) + 0x20000;
			pZ = clamp(z, 0, 0x3FFFF);
		}
		if (HAS_TEX && (flags & 2) != 0) {
			tex = true;
			lod = HAS_LOD && (((omHi >> 16) & 1u) != 0u || lodFracUsed());
			bool persp = HAS_PERSP && ((omHi >> 19) & 1u) != 0u;
			ivec3 stw, stwX, stwY;
			for (int i = 0; i < 3; ++i) {
				int step = data[p + P_TDX + i] & ~0x1F;
				stw[i] = rowAttribute(data[t + T_TV + i], data[t + T_TDE + i], data[p + P_TDY + i], data[p + P_TDX + i], dy, xfrac, doOffset) + step * dxp;
				stwX[i] = stw[i] + (flip ? step : -step);
				stwY[i] = stw[i] + (data[p + P_TDY + i] & ~0x7FFF);
			}
			st = texCoords(stw.x, stw.y, stw.z, persp);
			if (lod) {
				sx = texCoords(stwX.x, stwX.y, stwX.z, persp);
				sy = texCoords(stwY.x, stwY.y, stwY.z, persp);
			}
		}
	} else if (HAS_TEXRECT && kind == KIND_TEXRECT) {
		int cycleType = int(omHi >> 20) & 3;
		int s0 = data[p + P_S0], t0 = data[p + P_T0], dsdx = data[p + P_DSDX], dtdy = data[p + P_DTDY];
		bool flip = data[p + P_FLIP] != 0;
		int dx = x - data[p + P_CX0], dy = y - data[p + P_CY0];
		int s = ((s0 << 5) + (flip ? dy * dsdx : dx * dsdx)) >> 5;
		int t = ((t0 << 5) + (flip ? dx * dtdy : dy * dtdy)) >> 5;
		st = ivec2(s, t);
		tex = true;
		lod = HAS_LOD && cycleType < 2 && (((omHi >> 16) & 1u) != 0u || lodFracUsed());
		sx = ivec2(s + (flip ? 0 : dsdx >> 5), t + (flip ? dtdy >> 5 : 0));
		sy = ivec2(s + (flip ? dsdx >> 5 : 0), t + (flip ? 0 : dtdy >> 5));
		copy = HAS_COPY && cycleType == 2;
		pZ = data[gS + S_PRIMZ];
	} else {
		pZ = data[gS + S_PRIMZ];
	}
	if (HAS_TEX && tex) {
		int t1 = tile, t2 = tile + 1;
		if (lod) {
			pLodFrac = levelOfDetail(tile, st.x, st.y, sx.x, sx.y, sy.x, sy.y, t1, t2);
		}
		// Each cycle filters or converts from YUV; convert-one converts the first cycle's texel.
		int cycleType = int(omHi >> 20) & 3;
		bool filter0 = cycleType >= 2 || ((omHi >> 11) & 1u) != 0u, filter1 = ((omHi >> 10) & 1u) != 0u;
		bool convertOne = HAS_CONV && ((omHi >> 9) & 1u) != 0u;
		int samples = HAS_TWO && cycleType == 1 ? 2 : 1;
		ivec4 first = ivec4(0);
		for (int i = 0; i < samples; ++i) {
			ivec4 c;
			bool conv;
			if (i == 1 && convertOne) {
				c = first;
				conv = true;
			} else {
				bool f = i == 0 ? filter0 : filter1;
				c = sampleTile(i == 0 ? t1 : t2, st.x, st.y, f);
				conv = !f;
				if (i == 0) {
					first = c;
				}
			}
			if (HAS_CONV && conv) {
				c = convertYuv(c);
			}
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
	if (HAS_COPY && copy) {
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
shared ivec4 spans[256];
shared uint colorOut[256], zOut[256];

// A halfword and its hidden byte into the word at index w, when dirty.
uint putHalf(uint word, uint v, int k) {
	uint s = uint(k & 1) * 16u;
	return (v & (1u << 18)) == 0u ? word : (word & ~(0xFFFFu << s)) | (((v >> 8) & 255u) | (v & 255u) << 8) << s;
}

uint putHidden(uint word, uint v, int k) {
	uint s = uint(k) * 8u;
	return (v & (1u << 18)) == 0u ? word : (word & ~(255u << s)) | ((v >> 16) & 3u) << s;
}

// The tile's 16-bit pixels back to RDRAM, each word by the thread of its first pixel: four pixels'
// color and depth (two words each) and hidden bytes (a word each).
void storeWords(int lx, int row) {
	colorOut[row + lx] = fbColor | fbHidden << 16 | (colorDirty ? 1u << 18 : 0u);
	zOut[row + lx] = fbZ | fbZHidden << 16 | (HAS_ZBUF && zDirty ? 1u << 18 : 0u);
	barrier();
	if ((lx & 3) != 0) {
		return;
	}
	uvec4 c = uvec4(colorOut[row + lx], colorOut[row + lx + 1], colorOut[row + lx + 2], colorOut[row + lx + 3]);
	if (((c.x | c.y | c.z | c.w) & (1u << 18)) != 0u) {
		uint w = colorAt >> 2, h = hiddenAt() >> 2;
		rdram[w] = putHalf(putHalf(rdram[w], c.x, 0), c.y, 1);
		rdram[w + 1u] = putHalf(putHalf(rdram[w + 1u], c.z, 0), c.w, 1);
		rdram[h] = putHidden(putHidden(putHidden(putHidden(rdram[h], c.x, 0), c.y, 1), c.z, 2), c.w, 3);
	}
	uvec4 z = uvec4(zOut[row + lx], zOut[row + lx + 1], zOut[row + lx + 2], zOut[row + lx + 3]);
	if (HAS_ZBUF && ((z.x | z.y | z.z | z.w) & (1u << 18)) != 0u) {
		uint w = zAt >> 2, h = (RDRAM_MASK + 1u + (zAt >> 1)) >> 2;
		rdram[w] = putHalf(putHalf(rdram[w], z.x, 0), z.y, 1);
		rdram[w + 1u] = putHalf(putHalf(rdram[w + 1u], z.z, 0), z.w, 1);
		rdram[h] = putHidden(putHidden(putHidden(putHidden(rdram[h], z.x, 0), z.y, 1), z.z, 2), z.w, 3);
	}
}

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
	// Sixteen primitives at a time: each thread works out one's span on its row for the others.
	int lx = int(gl_LocalInvocationID.x), row = int(gl_LocalInvocationID.y) * 16;
	for (int base = 0; base < count; base += 16) {
		int n = min(16, count - base);
		if (HAS_TRI) {
			ivec4 span = ivec4(0);
			if (lx < n) {
				int p = data[u[1] + first + base + lx] * PRIM_SIZE;
				if (data[p + P_KIND] == KIND_TRIANGLE && y >= data[p + P_Y0] && y <= data[p + P_Y1]) {
					span = rowSpan(p, y);
				}
			}
			spans[row + lx] = span;
			barrier();
		}
		for (int j = 0; j < n; ++j) {
			int index = data[u[1] + first + base + j];
			int p = index * PRIM_SIZE;
			if (x < data[p + P_X0] || x > data[p + P_X1] || y < data[p + P_Y0] || y > data[p + P_Y1]) {
				continue;
			}
			shaded = false;
			drawPrim(p, x, y, HAS_TRI ? spans[row + j] : ivec4(0));
			if (shaded) {
				key = uint(index + 1) << 20 | uint(y) << 10 | uint(x);
				keyColor = lastCombined;
			}
		}
		if (HAS_TRI) {
			barrier();
		}
	}
	if (u[10] == 0) {
		if (count > 0) {
			storePixel();
		}
	} else if (count > 0) {
		storeWords(lx, row);
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
