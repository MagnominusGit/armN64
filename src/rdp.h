#pragma once

#include "n64.h"

// The RDP's state (rdp.c), shared with the GL renderer (glr.c).

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
	// Samples of the pixel's 8 the primitive covers, and whether the first is one of them.
	int cvg;
	bool cvbit;
	// Where it is (the combiner's noise).
	int x, y;
};

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
	// An input is the combined value, which in the first cycle is the previous pixel's.
	bool readsCombined;
};

struct RDP {
	uint8_t tmem[0x1000];
	// While set, loads keep each TMEM byte's RDRAM address here (the GPU RDP's loads of what it drew).
	uint32_t* tmemSource;
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
	// The last triangle's depth slope (dzpix), and the blender's shifts for memory alpha from the
	// depth test.
	int dzpix, blShiftA, blShiftB;
	int minLevel, maxLevel;
	// The combiner's last output: what "combined" reads in one-cycle mode and in the first cycle.
	struct Color lastCombined;
	int k4, k5;
	// The texture convert unit's YUV factors as it uses them (2K + 1).
	int k0, k1, k2, k3;
	// RDP clocks spent on the commands so far.
	uint64_t clocks;
	// The pixel being drawn, and where the combiner's inputs come from for the current combine mode.
	struct Pixel pix;
	struct Stage stage[2];
	// otherModes taken apart (setupModes): the cycle type, depth and alpha tests, and the blender's
	// P, A, M and B inputs for each cycle.
	int cycleType, zMode, cvgDest;
	bool antialias, imageRead, colorOnCvg, zCompare, zUpdate, zPrim, alphaCompare, cvgAlpha, cvgTimesAlpha, forceBlend, texConvert;
	// The dither modes for colour and alpha (other modes' rgb_dither_sel and alpha_dither_sel).
	int ditherRgb, ditherAlpha;
	uint8_t blendSel[2][4];
	// The rows this renderer draws: y % rowCount == rowIndex (rowCount 0: none).
	int rowIndex, rowCount;
	// Primitives drawn so far, and the primitive, y and x of the pixel that left lastCombined, for
	// finding the latest across renderers.
	uint64_t seq, lastPos;
	// Other threads drawing the commands, or the GPU (the main thread's state only).
	struct Renderers* renderers;
	struct Gpu* gpu;
	// Bumped by texture loads and tile changes (the GL renderer's texture lookups).
	uint32_t tmemGen;
	// Bumped by every command but a primitive (the GL renderer's state is the same while it is).
	uint32_t stateGen;
};

// The HLE's corners of a triangle for the GL renderer, beside the RDP triangle they stand for
// (found by its edge words): depth, texture and shade (bits 0-2), tile (8-10), level (12-14) and
// perspective (16); per corner x, y, z, 1/w, r, g, b, a, s, t.
struct GlrCorners {
	uint64_t edges[4];
	uint32_t flags;
	float v[3][10];
};

// The GL renderer (glr.c), for HLE: N64_RDP_GL.
struct Glr* glrCreate(struct N64*);
void glrDestroy(struct Glr*);
// A primitive: an RDP one, drawn from the HLE's corners when it queued them.
void glrDraw(struct N64*, struct RDP*, const uint64_t* w, int op);
// The HLE's corners for a triangle it sends next (the GL renderer made if there is none yet), the
// triangle a skipped frame leaves undrawn, and corners for triangles that won't come.
void glrQueue(struct N64*, const struct GlrCorners*);
void glrSkip(struct N64*, const uint64_t* w, int op);
void glrQueueClear(struct N64*);
// Brings what the GPU drew back to RDRAM.
void glrSync(struct N64*);
void glrStateLoaded(struct Glr*);
bool glrRead(struct N64*, unsigned framebuffer, int x, int y, int w, int h, uint32_t* out);
// Whether the VI's framebuffer (origin) is one the GPU draws: then the picture is the GPU's,
// without bringing it back (a frontend shows that texture).
// A texture rectangle's pixels (inclusive, scissored) and the corner its coordinates start at.
void rdpRectBounds(struct RDP*, uint64_t w0, int* cx0, int* cy0, int* x0, int* y0, int* x1, int* y1);
// Whether the GPU has drawn RDRAM [addr, addr + len) and not given it back.
bool glrDrew(struct N64*, uint32_t addr, uint32_t len);
bool glrShows(struct N64*, uint32_t origin);
void glrSyncRange(struct N64*, uint32_t addr, uint32_t len);
unsigned glrPicture(struct N64*, uint32_t origin, uint32_t stride, int size, int rows, int* x, int* y);
// The RDP's own drawing of a primitive, into RDRAM.
void rdpDrawSoftware(struct N64*, struct RDP*, const uint64_t* w, int op);
// The depth image's depth at a pixel, 0 to 1.
float rdpDepthAt(struct N64*, struct RDP*, int x, int y);

// The RGBA8 texels (red in the low byte) of a tile's first w x h, as the samplers read TMEM.
void rdpTileImage(struct RDP*, int tile, int w, int h, bool convert, uint32_t* out);
