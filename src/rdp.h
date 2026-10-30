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
};

enum { GLR_TRIANGLE_WORDS = 16 };

// The GL renderer (glr.c), for HLE: N64_RDP_GL.
struct Glr* glrCreate(struct N64*);
void glrDestroy(struct Glr*);
// A primitive: the HLE's triangle (op 0x01: three projected vertices), or an RDP one.
void glrDraw(struct N64*, struct RDP*, const uint64_t* w, int op);
// Brings what the GPU drew back to RDRAM.
void glrSync(struct N64*);
// Whether the VI's framebuffer (origin) is one the GPU draws: then the picture is the GPU's,
// without bringing it back (a frontend shows that texture).
bool glrShows(struct N64*, uint32_t origin);
// The RDP's own drawing of a primitive, into RDRAM.
void rdpDrawSoftware(struct N64*, struct RDP*, const uint64_t* w, int op);

// The RGBA8 texels (red in the low byte) of a tile's first w x h, as the samplers read TMEM.
void rdpTileImage(struct RDP*, int tile, int w, int h, uint32_t* out);
