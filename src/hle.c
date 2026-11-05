#include "n64.h"
#include "rdp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// High-level graphics (N64_ACCURACY=hle): a graphics task's display list run here instead of by its
// microcode, for the Fast3D family (Fast3D, F3DEX, F3DEX2), with the triangles and other commands
// it makes drawn by the RDP as usual. Microcodes it doesn't know run on the RSP.

// F3DPD is Perfect Dark's Fast3D (as GLideN64 has it, GPL-2.0): 12-byte vertices whose colour or
// normal comes from a table (0x07 sets its base), and TRIX (0xB1) for up to eight triangles.
// S2DEX and S2DEX2 are the 2D microcodes (F3DEX's and F3DEX2's command numbers for what they share):
// backgrounds so far, other commands leave the task to the RSP.
// ZSort draws objects the game sorted from vertices it transformed here first (its DMEM in between).
enum { FAMILY_NONE, FAMILY_F3D, FAMILY_F3DEX, FAMILY_F3DEX2, FAMILY_F3DPD, FAMILY_S2DEX, FAMILY_S2DEX2, FAMILY_ZSORT, FAMILY_TURBO3D, FAMILY_T3DUX, FAMILY_ZSORTBOSS, FAMILY_F5ROGUE, FAMILY_F5INDI, FAMILY_GBPPU };

// Geometry mode bits, kept in the F3DEX2 layout.
enum {
	GEOM_ZBUFFER = 0x1, GEOM_SHADE = 0x4, GEOM_CULL_FRONT = 0x200, GEOM_CULL_BACK = 0x400,
	GEOM_FOG = 0x10000, GEOM_LIGHTING = 0x20000, GEOM_TEXTURE_GEN = 0x40000,
	GEOM_TEXTURE_GEN_LINEAR = 0x80000, GEOM_SHADING_SMOOTH = 0x200000, GEOM_POINT_LIGHTING = 0x400000,
};

enum { VERTICES = 64, MATRICES = 32, LIGHTS = 12, LIST_MAX = 1 << 16, DL_DEPTH = 18 };

struct Vertex {
	float x, y, z, w;
	float r, g, b, a;
	float s, t;
};

// A vertex on screen: pixels, depth 0-1, 1/w, colour 0-1 and texels.
struct ScreenVertex {
	float x, y, z, invW;
	float r, g, b, a;
	float s, t;
};

struct Hle {
	// The display list of the last task drawn here, for a yielded copy of it.
	uint32_t doneList;
	// The task's output buffer (FIFO microcodes), when it has one in RDRAM.
	uint32_t fifoStart, fifoEnd;
	int family;
	uint32_t segments[16];
	float modelview[MATRICES][16];
	int depth;
	float projection[16], mvp[16];
	struct Vertex v[VERTICES];
	float vscale[3], vtrans[3];
	int lights;
	// ZSort: its DMEM (the data first), the viewport in quarter pixels, and the RDP lists last run for
	// an object's three.
	uint8_t zDmem[0x1000];
	// What ZSort's and ZSortBOSS's DMEM commands work on: zDmem, or (ZSortBOSS, not dry) the SP's own DMEM,
	// which the CPU reads.
	uint8_t* zMem;
	float zScale[2], zTrans[2];
	uint32_t zRdp[3];
	float lightColor[LIGHTS + 1][3], lightDir[LIGHTS + 1][3], lookat[2][3];
	uint32_t geometry;
	// F3DPD's table of vertex colours.
	uint32_t colorBase;
	// L3DEX and L3DEX2 (F3DEX's and F3DEX2's line microcodes: triangles as their edges) and F3DTEX/A
	// (F3DEX with its own texture load and tile size commands), by name.
	bool lines, texa;
	// F3DAM (Hey You, Pikachu!'s F3DEX2): texture coordinates through a 2x2 matrix of s.15 halves
	// (G_TEXTURE's word, then the one G_MOVEWORD sets at fog + 0x10) plus the offset at fog + 0x0C.
	bool am;
	uint32_t amScale[3];
	// The NoN microcodes (Star Fox 64's F3DEX, Ocarina of Time's F3DZEX) don't clip at the near
	// plane: what lies in front of it is drawn, its depth held at the near end.
	bool noNear;
	// S2DEX: the 2D matrix (A to D, the offset in pixels, the base scales), the object render mode,
	// the state words texture loads and G_SELECT_DL test, and G_RDPHALF_0's half of a selection.
	float objA, objB, objC, objD, objX, objY, objBaseX, objBaseY;
	// Triangles S2DEX draws for an object: 1/w whole (0x7FFF), the texture coordinates as they are.
	bool objTriangles;
	// T3DUX: an object's vertices (up to 255), and the last SETTILE its RDP lists gave.
	struct Vertex t3dux[256];
	uint32_t t3duxTile[2];
	// ZSortBOSS: where its main and sub lists are and which runs, whether each has ended, a signal
	// waited on, the other-mode bits G_RDPSETOTHERMODE takes, the 1/w scale, the fog table and the
	// audio's ADPCM table (the last four kept from task to task, as GLideN64 has them).
	uint32_t bossPc[2];
	int bossList;
	bool bossEnded[2], bossSignal, bossDone, bossInit;
	uint32_t bossMask[2];
	float bossInvw;
	uint8_t bossFog[256];
	int16_t bossTable[8][8];
	// Factor 5's: each list level's jump target (its first word), the vertex colour table, the texture
	// coordinate scales, and where the viewport and the projection matrix were.
	uint32_t f5Jump[DL_DEPTH + 2], f5Colors, f5TexScale[2], f5Viewport, f5Persp;
	// Factor 5's second: its DMEM (as GLideN64 keeps it), the matrices it generates vertices and texture
	// coordinates through, and Battle for Naboo's terrain walk.
	uint8_t f5iDmem[0x1000];
	float f5iGenMtx[16];
	int32_t f5iStMtx[16];
	uint16_t nabooAA;
	int16_t nabooBB, nabooEE, nabooHH;
	// gb64's Game Boy picture processor, one line at a time as the CPU signals: its task (the task
	// data's address, and the data), the sprites of the next line, the tile cache and each tile's
	// offset, the attribute cache, the tilemap and window rows, the line (with room either side),
	// the lines begun, the window's line, and where the caches stand.
	uint32_t gbData;
	uint8_t gbTask[24], gbSprites[160], gbTiles[32][16], gbAttrs[24], gbMap[64], gbWindow[64], gbLine[8 + 160 + 16];
	uint16_t gbTileIds[32];
	uint32_t gbLines, gbWindowY;
	int gbTile, gbAttr, gbState;
	uint64_t gbReadyAt;
	uint8_t nabooDD, nabooCC, nabooTT;
	uint32_t objMode, objStatus[4], half0w0, half0w1;
	// F3DFLX (F-Zero X's F3DEX2): light k in slot k + 1, slot 1 its alpha light, a direction whose
	// angle to the normal picks each vertex's alpha from a table at the G_DMA_IO address (in place of
	// generated texture coordinates).
	bool flx;
	// The early Fast3D of Wave Race 64 and Shadows of the Empire (known by their code; GLideN64's
	// F3DBETA): vertex indices times 5 (G_CULLDL's offsets times 20), G_VTX's count in its length (n times
	// 0x210, less one), F3DEX's G_QUAD, a G_TRI2 at 0xB1 and G_RDPHALF_1 at 0xB3.
	bool wrus;
	// Conker's F3DEXBG (as GLideN64 has it): normals in their own buffer (two bytes a vertex, the third
	// the low byte of the vertex's flag), vertex colours scaled by the light, 48-byte lights with a
	// position and falloff, the coordinate modifiers the light distances go through, and four triangles
	// a command. Its G_LOAD_UCODE turns on the advanced lighting (the last light directional).
	bool cbfd, cbfdAdvanced;
	uint32_t cbfdNormals;
	float cbfdMod[16], lightPos[LIGHTS + 1][3], lightFalloff[LIGHTS + 1];
	// Rare's DMA microcodes, Diddy Kong Racing's and Jet Force Gemini's (as GLideN64 has them, known by
	// their code): matrices in slots that hold the projection too, 10-byte coloured vertices added to
	// the buffer (a billboard's to its first vertex), triangles from memory with their own texture
	// coordinates and culling, offsets added to matrix and vertex addresses, and a table of texture
	// address shifts.
	bool dkr, jfg, dkrBillboard;
	int dkrVertex;
	uint32_t dkrMtxOffset, dkrVtxOffset, dkrTexTable, dkrTexShift, dkrTexCount, dkrImage[2];
	float flxDir[3];
	uint32_t dmaIo;
	float texScaleS, texScaleT;
	int texTile, texLevel;
	bool texOn;
	float fogMul, fogOffset;
	// G_MW_CLIP's ratio: x and y clip at that many times w.
	float clipRatio;
	uint32_t modeH, modeL;
	uint32_t half1;
	// Fast3D's Sprite2D: the sprite its base command named; the scale-flip and draw commands
	// that follow it.
	uint32_t sprite;
	bool spriteMode, spriteFlipS, spriteFlipT;
	float spriteScaleX, spriteScaleY;
	// The display list's call stack, and where it branches to itself (a list the CPU is still
	// writing: the task waits there until that command changes).
	uint32_t stack[DL_DEPTH];
	int sp;
	bool waiting;
	uint32_t waitPc;
	// Microcodes (by a hash of their first code) whose lists broke a family's rules: theirs run on
	// the RSP.
	uint32_t ucode, rejected[8];
	int rejectedCount;
	bool rejectedNow;
	// Fast3D triangles this task with vertex numbers that aren't multiples of 10.
	int oddTriangles;
	// Why the list was left to the RSP (for the frontend).
	char why[48];
	// Counting the commands instead of drawing them (N64HLECHECK).
	bool dry;
	// The color image is 8-bit: the GL renderer leaves those triangles to the RDP (no corners).
	bool image8;
	// The output buffer full while the RDP is frozen and hasn't taken the last fill: the microcode
	// waits for it, the rest of the list kept until it has.
	bool stalled;
	uint64_t list[LIST_MAX];
	size_t count;
	// The task's time: the RSP's work in CPU cycles where the microcode's costs are known, otherwise the
	// list's length as the RDP's commands fill it; and the words left of the command the list ends in.
	size_t timeCount, left;
};

static uint32_t rd32(struct N64* n, uint32_t a) {
	return be32(&n->rdram[a & (RDRAM_SIZE - 4)]);
}

static int16_t rd16(struct N64* n, uint32_t a) {
	a &= RDRAM_SIZE - 2;
	return (int16_t) (n->rdram[a] << 8 | n->rdram[a + 1]);
}

static int8_t rd8(struct N64* n, uint32_t a) {
	return (int8_t) n->rdram[a & (RDRAM_SIZE - 1)];
}

static uint32_t address(struct Hle* h, uint32_t a) {
	return (h->segments[(a >> 24) & 15] + (a & 0xFFFFFF)) & 0xFFFFFF;
}

// The words of the command a list word starts.
static size_t commandWords(uint64_t w) {
	int op = (w >> 56) & 0x3F;
	return op >= 0x08 && op <= 0x0F ? 4 + (op & 4 ? 8 : 0) + (op & 2 ? 8 : 0) + (op & 1 ? 2 : 0) :
	       op == 0x24 || op == 0x25 ? 2 : 1;
}

static void flush(struct N64* n, struct Hle* h, bool canStall) {
	// Whole commands only: a full list may end inside one, which waits for the next flush.
	size_t total = h->count, whole = 0, sent;
	while (whole < total && whole + commandWords(h->list[whole]) <= total) {
		whole += commandWords(h->list[whole]);
	}
	h->count = sent = whole;
	h->stalled = false;
	size_t capacity = (h->fifoEnd - h->fifoStart) / 8;
	if (!h->dry && capacity && h->count) {
		// A FIFO microcode's commands go through its output buffer in RDRAM, the RDP's registers
		// pointing at them, where programs can see them (DPC_CURRENT, the buffer); a list longer
		// than the buffer goes round it, the RDP taking each fill. A frozen RDP takes none, so the
		// microcode waits before writing over a fill (Banjo-Kazooie's intro).
		for (size_t done = 0; done < h->count;) {
			if (canStall && (n->dp.status & 2) && n->dp.current < n->dp.end) {
				h->stalled = true;
				sent = done;
				break;
			}
			size_t end = done;
			while (end < h->count && end - done + commandWords(h->list[end]) <= capacity) {
				end += commandWords(h->list[end]);
			}
			if (end == done) {
				break;
			}
			end = end > h->count ? h->count : end;
			n64RdramWrite(n, h->fifoStart, (uint32_t) (end - done) * 8);
			for (size_t i = done; i < end; ++i) {
				uint8_t* p = &n->rdram[h->fifoStart + (i - done) * 8];
				for (int b = 0; b < 8; ++b) {
					p[b] = (uint8_t) (h->list[i] >> (56 - 8 * b));
				}
			}
			n->dp.status &= ~1u;
			n->dp.start = n->dp.current = h->fifoStart;
			n->dp.end = h->fifoStart + (uint32_t) (end - done) * 8;
			rdpRun(n);
			done = end;
		}
	} else if (!h->dry) {
		rdpCommands(n, h->list, h->count);
	} else {
		for (size_t i = 0; i < h->count;) {
			int op = (h->list[i] >> 56) & 0x3F;
			++n->checkCounts[1][op];
			if (n->checkLen[1] < 4096) {
				n->checkSeq[1][n->checkLen[1]++] = h->list[i];
			}
			i += commandWords(h->list[i]);
		}
	}
	memmove(h->list, h->list + sent, (total - sent) * sizeof(h->list[0]));
	h->count = total - sent;
}

// The RSP's cycles for each vertex, triangle and display list command, measured against the
// microcodes' own runs (CPU cycles; Perfect Dark's from its game alone).
static bool costed(struct N64* n, struct Hle* h) {
	return !n->hleShortTasks && (h->family == FAMILY_F3D || h->family == FAMILY_F3DEX || h->family == FAMILY_F3DEX2 || h->family == FAMILY_F3DPD);
}

static void charge(struct N64* n, struct Hle* h, int vertices, int triangles, int commands) {
	static const int costs[][3] = {
		[FAMILY_F3D] = { 320, 63, 165 }, [FAMILY_F3DEX] = { 160, 199, 216 }, [FAMILY_F3DEX2] = { 160, 175, 64 }, [FAMILY_F3DPD] = { 480, 0, 70 },
	};
	if (costed(n, h)) {
		h->timeCount += (size_t) (vertices * costs[h->family][0] + triangles * costs[h->family][1] + commands * costs[h->family][2]);
	}
}

static void emit(struct N64* n, struct Hle* h, uint64_t w) {
	if (h->count == LIST_MAX) {
		flush(n, h, false);
	}
	h->list[h->count++] = w;
	if (!h->left) {
		// A full list keeps the words of the command it ends in.
		h->left = commandWords(w);
		if (!costed(n, h)) {
			h->timeCount = h->timeCount + h->left > LIST_MAX ? h->left : h->timeCount + h->left;
		}
	}
	--h->left;
}

// ---- Matrices (row vectors: v' = v M)

static void multiply(float* out, const float* a, const float* b) {
	float r[16];
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			r[i * 4 + j] = a[i * 4] * b[j] + a[i * 4 + 1] * b[4 + j] + a[i * 4 + 2] * b[8 + j] + a[i * 4 + 3] * b[12 + j];
		}
	}
	memcpy(out, r, sizeof(r));
}

// An s15.16 matrix: the integer halves, then the fractions.
static void readMatrix(struct N64* n, uint32_t a, float* m) {
	for (int i = 0; i < 16; ++i) {
		int16_t whole = rd16(n, a + i * 2);
		uint16_t frac = (uint16_t) rd16(n, a + 32 + i * 2);
		m[i] = (float) ((int32_t) ((uint32_t) (uint16_t) whole << 16 | frac)) / 65536.0f;
	}
}

static void updateMvp(struct Hle* h) {
	multiply(h->mvp, h->modelview[h->depth], h->projection);
}

static void matrix(struct N64* n, struct Hle* h, uint32_t a, bool projection, bool load, bool push) {
	float m[16];
	readMatrix(n, address(h, a), m);
	if (projection) {
		if (load) {
			memcpy(h->projection, m, sizeof(m));
		} else {
			multiply(h->projection, m, h->projection);
		}
	} else {
		if (push && h->depth < MATRICES - 1) {
			memcpy(h->modelview[h->depth + 1], h->modelview[h->depth], sizeof(m));
			++h->depth;
		}
		if (load) {
			memcpy(h->modelview[h->depth], m, sizeof(m));
		} else {
			multiply(h->modelview[h->depth], m, h->modelview[h->depth]);
		}
	}
	updateMvp(h);
}

static void popMatrix(struct Hle* h, int count) {
	h->depth = count > h->depth ? 0 : h->depth - count;
	updateMvp(h);
}

// ---- Vertices

static void normalize(float* v) {
	float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (l > 0) {
		v[0] /= l, v[1] /= l, v[2] /= l;
	}
}

// Rare's G_DMA_MTX: a matrix into a slot, or (Jet Force Gemini's) times slot 0's; the slot is the
// one drawn with, the projection in it.
static void dkrMatrix(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	if ((w0 & 0xFFFF) != 64) {
		return;
	}
	int index = (w0 >> 16) & 15;
	bool mul = false;
	if (index == 0) {
		index = (w0 >> 22) & 3;
	} else {
		mul = (w0 >> 23) & 1;
	}
	float m[16];
	readMatrix(n, (address(h, w1) + h->dkrMtxOffset) & (RDRAM_SIZE - 1), m);
	if (mul) {
		multiply(h->modelview[index], m, h->modelview[0]);
	} else {
		memcpy(h->modelview[index], m, sizeof(m));
	}
	static const float identity[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
	memcpy(h->projection, identity, sizeof(identity));
	h->depth = index;
	updateMvp(h);
}

// Rare's G_DMA_VTX: x, y, z and a colour, 10 bytes each, after the ones loaded since the last
// triangles (a billboard's after its first vertex, and moved by it).
// A vertex's fog, in its alpha, from its depth on screen.
static float fogAlpha(const struct Hle* h, const struct Vertex* v) {
	float f = (v->w != 0 ? v->z / v->w : 0) * h->fogMul + h->fogOffset;
	return (f < 0 ? 0 : f > 255 ? 255 : f) / 255.0f;
}

static void dkrVertices(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	if (w0 & 0x10000) {
		h->dkrVertex = h->dkrBillboard ? 1 : h->dkrVertex;
	} else {
		h->dkrVertex = 0;
	}
	int const count = (int) ((w0 >> 19) & 31) + (h->jfg ? 0 : 1), first = h->dkrVertex + (int) ((w0 >> 9) & 31);
	charge(n, h, count, 0, 0);
	uint32_t a = address(h, w1) + h->dkrVtxOffset;
	const float* m = h->mvp;
	for (int i = 0; i < count && first + i < VERTICES; ++i, a += 10) {
		struct Vertex* v = &h->v[first + i];
		float x = rd16(n, a), y = rd16(n, a + 2), z = rd16(n, a + 4);
		v->x = x * m[0] + y * m[4] + z * m[8] + m[12];
		v->y = x * m[1] + y * m[5] + z * m[9] + m[13];
		v->z = x * m[2] + y * m[6] + z * m[10] + m[14];
		v->w = x * m[3] + y * m[7] + z * m[11] + m[15];
		if (h->dkrBillboard) {
			v->x += h->v[0].x, v->y += h->v[0].y, v->z += h->v[0].z, v->w += h->v[0].w;
		}
		v->r = n->rdram[(a + 6) & (RDRAM_SIZE - 1)] / 255.0f, v->g = n->rdram[(a + 7) & (RDRAM_SIZE - 1)] / 255.0f;
		v->b = n->rdram[(a + 8) & (RDRAM_SIZE - 1)] / 255.0f, v->a = n->rdram[(a + 9) & (RDRAM_SIZE - 1)] / 255.0f;
		v->s = v->t = 0;
		if (h->geometry & GEOM_FOG) {
			v->a = fogAlpha(h, v);
		}
	}
	h->dkrVertex += count;
}

// Rare's G_DMA_TRI: 16-byte triangles (flag, three vertices, then S and T for each, s10.5), culled
// by the viewport's facing unless the flag's 0x40 is set.
static void drawTriangle(struct N64* n, struct Hle* h, int i0, int i1, int i2);
static void dkrTriangles(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	h->texOn = (w0 >> 16) & 15;
	int const count = (int) ((w0 >> 20) & 15) + 1;
	uint32_t a = address(h, w1);
	for (int i = 0; i < count; ++i, a += 16) {
		uint32_t const head = rd32(n, a);
		int const v[3] = { (head >> 16) & 0xFF, (head >> 8) & 0xFF, head & 0xFF };
		if (v[0] >= VERTICES || v[1] >= VERTICES || v[2] >= VERTICES) {
			continue;
		}
		h->geometry &= ~(GEOM_CULL_FRONT | GEOM_CULL_BACK);
		if (!(head & 0x40000000)) {
			h->geometry |= h->vscale[0] > 0 ? GEOM_CULL_BACK : GEOM_CULL_FRONT;
		}
		for (int k = 0; k < 3; ++k) {
			h->v[v[k]].s = rd16(n, a + 4 + 4 * (uint32_t) k) / 32.0f, h->v[v[k]].t = rd16(n, a + 6 + 4 * (uint32_t) k) / 32.0f;
		}
		drawTriangle(n, h, v[0], v[1], v[2]);
	}
	h->dkrVertex = 0;
}

// Conker's G_MW_COORD_MOD: the integer and fraction halves of the scale the light distances take, and
// the offset added first (GLideN64's gSP.cbfd.vertexCoordMod).
static void cbfdCoordMod(struct Hle* h, uint32_t w0, uint32_t w1) {
	if (w0 & 8) {
		return;
	}
	int const idx = (w0 >> 1) & 3, pos = w0 & 0x30;
	float* m = h->cbfdMod;
	if (pos == 0) {
		m[idx] = (int16_t) (w1 >> 16), m[1 + idx] = (int16_t) w1;
	} else if (pos == 0x10) {
		m[4 + idx] = (w1 >> 16) / 65536.0f, m[5 + idx] = (w1 & 0xFFFF) / 65536.0f;
		m[12 + idx] = m[idx] + m[4 + idx], m[13 + idx] = m[1 + idx] + m[5 + idx];
	} else if (pos == 0x20) {
		m[8 + idx] = (int16_t) (w1 >> 16), m[9 + idx] = (int16_t) w1;
	}
}

// Conker's lighting, on the transformed position: the ambient, the last light directional (advanced
// lighting only), the others point lights falling off with the square of the distance; the vertex's
// colour scaled by the sum.
static void cbfdLight(struct Hle* h, struct Vertex* v, const float* e, float dirs[][3]) {
	const float* m = h->cbfdMod;
	float const p[3] = { (v->x + m[8]) * m[12], (v->y + m[9]) * m[13], (v->z + m[10]) * m[14] };
	float rgb[3] = { h->lightColor[h->lights][0], h->lightColor[h->lights][1], h->lightColor[h->lights][2] };
	int l = h->lights - 1;
	if (h->cbfdAdvanced && l >= 0) {
		float d = e[0] * dirs[l][0] + e[1] * dirs[l][1] + e[2] * dirs[l][2];
		d = d > 1 ? 1 : d;
		if (d > 0) {
			rgb[0] += h->lightColor[l][0] * d, rgb[1] += h->lightColor[l][1] * d, rgb[2] += h->lightColor[l][2] * d;
		}
	}
	for (--l; l >= 0; --l) {
		float const dx = p[0] - h->lightPos[l][0], dy = p[1] - h->lightPos[l][1], dz = p[2] - h->lightPos[l][2];
		float const len = 2 * (dx * dx + dy * dy + dz * dz) / 65536.0f;
		float k = len > 0 ? h->lightFalloff[l] / len : 1;
		k = k > 1 ? 1 : k;
		if (h->cbfdAdvanced && (h->geometry & GEOM_POINT_LIGHTING)) {
			float const d = e[0] * dirs[l][0] + e[1] * dirs[l][1] + e[2] * dirs[l][2];
			k *= d > 1 ? 1 : d;
		}
		if (k > 0) {
			rgb[0] += h->lightColor[l][0] * k, rgb[1] += h->lightColor[l][1] * k, rgb[2] += h->lightColor[l][2] * k;
		}
	}
	v->r *= rgb[0] > 1 ? 1 : rgb[0], v->g *= rgb[1] > 1 ? 1 : rgb[1], v->b *= rgb[2] > 1 ? 1 : rgb[2];
}

static void loadVertices(struct N64* n, struct Hle* h, uint32_t a, int first, int count) {
	charge(n, h, count, 0, 0);
	a = address(h, a);
	// Vertices before the buffer (a start the list can't mean) go nowhere.
	if (first < 0) {
		a += (uint32_t) -first * (h->family == FAMILY_F3DPD ? 12 : 16);
		count += first, first = 0;
	}
	const float* mv = h->modelview[h->depth];
	// Lighting in model space, as the microcodes light: the lights' and lookat's directions through
	// the modelview's transpose, against the normal as it is.
	float dirs[LIGHTS + 2][3];
	if (h->geometry & (GEOM_LIGHTING | GEOM_TEXTURE_GEN)) {
		for (int l = 0; l < h->lights + 2; ++l) {
			const float* d = l < h->lights ? h->lightDir[l] : h->lookat[l - h->lights];
			for (int k = 0; k < 3; ++k) {
				dirs[l][k] = mv[k * 4] * d[0] + mv[k * 4 + 1] * d[1] + mv[k * 4 + 2] * d[2];
			}
			normalize(dirs[l]);
		}
	}
	bool const pd = h->family == FAMILY_F3DPD;
	for (int i = 0; i < count && first + i < VERTICES; ++i, a += pd ? 12 : 16) {
		struct Vertex* v = &h->v[first + i];
		// The colour (or normal): in the vertex, or for F3DPD in the table at its index.
		uint32_t const ca = pd ? h->colorBase + (rd16(n, a + 6) & 0xFF) : a + 12;
		float x = rd16(n, a), y = rd16(n, a + 2), z = rd16(n, a + 4);
		const float* m = h->mvp;
		v->x = x * m[0] + y * m[4] + z * m[8] + m[12];
		v->y = x * m[1] + y * m[5] + z * m[9] + m[13];
		v->z = x * m[2] + y * m[6] + z * m[10] + m[14];
		v->w = x * m[3] + y * m[7] + z * m[11] + m[15];
		float s = rd16(n, a + 8), t = rd16(n, a + 10);
		uint8_t c[4] = { n->rdram[ca & (RDRAM_SIZE - 1)], n->rdram[(ca + 1) & (RDRAM_SIZE - 1)], n->rdram[(ca + 2) & (RDRAM_SIZE - 1)], n->rdram[(ca + 3) & (RDRAM_SIZE - 1)] };
		v->a = c[3] / 255.0f;
		if (h->cbfd && (h->geometry & (GEOM_LIGHTING | GEOM_TEXTURE_GEN))) {
			uint32_t const na = h->cbfdNormals + (uint32_t) (first + i) * 2;
			float e[3] = { rd8(n, na), rd8(n, na + 1), rd8(n, a + 7) };
			normalize(e);
			v->r = c[0] / 255.0f, v->g = c[1] / 255.0f, v->b = c[2] / 255.0f;
			if ((h->geometry & GEOM_LIGHTING) && rd16(n, a + 6) >= 0) {
				cbfdLight(h, v, e, dirs);
			}
			if (h->geometry & GEOM_TEXTURE_GEN) {
				const float* lx = dirs[h->lights], * ly = dirs[h->lights + 1];
				float gx = e[0] * lx[0] + e[1] * lx[1] + e[2] * lx[2], gy = e[0] * ly[0] + e[1] * ly[1] + e[2] * ly[2];
				s = (gx + 1) * 512, t = (gy + 1) * 512;
			}
		} else if (h->geometry & (GEOM_LIGHTING | GEOM_TEXTURE_GEN)) {
			float e[3] = { rd8(n, ca), rd8(n, ca + 1), rd8(n, ca + 2) };
			normalize(e);
			if (h->geometry & GEOM_LIGHTING) {
				float rgb[3] = { h->lightColor[h->lights][0], h->lightColor[h->lights][1], h->lightColor[h->lights][2] };
				for (int l = 0; l < h->lights; ++l) {
					float d = e[0] * dirs[l][0] + e[1] * dirs[l][1] + e[2] * dirs[l][2];
					if (d > 0) {
						rgb[0] += d * h->lightColor[l][0], rgb[1] += d * h->lightColor[l][1], rgb[2] += d * h->lightColor[l][2];
					}
				}
				v->r = rgb[0] > 1 ? 1 : rgb[0];
				v->g = rgb[1] > 1 ? 1 : rgb[1];
				v->b = rgb[2] > 1 ? 1 : rgb[2];
			} else {
				v->r = c[0] / 255.0f, v->g = c[1] / 255.0f, v->b = c[2] / 255.0f;
			}
			if ((h->geometry & GEOM_TEXTURE_GEN) && h->flx) {
				float d[3];
				for (int k = 0; k < 3; ++k) {
					d[k] = mv[k * 4] * h->flxDir[0] + mv[k * 4 + 1] * h->flxDir[1] + mv[k * 4 + 2] * h->flxDir[2];
				}
				normalize(d);
				int index = (int16_t) ((e[0] * d[0] + e[1] * d[1] + e[2] * d[2]) * 128);
				v->a = n->rdram[(h->dmaIo + 128 + (uint32_t) index) & (RDRAM_SIZE - 1)] / 255.0f;
			} else if (h->geometry & GEOM_TEXTURE_GEN) {
				const float* lx = dirs[h->lights], * ly = dirs[h->lights + 1];
				float gx = e[0] * lx[0] + e[1] * lx[1] + e[2] * lx[2];
				float gy = e[0] * ly[0] + e[1] * ly[1] + e[2] * ly[2];
				if (h->geometry & GEOM_TEXTURE_GEN_LINEAR) {
					s = (float) acos(gx < -1 ? -1 : gx > 1 ? 1 : gx) * 325.94931f;
					t = (float) acos(gy < -1 ? -1 : gy > 1 ? 1 : gy) * 325.94931f;
				} else {
					s = (gx + 1) * 512, t = (gy + 1) * 512;
				}
			}
		} else {
			v->r = c[0] / 255.0f, v->g = c[1] / 255.0f, v->b = c[2] / 255.0f;
		}
		// Texture coordinates are s10.5 (generated ones texels), scaled by G_TEXTURE.
		if (h->am && !(h->geometry & GEOM_TEXTURE_GEN)) {
			uint32_t const s0 = (uint32_t) (int32_t) s, t0 = (uint32_t) (int32_t) t;
			uint32_t const* k = h->amScale;
			s = (int16_t) (((((k[0] >> 16) * s0 << 1) + ((k[0] & 0xFFFF) * t0 << 1) + 0x8000) >> 16) + (k[1] >> 16));
			t = (int16_t) (((((k[2] >> 16) * s0 << 1) + ((k[2] & 0xFFFF) * t0 << 1) + 0x8000) >> 16) + k[1]);
		}
		float unit = (h->geometry & GEOM_TEXTURE_GEN) && !h->flx ? 1 : 1 / 32.0f;
		v->s = s * h->texScaleS * unit;
		v->t = t * h->texScaleT * unit;
		if (h->geometry & GEOM_FOG) {
			v->a = fogAlpha(h, v);
		}
	}
}

// G_MODIFYVTX (and F3DEX's G_MOVEWORD to the points): a loaded vertex's colour, texture
// coordinates (already scaled) or position on screen (x and y s13.2, z 16.16 in the viewport's range,
// kept by moving it in clip space at its own w).
// G_MW_MATRIX: a word into the combined matrix as the microcode keeps it (16 s15.16 elements, their
// integer halves then their fractions): two elements' integer or fraction halves.
static void insertMatrix(struct Hle* h, int where, uint32_t value) {
	if ((where & 3) || where > 0x3C) {
		return;
	}
	int const e = (where & 0x1F) >> 1;
	for (int k = 0; k < 2; ++k) {
		uint32_t const half = k ? value & 0xFFFF : value >> 16;
		int32_t fixed = (int32_t) lrintf(h->mvp[e + k] * 65536.0f);
		fixed = where < 0x20 ? (int32_t) (half << 16 | (fixed & 0xFFFF)) : (int32_t) (((uint32_t) fixed & 0xFFFF0000u) | half);
		h->mvp[e + k] = fixed / 65536.0f;
	}
}

static void modifyVertex(struct Hle* h, int index, int where, uint32_t value) {
	if (index >= VERTICES) {
		return;
	}
	struct Vertex* v = &h->v[index];
	switch (where) {
	case 0x10:
		v->r = (value >> 24) / 255.0f, v->g = ((value >> 16) & 0xFF) / 255.0f;
		v->b = ((value >> 8) & 0xFF) / 255.0f, v->a = (value & 0xFF) / 255.0f;
		break;
	case 0x14:
		v->s = (int16_t) (value >> 16) / 32.0f, v->t = (int16_t) value / 32.0f;
		break;
	case 0x18:
		if (h->vscale[0] != 0 && h->vscale[1] != 0) {
			v->x = ((int16_t) (value >> 16) / 4.0f - h->vtrans[0]) / h->vscale[0] * v->w;
			v->y = -((int16_t) value / 4.0f - h->vtrans[1]) / h->vscale[1] * v->w;
		}
		break;
	case 0x1C:
		if (h->vscale[2] != 0) {
			v->z = ((int32_t) value / 65536.0f - h->vtrans[2]) / h->vscale[2] * v->w;
		}
		break;
	}
}

// ---- Triangles

static int32_t fixed16(float f) {
	if (f >= 32768.0f) {
		return 0x7FFFFFFF;
	}
	if (f < -32768.0f) {
		return (int32_t) 0x80000000u;
	}
	return (int32_t) floorf(f * 65536.0f);
}

// The edge setup of an RDP triangle: the corners sorted top to bottom, the command's first four
// words, and what the coefficients need from them.
struct EdgeSetup {
	const struct ScreenVertex *a, *b, *c;
	uint64_t w[4];
	float hx, hy, mx, my, attr, ish, fy;
};

static void edgeSetup(const struct Hle* h, const struct ScreenVertex* a, const struct ScreenVertex* b, const struct ScreenVertex* c, struct EdgeSetup* e) {
	if (a->y > b->y) {
		const struct ScreenVertex* t = a; a = b; b = t;
	}
	if (b->y > c->y) {
		const struct ScreenVertex* t = b; b = c; c = t;
	}
	if (a->y > b->y) {
		const struct ScreenVertex* t = a; a = b; b = t;
	}
	bool shade = h->geometry & GEOM_SHADE, tex = h->texOn, z = h->geometry & GEOM_ZBUFFER;
	int op = 0x08 | (shade ? 4 : 0) | (tex ? 2 : 0) | (z ? 1 : 0);
	float y1 = floorf(a->y * 4) / 4, y2 = floorf(b->y * 4) / 4, y3 = floorf(c->y * 4) / 4;
	int32_t y1f = (int32_t) floorf(a->y * 4), y2f = (int32_t) floorf(b->y * 4), y3f = (int32_t) floorf(c->y * 4);
	y1f = y1f < -16384 ? -16384 : y1f > 16380 ? 16380 : y1f;
	y2f = y2f < -16384 ? -16384 : y2f > 16380 ? 16380 : y2f;
	y3f = y3f < -16384 ? -16384 : y3f > 16380 ? 16380 : y3f;
	float hx = c->x - a->x, hy = y3 - y1, mx = b->x - a->x, my = y2 - y1, lx = c->x - b->x, ly = y3 - y2;
	float nz = hx * my - hy * mx;
	uint32_t lft = nz < 0;
	float ish = fabsf(hy) > 1e-20f ? hx / hy : 0, ism = fabsf(my) > 1e-20f ? mx / my : 0, isl = fabsf(ly) > 1e-20f ? lx / ly : 0;
	float fy = floorf(y1) - y1;
	float xh = a->x + fy * ish, xm = a->x + fy * ism, xl = b->x;
	*e = (struct EdgeSetup) { a, b, c, { 0 }, hx, hy, mx, my, fabsf(nz) > 1e-20f ? -1.0f / nz : 0, ish, fy };
	e->w[0] = (uint64_t) ((uint32_t) op << 24 | lft << 23 | (uint32_t) (h->texLevel & 7) << 19 | (uint32_t) (h->texTile & 7) << 16 | (y3f & 0x3FFF)) << 32 | (uint32_t) ((y2f & 0x3FFF) << 16 | (y1f & 0x3FFF));
	e->w[1] = (uint64_t) (uint32_t) fixed16(xl) << 32 | (uint32_t) fixed16(isl);
	e->w[2] = (uint64_t) (uint32_t) fixed16(xh) << 32 | (uint32_t) fixed16(ish);
	e->w[3] = (uint64_t) (uint32_t) fixed16(xm) << 32 | (uint32_t) fixed16(ism);
}

// One RDP triangle (after libdragon's rdpq_triangle_cpu): edges, then shade, texture and depth
// coefficients as the command asks for them.
static void triangle(struct N64* n, struct Hle* h, const struct ScreenVertex* a, const struct ScreenVertex* b, const struct ScreenVertex* c, const struct ScreenVertex* flat) {
	// The microcodes scale 1/w so the corners' largest is a half (S2DEX's objects: whole), and send S
	// and T times it: the RDP divides that out with perspective correction and takes S and T as they
	// are without.
	float maxW = a->invW > b->invW ? (a->invW > c->invW ? a->invW : c->invW) : (b->invW > c->invW ? b->invW : c->invW);
	float k = maxW > 0 ? (h->objTriangles ? 1.0f : 0.5f) / maxW : 0.5f;
	struct EdgeSetup edge;
	edgeSetup(h, a, b, c, &edge);
	if (n->rdpGl && !h->image8 && !h->dry) {
		// For the GL renderer: the corners as they are (flat shading takes the one colour), beside
		// the RDP triangle, which keeps the list and its time the same.
		bool persp = h->modeH & 0x80000;
		struct GlrCorners q = { { edge.w[0], edge.w[1], edge.w[2], edge.w[3] },
		                        (uint32_t) ((h->geometry & GEOM_SHADE ? 4 : 0) | (h->texOn ? 2 : 0) | (h->geometry & GEOM_ZBUFFER ? 1 : 0)) |
		                        (uint32_t) (h->texTile & 7) << 8 | (uint32_t) (h->texLevel & 7) << 12 | (uint32_t) persp << 16, { { 0 } } };
		const struct ScreenVertex* corners[3] = { a, b, c };
		for (int j = 0; j < 3; ++j) {
			struct ScreenVertex v = *corners[j];
			if (flat) {
				v.r = flat->r, v.g = flat->g, v.b = flat->b, v.a = flat->a;
			}
			if (!persp) {
				v.s *= v.invW * k, v.t *= v.invW * k;
			}
			const float f[10] = { v.x, v.y, v.z, v.invW, v.r, v.g, v.b, v.a, v.s, v.t };
			memcpy(q.v[j], f, sizeof(f));
		}
		glrQueue(n, &q);
	}
	a = edge.a, b = edge.b, c = edge.c;
	bool shade = h->geometry & GEOM_SHADE, tex = h->texOn, z = h->geometry & GEOM_ZBUFFER;
	float hx = edge.hx, hy = edge.hy, mx = edge.mx, my = edge.my, attr = edge.attr, ish = edge.ish, fy = edge.fy;
	for (int i = 0; i < 4; ++i) {
		emit(n, h, edge.w[i]);
	}
	// One attribute's start, x and edge (y) slopes in s15.16 from its values at the three corners.
	#define GRADIENT(v1, v2, v3, start, dx, de, dy) do { \
		float m_ = (v2) - (v1), h_ = (v3) - (v1); \
		float dx_ = (hy * m_ - my * h_) * attr, dy_ = (mx * h_ - hx * m_) * attr; \
		float de_ = dy_ + dx_ * ish; \
		start = fixed16((v1) + fy * de_), dx = fixed16(dx_), de = fixed16(de_), dy = fixed16(dy_); \
	} while (0)
	if (shade) {
		const struct ScreenVertex* s1 = flat ? flat : a, * s2 = flat ? flat : b, * s3 = flat ? flat : c;
		int32_t st[4], dx[4], de[4], dy[4];
		GRADIENT(s1->r * 255, s2->r * 255, s3->r * 255, st[0], dx[0], de[0], dy[0]);
		GRADIENT(s1->g * 255, s2->g * 255, s3->g * 255, st[1], dx[1], de[1], dy[1]);
		GRADIENT(s1->b * 255, s2->b * 255, s3->b * 255, st[2], dx[2], de[2], dy[2]);
		GRADIENT(s1->a * 255, s2->a * 255, s3->a * 255, st[3], dx[3], de[3], dy[3]);
		const int32_t* rows[4] = { st, dx, de, dy };
		for (int r = 0; r < 4; r += 2) {
			const int32_t* p = rows[r], * q = rows[r + 1];
			emit(n, h, (uint64_t) (((uint32_t) p[0] & 0xFFFF0000u) | ((uint32_t) p[1] >> 16)) << 32 | (((uint32_t) p[2] & 0xFFFF0000u) | ((uint32_t) p[3] >> 16)));
			emit(n, h, (uint64_t) (((uint32_t) q[0] & 0xFFFF0000u) | ((uint32_t) q[1] >> 16)) << 32 | (((uint32_t) q[2] & 0xFFFF0000u) | ((uint32_t) q[3] >> 16)));
			emit(n, h, (uint64_t) ((uint32_t) p[0] << 16 | ((uint32_t) p[1] & 0xFFFF)) << 32 | ((uint32_t) p[2] << 16 | ((uint32_t) p[3] & 0xFFFF)));
			emit(n, h, (uint64_t) ((uint32_t) q[0] << 16 | ((uint32_t) q[1] & 0xFFFF)) << 32 | ((uint32_t) q[2] << 16 | ((uint32_t) q[3] & 0xFFFF)));
		}
	}
	if (tex) {
		float w1 = a->invW * k, w2 = b->invW * k, w3 = c->invW * k;
		int32_t st[3], dx[3], de[3], dy[3];
		GRADIENT(a->s * 32 * w1, b->s * 32 * w2, c->s * 32 * w3, st[0], dx[0], de[0], dy[0]);
		GRADIENT(a->t * 32 * w1, b->t * 32 * w2, c->t * 32 * w3, st[1], dx[1], de[1], dy[1]);
		GRADIENT(w1 * 0x7FFF, w2 * 0x7FFF, w3 * 0x7FFF, st[2], dx[2], de[2], dy[2]);
		const int32_t* rows[4] = { st, dx, de, dy };
		for (int r = 0; r < 4; r += 2) {
			const int32_t* p = rows[r], * q = rows[r + 1];
			emit(n, h, (uint64_t) (((uint32_t) p[0] & 0xFFFF0000u) | ((uint32_t) p[1] >> 16)) << 32 | ((uint32_t) p[2] & 0xFFFF0000u));
			emit(n, h, (uint64_t) (((uint32_t) q[0] & 0xFFFF0000u) | ((uint32_t) q[1] >> 16)) << 32 | ((uint32_t) q[2] & 0xFFFF0000u));
			emit(n, h, (uint64_t) ((uint32_t) p[0] << 16 | ((uint32_t) p[1] & 0xFFFF)) << 32 | ((uint32_t) p[2] << 16));
			emit(n, h, (uint64_t) ((uint32_t) q[0] << 16 | ((uint32_t) q[1] & 0xFFFF)) << 32 | ((uint32_t) q[2] << 16));
		}
	}
	if (z) {
		int32_t st, dx, de, dy;
		GRADIENT(a->z * 0x7FFF, b->z * 0x7FFF, c->z * 0x7FFF, st, dx, de, dy);
		emit(n, h, (uint64_t) (uint32_t) st << 32 | (uint32_t) dx);
		emit(n, h, (uint64_t) (uint32_t) de << 32 | (uint32_t) dy);
	}
	#undef GRADIENT
}

static struct ScreenVertex project(struct Hle* h, const struct Vertex* v) {
	struct ScreenVertex s;
	float iw = 1 / v->w;
	// On screen in quarter pixels, as the microcodes keep it.
	s.x = floorf((v->x * iw * h->vscale[0] + h->vtrans[0]) * 4) / 4;
	s.y = floorf((-v->y * iw * h->vscale[1] + h->vtrans[1]) * 4) / 4;
	float z = (v->z * iw * h->vscale[2] + h->vtrans[2]) * 32 / 0x7FFF;
	s.z = z < 0 ? 0 : z > 1 ? 1 : z;
	s.invW = iw;
	s.r = v->r, s.g = v->g, s.b = v->b, s.a = v->a;
	s.s = v->s, s.t = v->t;
	return s;
}

static struct Vertex lerp(const struct Vertex* a, const struct Vertex* b, float t) {
	struct Vertex v;
	const float* p = &a->x, * q = &b->x;
	float* o = &v.x;
	for (int i = 0; i < 10; ++i) {
		o[i] = p[i] + (q[i] - p[i]) * t;
	}
	return v;
}

// Clips against the near plane and a wide guard band, culls by facing on screen, and draws the
// rest as a fan.
static void drawTriangle(struct N64* n, struct Hle* h, int i0, int i1, int i2) {
	charge(n, h, 0, 1, 0);
	if (i0 >= VERTICES || i1 >= VERTICES || i2 >= VERTICES) {
		return;
	}
	// All three outside the same side of the view: rejected, as the microcodes do.
	uint32_t outside = 0x3F;
	for (int k = 0; k < 3; ++k) {
		const struct Vertex* v = &h->v[k == 0 ? i0 : k == 1 ? i1 : i2];
		outside &= (v->x < -v->w) | (v->x > v->w) << 1 | (v->y < -v->w) << 2 | (v->y > v->w) << 3 | (v->z < -v->w) << 4 | (v->z > v->w) << 5;
	}
	if (outside & 0xF) {
		return;
	}
	struct Vertex poly[2][9];
	int count = 3;
	poly[0][0] = h->v[i0], poly[0][1] = h->v[i1], poly[0][2] = h->v[i2];
	int cur = 0;
	// Planes as (x, y, z, w) weights: near, then x and y within the clip ratio's screens.
	float const r = h->clipRatio;
	float const planes[5][4] = { { 0, 0, 1, 1 }, { 1, 0, 0, r }, { -1, 0, 0, r }, { 0, 1, 0, r }, { 0, -1, 0, r } };
	// NoN skips the near plane only for a triangle wholly in front of the eye (one reaching behind it is
	// clipped there as usual: Conker's floors).
	bool const noNear = h->noNear && poly[0][0].w > 0 && poly[0][1].w > 0 && poly[0][2].w > 0;
	// Corners inside every plane leave nothing to clip.
	bool inside = true;
	for (int k = 0; k < 3 && inside; ++k) {
		const struct Vertex* v = &poly[0][k];
		inside = isfinite(v->z) && (noNear || v->z + v->w + 1e-4f * fabsf(v->w) >= 0) && fabsf(v->x) <= r * v->w && fabsf(v->y) <= r * v->w;
	}
	for (int p = inside ? 5 : 0; p < 5 && count >= 3; ++p) {
		const float* pl = planes[p];
		int out = 0;
		for (int i = 0; i < count; ++i) {
			const struct Vertex* a = &poly[cur][i], * b = &poly[cur][(i + 1) % count];
			// The near plane keeps what lies on it (2D layers at z = -w), give or take rounding.
			float da = a->x * pl[0] + a->y * pl[1] + a->z * pl[2] + a->w * pl[3] + (p == 0 ? 1e-4f * fabsf(a->w) : 0);
			float db = b->x * pl[0] + b->y * pl[1] + b->z * pl[2] + b->w * pl[3] + (p == 0 ? 1e-4f * fabsf(b->w) : 0);
			if (p == 0 && noNear) {
				da = db = 1;
			}
			if (da >= 0) {
				poly[cur ^ 1][out++] = *a;
			}
			if ((da >= 0) != (db >= 0) && out < 9) {
				poly[cur ^ 1][out++] = lerp(a, b, da / (da - db));
				// A new corner's fog comes from its own depth, as the microcodes work it out.
				if (h->geometry & GEOM_FOG) {
					poly[cur ^ 1][out - 1].a = fogAlpha(h, &poly[cur ^ 1][out - 1]);
				}
			}
		}
		count = out;
		cur ^= 1;
	}
	if (count < 3) {
		return;
	}
	struct ScreenVertex s[9];
	for (int i = 0; i < count; ++i) {
		s[i] = project(h, &poly[cur][i]);
	}
	float area = 0;
	for (int i = 1; i + 1 < count; ++i) {
		area += (s[i].x - s[0].x) * (s[i + 1].y - s[0].y) - (s[i + 1].x - s[0].x) * (s[i].y - s[0].y);
	}
	// Fast3D and F3DEX cull nothing with both faces' bits set.
	uint32_t cull = h->geometry & (GEOM_CULL_FRONT | GEOM_CULL_BACK);
	if (h->family != FAMILY_F3DEX2 && cull == (GEOM_CULL_FRONT | GEOM_CULL_BACK)) {
		cull = 0;
	}
	if ((area > 0 && (cull & GEOM_CULL_BACK)) || (area < 0 && (cull & GEOM_CULL_FRONT)) || area == 0) {
		return;
	}
	// Flat shading takes the colour of the triangle's first vertex.
	struct ScreenVertex flat;
	const struct ScreenVertex* f = NULL;
	if (!(h->geometry & GEOM_SHADING_SMOOTH)) {
		flat = project(h, &h->v[h->family == FAMILY_F3DEX2 ? i0 : i0]);
		f = &flat;
	}
	for (int i = 1; i + 1 < count; ++i) {
		triangle(n, h, &s[0], &s[i], &s[i + 1], f);
	}
}

// A line between two vertices as L3DEX draws it (GLideN64's gSPLine3D): 1.5 + width / 2 pixels wide,
// in the colour of the flag vertex; a quad of two triangles across it, the part behind the near
// plane cut off.
static void drawLine(struct N64* n, struct Hle* h, int i0, int i1, int width, int flag) {
	if (i0 >= VERTICES || i1 >= VERTICES || flag >= VERTICES) {
		return;
	}
	struct Vertex a = h->v[i0], b = h->v[i1];
	float const da = a.z + a.w, db = b.z + b.w;
	if (da < 0 && db < 0) {
		return;
	}
	if (da < 0) {
		a = lerp(&a, &b, da / (da - db));
	} else if (db < 0) {
		b = lerp(&b, &a, db / (db - da));
	}
	struct ScreenVertex sa = project(h, &a), sb = project(h, &b), c = project(h, &h->v[flag]);
	float dx = sb.x - sa.x, dy = sb.y - sa.y, length = sqrtf(dx * dx + dy * dy);
	if (!(length > 0)) {
		return;
	}
	float const half = (1.5f + width * 0.5f) / 2;
	float const px = -dy / length * half, py = dx / length * half;
	struct ScreenVertex q[4] = { sa, sa, sb, sb };
	q[0].x += px, q[0].y += py, q[1].x -= px, q[1].y -= py, q[2].x -= px, q[2].y -= py, q[3].x += px, q[3].y += py;
	triangle(n, h, &q[0], &q[1], &q[2], &c);
	triangle(n, h, &q[0], &q[2], &q[3], &c);
}

// L3DEX's triangle: its three edges, coloured by its first vertex.
static void drawEdges(struct N64* n, struct Hle* h, int v0, int v1, int v2, int width) {
	if (v0 != v1) {
		drawLine(n, h, v0, v1, width, v0);
	}
	if (v1 != v2) {
		drawLine(n, h, v1, v2, width, v0);
	}
	if (v2 != v0) {
		drawLine(n, h, v2, v0, width, v0);
	}
}

// ---- Display lists

static void setGeometry(struct Hle* h, uint32_t clear, uint32_t set, bool f3dex2) {
	if (!f3dex2) {
		// Fast3D and F3DEX bits to F3DEX2's.
		uint32_t bits[2] = { clear, set };
		for (int i = 0; i < 2; ++i) {
			uint32_t v = bits[i];
			bits[i] = (v & 0x1) | (v & 0x4) | (v & 0x1000 ? GEOM_CULL_FRONT : 0) | (v & 0x2000 ? GEOM_CULL_BACK : 0) |
			          (v & 0xF0000) | (v & 0x200 ? GEOM_SHADING_SMOOTH : 0);
		}
		clear = bits[0], set = bits[1];
	}
	h->geometry = (h->geometry & ~clear) | set;
}

static void readLight(struct N64* n, struct Hle* h, uint32_t a, int index) {
	if (index > LIGHTS) {
		return;
	}
	for (int i = 0; i < 3; ++i) {
		h->lightColor[index][i] = n->rdram[(a + i) & (RDRAM_SIZE - 1)] / 255.0f;
		h->lightDir[index][i] = rd8(n, a + 8 + i);
	}
	normalize(h->lightDir[index]);
}

static void readLookat(struct N64* n, struct Hle* h, uint32_t a, int index);

// G_MW_LIGHTCOL: a light's colour (the ambient one after the last light) replaced.
static void setLightColor(struct Hle* h, int index, uint32_t rgb) {
	if (index > LIGHTS) {
		return;
	}
	for (int i = 0; i < 3; ++i) {
		h->lightColor[index][i] = ((rgb >> (24 - 8 * i)) & 0xFF) / 255.0f;
	}
}

// The lookats and lights each task starts with: Fast3D's and F3DEX's data (DMEM from 0) as the
// microcode loads it, found by its G_MOVEMEM table's lookat Y, X and light places, ten 0x20 apart
// (Mario Kart 64's logo is lit and mapped by these alone).
static void defaultLights(struct N64* n, struct Hle* h, uint32_t data, uint32_t size) {
	for (uint32_t i = 0; i + 20 <= size; i += 2) {
		int k = 1;
		while (k < 10 && (uint16_t) rd16(n, data + i + 2 * k) == (uint16_t) rd16(n, data + i + 2 * k - 2) + 0x20) {
			++k;
		}
		uint32_t const last = (uint16_t) rd16(n, data + i + 18);
		if (k == 10 && last + 16 <= size) {
			readLookat(n, h, data + (uint16_t) rd16(n, data + i), 1);
			readLookat(n, h, data + (uint16_t) rd16(n, data + i + 2), 0);
			for (int l = 0; l < 8; ++l) {
				readLight(n, h, data + (uint16_t) rd16(n, data + i + 4 + 2 * l), l);
			}
			return;
		}
	}
}

static void readLookat(struct N64* n, struct Hle* h, uint32_t a, int index) {
	for (int i = 0; i < 3; ++i) {
		h->lookat[index][i] = rd8(n, a + 8 + i);
	}
	normalize(h->lookat[index]);
}

static void viewport(struct N64* n, struct Hle* h, uint32_t a) {
	for (int i = 0; i < 3; ++i) {
		h->vscale[i] = rd16(n, a + i * 2) / (i < 2 ? 4.0f : 1.0f);
		h->vtrans[i] = rd16(n, a + 8 + i * 2) / (i < 2 ? 4.0f : 1.0f);
	}
}

// G_BRANCH_Z: whether the vertex's depth on screen (16.16, the viewport's 0-0x3FF range) is at most
// the limit gSPBranchLessZ made.
static bool branchZ(const struct Hle* h, int index, uint32_t limit) {
	const struct Vertex* v = &h->v[index];
	float z = (v->w != 0 ? v->z / v->w : 0) * h->vscale[2] + h->vtrans[2];
	int64_t fixed = (int64_t) (z * 65536.0f);
	return fixed <= (int64_t) limit;
}

// The mask's bits cleared, then the data put in as it is (the microcodes don't mask it).
static void otherModeMask(struct N64* n, struct Hle* h, bool high, uint32_t mask, uint32_t bits) {
	uint32_t* m = high ? &h->modeH : &h->modeL;
	*m = (*m & ~mask) | bits;
	emit(n, h, (uint64_t) (0xEF000000u | (h->modeH & 0xFFFFFF)) << 32 | h->modeL);
}

static void otherMode(struct N64* n, struct Hle* h, bool high, int shift, int len, uint32_t bits) {
	otherModeMask(n, h, high, (len >= 32 ? 0xFFFFFFFFu : ((1u << len) - 1)) << shift, bits);
}

// The RDP commands a display list passes on as they are, with segmented image addresses resolved;
// a texture rectangle takes its second word from the next two commands.
static uint32_t rdpCommand(struct N64* n, struct Hle* h, uint32_t pc, uint32_t w0, uint32_t w1) {
	int op = w0 >> 24;
	if (op == 0xE4 || op == 0xE5) {
		uint32_t st = rd32(n, pc + 12), d = rd32(n, pc + 20);
		emit(n, h, (uint64_t) w0 << 32 | w1);
		emit(n, h, (uint64_t) st << 32 | d);
		return pc + 24;
	}
	if (op == 0xFD || op == 0xFE || op == 0xFF) {
		w1 = address(h, w1);
	}
	// Rare's texture shifts: each RGBA image moved by the table's next entry, unless its load block
	// doesn't come out whole lines (then the image as it was, and no more shifts).
	if (op == 0xFD && h->dkrTexTable) {
		if (((w0 >> 21) & 7) == 0) {
			h->dkrTexShift = (uint16_t) rd16(n, h->dkrTexTable + h->dkrTexCount * 2);
			h->dkrImage[0] = w0, h->dkrImage[1] = w1;
			w1 += h->dkrTexShift;
		} else {
			h->dkrTexTable = 0;
		}
	}
	if (op == 0xF3 && h->dkrTexTable) {
		if (h->dkrTexShift % (((((w1 >> 12) & 0xFFF) >> 2) + 1) << 3)) {
			emit(n, h, (uint64_t) h->dkrImage[0] << 32 | h->dkrImage[1]);
			h->dkrTexTable = 0;
		} else {
			++h->dkrTexCount;
		}
	}
	if (op == 0xFF) {
		h->image8 = ((w0 >> 19) & 3) == 1;
	}
	if (op == 0xEF) {
		h->modeH = w0 & 0xFFFFFF, h->modeL = w1;
	}
	emit(n, h, (uint64_t) w0 << 32 | w1);
	return pc + 8;
}

static void cullOrEnd(struct Hle* h, int first, int last, bool* end) {
	if (first < 0 || last >= VERTICES || first > last) {
		return;
	}
	// Ends the list when every vertex is outside the same side of the view (NoN draws what lies in
	// front of the near plane, so that side doesn't count).
	uint32_t all = h->noNear ? 0x2F : 0x3F;
	for (int i = first; i <= last; ++i) {
		const struct Vertex* v = &h->v[i];
		uint32_t code = (v->x < -v->w) | (v->x > v->w) << 1 | (v->y < -v->w) << 2 | (v->y > v->w) << 3 | (v->z < -v->w) << 4 | (v->z > v->w) << 5;
		all &= code;
	}
	if (all) {
		*end = true;
	}
}

// A Sprite2D sprite (image, palette, stride, width, height, format, size) from its first S and T,
// at x, y (10.2): its rows loaded in bands TMEM holds, each drawn as a texture rectangle.
static void spriteDraw(struct N64* n, struct Hle* h, int s0, int t0, int x, int y, bool older) {
	uint32_t a = h->sprite, w8 = rd32(n, a + 8), w12 = rd32(n, a + 12);
	uint32_t image = address(h, rd32(n, a)), tlut = rd32(n, a + 4) ? address(h, rd32(n, a + 4)) : 0;
	int stride = (int16_t) (w8 >> 16), width = (int16_t) w8, height = (int16_t) (w12 >> 16);
	int format = (w12 >> 8) & 0xFF, size = w12 & 0xFF;
	if (width <= 0 || height <= 0 || stride <= 0 || format > 4 || size > 3 || x < 0 || y < 0) {
		return;
	}
	// 4-bit texels load as 8-bit pairs; 32-bit ones fill both halves of TMEM.
	int loadSize = size ? size : 1, loadStride = size ? stride : stride / 2;
	int loadS0 = size ? s0 : s0 / 2, loadS1 = size ? s0 + width - 1 : (s0 + width - 1) / 2;
	uint32_t line = (uint32_t) ((loadS1 - loadS0 + 1) * (size == 3 ? 2 : 1 << (loadSize - 1)) + 7) / 8;
	uint32_t room = format == 2 || size == 3 ? 256 : 512;
	int band = (int) (room / line);
	if (band < 1) {
		return;
	}
	// A color-indexed sprite's palette, then RGBA16 colors looked up in it; other sprites without the
	// lookup (the older sprite command sets that only when it differs), as the microcode does.
	if (format == 2 && tlut) {
		emit(n, h, (uint64_t) 0x3D << 56 | (uint64_t) 2 << 51 | tlut);
		emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) 0x100 << 32 | (uint64_t) 7 << 24);
		emit(n, h, (uint64_t) 0x30 << 56 | (uint64_t) 7 << 24 | (uint64_t) ((size ? 255 : 15) << 2) << 12);
	}
	uint32_t lookup = format == 2 ? 2u << 14 : 0;
	if (format == 2 || !older || (h->modeH & (3u << 14)) != lookup) {
		otherMode(n, h, true, 14, 2, lookup);
	}
	bool copy = ((h->modeH >> 20) & 3) == 2;
	// The scales are texels per pixel.
	int dsdx = copy ? 4 << 10 : (int) (1024 * h->spriteScaleX), dtdy = copy ? 1 << 10 : (int) (1024 * h->spriteScaleY);
	float stepX = copy ? 1 : 1 / h->spriteScaleX, stepY = copy ? 1 : 1 / h->spriteScaleY;
	for (int t = 0; t < height; t += band) {
		int rows = height - t < band ? height - t : band;
		emit(n, h, (uint64_t) 0x3D << 56 | (uint64_t) format << 53 | (uint64_t) loadSize << 51 | (uint64_t) (loadStride - 1) << 32 | image);
		emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) format << 53 | (uint64_t) loadSize << 51 | (uint64_t) line << 41 | (uint64_t) 7 << 24 | 2 << 18 | 2 << 8);
		// 4-bit texels' columns as halves of the 8-bit ones (10.2), as the microcode gives them.
		int sl = size ? loadS0 << 2 : s0 * 2, sh = size ? loadS1 << 2 : (s0 + width - 1) * 2;
		emit(n, h, (uint64_t) 0x34 << 56 | (uint64_t) sl << 44 | (uint64_t) ((t0 + t) << 2) << 32 | (uint64_t) 7 << 24 | (uint64_t) sh << 12 |
		           (uint64_t) ((t0 + t + rows - 1) << 2));
		emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) format << 53 | (uint64_t) size << 51 | (uint64_t) line << 41 | 2 << 18 | 2 << 8);
		// The tile from the band's first texel (the loaded pair's first for 4-bit ones), as the
		// microcode makes it.
		int first = size ? 0 : s0 & 1;
		emit(n, h, (uint64_t) 0x32 << 56 | (uint64_t) ((width - 1) << 2) << 12 | (uint64_t) ((rows - 1) << 2));
		int yh = y + (int) (t * stepY * 4), yl = y + (int) ((t + rows) * stepY * 4) - (copy ? 4 : 0);
		int xh = x, xl = x + (int) (width * stepX * 4) - (copy ? 4 : 0);
		int s = (h->spriteFlipS ? first + width - 1 : first) << 5, tt = (h->spriteFlipT ? rows - 1 : 0) << 5;
		int ds = h->spriteFlipS ? -dsdx : dsdx, dt = h->spriteFlipT ? -dtdy : dtdy;
		emit(n, h, (uint64_t) 0x24 << 56 | (uint64_t) (xl & 0xFFF) << 44 | (uint64_t) (yl & 0xFFF) << 32 | (uint64_t) (xh & 0xFFF) << 12 | (uint64_t) (yh & 0xFFF));
		emit(n, h, (uint64_t) (s & 0xFFFF) << 48 | (uint64_t) (tt & 0xFFFF) << 32 | (uint64_t) (ds & 0xFFFF) << 16 | (uint64_t) (dt & 0xFFFF));
	}
}

// An S2DEX background (uObjBg at a): the image from (imageX, imageY) into the frame, wrapping round at
// the image's height; copy mode, or scaled (one cycle, scaleW and scaleH texels a pixel, 5.10). Rows
// of the image go through TMEM in bands, each drawn with a texture rectangle.
static void bgDraw(struct N64* n, struct Hle* h, uint32_t a, bool scaled) {
	a = address(h, a);
	int const imageX = (uint16_t) rd16(n, a), imageW = (uint16_t) rd16(n, a + 2), frameX = rd16(n, a + 4), frameW = (uint16_t) rd16(n, a + 6);
	int const imageY = (uint16_t) rd16(n, a + 8), imageH = (uint16_t) rd16(n, a + 10), frameY = rd16(n, a + 12), frameH = (uint16_t) rd16(n, a + 14);
	uint32_t const image = address(h, rd32(n, a + 16));
	int const format = (uint8_t) rd8(n, a + 22), size = (uint8_t) rd8(n, a + 23), pal = (uint16_t) rd16(n, a + 24) & 15;
	bool const flipS = rd16(n, a + 26) & 1;
	int const scaleW = scaled ? (uint16_t) rd16(n, a + 28) : 1024, scaleH = scaled ? (uint16_t) rd16(n, a + 30) : 1024;
	// Texels (the image's 10.2 sizes, its 10.5 position) and pixels (the frame's 10.2).
	int const width = imageW >> 2, height = imageH >> 2, cols = frameW >> 2, lines = frameH >> 2, s0 = imageX >> 5;
	if (width <= 0 || height <= 0 || cols <= 0 || lines <= 0 || format > 4 || size > 3 || !scaleW || !scaleH || s0 >= width) {
		return;
	}
	int texCols = (int) (((int64_t) cols * scaleW + 1023) >> 10) + (scaled ? 1 : 0);
	texCols = s0 + texCols > width ? width - s0 : texCols;
	// 4-bit texels load as 8-bit pairs; 32-bit and color-indexed ones have half of TMEM. A row of the
	// image is its bytes rounded down to a whole 8 (the microcode's step; a 1-cycle image's width has a
	// texel more for the filter).
	int const loadSize = size ? size : 1, loadStride = ((((width << size) >> 1) & ~7) >> (loadSize - 1));
	int const loadS0 = size ? s0 : s0 / 2, loadS1 = size ? s0 + texCols - 1 : (s0 + texCols - 1) / 2;
	uint32_t const line = (uint32_t) ((loadS1 - loadS0 + 1) * (size == 3 ? 2 : 1 << (loadSize - 1)) + 7) / 8;
	// A scaled band loads a row more for the filter, which has to fit too (or a palette is overwritten).
	int const band = (int) ((format == 2 || size == 3 ? 256u : 512u) / line) - (scaled ? 1 : 0);
	if (band < 1) {
		return;
	}
	bool const copy = ((h->modeH >> 20) & 3) == 2;
	int const dsdx = copy ? 4 << 10 : scaleW;
	// Down the frame: each band of image rows from where the frame has got to, up to the image's end.
	int64_t t = (int64_t) imageY << 5;
	for (int y = 0; y < lines;) {
		int64_t const tWrapped = t % ((int64_t) height << 10);
		int const row = (int) (tWrapped >> 10);
		int rows = height - row < band ? height - row : band;
		// The pixels this band covers (the next band takes over at its first row).
		int screen = (int) ((((int64_t) rows << 10) - (tWrapped & 1023) + scaleH - 1) / scaleH);
		screen = screen < 1 ? 1 : y + screen > lines ? lines - y : screen;
		emit(n, h, (uint64_t) 0x3D << 56 | (uint64_t) format << 53 | (uint64_t) loadSize << 51 | (uint64_t) (loadStride - 1) << 32 | image);
		emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) format << 53 | (uint64_t) loadSize << 51 | (uint64_t) line << 41 | (uint64_t) 7 << 24 | 2 << 18 | 2 << 8);
		int const sl = size ? loadS0 << 2 : s0 * 2, sh = size ? loadS1 << 2 : (s0 + texCols - 1) * 2;
		int const rowsLoaded = rows + (scaled && row + rows < height ? 1 : 0);
		emit(n, h, (uint64_t) 0x34 << 56 | (uint64_t) sl << 44 | (uint64_t) (row << 2) << 32 | (uint64_t) 7 << 24 | (uint64_t) sh << 12 |
		           (uint64_t) ((row + rowsLoaded - 1) << 2));
		emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) format << 53 | (uint64_t) size << 51 | (uint64_t) line << 41 | (uint64_t) pal << 20 | 2 << 18 | 2 << 8);
		emit(n, h, (uint64_t) 0x32 << 56 | (uint64_t) ((texCols - 1) << 2) << 12 | (uint64_t) ((rowsLoaded - 1) << 2));
		int const first = size ? 0 : s0 & 1;
		int xh = frameX, yh = frameY + (y << 2);
		int const xl = frameX + (cols << 2) - (copy ? 4 : 0), yl = frameY + ((y + screen) << 2) - (copy ? 4 : 0);
		int s = (flipS ? first + texCols - 1 : first) << 5, tt = (int) ((tWrapped & 1023) >> 5);
		int const ds = flipS ? -dsdx : dsdx, dt = copy ? 1 << 10 : scaleH;
		// A frame starting left of or above the screen: the rectangle from the edge, its texture as far on.
		if (xh < 0) {
			s += -xh * ds / 128, xh = 0;
		}
		if (yh < 0) {
			tt += -yh * dt / 128, yh = 0;
		}
		if (xl <= xh || yl < yh) {
			y += screen;
			t += (int64_t) screen * (copy ? 1024 : scaleH);
			continue;
		}
		emit(n, h, (uint64_t) 0x24 << 56 | (uint64_t) (xl & 0xFFF) << 44 | (uint64_t) (yl & 0xFFF) << 32 | (uint64_t) (xh & 0xFFF) << 12 | (uint64_t) (yh & 0xFFF));
		emit(n, h, (uint64_t) (s & 0xFFFF) << 48 | (uint64_t) (tt & 0xFFFF) << 32 | (uint64_t) (ds & 0xFFFF) << 16 | (uint64_t) (dt & 0xFFFF));
		y += screen;
		t += (int64_t) screen * (copy ? 1024 : scaleH);
	}
}

// S2DEX's texture load (uObjTxtr at a): a block, tile or palette into TMEM, unless the state word says
// it is there already.
static void objLoad(struct N64* n, struct Hle* h, uint32_t a) {
	a = address(h, a);
	uint32_t const type = rd32(n, a), image = address(h, rd32(n, a + 4)), sid = (uint16_t) rd16(n, a + 14) & 12;
	uint32_t const flag = rd32(n, a + 16), mask = rd32(n, a + 20), w8 = (uint16_t) rd16(n, a + 8);
	uint32_t const w10 = (uint16_t) rd16(n, a + 10), w12 = (uint16_t) rd16(n, a + 12);
	uint32_t* status = &h->objStatus[sid >> 2];
	if ((*status & mask) == flag) {
		return;
	}
	if (type == 0x1033) {
		emit(n, h, (uint64_t) 0x3D << 56 | (uint64_t) 2 << 51 | (uint64_t) w10 << 32 | image);
		emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) 2 << 51 | (uint64_t) w8 << 32 | (uint64_t) 7 << 24);
		// tsize counts 16-bit texels in fours.
		emit(n, h, (uint64_t) 0x33 << 56 | (uint64_t) 7 << 24 | (uint64_t) ((w10 << 2) & 0xFFF) << 12 | (w12 & 0xFFF));
	} else if (type == 0xFC1034) {
		// A tile: twidth + 1 16-bit texels a row (a line of a quarter of that), theight its 10.2 height.
		emit(n, h, (uint64_t) 0x3D << 56 | (uint64_t) 2 << 51 | (uint64_t) (w10 & 0x3FF) << 32 | image);
		emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) 2 << 51 | (uint64_t) ((w10 + 1) >> 2) << 41 | (uint64_t) w8 << 32 | (uint64_t) 7 << 24);
		emit(n, h, (uint64_t) 0x34 << 56 | (uint64_t) 7 << 24 | (uint64_t) ((w10 << 2) & 0xFFF) << 12 | (w12 & 0xFFF));
	} else if (type == 0x30) {
		emit(n, h, (uint64_t) 0x3D << 56 | (uint64_t) 2 << 51 | image);
		emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) w8 << 32 | (uint64_t) 7 << 24);
		emit(n, h, (uint64_t) 0x30 << 56 | (uint64_t) 7 << 24 | (uint64_t) ((w10 << 2) & 0xFFF) << 12);
	}
	*status = (*status & ~mask) | (flag & mask);
}

// An S2DEX object (uObjSprite at a) drawn from TMEM with a texture rectangle: at its place and scale,
// or through the 2D matrix (sub: its offset and base scales; full: also A and D, when it doesn't
// rotate). False for a rotated one.
// An S2DEX object the 2D matrix rotates or mirrors: its corners through the matrix, drawn as two
// textured triangles (as GLideN64 does). Copy mode draws only rectangles: false.
static bool objRotated(struct N64* n, struct Hle* h, float objX, float objY, int width, int height, float dsdx, float dtdy, int stride, int tmem,
                       int format, int size, int pal, int flags) {
	if (((h->modeH >> 20) & 3) == 2) {
		return false;
	}
	bool const clamp = !(h->objMode & 0x01), flipS = flags & 0x01, flipT = flags & 0x10;
	emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) format << 53 | (uint64_t) size << 51 | (uint64_t) stride << 41 | (uint64_t) tmem << 32 |
	           (uint64_t) pal << 20 | (uint64_t) clamp << 19 | (uint64_t) clamp << 9);
	emit(n, h, (uint64_t) 0x32 << 56 | (uint64_t) ((width - 1) << 2) << 12 | (uint64_t) ((height - 1) << 2));
	// The far corner a pixel in, its texel where the scale puts it there.
	float const lx[2] = { objX, objX + width / dsdx - 1 }, ly[2] = { objY, objY + height / dtdy - 1 };
	float const ls[2] = { 0, width - dsdx }, lt[2] = { 0, height - dtdy };
	struct ScreenVertex v[4];
	for (int i = 0; i < 4; ++i) {
		float const px = lx[i & 1], py = ly[i >> 1];
		v[i] = (struct ScreenVertex) { h->objA * px + h->objB * py + h->objX, h->objC * px + h->objD * py + h->objY, 0, 1, 1, 1, 1, 1,
		                               flipS ? ls[!(i & 1)] : ls[i & 1], flipT ? lt[!(i >> 1)] : lt[i >> 1] };
	}
	uint32_t const geometry = h->geometry;
	bool const texOn = h->texOn;
	int const texTile = h->texTile, texLevel = h->texLevel;
	h->geometry &= ~(uint32_t) (GEOM_SHADE | GEOM_ZBUFFER);
	h->texOn = true, h->texTile = 0, h->texLevel = 0, h->objTriangles = true;
	triangle(n, h, &v[0], &v[1], &v[2], NULL);
	triangle(n, h, &v[1], &v[3], &v[2], NULL);
	h->geometry = geometry, h->texOn = texOn, h->texTile = texTile, h->texLevel = texLevel, h->objTriangles = false;
	return true;
}

static bool objDraw(struct N64* n, struct Hle* h, uint32_t a, int matrix) {
	a = address(h, a);
	float const objX = rd16(n, a) / 4.0f, objY = rd16(n, a + 8) / 4.0f;
	int const scaleW = (uint16_t) rd16(n, a + 2) ? (uint16_t) rd16(n, a + 2) : 1, scaleH = (uint16_t) rd16(n, a + 10) ? (uint16_t) rd16(n, a + 10) : 1;
	int const width = (uint16_t) rd16(n, a + 4) >> 5, height = (uint16_t) rd16(n, a + 12) >> 5;
	int const stride = (uint16_t) rd16(n, a + 16), tmem = (uint16_t) rd16(n, a + 18);
	int const format = (uint8_t) rd8(n, a + 20), size = (uint8_t) rd8(n, a + 21), pal = (uint8_t) rd8(n, a + 22) & 15, flags = (uint8_t) rd8(n, a + 23);
	if (width <= 0 || height <= 0) {
		return true;
	}
	// Texels a pixel (5.10) and the upper-left corner in pixels.
	float dsdx = scaleW / 1024.0f, dtdy = scaleH / 1024.0f, x = objX, y = objY;
	if (matrix == 1) {
		x = objX * 1024.0f / h->objBaseX + h->objX, y = objY * 1024.0f / h->objBaseY + h->objY;
		dsdx *= h->objBaseX / 1024.0f, dtdy *= h->objBaseY / 1024.0f;
	} else if (matrix == 2) {
		if (h->objB != 0 || h->objC != 0 || h->objA <= 0 || h->objD <= 0) {
			return objRotated(n, h, objX, objY, width, height, dsdx, dtdy, stride, tmem, format, size, pal, flags);
		}
		x = objX * h->objA + h->objX, y = objY * h->objD + h->objY;
		dsdx /= h->objA, dtdy /= h->objD;
	}
	bool const clamp = !(h->objMode & 0x01), copy = ((h->modeH >> 20) & 3) == 2;
	emit(n, h, (uint64_t) 0x35 << 56 | (uint64_t) format << 53 | (uint64_t) size << 51 | (uint64_t) stride << 41 | (uint64_t) tmem << 32 |
	           (uint64_t) pal << 20 | (uint64_t) clamp << 19 | (uint64_t) clamp << 9);
	emit(n, h, (uint64_t) 0x32 << 56 | (uint64_t) ((width - 1) << 2) << 12 | (uint64_t) ((height - 1) << 2));
	int const xh = (int) (x * 4), yh = (int) (y * 4);
	int const xl = (int) ((x + width / dsdx) * 4) - (copy ? 4 : 0), yl = (int) ((y + height / dtdy) * 4) - (copy ? 4 : 0);
	bool const flipS = flags & 0x01, flipT = flags & 0x10;
	int const s = flipS ? (width - 1) << 5 : 0, t = flipT ? (height - 1) << 5 : 0;
	int const ds = (int) ((copy ? 4 : dsdx) * 1024) * (flipS ? -1 : 1), dt = (int) ((copy ? 1 : dtdy) * 1024) * (flipT ? -1 : 1);
	emit(n, h, (uint64_t) 0x24 << 56 | (uint64_t) (xl & 0xFFF) << 44 | (uint64_t) (yl & 0xFFF) << 32 | (uint64_t) (xh & 0xFFF) << 12 | (uint64_t) (yh & 0xFFF));
	emit(n, h, (uint64_t) (s & 0xFFFF) << 48 | (uint64_t) (t & 0xFFFF) << 32 | (uint64_t) (ds & 0xFFFF) << 16 | (uint64_t) (dt & 0xFFFF));
	return true;
}

// The task's microcode left to the RSP, this task (from the start) and later ones.
static void reject(struct Hle* h) {
	if (h->rejectedCount < 8) {
		h->rejected[h->rejectedCount++] = h->ucode;
	}
	h->rejectedNow = true;
}

static int family(struct N64* n, uint32_t code, uint32_t data, uint32_t size);
static bool isFlx(struct N64* n, uint32_t code, uint32_t data, uint32_t size);
static void ucodeFlags(struct N64* n, struct Hle* h, uint32_t code, uint32_t data, uint32_t size);
static void ucodeName(struct N64* n, uint32_t code, uint32_t data, uint32_t dataSize, char* out, size_t size);

// G_LOAD_UCODE: the list goes on in another microcode (its data named by G_RDPHALF_1); F3DEX's takes
// the new data's other modes with it, F3DEX2's and S2DEX2's keep their own (Kirby 64's menus). One of
// no family here leaves the task to the RSP.
static bool loadMicrocode(struct N64* n, struct Hle* h, uint32_t code, uint32_t dataSize) {
	// Conker's loads its advanced lighting variant, with the same commands (its data names itself
	// F3DEX2, or nothing).
	if (h->cbfd) {
		h->cbfdAdvanced = true;
		return true;
	}
	int f = family(n, address(h, code), address(h, h->half1), dataSize);
	h->flx = isFlx(n, address(h, code), address(h, h->half1), dataSize);
	ucodeFlags(n, h, address(h, code), address(h, h->half1), dataSize);
	if (f == FAMILY_NONE) {
		char name[40];
		ucodeName(n, address(h, code), address(h, h->half1), dataSize, name, sizeof(name));
		snprintf(h->why, sizeof(h->why), "to %s", name);
		reject(h);
		return false;
	}
	if (h->family != FAMILY_F3DEX2 && h->family != FAMILY_S2DEX2) {
		h->modeH = 0x080CFF, h->modeL = 0;
	}
	h->family = f;
	return true;
}

// ---- ZSort (as GLideN64 has it): DMEM fields carry a 0x400 bias; matrices are loaded and combined
// on command; the game reads back what it had transformed, sorts, and sends objects in screen space.

static uint32_t zField(uint32_t field) {
	return (field - 0x400) & 0xFFF;
}

static uint32_t zRd16(const struct Hle* h, uint32_t a) {
	return (uint32_t) h->zMem[a & 0xFFF] << 8 | h->zMem[(a + 1) & 0xFFF];
}

static void zWr16(struct Hle* h, uint32_t a, uint32_t v) {
	h->zMem[a & 0xFFF] = (uint8_t) (v >> 8), h->zMem[(a + 1) & 0xFFF] = (uint8_t) v;
}

// An RDP command list in RDRAM, to the end command (0xDF).
static void zsortRdp(struct N64* n, struct Hle* h, uint32_t a) {
	for (int k = 0; a && k < 0x10000; ++k) {
		uint32_t const w0 = rd32(n, a), w1 = rd32(n, a + 4);
		int const op = w0 >> 24;
		if (op == 0xDF) {
			break;
		}
		if (op == 0xE2 || op == 0xE3) {
			otherMode(n, h, op == 0xE3, (w0 >> 8) & 0xFF, w0 & 0xFF, w1);
			a += 8;
		} else {
			a = rdpCommand(n, h, a, w0, w1);
		}
	}
}

// One object's triangle or quad (a strip): screen positions (10.2), colours and, textured, S and T
// (10.5 with perspective, else 10.6) and the 1/w the transform gave (as 0x7FFFFFFF / (w * 31)).
static void zsortDraw(struct N64* n, struct Hle* h, uint32_t a, int type) {
	bool const textured = type == 2 || type == 4;
	int const count = type <= 2 ? 3 : 4, size = textured ? 16 : 8;
	bool const persp = h->modeH & 0x80000;
	struct ScreenVertex v[4];
	for (int i = 0; i < count; ++i, a += (uint32_t) size) {
		float w = 1;
		if (textured) {
			int32_t const invw = (int32_t) rd32(n, a + 12);
			w = invw ? (float) (0x7FFFFFFF / invw) / 31.0f : 1;
		}
		v[i] = (struct ScreenVertex) { rd16(n, a) / 4.0f, rd16(n, a + 2) / 4.0f, 0, w != 0 ? 1 / w : 1,
		                               n->rdram[(a + 4) & (RDRAM_SIZE - 1)] / 255.0f, n->rdram[(a + 5) & (RDRAM_SIZE - 1)] / 255.0f,
		                               n->rdram[(a + 6) & (RDRAM_SIZE - 1)] / 255.0f, n->rdram[(a + 7) & (RDRAM_SIZE - 1)] / 255.0f,
		                               textured ? rd16(n, a + 8) / (persp ? 32.0f : 64.0f) : 0, textured ? rd16(n, a + 10) / (persp ? 32.0f : 64.0f) : 0 };
	}
	bool const texOn = h->texOn;
	h->texOn = textured;
	triangle(n, h, &v[0], &v[1], &v[2], NULL);
	if (count == 4) {
		triangle(n, h, &v[2], &v[1], &v[3], NULL);
	}
	h->texOn = texOn;
}

// G_ZOBJ: a list of objects, each with its RDP lists (run when they changed) and its vertices.
static void zsortObjects(struct N64* n, struct Hle* h, uint32_t header) {
	for (int k = 0; header && k < 0x10000; ++k) {
		uint32_t const a = header & ~7u;
		int const type = header & 7;
		int const lists = type == 1 || type == 3 ? 1 : 3;
		for (int i = 0; i < lists; ++i) {
			uint32_t const w1 = rd32(n, a + 4 + 4 * (uint32_t) i);
			if (w1 != h->zRdp[i]) {
				h->zRdp[i] = w1;
				zsortRdp(n, h, address(h, w1));
			}
		}
		if (type >= 1 && type <= 4) {
			zsortDraw(n, h, a + (lists == 1 ? 8 : 16), type);
		}
		uint32_t const next = rd32(n, a);
		header = next ? address(h, next) : 0;
	}
}

// G_ZMULT_MPMTX: vertices (three s16 each) through the combined matrix into screen records: x and y
// on screen (10.2), 1/w, x, y and w as integers, the clip flags and fog.
static void zsortTransform(struct Hle* h, uint32_t w1) {
	int const count = 1 + (int) (w1 >> 24);
	uint32_t src = zField((w1 >> 12) & 0xFFF), dst = zField(w1 & 0xFFF);
	const float* m = h->mvp;
	for (int i = 0; i < count; ++i, src += 6, dst += 16) {
		float const x = (int16_t) zRd16(h, src), y = (int16_t) zRd16(h, src + 2), z = (int16_t) zRd16(h, src + 4);
		float const X = x * m[0] + y * m[4] + z * m[8] + m[12], Y = x * m[1] + y * m[5] + z * m[9] + m[13];
		float const Z = x * m[2] + y * m[6] + z * m[10] + m[14], W = x * m[3] + y * m[7] + z * m[11] + m[15];
		int const sx = W != 0 ? (int) (h->zTrans[0] + X / W * h->zScale[0]) : 0, sy = W != 0 ? (int) (h->zTrans[1] + Y / W * h->zScale[1]) : 0;
		int const w31 = (int) (W * 31.0f);
		uint32_t const invw = w31 ? (uint32_t) (0x7FFFFFFF / w31) : 0x7FFFFFFFu;
		int fog = 0;
		if (W >= 0 && W != 0) {
			fog = (int) (Z / W * h->fogMul + h->fogOffset);
			fog = fog > 255 ? 255 : fog < 0 ? 0 : fog;
		}
		int const cc = (X < -W ? 0x10 : 0) | (X > W ? 0x01 : 0) | (Y < -W ? 0x20 : 0) | (Y > W ? 0x02 : 0) | (W < 0.1f ? 0x04 : 0);
		zWr16(h, dst, (uint32_t) sx), zWr16(h, dst + 2, (uint32_t) sy);
		zWr16(h, dst + 4, invw >> 16), zWr16(h, dst + 6, invw);
		zWr16(h, dst + 8, (uint32_t) (int) X), zWr16(h, dst + 10, (uint32_t) (int) Y);
		h->zMem[(dst + 12) & 0xFFF] = (uint8_t) cc, h->zMem[(dst + 13) & 0xFFF] = (uint8_t) fog;
		zWr16(h, dst + 14, (uint32_t) (int) W);
	}
}

static float* zsortMatrix(struct Hle* h, int which) {
	return which == 4 ? h->modelview[h->depth] : which == 6 ? h->projection : which == 8 ? h->mvp : NULL;
}

// G_ZXFMLIGHT: the ambient colour, the lights (colour, direction) and the two lookat directions, from
// DMEM.
static void zsortLights(struct Hle* h, uint32_t w1) {
	int const count = 1 + (int) ((w1 >> 12) & 0xFF);
	uint32_t a = zField(w1 & 0xFFF);
	h->lights = count > LIGHTS ? LIGHTS : count;
	for (int k = 0; k < 3; ++k) {
		h->lightColor[h->lights][k] = h->zMem[(a + (uint32_t) k) & 0xFFF] / 255.0f;
	}
	a += 8;
	for (int l = 0; l < count; ++l, a += 24) {
		for (int k = 0; l < LIGHTS && k < 3; ++k) {
			h->lightColor[l][k] = h->zMem[(a + (uint32_t) k) & 0xFFF] / 255.0f;
			h->lightDir[l][k] = (int8_t) h->zMem[(a + 8 + (uint32_t) k) & 0xFFF];
		}
		if (l < LIGHTS) {
			normalize(h->lightDir[l]);
		}
	}
	for (int l = 0; l < 2; ++l, a += 24) {
		for (int k = 0; k < 3; ++k) {
			h->lookat[l][k] = (int8_t) h->zMem[(a + 8 + (uint32_t) k) & 0xFFF];
		}
		normalize(h->lookat[l]);
	}
}

// G_ZLIGHTING: each vertex's normal through the model matrix, lit, times its material colour; S and T
// from the normal through the projection against the lookat directions (sphere mapping).
static void zsortLighting(struct Hle* h, uint32_t w0, uint32_t w1) {
	int const count = 1 + (int) (w1 >> 24);
	uint32_t csrs = zField((w0 >> 12) & 0xFFF), nsrs = zField(w0 & 0xFFF), cdest = zField((w1 >> 12) & 0xFFF), tdest = zField(w1 & 0xFFF);
	const float* mv = h->modelview[h->depth];
	const float* pr = h->projection;
	for (int i = 0; i < count; ++i, nsrs += 3, csrs += 4, cdest += 4, tdest += 4) {
		float const nx = (int8_t) h->zMem[nsrs & 0xFFF], ny = (int8_t) h->zMem[(nsrs + 1) & 0xFFF], nz = (int8_t) h->zMem[(nsrs + 2) & 0xFFF];
		float e[3] = { nx * mv[0] + ny * mv[4] + nz * mv[8], nx * mv[1] + ny * mv[5] + nz * mv[9], nx * mv[2] + ny * mv[6] + nz * mv[10] };
		normalize(e);
		float rgb[3] = { h->lightColor[h->lights][0], h->lightColor[h->lights][1], h->lightColor[h->lights][2] };
		for (int l = 0; l < h->lights; ++l) {
			float const d = e[0] * h->lightDir[l][0] + e[1] * h->lightDir[l][1] + e[2] * h->lightDir[l][2];
			if (d > 0) {
				rgb[0] += d * h->lightColor[l][0], rgb[1] += d * h->lightColor[l][1], rgb[2] += d * h->lightColor[l][2];
			}
		}
		float p[3] = { e[0] * pr[0] + e[1] * pr[4] + e[2] * pr[8], e[0] * pr[1] + e[1] * pr[5] + e[2] * pr[9], e[0] * pr[2] + e[1] * pr[6] + e[2] * pr[10] };
		normalize(p);
		float const x = p[0] * h->lookat[0][0] + p[1] * h->lookat[0][1] + p[2] * h->lookat[0][2];
		float const y = p[0] * h->lookat[1][0] + p[1] * h->lookat[1][1] + p[2] * h->lookat[1][2];
		for (int k = 0; k < 3; ++k) {
			float const c = (rgb[k] > 1 ? 1 : rgb[k]) * h->zMem[(csrs + (uint32_t) k) & 0xFFF] / 255.0f;
			h->zMem[(cdest + (uint32_t) k) & 0xFFF] = (uint8_t) (c * 255.0f);
		}
		h->zMem[(cdest + 3) & 0xFFF] = h->zMem[(csrs + 3) & 0xFFF];
		zWr16(h, tdest, (uint32_t) (int) ((x + 1) * 512 * 32)), zWr16(h, tdest + 2, (uint32_t) (int) ((y + 1) * 512 * 32));
	}
}

// G_ZMOVEMEM: DMEM loaded from or saved to RDRAM, a matrix, or the viewport (which also turns on the
// texture and smooth shading the objects take).
static void zsortMoveMem(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	int const index = w0 & 0x0E;
	uint32_t const offset = ((w0 >> 6) & 0x1FF) << 3, length = 1 + (((w0 >> 15) & 0x1FF) << 3), a = address(h, w1);
	if (index == 0) {
		if (!(w0 & 1)) {
			for (uint32_t i = 0; i < length; ++i) {
				h->zMem[(offset + i) & 0xFFF] = n->rdram[(a + i) & (RDRAM_SIZE - 1)];
			}
		} else if (!h->dry) {
			n64RdramWrite(n, a, length);
			for (uint32_t i = 0; i < length; ++i) {
				n->rdram[(a + i) & (RDRAM_SIZE - 1)] = h->zMem[(offset + i) & 0xFFF];
			}
		}
	} else if (zsortMatrix(h, index)) {
		readMatrix(n, a, zsortMatrix(h, index));
	} else if (index == 12) {
		viewport(n, h, a);
		h->fogMul = rd16(n, a + 6), h->fogOffset = rd16(n, a + 14);
		for (int k = 0; k < 2; ++k) {
			h->zScale[k] = h->vscale[k] * 4, h->zTrans[k] = h->vtrans[k] * 4;
		}
		h->texScaleS = h->texScaleT = 1, h->texLevel = 0, h->texTile = 0, h->texOn = true;
		h->geometry |= GEOM_SHADE | GEOM_SHADING_SMOOTH;
	}
}

// ---- Turbo3D (Dark Rift's; libultra's gt.h, as GLideN64's Turbo3D.cpp runs it)

// An RDP list as it is (a texture rectangle's second word next), up to two zero words.
static void turbo3dRdp(struct N64* n, struct Hle* h, uint32_t a) {
	if (!a) {
		return;
	}
	a = address(h, a);
	for (int steps = 0; steps < 1 << 16; ++steps) {
		uint32_t const w0 = rd32(n, a), w1 = rd32(n, a + 4);
		if (!w0 && !w1) {
			return;
		}
		int const op = w0 >> 24;
		if (op == 0xE4 || op == 0xE5) {
			emit(n, h, (uint64_t) w0 << 32 | w1);
			emit(n, h, (uint64_t) rd32(n, a + 8) << 32 | rd32(n, a + 12));
			a += 16;
		} else {
			if (op == 0xF5) {
				h->t3duxTile[0] = w0, h->t3duxTile[1] = w1;
			}
			a = rdpCommand(n, h, a, w0, w1);
		}
	}
}

// An object (gtStateL at state): its other modes and matrix, vertices, RDP list, then triangles.
static void turbo3dObject(struct N64* n, struct Hle* h, uint32_t state, uint32_t vtx, uint32_t tri) {
	uint32_t const a = address(h, state);
	uint32_t const render = rd32(n, a), texture = rd32(n, a + 4);
	int const count = rd8(n, a + 8) & 0xFF, first = rd8(n, a + 9) & 0xFF, tris = rd8(n, a + 10) & 0xFF, flag = rd8(n, a + 11) & 0xFF;
	h->texTile = texture & 7, h->texLevel = 0, h->texOn = (render & 2) != 0;
	h->texScaleS = h->texScaleT = 1;
	h->modeH = rd32(n, a + 16) & 0xFFFFFF, h->modeL = rd32(n, a + 20);
	emit(n, h, (uint64_t) (0xEF000000u | h->modeH) << 32 | h->modeL);
	// GT_FLAG_NOMTX keeps the last matrix; otherwise the combined one follows the state.
	if (flag != 1) {
		readMatrix(n, a + 24, h->mvp);
	}
	setGeometry(h, GEOM_LIGHTING | GEOM_FOG, render | 0x2205, false);
	// GT_FLAG_NO_XFM: the vertices are on the screen already (gtVtxOut).
	if (flag == 2 && vtx && tri) {
		uint32_t const v = address(h, vtx), t = address(h, tri);
		turbo3dRdp(n, h, rd32(n, a + 12));
		for (int i = 0; i < tris; ++i) {
			int const k[3] = { rd8(n, t + 4 * (uint32_t) i) & 0xFF, rd8(n, t + 4 * (uint32_t) i + 1) & 0xFF, rd8(n, t + 4 * (uint32_t) i + 2) & 0xFF };
			if (k[0] >= count || k[1] >= count || k[2] >= count) {
				continue;
			}
			struct ScreenVertex c[3];
			for (int j = 0; j < 3; ++j) {
				uint32_t const e = v + 16 * (uint32_t) k[j];
				float const z = (int32_t) rd32(n, e + 4) / 65536.0f * 32 / 0x7FFF;
				c[j] = (struct ScreenVertex) { rd16(n, e) / 4.0f, rd16(n, e + 2) / 4.0f, z < 0 ? 0 : z > 1 ? 1 : z, 1,
				                               (rd8(n, e + 12) & 0xFF) / 255.0f, (rd8(n, e + 13) & 0xFF) / 255.0f, (rd8(n, e + 14) & 0xFF) / 255.0f,
				                               (rd8(n, e + 15) & 0xFF) / 255.0f, rd16(n, e + 8) / 32.0f, rd16(n, e + 10) / 32.0f };
			}
			triangle(n, h, &c[0], &c[1], &c[2], NULL);
		}
		return;
	}
	if (vtx) {
		loadVertices(n, h, vtx, first, count);
	}
	turbo3dRdp(n, h, rd32(n, a + 12));
	if (tri) {
		uint32_t const t = address(h, tri);
		for (int i = 0; i < tris; ++i) {
			drawTriangle(n, h, rd8(n, t + 4 * (uint32_t) i) & 0xFF, rd8(n, t + 4 * (uint32_t) i + 1) & 0xFF, rd8(n, t + 4 * (uint32_t) i + 2) & 0xFF);
		}
	}
}

// T3DUX's object (Toukon Road's; GLideN64's T3DUX.cpp): 8-byte vertices (x, y, z, flag) with their
// colours in the table at col, and 8-byte triangles (vertices, flat colour index, the table's texture
// coordinates for each corner, a palette).
static void t3duxObject(struct N64* n, struct Hle* h, uint32_t state, uint32_t vtx, uint32_t tri, uint32_t col) {
	uint32_t const a = address(h, state), colors = address(h, col);
	uint32_t const render = rd32(n, a);
	int const geomMode = rd8(n, a + 4) & 0xFF, texMode = rd8(n, a + 5) & 0xFF, count = rd8(n, a + 6) & 0xFF;
	int const tris = rd8(n, a + 8) & 0xFF, matrixFlag = rd8(n, a + 9) & 0xFF;
	bool const texturing = texMode != 1, flat = !(geomMode & 15);
	h->texTile = 0, h->texLevel = 0, h->texOn = texturing;
	h->texScaleS = h->texScaleT = 1;
	h->modeH = rd32(n, a + 16) & 0xFFFFFF, h->modeL = rd32(n, a + 20);
	emit(n, h, (uint64_t) (0xEF000000u | h->modeH) << 32 | h->modeL);
	if (!(matrixFlag & 1)) {
		readMatrix(n, a + 24, h->mvp);
	}
	setGeometry(h, GEOM_LIGHTING | GEOM_FOG, render | 0x2205, false);
	if (vtx) {
		uint32_t const v = address(h, vtx);
		const float* m = h->mvp;
		for (int i = 0; i < count; ++i) {
			float const x = rd16(n, v + 8 * (uint32_t) i), y = rd16(n, v + 8 * (uint32_t) i + 2), z = rd16(n, v + 8 * (uint32_t) i + 4);
			uint32_t const c = colors + 4 * (uint32_t) i;
			h->t3dux[i] = (struct Vertex) { x * m[0] + y * m[4] + z * m[8] + m[12], x * m[1] + y * m[5] + z * m[9] + m[13],
			                                x * m[2] + y * m[6] + z * m[10] + m[14], x * m[3] + y * m[7] + z * m[11] + m[15],
			                                (rd8(n, c) & 0xFF) / 255.0f, (rd8(n, c + 1) & 0xFF) / 255.0f, (rd8(n, c + 2) & 0xFF) / 255.0f,
			                                (rd8(n, c + 3) & 0xFF) / 255.0f, 0, 0 };
		}
	}
	turbo3dRdp(n, h, rd32(n, a + 12));
	if (!tri) {
		return;
	}
	// A triangle's palette byte, when it has one, goes in at the tile's palette (bit 20) of the SETTILE
	// the RDP list gave.
	uint32_t tileW1 = h->t3duxTile[1];
	h->t3duxTile[1] &= 0xFF0FFFFF;
	uint32_t const t = address(h, tri);
	for (int i = 0; i < tris; ++i) {
		uint32_t const e = t + 8 * (uint32_t) i;
		int const k[3] = { rd8(n, e) & 0xFF, rd8(n, e + 1) & 0xFF, rd8(n, e + 2) & 0xFF }, flag = rd8(n, e + 3) & 0xFF;
		int const tex[3] = { rd8(n, e + 4) & 0xFF, rd8(n, e + 5) & 0xFF, rd8(n, e + 6) & 0xFF }, triPal = rd8(n, e + 7) & 0xFF;
		if (texturing && triPal && (h->t3duxTile[1] | (uint32_t) triPal << 20) != tileW1) {
			tileW1 = h->t3duxTile[1] | (uint32_t) triPal << 20;
			emit(n, h, (uint64_t) h->t3duxTile[0] << 32 | tileW1);
		}
		if (k[0] >= count || k[1] >= count || k[2] >= count) {
			continue;
		}
		uint32_t const flatColor = colors + (((uint32_t) flag << 2) & 0x3FC);
		for (int j = 0; j < 3; ++j) {
			struct Vertex* v = &h->v[j];
			*v = h->t3dux[k[j]];
			if (texturing) {
				uint32_t const st = rd32(n, colors + ((uint32_t) tex[j] << 2));
				v->s = (int16_t) (st >> 16) / 32.0f, v->t = (int16_t) st / 32.0f;
			}
			if (flat) {
				v->r = (rd8(n, flatColor) & 0xFF) / 255.0f, v->g = (rd8(n, flatColor + 1) & 0xFF) / 255.0f;
				v->b = (rd8(n, flatColor + 2) & 0xFF) / 255.0f, v->a = (rd8(n, flatColor + 3) & 0xFF) / 255.0f;
			}
		}
		drawTriangle(n, h, 0, 1, 2);
	}
}

// The task's list: gtGfx entries (global state, object state, vertices, triangles; T3DUX's also a
// colour table and a store, 24 bytes) up to an object state of zero. A global state brings other
// modes, segments, the viewport and an RDP list.
static void turbo3dList(struct N64* n, struct Hle* h, uint32_t pc) {
	bool const ux = h->family == FAMILY_T3DUX;
	for (int steps = 0; steps < 1 << 16; ++steps, pc += ux ? 24 : 16) {
		uint32_t const global = rd32(n, pc), state = rd32(n, pc + 4), vtx = rd32(n, pc + 8), tri = rd32(n, pc + 12);
		if (!state) {
			return;
		}
		if (global) {
			uint32_t const g = address(h, global);
			h->modeH = rd32(n, g + 8) & 0xFFFFFF, h->modeL = rd32(n, g + 12);
			emit(n, h, (uint64_t) (0xEF000000u | h->modeH) << 32 | h->modeL);
			for (int i = 0; i < 16; ++i) {
				h->segments[i] = rd32(n, g + 16 + 4 * (uint32_t) i) & 0xFFFFFF;
			}
			viewport(n, h, address(h, global + 80));
			turbo3dRdp(n, h, rd32(n, g + 96));
		}
		if (ux) {
			t3duxObject(n, h, state, vtx, tri, rd32(n, pc + 16));
		} else {
			turbo3dObject(n, h, state, vtx, tri);
		}
	}
}

// ---- ZSortBOSS (World Driver Championship's and Stunt Racer 64's; GLideN64's ZSortBOSS.cpp)

// SP_STATUS's signals the CPU and this microcode pass each other.
enum { SIG0 = 0x80, SIG1 = 0x100, SIG2 = 0x200, SIG3 = 0x400, SIG4 = 0x800 };

static uint32_t zRd32(const struct Hle* h, uint32_t a) {
	return zRd16(h, a) << 16 | zRd16(h, a + 2);
}

static void zWr32(struct Hle* h, uint32_t a, uint32_t v) {
	zWr16(h, a, v >> 16), zWr16(h, a + 2, v);
}

static float* bossMatrix(struct Hle* h, uint32_t which) {
	return which == 0x830 ? h->modelview[h->depth] : which == 0x870 ? h->projection : which == 0x8B0 ? h->mvp : NULL;
}

static int32_t bossInvw(int32_t w) {
	return w ? 0x7FFFFFFF / w : 0x7FFFFFFF;
}

// Its RDP-side commands: other modes (F3DEX2's shifts; G_RDPSETOTHERMODE through the update mask), the
// update mask, the triangles' tile and level, or an RDP command as it is. The next command's address.
static uint32_t bossRdpCommand(struct N64* n, struct Hle* h, uint32_t pc, uint32_t w0, uint32_t w1) {
	switch (w0 >> 24) {
	case 0xE2:
	case 0xE3:
		otherModeMask(n, h, (w0 >> 24) == 0xE3, (uint32_t) ((int32_t) 0x80000000 >> (w0 & 31)) >> ((w0 >> 8) & 31), w1);
		return pc + 8;
	case 0xEF:
		h->modeH = ((w0 & h->bossMask[0]) | (h->modeH & ~h->bossMask[0])) & 0xFFFFFF;
		h->modeL = (w1 & h->bossMask[1]) | (h->modeL & ~h->bossMask[1]);
		emit(n, h, (uint64_t) (0xEF000000u | h->modeH) << 32 | h->modeL);
		return pc + 8;
	case 0xDD:
		h->bossMask[0] = w0 | 0xFF000000u, h->bossMask[1] = w1;
		return pc + 8;
	case 0xDE:
		h->texScaleS = h->texScaleT = 1, h->texLevel = (w1 >> 3) & 7, h->texOn = true, h->texTile = w1 & 7;
		h->geometry |= GEOM_SHADE | GEOM_SHADING_SMOOTH;
		return pc + 8;
	case 0xDF:
	case 0xE1:
	case 0xF1:
		return pc + 8;
	}
	return rdpCommand(n, h, pc, w0, w1);
}

// An RDP list, to its 0xDF.
static void bossRdp(struct N64* n, struct Hle* h, uint32_t a) {
	for (int k = 0; a && k < 0x10000; ++k) {
		uint32_t const w0 = rd32(n, a), w1 = rd32(n, a + 4);
		if (w0 >> 24 == 0xDF) {
			break;
		}
		a = bossRdpCommand(n, h, a, w0, w1);
	}
}

// An object's triangle or quad on the screen: x and y (10.2), colour, S and T (10.5 with perspective,
// else 10.6) and 1/w (scaled by the list's factor; equal to the colour word or negative: none).
static void bossDraw(struct N64* n, struct Hle* h, uint32_t a, int count) {
	bool const persp = h->modeH & 0x80000;
	struct ScreenVertex v[4];
	for (int i = 0; i < count; ++i, a += 16) {
		int32_t const invw = (int32_t) rd32(n, a + 12), rgba = (int32_t) rd32(n, a + 4);
		float const w = invw == rgba || invw < 0 ? 1 : bossInvw(invw) / h->bossInvw;
		v[i] = (struct ScreenVertex) { rd16(n, a) / 4.0f, rd16(n, a + 2) / 4.0f, 0, w != 0 ? 1 / w : 1,
		                               (rd8(n, a + 4) & 0xFF) / 255.0f, (rd8(n, a + 5) & 0xFF) / 255.0f, (rd8(n, a + 6) & 0xFF) / 255.0f,
		                               (rd8(n, a + 7) & 0xFF) / 255.0f, rd16(n, a + 8) / (persp ? 32.0f : 64.0f), rd16(n, a + 10) / (persp ? 32.0f : 64.0f) };
	}
	bool const texOn = h->texOn;
	h->texOn = true;
	triangle(n, h, &v[0], &v[1], &v[2], NULL);
	if (count == 4) {
		triangle(n, h, &v[2], &v[1], &v[3], NULL);
	}
	h->texOn = texOn;
}

// G_ZOBJ: two lists of objects, each with three RDP lists (run when they changed) and a textured
// triangle or quad.
static void bossObjects(struct N64* n, struct Hle* h, uint32_t header) {
	for (int k = 0; header && k < 0x10000; ++k) {
		uint32_t const a = header & ~7u;
		int const type = (int) (header & 7) << 1;
		if (type == 0 || type == 4 || type == 8) {
			for (int i = 0; i < 3; ++i) {
				uint32_t const w1 = rd32(n, a + 4 + 4 * (uint32_t) i);
				if (w1 != h->zRdp[i]) {
					h->zRdp[i] = w1;
					bossRdp(n, h, w1 ? address(h, w1) : 0);
				}
			}
			if (type) {
				bossDraw(n, h, a + 16, type == 4 ? 3 : 4);
			}
		}
		header = address(h, rd32(n, a));
	}
}

// G_ZMULT_MPMTX: vertices (three s16 each) through the combined matrix into screen records: x and y
// (10.2), 1/w, x and y as integers, clip codes, fog (through the fog table) and w.
static void bossTransform(struct Hle* h, uint32_t w1) {
	int const count = 1 + (int) (w1 >> 24);
	uint32_t src = (w1 >> 12) & 0xFFF, dst = w1 & 0xFFF;
	const float* m = h->mvp;
	float const f = h->bossInvw;
	for (int i = 0; i < count; ++i, src += 6, dst += 16) {
		float const x = (int16_t) zRd16(h, src), y = (int16_t) zRd16(h, src + 2), z = (int16_t) zRd16(h, src + 4);
		float const X = x * m[0] + y * m[4] + z * m[8] + m[12], Y = x * m[1] + y * m[5] + z * m[9] + m[13];
		float const Z = x * m[2] + y * m[6] + z * m[10] + m[14], W = x * m[3] + y * m[7] + z * m[11] + m[15];
		float const iw = W <= 0 ? f : 1 / W;
		float const xw = X * iw < -f ? -f : X * iw > f ? f : X * iw, yw = Y * iw < -f ? -f : Y * iw > f ? f : Y * iw;
		int fog = (int) (W * (h->fogMul / 65536.0f) + h->fogOffset);
		fog = fog > 127 ? 127 : fog < -128 ? -128 : fog;
		int const cc = (X >= W ? 0x10 : 0) | (Y >= W ? 0x20 : 0) | (Z >= W ? 0x40 : 0) | (X <= -W ? 0x01 : 0) | (Y <= -W ? 0x02 : 0) | (Z <= -W ? 0x04 : 0);
		zWr16(h, dst, (uint32_t) (int16_t) (int32_t) (h->zTrans[0] + xw * h->zScale[0]));
		zWr16(h, dst + 2, (uint32_t) (int16_t) (int32_t) (h->zTrans[1] + yw * h->zScale[1]));
		zWr32(h, dst + 4, (uint32_t) bossInvw((int32_t) (W * f)));
		zWr16(h, dst + 8, (uint32_t) (int16_t) (int32_t) X), zWr16(h, dst + 10, (uint32_t) (int16_t) (int32_t) Y);
		h->zMem[(dst + 12) & 0xFFF] = (uint8_t) cc, h->zMem[(dst + 13) & 0xFFF] = h->bossFog[fog + 128];
		zWr16(h, dst + 14, (uint32_t) (int16_t) (int32_t) W);
	}
}

// G_ZMOVEMEM: a matrix loaded (the combined one also saved), the viewport, or bytes between RDRAM and
// DMEM (the fog table kept as it goes in).
static void bossMoveMem(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	bool const save = (w0 >> 23) & 1;
	uint32_t const length = 1 + ((w0 >> 12) & 0x7FF), offset = w0 & 0xFFF, a = address(h, w1);
	float* m = bossMatrix(h, offset);
	if (m && !save) {
		readMatrix(n, a, m);
		return;
	}
	if (m) {
		if (!h->dry) {
			n64RdramWrite(n, a, 64);
			for (int i = 0; i < 16; ++i) {
				int32_t const v = (int32_t) lrintf(m[i] * 65536.0f);
				uint32_t const at = (a + 2 * (uint32_t) i) & (RDRAM_SIZE - 1);
				n->rdram[at] = (uint8_t) (v >> 24), n->rdram[(at + 1) & (RDRAM_SIZE - 1)] = (uint8_t) (v >> 16);
				n->rdram[(at + 32) & (RDRAM_SIZE - 1)] = (uint8_t) (v >> 8), n->rdram[(at + 33) & (RDRAM_SIZE - 1)] = (uint8_t) v;
			}
		}
		return;
	}
	if (offset == 0) {
		viewport(n, h, a);
		h->fogMul = rd16(n, a + 6), h->fogOffset = rd16(n, a + 14);
		for (int k = 0; k < 2; ++k) {
			h->zScale[k] = h->vscale[k] * 4, h->zTrans[k] = h->vtrans[k] * 4;
		}
		return;
	}
	if (offset + length > 0x1000) {
		return;
	}
	if (offset == 0x730) {
		for (uint32_t i = 0; i < 256 && i < length; ++i) {
			h->bossFog[i] = n->rdram[(a + i) & (RDRAM_SIZE - 1)];
		}
	}
	if (!save) {
		for (uint32_t i = 0; i < length; ++i) {
			h->zMem[offset + i] = n->rdram[(a + i) & (RDRAM_SIZE - 1)];
		}
	} else if (!h->dry) {
		n64RdramWrite(n, a, length);
		for (uint32_t i = 0; i < length; ++i) {
			n->rdram[(a + i) & (RDRAM_SIZE - 1)] = h->zMem[offset + i];
		}
	}
}

// G_ZLIGHTING: S and T for each normal (s8 in DMEM) against the lookat directions, into DMEM.
static void bossLighting(struct Hle* h, uint32_t w0, uint32_t w1) {
	int const count = 1 + (int) (w1 >> 24);
	uint32_t normals = w0 & 0xFFF, st = w1 & 0xFFF;
	for (int i = 0; i < count; ++i, normals += 3, st += 4) {
		float const e[3] = { (int8_t) h->zMem[normals & 0xFFF] / 256.0f, (int8_t) h->zMem[(normals + 1) & 0xFFF] / 256.0f,
		                     (int8_t) h->zMem[(normals + 2) & 0xFFF] / 256.0f };
		float const x = e[0] * h->lookat[0][0] + e[1] * h->lookat[0][1] + e[2] * h->lookat[0][2];
		float const y = e[0] * h->lookat[1][0] + e[1] * h->lookat[1][1] + e[2] * h->lookat[1][2];
		zWr16(h, st, (uint32_t) (int16_t) (int32_t) ((x + 0.5f) * 1024)), zWr16(h, st + 2, (uint32_t) (int16_t) (int32_t) ((y + 0.5f) * 1024));
	}
}

// G_ZTRANSFORMLIGHTS: the two lookat directions (s8 in DMEM) through the model matrix, normalized (the
// microcode's lights, which no game here has, are left out as GLideN64 leaves them).
static void bossLookat(struct Hle* h, uint32_t w1) {
	uint32_t a = w1 & 0xFFF;
	int const lights = 1 - (int) (w1 >> 12);
	a += 24 * (uint32_t) (lights > 0 && lights < 12 ? lights : 0);
	const float* m = h->modelview[h->depth];
	for (int l = 0; l < 2; ++l, a += 24) {
		float const d[3] = { (int8_t) h->zMem[(a + 16) & 0xFFF] / 256.0f, (int8_t) h->zMem[(a + 17) & 0xFFF] / 256.0f,
		                     (int8_t) h->zMem[(a + 18) & 0xFFF] / 256.0f };
		float r[3] = { m[0] * d[0] + m[4] * d[1] + m[8] * d[2], m[1] * d[0] + m[5] * d[1] + m[9] * d[2], m[2] * d[0] + m[6] * d[1] + m[10] * d[2] };
		float const len = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
		float recip = len != 0 ? 1 / sqrtf(len) : 256;
		recip = recip > 256 ? 256 : recip;
		for (int k = 0; k < 3; ++k) {
			h->lookat[l][k] = r[k] * recip;
		}
	}
}

// The audio commands, in DMEM as GLideN64 has them. G_ZAUDIO1: a word of DMEM to RDRAM (with the next).
static void bossAudio1(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	uint32_t const a = address(h, w1);
	zWr32(h, 0, zRd32(h, w0 & 0xFFC));
	if (!h->dry) {
		n64RdramWrite(n, a, 8);
		for (uint32_t i = 0; i < 8; ++i) {
			n->rdram[(a + i) & (RDRAM_SIZE - 1)] = h->zMem[i];
		}
	}
}

// G_ZAUDIO2: samples resampled (a step and a start, 8.16) between the two at 0x30 and mixed into the
// buffer the word at DMEM 0x10 points at, by the two volumes at 0x904.
static void bossAudio2(struct Hle* h, uint32_t w0, uint32_t w1) {
	int const length = (int) (w1 >> 24);
	uint32_t dst = zRd32(h, 0x10);
	float const step = (float) ((w0 >> 16) & 0xFF) + (float) (w0 & 0xFFFF) / 65536.0f, start = (float) ((w1 >> 16) & 0xFF) + (float) (w1 & 0xFFFF) / 65536.0f;
	uint16_t const volume[2] = { (uint16_t) zRd16(h, 0x904), (uint16_t) zRd16(h, 0x906) };
	for (int i = 0; i < length; i += 4) {
		for (int j = 0; j < 4; ++j) {
			float whole;
			float const frac = fabsf(modff(i * step + j * step + start, &whole));
			int const index = (int) whole << 1;
			int16_t const v9 = (int16_t) zRd16(h, 0x30 + (uint32_t) index), v10 = (int16_t) zRd16(h, 0x32 + (uint32_t) index);
			int16_t const v12 = (int16_t) (v10 - v9), v13 = (int16_t) zRd16(h, dst);
			for (int k = 0; k < 2; ++k, dst += 2) {
				int16_t const sample = (int16_t) (((int32_t) ((uint32_t) v9 << 16) + v12 * (int32_t) (uint16_t) (frac * 65536.0f)) >> 16);
				zWr16(h, dst, (uint32_t) (int16_t) (((int32_t) ((uint32_t) v13 << 16) + volume[k] * sample) >> 16));
			}
		}
	}
}

// G_ZAUDIO3: the ADPCM table (8 x 8) from RDRAM, and 8 bytes from another address into DMEM 0, that
// address after them.
static void bossAudio3(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	uint32_t a = address(h, w0);
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			h->bossTable[i][j] = rd16(n, a + (uint32_t) (i << 4) + (uint32_t) (j << 1));
		}
	}
	a = address(h, w1);
	for (uint32_t i = 0; i < 8; ++i) {
		h->zMem[i] = n->rdram[(a + i) & (RDRAM_SIZE - 1)];
	}
	zWr32(h, 8, a);
}

// G_ZAUDIO4: ADPCM frames (9 bytes: scale and predictor, 16 samples) decoded into DMEM 0x30, from the
// two samples at DMEM 0.
static void bossAudio4(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	uint32_t src = address(h, w1) + ((w0 & 0xF000) >> 12), dst = 0x30;
	int16_t v1 = (int16_t) zRd16(h, 0), v2 = (int16_t) zRd16(h, 2);
	for (int left = (int) (w0 & 0xFFF); left > 0; left -= 9) {
		uint32_t const head = n->rdram[src++ & (RDRAM_SIZE - 1)];
		int const index = (int) (head & 15) << 1;
		if (index > 6) {
			break;
		}
		int16_t const scale = (int16_t) (1 << ((head >> 4) & 31));
		for (int half = 0; half < 2; ++half, dst += 16) {
			int16_t coeff[8];
			for (int b = 0; b < 4; ++b) {
				int32_t const v = (int8_t) n->rdram[src++ & (RDRAM_SIZE - 1)];
				coeff[2 * b] = (int16_t) (v >> 4), coeff[2 * b + 1] = (int16_t) ((int32_t) ((uint32_t) v << 28) >> 28);
			}
			int16_t out[8];
			for (int i = 0; i < 8; ++i) {
				int32_t sum = 0;
				for (int j = 0, k = i; j < i; ++j, --k) {
					sum += h->bossTable[index + 1][k - 1] * coeff[j];
				}
				sum += coeff[i] * 0x0800;
				sum *= scale;
				int32_t const prediction = v1 * h->bossTable[index][i] + v2 * h->bossTable[index + 1][i];
				out[i] = (int16_t) ((sum * 0x20 + prediction * 0x20) >> 16);
				zWr16(h, dst + 2 * (uint32_t) i, (uint32_t) out[i]);
			}
			v1 = out[6], v2 = out[7];
		}
	}
}

// The task's two lists, from where they were: the main list ends waiting for the CPU's SIG0, then the
// sub list runs; G_ZWAITSIGNAL raises SIG3 (and an SP interrupt) and waits for the CPU to clear it. A wait leaves the task
// waiting (hleResume comes back here).
static void bossRun(struct N64* n, struct Hle* h) {
	h->waiting = false;
	h->zMem = h->dry ? h->zDmem : n->sp.mem;
	for (int steps = 0; steps < 1 << 22; ++steps) {
		uint32_t const pc = h->bossPc[h->bossList], w0 = rd32(n, pc), w1 = rd32(n, pc + 4);
		uint32_t next = pc + 8;
		switch (w0 >> 24) {
		case 0x02:
			if (h->bossEnded[1]) {
				h->bossDone = true;
				return;
			}
			h->bossEnded[0] = true;
			if (!(n->sp.status & SIG0)) {
				h->waiting = !h->dry;
				return;
			}
			if (!h->dry) {
				n->sp.status &= ~(uint32_t) SIG0;
			}
			h->bossPc[0] = next, h->bossList = 1;
			continue;
		case 0x1A:
			if (h->bossEnded[0]) {
				h->bossDone = true;
				return;
			}
			h->bossEnded[1] = true;
			h->bossPc[1] = next, h->bossList = 0;
			continue;
		case 0x12:
			// The CPU clears SIG3 in its SP interrupt, which the microcode raises with it.
			if (!h->bossSignal && !h->dry) {
				n->sp.status = (n->sp.status & ~(uint32_t) (SIG1 | SIG2)) | SIG3;
				n64Interrupt(n, MI_SP, true);
			}
			if (h->dry || (n->sp.status & SIG3)) {
				h->bossSignal = true;
				h->waiting = !h->dry;
				return;
			}
			h->bossSignal = false;
			break;
		case 0x04:
			bossMoveMem(n, h, w0, w1);
			break;
		case 0x06:
			if ((w0 & 0xFFF) == 0x10 && rd32(n, next) >> 24 == 0x04) {
				h->bossInvw = (float) w1;
			}
			zWr32(h, w0 & 0xFFC, w1);
			break;
		case 0x08: {
			float* m = bossMatrix(h, w1 & 0xFFF);
			if (m) {
				float t[16];
				memcpy(t, m, sizeof(t));
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						m[j * 4 + i] = t[i * 4 + j];
					}
				}
			}
			break;
		}
		case 0x0A: {
			// G_ZMTXCAT: D = T x S (S in the second word's top half, T in the first word).
			float* a = bossMatrix(h, (w1 >> 16) & 0xFFF), * b = bossMatrix(h, w0 & 0xFFF), * d = bossMatrix(h, w1 & 0xFFF);
			if (a && b && d) {
				float m[16];
				multiply(m, b, a);
				memcpy(d, m, sizeof(m));
			}
			break;
		}
		case 0x0C:
			bossTransform(h, w1);
			break;
		case 0x0E:
			bossRdp(n, h, w1 ? address(h, w1) : 0);
			break;
		case 0x10:
			bossObjects(n, h, address(h, w0));
			bossObjects(n, h, address(h, w1));
			break;
		case 0x14:
			bossLighting(h, w0, w1);
			break;
		case 0x18:
			bossLookat(h, w1);
			break;
		case 0x1C:
			bossAudio2(h, w0, w1);
			break;
		case 0x1E:
			memset(&h->zMem[0xC20], 0, 512);
			break;
		case 0x22:
			bossAudio3(n, h, w0, w1);
			break;
		case 0x24:
			bossAudio4(n, h, w0, w1);
			break;
		case 0x26:
			bossAudio1(n, h, w0, w1);
			break;
		default:
			if (w0 >> 24 >= 0xDD) {
				next = bossRdpCommand(n, h, pc, w0, w1);
			}
			break;
		}
		h->bossPc[h->bossList] = next;
	}
	h->bossDone = true;
}

// ---- Factor 5's (Rogue Squadron's; GLideN64's F5Rogue.cpp, after Lemmy's LemNemu and olivieryuyu's
// decoding). GLideN64 keeps the generated terrain's arrays index-swapped (^1); so does this, the same
// raw indices reaching them.

struct F5Triangle {
	int v[3], z;
};

// A vertex (x, y, z) through the combined matrix, y the other way up (as Factor 5's matrices have it).
static void f5Vertex(struct Hle* h, int i, float x, float y, float z) {
	const float* m = h->mvp;
	struct Vertex* v = &h->v[i];
	v->x = x * m[0] + y * m[4] + z * m[8] + m[12];
	v->y = -(x * m[1] + y * m[5] + z * m[9] + m[13]);
	v->z = x * m[2] + y * m[6] + z * m[10] + m[14];
	v->w = x * m[3] + y * m[7] + z * m[11] + m[15];
}

// The colours (RGBA words) and, textured, the texture coordinates (S and T words through the two
// scales, as the RSP's multiplies keep them: 10.5 with perspective, else 11.4) of the vertices; fog in
// the alpha when it is on.
static void f5Prepare(struct Hle* h, const int* index, const uint32_t* colors, const uint32_t* texCoords, bool persp, int count) {
	uint32_t const sscale = h->f5TexScale[0] >> 16, tscale = h->f5TexScale[0] & 0xFFFF, sscale1 = h->f5TexScale[1] >> 16, tscale1 = h->f5TexScale[1] & 0xFFFF;
	for (int i = 0; i < count; ++i) {
		struct Vertex* v = &h->v[index[i]];
		uint32_t const c = colors[i];
		v->r = (c >> 24) / 255.0f, v->g = ((c >> 16) & 0xFF) / 255.0f, v->b = ((c >> 8) & 0xFF) / 255.0f, v->a = (c & 0xFF) / 255.0f;
		if (h->geometry & GEOM_FOG) {
			float const f = (v->w != 0 ? v->z / v->w : 0) * h->fogMul + h->fogOffset;
			v->a = (f < 0 ? 0 : f > 255 ? 255 : f) / 255.0f;
		}
		if (texCoords) {
			uint32_t const st = texCoords[i], s0 = (uint32_t) (int32_t) (int16_t) (st >> 16), t0 = (uint32_t) (int32_t) (int16_t) st;
			int16_t const s = (int16_t) ((int16_t) ((s0 * sscale) >> 16) + (int16_t) (s0 * sscale1));
			int16_t const t = (int16_t) ((int16_t) ((t0 * tscale) >> 16) + (int16_t) (t0 * tscale1));
			v->s = s / (persp ? 32.0f : 16.0f), v->t = t / (persp ? 32.0f : 16.0f);
		}
	}
}

static int f5Z(const struct Hle* h, int i) {
	const struct Vertex* v = &h->v[i];
	return (int) (((v->w != 0 ? v->z / v->w : 0) * h->vscale[2] + h->vtrans[2]) * 32768.0f);
}

// The generated triangles back to front (by their corners' average depth), then drawn.
static void f5DrawSorted(struct N64* n, struct Hle* h, struct F5Triangle* t, int count) {
	for (int i = 0; i < count; ++i) {
		t[i].z = (f5Z(h, t[i].v[0]) + f5Z(h, t[i].v[1]) + f5Z(h, t[i].v[2])) / 3;
	}
	for (int i = 1; i < count; ++i) {
		struct F5Triangle const x = t[i];
		int j = i;
		while (j > 0 && t[j - 1].z < x.z) {
			t[j] = t[j - 1];
			--j;
		}
		t[j] = x;
	}
	for (int i = 0; i < count; ++i) {
		drawTriangle(n, h, t[i].v[0], t[i].v[1], t[i].v[2]);
	}
}

static void f5HeightAverage(uint16_t* d, int a, int b, int to) {
	d[to] = (uint16_t) (int16_t) (((int16_t) d[a] + (int16_t) d[b]) / 2);
}

static void f5ColorAverage(uint32_t* c, int a, int b, int to) {
	uint32_t r = 0;
	for (int k = 0; k < 32; k += 8) {
		r |= ((((c[a] >> k) & 0xFF) + ((c[b] >> k) & 0xFF)) >> 1) << k;
	}
	c[to] = r;
}

static void f5HeightWeight(uint16_t* d, int a, int b, int to, uint16_t w) {
	int32_t const v = (int16_t) d[to] * (0x10000 - w) + ((((int16_t) d[a] + (int16_t) d[b]) * w) >> 1);
	d[to] = (uint16_t) (v >> 16);
}

static void f5ColorWeight(uint32_t* c, int a, int b, int to, uint16_t w) {
	uint32_t r = 0;
	for (int k = 0; k < 32; k += 8) {
		uint32_t const v = ((c[to] >> k) & 0xFF) * (0x10000u - w) + (((((c[a] >> k) & 0xFF) + ((c[b] >> k) & 0xFF)) * w) >> 1);
		r |= ((v >> 16) & 0xFF) << k;
	}
	c[to] = r;
}

// A terrain patch's corner fix: when the option byte matches, two edge heights and colours averaged
// and two triangles more.
static void f5Option(uint32_t param, uint32_t mask, uint32_t want, uint16_t* d, uint32_t* c, const int (*h)[3], const int (*col)[3], int pairs,
                     const int (*tris)[3], int trisCount, struct F5Triangle* t, int* count) {
	if ((param & mask) != want) {
		return;
	}
	for (int i = 0; i < pairs; ++i) {
		f5HeightAverage(d, h[i][0], h[i][1], h[i][2]);
		f5ColorAverage(c, col[i][0], col[i][1], col[i][2]);
	}
	for (int i = 0; i < trisCount; ++i) {
		t[(*count)++] = (struct F5Triangle) { { tris[i][0], tris[i][1], tris[i][2] }, 0 };
	}
}

static void f5Weight(uint16_t w, uint16_t* d, uint32_t* c, const int (*h)[3], const int (*col)[3], int pairs) {
	if (!w) {
		return;
	}
	for (int i = 0; i < pairs; ++i) {
		f5HeightWeight(d, h[i][0], h[i][1], h[i][2], w);
		f5ColorWeight(c, col[i][0], col[i][1], col[i][2], w);
	}
}

// The grid's vertices (step x step): x from params[8] by params[9]'s step a column, z from params[9]
// by the same a row, y each height plus params[8]'s low half times 16 (16-bit lanes as the RSP adds
// them).
static int f5Grid(struct Hle* h, const uint32_t* params, int step, const uint16_t* heights) {
	uint16_t const s = (uint16_t) params[9], y0 = (uint16_t) ((params[8] & 0xFFFF) << 4), x0 = (uint16_t) (params[8] >> 16), z0 = (uint16_t) (params[9] >> 16);
	int count = 0;
	uint16_t x = 0;
	for (int column = 0; column < step; ++column, x = (uint16_t) (x + s)) {
		uint16_t z = 0;
		for (int row = 0; row < step * (step - 1) + 1; row += step, z = (uint16_t) (z + s)) {
			f5Vertex(h, count++, (int16_t) (uint16_t) (x + x0), (int16_t) (uint16_t) (heights[(column + row) ^ 1] + y0), (int16_t) (uint16_t) (z + z0));
		}
	}
	return count;
}

// The colours and texture coordinates of a step x step grid, then its triangles sorted and drawn.
static void f5GridDraw(struct N64* n, struct Hle* h, const uint32_t* params, int step, const uint32_t* colorData, struct F5Triangle* t, int count,
                       const struct F5Triangle* order, int orderCount) {
	int const vertices = step * step;
	int index[25];
	uint32_t colors[25], texCoords[25];
	uint16_t const tex = (uint16_t) params[7];
	int k = 0;
	for (int i = 0, start = 0; i < vertices; i += step, ++start) {
		for (int j = 0; j < step; ++j, ++k) {
			index[k] = k, colors[k] = colorData[start + j * step];
		}
	}
	k = 0;
	for (int i = 0; i < step; ++i) {
		uint32_t st = (uint32_t) (uint16_t) (tex * i) << 16 | (uint16_t) (tex * (step - 1));
		for (int j = 0; j < step; ++j, st -= tex) {
			texCoords[k++] = st;
		}
	}
	f5Prepare(h, index, colors, texCoords, false, vertices);
	for (int i = 0; i < orderCount; ++i) {
		t[count++] = order[i];
	}
	f5DrawSorted(n, h, t, count);
}

// G_TRIGEN 00 00: a 5 x 5 terrain patch from 25 heights (bytes) and 25 colours.
static void f5Patch5(struct N64* n, struct Hle* h, const uint32_t* params) {
	static const struct F5Triangle order[32] = {
		{ { 0, 6, 1 }, 0 }, { { 0, 5, 6 }, 0 }, { { 1, 7, 2 }, 0 }, { { 1, 6, 7 }, 0 }, { { 2, 8, 3 }, 0 }, { { 5, 11, 6 }, 0 }, { { 2, 7, 8 }, 0 },
		{ { 5, 10, 11 }, 0 }, { { 3, 9, 4 }, 0 }, { { 3, 8, 9 }, 0 }, { { 6, 12, 7 }, 0 }, { { 6, 11, 12 }, 0 }, { { 10, 16, 11 }, 0 }, { { 7, 13, 8 }, 0 },
		{ { 10, 15, 16 }, 0 }, { { 7, 12, 13 }, 0 }, { { 8, 14, 9 }, 0 }, { { 11, 17, 12 }, 0 }, { { 8, 13, 14 }, 0 }, { { 11, 16, 17 }, 0 }, { { 15, 21, 16 }, 0 },
		{ { 12, 18, 13 }, 0 }, { { 15, 20, 21 }, 0 }, { { 12, 17, 18 }, 0 }, { { 13, 19, 14 }, 0 }, { { 16, 22, 17 }, 0 }, { { 16, 21, 22 }, 0 }, { { 13, 18, 19 }, 0 },
		{ { 17, 23, 18 }, 0 }, { { 17, 22, 23 }, 0 }, { { 18, 24, 19 }, 0 }, { { 18, 23, 24 }, 0 } };
	uint32_t colorData[25];
	uint16_t d[26] = { 0 };
	uint32_t const colorAt = params[3] & 0xFFFFFF, heightAt = params[2] & 0xFFFFF8;
	for (int i = 0; i < 25; ++i) {
		colorData[i] = rd32(n, colorAt + 4 * (uint32_t) i);
		int8_t const v = rd8(n, heightAt + (params[2] & 7) + (uint32_t) i);
		d[i ^ 1] = (uint16_t) ((v & 0xFF) << 4 | (v < 0 ? 0xF000 : 0));
	}
	struct F5Triangle t[64];
	int count = 0;
	static const int o1h[2][3] = { { 1, 3, 0 }, { 3, 5, 2 } }, o1c[2][3] = { { 0, 2, 1 }, { 2, 4, 3 } }, o1t[2][3] = { { 0, 10, 5 }, { 10, 20, 15 } };
	static const int o2h[2][3] = { { 21, 23, 20 }, { 23, 25, 22 } }, o2c[2][3] = { { 20, 22, 21 }, { 22, 24, 23 } }, o2t[2][3] = { { 4, 9, 14 }, { 14, 19, 24 } };
	static const int o3h[2][3] = { { 1, 11, 4 }, { 11, 21, 14 } }, o3c[2][3] = { { 0, 10, 5 }, { 10, 20, 15 } }, o3t[2][3] = { { 0, 1, 2 }, { 2, 3, 4 } };
	static const int o4h[2][3] = { { 5, 15, 8 }, { 15, 25, 18 } }, o4c[2][3] = { { 4, 14, 9 }, { 14, 24, 19 } }, o4t[2][3] = { { 22, 24, 23 }, { 20, 22, 21 } };
	static const int w5h[8][3] = { { 1, 13, 7 }, { 3, 13, 6 }, { 3, 15, 9 }, { 11, 13, 10 }, { 13, 15, 12 }, { 11, 23, 17 }, { 13, 23, 16 }, { 13, 25, 19 } };
	static const int w5c[8][3] = { { 0, 12, 6 }, { 2, 12, 7 }, { 2, 14, 8 }, { 10, 12, 11 }, { 12, 14, 13 }, { 10, 22, 16 }, { 12, 22, 17 }, { 12, 24, 18 } };
	f5Option(params[1], 0xFF00, 0x0100, d, colorData, o1h, o1c, 2, o1t, 2, t, &count);
	f5Weight((uint16_t) params[6], d, colorData, o1h, o1c, 2);
	f5Option(params[1], 0xFF, 0x01, d, colorData, o2h, o2c, 2, o2t, 2, t, &count);
	f5Weight((uint16_t) (params[7] >> 16), d, colorData, o2h, o2c, 2);
	f5Option(params[1], 0xFF000000u, 0x01000000u, d, colorData, o3h, o3c, 2, o3t, 2, t, &count);
	f5Weight((uint16_t) params[5], d, colorData, o3h, o3c, 2);
	f5Option(params[1], 0xFF0000, 0x010000, d, colorData, o4h, o4c, 2, o4t, 2, t, &count);
	f5Weight((uint16_t) (params[6] >> 16), d, colorData, o4h, o4c, 2);
	f5Weight((uint16_t) (params[5] >> 16), d, colorData, w5h, w5c, 8);
	f5Grid(h, params, 5, d);
	f5GridDraw(n, h, params, 5, colorData, t, count, order, 32);
}

// G_TRIGEN 00 01: a 3 x 3 patch, its heights and colours spread out in memory.
static void f5Patch3(struct N64* n, struct Hle* h, const uint32_t* params) {
	static const struct F5Triangle order[8] = { { { 0, 3, 4 }, 0 }, { { 0, 4, 1 }, 0 }, { { 1, 4, 5 }, 0 }, { { 1, 5, 2 }, 0 },
	                                            { { 3, 6, 7 }, 0 }, { { 3, 7, 4 }, 0 }, { { 4, 7, 8 }, 0 }, { { 4, 8, 5 }, 0 } };
	uint32_t colorData[32] = { 0 };
	uint16_t d[64] = { 0 };
	uint32_t const end = ((params[4] & 0xFFFF) + 0x13) & 0xFF0, offset = params[3] & 7;
	if (end < offset) {
		return;
	}
	uint32_t const size = (end - offset) / 4, colorAt = (params[3] & 0xFFFFF8) + offset;
	int colorCount = 0;
	for (uint32_t i = 0; colorCount + 3 <= 32;) {
		colorData[colorCount++] = rd32(n, colorAt + 4 * i), i += 2;
		colorData[colorCount++] = rd32(n, colorAt + 4 * i), i += 2;
		colorData[colorCount++] = rd32(n, colorAt + 4 * i), i += 6;
		if (size <= i) {
			break;
		}
	}
	uint32_t const heightSize = ((params[4] >> 16) + 0x16) & 0xFF0, heightAt = params[2] & 0xFFFFF8;
	uint32_t at = params[2] & 7;
	for (int i = 0; i < 64;) {
		for (int k = 0; k < 3 && i < 64; ++k, ++i, at += 2) {
			int8_t const v = rd8(n, heightAt + at);
			d[i ^ 1] = (uint16_t) ((v & 0xFF) << 4 | (v < 0 ? 0xF000 : 0));
		}
		at += 4;
		if (heightSize <= at) {
			break;
		}
	}
	struct F5Triangle t[32];
	int count = 0;
	static const int o1h[1][3] = { { 1, 3, 0 } }, o1c[1][3] = { { 0, 2, 1 } }, o1t[1][3] = { { 0, 6, 3 } };
	static const int o2h[1][3] = { { 7, 9, 8 } }, o2c[1][3] = { { 6, 8, 7 } }, o2t[1][3] = { { 2, 5, 8 } };
	static const int w2h[1][3] = { { 7, 9, 6 } };
	static const int o3h[1][3] = { { 1, 7, 2 } }, o3c[1][3] = { { 0, 6, 3 } }, o3t[1][3] = { { 0, 1, 2 } };
	static const int o4h[1][3] = { { 3, 9, 4 } }, o4c[1][3] = { { 2, 8, 5 } }, o4t[1][3] = { { 6, 8, 7 } };
	static const int w5h[1][3] = { { 1, 9, 5 } }, w5c[1][3] = { { 0, 8, 4 } };
	f5Option(params[1], 0xFF00, 0x0200, d, colorData, o1h, o1c, 1, o1t, 1, t, &count);
	f5Weight((uint16_t) params[6], d, colorData, o1h, o1c, 1);
	f5Option(params[1], 0xFF, 0x02, d, colorData, o2h, o2c, 1, o2t, 1, t, &count);
	f5Weight((uint16_t) (params[7] >> 16), d, colorData, w2h, o2c, 1);
	f5Option(params[1], 0xFF000000u, 0x02000000u, d, colorData, o3h, o3c, 1, o3t, 1, t, &count);
	f5Weight((uint16_t) params[5], d, colorData, o3h, o3c, 1);
	f5Option(params[1], 0x00FF0000, 0x00020000, d, colorData, o4h, o4c, 1, o4t, 1, t, &count);
	f5Weight((uint16_t) (params[6] >> 16), d, colorData, o4h, o4c, 1);
	f5Weight((uint16_t) (params[5] >> 16), d, colorData, w5h, w5c, 1);
	f5Grid(h, params, 3, d);
	f5GridDraw(n, h, params, 3, colorData, t, count, order, 8);
}

// G_TRIGEN 02: a quad, its four corners from params[1] and [2] on the grid params[8] and [9] give.
static void f5Quad(struct N64* n, struct Hle* h, const uint32_t* params) {
	uint16_t const s = (uint16_t) params[9], y0 = (uint16_t) ((params[8] & 0xFFFF) << 4), x0 = (uint16_t) (params[8] >> 16), z0 = (uint16_t) (params[9] >> 16);
	uint16_t const heights[4] = { (uint16_t) (params[1] >> 16), (uint16_t) params[1], (uint16_t) (params[2] >> 16), (uint16_t) params[2] };
	uint16_t const dx[4] = { 0, s, 0, s }, dz[4] = { 0, 0, s, s };
	for (int i = 0; i < 4; ++i) {
		f5Vertex(h, i, (int16_t) (uint16_t) (x0 + dx[i]), (int16_t) (uint16_t) (y0 + heights[i]), (int16_t) (uint16_t) (z0 + dz[i]));
	}
	uint32_t const tex = (uint16_t) params[7];
	int const index[4] = { 0, 1, 2, 3 };
	uint32_t const colors[4] = { params[3], params[4], params[5], params[6] }, texCoords[4] = { tex, tex | tex << 16, 0, tex << 16 };
	f5Prepare(h, index, colors, texCoords, false, 4);
	const struct Vertex* b = &h->v[1], * c = &h->v[2];
	if ((c->w != 0 ? c->z / c->w : 0) > (b->w != 0 ? b->z / b->w : 0)) {
		drawTriangle(n, h, 0, 1, 3), drawTriangle(n, h, 0, 3, 2);
	} else {
		drawTriangle(n, h, 0, 3, 2), drawTriangle(n, h, 0, 1, 3);
	}
}

// G_TRI1 and G_TRI2: vertices (DMEM offsets / 40), their colours from the table by the next word's
// indices, texture coordinates after it when textured; flat shading takes the colour of the corner the
// word's top bits name.
static uint32_t f5Triangles(struct N64* n, struct Hle* h, uint32_t pc, uint32_t w0, uint32_t w1, bool two) {
	int const v[4] = { (int) (((w1 >> 13) & 0x7F8) / 40), (int) (((w1 >> 5) & 0x7F8) / 40), (int) (((w1 << 3) & 0x7F8) / 40), (int) (((w1 >> 21) & 0x7F8) / 40) };
	uint32_t const param = rd32(n, pc + 8);
	uint32_t const index[4] = { (param >> 16) & 0xFF, (param >> 8) & 0xFF, param & 0xFF, param >> 24 };
	bool const textured = w0 & 2;
	uint32_t colors[4], texCoords[4];
	for (int i = 0; i < 4; ++i) {
		colors[i] = rd32(n, h->f5Colors + index[i]);
		texCoords[i] = textured ? rd32(n, pc + 16 + 4 * (uint32_t) i) : 0;
	}
	for (int i = 0; i < 4; ++i) {
		if (v[i] >= VERTICES) {
			return pc + (textured ? 32 : 16);
		}
	}
	f5Prepare(h, v, colors, textured ? texCoords : NULL, h->modeH & 0x80000, two ? 4 : 3);
	uint32_t const geometry = h->geometry;
	if ((h->geometry & (GEOM_CULL_FRONT | GEOM_CULL_BACK)) == (GEOM_CULL_FRONT | GEOM_CULL_BACK)) {
		h->geometry &= ~(uint32_t) GEOM_CULL_FRONT;
	}
	int const first = (int) ((w1 >> 24) & 3) % 3;
	int const tris[2][3] = { { v[0], v[1], v[2] }, { v[0], v[2], v[3] } };
	for (int k = 0; k < (two ? 2 : 1); ++k) {
		drawTriangle(n, h, tris[k][first], tris[k][(first + 1) % 3], tris[k][(first + 2) % 3]);
	}
	h->geometry = geometry;
	return pc + (textured ? 32 : 16);
}

// G_TEXRECT_GEN: a sprite at a vertex, its size through the viewport and the projection's scale over w,
// its texture steps from the command over the size (the combined matrix's fractions below them).
static void f5Sprite(struct N64* n, struct Hle* h, uint32_t pc, uint32_t w0) {
	int const index = (int) (((w0 >> 5) & 0x7F8) / 40);
	if (index >= VERTICES) {
		return;
	}
	const struct Vertex* v = &h->v[index];
	if (v->x > v->w || v->x < -v->w || v->y > v->w || v->y < -v->w || v->w < 0.01f) {
		return;
	}
	uint32_t const p1 = rd32(n, pc + 4), p3 = rd32(n, pc + 12), p4 = rd32(n, pc + 16);
	float const screenX = v->x / v->w * h->vscale[0] + h->vtrans[0], screenY = -v->y / v->w * h->vscale[1] + h->vtrans[1];
	bool const flip = w0 & 1;
	uint64_t const w = v->w > 1 ? (uint64_t) v->w : 1;
	uint32_t const vp = rd32(n, h->f5Viewport), viewX = (vp >> 17) & 0x7FFF, viewY = (vp >> 1) & 0x7FFF;
	uint32_t const pm = h->f5Persp;
	uint32_t const perspX = (rd32(n, pm) & 0xFFFF0000u) | rd32(n, pm + 32) >> 16, perspY = rd32(n, pm + 8) << 16 | (rd32(n, pm + 40) & 0xFFFF);
	uint64_t p3x = p3 >> 16, p3y = p3 & 0xFFFF;
	if (flip) {
		uint64_t const t = p3x;
		p3x = p3y, p3y = t;
	}
	uint32_t ox = (uint32_t) (((p3x * viewX * perspX) / w) >> 16), oy = (uint32_t) (((p3y * viewY * perspY) / w) >> 16);
	float const ulx = screenX - ox / 4.0f, lrx = screenX + ox / 4.0f, uly = screenY - oy / 4.0f, lry = screenY + oy / 4.0f;
	if (lrx - ulx <= 0 || lry - uly <= 0 || !ox || !oy) {
		return;
	}
	float whole;
	uint32_t const p4x = (p4 & 0xFFFF0000u) | (uint32_t) (fabsf(modff(h->mvp[0], &whole)) * 65536.0f);
	uint32_t const p4y = p4 << 16 | (uint32_t) (fabsf(modff(h->mvp[1], &whole)) * 65536.0f);
	if (flip) {
		uint32_t const t = ox;
		ox = oy, oy = t;
	}
	uint16_t dsdx = (uint16_t) ((p4x / ox) >> 10), dtdy = (uint16_t) ((p4y / oy) >> 10), e = 0, f = 0;
	if (w0 & 2) {
		dsdx = (uint16_t) -dsdx, e = (uint16_t) (p4 >> 16);
	}
	if (w0 & 4) {
		dtdy = (uint16_t) -dtdy, f = (uint16_t) p4;
	}
	uint16_t const rectDsdx = dsdx, rectDtdy = dtdy;
	if (flip) {
		uint16_t const t = dsdx;
		dsdx = dtdy, dtdy = t;
	}
	uint16_t const s = ulx > 0 ? (uint16_t) (e + 0xFFF0) : (uint16_t) (((0 - (dsdx << 6) * (int) (ulx * 4)) << 3) + e + 0xFFF0);
	uint16_t const t = uly > 0 ? (uint16_t) (f + 0xFFF0) : (uint16_t) (((0 - (dtdy << 6) * (int) (uly * 4)) << 3) + f + 0xFFF0);
	float const z = v->z / v->w * h->vscale[2] + h->vtrans[2];
	emit(n, h, (uint64_t) 0x2E << 56 | (uint64_t) (uint16_t) (int) (z * 32) << 16);
	emit(n, h, (uint64_t) 0x3A << 56 | p1);
	if (h->geometry & GEOM_FOG) {
		emit(n, h, (uint64_t) 0x38 << 56 | (p1 & 0xFFFFFF00u) | (uint32_t) (v->a * 255.0f));
	}
	int const xh = ulx > 0 ? (int) (ulx * 4) : 0, yh = uly > 0 ? (int) (uly * 4) : 0, xl = (int) (lrx * 4), yl = (int) (lry * 4);
	emit(n, h, (uint64_t) (flip ? 0x25 : 0x24) << 56 | (uint64_t) (xl & 0xFFF) << 44 | (uint64_t) (yl & 0xFFF) << 32 | (uint64_t) (h->texTile & 7) << 24 |
	           (uint64_t) (xh & 0xFFF) << 12 | (uint64_t) (yh & 0xFFF));
	emit(n, h, (uint64_t) s << 48 | (uint64_t) t << 32 | (uint64_t) rectDsdx << 16 | rectDtdy);
}

// The list (and those it calls): each starts with a word the jump command goes to, which a call or
// branch skips.
static void f5List(struct N64* n, struct Hle* h, uint32_t pc) {
	uint32_t stack[DL_DEPTH];
	int sp = 0;
	h->f5Jump[0] = rd32(n, pc) & 0xFFFFFF;
	pc += 8;
	for (int steps = 0; steps < 1 << 22; ++steps) {
		uint32_t const w0 = rd32(n, pc), w1 = rd32(n, pc + 4);
		int const op = w0 >> 24;
		uint32_t next = pc + 8;
		switch (op) {
		case 0x00:
			break;
		case 0x01: {
			int const p = (w0 >> 16) & 0xFF;
			if (p & 1) {
				h->f5Persp = w1 & 0xFFFFFF;
			}
			matrix(n, h, w1, p & 1, p & 2, p & 4);
			break;
		}
		case 0x02:
			h->f5Colors = address(h, w1);
			break;
		case 0x03:
			if (((w0 >> 16) & 0xFF) == 0x80) {
				h->f5Viewport = pc + 8;
				viewport(n, h, pc + 8);
			} else if (((w0 >> 16) & 0xFF) == 0x82) {
				h->f5TexScale[0] = rd32(n, pc + 16), h->f5TexScale[1] = rd32(n, pc + 12);
			}
			next = pc + 24;
			break;
		case 0x04: {
			uint32_t const a = address(h, w1);
			int const count = (int) ((w0 >> 10) & 0x3F);
			for (int i = 0; i < count && i < VERTICES; ++i) {
				f5Vertex(h, i, rd16(n, a + 8 * (uint32_t) i), rd16(n, a + 8 * (uint32_t) i + 2), rd16(n, a + 8 * (uint32_t) i + 4));
			}
			break;
		}
		case 0x05: {
			uint32_t params[10];
			for (int i = 0; i < 10; ++i) {
				params[i] = rd32(n, pc + 4 * (uint32_t) i);
			}
			int const mode = (w0 >> 8) & 0xFF;
			if (mode == 0 && (w0 & 0xFF) == 0) {
				f5Patch5(n, h, params);
			} else if (mode == 0 && (w0 & 0xFF) == 1) {
				f5Patch3(n, h, params);
			} else if (mode == 2) {
				f5Quad(n, h, params);
			}
			next = pc + 40;
			break;
		}
		case 0x06:
		case 0x07:
			if (op == 0x06) {
				if (sp >= DL_DEPTH) {
					break;
				}
				stack[sp++] = next;
			}
			next = address(h, w1);
			h->f5Jump[sp] = rd32(n, next) & 0xFFFFFF;
			next += 8;
			break;
		case 0xB5:
			next = h->f5Jump[sp];
			h->f5Jump[sp] = rd32(n, next) & 0xFFFFFF;
			next += 8;
			break;
		case 0xB8:
			if (!sp) {
				return;
			}
			next = stack[--sp];
			break;
		case 0xB3:
		case 0xBE:
			otherModeMask(n, h, op == 0xBE, ~rd32(n, pc + 8), w1);
			next = pc + 16;
			break;
		case 0xB4:
		case 0xBF:
			next = f5Triangles(n, h, pc, w0, w1, op == 0xB4);
			break;
		case 0xB6:
			setGeometry(h, w1, 0, false);
			break;
		case 0xB7:
			setGeometry(h, 0, w1, false);
			break;
		case 0xB9:
		case 0xBA:
			otherMode(n, h, op == 0xBA, (w0 >> 8) & 0xFF, w0 & 0xFF, w1);
			break;
		case 0xBB:
			h->texScaleS = (w1 >> 16) / 65536.0f, h->texScaleT = (w1 & 0xFFFF) / 65536.0f;
			h->texLevel = (w0 >> 11) & 7, h->texTile = (w0 >> 8) & 7, h->texOn = w0 & 0xFF;
			break;
		case 0xBC:
			if ((w0 & 0xFF) == 0x06) {
				h->segments[(((w0 >> 8) & 0xFFFF) >> 2) & 15] = w1 & 0xFFFFFF;
			} else if ((w0 & 0xFF) == 0x08) {
				h->fogMul = (int32_t) w1 / 256.0f;
			} else if ((w0 & 0xFF) == 0x0A) {
				h->fogOffset = (int32_t) w1 / 256.0f;
			}
			break;
		case 0xBD:
			f5Sprite(n, h, pc, w0);
			next = pc + 24;
			break;
		case 0xE4:
		case 0xE5:
			// A texture rectangle's other words follow it as they are.
			emit(n, h, (uint64_t) w0 << 32 | w1);
			emit(n, h, (uint64_t) rd32(n, pc + 8) << 32 | rd32(n, pc + 12));
			next = pc + 16;
			break;
		default:
			if (op >= 0xC0) {
				next = rdpCommand(n, h, pc, w0, w1);
			}
			break;
		}
		pc = next;
	}
}

// ---- Factor 5's second microcode (Indiana Jones and Battle for Naboo; GLideN64's F5Indi_Naboo.cpp,
// olivieryuyu's decoding). Its DMEM is kept as GLideN64 keeps it (each word's bytes the other way
// round: byte a there is the RSP's byte a ^ 3), so the decoding's offsets and swaps carry over as
// they are; RDRAM is read the same way.

static uint32_t f5iWord(const struct Hle* h, uint32_t a) {
	const uint8_t* m = h->f5iDmem;
	return m[a & 0xFFF] | m[(a + 1) & 0xFFF] << 8 | m[(a + 2) & 0xFFF] << 16 | (uint32_t) m[(a + 3) & 0xFFF] << 24;
}

static void f5iSetWord(struct Hle* h, uint32_t a, uint32_t v) {
	for (int k = 0; k < 4; ++k) {
		h->f5iDmem[(a + (uint32_t) k) & 0xFFF] = (uint8_t) (v >> (8 * k));
	}
}

static uint16_t f5iHalf(const struct Hle* h, uint32_t a) {
	return (uint16_t) (h->f5iDmem[a & 0xFFF] | h->f5iDmem[(a + 1) & 0xFFF] << 8);
}

static void f5iSetHalf(struct Hle* h, uint32_t a, uint16_t v) {
	h->f5iDmem[a & 0xFFF] = (uint8_t) v, h->f5iDmem[(a + 1) & 0xFFF] = (uint8_t) (v >> 8);
}

static uint64_t f5iQuad(const struct Hle* h, uint32_t a) {
	return f5iWord(h, a) | (uint64_t) f5iWord(h, a + 4) << 32;
}

// RDRAM as GLideN64 reads it: a word as it is, a halfword or byte from the other end of its word.
static uint16_t f5iRdHalf(struct N64* n, uint32_t a) {
	return (uint16_t) rd16(n, a ^ 2);
}

static uint8_t f5iRdByte(struct N64* n, uint32_t a) {
	return n->rdram[(a ^ 3) & (RDRAM_SIZE - 1)];
}

static void f5iCopyIn(struct N64* n, struct Hle* h, uint32_t dmem, uint32_t rdram, uint32_t length) {
	for (uint32_t i = 0; i < length; ++i) {
		h->f5iDmem[(dmem + i) & 0xFFF] = f5iRdByte(n, rdram + i);
	}
}

static void f5iCopyOut(struct N64* n, struct Hle* h, uint32_t rdram, uint32_t dmem, uint32_t length) {
	if (h->dry || !length) {
		return;
	}
	n64RdramWrite(n, rdram & ~7u, length + 16);
	for (uint32_t i = 0; i < length; ++i) {
		n->rdram[((rdram + i) ^ 3) & (RDRAM_SIZE - 1)] = h->f5iDmem[(dmem + i) & 0xFFF];
	}
}

// A vertex as GLideN64's SWVertex has it in such memory (y, x, flag, z halfwords): through the
// combined matrix into slot i.
static void f5iVertexDmem(struct Hle* h, int i, uint32_t a) {
	if (i < VERTICES) {
		f5Vertex(h, i, (int16_t) f5iHalf(h, a + 2), (int16_t) f5iHalf(h, a), (int16_t) f5iHalf(h, a + 6));
	}
}

static void f5iVertexRdram(struct N64* n, struct Hle* h, int i, uint32_t a) {
	if (i < VERTICES) {
		f5Vertex(h, i, rd16(n, a), rd16(n, a + 2), rd16(n, a + 4));
	}
}

// GLideN64's clip flags: past each side, or w under 0.01.
static uint32_t f5iClip(const struct Hle* h, int i) {
	const struct Vertex* v = &h->v[i];
	return (v->x > v->w) | (v->x < -v->w) << 1 | (v->y > v->w) << 2 | (v->y < -v->w) << 3 | (v->w < 0.01f) << 4;
}

static bool f5iClipped(const struct Hle* h, int a, int b, int c) {
	return a >= VERTICES || b >= VERTICES || c >= VERTICES || (f5iClip(h, a) & f5iClip(h, b) & f5iClip(h, c));
}

// A triangle as this microcode draws it: shading smooth whenever shaded (its G_SHADING_SMOOTH is
// G_SHADE), untextured ones not culled, textured ones with both culls set culled at the back only.
static void f5iTriangle(struct N64* n, struct Hle* h, int a, int b, int c, bool textured) {
	uint32_t const geometry = h->geometry;
	if (h->geometry & GEOM_SHADE) {
		h->geometry |= GEOM_SHADING_SMOOTH;
	}
	if (!textured) {
		h->geometry &= ~(uint32_t) (GEOM_CULL_FRONT | GEOM_CULL_BACK);
	} else if ((h->geometry & (GEOM_CULL_FRONT | GEOM_CULL_BACK)) == (GEOM_CULL_FRONT | GEOM_CULL_BACK)) {
		h->geometry &= ~(uint32_t) GEOM_CULL_FRONT;
	}
	drawTriangle(n, h, a, b, c);
	h->geometry = geometry;
}

static void f5iLoadStMatrix(struct Hle* h) {
	for (uint32_t i = 0; i < 16; ++i) {
		h->f5iStMtx[i] = (int32_t) ((uint32_t) f5iHalf(h, 0xE40 + 2 * (i ^ 1)) << 16 | f5iHalf(h, 0xE60 + 2 * (i ^ 1)));
	}
}

static void f5iDmaDirect(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	f5iCopyIn(n, h, (w0 >> 8) & 0xFFF, w1 & 0xFFFFFF, (((w0 & 0xFF) + 1) >> 2) * 4);
}

// The DMA from the list set up by G_SET_DLIST_ADDR (0xFD4: where, 0xFD8: the next block's link,
// 0xFDC: what is left of this block), going on into the next block as one runs out.
static void f5iDmaSegmented(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	if (!(w0 & 0xFF)) {
		return;
	}
	int32_t a0 = (int32_t) (w0 & 0xFF);
	uint32_t a2 = f5iWord(h, 0xFDC);
	int32_t a3 = a0 + 1, a1 = (int32_t) a2 - a3;
	uint32_t v1 = (w0 >> 8) & 0xFFF;
	for (int guard = 0; a1 < 0 && guard < 4096; ++guard) {
		uint32_t const v0 = f5iWord(h, 0xFD4);
		a3 = a0 - (int32_t) a2;
		f5iCopyIn(n, h, v1, v0 & 0xFFFFFF, a2);
		v1 += a2;
		a0 = a3;
		w1 = f5iWord(h, 0xFD8);
		f5iSetWord(h, 0xFD8, rd32(n, w1 & 0xFFFFFF));
		f5iSetWord(h, 0xFDC, 0x100);
		f5iSetWord(h, 0xFD4, w1 + 8);
		a2 = f5iWord(h, 0xFDC);
		a3 = a0 + 1;
		a1 = (int32_t) a2 - a3;
	}
	uint32_t const v0 = f5iWord(h, 0xFD4);
	f5iCopyIn(n, h, v1, v0 & 0xFFFFFF, (uint32_t) (a0 + 1));
	f5iSetWord(h, 0xFDC, (uint32_t) a1);
	f5iSetWord(h, 0xFD4, v0 + (uint32_t) a3);
}

static uint32_t minu(uint32_t a, uint32_t b) {
	return a < b ? a : b;
}

// The lighting overlays: colours at 0xD40 for the vertices, from their colours (at the command's
// address), the factors at 0x158 and the lights from 0xB10 (24 bytes each, their count at 0x58B).
static void f5iLighting1(struct Hle* h, uint32_t w0, uint32_t w1) {
	uint32_t const size = ((((w1 >> 24) & 0x7F) << 2) + 4) & 0xFFF8;
	for (uint32_t i = 0; i < size; ++i) {
		h->f5iDmem[(0xD40 + i) & 0xFFF] = 0;
	}
	uint8_t lights = h->f5iDmem[0x58B ^ 3];
	if (!lights) {
		f5iSetWord(h, 0x5B0, 0), f5iSetWord(h, 0x5B4, 0);
		return;
	}
	uint32_t light = 0xB10;
	uint32_t const dataAt = (w0 >> 8) & 0xFFF;
	uint8_t* m = h->f5iDmem;
	for (;;) {
		uint32_t dst = 0xD40, data = dataAt;
		if (!m[light]) {
			for (uint32_t i = 0; i < size >> 3; ++i) {
				uint32_t factor = 0x158;
				for (int half = 0; half < 2; ++half, factor += 4, dst += 4, data += 4) {
					m[dst & 0xFFF] = (uint8_t) ((m[data & 0xFFF] * m[factor]) >> 8);
					for (uint32_t j = 1; j < 4; ++j) {
						m[(dst + j) & 0xFFF] = (uint8_t) minu(((m[(data + j) & 0xFFF] * m[factor + j] * m[(light + j) & 0xFFF]) >> 16) + m[(dst + j) & 0xFFF], 0xFF);
					}
				}
			}
		} else {
			uint32_t offsetIndex = 0;
			uint64_t positions = f5iQuad(h, 0x5B0);
			uint32_t const factors[2] = { f5iWord(h, 0x158), f5iWord(h, 0x15C) };
			for (uint32_t i = 0; i < size >> 3; ++i) {
				uint32_t at[2];
				switch (positions & 3) {
				case 0:
					at[0] = at[1] = light - 0x10;
					break;
				case 1:
					at[0] = light - 0x08, at[1] = light - 0x10;
					break;
				case 2:
					at[0] = light - 0x10, at[1] = light - 0x08;
					break;
				default:
					at[0] = at[1] = light - 0x08;
					break;
				}
				positions >>= 2;
				for (int j = 0; j < 2; ++j) {
					int16_t const x1 = (int16_t) (f5iWord(h, at[j]) >> 16), y1 = (int16_t) f5iWord(h, at[j]), z1 = (int16_t) (f5iWord(h, at[j] + 4) >> 16);
					uint32_t const a = (0x170 + m[(0x380 + (offsetIndex ^ 3)) & 0xFFF]) & 0xFFC;
					++offsetIndex;
					int16_t const x = (int16_t) ((int16_t) (f5iWord(h, a) >> 16) - x1), y = (int16_t) ((int16_t) f5iWord(h, a) - y1);
					int16_t const z = (int16_t) ((int16_t) (f5iWord(h, a + 4) >> 16) - z1);
					uint64_t const t = 0xFFFF000000000000ull | (uint64_t) f5iWord(h, light + 4) << 16;
					uint64_t const k = (uint64_t) (int64_t) (x * x + y * y + z * z) * t;
					uint32_t const p = (uint32_t) (k >> 32), p1 = (p & 0xFFFF0000u) == 0xFFFF0000u ? p & 0xFFFF : 0;
					uint32_t const v = f5iWord(h, light);
					uint32_t d = ((v >> 24) * p1) >> 16, e = (((v >> 16) & 0xFF) * p1) >> 16, f = (((v >> 8) & 0xFF) * p1) >> 16;
					uint32_t const jj = ((factors[j] & 0xFF) * m[data & 0xFFF]) >> 8;
					if (m[light] == 4) {
						m[dst & 0xFFF] = (uint8_t) jj;
						m[(dst + 1) & 0xFFF] = (uint8_t) minu(m[(dst + 1) & 0xFFF] + f, 0xFF);
						m[(dst + 2) & 0xFFF] = (uint8_t) minu(m[(dst + 2) & 0xFFF] + e, 0xFF);
						m[(dst + 3) & 0xFFF] = (uint8_t) minu(m[(dst + 3) & 0xFFF] + d, 0xFF);
					} else {
						d += ((factors[j] >> 24) * m[(data + 3) & 0xFFF]) >> 8;
						e += (((factors[j] >> 16) & 0xFF) * m[(data + 2) & 0xFFF]) >> 8;
						f += (((factors[j] >> 8) & 0xFF) * m[(data + 1) & 0xFFF]) >> 8;
						m[dst & 0xFFF] = (uint8_t) jj;
						m[(dst + 1) & 0xFFF] = (uint8_t) minu(f >> 1, 0xFF);
						m[(dst + 2) & 0xFFF] = (uint8_t) minu(e >> 1, 0xFF);
						m[(dst + 3) & 0xFFF] = (uint8_t) minu(d >> 1, 0xFF);
					}
					dst += 4, data += 4;
				}
			}
		}
		if (!--lights) {
			f5iSetWord(h, 0x5B0, 0), f5iSetWord(h, 0x5B4, 0);
			if (f5iWord(h, 0x5A8) == 0xFFFFFFFFu) {
				return;
			}
			for (uint32_t i = 0; i < size; ++i) {
				m[(0xD40 + i) & 0xFFF] = (uint8_t) ((m[(0xD40 + i) & 0xFFF] * m[0x5A8 + (i & 3)]) >> 8);
			}
			return;
		}
		light += 0x18;
	}
}

static void f5iLighting2(struct Hle* h, uint32_t w0, uint32_t w1) {
	uint32_t const size = ((((w1 >> 24) & 0x7F) << 2) + 4) & 0xFFF8;
	for (uint32_t i = 0; i < size + 8; ++i) {
		h->f5iDmem[(0xD40 + i) & 0xFFF] = 0;
	}
	uint8_t lights = h->f5iDmem[0x58B ^ 3];
	if (!lights) {
		f5iSetWord(h, 0x5B0, 0), f5iSetWord(h, 0x5B4, 0);
		return;
	}
	uint32_t light = 0xB10;
	uint32_t const dataAt = (w0 >> 8) & 0xFFF;
	uint8_t* m = h->f5iDmem;
	for (;;) {
		uint32_t dst = 0xD40, data = dataAt;
		if (!m[light]) {
			for (uint32_t i = 0; i < size >> 3; ++i) {
				uint32_t factor = 0x158;
				for (int half = 0; half < 2; ++half, factor += 4, dst += 4, data += 4) {
					m[dst & 0xFFF] = (uint8_t) ((m[data & 0xFFF] * m[factor]) >> 8);
					for (uint32_t j = 1; j < 4; ++j) {
						m[(dst + j) & 0xFFF] = (uint8_t) minu(((m[(data + j) & 0xFFF] * m[factor + j] * m[(light + j) & 0xFFF]) >> 16) + m[(dst + j) & 0xFFF], 0xFF);
					}
				}
			}
		} else if (m[light] == 1) {
			uint32_t factor2 = 0x380;
			uint64_t positions = f5iQuad(h, 0x5B0);
			uint32_t const factors[2] = { f5iWord(h, 0x158), f5iWord(h, 0x15C) };
			for (uint32_t i = 0; i < size >> 3; ++i) {
				uint32_t at[2];
				switch (positions & 3) {
				case 0:
					at[0] = at[1] = light - 0x10;
					break;
				case 1:
					at[0] = light - 0x08, at[1] = light - 0x10;
					break;
				case 2:
					at[0] = light - 0x10, at[1] = light - 0x08;
					break;
				default:
					at[0] = at[1] = light - 0x08;
					break;
				}
				positions >>= 2;
				for (int j = 0; j < 2; ++j, factor2 += 4) {
					int16_t const x = (int16_t) (f5iWord(h, at[j]) >> 16), y = (int16_t) f5iWord(h, at[j]), z = (int16_t) (f5iWord(h, at[j] + 4) >> 16);
					uint32_t v = f5iWord(h, factor2);
					uint32_t in = (uint32_t) (((x * (int8_t) (v >> 24) + y * (int8_t) (v >> 16) + z * (int8_t) (v >> 8)) >> 8) & 0xFFFF);
					in = in & 0x8000 ? 0 : in;
					v = f5iWord(h, light);
					uint32_t const d = ((v >> 24) * in) >> 8, e = (((v >> 16) & 0xFF) * in) >> 8, f = (((v >> 8) & 0xFF) * in) >> 8;
					v = factors[j];
					uint32_t const k = (v & 0xFF) * m[data & 0xFFF], jj = ((v >> 8) & 0xFF) * m[(data + 1) & 0xFFF];
					uint32_t const hh = ((v >> 16) & 0xFF) * m[(data + 2) & 0xFFF], g = (v >> 24) * m[(data + 3) & 0xFFF];
					data += 4;
					m[dst & 0xFFF] = (uint8_t) (k >> 8);
					m[(dst + 1) & 0xFFF] = (uint8_t) minu(m[(dst + 1) & 0xFFF] + ((f * jj) >> 22), 0xFF);
					m[(dst + 2) & 0xFFF] = (uint8_t) minu(m[(dst + 2) & 0xFFF] + ((e * hh) >> 22), 0xFF);
					m[(dst + 3) & 0xFFF] = (uint8_t) minu(m[(dst + 3) & 0xFFF] + ((d * g) >> 22), 0xFF);
					dst += 4;
				}
			}
		} else {
			uint32_t offsetAt = 0x380;
			for (uint32_t i = 0; i < size >> 3; ++i, offsetAt += 8) {
				for (uint32_t j = 0; j < 2; ++j) {
					uint32_t const at = light - 0x10;
					int16_t const x1 = (int16_t) (f5iWord(h, at) >> 16), y1 = (int16_t) f5iWord(h, at), z1 = (int16_t) (f5iWord(h, at + 4) >> 16);
					uint32_t const a = 0x170 + (f5iWord(h, offsetAt + j * 4) & 0xFF);
					int16_t const x = (int16_t) ((int16_t) (f5iWord(h, a) >> 16) - x1), y = (int16_t) ((int16_t) f5iWord(h, a) - y1);
					int16_t const z = (int16_t) ((int16_t) (f5iWord(h, a + 4) >> 16) - z1);
					uint64_t const t = 0xFFFF000000000000ull | (uint64_t) f5iWord(h, light + 4) << 16;
					uint64_t const k = (uint64_t) (int64_t) (x * x + y * y + z * z) * t;
					uint32_t const p = (uint32_t) (k >> 32), p1 = (p & 0xFFFF0000u) == 0xFFFF0000u ? p & 0xFFFF : 0;
					uint32_t const factor = 0x158 + j * 4;
					m[dst & 0xFFF] = (uint8_t) ((m[data & 0xFFF] * m[factor]) >> 8);
					for (uint32_t c = 1; c < 4; ++c) {
						m[(dst + c) & 0xFFF] = (uint8_t) minu(((((m[(light + c) & 0xFFF] * p1) >> 16) * m[(data + c) & 0xFFF] * m[factor + c] + 0x8000) >> 16) + m[(dst + c) & 0xFFF], 0xFF);
					}
					dst += 4, data += 4;
				}
			}
		}
		if (!--lights) {
			f5iSetWord(h, 0x5B0, 0), f5iSetWord(h, 0x5B4, 0);
			return;
		}
		light += 0x18;
	}
}

static void f5iLightingBasic(struct Hle* h, uint32_t w0, uint32_t w1) {
	uint32_t dst = 0xD40, data = (w0 >> 8) & 0xFFF;
	for (int32_t count = (int32_t) ((w1 >> 24) & 0x7F); count > 0; count -= 4) {
		for (uint32_t j = 0; j < 16; ++j) {
			h->f5iDmem[(dst + j) & 0xFFF] = (uint8_t) ((h->f5iDmem[(data + j) & 0xFFF] * h->f5iDmem[0x158 + (j & 7)]) >> 8);
		}
		dst += 16, data += 16;
	}
}

// RGB565 colours at the address rebuilt as RGBA words in place.
static void f5iRebuildColors(struct Hle* h, uint32_t w0, uint32_t w1) {
	uint32_t const a = (w0 >> 8) & 0xFFF, count = minu(w1 >> 24, (0x588 - a) >> 2);
	uint32_t colors[256];
	for (uint32_t i = 0; i < count && i < 256; ++i) {
		uint16_t const v = f5iHalf(h, a + 2 * (i ^ 1));
		colors[i] = (uint32_t) ((v >> 8) & 0xF8) << 24 | (uint32_t) ((v & 0x7E0) >> 3) << 16 | (uint32_t) ((v & 0x1F) << 3) << 8 | 0xFF;
	}
	for (uint32_t i = 0; i < count && i < 256; ++i) {
		f5iSetWord(h, a + 4 * i, colors[i]);
	}
}

// The vertex base (0x128) every loaded vertex is moved by.
static void f5iBase(const struct Hle* h, uint16_t* x, uint16_t* y, uint16_t* z) {
	*x = f5iHalf(h, 0x128 + 2), *y = f5iHalf(h, 0x128), *z = f5iHalf(h, 0x128 + 6);
}

// Vertices by index: each six bytes into the RDRAM table at 0x154, moved by the base, into slots 0 on.
static void f5iLoadVertices(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	uint32_t const count = (w1 >> 24) & 0x7F, at = (w0 >> 8) & 0xFFF, base = f5iWord(h, 0x154);
	uint16_t x0, y0, z0;
	f5iBase(h, &x0, &y0, &z0);
	for (uint32_t i = 0; i < count && i < VERTICES; ++i) {
		uint32_t const a = f5iHalf(h, at + 2 * (i ^ 1)) * 6u + base, row = a & 0xFFFFF8, shift = (a & 7) >> 1;
		uint16_t const x = (uint16_t) (f5iRdHalf(n, row + 2 * ((shift + 0) ^ 1)) + x0), y = (uint16_t) (f5iRdHalf(n, row + 2 * ((shift + 1) ^ 1)) + y0);
		uint16_t const z = (uint16_t) (f5iRdHalf(n, row + 2 * ((shift + 2) ^ 1)) + z0);
		f5Vertex(h, (int) i, (int16_t) x, (int16_t) y, (int16_t) z);
	}
}

// Vertices in DMEM, moved by the base when it isn't zero.
static void f5iAdjustVertices(struct Hle* h, uint32_t w0, uint32_t w1) {
	uint32_t const count = (w1 >> 24) & 0x7F, at = (w0 >> 8) & 0xFFF;
	uint16_t x0 = 0, y0 = 0, z0 = 0;
	if (f5iQuad(h, 0x128)) {
		f5iBase(h, &x0, &y0, &z0);
	}
	for (uint32_t i = 0; i < count && i < VERTICES; ++i) {
		uint32_t const a = at + 8 * i;
		f5Vertex(h, (int) i, (int16_t) (f5iHalf(h, a + 2) + x0), (int16_t) (f5iHalf(h, a) + y0), (int16_t) (f5iHalf(h, a + 6) + z0));
	}
}

// G_MOVEMEM: a DMA into DMEM, then by its kind the viewport, the combined matrix, the matrix vertices
// are generated through (and the texture coordinate matrix in DMEM), vertices, the light count or
// the lighting.
static void f5iMoveMem(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	if (w1 & 0xFFFFFF) {
		f5iDmaDirect(n, h, w0, w1);
	} else {
		f5iDmaSegmented(n, h, w0, w1);
	}
	switch (w0 & 0x00F00000) {
	case 0x00000000:
		if (w0 == 0x0101300F) {
			viewport(n, h, address(h, w1));
		} else if (w0 == 0x0105C03F) {
			readMatrix(n, address(h, w1), h->mvp);
		} else if (w0 == 0x010E403F) {
			readMatrix(n, w1 & 0xFFFFFF, h->f5iGenMtx);
			f5iLoadStMatrix(h);
		}
		break;
	case 0x00100000:
		f5iAdjustVertices(h, w0, w1);
		break;
	case 0x00300000:
		h->f5iDmem[0x58B ^ 3] = (uint8_t) (w1 >> 24);
		break;
	case 0x00500000:
		f5iLoadVertices(n, h, w0, w1);
		break;
	default:
		if (((w0 >> 8) & 0xFFF) == 0x480) {
			uint32_t const f = w0 >> 22;
			if (f & 2) {
				f5iRebuildColors(h, w0, w1);
			}
			if (!(((w1 >> 24) | h->f5iDmem[0x58B ^ 3]) & 0x80)) {
				if (f & 1) {
					f5iLighting1(h, w0, w1);
				} else {
					f5iLighting2(h, w0, w1);
				}
			} else {
				f5iLightingBasic(h, w0, w1);
			}
		}
		break;
	}
}

// Texture coordinates from positions: the vertex (at 0x170 by the parameter's index) through the
// texture matrix, less the eye (0x0F0), normalized and scaled (0x0EC).
static void f5iCalcSt(struct Hle* h, const uint32_t* params, int count, uint32_t* st) {
	const int32_t* mtx = h->f5iStMtx;
	for (int i = 0; i < count; ++i) {
		uint32_t const a = 0x170 + (params[i + 4] & 0xFF);
		int32_t const x = (int16_t) f5iHalf(h, a + 2), y = (int16_t) f5iHalf(h, a), z = (int16_t) f5iHalf(h, a + 6);
		int32_t x1 = (int32_t) ((uint32_t) x * (uint32_t) mtx[0] + (uint32_t) y * (uint32_t) mtx[4] + (uint32_t) z * (uint32_t) mtx[8] + (uint32_t) mtx[12]);
		int32_t y1 = (int32_t) ((uint32_t) x * (uint32_t) mtx[1] + (uint32_t) y * (uint32_t) mtx[5] + (uint32_t) z * (uint32_t) mtx[9] + (uint32_t) mtx[13]);
		int32_t z1 = (int32_t) ((uint32_t) x * (uint32_t) mtx[2] + (uint32_t) y * (uint32_t) mtx[6] + (uint32_t) z * (uint32_t) mtx[10] + (uint32_t) mtx[14]);
		x1 = (int32_t) ((uint32_t) x1 - ((uint32_t) f5iHalf(h, 0x0F0 + 2) << 16));
		y1 = (int32_t) ((uint32_t) y1 - ((uint32_t) f5iHalf(h, 0x0F0) << 16));
		z1 = (int32_t) ((uint32_t) z1 - ((uint32_t) f5iHalf(h, 0x0F0 + 6) << 16));
		uint64_t const x2 = (uint64_t) (int64_t) x1, y2 = (uint64_t) (int64_t) y1, z2 = (uint64_t) (int64_t) z1;
		uint64_t const r = ((x2 * x2) >> 16) + ((y2 * y2) >> 16) + ((z2 * z2) >> 16);
		uint32_t r1 = (uint32_t) r;
		if (r > 0xFFFFFFFFull) {
			r1 = 0x7FFF0000u | (uint32_t) (r & 0xFFFF);
		}
		uint32_t d = (uint32_t) sqrt((double) r1);
		d = d ? 0xFFFFFFFFu / (d * 0x300) : 0xFFFFFFFFu;
		uint32_t const v = (uint32_t) ((d * x2) >> 16), w = (uint32_t) ((d * y2) >> 16);
		uint32_t const s = (v * f5iHalf(h, 0x0EC + 2)) >> 16, t = (w * f5iHalf(h, 0x0EC)) >> 16;
		st[i] = s << 16 | t;
	}
}

// The commands one at a time from where each list level is (shared by the main list and the sub-lists
// a triangle or rectangle starts).
struct F5iRun {
	uint32_t pc[DL_DEPTH + 2];
	int level;
	bool halt;
};

static void f5iCommand(struct N64* n, struct Hle* h, struct F5iRun* run);

// The sub-list at 0x58C, if one is set: run up to its end (0xB8 with -1) or until its next command is
// Naboo's 0xBD (that command then kept at 0xE58).
static void f5iSubList(struct N64* n, struct Hle* h, struct F5iRun* run) {
	uint32_t const list = f5iWord(h, 0x58C) & 0xFFFFFF;
	if (!list || run->level + 1 > DL_DEPTH) {
		return;
	}
	int const level = ++run->level;
	h->f5Jump[level] = rd32(n, list) & 0xFFFFFF;
	run->pc[level] = list + 8;
	for (int steps = 0; steps < 1 << 20 && !run->halt && run->level >= level; ++steps) {
		uint32_t const pc = run->pc[run->level], w0 = rd32(n, pc), w1 = rd32(n, pc + 4);
		if (w0 == 0xB8000000u && w1 == 0xFFFFFFFFu) {
			break;
		}
		int const next = (int) (rd32(n, pc + 8) >> 24);
		f5iCommand(n, h, run);
		if (next == 0xBD) {
			f5iSetWord(h, 0xE58, w0), f5iSetWord(h, 0xE5C, w1);
			break;
		}
	}
	run->level = level - 1;
	f5iSetWord(h, 0x58C, 0);
}

// G_TRI1 and G_TRI2: vertices by DMEM address ((a - 0x600) / 40), colours from 0xD40 by the second
// word's indices, texture coordinates after the command or computed; a sub-list runs before a
// triangle is drawn.
static uint32_t f5iTriangles(struct N64* n, struct Hle* h, struct F5iRun* run, uint32_t pc, bool two) {
	uint32_t params[8];
	for (int i = 0; i < 8; ++i) {
		params[i] = rd32(n, pc + 4 * (uint32_t) i);
	}
	uint32_t const w0 = params[0], w1 = params[1], w2 = params[2], w3 = params[3];
	int const v[4] = { (int) ((((w1 >> 16) & 0xFFF) - 0x600) / 40), (int) (((w1 & 0xFFF) - 0x600) / 40), (int) ((((w3 >> 16) & 0xFFF) - 0x600) / 40),
	                   (int) (((w3 & 0xFFF) - 0x600) / 40) };
	uint32_t const index[4] = { (w2 >> 16) & 0xFF, (w2 >> 8) & 0xFF, w2 & 0xFF, w2 >> 24 };
	bool const textured = w0 & 0x200;
	int const count = two ? 4 : 3;
	uint32_t st[4];
	if (w0 & 0x800) {
		f5iCalcSt(h, params, count, st);
	} else {
		for (int i = 0; i < 4; ++i) {
			st[i] = params[i + 4];
		}
	}
	bool const persp = h->modeH & 0x80000;
	for (int i = 0; i < count; ++i) {
		if (v[i] < 0 || v[i] >= VERTICES) {
			return pc + (textured ? 32 : 16);
		}
		struct Vertex* p = &h->v[v[i]];
		uint32_t const c = f5iWord(h, 0xD40 + index[i]);
		p->r = (c >> 24) / 255.0f, p->g = ((c >> 16) & 0xFF) / 255.0f, p->b = ((c >> 8) & 0xFF) / 255.0f, p->a = (c & 0xFF) / 255.0f;
		if (textured) {
			p->s = (int16_t) (st[i] >> 16) / (persp ? 32.0f : 16.0f), p->t = (int16_t) st[i] / (persp ? 32.0f : 16.0f);
		}
	}
	int const tris[2][3] = { { v[0], v[1], v[2] }, { v[0], v[2], v[3] } };
	bool const drawn[2] = { !f5iClipped(h, v[0], v[1], v[2]), two && !f5iClipped(h, v[0], v[2], v[3]) };
	if (drawn[0] || drawn[1]) {
		f5iSubList(n, h, run);
	}
	for (int k = 0; k < 2; ++k) {
		if (drawn[k]) {
			f5iTriangle(n, h, tris[k][0], tris[k][1], tris[k][2], textured);
		}
	}
	return pc + (textured ? 32 : 16);
}

// G_GEOMETRY_GEN 0x18: the vertices at 0x170 the mask names, through the generating matrix.
static void f5iGenVertices(struct Hle* h, uint32_t w0, uint32_t w1) {
	float combined[16];
	memcpy(combined, h->mvp, sizeof(combined));
	memcpy(h->mvp, h->f5iGenMtx, sizeof(combined));
	uint32_t a = (w0 & 0xFFFF) | (w1 & 0xFFFF0000u);
	for (int i = 0; a && i < 32; ++i) {
		uint32_t const c = a & (1u << i);
		if (c) {
			f5iVertexDmem(h, i, 0x170 + 8 * (uint32_t) i);
			a ^= c;
		}
	}
	memcpy(h->mvp, combined, sizeof(combined));
}

// G_GEOMETRY_GEN 0x27: particles moved along (their states in RDRAM, written back), respawned from the
// clock where they ran out.
static void f5iParticles(struct N64* n, struct Hle* h, uint32_t pc) {
	uint32_t params[6];
	for (int i = 0; i < 6; ++i) {
		params[i] = rd32(n, pc + 4 * (uint32_t) i);
	}
	f5iSetWord(h, 0xC10, rd32(n, params[3] & 0xFFFFFF)), f5iSetWord(h, 0xC14, rd32(n, (params[3] & 0xFFFFFF) + 4));
	uint32_t const live = f5iWord(h, 0xC10);
	if (!live) {
		return;
	}
	f5iCopyIn(n, h, 0x170, params[2] & 0xFFFFFF, 256);
	uint32_t const corrector = params[0] & 0xFFFF, m = params[5] & 0xFFFF;
	uint32_t vtx = 0x170, light = 0xB00, factorAt = 0x380, colorAt = 0x480, count = 0, gone = 0;
	for (uint32_t u = live; u; u >>= 1, vtx += 8, light += 8, factorAt += 8, colorAt += 8, ++count) {
		uint16_t f = (uint16_t) ((((uint32_t) f5iHalf(h, light + 2) << 12) + f5iHalf(h, light) * corrector) << 4 >> 16);
		bool ended = false;
		if (!(m & 0x2000)) {
			int16_t f1;
			do {
				f = (uint16_t) ((((uint32_t) f5iHalf(h, light + 2) << 12) + f5iHalf(h, light) * corrector) << 4 >> 16);
				f1 = (int16_t) (f + 0xF000);
				if (f1 < 0) {
					f5iSetHalf(h, light + 2, f);
					uint16_t flag = f5iHalf(h, vtx + 4);
					if (m & 0x40) {
						flag = (uint16_t) ((((f >> 4) & 0xFC) << 8) | (flag & 0xFF));
					}
					if (m & 0x20) {
						flag = (uint16_t) (((f >> 4) & 0xFC) | (flag & 0xFF00));
					}
					f5iSetHalf(h, vtx + 4, flag);
				} else {
					f5iSetHalf(h, light + 2, 0);
					if (m & 0x4000) {
						uint32_t const clock = busRead32(n, 0x04100010);
						uint32_t const v0 = f5iWord(h, 0xC18), v1 = f5iWord(h, 0xC1C), v2 = f5iWord(h, 0xC20), v3 = f5iWord(h, 0xC24);
						f5iSetHalf(h, vtx + 2, (uint16_t) ((clock & (v0 >> 16)) + (v2 >> 16)));
						f5iSetHalf(h, vtx, (uint16_t) (((clock >> 3) & (v0 & 0xFFFF)) + (v2 & 0xFFFF)));
						f5iSetHalf(h, vtx + 6, (uint16_t) (((clock >> 6) & (v1 >> 16)) + (v3 >> 16)));
						f5iSetHalf(h, vtx + 4, (uint16_t) (f5iHalf(h, vtx + 4) & (v1 & 0xFFFF)));
					} else {
						gone |= 1u << count;
						ended = true;
						break;
					}
				}
			} while (f1 >= 0);
		}
		if (ended) {
			continue;
		}
		uint32_t x = ((uint32_t) f5iHalf(h, vtx + 2) << 12) + (uint32_t) ((int16_t) (params[4] >> 16) * (int32_t) corrector);
		uint32_t y = ((uint32_t) f5iHalf(h, vtx) << 12) + (uint32_t) ((int16_t) params[4] * (int32_t) corrector);
		uint32_t z = ((uint32_t) f5iHalf(h, vtx + 6) << 12) + (uint32_t) ((int16_t) (params[5] >> 16) * (int32_t) corrector);
		if (m & 0x0380) {
			uint32_t offset = factorAt - 0x380;
			if (!(m & 0x0200)) {
				offset = m & 0x080 ? (uint32_t) ((f >> 4) & 0xF8) : (f5iWord(h, light + 4) >> 24) << 3;
			}
			uint32_t const w0 = f5iWord(h, 0x380 + offset), w1 = f5iWord(h, 0x380 + offset + 4);
			x += (uint32_t) ((int16_t) (w0 >> 16) * (int32_t) corrector), y += (uint32_t) ((int16_t) w0 * (int32_t) corrector);
			z += (uint32_t) ((int16_t) (w1 >> 16) * (int32_t) corrector);
		}
		if (m & 0x1C00) {
			uint32_t offset = colorAt - 0x480;
			if (!(m & 0x1000)) {
				offset = m & 0x0400 ? (uint32_t) ((f >> 4) & 0xF8) : ((f5iWord(h, light + 4) >> 16) & 0xFF) << 3;
			}
			uint32_t const w0 = f5iWord(h, 0x480 + offset), w1 = f5iWord(h, 0x480 + offset + 4);
			x += (uint32_t) ((int16_t) (w0 >> 16) * (int32_t) corrector), y += (uint32_t) ((int16_t) w0 * (int32_t) corrector);
			z += (uint32_t) ((int16_t) (w1 >> 16) * (int32_t) corrector);
		}
		int16_t vx = (int16_t) ((x << 4) >> 16), vy = (int16_t) ((y << 4) >> 16), vz = (int16_t) ((z << 4) >> 16);
		if (m & 0x0008) {
			uint32_t const v0 = f5iWord(h, 0xC00), v1 = f5iWord(h, 0xC04), v2 = f5iWord(h, 0xC08), v3 = f5iWord(h, 0xC0C);
			int16_t const lo[3] = { (int16_t) (v0 >> 16), (int16_t) v0, (int16_t) (v1 >> 16) }, span[3] = { (int16_t) (v2 >> 16), (int16_t) v2, (int16_t) (v3 >> 16) };
			int16_t* c[3] = { &vx, &vy, &vz };
			for (int k = 0; k < 3; ++k) {
				if (*c[k] < lo[k]) {
					*c[k] = (int16_t) (*c[k] + span[k]);
				}
				if (*c[k] > lo[k] + span[k]) {
					*c[k] = (int16_t) (*c[k] - span[k]);
				}
			}
		}
		f5iSetHalf(h, vtx + 2, (uint16_t) vx), f5iSetHalf(h, vtx, (uint16_t) vy), f5iSetHalf(h, vtx + 6, (uint16_t) vz);
	}
	f5iCopyOut(n, h, params[2] & 0xFFFFFF, 0x170, 256);
	if (!(m & 0x04) && !h->dry) {
		uint32_t const a = params[3] & 0xFFFFFF;
		n64RdramWrite(n, a, 4);
		uint32_t const v = live & ~gone;
		for (int k = 0; k < 4; ++k) {
			n->rdram[(a + (uint32_t) k) & (RDRAM_SIZE - 1)] = (uint8_t) (v >> (24 - 8 * k));
		}
		f5iCopyOut(n, h, (params[1] >> 8) & 0xFFFFFF, 0xB00, count * 8);
	}
}

// A texture rectangle as gDPTextureRectangle has it: from (ulx, uly) to (lrx, lry), S and T at its
// corner (10.5), the steps (5.10).
static void f5iRect(struct N64* n, struct Hle* h, float ulx, float uly, float lrx, float lry, int tile, int16_t s, int16_t t, float dsdx, float dtdy, bool flip) {
	int const xh = ulx > 0 ? (int) (ulx * 4) : 0, yh = uly > 0 ? (int) (uly * 4) : 0, xl = (int) (lrx * 4), yl = (int) (lry * 4);
	if (xl <= xh || yl <= yh) {
		return;
	}
	emit(n, h, (uint64_t) (flip ? 0x25 : 0x24) << 56 | (uint64_t) (xl & 0xFFF) << 44 | (uint64_t) (yl & 0xFFF) << 32 | (uint64_t) (tile & 7) << 24 |
	           (uint64_t) (xh & 0xFFF) << 12 | (uint64_t) (yh & 0xFFF));
	emit(n, h, (uint64_t) (uint16_t) s << 48 | (uint64_t) (uint16_t) t << 32 | (uint64_t) (uint16_t) (int16_t) (dsdx * 1024) << 16 | (uint16_t) (int16_t) (dtdy * 1024));
}

static void f5iPrim(struct N64* n, struct Hle* h, const struct Vertex* v, uint32_t color) {
	float const z = v->z / v->w * h->vscale[2] + h->vtrans[2];
	emit(n, h, (uint64_t) 0x2E << 56 | (uint64_t) (uint16_t) (int) (z * 32) << 16);
	emit(n, h, (uint64_t) 0x3A << 56 | color);
}

// Whether a vertex (in RDRAM) lies inside the box at lim.
static uint32_t f5iMask(struct N64* n, struct Hle* h, uint32_t mask, uint32_t lim, uint32_t vtx) {
	int16_t const xl = (int16_t) (f5iWord(h, lim) >> 16), yl = (int16_t) f5iWord(h, lim), zl = (int16_t) (f5iWord(h, lim + 4) >> 16);
	int16_t const xr = (int16_t) (f5iWord(h, lim + 8) >> 16), yr = (int16_t) f5iWord(h, lim + 8), zr = (int16_t) (f5iWord(h, lim + 12) >> 16);
	uint32_t s = 0xFFFFFFFEu;
	for (uint32_t i = 0; i < 0x20; ++i, s = s << 1 | 1) {
		int16_t const x = rd16(n, vtx + 8 * i), y = rd16(n, vtx + 8 * i + 2), z = rd16(n, vtx + 8 * i + 4);
		if (x > xl && x < xr && y > yl && y < yr && z > zl && z < zr) {
			mask &= s;
		}
	}
	return mask;
}

// G_GEOMETRY_GEN 0x24: Indiana Jones's sprites, a rectangle at each of up to 32 vertices a block, sized
// over w and textured as the command and the vertex's flag (colour, size) say.
static void f5iSprites(struct N64* n, struct Hle* h, uint32_t pc) {
	uint32_t params[10];
	for (int i = 0; i < 10; ++i) {
		params[i] = rd32(n, pc + 4 * (uint32_t) i);
	}
	uint32_t const vp = f5iWord(h, 0x130), vpx = (vp >> 17) & 0x7FFF, vpy = (vp >> 1) & 0x7FFF;
	uint32_t const maxX = ((params[0] & 0xFFFF) * vpx) >> 16, maxY = ((params[1] >> 16) * vpy) >> 16;
	uint64_t const mx = (params[4] & 0xFFFF0000u) | (params[5] >> 16), xScale = mx * (params[7] >> 16) * vpx;
	uint64_t const my = (params[4] << 16) | (params[5] & 0xFFFF), yScale = my * (params[7] & 0xFFFF) * vpy;
	uint32_t const mp = f5iWord(h, 0x5C8);
	uint64_t const q = (params[6] & 0xFFFF0000u) | (mp >> 16), r = (params[6] << 16) | (mp & 0xFFFF);
	uint32_t const scale = params[3];
	uint32_t vtx = params[2] & 0xFFFFFF;
	for (int32_t count = (int32_t) ((params[0] >> 16) & 0xFF); count > 0; --count, vtx += 0x100) {
		for (int i = 0; i < 0x20; ++i) {
			f5iVertexRdram(n, h, i, vtx + 8 * (uint32_t) i);
		}
		uint32_t k = params[8], lim = 0xC28;
		for (uint32_t b = params[9] & 0xFF; b; --b, lim += 0x10) {
			k = f5iMask(n, h, k, lim, vtx);
		}
		for (int i = 0; i < 0x20 && i < VERTICES; ++i, k >>= 1) {
			if (!(k & 1) || f5iClip(h, i)) {
				continue;
			}
			const struct Vertex* v = &h->v[i];
			float const sx = v->x / v->w * h->vscale[0] + h->vtrans[0], sy = -v->y / v->w * h->vscale[1] + h->vtrans[1];
			uint32_t const w = v->w > 1 ? (uint32_t) v->w : 1, invW = 0xFFFFFFFFu / w;
			uint16_t const flag = (uint16_t) rd16(n, vtx + 8 * (uint32_t) i + 6);
			uint32_t const val = f5iWord(h, 0x480 + (flag >> 8)), a = val >> 16, c = val & 0xFFFF;
			uint64_t ox = (xScale * invW) >> 32;
			ox = (ox * a) >> 32;
			ox = ox > maxX ? maxX : ox < 4 ? 4 : ox;
			float const ulx = sx - ox / 4.0f, lrx = sx + ox / 4.0f > 0 ? sx + ox / 4.0f : 0;
			int32_t const g = ulx > 0 ? 0 : (int32_t) ulx;
			if (lrx - ulx <= 0) {
				continue;
			}
			uint64_t oy = (yScale * invW) >> 32;
			oy = (oy * c) >> 32;
			oy = oy > maxY ? maxY : oy < 4 ? 4 : oy;
			float const uly = sy - oy / 4.0f, lry = sy + oy / 4.0f > 0 ? sy + oy / 4.0f : 0;
			int32_t const hh = uly > 0 ? 0 : (int32_t) uly;
			if (lry - uly <= 0) {
				continue;
			}
			uint64_t const q1 = q * 0x40 / ox, r1 = r * 0x40 / oy;
			uint16_t const dsdx = (uint16_t) (q1 >> 16), dtdy = (uint16_t) (r1 >> 16);
			uint64_t const u = q1 * 0x40 < 0x7FFFFFFFu ? q1 * 0x40 : 0x7FFFFFFFu, vv = r1 * 0x40 < 0x7FFFFFFFu ? r1 * 0x40 : 0x7FFFFFFFu;
			int16_t const s = (int16_t) (((uint16_t) (((0 - u) * (uint64_t) (int64_t) g) >> 16) - 2) * 8);
			int16_t const t = (int16_t) (((uint16_t) (((0 - vv) * (uint64_t) (int64_t) hh) >> 16) - 2) * 8);
			uint32_t const prim = f5iWord(h, 0x380 + (flag & 0xFF));
			uint32_t color = 0;
			for (int ch = 24; ch >= 0; ch -= 8) {
				color |= (((prim >> ch) & 0xFF) * ((scale >> ch) & 0xFF) >> 8) << ch;
			}
			f5iPrim(n, h, v, color);
			f5iRect(n, h, ulx, uly, lrx, lry, 0, s, t, (int16_t) dsdx / 1024.0f, (int16_t) dtdy / 1024.0f, false);
		}
	}
}

// The state-only commands (geometry, other modes, texture, words into DMEM, RDP state), shared with the
// pairs of commands Battle for Naboo keeps in DMEM.
static void f5iState(struct N64* n, struct Hle* h, uint32_t w0, uint32_t w1) {
	switch (w0 >> 24) {
	case 0xB6:
		setGeometry(h, ~w1, 0, false);
		break;
	case 0xB7:
		setGeometry(h, 0, w1, false);
		break;
	case 0xB9:
		if (((w0 >> 23) & 1) != f5iWord(h, 0x11C)) {
			break;
		}
		// fall through
	case 0xBA: {
		uint32_t const a0 = (w0 >> 16) & 7, mask = (uint32_t) ((int32_t) 0x80000000 >> (w0 & 31)) >> ((w0 >> 8) & 31);
		if (a0 == 0 || a0 == 4) {
			otherModeMask(n, h, a0 == 0, mask, w1);
		}
		break;
	}
	case 0xBB:
		h->texScaleS = (w1 >> 16) / 65536.0f, h->texScaleT = (w1 & 0xFFFF) / 65536.0f;
		h->texLevel = (w0 >> 11) & 7, h->texTile = (w0 >> 8) & 7, h->texOn = w0 & 0xFF;
		f5iSetWord(h, 0x148, w0);
		break;
	case 0xBC:
		f5iSetWord(h, w0 & 0xFFF, w1);
		if ((w0 & 0xFFF) == 0x160) {
			h->fogMul = (int32_t) w1 / 256.0f;
		} else if ((w0 & 0xFFF) == 0x164) {
			h->fogOffset = (int32_t) w1 / 256.0f;
		}
		break;
	default:
		if (w0 >> 24 >= 0xC0 && w0 >> 24 != 0xE4 && w0 >> 24 != 0xE5) {
			rdpCommand(n, h, 0, w0, w1);
		}
		break;
	}
}

// The pair of commands kept at a: sent to the RDP as they are, the other modes the microcode keeps
// left as they were.
static void f5nCommands(struct N64* n, struct Hle* h, uint32_t a) {
	uint32_t const modeH = h->modeH, modeL = h->modeL;
	f5iState(n, h, f5iWord(h, a), f5iWord(h, a + 4));
	f5iState(n, h, f5iWord(h, a + 8), f5iWord(h, a + 12));
	h->modeH = modeH, h->modeL = modeL;
}

// Battle for Naboo's terrain: a triangle drawn, the commands its kind keeps at 0xE50 (by 0x100's flags)
// first, or after the sub-list with the other modes saved there.
static void f5nTriangle(struct N64* n, struct Hle* h, struct F5iRun* run, const int* v) {
	if (f5iClipped(h, v[0], v[1], v[2])) {
		return;
	}
	uint32_t const a = (f5iWord(h, 0x100) >> 8) & 0xFFFF, b = h->f5iDmem[0x103 ^ 3], c = a | b;
	f5iSetHalf(h, 0x100 + 2 * (0 ^ 1), (uint16_t) c);
	if (!(f5iWord(h, 0x58C) & 0xFFFFFF)) {
		if ((c << 8) != b) {
			f5nCommands(n, h, 0xE50 + b);
		}
		f5iTriangle(n, h, v[0], v[1], v[2], true);
		return;
	}
	f5iSubList(n, h, run);
	uint32_t const modeH = 0xEF000000u | h->modeH;
	f5iSetWord(h, 0xE50, modeH), f5iSetWord(h, 0xE54, h->modeL);
	uint32_t const e = modeH & 0xFFCFFFFFu;
	f5iSetWord(h, 0xE70, e), f5iSetWord(h, 0xE60, e | 0x00100000u);
	uint32_t const g = h->f5iDmem[0x101 ^ 3];
	if (g) {
		f5nCommands(n, h, 0xE50 + g);
	}
	f5iTriangle(n, h, v[0], v[1], v[2], true);
}

// One of a terrain cell's triangles: its corners by index (through the table at 0xD58), their colours
// from 0xB78 and, at a transition, their alphas from 0xB00.
static void f5nCell(struct N64* n, struct Hle* h, struct F5iRun* run, uint32_t gg) {
	uint32_t const idx[3] = { f5iHalf(h, gg + 16) + (uint32_t) h->nabooCC, f5iHalf(h, gg + 22) + (uint32_t) h->nabooCC, f5iHalf(h, gg + 20) + (uint32_t) h->nabooCC };
	int v[3];
	uint32_t alphaSum = 0;
	for (int i = 0; i < 3; ++i) {
		uint32_t const at = f5iHalf(h, 0xD58 + 2 * ((idx[i] / 2) ^ 1));
		v[i] = (int) ((at - 0x600) / 40);
		if (v[i] < 0 || v[i] >= VERTICES) {
			return;
		}
		uint32_t const c = 0xB78 + idx[i] * 2 + ((uint32_t) h->nabooHH & 0xFFF);
		struct Vertex* p = &h->v[v[i]];
		p->r = h->f5iDmem[(c + 3) & 0xFFF] / 255.0f, p->g = h->f5iDmem[(c + 2) & 0xFFF] / 255.0f;
		p->b = h->f5iDmem[(c + 1) & 0xFFF] / 255.0f, p->a = h->f5iDmem[c & 0xFFF] / 255.0f;
		alphaSum += h->f5iDmem[c & 0xFFF];
	}
	uint8_t* flags = &h->f5iDmem[0x103 ^ 3];
	if (!h->nabooHH) {
		f5iSetWord(h, 0x100, 0);
	} else if (!alphaSum) {
		*flags = 0x00;
	} else if (h->nabooHH > 0) {
		*flags = 0x10;
	} else if (alphaSum != 0x2FD) {
		*flags = 0x00;
	} else {
		uint8_t alphas[3];
		uint32_t sum = 0;
		for (int i = 0; i < 3; ++i) {
			alphas[i] = h->f5iDmem[(0xB00 + idx[i] * 2 + ((uint32_t) h->nabooHH & 0xFFF)) & 0xFFF];
			sum += alphas[i];
		}
		if (!sum) {
			h->v[v[0]].a = alphas[0] / 255.0f;
			*flags = 0x20;
		} else {
			*flags = 0x10;
			for (int i = 0; i < 3; ++i) {
				h->v[v[i]].a = alphas[i] / 255.0f;
			}
		}
	}
	f5nTriangle(n, h, run, v);
}

// The terrain's polygons: cells walked by the two masks (AA, BB), two triangles a cell; 0xBE sets the
// texture coordinates of the corners first.
static void f5nPolygons(struct N64* n, struct Hle* h, struct F5iRun* run, uint32_t pc) {
	uint32_t const p0 = rd32(n, pc), p1 = rd32(n, pc + 4);
	int step = 5;
	if (p0 >> 24 == 0xBE) {
		h->nabooHH = (int16_t) p0;
		uint32_t const gg = (uint32_t) (h->nabooEE - h->nabooDD);
		uint32_t const idx[4] = { (f5iHalf(h, gg + 16) + h->nabooCC) >> 1, (f5iHalf(h, gg + 22) + h->nabooCC) >> 1, (f5iHalf(h, gg + 20) + h->nabooCC) >> 1,
		                          (f5iHalf(h, gg + 30) + h->nabooCC) >> 1 };
		int v[4];
		for (int i = 0; i < 4; ++i) {
			v[i] = (int) ((f5iHalf(h, 0xD58 + 2 * (idx[i] ^ 1)) - 0x600) / 40);
			if (v[i] < 0 || v[i] >= VERTICES) {
				return;
			}
		}
		float const t1 = (int16_t) p1 / 32.0f, s2 = (int16_t) (p1 >> 16) / 32.0f;
		h->v[v[0]].s = 0, h->v[v[0]].t = t1;
		h->v[v[1]].s = s2, h->v[v[1]].t = 0;
		h->v[v[2]].s = 0, h->v[v[2]].t = 0;
		h->v[v[3]].s = s2, h->v[v[3]].t = t1;
		step = 3;
	}
	for (int guard = 0; guard < 1 << 12; ++guard) {
		switch (step) {
		case 3:
			f5nCell(n, h, run, (uint32_t) (h->nabooEE - h->nabooDD));
			f5nCell(n, h, run, (uint32_t) (h->nabooEE - h->nabooDD + 8));
			step = 4;
			break;
		case 4:
			if (h->nabooDD == 0x10) {
				step = 6;
			} else {
				h->nabooCC = (uint8_t) (h->nabooCC + h->nabooTT);
				if (!((h->nabooAA & 0xFF) | ((uint16_t) h->nabooBB & 0xFF))) {
					return;
				}
				step = 5;
			}
			break;
		case 5:
			h->nabooDD = 0x10;
			if (!(h->nabooAA & 0x80)) {
				h->nabooAA = (uint16_t) (h->nabooAA << 1), step = 6;
			} else if (!(h->nabooAA & 0x8000)) {
				h->nabooAA = (uint16_t) (h->nabooAA << 1), step = 3;
			} else {
				h->nabooAA = (uint16_t) (h->nabooAA << 1);
				return;
			}
			break;
		default:
			h->nabooDD = 0;
			if (!(h->nabooBB & 0x80)) {
				h->nabooBB = (int16_t) ((uint16_t) h->nabooBB << 1), step = 4;
			} else if (!(h->nabooBB & 0x8000)) {
				h->nabooBB = (int16_t) ((uint16_t) h->nabooBB << 1), step = 3;
			} else {
				h->nabooBB = (int16_t) ((uint16_t) h->nabooBB << 1);
				return;
			}
			break;
		}
	}
}

// G_GEOMETRY_GEN 0x0F, first time: the terrain's tables at 0xDD0 and 0xC68 and the corner table at
// 0xD58 set up from the command (64 bytes).
static void f5nSetup(struct N64* n, struct Hle* h, uint32_t pc) {
	uint32_t p[16];
	for (int i = 0; i < 16; ++i) {
		p[i] = rd32(n, pc + 4 * (uint32_t) i);
	}
	f5iSetWord(h, 0x0F8, p[3]);
	f5iSetWord(h, 0x0FC, ((p[0] >> 16) & 0xFF) << 24 | (p[10] & 0xFF) << 16);
	uint32_t d[32];
	for (int i = 0; i < 32; ++i) {
		d[i] = f5iWord(h, 0xDD0 + 4 * (uint32_t) i);
	}
	d[0] = (p[2] & 0xFFFF0000u) | (p[14] >> 16);
	d[1] = (p[2] & 0xFFFF) << 16 | (d[1] & 0xFFFF);
	d[2] = (p[2] & 0xFFFF0000u) | (p[14] & 0xFFFF);
	d[3] = (p[2] & 0xFFFF) << 16 | 0x0100;
	d[4] = (p[4] & 0xFFFF0000u) | (d[4] & 0xFFFF), d[6] = (p[4] & 0xFFFF0000u) | (d[6] & 0xFFFF);
	d[8] = (p[4] & 0xFFFF) << 16 | (d[8] & 0xFFFF), d[10] = (p[4] & 0xFFFF) << 16 | (d[10] & 0xFFFF);
	d[12] = (p[5] & 0xFFFF0000u) | (d[12] & 0xFFFF), d[14] = (p[5] & 0xFFFF0000u) | (d[14] & 0xFFFF);
	d[16] = (p[5] & 0xFFFF) << 16 | (d[16] & 0xFFFF), d[18] = (p[5] & 0xFFFF) << 16 | (d[18] & 0xFFFF);
	d[5] = d[7] = (p[6] & 0xFFFF0000u) | (p[8] >> 16);
	d[9] = d[11] = (p[6] & 0xFFFF) << 16 | (p[8] & 0xFFFF);
	d[13] = d[15] = (p[7] & 0xFFFF0000u) | (p[9] >> 16);
	d[17] = d[19] = (p[7] & 0xFFFF) << 16 | (p[9] & 0xFFFF);
	d[20] = (p[0] & 0xFFFF) << 16 | (p[13] >> 16);
	d[21] = (p[15] & 0xFFFF0000u) | (d[21] & 0xFFFF);
	d[22] = p[11] & 0xFF;
	if (d[22] > 127) {
		d[22] |= 0xFF00;
	}
	d[22] |= (((p[0] & 0xFFFF) + 0x00FF) & 0xFF00) << 16;
	d[23] = (p[15] & 0xFFFF) << 16 | (d[23] & 0xFFFF);
	d[24] = d[26] = 0x7FFF7FFF;
	d[25] = d[27] = p[12];
	d[28] = d[29] = p[10];
	d[30] = d[31] = p[11];
	for (int i = 0; i < 32; ++i) {
		f5iSetWord(h, 0xDD0 + 4 * (uint32_t) i, d[i]);
	}
	f5iCopyIn(n, h, 0xC68, (p[1] >> 8) & 0xFFFFFF, 0xF0);
	h->f5iDmem[0x37E ^ 3] = 0xFF;
	static const uint32_t corners[14] = { 0x06000628, 0x06500678, 0x06A006C8, 0x06F00718, 0x07400768, 0x079007B8, 0x07E00808,
	                                      0x08300858, 0x088008A8, 0x08D008F8, 0x09200948, 0x09700998, 0x09C009E8, 0x0A100A38 };
	for (int i = 0; i < 14; ++i) {
		f5iSetWord(h, 0xD58 + 4 * (uint32_t) i, corners[i]);
	}
	f5iSetWord(h, 0xD90, 0x0A600000u | (f5iWord(h, 0xD90) & 0xFFFF));
}

// G_GEOMETRY_GEN 0x0F, later times: the terrain's step values at 0xDB8 from the command.
static void f5nUpdate(struct N64* n, struct Hle* h, uint32_t pc) {
	uint32_t const p0 = rd32(n, pc), p1 = rd32(n, pc + 4), p2 = rd32(n, pc + 8), p3 = rd32(n, pc + 12);
	f5iSetWord(h, 0x100, 0);
	uint32_t d[30];
	for (int i = 0; i < 30; ++i) {
		d[i] = f5iWord(h, 0xDB8 + 4 * (uint32_t) i);
	}
	d[0] = d[0] & 0xFFFF;
	d[1] = (p0 & 0xFFFF) << 16 | (d[1] & 0xFFFF);
	uint32_t v = (p0 >> 16) & 0xFF;
	d[2] = v << 24 | 0x00FFFF, d[3] = v * 2, d[4] = v << 24;
	v = p1 >> 24;
	d[5] = v, d[7] = (d[7] & 0xFFFF0000u) | (v * 2);
	v = p2 >> 16;
	d[10] = (d[10] & 0xFFFF0000u) | v, d[12] = (d[12] & 0xFFFF0000u) | v;
	v = p2 & 0xFFFF;
	d[14] = (d[14] & 0xFFFF0000u) | v, d[16] = (d[16] & 0xFFFF0000u) | v;
	v = p3 >> 16;
	d[18] = (d[18] & 0xFFFF0000u) | v, d[20] = (d[20] & 0xFFFF0000u) | v;
	v = p3 & 0xFFFF;
	d[22] = (d[22] & 0xFFFF0000u) | v, d[24] = (d[24] & 0xFFFF0000u) | v;
	v = (p1 >> 8) & 0xFFFF;
	d[27] = (d[27] & 0xFFFF0000u) | v, d[29] = (d[29] & 0xFFFF0000u) | v;
	for (int i = 0; i < 30; ++i) {
		f5iSetWord(h, 0xDB8 + 4 * (uint32_t) i, d[i]);
	}
}

struct F5nVertex {
	uint16_t x, y, z, flag;
	uint32_t e;
	uint16_t k, l, m, n, j;
	uint8_t c1[4], c2[4];
};

static int32_t f5nMultClip(int32_t s, int32_t t, int32_t max) {
	if (((uint32_t) s ^ (uint32_t) t) & 0x80000000u) {
		return 0;
	}
	int64_t const r = ((int64_t) s * t) >> 32;
	return (int32_t) (r < max ? r : max);
}

static uint32_t f5nColor(const uint8_t* c) {
	return (uint32_t) c[0] << 24 | (uint32_t) c[1] << 16 | (uint32_t) c[2] << 8 | c[3];
}

// G_GEOMETRY_GEN 0x09 and 0x0C: two rows of terrain vertices from heights and colours in RDRAM, with
// distance fades (K to N, J), written to DMEM, then loaded and the cells drawn.
static void f5nGenerate(struct N64* n, struct Hle* h, struct F5iRun* run, uint32_t pc, bool lit) {
	uint32_t p[6];
	for (int i = 0; i < 6; ++i) {
		p[i] = rd32(n, pc + 4 * (uint32_t) i);
	}
	if (lit) {
		f5iCopyIn(n, h, 0x380, f5iWord(h, 0x0F8) & 0xFFFFFF, 512);
	}
	uint32_t const src = p[0] & 0xFFF, dst = f5iWord(h, src) >> 16, from = p[4] & 0xFFFFFF;
	uint32_t const dst2 = f5iWord(h, src + 8) >> 16, shift = f5iWord(h, 0xDBC) >> 16;
	f5iCopyIn(n, h, dst, from, 40);
	f5iCopyIn(n, h, dst2, from + shift, 40);
	if (((p[0] >> 12) & 15) > 7) {
		uint32_t const from2 = p[5] & 0xFFFFFF;
		f5iCopyIn(n, h, dst + 0x78, from2, 40);
		f5iCopyIn(n, h, dst2 + 0x78, from2 + shift, 40);
	}
	uint32_t const c32 = (uint32_t) ((int8_t) (p[1] >> 24) * 0x10000) - (f5iWord(h, 0xDD0) >> 16);
	int64_t const c64 = (int32_t) c32;
	uint32_t const c2 = (uint32_t) ((c64 * c64) >> 16);
	uint32_t g = c32 * (f5iWord(h, 0xDC4) & 0xFFFF) + ((f5iWord(h, 0xDCC) & 0xFFFF) << 16);
	struct F5nVertex vd[2];
	uint32_t const starts[2] = { p[1] + (f5iWord(h, 0xDC0) >> 16), p[1] }, bases[2] = { 0xDD0, 0xDD8 };
	for (int k = 0; k < 2; ++k) {
		uint32_t const v = starts[k];
		memset(&vd[k], 0, sizeof(vd[k]));
		vd[k].x = (uint16_t) (((v >> 24) & 0xFF) << 8), vd[k].z = (uint16_t) (((v >> 8) & 0xFF) << 8), vd[k].flag = (uint16_t) ((v & 0xFF) << 8);
		int64_t const d64 = (int32_t) ((uint32_t) ((int8_t) (v >> 8) * 0x10000) - (f5iWord(h, bases[k] + 4) >> 16));
		vd[k].e = c2 + (uint32_t) ((d64 * d64) >> 16);
	}
	uint32_t const e20 = f5iWord(h, 0xE20), e24 = f5iWord(h, 0xE24), e28 = f5iWord(h, 0xE28), e2c = f5iWord(h, 0xE2C);
	uint32_t const ii = (e2c & 0xFFFF0000u) | (e24 >> 16);
	uint32_t const de0 = f5iWord(h, 0xDE0), de4 = f5iWord(h, 0xDE4), df0 = f5iWord(h, 0xDF0), df4 = f5iWord(h, 0xDF4);
	int32_t const kk = (int32_t) ((df0 & 0xFFFF0000u) | (de0 >> 16)), ll = (int32_t) ((df0 & 0xFFFF) << 16 | (de0 & 0xFFFF));
	int32_t const mm = (int32_t) ((df4 & 0xFFFF0000u) | (de4 >> 16)), nn = (int32_t) ((df4 & 0xFFFF) << 16 | (de4 & 0xFFFF));
	uint32_t const e00 = f5iWord(h, 0xE00), e04 = f5iWord(h, 0xE04), e10 = f5iWord(h, 0xE10), e14 = f5iWord(h, 0xE14);
	int32_t const oo = (int32_t) ((e10 & 0xFFFF0000u) | (e00 >> 16)), pp = (int32_t) ((e10 & 0xFFFF) << 16 | (e00 & 0xFFFF));
	int32_t const qq = (int32_t) ((e14 & 0xFFFF0000u) | (e04 >> 16)), rr = (int32_t) ((e14 & 0xFFFF) << 16 | (e04 & 0xFFFF));
	uint32_t const e34 = f5iWord(h, 0xE34), e40 = f5iWord(h, 0xE40), e48 = f5iWord(h, 0xE48);
	uint32_t const dd0 = f5iWord(h, 0xDD0), dd6 = (f5iWord(h, 0xDD4) & 0xFFFF) << 16, dd8 = f5iWord(h, 0xDD8);
	uint8_t const shiftR = h->f5iDmem[0x0FC ^ 3], shiftH = h->f5iDmem[0x0FD ^ 3];
	for (uint32_t at = p[4] >> 24; at <= p[5] >> 24; at += 4) {
		for (int k = 0; k < 2; ++k) {
			struct F5nVertex* d = &vd[k];
			uint64_t j = ((uint64_t) d->e * ii) >> 16;
			j *= j;
			d->k = (uint16_t) f5nMultClip(kk - (int32_t) d->e, oo, 0x7FFF), d->l = (uint16_t) f5nMultClip(ll - (int32_t) d->e, pp, 0x7FFF);
			d->m = (uint16_t) f5nMultClip(mm - (int32_t) d->e, qq, (int32_t) (e34 >> 16)), d->n = (uint16_t) f5nMultClip(nn - (int32_t) d->e, rr, (int32_t) (e34 & 0xFFFF));
			uint16_t const jj = (uint16_t) (j >> 47);
			d->j = jj < 0x7FFF ? jj : 0x7FFF;
		}
		for (int k = 0; k < 2; ++k) {
			struct F5nVertex* d = &vd[k];
			uint32_t const a = (f5iWord(h, src + 8 * (uint32_t) k) >> 16) + at;
			uint32_t const factor = 0xFFFE - d->m - d->n, f3 = 0x7FFF - d->k;
			if (lit) {
				uint32_t const u1 = f5iWord(h, a), u2 = f5iWord(h, a + 0x78);
				for (int c = 0; c < 3; ++c) {
					int const sh = 24 - 8 * c;
					d->c1[c] = (uint8_t) ((factor * ((u1 >> sh) & 0xFF) + d->m * ((e40 >> sh) & 0xFF) + d->n * ((e48 >> sh) & 0xFF)) >> 16);
				}
				uint8_t const i1 = (uint8_t) ((u1 & 0xFF) + shiftR), i2 = (uint8_t) ((u2 & 0xFF) + shiftH);
				uint32_t const u3 = (uint32_t) (int8_t) h->f5iDmem[(0x380 + i1) ^ 3], u4 = (uint32_t) (int8_t) h->f5iDmem[(0x380 + i2) ^ 3];
				uint32_t const l1 = (e28 & 0xFFFF) << 16 | (e20 & 0xFFFF), l2 = (dd8 & 0xFFFF) << 16 | (dd0 & 0xFFFF);
				uint16_t const u5 = (uint16_t) ((u3 * l1 + u4 * l2) >> 16), u6 = (uint16_t) (u5 + (e20 >> 16));
				uint16_t const u7 = (uint16_t) (((0x7FFF - d->k) * (uint32_t) u6 + d->k * (e28 >> 16)) >> 16);
				d->y = (uint16_t) (((0x7FFF - d->j) * (uint32_t) u7 + d->j * (e24 & 0xFFFF)) >> 16);
				uint8_t const a1 = h->f5iDmem[(0x4FF + u4) ^ 3];
				uint16_t const f1 = (uint16_t) (0xFF - a1), f2 = (uint16_t) (0xFF * a1);
				for (int c = 0; c < 3; ++c) {
					int const sh = 24 - 8 * c;
					uint8_t const vc = (uint8_t) ((f1 * ((u2 >> sh) & 0xFF) + f2) >> 8);
					d->c2[c] = (uint8_t) ((factor * vc + d->m * ((e40 >> sh) & 0xFF) + d->n * ((e48 >> sh) & 0xFF)) >> 16);
					d->c2[c] = (uint8_t) ((((f3 * d->c2[c] + d->k * d->c1[c]) << 1) >> 16) + 1);
				}
				d->c1[3] = (uint8_t) ((d->k * d->k) >> 22);
				d->c2[3] = (uint8_t) (((((0x7FFF - d->l) * (uint32_t) a1 + d->l * 0xFFu) << 1) >> 16) + 1);
			} else {
				uint32_t const vv = f5iWord(h, src + 0x20 + 4 * (uint32_t) k);
				uint32_t const a3 = (vv >> 16) + at + (at & 4), a4 = (vv & 0xFFFF) + at - (at & 4), f1 = 0x7FFF - d->l;
				uint16_t ys[2];
				for (int half = 0; half < 2; ++half) {
					uint32_t const off = half ? 0x78 : 0, u = f5iWord(h, a + off);
					uint32_t t = 0;
					for (int b = 0; b < 4; ++b) {
						t |= (uint32_t) ((h->f5iDmem[(a3 + off + (uint32_t) b) & 0xFFF] + h->f5iDmem[(a4 + off + (uint32_t) b) & 0xFFF]) >> 1) << (8 * b);
					}
					uint8_t* col = half ? d->c2 : d->c1;
					for (int c = 0; c < 3; ++c) {
						int const sh = 24 - 8 * c;
						uint64_t va = (uint64_t) (f1 * ((u >> sh) & 0xFF) + d->l * ((t >> sh) & 0xFF));
						va = (va * factor) >> 16;
						col[c] = (uint8_t) (((va << 1) + d->m * ((e40 >> sh) & 0xFF) + d->n * ((e48 >> sh) & 0xFF)) >> 16);
					}
					uint32_t const ta = ((f5iWord(h, a3 + off) & 0xFF) << 7) + ((f5iWord(h, a4 + off) & 0xFF) << 7);
					uint32_t const vy = ((f1 * ((u & 0xFF) << 8)) >> 16) + ((d->l * ta) >> 16);
					ys[half] = (uint16_t) ((vy * (0x7FFF - d->j) + d->j * (e24 & 0xFFFF)) >> 16);
				}
				for (int c = 0; c < 3; ++c) {
					d->c2[c] = (uint8_t) ((((f3 * d->c2[c] + d->k * d->c1[c]) << 1) >> 16) + 1);
				}
				d->y = (uint16_t) (((f3 * ys[1] + d->k * ys[0]) << 1) >> 16);
				d->c1[3] = d->c2[3] = (uint8_t) ((d->k * d->k) >> 22);
			}
			f5iSetWord(h, a + 0x0F0, f5nColor(d->c1));
			f5iSetWord(h, a + 0x168, f5nColor(d->c2));
		}
		for (int k = 0; k < 2; ++k) {
			uint32_t const v = (k ? 0x1C0 : 0x170) + at * 2;
			f5iSetHalf(h, v, vd[k].y), f5iSetHalf(h, v + 2, vd[k].x), f5iSetHalf(h, v + 4, vd[k].flag), f5iSetHalf(h, v + 6, vd[k].z);
		}
		uint32_t const v0 = f5iWord(h, 0xDC0);
		vd[0].x = (uint16_t) (vd[0].x + (v0 >> 16)), vd[0].flag = (uint16_t) (vd[0].flag + (f5iWord(h, 0xDC4) & 0xFFFF));
		vd[1].x = (uint16_t) (vd[1].x + (v0 >> 16)), vd[1].flag = (uint16_t) (vd[1].flag + (f5iWord(h, 0xDCC) & 0xFFFF));
		vd[0].e += g, vd[1].e += g;
		g += dd6;
	}
	// The rows loaded (moved by the base, x by its z too, as GLideN64 has it).
	uint16_t x0, y0, z0;
	f5iBase(h, &x0, &y0, &z0);
	uint32_t const rows[2][3] = { { p[2] >> 16, 0x170, 0x10 }, { p[2] & 0xFFFF, 0x1C0, 0x18 } };
	for (int r = 0; r < 2; ++r) {
		uint32_t const count = rows[r][0] & 0x1F, offset = rows[r][0] >> 5;
		uint32_t const from2 = rows[r][1] + offset, to = (f5iWord(h, src + rows[r][2]) >> 16) + offset * 5;
		for (uint32_t i = 0; i < count; ++i) {
			uint32_t const a = from2 + 8 * i;
			int const slot = (int) ((to - 0x600) / 40 + i);
			if (slot >= 0 && slot < VERTICES) {
				f5Vertex(h, slot, (int16_t) (f5iHalf(h, a + 2) + x0 + z0), (int16_t) (f5iHalf(h, a) + y0), (int16_t) f5iHalf(h, a + 6));
			}
		}
	}
	h->nabooAA = f5iRdHalf(n, pc + 14), h->nabooBB = (int16_t) f5iRdHalf(n, pc + 12);
	h->nabooCC = f5iRdByte(n, pc + 2), h->nabooTT = f5iRdByte(n, pc + 6);
	h->nabooEE = (int16_t) src, h->nabooHH = (int16_t) (f5iWord(h, 0xDB8) >> 16), h->nabooDD = 0;
	f5nPolygons(n, h, run, pc);
}

// G_GEOMETRY_GEN 0x15: Battle for Naboo's sprite at a vertex (as Rogue Squadron's G_TEXRECT_GEN, its
// projection scale in the command), after the sub-list.
static void f5nSprite(struct N64* n, struct Hle* h, struct F5iRun* run, uint32_t pc) {
	uint32_t p[8];
	for (int i = 0; i < 8; ++i) {
		p[i] = rd32(n, pc + 4 * (uint32_t) i);
	}
	int const index = (int) (((p[0] >> 5) & 0x7F8) / 40);
	if (index >= VERTICES || f5iClip(h, index)) {
		return;
	}
	const struct Vertex* v = &h->v[index];
	float const sx = v->x / v->w * h->vscale[0] + h->vtrans[0], sy = -v->y / v->w * h->vscale[1] + h->vtrans[1];
	bool const flip = p[0] & 1;
	uint64_t const w = v->w > 1 ? (uint64_t) v->w : 1;
	uint32_t const vp = f5iWord(h, 0x130), viewX = (vp >> 17) & 0x7FFF, viewY = (vp >> 1) & 0x7FFF;
	uint32_t const perspX = (p[5] & 0xFFFF0000u) | (p[6] >> 16), perspY = (p[5] & 0xFFFF) << 16 | (p[6] & 0xFFFF);
	uint64_t p3x = p[3] >> 16, p3y = p[3] & 0xFFFF;
	if (flip) {
		uint64_t const t = p3x;
		p3x = p3y, p3y = t;
	}
	uint32_t ox = (uint32_t) (((p3x * viewX * perspX) / w) >> 16), oy = (uint32_t) (((p3y * viewY * perspY) / w) >> 16);
	float const ulx = sx - ox / 4.0f, lrx = sx + ox / 4.0f, uly = sy - oy / 4.0f, lry = sy + oy / 4.0f;
	if (lrx - ulx <= 0 || lry - uly <= 0 || !ox || !oy) {
		return;
	}
	float whole;
	uint32_t const p4x = (p[4] & 0xFFFF0000u) | (uint32_t) (fabsf(modff(h->mvp[0], &whole)) * 65536.0f);
	uint32_t const p4y = (p[4] & 0xFFFF) << 16 | (uint32_t) (fabsf(modff(h->mvp[1], &whole)) * 65536.0f);
	if (flip) {
		uint32_t const t = ox;
		ox = oy, oy = t;
	}
	uint16_t dsdx = (uint16_t) ((p4x / ox) >> 10), dtdy = (uint16_t) ((p4y / oy) >> 10), e = 0, f = 0;
	if (p[0] & 2) {
		dsdx = (uint16_t) -dsdx, e = (uint16_t) (p[4] >> 16);
	}
	if (p[0] & 4) {
		dtdy = (uint16_t) -dtdy, f = (uint16_t) p[4];
	}
	float const rectDsdx = (int16_t) dsdx / 1024.0f, rectDtdy = (int16_t) dtdy / 1024.0f;
	if (flip) {
		uint16_t const t = dsdx;
		dsdx = dtdy, dtdy = t;
	}
	int16_t const s = (int16_t) (ulx > 0 ? (uint16_t) (e + 0xFFF0) : (uint16_t) (((0 - (dsdx << 6) * (int) (ulx * 4)) << 3) + e + 0xFFF0));
	int16_t const t = (int16_t) (uly > 0 ? (uint16_t) (f + 0xFFF0) : (uint16_t) (((0 - (dtdy << 6) * (int) (uly * 4)) << 3) + f + 0xFFF0));
	f5iSubList(n, h, run);
	f5iPrim(n, h, v, p[7]);
	if (h->geometry & GEOM_FOG) {
		emit(n, h, (uint64_t) 0x38 << 56 | (p[7] & 0xFFFFFF00u) | (uint32_t) (v->a * 255.0f));
	}
	f5iRect(n, h, ulx, uly, lrx, lry, h->texTile, s, t, rectDsdx, rectDtdy, flip);
}

// One command at the current level's place; it moves that place on (or to another level).
static void f5iCommand(struct N64* n, struct Hle* h, struct F5iRun* run) {
	int const level = run->level;
	uint32_t const pc = run->pc[level], w0 = rd32(n, pc), w1 = rd32(n, pc + 4);
	uint32_t next = pc + 8;
	switch (w0 >> 24) {
	case 0x00:
		break;
	case 0x01:
		f5iMoveMem(n, h, w0, w1);
		break;
	case 0x02:
		f5iSetWord(h, 0xFD4, w1 + 8), f5iSetWord(h, 0xFD8, rd32(n, w1 & 0xFFFFFF)), f5iSetWord(h, 0xFDC, w0 & 0x1FF);
		break;
	case 0x05:
		switch (w1 & 0xFF) {
		case 0x09:
		case 0x0C:
			f5nGenerate(n, h, run, pc, (w1 & 0xFF) == 0x0C);
			next = pc + 24;
			break;
		case 0x0F:
			if (!h->f5iDmem[0x37E ^ 3]) {
				f5nSetup(n, h, pc);
				next = pc + 64;
			} else {
				f5nUpdate(n, h, pc);
				next = pc + 16;
			}
			break;
		case 0x15:
			f5nSprite(n, h, run, pc);
			next = pc + 32;
			break;
		case 0x18:
			f5iGenVertices(h, w0, w1);
			break;
		case 0x24:
			f5iSubList(n, h, run);
			f5iSprites(n, h, pc);
			next = pc + 40;
			break;
		case 0x27:
			f5iParticles(n, h, pc);
			next = pc + 24;
			break;
		case 0x4F:
			h->f5iDmem[0x37E ^ 3] = 0;
			break;
		}
		break;
	case 0x06:
	case 0x07:
		if ((w0 >> 24) == 0x06) {
			if (level + 1 > DL_DEPTH) {
				break;
			}
			run->pc[level] = next;
			run->level = level + 1;
		}
		next = address(h, w1);
		h->f5Jump[run->level] = rd32(n, next) & 0xFFFFFF;
		run->pc[run->level] = next + 8;
		return;
	case 0xB4:
	case 0xBF:
		next = f5iTriangles(n, h, run, pc, (w0 >> 24) == 0xB4);
		break;
	case 0xB5:
		next = h->f5Jump[level];
		h->f5Jump[level] = rd32(n, next) & 0xFFFFFF;
		next += 8;
		break;
	case 0xB8:
		if (!level) {
			run->halt = true;
			return;
		}
		run->level = level - 1;
		return;
	case 0xBE:
		f5nPolygons(n, h, run, pc);
		break;
	case 0xE4:
	case 0xE5:
		if ((w0 >> 24) == 0xE4) {
			f5iSubList(n, h, run);
		}
		emit(n, h, (uint64_t) w0 << 32 | w1);
		emit(n, h, (uint64_t) rd32(n, pc + 8) << 32 | rd32(n, pc + 12));
		next = pc + 16;
		break;
	default:
		f5iState(n, h, w0, w1);
		break;
	}
	if (run->level == level) {
		run->pc[level] = next;
	}
}

static void f5iList(struct N64* n, struct Hle* h, uint32_t pc) {
	struct F5iRun run = { { 0 }, 0, false };
	h->f5Jump[0] = rd32(n, pc) & 0xFFFFFF;
	run.pc[0] = pc + 8;
	static const uint32_t cleared[7] = { 0x11C, 0x120, 0x124, 0x37C, 0x58C, 0x5B0, 0x5B4 };
	for (int i = 0; i < 7; ++i) {
		f5iSetWord(h, cleared[i], 0);
	}
	for (int steps = 0; steps < 1 << 22 && !run.halt; ++steps) {
		f5iCommand(n, h, &run);
	}
}

static void zsortList(struct N64* n, struct Hle* h, uint32_t pc) {
	int sp = 0;
	for (int steps = 0; pc && steps < 1 << 20; ++steps) {
		uint32_t const w0 = rd32(n, pc), w1 = rd32(n, pc + 4);
		uint32_t next = pc + 8;
		int const op = w0 >> 24;
		switch (op) {
		case 0x80:
			zsortObjects(n, h, address(h, w0));
			zsortObjects(n, h, address(h, w1));
			break;
		case 0x81:
			zsortRdp(n, h, address(h, w1));
			break;
		case 0xDC:
			zsortMoveMem(n, h, w0, w1);
			break;
		case 0xD6:
			zsortTransform(h, w1);
			break;
		case 0xD5: {
			float* s = zsortMatrix(h, w0 & 15), * t = zsortMatrix(h, (w1 >> 16) & 15), * d = zsortMatrix(h, w1 & 15);
			if (s && t && d) {
				float m[16];
				multiply(m, s, t);
				memcpy(d, m, sizeof(m));
			}
			break;
		}
		case 0xD1:
			zsortLights(h, w1);
			break;
		case 0xD2:
			zsortLighting(h, w0, w1);
			break;
		case 0xDE:
			if (!((w0 >> 16) & 0xFF) && sp < DL_DEPTH) {
				h->stack[sp++] = next;
			}
			next = address(h, w1);
			break;
		case 0xDF:
			if (!sp) {
				return;
			}
			next = h->stack[--sp];
			break;
		case 0xDB:
			if ((w0 & 0xFF) == 6) {
				h->segments[((w0 >> 8) & 0xFFFF) / 4 & 15] = w1 & 0xFFFFFF;
			} else if ((w0 & 0xFF) == 8) {
				h->fogMul = (int16_t) (w1 >> 16), h->fogOffset = (int16_t) w1;
			}
			break;
		case 0xBB:
			h->texScaleS = (w1 >> 16) / 65536.0f, h->texScaleT = (w1 & 0xFFFF) / 65536.0f;
			h->texLevel = (w0 >> 11) & 7, h->texTile = (w0 >> 8) & 7, h->texOn = w0 & 0xFF;
			break;
		case 0xB6:
		case 0xB7:
			setGeometry(h, op == 0xB6 ? w1 : 0, op == 0xB7 ? w1 : 0, false);
			break;
		case 0xE2:
		case 0xE3:
			otherMode(n, h, op == 0xE3, (w0 >> 8) & 0xFF, w0 & 0xFF, w1);
			break;
		// Interpolation, the transpose, sub-lists and signals (as GLideN64 has them), the RDP halves,
		// G_CULLDL and no-ops: nothing here.
		case 0x00: case 0xB2: case 0xB3: case 0xB4: case 0xBE:
		case 0xD0: case 0xD3: case 0xD4: case 0xD7: case 0xD8: case 0xD9: case 0xDA:
			break;
		default:
			if (op >= 0xE4) {
				next = rdpCommand(n, h, pc, w0, w1);
				break;
			}
			snprintf(h->why, sizeof(h->why), "ZSort op %02X", op);
			reject(h);
			return;
		}
		pc = next;
	}
}

static void runList(struct N64* n, struct Hle* h, uint32_t pc) {
	uint32_t* stack = h->stack;
	int sp = h->sp;
	h->waiting = false;
	bool f3dex2 = h->family == FAMILY_F3DEX2 || h->family == FAMILY_S2DEX2, f3dex = h->family == FAMILY_F3DEX || h->family == FAMILY_S2DEX;
	// F3DEX2's 0xD5: so many commands from an address, then back.
	int counted = -1;
	uint32_t countedReturn = 0;
	for (int guard = 0; guard < 1 << 22; ++guard) {
		if (counted == 0) {
			pc = countedReturn;
		}
		counted -= counted >= 0;
		uint32_t w0 = rd32(n, pc), w1 = rd32(n, pc + 4);
		int op = w0 >> 24;
		uint32_t next = pc + 8;
		bool end = false;
		charge(n, h, 0, 0, 1);
		bool const s2dex = h->family == FAMILY_S2DEX || h->family == FAMILY_S2DEX2, two = h->family == FAMILY_S2DEX2;
		// S2DEX's G_RDPHALF_0, half of a G_SELECT_DL (0xE4, otherwise a texture rectangle).
		if (s2dex && op == 0xE4 && rd32(n, next) >> 24 == (two ? 0x04u : 0xB0u)) {
			h->half0w0 = w0, h->half0w1 = w1;
			pc = next;
			continue;
		}
		if (op >= 0xE4 && !(f3dex2 && op == 0xF1)) {
			pc = rdpCommand(n, h, pc, w0, w1);
			continue;
		}
		// S2DEX's own commands, what it shares with F3DEX or F3DEX2 as they have it.
		if (s2dex) {
			bool handled = true, drawn = true;
			if (op == (two ? 0x0A : 0x02) || op == (two ? 0x09 : 0x01)) {
				bgDraw(n, h, w1, op == (two ? 0x09 : 0x01));
			} else if (op == (two ? 0x0B : 0xB1)) {
				h->objMode = w1;
			} else if (op == (two ? 0x05 : 0xC1)) {
				objLoad(n, h, w1);
			} else if (op == (two ? 0x01 : 0x03) || op == (two ? 0xDA : 0xB2) || op == (two ? 0x02 : 0x04)) {
				drawn = objDraw(n, h, w1, op == (two ? 0x01 : 0x03) ? 0 : op == (two ? 0x02 : 0x04) ? 2 : 1);
			} else if (op == (two ? 0x07 : 0xC3) || op == (two ? 0x08 : 0xC4) || op == (two ? 0x06 : 0xC2)) {
				// A texture load, then the object after it (uObjTxtr is 24 bytes).
				objLoad(n, h, w1);
				drawn = objDraw(n, h, w1 + 24, op == (two ? 0x07 : 0xC3) ? 0 : op == (two ? 0x06 : 0xC2) ? 2 : 1);
			} else if (op == (two ? 0xDC : 0x05)) {
				// The 2D matrix (index 0, in the low half; the size above it) or its offset and base
				// scales alone (2).
				uint32_t const m = address(h, w1);
				if ((w0 & 0xFFFF) == 0) {
					h->objA = (int32_t) rd32(n, m) / 65536.0f, h->objB = (int32_t) rd32(n, m + 4) / 65536.0f;
					h->objC = (int32_t) rd32(n, m + 8) / 65536.0f, h->objD = (int32_t) rd32(n, m + 12) / 65536.0f;
					h->objX = rd16(n, m + 16) / 4.0f, h->objY = rd16(n, m + 18) / 4.0f;
					h->objBaseX = (uint16_t) rd16(n, m + 20), h->objBaseY = (uint16_t) rd16(n, m + 22);
				} else {
					h->objX = rd16(n, m) / 4.0f, h->objY = rd16(n, m + 2) / 4.0f;
					h->objBaseX = (uint16_t) rd16(n, m + 4), h->objBaseY = (uint16_t) rd16(n, m + 6);
				}
				h->objBaseX = h->objBaseX ? h->objBaseX : 1024, h->objBaseY = h->objBaseY ? h->objBaseY : 1024;
			} else if (op == (two ? 0x04 : 0xB0)) {
				// G_SELECT_DL: the list (its halves in both commands) runs when the state word doesn't
				// already hold flag, which it then does.
				uint32_t const sid = (h->half0w0 >> 16) & 12, flag = h->half0w1, mask = w1;
				uint32_t* status = &h->objStatus[sid >> 2];
				if ((*status & mask) != flag) {
					*status = (*status & ~mask) | (flag & mask);
					if (!((w0 >> 16) & 0xFF) && sp < DL_DEPTH) {
						stack[sp++] = next;
					}
					next = address(h, (w0 & 0xFFFF) << 16 | (h->half0w0 & 0xFFFF));
				}
			} else if ((two ? op == 0xDB && ((w0 >> 16) & 0xFF) == 8 : op == 0xBC && (w0 & 0xFF) == 8)) {
				// G_MW_GENSTAT: a state word set.
				h->objStatus[((two ? w0 & 0xFFFF : (w0 >> 8) & 0xFFFF) >> 2) & 3] = w1;
			} else {
				handled = false;
			}
			if (!drawn) {
				snprintf(h->why, sizeof(h->why), "S2DEX%s rotated sprite", two ? "2" : "");
				reject(h);
				return;
			}
			if (handled) {
				pc = next;
				continue;
			}
			// The 3D microcodes' geometry mode and texture commands: S2DEX has nothing they set.
			if (two ? op == 0xD7 || op == 0xD9 : op == 0xB6 || op == 0xB7 || op == 0xBB) {
				pc = next;
				continue;
			}
			bool const shared = two ? op == 0x00 || (op >= 0xDB && op <= 0xDF && op != 0xDC) || op == 0xE1 || op == 0xE2 || op == 0xE3
			                        : op == 0x00 || op == 0x06 || op == 0xAF || op == 0xB4 || op == 0xB8 || op == 0xB9 || op == 0xBA || op == 0xBC || op == 0xC0;
			if (!shared) {
				snprintf(h->why, sizeof(h->why), "S2DEX%s op %02X", two ? "2" : "", op);
				reject(h);
				return;
			}
		}
		// The line microcodes' triangles and lines, and F3DTEX/A's own commands (as GLideN64 has them).
		if (h->lines && (f3dex2 ? op == 0x05 || op == 0x06 || op == 0x08 : op == 0xBF || op == 0xB1 || op == 0xB5)) {
			if (f3dex2 && op == 0x08) {
				drawLine(n, h, (w0 >> 17) & 0x7F, (w0 >> 9) & 0x7F, (int8_t) w0, (w0 >> 17) & 0x7F);
			} else if (!f3dex2 && op == 0xB5) {
				drawLine(n, h, (w1 >> 17) & 0x7F, (w1 >> 9) & 0x7F, (int8_t) w1, (w1 >> 17) & 0x7F);
			} else {
				int const width = f3dex2 ? (int8_t) (w1 >> 24) : 0;
				uint32_t const first = f3dex2 || op == 0xB1 ? w0 : w1;
				drawEdges(n, h, (first >> 17) & 0x7F, (first >> 9) & 0x7F, (first >> 1) & 0x7F, width);
				if (op == (f3dex2 ? 0x06 : 0xB1)) {
					drawEdges(n, h, (w1 >> 17) & 0x7F, (w1 >> 9) & 0x7F, (w1 >> 1) & 0x7F, width);
				}
			}
			pc = next;
			continue;
		}
		if (h->texa && op == 0xB5) {
			emit(n, h, (uint64_t) 0x3D100000 << 32 | address(h, w1));
			emit(n, h, (uint64_t) 0x35100000 << 32 | 0x07000000);
			emit(n, h, (uint64_t) 0x33000000 << 32 | 0x27000000 | (w0 & 0xFFFFFF));
			pc = next;
			continue;
		}
		if (h->texa && op == 0xBE) {
			emit(n, h, (uint64_t) (0x35400000 | ((w1 >> 24) & 0xFF) << 9) << 32 | (w0 & 0xFFFFFF));
			emit(n, h, (uint64_t) 0x32000000 << 32 | (w1 & 0xFFFFFF));
			pc = next;
			continue;
		}
		if (h->cbfd && op >= 0x10 && op <= 0x1F) {
			drawTriangle(n, h, (w0 >> 23) & 31, (w0 >> 18) & 31, ((w0 >> 15) & 7) << 2 | w1 >> 30);
			drawTriangle(n, h, (w0 >> 10) & 31, (w0 >> 5) & 31, w0 & 31);
			drawTriangle(n, h, (w1 >> 25) & 31, (w1 >> 20) & 31, (w1 >> 15) & 31);
			drawTriangle(n, h, (w1 >> 10) & 31, (w1 >> 5) & 31, w1 & 31);
			pc = next;
			continue;
		}
		if (f3dex2) {
			switch (op) {
			case 0x00:
				emit(n, h, 0);
				break;
			case 0x01: {
				int count = (w0 >> 12) & 0xFF, last = (w0 >> 1) & 0x7F;
				loadVertices(n, h, w1, last - count, count);
				break;
			}
			case 0x03:
				cullOrEnd(h, (w0 & 0xFFFF) / 2, (w1 & 0xFFFF) / 2, &end);
				break;
			case 0x04:
				if (branchZ(h, ((w0 >> 1) & 0x7FF) % VERTICES, w1)) {
					next = address(h, h->half1);
				}
				break;
			case 0x05:
				drawTriangle(n, h, ((w0 >> 16) & 0xFF) / 2, ((w0 >> 8) & 0xFF) / 2, (w0 & 0xFF) / 2);
				break;
			case 0x06:
			case 0x07:
				// The microcode sends the second triangle first.
				drawTriangle(n, h, ((w1 >> 16) & 0xFF) / 2, ((w1 >> 8) & 0xFF) / 2, (w1 & 0xFF) / 2);
				drawTriangle(n, h, ((w0 >> 16) & 0xFF) / 2, ((w0 >> 8) & 0xFF) / 2, (w0 & 0xFF) / 2);
				break;
			case 0xD7:
				if (h->am) {
					h->amScale[0] = w1;
					h->texScaleS = h->texScaleT = 1;
				} else {
					h->texScaleS = (w1 >> 16) / 65536.0f, h->texScaleT = (w1 & 0xFFFF) / 65536.0f;
				}
				h->texLevel = (w0 >> 11) & 7, h->texTile = (w0 >> 8) & 7, h->texOn = (w0 >> 1) & 1;
				break;
			case 0xD8:
				popMatrix(h, (int) (w1 / 64));
				break;
			case 0xD9:
				setGeometry(h, ~(w0 & 0xFFFFFF), w1, true);
				break;
			case 0xDA: {
				int p = (w0 & 0xFF) ^ 1;
				matrix(n, h, w1, p & 4, p & 2, p & 1);
				break;
			}
			case 0xDB: {
				int index = (w0 >> 16) & 0xFF, offset = w0 & 0xFFFF;
				if (index == 0) {
					insertMatrix(h, offset, w1);
				} else if (index == 2) {
					int const l = (int) (w1 / (h->cbfd ? 48 : 24));
					h->lights = l > LIGHTS ? LIGHTS : l;
				} else if (index == 0x0A && !h->cbfd && !h->flx && offset % 24 == 0) {
					setLightColor(h, offset / 24, w1);
				} else if (index == 0x10 && h->cbfd) {
					cbfdCoordMod(h, w0, w1);
				} else if (index == 4 && offset == 0x14) {
					h->clipRatio = (float) ((int16_t) w1 > 0 ? (int16_t) w1 : 1);
				} else if (index == 6) {
					h->segments[(offset / 4) & 15] = w1 & 0xFFFFFF;
				} else if (index == 8 && h->am && (offset & 0xFF) == 0x0C) {
					h->amScale[1] = w1;
				} else if (index == 8 && h->am && (offset & 0xFF) == 0x10) {
					h->amScale[2] = w1;
				} else if (index == 8) {
					h->fogMul = (int16_t) (w1 >> 16), h->fogOffset = (int16_t) w1;
				}
				break;
			}
			case 0xDC: {
				int index = w0 & 0xFF, offset = ((w0 >> 8) & 0xFF) * 8;
				uint32_t a = address(h, w1);
				if (index == 8) {
					viewport(n, h, a);
				} else if (index == 14 && h->cbfd) {
					h->cbfdNormals = a;
				} else if (index == 10 && h->cbfd) {
					int const l = offset / 48;
					if (l < 2) {
						readLookat(n, h, a, l);
					} else if (l - 2 <= LIGHTS) {
						readLight(n, h, a, l - 2);
						for (int k = 0; k < 3; ++k) {
							h->lightPos[l - 2][k] = rd16(n, a + 32 + 2 * (uint32_t) k);
						}
						h->lightFalloff[l - 2] = n->rdram[(a + 12) & (RDRAM_SIZE - 1)] / 16.0f;
					}
				} else if (index == 14) {
					// gSPForceMatrix: the combined matrix itself, until the next matrix command.
					readMatrix(n, a, h->mvp);
				} else if (index == 10) {
					int l = offset / 24;
					if (h->flx) {
						// The alpha light (its direction s7.8 at 8), then the lights.
						if (l == 1) {
							for (int k = 0; k < 3; ++k) {
								h->flxDir[k] = rd16(n, a + 8 + 2 * (uint32_t) k) / 256.0f;
							}
							normalize(h->flxDir);
						} else if (l > 1) {
							readLight(n, h, a, l - 1);
						}
					} else if (l < 2) {
						readLookat(n, h, a, l);
					} else {
						readLight(n, h, a, l - 2);
					}
				}
				break;
			}
			case 0xD6:
				h->dmaIo = address(h, w1);
				break;
			case 0xDE:
				if (!((w0 >> 16) & 0xFF) && sp < DL_DEPTH) {
					stack[sp++] = next;
				} else if (address(h, w1) == pc) {
					h->waiting = true, h->waitPc = pc, h->sp = sp;
					return;
				}
				next = address(h, w1);
				break;
			case 0x02:
				modifyVertex(h, (int) (w0 & 0xFFFF) / 2, (w0 >> 16) & 0xFF, w1);
				break;
			case 0xD5:
				if (counted < 0) {
					counted = w0 & 0xFF;
					countedReturn = next;
					next = address(h, w1);
				}
				break;
			case 0xDF:
				end = true;
				break;
			case 0xE1:
				h->half1 = w1;
				break;
			case 0xDD:
				if (!loadMicrocode(n, h, w1, (w0 & 0xFFFF) + 1)) {
					return;
				}
				f3dex2 = h->family == FAMILY_F3DEX2 || h->family == FAMILY_S2DEX2, f3dex = h->family == FAMILY_F3DEX || h->family == FAMILY_S2DEX;
				break;
			case 0xE2:
			case 0xE3:
				// The mask as the microcode makes it: the top bit shifted right by len - 1, with sign,
				// then by 32 - shift - len (shift and len give no mask past bit 0).
				otherModeMask(n, h, op == 0xE3, (uint32_t) ((int32_t) 0x80000000 >> (w0 & 31)) >> ((w0 >> 8) & 31), w1);
				break;
			}
		} else {
			int scale = f3dex ? 2 : h->wrus ? 5 : 10;
			if (h->dkr && (op == 0x01 || op == 0x02 || op == 0x04 || op == 0x05 || op == 0x07 || op == 0xBF ||
			               (op == 0xBC && ((w0 & 0xFF) == 0x02 || (w0 & 0xFF) == 0x0A)))) {
				if (op == 0x01) {
					dkrMatrix(n, h, w0, w1);
				} else if (op == 0x02) {
					h->dkrTexTable = address(h, w1), h->dkrTexShift = h->dkrTexCount = 0;
				} else if (op == 0x04) {
					dkrVertices(n, h, w0, w1);
				} else if (op == 0x05) {
					dkrTriangles(n, h, w0, w1);
				} else if (op == 0x07) {
					// So many commands from an address, then back.
					if (counted < 0) {
						counted = (w0 >> 16) & 0xFF;
						countedReturn = next;
						next = address(h, w1);
					}
				} else if (op == 0xBF) {
					h->dkrMtxOffset = w0 & 0xFFFFFF, h->dkrVtxOffset = w1 & 0xFFFFFF;
				} else if ((w0 & 0xFF) == 0x02) {
					h->dkrBillboard = w1 & 1;
				} else {
					h->depth = (w1 >> 6) & 3;
					updateMvp(h);
				}
				pc = next;
				continue;
			}
			switch (op) {
			case 0xC0:
				emit(n, h, 0);
				break;
			case 0x01: {
				int p = (w0 >> 16) & 0xFF;
				matrix(n, h, w1, p & 1, p & 2, p & 4);
				break;
			}
			case 0x03: {
				int index = (w0 >> 16) & 0xFF;
				uint32_t a = address(h, w1);
				if (index == 0x80) {
					viewport(n, h, a);
				} else if (index == 0x9E) {
					// gSPForceMatrix sends the combined matrix in four parts from one place; the first
					// takes all of it, until the next matrix command.
					readMatrix(n, a, h->mvp);
				} else if (index == 0x82 || index == 0x84) {
					readLookat(n, h, a, index == 0x84 ? 0 : 1);
				} else if (index >= 0x86 && index <= 0x94) {
					readLight(n, h, a, (index - 0x86) / 2);
				}
				break;
			}
			case 0x04:
				if (f3dex) {
					loadVertices(n, h, w1, ((w0 >> 16) & 0xFF) / 2, (w0 >> 10) & 0x3F);
				} else if (h->wrus) {
					loadVertices(n, h, w1, ((w0 >> 16) & 0xFF) / 5, (int) ((w0 & 0xFFFF) + 1) / 0x210);
				} else {
					loadVertices(n, h, w1, (w0 >> 16) & 0xF, ((w0 >> 20) & 0xF) + 1);
				}
				break;
			case 0x09:
				if (h->family == FAMILY_F3D) {
					h->sprite = address(h, w1);
					h->spriteMode = true;
					h->spriteScaleX = h->spriteScaleY = 1;
					h->spriteFlipS = h->spriteFlipT = false;
					// The older 32-byte sprite draws at once: after the uSprite fields its scales
					// (5.10), first S and T, and x and y (10.2).
					if ((w0 & 0xFFFF) == 0x20) {
						uint32_t a = h->sprite, w16 = rd32(n, a + 16), w20 = rd32(n, a + 20), w24 = rd32(n, a + 24), w28 = rd32(n, a + 28);
						h->spriteScaleX = (w16 >> 16) ? (w16 >> 16) / 1024.0f : 1, h->spriteScaleY = (w16 & 0xFFFF) ? (w16 & 0xFFFF) / 1024.0f : 1;
						spriteDraw(n, h, (int16_t) w20, (int16_t) (w24 >> 16), (int16_t) w24, (int16_t) (w28 >> 16), true);
						h->spriteMode = false;
					}
				}
				break;
			case 0x06:
				if (!((w0 >> 16) & 0xFF) && sp < DL_DEPTH) {
					stack[sp++] = next;
				} else if (address(h, w1) == pc) {
					h->waiting = true, h->waitPc = pc, h->sp = sp;
					return;
				}
				next = address(h, w1);
				break;
			case 0xB0:
				if (f3dex && (uint32_t) (h->v[((w0 >> 1) & 0x7FF) % VERTICES].z) <= w1) {
					next = address(h, h->half1);
				}
				break;
			case 0xB1:
				if (f3dex || h->wrus) {
					int const k = f3dex ? 2 : 5;
					drawTriangle(n, h, ((w0 >> 16) & 0xFF) / k, ((w0 >> 8) & 0xFF) / k, (w0 & 0xFF) / k);
					drawTriangle(n, h, ((w1 >> 16) & 0xFF) / k, ((w1 >> 8) & 0xFF) / k, (w1 & 0xFF) / k);
				} else if (h->family == FAMILY_F3DPD) {
					// TRIX: each triangle's first two vertices from w1's nibbles, its third from w0's.
					for (uint32_t a1 = w1, a0 = w0; a1; a1 >>= 8, a0 >>= 4) {
						drawTriangle(n, h, a1 & 15, (a1 >> 4) & 15, a0 & 15);
					}
				} else if (h->family == FAMILY_F3D) {
					// Rare's TRI4 (GoldenEye, as GLideN64 has it): four triangles, each a nibble of w0 and
					// two of w1.
					for (int k = 0; k < 4; ++k) {
						drawTriangle(n, h, (w0 >> (4 * k)) & 15, (w1 >> (8 * k)) & 15, (w1 >> (8 * k + 4)) & 15);
					}
				}
				break;
			case 0x07:
				if (h->family == FAMILY_F3DPD) {
					h->colorBase = address(h, w1);
				}
				break;
			case 0xB4:
				if (!h->wrus) {
					h->half1 = w1;
				}
				break;
			case 0xB3:
				// Fast3D's G_RDPHALF_2 and G_RDPHALF_CONT send both halves to the RDP.
				if (h->wrus) {
					h->half1 = w1;
				} else if (!f3dex && !f3dex2) {
					emit(n, h, (uint64_t) h->half1 << 32 | w1);
				}
				break;
			case 0xB2:
				// F3DEX's G_MODIFYVTX.
				if (f3dex) {
					modifyVertex(h, (int) (w0 & 0xFFFF) / 2, (w0 >> 16) & 0xFF, w1);
				} else if (!f3dex2 && !h->wrus) {
					emit(n, h, (uint64_t) h->half1 << 32 | w1);
				}
				break;
			case 0xAF:
				if (!loadMicrocode(n, h, w1, (w0 & 0xFFFF) + 1)) {
					return;
				}
				f3dex2 = h->family == FAMILY_F3DEX2 || h->family == FAMILY_S2DEX2, f3dex = h->family == FAMILY_F3DEX || h->family == FAMILY_S2DEX;
				break;
			case 0xB5:
				// F3DEX's G_QUAD, and Wave Race 64's.
				if (f3dex || h->wrus) {
					int v0 = ((w1 >> 24) & 0xFF) / scale, v1 = ((w1 >> 16) & 0xFF) / scale, v2 = ((w1 >> 8) & 0xFF) / scale, v3 = (w1 & 0xFF) / scale;
					drawTriangle(n, h, v0, v1, v2);
					drawTriangle(n, h, v0, v2, v3);
				}
				break;
			case 0xB6:
				setGeometry(h, w1, 0, false);
				break;
			case 0xB7:
				setGeometry(h, 0, w1, false);
				break;
			case 0xB8:
				end = true;
				break;
			case 0xB9:
			case 0xBA:
				otherMode(n, h, op == 0xBA, (w0 >> 8) & 0xFF, w0 & 0xFF, w1);
				break;
			case 0xBB:
				h->texScaleS = (w1 >> 16) / 65536.0f, h->texScaleT = (w1 & 0xFFFF) / 65536.0f;
				h->texLevel = (w0 >> 11) & 7, h->texTile = (w0 >> 8) & 7, h->texOn = w0 & 0xFF;
				break;
			case 0xBC: {
				int index = w0 & 0xFF, offset = (w0 >> 8) & 0xFFFF;
				if (index == 0) {
					insertMatrix(h, offset, w1);
				} else if (index == 2) {
					int l = (int) (((w1 - 0x80000000u) >> 5) - 1);
					h->lights = l < 0 ? 0 : l > LIGHTS ? LIGHTS : l;
				} else if (index == 4 && offset == 0x14) {
					h->clipRatio = (float) ((int16_t) w1 > 0 ? (int16_t) w1 : 1);
				} else if (index == 6) {
					h->segments[(offset / 4) & 15] = w1 & 0xFFFFFF;
				} else if (index == 8) {
					h->fogMul = (int16_t) (w1 >> 16), h->fogOffset = (int16_t) w1;
				} else if (index == 0x0A && offset % 32 == 0) {
					setLightColor(h, offset / 32, w1);
				} else if (index == 0x0C) {
					modifyVertex(h, offset / 40, offset % 40, w1);
				}
				break;
			}
			case 0xBD:
				if (h->spriteMode) {
					uint32_t w16 = rd32(n, h->sprite + 16);
					spriteDraw(n, h, (int16_t) (w16 >> 16), (int16_t) w16, (int16_t) (w1 >> 16), (int16_t) w1, false);
					h->spriteMode = false;
				} else {
					popMatrix(h, 1);
				}
				break;
			case 0xBE:
				if (h->spriteMode) {
					h->spriteScaleX = (w1 >> 16) ? (w1 >> 16) / 1024.0f : 1, h->spriteScaleY = (w1 & 0xFFFF) ? (w1 & 0xFFFF) / 1024.0f : 1;
					h->spriteFlipS = (w0 >> 8) & 0xFF, h->spriteFlipT = w0 & 0xFF;
				} else {
					cullOrEnd(h, (w0 & 0xFFFF) / (f3dex ? 2 : scale * 4), (w1 & 0xFFFF) / (f3dex ? 2 : scale * 4), &end);
				}
				break;
			case 0xBF:
				// Fast3D's indices are vertex numbers times 10: other ones are another microcode's. A few
				// (a hack's slip, read from the middle of a vertex) are left out.
				if (!f3dex && (((w1 >> 16) & 0xFF) % scale || ((w1 >> 8) & 0xFF) % scale || (w1 & 0xFF) % scale)) {
					if (++h->oddTriangles <= 4) {
						break;
					}
					snprintf(h->why, sizeof(h->why), "Fast3D-like %08x", h->ucode);
					reject(h);
					return;
				}
				drawTriangle(n, h, ((w1 >> 16) & 0xFF) / scale, ((w1 >> 8) & 0xFF) / scale, (w1 & 0xFF) / scale);
				break;
			}
		}
		if (end) {
			if (sp == 0) {
				return;
			}
			next = stack[--sp];
		}
		pc = next;
	}
}

// The name in a microcode's ID string: "RSP Gfx ucode NAME", or a tag in place of "ucode" (Ocarina
// of Time's "RSP Gfx[Safe] S2DEX").
static const char* gfxName(const char* text) {
	const char* id = strstr(text, "RSP Gfx");
	const char* name = id && (id[7] == ' ' || id[7] == '[') ? strchr(id + 8, ' ') : NULL;
	return name ? name + 1 : NULL;
}

// The name a microcode's data gives, where a patch blanked it (a fan translation, a ROM hack), known
// by the hash of its first code.
static const char* knownName(struct N64* n, uint32_t code) {
	uint32_t hash = 2166136261u;
	for (uint32_t i = 0; i < 1024; ++i) {
		hash = (hash ^ n->rdram[(code + i) & (RDRAM_SIZE - 1)]) * 16777619u;
	}
	static const struct {
		uint32_t hash;
		const char* name;
	} known[] = {
		{ 0x732CC90Fu, "RSP Gfx ucode F3DZEX.NoN  fifo 2.08J Yoshitaka Yasumoto/Kawasedo 1999." },
		{ 0xCDD276A8u, "RSP SW Version: 2.0G, 09-30-96" },
		{ 0xF5216367u, "RSP Gfx ucode F3DEX       fifo 2.08  Yoshitaka Yasumoto 1999 Nintendo." },
	};
	for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); ++i) {
		if (known[i].hash == hash) {
			return known[i].name;
		}
	}
	return NULL;
}

// The microcode family from the first ID string in its data (the block may run on into the next
// microcode's data, and its string).
// A microcode by the name its data gives (a Fast3D or F3DEX family line), else by the hash of its
// first code.
static void ucodeName(struct N64* n, uint32_t code, uint32_t data, uint32_t dataSize, char* out, size_t size) {
	char text[0x1001];
	uint32_t length = dataSize < 0x1000 ? dataSize : 0x1000;
	for (uint32_t i = 0; i < length; ++i) {
		uint8_t c = n->rdram[(data + i) & (RDRAM_SIZE - 1)];
		text[i] = c >= 32 && c < 127 ? (char) c : ' ';
	}
	text[length] = 0;
	const char* name = gfxName(text) ? strstr(text, "RSP Gfx") : strstr(text, "RSP SW Version");
	if (!name) {
		name = knownName(n, code);
	}
	if (name) {
		snprintf(out, size, "%.*s", (int) (size - 1), name);
		return;
	}
	uint32_t hash = 2166136261u;
	for (uint32_t i = 0; i < 1024; ++i) {
		hash = (hash ^ n->rdram[(code + i) & (RDRAM_SIZE - 1)]) * 16777619u;
	}
	snprintf(out, size, "unnamed %08x", hash);
}

// The graphics task's microcode, and why HLE left it to the RSP when it did.
void hleUcodeName(struct N64* n, char* out, size_t size) {
	const uint8_t* task = &n->sp.mem[0xFC0];
	if (n->hle && n->hle->rejectedNow && n->hle->why[0]) {
		snprintf(out, size, "%s", n->hle->why);
		return;
	}
	ucodeName(n, be32(task + 0x10) & 0xFFFFFF, be32(task + 0x18) & 0xFFFFFF, be32(task + 0x1C), out, size);
}

// The line microcodes, F3DTEX/A and F3DAM, by name.
static void ucodeFlags(struct N64* n, struct Hle* h, uint32_t code, uint32_t data, uint32_t size) {
	char text[48];
	ucodeName(n, code, data, size, text, sizeof(text));
	const char* name = gfxName(text);
	h->lines = name && !strncmp(name, "L3DEX", 5);
	h->texa = name && !strncmp(name, "F3DTEX/A", 8);
	h->am = name && !strncmp(name, "F3DAM", 5);
	h->noNear = name && strstr(name, ".NoN");
	h->cbfd = name && !strncmp(name, "F3DEXBG", 7);
}

static bool isFlx(struct N64* n, uint32_t code, uint32_t data, uint32_t size) {
	char text[48];
	ucodeName(n, code, data, size, text, sizeof(text));
	const char* name = gfxName(text);
	return name && !strncmp(name, "F3DFLX", 6);
}

static int family(struct N64* n, uint32_t code, uint32_t data, uint32_t size) {
	if (size > 0x1000) {
		size = 0x1000;
	}
	char text[0x1001];
	for (uint32_t i = 0; i < size; ++i) {
		uint8_t c = n->rdram[(data + i) & (RDRAM_SIZE - 1)];
		text[i] = c >= 32 && c < 127 ? (char) c : ' ';
	}
	text[size] = 0;
	if (!strstr(text, "RSP Gfx") && !strstr(text, "RSP SW Version") && knownName(n, code)) {
		snprintf(text, sizeof(text), "%s", knownName(n, code));
	}
	const char* name = gfxName(text);
	const char* sw = strstr(text, "RSP SW Version");
	if (sw && (!name || sw < name)) {
		return FAMILY_F3D;
	}
	if (!name) {
		return FAMILY_NONE;
	}
	if (!strncmp(name, "L3DEX", 5) || !strncmp(name, "F3DTEX/A", 8)) {
		for (const char* c = name; *c && c < name + 48; ++c) {
			if (c[0] == ' ' && c[1] >= '0' && c[1] <= '9' && c[2] == '.' && c[3] >= '0' && c[3] <= '9') {
				return c[1] >= '2' ? FAMILY_F3DEX2 : FAMILY_F3DEX;
			}
		}
		return FAMILY_F3DEX;
	}
	if (!strncmp(name, "ZSortp", 6)) {
		return FAMILY_ZSORT;
	}
	if (!strncmp(name, "S2DEX", 5) && (name[5] == ' ' || name[5] == '.' || name[5] == '_')) {
		for (const char* c = name; *c && c < name + 48; ++c) {
			if (c[0] == ' ' && c[1] >= '0' && c[1] <= '9' && c[2] == '.' && c[3] >= '0' && c[3] <= '9') {
				return c[1] >= '2' ? FAMILY_S2DEX2 : FAMILY_S2DEX;
			}
		}
		return FAMILY_S2DEX;
	}
	if (!strncmp(name, "F3DEXBG", 7)) {
		return FAMILY_F3DEX2;
	}
	if ((!strncmp(name, "F3DEX2", 6) || !strncmp(name, "F3DZEX", 6) || !strncmp(name, "F3DLX2", 6) || !strncmp(name, "F3DLP2", 6) ||
	     !strncmp(name, "F3DFLX", 6) || !strncmp(name, "F3DPEX", 6)) &&
	    (name[6] == ' ' || name[6] == '.' || name[6] == '_')) {
		return FAMILY_F3DEX2;
	}
	if (!strncmp(name, "F3DAM", 5)) {
		return FAMILY_F3DEX2;
	}
	if (strncmp(name, "F3DEX", 5) && strncmp(name, "F3DLX", 5) && strncmp(name, "F3DLP", 5)) {
		return FAMILY_NONE;
	}
	// Variants with their own names (F3DEX3, F3DEXBG...) are other microcodes.
	if (name[5] != ' ' && name[5] != '.' && name[5] != '_') {
		return FAMILY_NONE;
	}
	// F3DEX2's builds also call themselves F3DEX (F3DLX, F3DLP), with versions from 2.0; F3DEX's
	// are 0.x and 1.x.
	for (const char* c = name; *c && c < name + 48; ++c) {
		if (c[0] == ' ' && c[1] >= '0' && c[1] <= '9' && c[2] == '.' && c[3] >= '0' && c[3] <= '9') {
			return c[1] >= '2' ? FAMILY_F3DEX2 : FAMILY_F3DEX;
		}
	}
	return FAMILY_F3DEX;
}

// Runs the graphics task in DMEM at high level when its microcode is one of the known families;
// A task's list run (or run until it waits): the RSP's time for it, roughly. The RDP draws while
// the RSP makes its commands, so it can't be done before the RSP is; waiting, it looks again a
// little later.
// gb64's Game Boy picture processor (its rsp/ppu.s, MIT): the CPU signals each line's mode 2 (SIG1:
// the sprites) and mode 3 (SIG0: the line), or SIG2 to stop; each line goes to the output as 160
// palette indices (the background's 0-31, the sprites' 32-63).
enum { GB_COLOR = 1, GB_BG = 0x01, GB_OBJ = 0x02, GB_OBJ16 = 0x04, GB_BG_MAP = 0x08, GB_TILE_DATA = 0x10, GB_WINDOW = 0x20, GB_WINDOW_MAP = 0x40 };

static uint8_t gbLcdc(struct Hle* h) {
	return h->gbTask[14];
}

static bool gbColor(struct Hle* h) {
	return h->gbTask[13] & GB_COLOR;
}

static void gbRead(struct N64* n, uint8_t* to, uint32_t from, uint32_t length) {
	for (uint32_t i = 0; i < length; ++i) {
		to[i] = n->rdram[((from & 0xFFFFF8) + i) & (RDRAM_SIZE - 1)];
	}
}

// The next tile of the cache holds the tile at this offset (read again only when it held another).
static void gbRequestTile(struct N64* n, struct Hle* h, uint32_t tile) {
	int slot = h->gbTile++ & 31;
	if (h->gbTileIds[slot] != (uint16_t) tile) {
		h->gbTileIds[slot] = (uint16_t) tile;
		gbRead(n, h->gbTiles[slot], be32(&h->gbTask[8]) + tile, 16);
	}
}

// A tilemap row (and its attributes on the Game Boy Color).
static void gbPrecacheLine(struct N64* n, struct Hle* h, uint8_t* row, uint32_t y, uint8_t map) {
	uint32_t a = be32(&h->gbTask[8]) + 0x1800 + ((gbLcdc(h) & map) ? 0x400 : 0) + (y >> 3) * 32;
	gbRead(n, row, a, 32);
	if (gbColor(h)) {
		gbRead(n, row + 32, a + 0x2000, 32);
	}
}

static void gbPrecacheTiles(struct N64* n, struct Hle* h, const uint8_t* row, uint32_t x, uint32_t count) {
	for (; count; --count, x = (x + 1) & 31) {
		uint32_t at = (gbLcdc(h) & GB_TILE_DATA) ? 0 : 0x80;
		uint32_t tile = (((row[x] + at) & 0xFF) + at) << 4;
		uint8_t attr = row[32 + x];
		h->gbAttrs[h->gbAttr++ % 24] = attr;
		gbRequestTile(n, h, tile + ((attr & 8) << 10));
	}
}

// Bit x of a tile row's two planes (the first byte's the colour's low bit), flipped or not.
static int gbPixel(const uint8_t* row, int x, bool flip) {
	int b = flip ? x : 7 - x;
	return (row[0] >> b & 1) | (row[1] >> b & 1) << 1;
}

// The cached tiles drawn from pixel x on (the first one shifted left by shift), over the sprites.
static void gbTileLine(struct Hle* h, int x, int count, int y, int shift) {
	count += shift;
	int at = 8 + x - shift;
	uint8_t saved[8];
	memcpy(saved, &h->gbLine[at + count], 8);
	bool spritesFirst = gbColor(h) && !(gbLcdc(h) & GB_BG);
	for (; count > 0; count -= 8, at += 8) {
		uint8_t attr = h->gbAttrs[h->gbAttr++ % 24];
		const uint8_t* row = h->gbTiles[h->gbTile++ & 31] + ((attr & 0x40) ? 14 - 2 * y : 2 * y);
		for (int i = 0; i < 8; ++i) {
			int color = gbPixel(row, i, attr & 0x20);
			uint8_t bg = (uint8_t) (color | (attr & 7) << 2), was = h->gbLine[at + i], obj = was & 0x7F;
			uint8_t top = obj ? obj : bg;
			h->gbLine[at + i] = spritesFirst || !color || !((was & 0x80) | (attr & 0x80)) ? top : bg;
		}
	}
	memcpy(&h->gbLine[at + count], saved, 8);
}

static void gbSprites(struct Hle* h) {
	uint8_t ly = h->gbTask[15];
	for (const uint8_t* s = h->gbSprites; s < h->gbSprites + 160 && s[0]; s += 4) {
		int y = (ly - (s[0] - 16)) & 7;
		y = (s[3] & 0x40) ? 7 - y : y;
		uint8_t add = (uint8_t) ((gbColor(h) ? (s[3] & 7) << 2 : (s[3] & 0x10) >> 2) | (s[3] & 0x80) | 0x20);
		const uint8_t* row = h->gbTiles[h->gbTile++ & 31] + 2 * y;
		uint8_t* to = &h->gbLine[s[1]];
		for (int i = 0; i < 8; ++i) {
			int color = gbPixel(row, i, s[3] & 0x20);
			if (color && !to[i]) {
				to[i] = (uint8_t) (color | add);
			}
		}
	}
}

// Mode 2: the sprites read and the next line's (at most 10) kept, in x order but on the Game Boy Color.
static void gbMode2(struct N64* n, struct Hle* h) {
	gbRead(n, h->gbSprites, be32(&h->gbTask[4]) + 0xD0, 160);
	int height = (gbLcdc(h) & GB_OBJ16) ? 16 : 8, kept = 0;
	uint8_t* s = h->gbSprites;
	for (int i = 0; i < 40 && kept < 10; ++i) {
		int d = (int) h->gbLines - (s[4 * i] - 16);
		if (s[4 * i + 1] >= 168 || d < 0 || d >= height) {
			continue;
		}
		uint8_t sprite[4];
		memcpy(sprite, &s[4 * i], 4);
		int at = kept;
		if (!gbColor(h)) {
			for (at = 0; at < kept && sprite[1] >= s[4 * at + 1]; ++at) {
			}
			memmove(&s[4 * at + 4], &s[4 * at], (size_t) (4 * (kept - at)));
		}
		memcpy(&s[4 * at], sprite, 4);
		++kept;
	}
	memset(&s[4 * kept], 0, 4);
	++h->gbLines;
}

// Mode 3, after the task's data was read at its start: the line drawn and written out; true when
// it was the last.
static bool gbMode3(struct N64* n, struct Hle* h) {
	uint8_t lcdc = gbLcdc(h), ly = h->gbTask[15], scy = h->gbTask[16], scx = h->gbTask[17], wy = h->gbTask[18], wx = h->gbTask[19];
	gbPrecacheLine(n, h, h->gbMap, (uint8_t) (ly + scy), GB_BG_MAP);
	h->gbTile = h->gbAttr = 0;
	if (!(lcdc & GB_OBJ)) {
		memset(h->gbSprites, 0, 4);
	}
	for (const uint8_t* s = h->gbSprites; be32(s); s += 4) {
		uint32_t tile = s[2];
		if (lcdc & GB_OBJ16) {
			tile = (tile & 0xFE) ^ (((ly - (s[0] - 16)) & 8) >> 3) ^ ((s[3] & 0x40) >> 6);
		}
		gbRequestTile(n, h, (tile << 4) + (gbColor(h) && (s[3] & 8) ? 0x2000 : 0));
	}
	int window = 160;
	uint32_t windowY = h->gbWindowY;
	if ((lcdc & GB_WINDOW) && wy <= ly && wx - 7 < 160) {
		window = wx - 7;
		++h->gbWindowY;
		gbPrecacheLine(n, h, h->gbWindow, windowY, GB_WINDOW_MAP);
	}
	if (window > 0) {
		uint32_t count = ((scx + (uint32_t) window - 1) >> 3) - (scx >> 3) + 1;
		if ((lcdc & GB_BG) || gbColor(h)) {
			gbPrecacheTiles(n, h, h->gbMap, scx >> 3, count);
		} else {
			for (; count; --count) {
				int slot = h->gbTile++ & 31;
				memset(h->gbTiles[slot], 0, 16);
				h->gbTileIds[slot] = 0xFFFF;
			}
		}
	}
	if (window < 160) {
		gbPrecacheTiles(n, h, h->gbWindow, 0, ((159 - (uint32_t) window) >> 3) + 1);
	}
	n->sp.status &= ~(uint32_t) (SIG0 | SIG1);
	h->gbTile = h->gbAttr = 0;
	gbSprites(h);
	if (window > 0) {
		gbTileLine(h, 0, window, (ly + scy) & 7, scx & 7);
	}
	if (window < 160) {
		gbTileLine(h, window, 160 - window, windowY & 7, 0);
	}
	uint32_t to = (be32(h->gbTask) + ly * 160u) & 0xFFFFF8;
	n64RdramWrite(n, to, 160);
	for (int i = 0; i < 160; ++i) {
		n->rdram[(to + (uint32_t) i) & (RDRAM_SIZE - 1)] = h->gbLine[8 + i];
	}
	return ly == 143;
}

// The RSP's time for mode 2, and for mode 3 up to its clearing the signals (measured, CPU cycles):
// the CPU's handshake counts on it (its next mode 3 signal often comes while the RSP is busy).
enum { GB_MODE2_TIME = 1400, GB_MODE3_TIME = 1450 };
enum { GB_WAIT_MODE2, GB_MODE2, GB_WAIT_MODE3, GB_MODE3 };

// What the signals ask for, done as the RSP would by now; the task ends at the last line or SIG2
// (its wait counts written to the task's performance words, here each wait counted once).
static void gbRun(struct N64* n, struct Hle* h) {
	while (n->cycles >= h->gbReadyAt) {
		if ((h->gbState == GB_WAIT_MODE2 || h->gbState == GB_WAIT_MODE3) && (n->sp.status & SIG2)) {
			break;
		}
		if (h->gbState == GB_WAIT_MODE2 && (n->sp.status & SIG1)) {
			gbMode2(n, h);
			h->gbState = GB_MODE2, h->gbReadyAt = n->cycles + GB_MODE2_TIME;
		} else if (h->gbState == GB_MODE2) {
			h->gbState = GB_WAIT_MODE3;
		} else if (h->gbState == GB_WAIT_MODE3 && (n->sp.status & SIG0)) {
			gbRead(n, h->gbTask, h->gbData, 24);
			h->gbState = GB_MODE3, h->gbReadyAt = n->cycles + GB_MODE3_TIME;
		} else if (h->gbState == GB_MODE3) {
			if (gbMode3(n, h)) {
				break;
			}
			memset(&h->gbLine[8], 0, 160);
			h->gbState = GB_WAIT_MODE2;
		} else {
			n->sp.hleDoneAt = n->cycles + (1 << 20);
			return;
		}
	}
	if (n->cycles < h->gbReadyAt) {
		n->sp.hleDoneAt = h->gbReadyAt;
		return;
	}
	uint32_t perf = be32(&h->gbTask[20]) & 0xFFFFF8;
	n64RdramWrite(n, perf, 8);
	for (int i = 0; i < 8; ++i) {
		n->rdram[(perf + (uint32_t) i) & (RDRAM_SIZE - 1)] = (uint8_t) (h->gbLines >> (8 * (3 - (i & 3))));
	}
	h->waiting = false;
	n->sp.hleDoneAt = n->cycles + 2048;
}

static void gbStart(struct N64* n, struct Hle* h) {
	h->gbData = be32(&n->sp.mem[0xFF0]) & 0xFFFFFF;
	gbRead(n, h->gbTask, h->gbData, 24);
	memset(h->gbTileIds, 0xFF, sizeof(h->gbTileIds));
	memset(h->gbAttrs, 0, sizeof(h->gbAttrs));
	memset(h->gbMap, 0, sizeof(h->gbMap));
	memset(h->gbWindow, 0, sizeof(h->gbWindow));
	memset(h->gbLine, 0, sizeof(h->gbLine));
	h->gbLines = h->gbWindowY = 0;
	h->gbState = GB_WAIT_MODE2, h->gbReadyAt = 0;
	h->waiting = true;
	gbRun(n, h);
}

static void finishTask(struct N64* n, struct Hle* h) {
	if (!h->dry && h->waiting && h->family == FAMILY_ZSORTBOSS) {
		// Waiting on a signal: the CPU's write of it brings the task back (hleSignalled).
		n->sp.hleDoneAt = n->cycles + (1 << 20);
	} else if (!h->dry && h->waiting) {
		n->sp.hleDoneAt = n->cycles + 2048;
	} else if (!h->dry) {
		n->sp.hleDoneAt = n->cycles + 4096 + (costed(n, h) ? 1 : 64) * (uint64_t) h->timeCount;
	}
	bool const waiting = h->waiting;
	flush(n, h, !h->dry);
	// A task that waits or stalls keeps its time for when it ends.
	h->timeCount = !costed(n, h) ? h->count : waiting || h->stalled ? h->timeCount : 0;
	if (h->stalled) {
		n->sp.hleDoneAt = n->cycles + 2048;
	} else if (!h->dry && !h->waiting) {
		n->dp.busyUntil = n->dp.busyUntil > n->sp.hleDoneAt ? n->dp.busyUntil : n->sp.hleDoneAt;
	}
}

void hleSignalled(struct N64* n) {
	// The picture processor, when not busy, answers at once.
	if (n->hle && n->hle->family == FAMILY_GBPPU && n->hle->waiting && n->sp.hleDoneAt && n->cycles >= n->hle->gbReadyAt) {
		gbRun(n, n->hle);
		n64Event(n, n->sp.hleDoneAt);
		return;
	}
	if (n->hle && n->hle->family == FAMILY_ZSORTBOSS && n->hle->waiting && n->sp.hleDoneAt > n->cycles) {
		n->sp.hleDoneAt = n->cycles;
		n64Event(n, n->cycles);
	}
}

bool hleWaiting(struct N64* n) {
	return n->hle && (n->hle->waiting || n->hle->stalled);
}

// A waiting task looks at the command it waits on again, and goes on when the CPU changed it.
void hleResume(struct N64* n) {
	struct Hle* h = n->hle;
	if (h->stalled) {
		finishTask(n, h);
		return;
	}
	if (h->family == FAMILY_GBPPU) {
		gbRun(n, h);
		return;
	}
	if (h->family == FAMILY_ZSORTBOSS) {
		bossRun(n, h);
		finishTask(n, h);
		return;
	}
	if (rd32(n, h->waitPc) >> 24 == (h->family == FAMILY_F3DEX2 ? 0xDE : 0x06) && address(h, rd32(n, h->waitPc + 4)) == h->waitPc) {
		n->sp.hleDoneAt = n->cycles + 2048;
		return;
	}
	runList(n, h, h->waitPc);
	finishTask(n, h);
}

// false leaves it to the RSP. Dry, it only counts the RDP commands it would send.
bool hleGraphics(struct N64* n, bool dry) {
	const uint8_t* task = &n->sp.mem[0xFC0];
	if (be32(task) != 1) {
		return false;
	}
	// A microcode rejected before runs on the RSP, its yielded tasks too.
	uint32_t code = be32(task + 0x10) & 0xFFFFFF, ucode = 2166136261u;
	for (uint32_t i = 0; i < 1024; ++i) {
		ucode = (ucode ^ n->rdram[(code + i) & (RDRAM_SIZE - 1)]) * 16777619u;
	}
	for (int i = 0; n->hle && i < n->hle->rejectedCount; ++i) {
		if (n->hle->rejected[i] == ucode) {
			// Rejected before: the report keeps why.
			n->hle->rejectedNow = true;
			return false;
		}
	}
	// A yielded task started again: one run here drew all of it, so it is done (its data is the
	// yield buffer, which the microcode would restore itself from).
	uint32_t list = be32(task + 0x30) & 0xFFFFFF;
	if (be32(task + 4) & 1) {
		if (n->hle && n->hle->doneList == list && !dry) {
			n->sp.hleDoneAt = n->cycles + 4096;
			return true;
		}
		return false;
	}
	uint32_t data = be32(task + 0x18) & 0xFFFFFF, dataSize = be32(task + 0x1C);
	int f = family(n, code, data, dataSize);
	// Perfect Dark's microcode has no name: it is known by its code.
	if (f == FAMILY_NONE && ucode == 0xB0811E2Fu) {
		f = FAMILY_F3DPD;
	}
	// Eikou no Saint Andrews' Fast3D has no name either (GLideN64 knows it by its code too), nor do
	// Saikyou Habu Shougi's and Tetris 64's, or Pachinko 365 Nichi's and Morita Shougi 64's F3DEX (each
	// matches the microcode's pictures as that).
	if (f == FAMILY_NONE && (ucode == 0xE272C977u || ucode == 0x6F15E8F7u || ucode == 0x112F868Fu)) {
		f = FAMILY_F3D;
	}
	if (f == FAMILY_NONE && ucode == 0x67E5A46Fu) {
		f = FAMILY_F3DEX;
	}
	// Power League 64's is F3DEX under Fast3D's name (as GLideN64 has it).
	if (f == FAMILY_F3D && ucode == 0x7754FFF8u) {
		f = FAMILY_F3DEX;
	}
	// Dark Rift's Turbo3D has no name either.
	if (f == FAMILY_NONE && ucode == 0xA9DAF288u) {
		f = FAMILY_TURBO3D;
	}
	// The Toukon Road games' T3DUX (0.83 and 0.85 start alike).
	if (f == FAMILY_NONE && ucode == 0xEE50B3B1u) {
		f = FAMILY_T3DUX;
	}
	if (f == FAMILY_NONE && ucode == 0x57C8BE85u) {
		f = FAMILY_ZSORTBOSS;
	}
	// Rogue Squadron's (and its Japanese release's).
	if (f == FAMILY_NONE && ucode == 0xDB79F253u) {
		f = FAMILY_F5ROGUE;
	}
	// Battle for Naboo's and Indiana Jones's.
	if (f == FAMILY_NONE && (ucode == 0x5804274Fu || ucode == 0x3CCF37FFu)) {
		f = FAMILY_F5INDI;
	}
	// gb64's Game Boy picture processor, all in its boot code.
	if (f == FAMILY_NONE && !code) {
		uint32_t boot = be32(task + 8) & 0xFFFFFF, bootHash = 2166136261u;
		for (uint32_t i = 0; i < be32(task + 12) && i < 0x1000; ++i) {
			bootHash = (bootHash ^ n->rdram[(boot + i) & (RDRAM_SIZE - 1)]) * 16777619u;
		}
		f = bootHash == 0x8856B8EFu ? FAMILY_GBPPU : f;
	}
	bool const wrus = f == FAMILY_F3D && (ucode == 0xAC1B2CFEu || ucode == 0x106A093Eu);
	bool const dkr = f == FAMILY_NONE && (ucode == 0x5277D6E7u || ucode == 0x26EDE132u);
	if (dkr) {
		f = FAMILY_F3D;
	}
	if (f == FAMILY_NONE) {
		return false;
	}
	if (!n->hle) {
		n->hle = calloc(1, sizeof(struct Hle));
		if (!n->hle) {
			return false;
		}
	}
	struct Hle* h = n->hle;
	h->ucode = ucode;
	h->family = f;
	if (f == FAMILY_GBPPU) {
		if (dry) {
			return false;
		}
		gbStart(n, h);
		return true;
	}
	h->flx = isFlx(n, code, data, dataSize);
	h->wrus = wrus;
	ucodeFlags(n, h, code, data, dataSize);
	h->cbfdAdvanced = false;
	h->dkr = dkr, h->jfg = ucode == 0x26EDE132u;
	h->dkrBillboard = false, h->dkrVertex = 0, h->dkrMtxOffset = h->dkrVtxOffset = h->dkrTexTable = 0;
	h->dry = dry;
	memset(h->segments, 0, sizeof(h->segments));
	memset(h->objStatus, 0, sizeof(h->objStatus));
	h->clipRatio = 2;
	static const float identity[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
	h->depth = 0;
	memcpy(h->modelview[0], identity, sizeof(identity));
	memcpy(h->projection, identity, sizeof(identity));
	updateMvp(h);
	h->geometry = 0;
	h->texOn = false;
	h->texScaleS = h->texScaleT = 1;
	h->lights = 1;
	// The lookat directions the microcodes' data starts with (texture generation without a lookat).
	static const float lookat[2][3] = { { 1, 0, 0 }, { 0, 1, 0 } };
	memcpy(h->lookat, lookat, sizeof(lookat));
	if (f == FAMILY_F3D || f == FAMILY_F3DEX) {
		defaultLights(n, h, data, dataSize);
	}
	// So do the other modes (the data's set-other-modes command).
	h->modeH = 0x080CFF, h->modeL = 0;
	h->spriteMode = false;
	h->count = 0;
	uint32_t fifoStart = be32(task + 0x28) & 0xFFFFFF, fifoEnd = be32(task + 0x2C) & 0xFFFFFF;
	bool fifo = fifoStart && !(fifoStart & 7) && fifoEnd > fifoStart && fifoEnd <= RDRAM_SIZE;
	h->fifoStart = fifo ? fifoStart : 0;
	h->fifoEnd = fifo ? fifoEnd & ~7u : 0;
	h->sp = 0;
	h->rejectedNow = false;
	h->oddTriangles = 0;
	h->why[0] = 0;
	h->objBaseX = h->objBaseY = 1024, h->objA = h->objD = 1, h->objB = h->objC = 0, h->objX = h->objY = 0;
	if (f == FAMILY_ZSORT) {
		for (uint32_t i = 0; i < sizeof(h->zDmem); ++i) {
			h->zDmem[i] = i < dataSize ? n->rdram[(data + i) & (RDRAM_SIZE - 1)] : 0;
		}
		h->zMem = h->zDmem;
		h->zRdp[0] = h->zRdp[1] = h->zRdp[2] = 0;
		h->zScale[0] = h->zScale[1] = h->zTrans[0] = h->zTrans[1] = 0;
		zsortList(n, h, address(h, be32(task + 0x30)));
	} else if (f == FAMILY_ZSORTBOSS) {
		// The microcode's data in DMEM (the SP's own unless dry: the CPU reads the transformed vertices
		// there), below the task.
		h->zMem = dry ? h->zDmem : n->sp.mem;
		for (uint32_t i = 0; i < (dry ? sizeof(h->zDmem) : 0xFC0); ++i) {
			if (i < dataSize || dry) {
				h->zMem[i] = i < dataSize ? n->rdram[(data + i) & (RDRAM_SIZE - 1)] : 0;
			}
		}
		h->zRdp[0] = h->zRdp[1] = h->zRdp[2] = 0;
		if (!h->bossInit) {
			h->bossInit = true, h->bossInvw = 10, h->bossMask[0] = h->bossMask[1] = 0;
			memset(h->bossFog, 0, sizeof(h->bossFog)), memset(h->bossTable, 0, sizeof(h->bossTable));
		}
		// The sub list starts at the task's yield data.
		h->bossPc[0] = be32(task + 0x30) & 0xFFFFFF, h->bossPc[1] = be32(task + 0x38) & 0xFFFFFF;
		h->bossList = 0, h->bossEnded[0] = h->bossEnded[1] = h->bossSignal = h->bossDone = false;
		if (!dry) {
			n->sp.status = (n->sp.status & ~(uint32_t) (SIG1 | SIG2)) | SIG4;
		}
		bossRun(n, h);
	} else if (f == FAMILY_F5INDI) {
		// The microcode's data in DMEM, as the RSP loads it each task.
		for (uint32_t i = 0; i < dataSize && i < 0xFC0; ++i) {
			h->f5iDmem[i] = n->rdram[(data + (i ^ 3)) & (RDRAM_SIZE - 1)];
		}
		f5iList(n, h, be32(task + 0x30) & 0xFFFFFF);
	} else if (f == FAMILY_F5ROGUE) {
		h->f5Colors = h->f5Viewport = h->f5Persp = 0, h->f5TexScale[0] = h->f5TexScale[1] = 0;
		f5List(n, h, be32(task + 0x30) & 0xFFFFFF);
	} else if (f == FAMILY_TURBO3D || f == FAMILY_T3DUX) {
		turbo3dList(n, h, be32(task + 0x30) & 0xFFFFFF);
	} else {
		runList(n, h, address(h, be32(task + 0x30)));
	}
	if (h->rejectedNow) {
		// The RSP runs this task from the start (what the list made so far goes).
		h->count = 0;
		h->doneList = 0;
		if (n->rdpGl) {
			glrQueueClear(n);
		}
		return false;
	}
	finishTask(n, h);
	h->doneList = dry ? h->doneList : list;
	return true;
}

size_t hleStateSize(void) {
	return sizeof(struct Hle);
}
