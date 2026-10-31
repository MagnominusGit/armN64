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
enum { FAMILY_NONE, FAMILY_F3D, FAMILY_F3DEX, FAMILY_F3DEX2, FAMILY_F3DPD };

// Geometry mode bits, kept in the F3DEX2 layout.
enum {
	GEOM_ZBUFFER = 0x1, GEOM_SHADE = 0x4, GEOM_CULL_FRONT = 0x200, GEOM_CULL_BACK = 0x400,
	GEOM_FOG = 0x10000, GEOM_LIGHTING = 0x20000, GEOM_TEXTURE_GEN = 0x40000,
	GEOM_TEXTURE_GEN_LINEAR = 0x80000, GEOM_SHADING_SMOOTH = 0x200000,
};

enum { VERTICES = 64, MATRICES = 32, LIGHTS = 8, LIST_MAX = 1 << 16, DL_DEPTH = 18 };

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
	float lightColor[LIGHTS + 1][3], lightDir[LIGHTS + 1][3], lookat[2][3];
	uint32_t geometry;
	// F3DPD's table of vertex colours.
	uint32_t colorBase;
	float texScaleS, texScaleT;
	int texTile, texLevel;
	bool texOn;
	float fogMul, fogOffset;
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
	// Counting the commands instead of drawing them (N64HLECHECK).
	bool dry;
	uint64_t list[LIST_MAX];
	size_t count;
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
	return op == 0x01 ? GLR_TRIANGLE_WORDS : op >= 0x08 && op <= 0x0F ? 4 + (op & 4 ? 8 : 0) + (op & 2 ? 8 : 0) + (op & 1 ? 2 : 0) :
	       op == 0x24 || op == 0x25 ? 2 : 1;
}

static void flush(struct N64* n, struct Hle* h) {
	// Whole commands only: a full list may end inside one, which waits for the next flush.
	size_t total = h->count, whole = 0;
	while (whole < total && whole + commandWords(h->list[whole]) <= total) {
		whole += commandWords(h->list[whole]);
	}
	h->count = whole;
	size_t capacity = (h->fifoEnd - h->fifoStart) / 8;
	if (!h->dry && capacity && h->count) {
		// A FIFO microcode's commands go through its output buffer in RDRAM, the RDP's registers
		// pointing at them, where programs can see them (DPC_CURRENT, the buffer); a list longer
		// than the buffer goes round it, the RDP taking each fill.
		for (size_t done = 0; done < h->count;) {
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
	memmove(h->list, h->list + whole, (total - whole) * sizeof(h->list[0]));
	h->count = total - whole;
}

static void emit(struct N64* n, struct Hle* h, uint64_t w) {
	if (h->count == LIST_MAX) {
		flush(n, h);
	}
	h->list[h->count++] = w;
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

static void loadVertices(struct N64* n, struct Hle* h, uint32_t a, int first, int count) {
	a = address(h, a);
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
		if (h->geometry & (GEOM_LIGHTING | GEOM_TEXTURE_GEN)) {
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
			if (h->geometry & GEOM_TEXTURE_GEN) {
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
		float unit = h->geometry & GEOM_TEXTURE_GEN ? 1 : 1 / 32.0f;
		v->s = s * h->texScaleS * unit;
		v->t = t * h->texScaleT * unit;
		if (h->geometry & GEOM_FOG) {
			float f = (v->w != 0 ? v->z / v->w : 0) * h->fogMul + h->fogOffset;
			v->a = (f < 0 ? 0 : f > 255 ? 255 : f) / 255.0f;
		}
	}
}

// G_MODIFYVTX (and F3DEX's G_MOVEWORD to the points): a loaded vertex's colour, texture
// coordinates (already scaled) or position on screen (x and y s13.2, z 16.16 in the viewport's range,
// kept by moving it in clip space at its own w).
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

// One RDP triangle (after libdragon's rdpq_triangle_cpu): edges, then shade, texture and depth
// coefficients as the command asks for them.
static void triangle(struct N64* n, struct Hle* h, const struct ScreenVertex* a, const struct ScreenVertex* b, const struct ScreenVertex* c, const struct ScreenVertex* flat) {
	// The microcodes scale 1/w so the corners' largest is a half, and send S and T times it: the
	// RDP divides that out with perspective correction and takes S and T as they are without.
	float maxW = a->invW > b->invW ? (a->invW > c->invW ? a->invW : c->invW) : (b->invW > c->invW ? b->invW : c->invW);
	float k = maxW > 0 ? 0.5f / maxW : 0.5f;
	if (n->rdpGl) {
		// For the GL renderer: the corners as they are (flat shading takes the one colour).
		bool persp = h->modeH & 0x80000;
		uint64_t head = (uint64_t) 0x01 << 56 | (uint64_t) ((h->geometry & GEOM_SHADE ? 4 : 0) | (h->texOn ? 2 : 0) | (h->geometry & GEOM_ZBUFFER ? 1 : 0)) |
		                (uint64_t) (h->texTile & 7) << 8 | (uint64_t) (h->texLevel & 7) << 12 | (uint64_t) persp << 16;
		emit(n, h, head);
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
			for (int i = 0; i < 10; i += 2) {
				uint32_t hi, lo;
				memcpy(&hi, &f[i], 4);
				memcpy(&lo, &f[i + 1], 4);
				emit(n, h, (uint64_t) hi << 32 | lo);
			}
		}
		return;
	}
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
	float attr = fabsf(nz) > 1e-20f ? -1.0f / nz : 0;
	uint32_t lft = nz < 0;
	float ish = fabsf(hy) > 1e-20f ? hx / hy : 0, ism = fabsf(my) > 1e-20f ? mx / my : 0, isl = fabsf(ly) > 1e-20f ? lx / ly : 0;
	float fy = floorf(y1) - y1;
	float xh = a->x + fy * ish, xm = a->x + fy * ism, xl = b->x;
	emit(n, h, (uint64_t) ((uint32_t) op << 24 | lft << 23 | (uint32_t) (h->texLevel & 7) << 19 | (uint32_t) (h->texTile & 7) << 16 | (y3f & 0x3FFF)) << 32 | (uint32_t) ((y2f & 0x3FFF) << 16 | (y1f & 0x3FFF)));
	emit(n, h, (uint64_t) (uint32_t) fixed16(xl) << 32 | (uint32_t) fixed16(isl));
	emit(n, h, (uint64_t) (uint32_t) fixed16(xh) << 32 | (uint32_t) fixed16(ish));
	emit(n, h, (uint64_t) (uint32_t) fixed16(xm) << 32 | (uint32_t) fixed16(ism));
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
	// Planes as (x, y, z, w) weights: near, then x and y within four screens.
	static const float planes[5][4] = { { 0, 0, 1, 1 }, { 1, 0, 0, 4 }, { -1, 0, 0, 4 }, { 0, 1, 0, 4 }, { 0, -1, 0, 4 } };
	// Corners inside every plane leave nothing to clip.
	bool inside = true;
	for (int k = 0; k < 3 && inside; ++k) {
		const struct Vertex* v = &poly[0][k];
		inside = isfinite(v->z) && v->z + v->w + 1e-4f * fabsf(v->w) >= 0 && fabsf(v->x) <= 4 * v->w && fabsf(v->y) <= 4 * v->w;
	}
	for (int p = inside ? 5 : 0; p < 5 && count >= 3; ++p) {
		const float* pl = planes[p];
		int out = 0;
		for (int i = 0; i < count; ++i) {
			const struct Vertex* a = &poly[cur][i], * b = &poly[cur][(i + 1) % count];
			// The near plane keeps what lies on it (2D layers at z = -w), give or take rounding.
			float da = a->x * pl[0] + a->y * pl[1] + a->z * pl[2] + a->w * pl[3] + (p == 0 ? 1e-4f * fabsf(a->w) : 0);
			float db = b->x * pl[0] + b->y * pl[1] + b->z * pl[2] + b->w * pl[3] + (p == 0 ? 1e-4f * fabsf(b->w) : 0);
			if (da >= 0) {
				poly[cur ^ 1][out++] = *a;
			}
			if ((da >= 0) != (db >= 0) && out < 9) {
				poly[cur ^ 1][out++] = lerp(a, b, da / (da - db));
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
	// Ends the list when every vertex is outside the same side of the view.
	uint32_t all = 0x3F;
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

// The task's microcode left to the RSP, this task (from the start) and later ones.
static void reject(struct Hle* h) {
	if (h->rejectedCount < 8) {
		h->rejected[h->rejectedCount++] = h->ucode;
	}
	h->rejectedNow = true;
}

static int family(struct N64* n, uint32_t data, uint32_t size);

// G_LOAD_UCODE: the list goes on in another microcode (its data named by G_RDPHALF_1); F3DEX's takes
// the new data's other modes with it, F3DEX2's keeps its own. One of no family here leaves the task
// to the RSP.
static bool loadMicrocode(struct N64* n, struct Hle* h, uint32_t dataSize) {
	int f = family(n, address(h, h->half1), dataSize);
	if (f == FAMILY_NONE) {
		reject(h);
		return false;
	}
	if (h->family != FAMILY_F3DEX2) {
		h->modeH = 0x080CFF, h->modeL = 0;
	}
	h->family = f;
	return true;
}

static void runList(struct N64* n, struct Hle* h, uint32_t pc) {
	uint32_t* stack = h->stack;
	int sp = h->sp;
	h->waiting = false;
	bool f3dex2 = h->family == FAMILY_F3DEX2, f3dex = h->family == FAMILY_F3DEX;
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
		if (op >= 0xE4 && !(f3dex2 && op == 0xF1)) {
			pc = rdpCommand(n, h, pc, w0, w1);
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
				h->texScaleS = (w1 >> 16) / 65536.0f, h->texScaleT = (w1 & 0xFFFF) / 65536.0f;
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
				if (index == 2) {
					h->lights = (int) (w1 / 24) > LIGHTS ? LIGHTS : (int) (w1 / 24);
				} else if (index == 6) {
					h->segments[(offset / 4) & 15] = w1 & 0xFFFFFF;
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
				} else if (index == 14) {
					// gSPForceMatrix: the combined matrix itself, until the next matrix command.
					readMatrix(n, a, h->mvp);
				} else if (index == 10) {
					int l = offset / 24;
					if (l < 2) {
						readLookat(n, h, a, l);
					} else {
						readLight(n, h, a, l - 2);
					}
				}
				break;
			}
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
				if (!loadMicrocode(n, h, (w0 & 0xFFFF) + 1)) {
					return;
				}
				f3dex2 = h->family == FAMILY_F3DEX2, f3dex = h->family == FAMILY_F3DEX;
				break;
			case 0xE2:
			case 0xE3:
				// The mask as the microcode makes it: the top bit shifted right by len - 1, with sign,
				// then by 32 - shift - len (shift and len give no mask past bit 0).
				otherModeMask(n, h, op == 0xE3, (uint32_t) ((int32_t) 0x80000000 >> (w0 & 31)) >> ((w0 >> 8) & 31), w1);
				break;
			}
		} else {
			int scale = f3dex ? 2 : 10;
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
				if (f3dex) {
					drawTriangle(n, h, ((w0 >> 16) & 0xFF) / 2, ((w0 >> 8) & 0xFF) / 2, (w0 & 0xFF) / 2);
					drawTriangle(n, h, ((w1 >> 16) & 0xFF) / 2, ((w1 >> 8) & 0xFF) / 2, (w1 & 0xFF) / 2);
				} else if (h->family == FAMILY_F3DPD) {
					// TRIX: each triangle's first two vertices from w1's nibbles, its third from w0's.
					for (uint32_t a1 = w1, a0 = w0; a1; a1 >>= 8, a0 >>= 4) {
						drawTriangle(n, h, a1 & 15, (a1 >> 4) & 15, a0 & 15);
					}
				}
				break;
			case 0x07:
				if (h->family == FAMILY_F3DPD) {
					h->colorBase = address(h, w1);
				}
				break;
			case 0xB4:
				h->half1 = w1;
				break;
			case 0xB2:
				// F3DEX's G_MODIFYVTX (Fast3D's 0xB2 is the third texture rectangle word).
				if (f3dex) {
					modifyVertex(h, (int) (w0 & 0xFFFF) / 2, (w0 >> 16) & 0xFF, w1);
				}
				break;
			case 0xAF:
				if (!loadMicrocode(n, h, (w0 & 0xFFFF) + 1)) {
					return;
				}
				f3dex2 = h->family == FAMILY_F3DEX2, f3dex = h->family == FAMILY_F3DEX;
				break;
			case 0xB5:
				if (f3dex) {
					int v0 = ((w1 >> 24) & 0xFF) / 2, v1 = ((w1 >> 16) & 0xFF) / 2, v2 = ((w1 >> 8) & 0xFF) / 2, v3 = (w1 & 0xFF) / 2;
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
				if (index == 2) {
					int l = (int) (((w1 - 0x80000000u) >> 5) - 1);
					h->lights = l < 0 ? 0 : l > LIGHTS ? LIGHTS : l;
				} else if (index == 6) {
					h->segments[(offset / 4) & 15] = w1 & 0xFFFFFF;
				} else if (index == 8) {
					h->fogMul = (int16_t) (w1 >> 16), h->fogOffset = (int16_t) w1;
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
					cullOrEnd(h, (w0 & 0xFFFF) / (f3dex ? 2 : 40), (w1 & 0xFFFF) / (f3dex ? 2 : 40), &end);
				}
				break;
			case 0xBF:
				// Fast3D's indices are vertex numbers times 10: other ones are another microcode's.
				if (!f3dex && (((w1 >> 16) & 0xFF) % 10 || ((w1 >> 8) & 0xFF) % 10 || (w1 & 0xFF) % 10)) {
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

// The microcode family from the first ID string in its data (the block may run on into the next
// microcode's data, and its string).
// The graphics task's microcode by the name its data gives (a Fast3D or F3DEX family line), else
// by the hash of its first code.
void hleUcodeName(struct N64* n, char* out, size_t size) {
	const uint8_t* task = &n->sp.mem[0xFC0];
	uint32_t data = be32(task + 0x18) & 0xFFFFFF, dataSize = be32(task + 0x1C), code = be32(task + 0x10) & 0xFFFFFF;
	char text[0x1001];
	uint32_t length = dataSize < 0x1000 ? dataSize : 0x1000;
	for (uint32_t i = 0; i < length; ++i) {
		uint8_t c = n->rdram[(data + i) & (RDRAM_SIZE - 1)];
		text[i] = c >= 32 && c < 127 ? (char) c : ' ';
	}
	text[length] = 0;
	const char* name = strstr(text, "RSP Gfx ucode ");
	name = name ? name : strstr(text, "RSP SW Version");
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

static int family(struct N64* n, uint32_t data, uint32_t size) {
	if (size > 0x1000) {
		size = 0x1000;
	}
	char text[0x1001];
	for (uint32_t i = 0; i < size; ++i) {
		uint8_t c = n->rdram[(data + i) & (RDRAM_SIZE - 1)];
		text[i] = c >= 32 && c < 127 ? (char) c : ' ';
	}
	text[size] = 0;
	const char* id = strstr(text, "RSP Gfx ucode ");
	const char* sw = strstr(text, "RSP SW Version");
	if (sw && (!id || sw < id)) {
		return FAMILY_F3D;
	}
	if (!id) {
		return FAMILY_NONE;
	}
	const char* name = id + 14;
	if ((!strncmp(name, "F3DEX2", 6) || !strncmp(name, "F3DZEX", 6) || !strncmp(name, "F3DLX2", 6) || !strncmp(name, "F3DLP2", 6)) &&
	    (name[6] == ' ' || name[6] == '.' || name[6] == '_')) {
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
static void finishTask(struct N64* n, struct Hle* h) {
	if (!h->dry && h->waiting) {
		n->sp.hleDoneAt = n->cycles + 2048;
	} else if (!h->dry) {
		n->sp.hleDoneAt = n->cycles + 4096 + 64 * (uint64_t) h->count;
	}
	flush(n, h);
	if (!h->dry && !h->waiting) {
		n->dp.busyUntil = n->dp.busyUntil > n->sp.hleDoneAt ? n->dp.busyUntil : n->sp.hleDoneAt;
	}
}

bool hleWaiting(struct N64* n) {
	return n->hle && n->hle->waiting;
}

// A waiting task looks at the command it waits on again, and goes on when the CPU changed it.
void hleResume(struct N64* n) {
	struct Hle* h = n->hle;
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
	int f = family(n, data, dataSize);
	// Perfect Dark's microcode has no name: it is known by its code.
	if (f == FAMILY_NONE && ucode == 0xB0811E2Fu) {
		f = FAMILY_F3DPD;
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
	h->dry = dry;
	memset(h->segments, 0, sizeof(h->segments));
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
	runList(n, h, address(h, be32(task + 0x30)));
	if (h->rejectedNow) {
		// The RSP runs this task from the start (what the list made so far goes).
		h->count = 0;
		h->doneList = 0;
		return false;
	}
	finishTask(n, h);
	h->doneList = dry ? h->doneList : list;
	return true;
}

size_t hleStateSize(void) {
	return sizeof(struct Hle);
}
