#include "n64.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// High-level graphics (N64_ACCURACY=hle): a graphics task's display list run here instead of by its
// microcode, for the Fast3D family (Fast3D, F3DEX, F3DEX2), with the triangles and other commands
// it makes drawn by the RDP as usual. Microcodes it doesn't know run on the RSP.

enum { FAMILY_NONE, FAMILY_F3D, FAMILY_F3DEX, FAMILY_F3DEX2 };

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
	float texScaleS, texScaleT;
	int texTile, texLevel;
	bool texOn;
	float fogMul, fogOffset;
	uint32_t modeH, modeL;
	uint32_t half1;
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

static void flush(struct N64* n, struct Hle* h) {
	if (!h->dry) {
		rdpCommands(n, h->list, h->count);
	} else {
		for (size_t i = 0; i < h->count;) {
			int op = (h->list[i] >> 56) & 0x3F;
			++n->checkCounts[1][op];
			if (n->checkLen[1] < 4096) {
				n->checkSeq[1][n->checkLen[1]++] = h->list[i];
			}
			i += op >= 0x08 && op <= 0x0F ? 4 + (op & 4 ? 8 : 0) + (op & 2 ? 8 : 0) + (op & 1 ? 2 : 0) : op == 0x24 || op == 0x25 ? 2 : 1;
		}
	}
	h->count = 0;
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
	for (int i = 0; i < count && first + i < VERTICES; ++i, a += 16) {
		struct Vertex* v = &h->v[first + i];
		float x = rd16(n, a), y = rd16(n, a + 2), z = rd16(n, a + 4);
		const float* m = h->mvp;
		v->x = x * m[0] + y * m[4] + z * m[8] + m[12];
		v->y = x * m[1] + y * m[5] + z * m[9] + m[13];
		v->z = x * m[2] + y * m[6] + z * m[10] + m[14];
		v->w = x * m[3] + y * m[7] + z * m[11] + m[15];
		float s = rd16(n, a + 8), t = rd16(n, a + 10);
		uint8_t c[4] = { n->rdram[(a + 12) & (RDRAM_SIZE - 1)], n->rdram[(a + 13) & (RDRAM_SIZE - 1)], n->rdram[(a + 14) & (RDRAM_SIZE - 1)], n->rdram[(a + 15) & (RDRAM_SIZE - 1)] };
		v->a = c[3] / 255.0f;
		if (h->geometry & (GEOM_LIGHTING | GEOM_TEXTURE_GEN)) {
			float nrm[3] = { rd8(n, a + 12), rd8(n, a + 13), rd8(n, a + 14) };
			float e[3] = {
				nrm[0] * mv[0] + nrm[1] * mv[4] + nrm[2] * mv[8],
				nrm[0] * mv[1] + nrm[1] * mv[5] + nrm[2] * mv[9],
				nrm[0] * mv[2] + nrm[1] * mv[6] + nrm[2] * mv[10],
			};
			normalize(e);
			if (h->geometry & GEOM_LIGHTING) {
				float rgb[3] = { h->lightColor[h->lights][0], h->lightColor[h->lights][1], h->lightColor[h->lights][2] };
				for (int l = 0; l < h->lights; ++l) {
					float d = e[0] * h->lightDir[l][0] + e[1] * h->lightDir[l][1] + e[2] * h->lightDir[l][2];
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
				float gx = e[0] * h->lookat[0][0] + e[1] * h->lookat[0][1] + e[2] * h->lookat[0][2];
				float gy = e[0] * h->lookat[1][0] + e[1] * h->lookat[1][1] + e[2] * h->lookat[1][2];
				if (h->geometry & GEOM_TEXTURE_GEN_LINEAR) {
					s = acosf(gx < -1 ? -1 : gx > 1 ? 1 : gx) * 325.94931f;
					t = acosf(gy < -1 ? -1 : gy > 1 ? 1 : gy) * 325.94931f;
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
		// With perspective correction the RDP divides S and T by W: they go in times 1/w, scaled
		// so the largest 1/w is 0x7FFF.
		bool persp = h->modeH & 0x80000;
		float w1 = persp ? a->invW : 1, w2 = persp ? b->invW : 1, w3 = persp ? c->invW : 1;
		float maxW = w1 > w2 ? (w1 > w3 ? w1 : w3) : (w2 > w3 ? w2 : w3);
		float k = maxW > 0 ? 1 / maxW : 1;
		w1 *= k, w2 *= k, w3 *= k;
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
	s.x = v->x * iw * h->vscale[0] + h->vtrans[0];
	s.y = -v->y * iw * h->vscale[1] + h->vtrans[1];
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
	for (int p = 0; p < 5 && count >= 3; ++p) {
		const float* pl = planes[p];
		int out = 0;
		for (int i = 0; i < count; ++i) {
			const struct Vertex* a = &poly[cur][i], * b = &poly[cur][(i + 1) % count];
			float da = a->x * pl[0] + a->y * pl[1] + a->z * pl[2] + a->w * pl[3] - (p == 0 ? 1e-5f : 0);
			float db = b->x * pl[0] + b->y * pl[1] + b->z * pl[2] + b->w * pl[3] - (p == 0 ? 1e-5f : 0);
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
	if ((area > 0 && (h->geometry & GEOM_CULL_BACK)) || (area < 0 && (h->geometry & GEOM_CULL_FRONT)) || area == 0) {
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

static void otherMode(struct N64* n, struct Hle* h, bool high, int shift, int len, uint32_t bits) {
	uint32_t mask = (len >= 32 ? 0xFFFFFFFFu : ((1u << len) - 1)) << shift;
	uint32_t* m = high ? &h->modeH : &h->modeL;
	*m = (*m & ~mask) | (bits & mask);
	emit(n, h, (uint64_t) (0xEF000000u | (h->modeH & 0xFFFFFF)) << 32 | h->modeL);
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

static void runList(struct N64* n, struct Hle* h, uint32_t start) {
	uint32_t stack[DL_DEPTH];
	int sp = 0;
	uint32_t pc = address(h, start);
	bool f3dex2 = h->family == FAMILY_F3DEX2, f3dex = h->family == FAMILY_F3DEX;
	for (int guard = 0; guard < 1 << 22; ++guard) {
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
				if ((uint32_t) (h->v[((w0 >> 1) & 0x7FF) % VERTICES].z) <= w1) {
					next = address(h, h->half1);
				}
				break;
			case 0x05:
				drawTriangle(n, h, ((w0 >> 16) & 0xFF) / 2, ((w0 >> 8) & 0xFF) / 2, (w0 & 0xFF) / 2);
				break;
			case 0x06:
			case 0x07:
				drawTriangle(n, h, ((w0 >> 16) & 0xFF) / 2, ((w0 >> 8) & 0xFF) / 2, (w0 & 0xFF) / 2);
				drawTriangle(n, h, ((w1 >> 16) & 0xFF) / 2, ((w1 >> 8) & 0xFF) / 2, (w1 & 0xFF) / 2);
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
				}
				next = address(h, w1);
				break;
			case 0xDF:
				end = true;
				break;
			case 0xE1:
				h->half1 = w1;
				break;
			case 0xE2:
			case 0xE3: {
				int len = (w0 & 0xFF) + 1, shift = 32 - ((w0 >> 8) & 0xFF) - len;
				otherMode(n, h, op == 0xE3, shift, len, w1);
				break;
			}
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
			case 0x06:
				if (!((w0 >> 16) & 0xFF) && sp < DL_DEPTH) {
					stack[sp++] = next;
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
				}
				break;
			case 0xB4:
				h->half1 = w1;
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
				}
				break;
			}
			case 0xBD:
				popMatrix(h, 1);
				break;
			case 0xBE:
				cullOrEnd(h, (w0 & 0xFFFF) / (f3dex ? 2 : 40), (w1 & 0xFFFF) / (f3dex ? 2 : 40), &end);
				break;
			case 0xBF:
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

// The microcode family from the strings in its data; without them, from the commands the display
// list starts with (F3DEX2 ones from 0xD7 to 0xDF, Fast3D ones from 0xB6 to 0xBF).
static int family(struct N64* n, uint32_t data, uint32_t size, uint32_t list) {
	if (size > 0x1000) {
		size = 0x1000;
	}
	char text[0x1001];
	for (uint32_t i = 0; i < size; ++i) {
		uint8_t c = n->rdram[(data + i) & (RDRAM_SIZE - 1)];
		text[i] = c >= 32 && c < 127 ? (char) c : ' ';
	}
	text[size] = 0;
	if (strstr(text, "F3DEX2") || strstr(text, "F3DZEX") || strstr(text, "F3DLX2") || strstr(text, "F3DLP2")) {
		return FAMILY_F3DEX2;
	}
	if (strstr(text, "S2DEX") || strstr(text, "L3DEX") || strstr(text, "F3DEX3") || strstr(text, "Turbo")) {
		return FAMILY_NONE;
	}
	if (strstr(text, "F3DEX") || strstr(text, "F3DLX") || strstr(text, "F3DLP")) {
		return FAMILY_F3DEX;
	}
	if (strstr(text, "RSP Gfx") || strstr(text, "RSP SW Version")) {
		return strstr(text, "RSP SW Version") ? FAMILY_F3D : FAMILY_NONE;
	}
	int f3dex2 = 0, f3d = 0;
	for (int i = 0; i < 64; ++i) {
		int op = n->rdram[(list + i * 8) & (RDRAM_SIZE - 1)];
		f3dex2 += op >= 0xD7 && op <= 0xDF;
		f3d += op >= 0xB6 && op <= 0xBF;
	}
	return f3dex2 > f3d ? FAMILY_F3DEX2 : FAMILY_F3D;
}

// Runs the graphics task in DMEM at high level when its microcode is one of the known families;
// false leaves it to the RSP. Dry, it only counts the RDP commands it would send.
bool hleGraphics(struct N64* n, bool dry) {
	const uint8_t* task = &n->sp.mem[0xFC0];
	if (be32(task) != 1) {
		return false;
	}
	uint32_t data = be32(task + 0x18) & 0xFFFFFF, dataSize = be32(task + 0x1C);
	int f = family(n, data, dataSize, be32(task + 0x30) & 0xFFFFFF);
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
	h->count = 0;
	runList(n, h, be32(task + 0x30));
	flush(n, h);
	return true;
}
