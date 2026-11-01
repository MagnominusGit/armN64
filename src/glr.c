#include "rdp.h"
#include "gles.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The GL renderer (N64_RDP_GL, with hle): the GPU draws what the HLE makes and the RDP commands
// around it, into a texture for each color image, with a combiner shader made for each combine
// mode. The RDP keeps the state (rdp.c); what the GPU drew goes back to RDRAM when anything reads
// it (rdpWait), and a color image goes up again from RDRAM when the GPU next draws into it. A
// speed setting: not exact.

typedef unsigned GLenum;
typedef unsigned GLuint;
typedef int GLint;
typedef int GLsizei;
typedef unsigned GLbitfield;
typedef intptr_t GLintptr;
typedef intptr_t GLsizeiptr;
typedef unsigned char GLboolean;

enum {
	GL_TEXTURE_2D = 0x0DE1, GL_RGBA = 0x1908, GL_RGBA8 = 0x8058, GL_UNSIGNED_BYTE = 0x1401, GL_UNSIGNED_INT = 0x1405, GL_FLOAT = 0x1406,
	GL_FRAMEBUFFER = 0x8D40, GL_READ_FRAMEBUFFER = 0x8CA8, GL_DRAW_FRAMEBUFFER = 0x8CA9, GL_COLOR_ATTACHMENT0 = 0x8CE0, GL_DEPTH_ATTACHMENT = 0x8D00, GL_RENDERBUFFER = 0x8D41,
	GL_DEPTH_COMPONENT24 = 0x81A6, GL_FRAMEBUFFER_COMPLETE = 0x8CD5, GL_DEPTH_STENCIL_ATTACHMENT = 0x821A, GL_DEPTH24_STENCIL8 = 0x88F0,
	GL_STENCIL_TEST = 0x0B90, GL_STENCIL_BUFFER_BIT = 0x400, GL_KEEP = 0x1E00, GL_REPLACE = 0x1E01, GL_EQUAL = 0x202, GL_GEQUAL = 0x206,
	GL_TEXTURE_MIN_FILTER = 0x2801, GL_TEXTURE_MAG_FILTER = 0x2800, GL_NEAREST = 0x2600, GL_TEXTURE_WRAP_S = 0x2802,
	GL_TEXTURE_WRAP_T = 0x2803, GL_CLAMP_TO_EDGE = 0x812F, GL_TEXTURE0 = 0x84C0,
	GL_ARRAY_BUFFER = 0x8892, GL_STREAM_DRAW = 0x88E0, GL_TRIANGLES = 4, GL_MAP_WRITE_BIT = 2, GL_MAP_UNSYNCHRONIZED_BIT = 0x20,
	GL_SCISSOR_TEST = 0x0C11, GL_DEPTH_TEST = 0x0B71, GL_BLEND = 0x0BE2, GL_POLYGON_OFFSET_FILL = 0x8037,
	GL_DITHER = 0x0BD0, GL_COLOR_BUFFER_BIT = 0x4000, GL_DEPTH_BUFFER_BIT = 0x100,
	GL_LESS = 0x201, GL_LEQUAL = 0x203, GL_ALWAYS = 0x207,
	GL_ZERO = 0, GL_ONE = 1, GL_SRC_ALPHA = 0x302, GL_ONE_MINUS_SRC_ALPHA = 0x303, GL_CONSTANT_ALPHA = 0x8003,
	GL_VERTEX_SHADER = 0x8B31, GL_FRAGMENT_SHADER = 0x8B30, GL_COMPILE_STATUS = 0x8B81, GL_LINK_STATUS = 0x8B82,
	GL_PACK_ALIGNMENT = 0x0D05, GL_UNPACK_ALIGNMENT = 0x0CF5,
};

enum { TARGETS = 8, TARGET_ROWS = 480, PROGRAMS = 256, TEXTURES = 1024, TEXTURE_BUCKETS = 2048, BATCH_VERTICES = 1 << 16, VERTEX_FLOATS = 11,
       VERTEX_BUFFER_BYTES = BATCH_VERTICES * VERTEX_FLOATS * 4 * 2 };

struct Gl {
	void (*GenFramebuffers)(GLsizei, GLuint*);
	void (*BindFramebuffer)(GLenum, GLuint);
	void (*BlitFramebuffer)(GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum);
	void (*FramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint);
	void (*FramebufferRenderbuffer)(GLenum, GLenum, GLenum, GLuint);
	GLenum (*CheckFramebufferStatus)(GLenum);
	void (*DeleteFramebuffers)(GLsizei, const GLuint*);
	void (*GenRenderbuffers)(GLsizei, GLuint*);
	void (*BindRenderbuffer)(GLenum, GLuint);
	void (*RenderbufferStorage)(GLenum, GLenum, GLsizei, GLsizei);
	void (*GenTextures)(GLsizei, GLuint*);
	void (*DeleteTextures)(GLsizei, const GLuint*);
	void (*BindTexture)(GLenum, GLuint);
	void (*ActiveTexture)(GLenum);
	void (*TexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);
	void (*TexSubImage2D)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*);
	void (*TexParameteri)(GLenum, GLenum, GLint);
	void (*Viewport)(GLint, GLint, GLsizei, GLsizei);
	void (*Scissor)(GLint, GLint, GLsizei, GLsizei);
	void (*Enable)(GLenum);
	void (*Disable)(GLenum);
	void (*ClearColor)(float, float, float, float);
	void (*ClearDepthf)(float);
	void (*Clear)(GLbitfield);
	void (*DepthFunc)(GLenum);
	void (*DepthMask)(GLboolean);
	void (*BlendFunc)(GLenum, GLenum);
	void (*BlendFuncSeparate)(GLenum, GLenum, GLenum, GLenum);
	void (*BlendColor)(float, float, float, float);
	void (*PolygonOffset)(float, float);
	void (*StencilFunc)(GLenum, GLint, GLuint);
	void (*StencilOp)(GLenum, GLenum, GLenum);
	void (*ColorMask)(GLboolean, GLboolean, GLboolean, GLboolean);
	void (*ClearStencil)(GLint);
	void (*ReadPixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*);
	void (*PixelStorei)(GLenum, GLint);
	GLuint (*CreateShader)(GLenum);
	void (*ShaderSource)(GLuint, GLsizei, const char* const*, const GLint*);
	void (*CompileShader)(GLuint);
	void (*GetShaderiv)(GLuint, GLenum, GLint*);
	void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei*, char*);
	void (*DeleteShader)(GLuint);
	GLuint (*CreateProgram)(void);
	void (*AttachShader)(GLuint, GLuint);
	void (*LinkProgram)(GLuint);
	void (*GetProgramiv)(GLuint, GLenum, GLint*);
	void (*GetProgramInfoLog)(GLuint, GLsizei, GLsizei*, char*);
	void (*UseProgram)(GLuint);
	void (*DeleteProgram)(GLuint);
	GLint (*GetUniformLocation)(GLuint, const char*);
	void (*Uniform1i)(GLint, GLint);
	void (*Uniform1f)(GLint, float);
	void (*Uniform2f)(GLint, float, float);
	void (*Uniform4f)(GLint, float, float, float, float);
	void (*Uniform4iv)(GLint, GLsizei, const GLint*);
	void (*GenVertexArrays)(GLsizei, GLuint*);
	void (*BindVertexArray)(GLuint);
	void (*GenBuffers)(GLsizei, GLuint*);
	void (*BindBuffer)(GLenum, GLuint);
	void (*BufferData)(GLenum, GLsizeiptr, const void*, GLenum);
	void* (*MapBufferRange)(GLenum, GLintptr, GLsizeiptr, GLbitfield);
	GLboolean (*UnmapBuffer)(GLenum);
	void (*VertexAttribPointer)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
	void (*VertexAttribIPointer)(GLuint, GLint, GLenum, GLsizei, const void*);
	void (*EnableVertexAttribArray)(GLuint);
	void (*DrawArrays)(GLenum, GLint, GLsizei);
	void (*Flush)(void);
};

// A color image the GPU draws into: its texture and framebuffer, whether the texture has the
// RDRAM's pixels, and the rows drawn since they last went back.
struct Target {
	uint32_t addr, width;
	int size;
	GLuint texture, framebuffer;
	// Rows loaded from RDRAM since the GPU last gave its pixels back.
	int loadedRows;
	int dirtyFrom, dirtyTo;
	uint64_t used;
	// Rows whose pages are flagged for rdpWait since the last sync.
	int marked;
};

// What the fragment shader is made for.
struct ProgramKey {
	uint64_t combine;
	uint32_t modes;
	// The level of detail: worked out (bit 0), picking the tiles (1), sharpen (2), detail (3), the
	// primitive's levels (4-6) and the least (8-12).
	uint32_t lod;
};

// A program's uniforms as last set.
struct Uniforms {
	float size[2], prim[4], env[4], fog[4], blend[4], primLod, primDepth;
	int persp, tile[2][12], image[2][4];
};

struct Program {
	struct ProgramKey key;
	GLuint program;
	GLint size, persp, prim, env, fog, blend, primLod, primDepth, tile[2], image[2], texture[2];
	struct Uniforms set;
	bool uniformsSet;
};

// What GL is set to (-1: not known), so a batch changes only what differs.
struct Applied {
	GLuint framebuffer, program, texture[3];
	int unit, viewport, scissorOn, scissor[4], depth, depthMask, offset, blend;
	float blendConstant;
};

struct Texture {
	uint64_t hash;
	GLuint texture;
	// Its rows repeat every period (TMEM wraps): only those are decoded, 0 when all are.
	int width, height, period;
	uint64_t used;
	// The next texture in its hash bucket (index + 1, 0 for none).
	int next;
};

// Everything a batch of primitives is drawn with; a change draws the batch first.
struct DrawState {
	struct Target* target;
	struct Program* program;
	struct Texture* texture[2];
	int tile[2][12];
	float prim[4], env[4], fog[4], blend[4], primLod, primDepth;
	int scissor[4];
	bool depthTest, depthWrite, decal, depthLess, blendOn, persp;
	int blendB;
	// Memory as P: 1 with a per-pixel A, 2 with a constant one (blendConstant).
	int memoryFirst;
	float blendConstant;
};

struct Glr {
	struct Gles* gles;
	// Each tile's texture as of a TMEM generation.
	struct Texture* tileTexture[8];
	uint32_t tileGen[8], tileTlut[8];
	struct Gl gl;
	struct Target target[TARGETS];
	GLuint depth, vertexArray, vertexBuffer, vertexShader;
	// Targets drawn at scale times the RDP's resolution (N64_GL_SCALE); their pixels go to and from
	// RDRAM through a texture at its resolution.
	int scale;
	GLuint native, nativeFramebuffer;
	struct Program programs[PROGRAMS];
	int programCount;
	struct Texture textures[TEXTURES];
	int textureBucket[TEXTURE_BUCKETS];
	uint64_t clock;
	struct DrawState state;
	bool stateSet;
	float* batch;
	int batchCount;
	uint8_t* rows;
	uint32_t* image;
	struct Applied applied;
	size_t vertexOffset;
	uint64_t rawField;
	int rawTriangles;
	// The stencil value the last decal batch marked its pixels with.
	int stencilRef;
	// The primitive's number (the RDP's seq), for the noise.
	uint32_t seq;
	// The HLE's corners for triangles it has sent and the RDP hasn't reached yet (a ring).
	struct GlrCorners* queue;
	uint32_t queueRead, queueWrite;
	// The state of the last primitive, valid while the RDP's stateGen stays (and no sync).
	bool fastValid;
	uint32_t fastGen;
	int fastKey, fastRows;
};

static void useFramebuffer(struct Glr* g, GLuint framebuffer) {
	if (g->applied.framebuffer != framebuffer) {
		g->gl.BindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		g->applied.framebuffer = framebuffer;
	}
}

static void useScissor(struct Glr* g, bool on, int x, int y, int w, int h);

// Rows from to to - 1 between a target and the texture at the RDP's resolution, scaled (nearest).
static void blitRows(struct Glr* g, struct Target* t, int from, int to, bool toTarget) {
	struct Gl* gl = &g->gl;
	int const k = g->scale, w = (int) t->width;
	useScissor(g, false, 0, 0, 0, 0);
	gl->BindFramebuffer(GL_READ_FRAMEBUFFER, toTarget ? g->nativeFramebuffer : t->framebuffer);
	gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, toTarget ? t->framebuffer : g->nativeFramebuffer);
	if (toTarget) {
		gl->BlitFramebuffer(0, from, w, to, 0, from * k, w * k, to * k, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	} else {
		gl->BlitFramebuffer(0, from * k, w * k, to * k, 0, from, w, to, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	}
	gl->BindFramebuffer(GL_FRAMEBUFFER, g->nativeFramebuffer);
	g->applied.framebuffer = g->nativeFramebuffer;
}

static void useTexture(struct Glr* g, int unit, GLuint texture) {
	if (g->applied.texture[unit] != texture) {
		if (g->applied.unit != unit) {
			g->gl.ActiveTexture(GL_TEXTURE0 + (GLenum) unit);
			g->applied.unit = unit;
		}
		g->gl.BindTexture(GL_TEXTURE_2D, texture);
		g->applied.texture[unit] = texture;
	}
}

// Unit 2 for uploads, active.
static void uploadTexture(struct Glr* g, GLuint texture) {
	useTexture(g, 2, texture);
	if (g->applied.unit != 2) {
		g->gl.ActiveTexture(GL_TEXTURE0 + 2);
		g->applied.unit = 2;
	}
}

static void useScissor(struct Glr* g, bool on, int x, int y, int w, int h) {
	struct Applied* a = &g->applied;
	if (a->scissorOn != on) {
		on ? g->gl.Enable(GL_SCISSOR_TEST) : g->gl.Disable(GL_SCISSOR_TEST);
		a->scissorOn = on;
	}
	if (on && (a->scissor[0] != x || a->scissor[1] != y || a->scissor[2] != w || a->scissor[3] != h)) {
		g->gl.Scissor(x * g->scale, y * g->scale, w * g->scale, h * g->scale);
		a->scissor[0] = x, a->scissor[1] = y, a->scissor[2] = w, a->scissor[3] = h;
	}
}

static void useDepthMask(struct Glr* g, bool on) {
	if (g->applied.depthMask != on) {
		g->gl.DepthMask(on);
		g->applied.depthMask = on;
	}
}

static void uniform(struct Glr* g, GLint location, int count, const float* values, float* set) {
	if (memcmp(values, set, (size_t) count * sizeof(float))) {
		count == 1 ? g->gl.Uniform1f(location, values[0]) : count == 2 ? g->gl.Uniform2f(location, values[0], values[1])
		                                                     : g->gl.Uniform4f(location, values[0], values[1], values[2], values[3]);
		memcpy(set, values, (size_t) count * sizeof(float));
	}
}

static bool loadGl(struct Glr* g) {
	struct Gl* gl = &g->gl;
#define GLFN(name) (*(void**) &gl->name = glesProc(g->gles, "gl" #name), gl->name != NULL)
	return GLFN(GenFramebuffers) && GLFN(BindFramebuffer) && GLFN(BlitFramebuffer) && GLFN(FramebufferTexture2D) && GLFN(FramebufferRenderbuffer) &&
	       GLFN(CheckFramebufferStatus) && GLFN(DeleteFramebuffers) && GLFN(GenRenderbuffers) && GLFN(BindRenderbuffer) &&
	       GLFN(RenderbufferStorage) && GLFN(GenTextures) && GLFN(DeleteTextures) && GLFN(BindTexture) && GLFN(ActiveTexture) &&
	       GLFN(TexImage2D) && GLFN(TexSubImage2D) && GLFN(TexParameteri) && GLFN(Viewport) && GLFN(Scissor) && GLFN(Enable) &&
	       GLFN(Disable) && GLFN(ClearColor) && GLFN(ClearDepthf) && GLFN(Clear) && GLFN(DepthFunc) && GLFN(DepthMask) &&
	       GLFN(BlendFunc) && GLFN(BlendFuncSeparate) && GLFN(BlendColor) && GLFN(PolygonOffset) && GLFN(StencilFunc) && GLFN(StencilOp) && GLFN(ColorMask) && GLFN(ClearStencil) && GLFN(ReadPixels) && GLFN(PixelStorei) && GLFN(CreateShader) &&
	       GLFN(ShaderSource) && GLFN(CompileShader) && GLFN(GetShaderiv) && GLFN(GetShaderInfoLog) && GLFN(DeleteShader) &&
	       GLFN(CreateProgram) && GLFN(AttachShader) && GLFN(LinkProgram) && GLFN(GetProgramiv) && GLFN(GetProgramInfoLog) &&
	       GLFN(UseProgram) && GLFN(DeleteProgram) && GLFN(GetUniformLocation) && GLFN(Uniform1i) && GLFN(Uniform1f) &&
	       GLFN(Uniform2f) && GLFN(Uniform4f) && GLFN(Uniform4iv) && GLFN(GenVertexArrays) && GLFN(BindVertexArray) &&
	       GLFN(GenBuffers) && GLFN(BindBuffer) && GLFN(BufferData) && GLFN(MapBufferRange) && GLFN(UnmapBuffer) && GLFN(VertexAttribPointer) && GLFN(VertexAttribIPointer) &&
	       GLFN(EnableVertexAttribArray) && GLFN(DrawArrays) && GLFN(Flush);
#undef GLFN
}

// ---- Shaders

static const char vertexSource[] =
	"#version 300 es\n"
	"layout(location = 0) in vec4 aPos;\n"
	"layout(location = 1) in vec4 aShade;\n"
	"layout(location = 2) in vec2 aTex;\n"
	"layout(location = 3) in uint aSeq;\n"
	"uniform vec2 uSize;\n"
	"uniform int uPersp;\n"
	"out vec4 vShade;\n"
	"out vec3 vTex;\n"
	"flat out uint vSeq;\n"
	"void main() {\n"
	"	vSeq = aSeq;\n"
	// Depth as the RDP's 18 bits over a cleared pixel's 0x3FFFF (the corners' z is 15.3 over 0x7FFF).
	"	gl_Position = vec4(aPos.x / uSize.x * 2.0 - 1.0, aPos.y / uSize.y * 2.0 - 1.0, aPos.z * (262136.0 / 262143.0) * 2.0 - 1.0, 1.0);\n"
	"	float q = uPersp != 0 ? aPos.w : 1.0;\n"
	"	vShade = aShade;\n"
	"	vTex = vec3(aTex * q, q);\n"
	"}\n";

// Texture addressing as the RDP does it (rdp.c wrapCoord, sample): tile = shiftS, shiftT, sl, tl,
// sh, th, cmS, cmT, maskS, maskT, bilinear, unused. The RDP's coordinates are fixed point: a little
// bias keeps interpolation just under a whole step from flooring to the one before.
static const char fragmentHead[] =
	"#version 300 es\n"
	"precision highp float;\n"
	"precision highp int;\n"
	"in vec4 vShade;\n"
	"in vec3 vTex;\n"
	"flat in uint vSeq;\n"
	"uniform vec4 uPrim, uEnv, uFog, uBlend;\n"
	"uniform float uPrimLod, uPrimDepth;\n"
	"uniform int uPersp;\n"
	"uniform ivec4 uTile0[3], uTile1[3];\n"
	"uniform ivec4 uImage0, uImage1;\n"
	"uniform highp sampler2D uTex0, uTex1;\n"
	"out vec4 oColor;\n"
	"vec3 e8(vec3 v) { return floor(v * 255.0 + 0.5); }\n"
	// rdp.c's clamp9: the combiner keeps 9 bits; 0x100-0x17F saturate, 0x180-0x1FF are negative.
	"vec4 clamp9(vec4 v) { v = mod(v, 512.0); return mix(mix(v, vec4(255.0), step(256.0, v)), vec4(0.0), step(384.0, v)); }\n"
	"float e8(float v) { return floor(v * 255.0 + 0.5); }\n"
	"int wrapCoord(int c, int shift, int lo, int hi, int cm, int mask, out int frac) {\n"
	"	if (shift < 11) c >>= shift; else c <<= 16 - shift;\n"
	"	c -= lo << 3;\n"
	"	if ((cm & 2) != 0 || mask == 0) c = clamp(c, 0, (hi - lo) << 3);\n"
	"	frac = c & 31;\n"
	"	int i = c >> 5;\n"
	"	if (mask != 0) {\n"
	"		if ((cm & 1) != 0 && ((i >> mask) & 1) != 0) i = ~i;\n"
	"		i &= (1 << mask) - 1;\n"
	"	}\n"
	"	return i;\n"
	"}\n"
	"vec4 tap(sampler2D t, ivec4 image, int i, int j) {\n"
	"	j = clamp(j, 0, image.y - 1);\n"
	"	if (image.w != 0) j %= image.w;\n"
	"	return texelFetch(t, ivec2(clamp(i, 0, image.x - 1), j), 0);\n"
	"}\n"

	// The RDP's divide (rdp.c texCoords): w's top 15 bits normalized, a table's reciprocal, the
	// products saturating outside 16 bits.
	"const ivec2 rcpTable[64] = ivec2[64](\n"
	"	ivec2(0x4000, -1008), ivec2(0x3f04, -976), ivec2(0x3e10, -952), ivec2(0x3d22, -920), ivec2(0x3c3c, -892), ivec2(0x3b5d, -872),\n"
	"	ivec2(0x3a83, -840), ivec2(0x39b1, -820), ivec2(0x38e4, -800), ivec2(0x381c, -776), ivec2(0x375a, -756), ivec2(0x369d, -736),\n"
	"	ivec2(0x35e5, -716), ivec2(0x3532, -700), ivec2(0x3483, -680), ivec2(0x33d9, -664), ivec2(0x3333, -648), ivec2(0x3291, -628),\n"
	"	ivec2(0x31f4, -620), ivec2(0x3159, -600), ivec2(0x30c3, -588), ivec2(0x3030, -572), ivec2(0x2fa1, -560), ivec2(0x2f15, -548),\n"
	"	ivec2(0x2e8c, -536), ivec2(0x2e06, -524), ivec2(0x2d83, -512), ivec2(0x2d03, -500), ivec2(0x2c86, -492), ivec2(0x2c0b, -480),\n"
	"	ivec2(0x2b93, -468), ivec2(0x2b1e, -460), ivec2(0x2aab, -452), ivec2(0x2a3a, -440), ivec2(0x29cc, -432), ivec2(0x2960, -424),\n"
	"	ivec2(0x28f6, -416), ivec2(0x288e, -408), ivec2(0x2828, -400), ivec2(0x27c4, -392), ivec2(0x2762, -384), ivec2(0x2702, -376),\n"
	"	ivec2(0x26a4, -368), ivec2(0x2648, -364), ivec2(0x25ed, -356), ivec2(0x2594, -348), ivec2(0x253d, -344), ivec2(0x24e7, -340),\n"
	"	ivec2(0x2492, -332), ivec2(0x243f, -324), ivec2(0x23ee, -320), ivec2(0x239e, -316), ivec2(0x234f, -308), ivec2(0x2302, -304),\n"
	"	ivec2(0x22b6, -296), ivec2(0x226c, -296), ivec2(0x2222, -288), ivec2(0x21da, -284), ivec2(0x2193, -280), ivec2(0x214d, -276),\n"
	"	ivec2(0x2108, -268), ivec2(0x20c5, -268), ivec2(0x2082, -260), ivec2(0x2041, -260));\n"
	"int rdpDivide(int st, int w, bool carry, int shift, int rcp) {\n"
	"	if (carry) return 0x7FFF;\n"
	"	int prod = int(uint(st) * uint(rcp));\n"
	"	int mask = ((1 << 30) - 1) & -((1 << 29) >> shift);\n"
	"	int outside = prod & mask;\n"
	"	if (outside != mask && outside != 0) {\n"
	"		int shifted = shift != 14 ? prod >> (13 - shift) : prod;\n"
	"		return (shifted & (1 << 29)) == 0 ? 0x7FFF : -0x8000;\n"
	"	}\n"
	"	return shift != 14 ? prod >> (13 - shift) : int(uint(prod) << 1);\n"
	"}\n"
	"ivec2 texCoord() {\n"
	"	vec3 top = vTex - 0.5 * dFdx(vTex) - 0.5 * dFdy(vTex);\n"
	"	int s, tt;\n"
	"	if (uPersp != 0) {\n"
	"		int w = int(floor(top.z * 32767.0));\n"
	"		bool carry = w <= 0;\n"
	"		w &= 0x7FFF;\n"
	"		int msb = 14;\n"
	"		while (msb >= 0 && (w >> msb) == 0) --msb;\n"
	"		int shift = min(14 - msb, 14);\n"
	"		int norm = (w << shift) & 0x3FFF;\n"
	"		ivec2 e = rcpTable[norm >> 8];\n"
	"		int rcp = ((e.y * (norm & 0xFF)) >> 10) + e.x;\n"
	"		s = rdpDivide(int(floor(top.x * 32.0 + 0.01)), w, carry, shift, rcp);\n"
	"		tt = rdpDivide(int(floor(top.y * 32.0 + 0.01)), w, carry, shift, rcp);\n"
	"	} else {\n"
	"		s = int(floor(top.x * 32.0 + 0.01)), tt = int(floor(top.y * 32.0 + 0.01));\n"
	"	}\n"
	"	return ivec2(clamp(s, -0x10000, 0xFFFF), clamp(tt, -0x10000, 0xFFFF));\n"
	"}\n"
	"vec4 sampleTile(sampler2D t, ivec4 a, ivec4 b, ivec4 c, ivec4 image, ivec2 st) {\n"
	"	int s = st.x, tt = st.y, fs, ft, f;\n"
	"	int s0 = wrapCoord(s, a.x, a.z, b.x, b.z, c.x, fs);\n"
	"	int t0 = wrapCoord(tt, a.y, a.w, b.y, b.w, c.y, ft);\n"
	"	vec4 c0 = tap(t, image, s0, t0);\n"
	"	if (c.z == 0) return c0;\n"
	"	int s1 = wrapCoord(s + (32 << (a.x < 11 ? a.x : 0)), a.x, a.z, b.x, b.z, c.x, f);\n"
	"	int t1 = wrapCoord(tt + (32 << (a.y < 11 ? a.y : 0)), a.y, a.w, b.y, b.w, c.y, f);\n"
	"	vec4 c1 = tap(t, image, s1, t0), c2 = tap(t, image, s0, t1);\n"
	"	float x = float(fs) / 32.0, y = float(ft) / 32.0;\n"
	"	if (fs + ft < 32) return c0 + (c1 - c0) * x + (c2 - c0) * y;\n"
	"	vec4 c3 = tap(t, image, s1, t1);\n"
	"	return c3 + (c2 - c3) * (1.0 - x) + (c1 - c3) * (1.0 - y);\n"
	"}\n";

// modes: cycle type (bits 0-1), alpha compare (2), textured (3), shaded (4), the RDP's dither in
// the shader (5) with the Bayer matrix (6) or noise (31) into a 16-bit image (7), the blender's first
// cycle and last cycle selections (8-15, 16-23), blending with memory (24), depth from prim (25),
// coverage times alpha with antialiasing (26), alpha as coverage (27), the last cycle's blend formula
// in the shader (28).
enum {
	MODE_CYCLE = 3, MODE_ALPHA_COMPARE = 4, MODE_TEXTURED = 8, MODE_SHADED = 16, MODE_DITHER = 32, MODE_BAYER = 64, MODE_IMAGE16 = 128, MODE_BLEND = 1 << 24, MODE_PRIM_DEPTH = 1 << 25,
	MODE_CVG_ALPHA = 1 << 26, MODE_ALPHA_CVG = 1 << 27, MODE_BLEND_COLOR = 1 << 28, MODE_BLEND_MEMORY_FIRST = 1 << 29, MODE_BLEND_CONSTANT = 1 << 30,
	MODE_NOISE = 1u << 31,
};

static const char* rgbInput(int which, int sel) {
	static const char* common[6] = { "combined.rgb", "tex0.rgb", "tex1.rgb", "uPrim.rgb", "shade.rgb", "uEnv.rgb" };
	if (sel < 6) {
		return common[sel];
	}
	if (which == 2) {
		static const char* mul[10] = { "vec3(0.0)", "vec3(combined.a)", "vec3(tex0.a)", "vec3(tex1.a)", "vec3(uPrim.a)", "vec3(shade.a)",
		                               "vec3(uEnv.a)", "vec3(lodFrac)", "vec3(uPrimLod)", "vec3(0.0)" };
		return sel < 16 ? mul[sel - 6] : "vec3(0.0)";
	}
	if (which == 0 && sel == 7) {
		return "noise()";
	}
	return (which == 0 || which == 3) && sel == 6 ? "vec3(1.0)" : "vec3(0.0)";
}

static const char* alphaInput(int sel, bool multiplier) {
	static const char* common[6] = { "combined.a", "tex0.a", "tex1.a", "uPrim.a", "shade.a", "uEnv.a" };
	if (sel == 0 && multiplier) {
		return "lodFrac";
	}
	if (sel < 6) {
		return common[sel];
	}
	return sel == 6 ? (multiplier ? "uPrimLod" : "1.0") : "0.0";
}

// An input as the combiner's integer: 0-255, and 256 for one.
static const char* input8(char* out, size_t size, const char* input) {
	if (!strcmp(input, "vec3(1.0)") || !strcmp(input, "1.0")) {
		snprintf(out, size, "%s", input[0] == 'v' ? "vec3(256.0)" : "256.0");
	} else {
		snprintf(out, size, "e8(%s)", input);
	}
	return out;
}

// One combiner cycle as GLSL, in the RDP's integers: ((a - b) * m + d * 256 + 128) / 256 for color and
// alpha, kept in 9 bits and saturated as the RDP does.
static void cycleSource(char* out, size_t size, uint64_t c, int cycle) {
	int a = cycle ? (c >> 37) & 15 : (c >> 52) & 15, m = cycle ? (c >> 32) & 31 : (c >> 47) & 31;
	int b = cycle ? (c >> 24) & 15 : (c >> 28) & 15, d = cycle ? (c >> 6) & 7 : (c >> 15) & 7;
	int aa = cycle ? (c >> 21) & 7 : (c >> 44) & 7, ab = cycle ? (c >> 3) & 7 : (c >> 12) & 7;
	int am = cycle ? (c >> 18) & 7 : (c >> 41) & 7, ad = cycle ? c & 7 : (c >> 9) & 7;
	char in[8][64];
	snprintf(out, size, "	combined = clamp9(floor((vec4((%s - %s) * %s, (%s - %s) * %s) + vec4(%s, %s) * 256.0 + 128.0) / 256.0)) / 255.0;\n",
	         input8(in[0], 64, rgbInput(0, a)), input8(in[1], 64, rgbInput(1, b)), input8(in[2], 64, rgbInput(2, m)),
	         input8(in[3], 64, alphaInput(aa, false)), input8(in[4], 64, alphaInput(ab, false)), input8(in[5], 64, alphaInput(am, true)),
	         input8(in[6], 64, rgbInput(3, d)), input8(in[7], 64, alphaInput(ad, false)));
}

// The blender's P and A for a cycle (pixel, memory, blend color, fog): P as the color and A as the
// alpha the GPU's blending then uses with memory.
// The blender's inputs for one cycle's selections (P | A << 2 | M << 4 | B << 6): the pixel, memory
// (which the shader can't read: the pixel stands in), the blend color or fog for P and M; the pixel
// alpha, fog alpha, shade alpha or 0 for A; 1 - A, memory alpha (taken as 1), 1 or 0 for B.
static const char* blendColorInput(int sel) {
	static const char* inputs[4] = { "combined.rgb", "combined.rgb", "uBlend.rgb", "uFog.rgb" };
	return inputs[sel & 3];
}

static const char* blendAlphaInput(int sel) {
	static const char* inputs[4] = { "combined.a", "uFog.a", "shade.a", "0.0" };
	return inputs[sel & 3];
}

// The blender's factors as the RDP takes them, from alpha a (scaled by keep): a's top 5 bits (fa) for
// P, and B's top 5 bits plus one (fb1) for M, over 32 (memory alpha taken as 1). With the pixel's alpha
// as A, 1 - A as B and the pixel opaque, P passes (reject).
static void blendFactors(char* out, size_t size, int sel, const char* keep) {
	static const char* b[4] = { "floor((255.0 - blendA8) / 8.0) + 1.0", "32.0", "32.0", "1.0" };
	bool rejects = !((sel >> 2) & 3) && !((sel >> 6) & 3);
	snprintf(out, size, "float blendA8 = floor(%s * %s * 255.0 + 0.5); float fa = floor(blendA8 / 8.0), fb1 = %s; float reject = %s; "
	         "float blendA = mix(fa / 32.0, 1.0, reject), blendB = mix(fb1 / 32.0, 0.0, reject);",
	         blendAlphaInput(sel >> 2), keep, b[(sel >> 6) & 3], rejects ? "blendA8 >= 255.0 ? 1.0 : 0.0" : "0.0");
}

// P * A + M * B into combined.rgb, in the RDP's integers.
static void blendFormula(char* out, size_t size, int sel) {
	char factors[384];
	blendFactors(factors, sizeof(factors), sel, "1.0");
	snprintf(out, size, "	{ %s combined.rgb = mix(clamp(floor((e8(%s) * fa + e8(%s) * fb1) / 32.0), 0.0, 255.0) / 255.0, %s, reject); }\n", factors,
	         blendColorInput(sel), blendColorInput(sel >> 4), blendColorInput(sel));
}

static GLuint compile(struct Glr* g, GLenum type, const char* source) {
	struct Gl* gl = &g->gl;
	GLuint s = gl->CreateShader(type);
	gl->ShaderSource(s, 1, &source, NULL);
	gl->CompileShader(s);
	GLint ok = 0;
	gl->GetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[1024];
		gl->GetShaderInfoLog(s, sizeof(log), NULL, log);
		fprintf(stderr, "GL renderer: shader: %s\n", log);
		gl->DeleteShader(s);
		return 0;
	}
	return s;
}

static struct Program* program(struct Glr* g, struct ProgramKey key) {
	for (int i = 0; i < g->programCount; ++i) {
		if (g->programs[i].key.combine == key.combine && g->programs[i].key.modes == key.modes && g->programs[i].key.lod == key.lod) {
			return &g->programs[i];
		}
	}
	if (g->programCount == PROGRAMS) {
		return NULL;
	}
	struct Gl* gl = &g->gl;
	static char source[8192];
	char line[1024];
	int cycleType = key.modes & MODE_CYCLE;
	snprintf(source, sizeof(source), "%s", fragmentHead);
	// rdp.c's noise: a hash of the pixel (at the RDP's resolution), the primitive's number and what
	// it is for (the combiner's channels 0-2, the dither 4).
	snprintf(line, sizeof(line),
	         "uint pixelNoise(uint tag) {\n"
	         "	uvec2 p = uvec2(gl_FragCoord.xy) / %du;\n"
	         "	uint h = p.x * 0x9E3779B1u ^ p.y * 0x85EBCA77u ^ vSeq * 0xC2B2AE3Du ^ tag;\n"
	         "	h ^= h >> 15;\n"
	         "	h *= 0x2C1B3C6Du;\n"
	         "	return h ^ (h >> 12);\n"
	         "}\n"
	         "vec3 noise() { return vec3(uvec3(pixelNoise(0u), pixelNoise(1u), pixelNoise(2u)) & 255u) / 255.0; }\n", g->scale);
	strcat(source, line);
	strcat(source, "void main() {\n	vec4 combined = vec4(0.0);\n");
	strcat(source, key.modes & MODE_SHADED ? "	vec4 shade = vShade;\n" : "	vec4 shade = vec4(0.0);\n");
	strcat(source, "	float lodFrac = 0.0;\n");
	if (key.modes & MODE_TEXTURED) {
		strcat(source, "	ivec2 st = texCoord();\n	vec4 tex0 = sampleTile(uTex0, uTile0[0], uTile0[1], uTile0[2], uImage0, st);\n");
		bool second = cycleType == 1 || (key.lod & 2);
		strcat(source, second ? "	vec4 tex1 = sampleTile(uTex1, uTile1[0], uTile1[1], uTile1[2], uImage1, st);\n" : "	vec4 tex1 = tex0;\n");
		if (key.lod & 1) {
			// rdp.c levelOfDetail: the texel step to the next pixel and line, its level, the fraction
			// between levels and the tiles; this has the primitive's tile and the next one.
			int maxLevel = (int) (key.lod >> 4) & 7, minLevel = (int) (key.lod >> 8) & 31;
			bool sharpen = key.lod & 4, detail = key.lod & 8;
			snprintf(line, sizeof(line),
			         "	{ vec2 fst = vec2(st);\n"
			         "	int l = int(max(max(abs(dFdx(fst.x)), abs(dFdx(fst.y))), max(abs(dFdy(fst.x)), abs(dFdy(fst.y)))));\n"
			         "	l = l >= 0x4000 ? (l & 0x3FFF) | 0x4000 : l;\n"
			         "	bool magnify = l < 32;\n"
			         "	int level = 0;\n"
			         "	for (int v = (l >> 5) & 0xFF; !magnify && v > 1; v >>= 1) ++level;\n"
			         "	bool distant = (l & 0x4000) != 0 || %d == 0 || (!magnify && ((l & 0x6000) != 0 || level >= %d));\n"
			         "	int frac = (l & 0x4000) != 0 ? 0xFF : magnify ? %s : %s;\n"
			         "	lodFrac = float(frac) / 255.0;\n",
			         maxLevel, maxLevel, !sharpen && !detail ? "(distant ? 0xFF : 0)" : sharpen ? "(max(l, %d) << 3) | 0x100" : "max(l, %d) << 3",
			         !sharpen && !detail ? "(distant ? 0xFF : ((l << 3) >> level) & 0xFF)" : "((l << 3) >> level) & 0xFF");
			if (sharpen || detail) {
				// The minimum level goes where the format has it.
				char with[1024];
				snprintf(with, sizeof(with), line, minLevel);
				snprintf(line, sizeof(line), "%s", with);
			}
			strcat(source, line);
			if ((key.lod & 2) && !detail) {
				snprintf(line, sizeof(line),
				         "	int t1 = distant ? %d : level, t2 = distant || %s ? t1 : t1 + 1;\n"
				         "	vec4 base = tex0, next = tex1;\n"
				         "	tex0 = t1 == 0 ? base : next;\n"
				         "	tex1 = t2 == 0 ? base : next;\n",
				         maxLevel, sharpen ? "false" : "magnify");
				strcat(source, line);
			}
			strcat(source, "	}\n");
		}
	} else {
		strcat(source, "	vec4 tex0 = vec4(0.0), tex1 = vec4(0.0);\n");
	}
	if (cycleType == 2) {
		// Copy: the texel as it is.
		strcat(source, "	combined = tex0;\n");
	} else {
		if (cycleType == 1) {
			cycleSource(line, sizeof(line), key.combine, 0);
			strcat(source, line);
			// The second cycle sees the pipelined texels: texel 1 as texel 0, and texel 0 as texel 1
			// (rdp.c's shadePixel).
			strcat(source, "	vec4 swapped = tex0;\n	tex0 = tex1;\n	tex1 = swapped;\n");
		}
		cycleSource(line, sizeof(line), key.combine, 1);
		strcat(source, line);
	}
	if (key.modes & MODE_CVG_ALPHA) {
		// Coverage times alpha: no coverage left, no pixel.
		strcat(source, "	if (combined.a < 0.125) discard;\n");
	}
	if (key.modes & MODE_ALPHA_CVG) {
		// Alpha as coverage: full inside the primitive.
		strcat(source, "	combined.a = 1.0;\n");
	}
	if (key.modes & MODE_ALPHA_COMPARE) {
		// Copy mode drops texels with no alpha, the other modes those below the blend alpha.
		strcat(source, cycleType == 2 ? "	if (combined.a == 0.0) discard;\n" : "	if (combined.a < uBlend.a) discard;\n");
	}
	int first = (key.modes >> 8) & 0xFF, last = (key.modes >> 16) & 0xFF;
	// The first of two cycles always blends (fog, mostly). With memory as its M (and 1 - A as B), the
	// image's share carries to the last cycle's blend with it: P * a1 * a2 + memory * (1 - a1 * a2).
	strcat(source, "	float memoryKeep = 1.0;\n");
	if (cycleType == 1 && ((first >> 4) & 3) == 1 && !((first >> 6) & 3)) {
		snprintf(line, sizeof(line), "	memoryKeep = %s;\n	combined.rgb = %s;\n", blendAlphaInput(first >> 2), blendColorInput(first));
		strcat(source, line);
	} else if (cycleType == 1) {
		blendFormula(line, sizeof(line), first);
		strcat(source, line);
	}
	if (key.modes & MODE_DITHER) {
		// The matrix's value for all three channels, or three bits of noise for each.
		char matrix[256];
		if (key.modes & MODE_NOISE) {
			snprintf(matrix, sizeof(matrix), "uint h = pixelNoise(4u);\n	ivec3 ditherM = ivec3(uvec3(h, h >> 3, h >> 6) & 7u)");
		} else {
			snprintf(matrix, sizeof(matrix), "ivec3 ditherM = ivec3(%s[(px.y & 3) * 4 + (px.x & 3)])",
			         key.modes & MODE_BAYER ? "int[16](0, 4, 1, 5, 4, 0, 5, 1, 3, 7, 2, 6, 7, 3, 6, 2)" : "int[16](0, 6, 1, 7, 4, 2, 5, 3, 3, 5, 2, 4, 7, 1, 6, 0)");
		}
		snprintf(line, sizeof(line), "	ivec2 px = ivec2(gl_FragCoord.xy) / %d;\n	%s;\n	bool dithered = %s;\n", g->scale, matrix,
		         key.modes & MODE_BLEND ? "false" : "true");
		strcat(source, line);
	}
	if (cycleType < 2) {
		// The last cycle: with force blend and memory as M the GPU blends P with the image, other
		// inputs here; without, P passes.
		if (key.modes & MODE_BLEND_COLOR) {
			blendFormula(line, sizeof(line), last);
			strcat(source, line);
			snprintf(line, sizeof(line), "	oColor = vec4(combined.rgb, %s);\n", blendAlphaInput(last >> 2));
		} else if (key.modes & MODE_BLEND_MEMORY_FIRST) {
			// Memory as P: M * B here, the GPU adds the image times A (the blend constant when A is).
			char factors[384];
			blendFactors(factors, sizeof(factors), last, "1.0");
			snprintf(line, sizeof(line), "	{ %s oColor = vec4(%s * blendB, %s); }\n", factors, blendColorInput(last >> 4),
			         key.modes & MODE_BLEND_CONSTANT ? "1.0" : "blendA");
		} else if (key.modes & MODE_BLEND) {
			// The GPU blends P with the image: P * A here (a little low, so that rounding takes it
			// down as the RDP does), B as the destination's factor from A itself (the RDP's extra
			// 32nd of the image, rounded up, would show on dark colors over light ones).
			char factors[384];
			blendFactors(factors, sizeof(factors), last, "memoryKeep");
			snprintf(line, sizeof(line), "	{ %s oColor = vec4(%s * blendA - (1.0 - reject) * 0.49 / 255.0, mix(blendA8 / 255.0, 1.0, reject));\n", factors,
			         blendColorInput(last));
			strcat(source, line);
			// Where the GPU's blend leaves the image none of its own (B zero, or 1 - A with A whole: no
			// blend), the color is the shader's to dither.
			int keep = (last >> 6) & 3;
			if ((key.modes & MODE_DITHER) && (keep == 3 || keep == 0)) {
				snprintf(line, sizeof(line), "	dithered = %s;\n", keep == 3 ? "true" : "reject == 1.0");
				strcat(source, line);
			}
			snprintf(line, sizeof(line), "	}\n");
		} else {
			snprintf(line, sizeof(line), "	oColor = vec4(%s, %s * memoryKeep);\n", blendColorInput(last), blendAlphaInput(last >> 2));
		}
		strcat(source, line);
	} else {
		strcat(source, "	oColor = combined;\n");
	}
	if (key.modes & MODE_DITHER) {
		// rdp.c's dither: a channel whose low 3 bits are above the matrix's value rounds up; a 16-bit
		// image keeps 5 bits, as the RDP's would read back.
		strcat(source, "	if (dithered) { ivec3 o = ivec3(floor(oColor.rgb * 255.0 + 0.5));\n"
		               "	for (int i = 0; i < 3; ++i) if ((o[i] & 7) > ditherM[i]) o[i] = min((o[i] & 0xF8) + 8, 255);\n");
		strcat(source, key.modes & MODE_IMAGE16 ? "	o = (o >> 3) << 3 | o >> 5;\n	oColor.rgb = vec3(o) / 255.0; }\n" : "	oColor.rgb = vec3(o) / 255.0; }\n");
	}
	if (!(key.modes & MODE_BLEND) && (key.modes & (MODE_BLEND_MEMORY_FIRST | MODE_BLEND_CONSTANT)) != MODE_BLEND_MEMORY_FIRST) {
		// Alpha in the image marks the pixels drawn (blending adds the pixel's).
		strcat(source, "	oColor.a = 1.0;\n");
	}
	if (key.modes & MODE_PRIM_DEPTH) {
		strcat(source, "	gl_FragDepth = uPrimDepth;\n");
	}
	strcat(source, "}\n");
	GLuint fs = compile(g, GL_FRAGMENT_SHADER, source);
	if (!fs) {
		return NULL;
	}
	GLuint p = gl->CreateProgram();
	gl->AttachShader(p, g->vertexShader);
	gl->AttachShader(p, fs);
	gl->LinkProgram(p);
	gl->DeleteShader(fs);
	GLint ok = 0;
	gl->GetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[1024];
		gl->GetProgramInfoLog(p, sizeof(log), NULL, log);
		fprintf(stderr, "GL renderer: program: %s\n", log);
		gl->DeleteProgram(p);
		return NULL;
	}
	struct Program* pr = &g->programs[g->programCount++];
	pr->key = key;
	pr->program = p;
	pr->size = gl->GetUniformLocation(p, "uSize");
	pr->persp = gl->GetUniformLocation(p, "uPersp");
	pr->prim = gl->GetUniformLocation(p, "uPrim");
	pr->env = gl->GetUniformLocation(p, "uEnv");
	pr->fog = gl->GetUniformLocation(p, "uFog");
	pr->blend = gl->GetUniformLocation(p, "uBlend");
	pr->primLod = gl->GetUniformLocation(p, "uPrimLod");
	pr->primDepth = gl->GetUniformLocation(p, "uPrimDepth");
	pr->tile[0] = gl->GetUniformLocation(p, "uTile0");
	pr->tile[1] = gl->GetUniformLocation(p, "uTile1");
	pr->image[0] = gl->GetUniformLocation(p, "uImage0");
	pr->image[1] = gl->GetUniformLocation(p, "uImage1");
	pr->texture[0] = gl->GetUniformLocation(p, "uTex0");
	pr->texture[1] = gl->GetUniformLocation(p, "uTex1");
	gl->UseProgram(p);
	g->applied.program = p;
	gl->Uniform1i(pr->texture[0], 0);
	gl->Uniform1i(pr->texture[1], 1);
	return pr;
}

// ---- Color images

static void writePixels(struct N64* n, struct Target* t, int from, int to);

// The RDRAM pixels of rows from to to - 1, into the texture, with alpha 0: drawing sets it, and only
// pixels with it go back.
static void loadPixels(struct N64* n, struct Glr* g, struct Target* t, int from, int to) {
	n->stats.glTargetRows += (uint64_t) (to - from);
	uint32_t* out = g->image;
	for (int y = from; y < to; ++y) {
		for (uint32_t x = 0; x < t->width; ++x) {
			uint32_t i = y * t->width + x, c;
			if (t->size == 3) {
				const uint8_t* p = &n->rdram[(t->addr + i * 4) & (RDRAM_SIZE - 4)];
				c = (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16;
			} else if (t->size == 2) {
				uint32_t a = (t->addr + i * 2) & (RDRAM_SIZE - 2);
				uint16_t v = (uint16_t) (n->rdram[a] << 8 | n->rdram[a + 1]);
				int r = (v >> 11) & 31, gr = (v >> 6) & 31, b = (v >> 1) & 31;
				c = (uint32_t) (r << 3 | r >> 2) | (uint32_t) (gr << 3 | gr >> 2) << 8 | (uint32_t) (b << 3 | b >> 2) << 16;
			} else {
				uint8_t v = n->rdram[(t->addr + i) & (RDRAM_SIZE - 1)];
				c = v | (uint32_t) v << 8 | (uint32_t) v << 16;
			}
			out[(y - from) * t->width + x] = c;
		}
	}
	struct Gl* gl = &g->gl;
	uploadTexture(g, g->scale > 1 ? g->native : t->texture);
	gl->TexSubImage2D(GL_TEXTURE_2D, 0, 0, from, (GLsizei) t->width, to - from, GL_RGBA, GL_UNSIGNED_BYTE, out);
	if (g->scale > 1) {
		blitRows(g, t, from, to, true);
	}
}

static void flushBatch(struct N64* n, struct Glr* g);

// The target for a color image, its texture loaded from RDRAM (rows) when the GPU last gave its
// pixels back.
static struct Target* target(struct N64* n, struct Glr* g, uint32_t addr, uint32_t width, int size, int rows) {
	struct Target* t = NULL;
	for (int i = 0; i < TARGETS; ++i) {
		if (g->target[i].texture && g->target[i].addr == addr && g->target[i].width == width && g->target[i].size == size) {
			t = &g->target[i];
			break;
		}
	}
	struct Gl* gl = &g->gl;
	if (!t) {
		t = &g->target[0];
		for (int i = 0; i < TARGETS; ++i) {
			if (!g->target[i].texture || g->target[i].used < t->used) {
				t = &g->target[i];
				if (!t->texture) {
					break;
				}
			}
		}
		if (t->texture) {
			flushBatch(n, g);
			if (t->dirtyTo > t->dirtyFrom) {
				writePixels(n, t, t->dirtyFrom, t->dirtyTo);
			}
		} else {
			gl->GenTextures(1, &t->texture);
			gl->GenFramebuffers(1, &t->framebuffer);
		}
		uploadTexture(g, t->texture);
		gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei) width * g->scale, TARGET_ROWS * g->scale, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		useFramebuffer(g, t->framebuffer);
		gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->texture, 0);
		gl->FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, g->depth);
		t->addr = addr, t->width = width, t->size = size;
		t->loadedRows = 0;
		t->dirtyFrom = t->dirtyTo = 0;
	}
	t->used = ++g->clock;
	rows = rows > TARGET_ROWS ? TARGET_ROWS : rows;
	// Another color image over these rows (the same address at another width, say): what it drew
	// goes back to RDRAM first, and both load again from there.
	uint32_t rowBytes = width * (size == 3 ? 4 : size == 1 ? 1 : 2), from = addr, to = addr + (uint32_t) rows * rowBytes;
	for (int i = 0; i < TARGETS; ++i) {
		struct Target* o = &g->target[i];
		uint32_t oRow = o->width * (o->size == 3 ? 4 : o->size == 1 ? 1 : 2);
		if (o == t || !o->texture || o->dirtyTo <= o->dirtyFrom || o->addr + o->dirtyFrom * oRow >= to || from >= o->addr + o->dirtyTo * oRow) {
			continue;
		}
		flushBatch(n, g);
		writePixels(n, o, o->dirtyFrom, o->dirtyTo);
		o->dirtyFrom = o->dirtyTo = 0;
		o->loadedRows = 0;
		if (t->dirtyTo > t->dirtyFrom) {
			writePixels(n, t, t->dirtyFrom, t->dirtyTo);
			t->dirtyFrom = t->dirtyTo = 0;
		}
		t->loadedRows = 0;
	}
	if (rows > t->loadedRows) {
		loadPixels(n, g, t, t->loadedRows, rows);
		t->loadedRows = rows;
	}
	return t;
}

// The GPU's rows from to to - 1 back to RDRAM, as the RDP would store them (16-bit pixels fully
// covered).
static void writePixels(struct N64* n, struct Target* t, int from, int to) {
	struct Glr* g = n->glr;
	struct Gl* gl = &g->gl;
	uint32_t bytes = t->size == 3 ? 4 : t->size == 1 ? 1 : 2;
	uint32_t start = t->addr + (uint32_t) from * t->width * bytes;
	// First, as it may give back another target's rows (through g->image).
	n64RdramWrite(n, start & (RDRAM_SIZE - 1), (uint32_t) (to - from) * t->width * bytes);
	if (g->scale > 1) {
		blitRows(g, t, from, to, false);
	} else {
		useFramebuffer(g, t->framebuffer);
	}
	gl->ReadPixels(0, from, (GLsizei) t->width, to - from, GL_RGBA, GL_UNSIGNED_BYTE, g->image);
	n->stats.glRows += (uint64_t) (to - from);
	for (int y = from; y < to; ++y) {
		for (uint32_t x = 0; x < t->width; ++x) {
			uint32_t c = g->image[(y - from) * t->width + x], i = y * t->width + x;
			if (!(c >> 24)) {
				continue;
			}
			int r = c & 255, gr = (c >> 8) & 255, b = (c >> 16) & 255;
			if (t->size == 3) {
				uint8_t* p = &n->rdram[(t->addr + i * 4) & (RDRAM_SIZE - 4)];
				p[0] = (uint8_t) r, p[1] = (uint8_t) gr, p[2] = (uint8_t) b, p[3] = 0xE0;
			} else if (t->size == 2) {
				uint32_t a = (t->addr + i * 2) & (RDRAM_SIZE - 2);
				uint16_t v = (uint16_t) ((r >> 3) << 11 | (gr >> 3) << 6 | (b >> 3) << 1 | 1);
				n->rdram[a] = (uint8_t) (v >> 8), n->rdram[a + 1] = (uint8_t) v;
				n->hidden[a >> 1] = 3;
			} else {
				n->rdram[(t->addr + i) & (RDRAM_SIZE - 1)] = (uint8_t) r;
			}
		}
	}
}

// ---- Textures

static uint64_t hashBytes(uint64_t h, const uint8_t* p, size_t len) {
	for (size_t i = 0; i < len; ++i) {
		h = (h ^ p[i]) * 0x100000001B3ull;
	}
	return h;
}

static uint64_t mix(uint64_t h, const uint8_t* tmem, uint32_t at) {
	uint64_t v;
	memcpy(&v, &tmem[at & 0xFF8], 8);
	h = (h ^ v) * 0x100000001B3ull;
	return h ^ (h >> 29);
}

// TMEM from byte from, len bytes (wrapping), eight at a time in four lanes.
static uint64_t hashTmem(uint64_t h, const uint8_t* tmem, uint32_t from, uint32_t len) {
	uint64_t a = h, b = h ^ 1, c = h ^ 2, d = h ^ 3;
	uint32_t i = 0;
	for (; i + 32 <= len; i += 32) {
		a = mix(a, tmem, from + i), b = mix(b, tmem, from + i + 8);
		c = mix(c, tmem, from + i + 16), d = mix(d, tmem, from + i + 24);
	}
	for (; i < len; i += 8) {
		a = mix(a, tmem, from + i);
	}
	return ((((a * 0x100000001B3ull) ^ b) * 0x100000001B3ull ^ c) * 0x100000001B3ull ^ d) * 0x100000001B3ull;
}

// The texture for a tile: its texels (as many as its mask or clamp reaches) decoded from TMEM,
// found again by what they were made from.
static struct Texture* texture(struct N64* n, struct Glr* g, struct RDP* r, int index, bool convert, int* width, int* height) {
	const struct Tile* t = &r->tile[index & 7];
	uint32_t tlutMode = (uint32_t) (r->otherModes >> 46) & 3;
	struct Texture* known = g->tileTexture[index & 7];
	if (!convert && known && known->texture && g->tileGen[index & 7] == r->tmemGen && g->tileTlut[index & 7] == tlutMode) {
		*width = known->width, *height = known->height;
		known->used = ++g->clock;
		return known;
	}
	int w = t->maskS ? 1 << t->maskS : ((t->sh - t->sl) >> 2) + 1;
	int h = t->maskT ? 1 << t->maskT : ((t->th - t->tl) >> 2) + 1;
	w = w < 1 ? 1 : w > 1024 ? 1024 : w;
	h = h < 1 ? 1 : h > 1024 ? 1024 : h;
	// What the texels are decoded from (the tile's position, wrapping and clamping are the shader's,
	// so a scrolling tile keeps its texture).
	int decode[5] = { t->format, t->size, t->line, t->tmem, t->palette };
	uint64_t key = hashBytes(0xCBF29CE484222325ull, (const uint8_t*) decode, sizeof(decode));
	key = hashBytes(key, (const uint8_t*) &w, sizeof(w));
	key = hashBytes(key, (const uint8_t*) &h, sizeof(h));
	uint64_t tlut = (r->otherModes >> 46) & 3;
	key = hashBytes(key, (const uint8_t*) &tlut, sizeof(tlut));
	if (convert) {
		int k[4] = { r->k0, r->k1, r->k2, r->k3 };
		key = hashBytes(key, (const uint8_t*) k, sizeof(k));
	}
	// The bytes the tile's texels come from: its lines (both halves for 32-bit and YUV texels) and
	// for color-indexed ones the palette.
	uint32_t from = (uint32_t) t->tmem * 8, len = (uint32_t) (t->line ? t->line : 1) * 8 * (uint32_t) h;
	len = len > 0x1000 ? 0x1000 : len;
	key = hashTmem(key, r->tmem, from, len);
	if (t->size >= 2 && (t->size == 3 || t->format == 1)) {
		key = hashTmem(key, r->tmem, from | 0x800, len);
	}
	if (t->format == 2 || ((r->otherModes >> 47) & 1)) {
		key = t->size == 0 ? hashTmem(key, r->tmem, 0x800 + (uint32_t) t->palette * 128, 128) : hashTmem(key, r->tmem, 0x800, 0x800);
	}
	*width = w, *height = h;
	int* bucket = &g->textureBucket[key & (TEXTURE_BUCKETS - 1)];
	for (int i = *bucket; i; i = g->textures[i - 1].next) {
		struct Texture* x = &g->textures[i - 1];
		if (x->hash == key) {
			x->used = ++g->clock;
			g->tileTexture[index & 7] = x, g->tileGen[index & 7] = r->tmemGen, g->tileTlut[index & 7] = tlutMode;
			return x;
		}
	}
	// A new texture: in an unused slot or the one used longest ago, taken out of its bucket.
	struct Texture* slot = &g->textures[0];
	for (int i = 0; i < TEXTURES; ++i) {
		struct Texture* x = &g->textures[i];
		if (!x->texture || x->used < slot->used) {
			slot = x;
			if (!x->texture) {
				break;
			}
		}
	}
	if (slot->texture) {
		int self = (int) (slot - g->textures) + 1;
		for (int* link = &g->textureBucket[slot->hash & (TEXTURE_BUCKETS - 1)]; *link; link = &g->textures[*link - 1].next) {
			if (*link == self) {
				*link = slot->next;
				break;
			}
		}
	}
	struct Gl* gl = &g->gl;
	// The slot may be in the batch waiting to be drawn.
	if (g->batchCount && (g->state.texture[0] == slot || g->state.texture[1] == slot)) {
		flushBatch(n, g);
	}
	// A row's texels depend on it through its TMEM address (line * 8 bytes a row, wrapping at 4 KiB, or
	// 2 KiB for the halves 32-bit and YUV texels use) and whether it is odd: rows repeat after that.
	uint32_t wrap = t->size == 3 || (t->size == 2 && t->format == 1) ? 0x800 : 0x1000, common = wrap;
	for (uint32_t x = (uint32_t) t->line * 8 % wrap, y = wrap; y;) {
		uint32_t z = x % y;
		x = y, y = z;
		common = x;
	}
	int period = (int) (wrap / common);
	period += period & 1;
	int rows = period < h ? period : h;
	bool same = slot->texture && slot->width == w && (slot->period ? slot->period : slot->height) == rows;
	if (!slot->texture) {
		gl->GenTextures(1, &slot->texture);
	}
	uint32_t* pixels = malloc((size_t) w * rows * 4);
	if (!pixels) {
		return NULL;
	}
	rdpTileImage(r, index, w, rows, convert, pixels);
	++n->stats.glTextures;
	uploadTexture(g, slot->texture);
	if (same) {
		// The same size: the texels replaced in place (no new storage).
		gl->TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, rows, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	} else {
		gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}
	free(pixels);
	slot->hash = key, slot->width = w, slot->height = h, slot->period = rows < h ? rows : 0, slot->used = ++g->clock;
	slot->next = *bucket, *bucket = (int) (slot - g->textures) + 1;
	g->tileTexture[index & 7] = slot, g->tileGen[index & 7] = r->tmemGen, g->tileTlut[index & 7] = tlutMode;
	return slot;
}

// ---- Drawing

static void colorOf(const struct Color* c, float* out) {
	out[0] = c->r / 255.0f, out[1] = c->g / 255.0f, out[2] = c->b / 255.0f, out[3] = c->a / 255.0f;
}

static void flushBatch(struct N64* n, struct Glr* g) {
	if (!g->batchCount || !g->stateSet) {
		g->batchCount = 0;
		return;
	}
	struct Gl* gl = &g->gl;
	struct DrawState* s = &g->state;
	struct Program* p = s->program;
	struct Applied* a = &g->applied;
	useFramebuffer(g, s->target->framebuffer);
	if (a->viewport != (int) s->target->width) {
		gl->Viewport(0, 0, (GLsizei) s->target->width * g->scale, TARGET_ROWS * g->scale);
		a->viewport = (int) s->target->width;
	}
	useScissor(g, true, s->scissor[0], s->scissor[1], s->scissor[2] - s->scissor[0], s->scissor[3] - s->scissor[1]);
	if (a->program != p->program) {
		gl->UseProgram(p->program);
		a->program = p->program;
	}
	struct Uniforms* u = &p->set;
	if (!p->uniformsSet) {
		memset(u, 0xFF, sizeof(*u));
		p->uniformsSet = true;
	}
	const float size[2] = { (float) s->target->width, (float) TARGET_ROWS };
	uniform(g, p->size, 2, size, u->size);
	if (u->persp != s->persp) {
		gl->Uniform1i(p->persp, s->persp);
		u->persp = s->persp;
	}
	uniform(g, p->prim, 4, s->prim, u->prim);
	uniform(g, p->env, 4, s->env, u->env);
	uniform(g, p->fog, 4, s->fog, u->fog);
	uniform(g, p->blend, 4, s->blend, u->blend);
	uniform(g, p->primLod, 1, &s->primLod, &u->primLod);
	uniform(g, p->primDepth, 1, &s->primDepth, &u->primDepth);
	for (int k = 0; k < 2; ++k) {
		if (s->texture[k]) {
			useTexture(g, k, s->texture[k]->texture);
			if (memcmp(u->tile[k], s->tile[k], sizeof(u->tile[k]))) {
				gl->Uniform4iv(p->tile[k], 3, s->tile[k]);
				memcpy(u->tile[k], s->tile[k], sizeof(u->tile[k]));
			}
			int image[4] = { s->texture[k]->width, s->texture[k]->height, 0, s->texture[k]->period };
			if (memcmp(u->image[k], image, sizeof(image))) {
				gl->Uniform4iv(p->image[k], 1, image);
				memcpy(u->image[k], image, sizeof(image));
			}
		}
	}
	// The RDP passes what is in front (decals what is no farther, within the depth slope), and
	// anything on a cleared depth.
	int depth = s->depthLess ? 3 : s->depthTest ? 1 : s->depthWrite ? 2 : 0;
	if (a->depth != depth) {
		depth ? gl->Enable(GL_DEPTH_TEST) : gl->Disable(GL_DEPTH_TEST);
		if (depth) {
			gl->DepthFunc(depth == 1 ? GL_LEQUAL : depth == 3 ? GL_LESS : GL_ALWAYS);
		}
		a->depth = depth;
	}
	useDepthMask(g, s->depthWrite);
	if (a->offset != s->decal) {
		if (s->decal) {
			gl->Enable(GL_POLYGON_OFFSET_FILL);
			gl->PolygonOffset(-1, -1);
		} else {
			gl->Disable(GL_POLYGON_OFFSET_FILL);
		}
		a->offset = s->decal;
	}
	int blend = s->blendOn ? s->blendB : s->memoryFirst ? 3 + s->memoryFirst : -1;
	if (a->blend != blend) {
		if (blend < 0) {
			gl->Disable(GL_BLEND);
		} else if (blend >= 4) {
			gl->Enable(GL_BLEND);
			gl->BlendFuncSeparate(GL_ONE, blend == 4 ? GL_SRC_ALPHA : GL_CONSTANT_ALPHA, GL_ONE, GL_ONE);
		} else {
			gl->Enable(GL_BLEND);
			gl->BlendFuncSeparate(GL_ONE, blend == 0 ? GL_ONE_MINUS_SRC_ALPHA : blend == 3 ? GL_ZERO : GL_ONE, GL_ONE, GL_ONE);
		}
		a->blend = blend;
	}
	if (blend == 5 && a->blendConstant != s->blendConstant) {
		gl->BlendColor(0, 0, 0, s->blendConstant);
		a->blendConstant = s->blendConstant;
	}
	gl->BindVertexArray(g->vertexArray);
	gl->BindBuffer(GL_ARRAY_BUFFER, g->vertexBuffer);
	// Batches go one after another into one vertex buffer, new storage only when it is full (the
	// GPU may still be reading the old: a buffer a batch costs the kernel page work).
	size_t bytes = (size_t) g->batchCount * VERTEX_FLOATS * 4;
	if (g->vertexOffset + bytes > VERTEX_BUFFER_BYTES) {
		gl->BufferData(GL_ARRAY_BUFFER, VERTEX_BUFFER_BYTES, NULL, GL_STREAM_DRAW);
		g->vertexOffset = 0;
	}
	void* to = gl->MapBufferRange(GL_ARRAY_BUFFER, (GLintptr) g->vertexOffset, (GLsizeiptr) bytes, GL_MAP_WRITE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
	if (to) {
		memcpy(to, g->batch, bytes);
		gl->UnmapBuffer(GL_ARRAY_BUFFER);
		GLint first = (GLint) (g->vertexOffset / (VERTEX_FLOATS * 4));
		if (s->decal) {
			// The RDP's decal passes only within the depth slope either way (rdp.c, z mode 3): first
			// the pixels no more than that in front marked, then those no more than that behind drawn.
			if (++g->stencilRef > 255) {
				useScissor(g, false, 0, 0, 0, 0);
				gl->ClearStencil(0);
				gl->Clear(GL_STENCIL_BUFFER_BIT);
				useScissor(g, true, s->scissor[0], s->scissor[1], s->scissor[2] - s->scissor[0], s->scissor[3] - s->scissor[1]);
				g->stencilRef = 1;
			}
			gl->Enable(GL_STENCIL_TEST);
			gl->ColorMask(0, 0, 0, 0);
			gl->DepthMask(0);
			gl->DepthFunc(GL_GEQUAL);
			gl->PolygonOffset(1, 1);
			gl->StencilFunc(GL_ALWAYS, g->stencilRef, 0xFF);
			gl->StencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
			gl->DrawArrays(GL_TRIANGLES, first, g->batchCount);
			gl->ColorMask(1, 1, 1, 1);
			gl->DepthMask(s->depthWrite);
			gl->DepthFunc(GL_LEQUAL);
			gl->PolygonOffset(-1, -1);
			gl->StencilFunc(GL_EQUAL, g->stencilRef, 0xFF);
			gl->StencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
			gl->DrawArrays(GL_TRIANGLES, first, g->batchCount);
			gl->Disable(GL_STENCIL_TEST);
		} else {
			gl->DrawArrays(GL_TRIANGLES, first, g->batchCount);
		}
		++n->stats.glBatches;
		g->vertexOffset += bytes;
	}
	g->batchCount = 0;
}

// The state a primitive draws with, from the RDP's; the batch so far goes first when it differs.
static bool drawRows(struct N64* n, struct Glr* g, struct RDP* r, int rows, int y0, int y1);

static bool drawState(struct N64* n, struct Glr* g, struct RDP* r, bool textured, bool shaded, bool depth, int tile, bool persp,
                      int y0, int y1) {
	if (r->colorSize < 1) {
		return false;
	}
	// No command but primitives since the last: the same state.
	int primitive = textured | shaded << 1 | depth << 2 | tile << 3 | persp << 6 | r->maxLevel << 7;
	if (g->fastValid && g->stateSet && g->fastGen == r->stateGen && g->fastKey == primitive) {
		return drawRows(n, g, r, g->fastRows, y0, y1);
	}
	struct DrawState s;
	memset(&s, 0, sizeof(s));
	int rows = (r->scissorY1 + 3) >> 2;
	rows = rows < 1 ? 1 : rows > TARGET_ROWS ? TARGET_ROWS : rows;
	if (g->stateSet && g->state.target && (g->state.target->addr != r->colorAddr || g->state.target->width != r->colorWidth)) {
		flushBatch(n, g);
	}
	s.target = target(n, g, r->colorAddr, r->colorWidth, r->colorSize, rows);
	int cycleType = r->cycleType;
	uint32_t modes = (uint32_t) cycleType | (r->alphaCompare ? MODE_ALPHA_COMPARE : 0) | (textured ? MODE_TEXTURED : 0) | (shaded ? MODE_SHADED : 0);
	int lastCycle = cycleType == 1 ? 1 : 0;
	const uint8_t* first = r->blendSel[0], * last = r->blendSel[lastCycle];
	modes |= (uint32_t) (first[0] | first[1] << 2 | first[2] << 4 | first[3] << 6) << 8;
	modes |= (uint32_t) (last[0] | last[1] << 2 | last[2] << 4 | last[3] << 6) << 16;
	bool blendMemory = last[2] == 1 && last[0] != 1, memoryFirst = last[0] == 1 && last[2] != 1;
	s.blendOn = cycleType < 2 && r->forceBlend && blendMemory;
	s.blendB = last[3];
	if (cycleType < 2 && r->forceBlend && memoryFirst) {
		s.memoryFirst = last[1] == 1 || last[1] == 3 ? 2 : 1;
		s.blendConstant = last[1] == 1 ? (r->fog.a >> 3) / 32.0f : 0;
		modes |= MODE_BLEND_MEMORY_FIRST | (s.memoryFirst == 2 ? MODE_BLEND_CONSTANT : 0);
	} else if (cycleType < 2 && r->forceBlend && !blendMemory) {
		modes |= MODE_BLEND_COLOR;
	}
	if (s.blendOn) {
		modes |= MODE_BLEND;
	}
	// The dither where the shader makes the image's color, or the GPU blends into a 16-bit image.
	if (cycleType < 2 && r->ditherRgb < 3 && !(modes & MODE_BLEND_MEMORY_FIRST) && (!(modes & MODE_BLEND) || r->colorSize == 2)) {
		modes |= MODE_DITHER | (r->ditherRgb == 1 ? MODE_BAYER : 0) | (r->ditherRgb == 2 ? MODE_NOISE : 0) | (r->colorSize == 2 ? MODE_IMAGE16 : 0);
	}
	if (depth && r->zPrim) {
		modes |= MODE_PRIM_DEPTH;
	}
	if (r->cvgTimesAlpha && r->antialias) {
		modes |= MODE_CVG_ALPHA;
	}
	// With coverage times alpha too the alpha stays (the VI's antialiasing is what smooths those edges).
	if (r->cvgAlpha && !r->cvgTimesAlpha) {
		modes |= MODE_ALPHA_CVG;
	}
	// The level of detail where the combiner reads its fraction or it picks the tiles.
	uint64_t m = r->otherModes, cmb = r->combine;
	bool lodTiles = textured && cycleType < 2 && ((m >> 48) & 1);
	bool lodRead = ((cmb >> 47) & 31) == 13 || ((cmb >> 41) & 7) == 0 || (cycleType == 1 && (((cmb >> 32) & 31) == 13 || ((cmb >> 18) & 7) == 0));
	uint32_t lod = textured && cycleType < 2 && (lodTiles || lodRead)
	                   ? 1 | (lodTiles ? 2 : 0) | (uint32_t) ((m >> 49) & 1) << 2 | (uint32_t) ((m >> 50) & 1) << 3 | (uint32_t) r->maxLevel << 4 |
	                         (uint32_t) (r->minLevel & 31) << 8
	                   : 0;
	struct ProgramKey key = { cycleType < 2 ? r->combine : 0, modes, lod };
	s.program = program(g, key);
	if (!s.program) {
		return false;
	}
	if (textured) {
		for (int k = 0; k < (cycleType == 1 || lodTiles ? 2 : 1); ++k) {
			// Without filtering a cycle converts its texels from YUV. With the level of detail and
			// one level, both texels are that level's (rdp.c levelOfDetail: always distant).
			bool oneLevel = ((r->otherModes >> 48) & 1) && !((r->otherModes >> 50) & 1) && r->maxLevel == 0;
			int index = (tile + (oneLevel ? 0 : k)) & 7, w, h;
			bool filter = k == 0 || cycleType != 1 ? cycleType >= 2 || ((r->otherModes >> 43) & 1) : (r->otherModes >> 42) & 1;
			s.texture[k] = texture(n, g, r, index, !filter, &w, &h);
			const struct Tile* t = &r->tile[index];
			bool bilinear = filter && ((r->otherModes >> 45) & 1) && cycleType < 2;
			int params[12] = { t->shiftS, t->shiftT, t->sl, t->tl, t->sh, t->th, t->cmS, t->cmT, t->maskS, t->maskT, bilinear, 0 };
			memcpy(s.tile[k], params, sizeof(params));
		}
	}
	colorOf(&r->prim, s.prim), colorOf(&r->env, s.env), colorOf(&r->fog, s.fog), colorOf(&r->blend, s.blend);
	s.primLod = r->primLod / 255.0f;
	s.primDepth = (float) r->primZ / 0x3FFFF;
	s.depthTest = depth && r->zCompare;
	s.depthWrite = depth && r->zUpdate;
	s.decal = s.depthTest && r->zMode == 3;
	// Outside decal mode a covered pixel passes only what is in front of it (no second pass over
	// one's own shadow, NFL Quarterback Club 99).
	s.depthLess = s.depthTest && r->zMode != 3;
	s.persp = persp;
	s.scissor[0] = r->scissorX0 >> 2, s.scissor[1] = r->scissorY0 >> 2;
	s.scissor[2] = (r->scissorX1 + 3) >> 2, s.scissor[3] = rows;
	if (s.scissor[2] <= s.scissor[0] || s.scissor[3] <= s.scissor[1]) {
		return false;
	}
	if (!g->stateSet || memcmp(&s, &g->state, sizeof(s))) {
		flushBatch(n, g);
		g->state = s;
		g->stateSet = true;
	}
	g->fastValid = true, g->fastGen = r->stateGen, g->fastKey = primitive, g->fastRows = rows;
	return drawRows(n, g, r, rows, y0, y1);
}

// A primitive's rows y0 to y1 - 1 in the state's target: they go back to RDRAM, their pages wait.
static bool drawRows(struct N64* n, struct Glr* g, struct RDP* r, int rows, int y0, int y1) {
	if (g->batchCount + 6 > BATCH_VERTICES) {
		flushBatch(n, g);
	}
	struct Target* t = g->state.target;
	y0 = y0 < 0 ? 0 : y0;
	y1 = y1 > rows ? rows : y1;
	if (t->dirtyTo <= t->dirtyFrom) {
		t->dirtyFrom = y0, t->dirtyTo = y1;
	} else {
		t->dirtyFrom = y0 < t->dirtyFrom ? y0 : t->dirtyFrom;
		t->dirtyTo = y1 > t->dirtyTo ? y1 : t->dirtyTo;
	}
	// Reads and writes of the pages of the rows drawn wait for the GPU (rdpWaitFor).
	if (t->marked < t->dirtyTo) {
		uint32_t bytes = r->colorSize == 3 ? 4 : r->colorSize == 1 ? 1 : 2;
		uint32_t first = (r->colorAddr & (RDRAM_SIZE - 1)) >> 12, last = ((r->colorAddr + (uint32_t) t->dirtyTo * r->colorWidth * bytes - 1) & (RDRAM_SIZE - 1)) >> 12;
		for (uint32_t p = first;; p = (p + 1) & ((RDRAM_SIZE >> 12) - 1)) {
			n->pageFlags[p] |= PAGE_RDP_WRITES;
			if (p == last) {
				break;
			}
		}
		t->marked = t->dirtyTo;
	}
	n->rdpPending = true;
	return true;
}

static void vertex(struct Glr* g, float x, float y, float z, float invW, const float* rgba, float s, float t) {
	float* v = &g->batch[g->batchCount++ * VERTEX_FLOATS];
	v[0] = x, v[1] = y, v[2] = z, v[3] = invW;
	v[4] = rgba[0], v[5] = rgba[1], v[6] = rgba[2], v[7] = rgba[3];
	v[8] = s, v[9] = t;
	memcpy(&v[10], &g->seq, 4);
}

// A rectangle from (x0, y0) to (x1, y1) in pixels, with texture coordinates at its corners.
static void rectangle(struct Glr* g, float x0, float y0, float x1, float y1, float z, const float* s, const float* t, bool flip) {
	static const float none[4] = { 0, 0, 0, 0 };
	float s00 = s[0], t00 = t[0], s10 = flip ? s[0] : s[1], t10 = flip ? t[1] : t[0];
	float s01 = flip ? s[1] : s[0], t01 = flip ? t[0] : t[1], s11 = s[1], t11 = t[1];
	vertex(g, x0, y0, z, 1, none, s00, t00);
	vertex(g, x1, y0, z, 1, none, s10, t10);
	vertex(g, x1, y1, z, 1, none, s11, t11);
	vertex(g, x0, y0, z, 1, none, s00, t00);
	vertex(g, x1, y1, z, 1, none, s11, t11);
	vertex(g, x0, y1, z, 1, none, s01, t01);
}

enum { QUEUE = 1 << 14 };

// The corners queued for RDP triangle w, the ones before them dropped (triangles that never came);
// NULL for a triangle without (from a microcode run on the RSP).
static const struct GlrCorners* queued(struct Glr* g, const uint64_t* w) {
	for (uint32_t i = g->queueRead; i != g->queueWrite && i - g->queueRead < 8; ++i) {
		const struct GlrCorners* q = &g->queue[i & (QUEUE - 1)];
		if (!memcmp(q->edges, w, sizeof(q->edges))) {
			g->queueRead = i + 1;
			return q;
		}
	}
	return NULL;
}

void glrQueue(struct N64* n, const struct GlrCorners* q) {
	if (!n->glr) {
		n->glr = glrCreate(n);
	}
	struct Glr* g = n->glr;
	if (!g) {
		return;
	}
	if (g->queueWrite - g->queueRead == QUEUE) {
		++g->queueRead;
	}
	g->queue[g->queueWrite++ & (QUEUE - 1)] = *q;
}

void glrSkip(struct N64* n, const uint64_t* w, int op) {
	if (op >= 0x08 && op <= 0x0F) {
		queued(n->glr, w);
	}
}

void glrQueueClear(struct N64* n) {
	if (n->glr) {
		n->glr->queueRead = n->glr->queueWrite;
	}
}

// The HLE's triangle from its corners.
static void triangle(struct N64* n, struct Glr* g, struct RDP* r, const struct GlrCorners* q) {
	uint32_t c = q->flags;
	// Depth from the primitive's needs no depth per corner (NFL Quarterback Club 99's shadows).
	bool z = (c & 1) || (r->zAddr && r->zPrim), tex = c & 2, shade = c & 4, persp = (c >> 16) & 1;
	float top = 1e9f, bottom = -1e9f;
	for (int k = 0; k < 3; ++k) {
		top = q->v[k][1] < top ? q->v[k][1] : top;
		bottom = q->v[k][1] > bottom ? q->v[k][1] : bottom;
	}
	r->maxLevel = (int) ((c >> 12) & 7);
	if (!drawState(n, g, r, tex, shade, z, (c >> 8) & 7, persp, (int) floorf(top), (int) ceilf(bottom) + 1)) {
		return;
	}
	// 1/w scaled as the microcodes send it, the corners' largest a half (hle.c triangle).
	float maxW = fmaxf(q->v[0][3], fmaxf(q->v[1][3], q->v[2][3]));
	float k = maxW > 0 ? 0.5f / maxW : 0.5f;
	for (int i = 0; i < 3; ++i) {
		vertex(g, q->v[i][0], q->v[i][1], q->v[i][2], q->v[i][3] * k, &q->v[i][4], q->v[i][8], q->v[i][9]);
	}
}

void glrDraw(struct N64* n, struct RDP* r, const uint64_t* w, int op) {
	struct Glr* g = n->glr;
	glesUse(g->gles);
	g->seq = r->seq;
	uint64_t c = w[0];
	const struct GlrCorners* q = op >= 0x08 && op <= 0x0F ? queued(g, w) : NULL;
	if (q && r->colorSize != 1) {
		triangle(n, g, r, q);
		return;
	}
	if (r->colorSize == 1) {
		// 8-bit images (masks programs read back, Blast Corps'): the RDP's own pixels.
		if (n->rdpPending) {
			glrSync(n);
		}
		rdpDrawSoftware(n, r, w, op);
		return;
	}
	if (op >= 0x08 && op <= 0x0F) {
		// An RDP triangle (from a microcode run on the RSP): the RDP draws it in RDRAM; many in a
		// field and the RDP threads take over.
		if (g->rawField != n->fields) {
			g->rawField = n->fields;
			g->rawTriangles = 0;
		}
		if (++g->rawTriangles > 100) {
			n->glrStop = true;
			n->glrStopReason = "the microcode sends the RDP its own triangles";
		}
		glrSync(n);
		rdpDrawSoftware(n, r, w, op);
		for (int i = 0; i < TARGETS; ++i) {
			if (g->target[i].addr == r->colorAddr) {
				g->target[i].loadedRows = 0;
			}
		}
		return;
	}
	if (op == 0x36) {
		int x1 = (int) ((c >> 44) & 0xFFF), y1 = (int) ((c >> 32) & 0xFFF), x0 = (int) ((c >> 12) & 0xFFF), y0 = (int) (c & 0xFFF);
		if (r->cycleType == 3 && r->colorAddr == r->zAddr) {
			// The depth image cleared (the GPU's depth, shared by the color images).
			flushBatch(n, g);
			if (g->stateSet) {
				struct Gl* gl = &g->gl;
				useFramebuffer(g, g->state.target->framebuffer);
				useScissor(g, false, 0, 0, 0, 0);
				useDepthMask(g, true);
				gl->ClearDepthf(1.0f);
				gl->Clear(GL_DEPTH_BUFFER_BIT);
			}
			// And in RDRAM, for the triangles the RDP draws.
			rdpDrawSoftware(n, r, w, op);
			return;
		}
		// A fill taller than a target (several images at once), or one whose pixels aren't all the same
		// (an 8-bit image takes the fill's bytes in turn, a 16-bit one its two halves): in RDRAM.
		bool uniform = r->colorSize == 3 || (r->colorSize == 2 && (r->fillColor >> 16) == (r->fillColor & 0xFFFF));
		if (r->cycleType == 3 && ((y1 >> 2) >= TARGET_ROWS || !uniform)) {
			glrSync(n);
			rdpDrawSoftware(n, r, w, op);
			return;
		}
		if (r->cycleType == 3) {
			// Fill: the color image filled.
			int rows = (y1 >> 2) + 1;
			if (!drawState(n, g, r, false, false, false, 0, false, y0 >> 2, rows)) {
				return;
			}
			flushBatch(n, g);
			struct Gl* gl = &g->gl;
			useFramebuffer(g, g->state.target->framebuffer);
			int sx0 = x0 >> 2 > g->state.scissor[0] ? x0 >> 2 : g->state.scissor[0];
			// The scissor's right edge is in in fill mode.
			int sx1 = (x1 >> 2) < (int) (r->scissorX1 >> 2) ? (x1 >> 2) + 1 : (int) (r->scissorX1 >> 2) + 1;
			int sy0 = y0 >> 2 > g->state.scissor[1] ? y0 >> 2 : g->state.scissor[1];
			int sy1 = rows < g->state.scissor[3] ? rows : g->state.scissor[3];
			if (sx1 <= sx0 || sy1 <= sy0) {
				return;
			}
			useScissor(g, true, sx0, sy0, sx1 - sx0, sy1 - sy0);
			uint32_t f = r->fillColor;
			if (r->colorSize == 3) {
				gl->ClearColor((f >> 24) / 255.0f, ((f >> 16) & 255) / 255.0f, ((f >> 8) & 255) / 255.0f, 1.0f);
			} else {
				uint32_t v = f >> 16;
				gl->ClearColor(((v >> 11) & 31) / 31.0f, ((v >> 6) & 31) / 31.0f, ((v >> 1) & 31) / 31.0f, 1.0f);
			}
			gl->Clear(GL_COLOR_BUFFER_BIT);
			return;
		}
		// Depth from the primitive's, as texture rectangles have it (Robot Poncots' dimmed scene
		// behind its HUD).
		if (!drawState(n, g, r, false, false, r->zAddr && r->zPrim, 0, false, y0 >> 2, (y1 >> 2) + 1)) {
			return;
		}
		static const float zero[2] = { 0, 0 };
		rectangle(g, x0 / 4.0f, y0 / 4.0f, x1 / 4.0f, y1 / 4.0f, (float) (r->primZ >> 3) / 0x7FFF, zero, zero, false);
		return;
	}
	// Texture rectangles: the pixels the RDP covers, and the texels at their corners (the shader
	// samples at the top of a pixel, here half a pixel left); steps a pixel in s5.10, copy mode four.
	int tile = (c >> 24) & 7;
	uint64_t d = w[1];
	float s0 = (int16_t) (d >> 48) / 32.0f, t0 = (int16_t) (d >> 32) / 32.0f;
	float dsdx = (int16_t) (d >> 16) / 1024.0f, dtdy = (int16_t) d / 1024.0f;
	if (r->cycleType == 3) {
		return;
	}
	if (r->cycleType == 2) {
		dsdx /= 4;
	}
	int cx0, cy0, x0, y0, x1, y1;
	rdpRectBounds(r, c, &cx0, &cy0, &x0, &y0, &x1, &y1);
	if (x1 < x0 || y1 < y0) {
		return;
	}
	bool flip = op == 0x25;
	float ax[2] = { (float) (x0 - cx0), (float) (x1 + 1 - cx0) }, ay[2] = { (float) (y0 - cy0), (float) (y1 + 1 - cy0) };
	const float* along = flip ? ay : ax, * across = flip ? ax : ay;
	float ss[2] = { s0 + along[0] * dsdx, s0 + along[1] * dsdx }, ts[2] = { t0 + across[0] * dtdy, t0 + across[1] * dtdy };
	r->maxLevel = 0;
	if (!drawState(n, g, r, true, false, r->zAddr && r->zPrim, tile, false, y0, y1 + 1)) {
		return;
	}
	rectangle(g, (float) x0, (float) y0, (float) (x1 + 1), (float) (y1 + 1), (float) (r->primZ >> 3) / 0x7FFF, ss, ts, flip);
}

bool glrDrew(struct N64* n, uint32_t addr, uint32_t len) {
	struct Glr* g = n->glr;
	for (int i = 0; i < TARGETS; ++i) {
		struct Target* t = &g->target[i];
		uint32_t bytes = t->size == 3 ? 4 : t->size == 1 ? 1 : 2;
		uint32_t from = t->addr + (uint32_t) t->dirtyFrom * t->width * bytes, to = t->addr + (uint32_t) t->dirtyTo * t->width * bytes;
		if (t->texture && t->dirtyTo > t->dirtyFrom && addr < to && addr + len > from) {
			return true;
		}
	}
	return false;
}

// The framebuffer of the color image the VI shows from origin (rows rows, stride pixels a row, size
// 2 or 3), when the GPU drew all of those rows since its pixels last went back to RDRAM: x and y of
// the picture's first pixel in it. 0 otherwise: RDRAM has the picture.
unsigned glrPicture(struct N64* n, uint32_t origin, uint32_t stride, int size, int rows, int* x, int* y) {
	struct Glr* g = n->glr;
	uint32_t const bytes = size == 3 ? 4 : 2;
	for (int i = 0; i < TARGETS; ++i) {
		struct Target* t = &g->target[i];
		if (!t->texture || t->size != size || t->width != stride || t->dirtyTo <= t->dirtyFrom || origin < t->addr) {
			continue;
		}
		uint32_t const offset = origin - t->addr, row = offset / (stride * bytes);
		if ((int) row < t->dirtyFrom || (int) row + rows > t->dirtyTo) {
			continue;
		}
		glesUse(g->gles);
		flushBatch(n, g);
		*x = (int) (offset % (stride * bytes) / bytes);
		*y = (int) row;
		return t->framebuffer;
	}
	return 0;
}

// A picture from a target as drawn (scale times its size, bottom row first), for tests.
bool glrRead(struct N64* n, unsigned framebuffer, int x, int y, int w, int h, uint32_t* out) {
	struct Glr* g = n->glr;
	if (!g) {
		return false;
	}
	int const k = g->scale;
	glesUse(g->gles);
	flushBatch(n, g);
	useFramebuffer(g, framebuffer);
	g->gl.ReadPixels(x * k, y * k, w * k, h * k, GL_RGBA, GL_UNSIGNED_BYTE, out);
	return true;
}

bool glrShows(struct N64* n, uint32_t origin) {
	struct Glr* g = n->glr;
	for (int i = 0; i < TARGETS; ++i) {
		struct Target* t = &g->target[i];
		if (t->texture && origin >= t->addr && origin - t->addr < t->width * 4 * 8) {
			glesUse(g->gles);
			flushBatch(n, g);
			g->gl.Flush();
			return true;
		}
	}
	return false;
}

// RDRAM [addr, addr + len) about to be read: the GPU's rows there go back. Rows at the end of what a
// color image drew go back alone (the VI's filters read a row of the image before the one shown,
// which the GPU may still be drawing); anything else gives everything back.
void glrSyncRange(struct N64* n, uint32_t addr, uint32_t len) {
	struct Glr* g = n->glr;
	for (int i = 0; i < TARGETS; ++i) {
		struct Target* t = &g->target[i];
		uint32_t const row = t->width * (t->size == 3 ? 4 : t->size == 1 ? 1 : 2);
		uint32_t const from = t->addr + (uint32_t) t->dirtyFrom * row, to = t->addr + (uint32_t) t->dirtyTo * row;
		if (!t->texture || t->dirtyTo <= t->dirtyFrom || addr >= to || addr + len <= from) {
			continue;
		}
		int const first = (int) ((addr > from ? addr - t->addr : from - t->addr) / row);
		if (first <= t->dirtyFrom || addr + len < to) {
			glrSync(n);
			return;
		}
		glesUse(g->gles);
		flushBatch(n, g);
		writePixels(n, t, first, t->dirtyTo);
		t->dirtyTo = first;
	}
}

void glrSync(struct N64* n) {
	struct Glr* g = n->glr;
	g->fastValid = false;
	++n->stats.glSyncs;
	glesUse(g->gles);
	flushBatch(n, g);
	for (int i = 0; i < TARGETS; ++i) {
		struct Target* t = &g->target[i];
		int rows = t->marked > t->dirtyTo ? t->marked : t->dirtyTo;
		if (t->texture && rows) {
			uint32_t bytes = t->size == 3 ? 4 : t->size == 1 ? 1 : 2;
			uint32_t first = (t->addr & (RDRAM_SIZE - 1)) >> 12, last = ((t->addr + (uint32_t) rows * t->width * bytes - 1) & (RDRAM_SIZE - 1)) >> 12;
			for (uint32_t p = first;; p = (p + 1) & ((RDRAM_SIZE >> 12) - 1)) {
				n->pageFlags[p] &= ~PAGE_RDP;
				if (p == last) {
					break;
				}
			}
		}
		if (t->texture && t->dirtyTo > t->dirtyFrom) {
			writePixels(n, t, t->dirtyFrom, t->dirtyTo);
			t->dirtyFrom = t->dirtyTo = 0;
		}
		// RDRAM may change before the GPU draws here again.
		t->loadedRows = 0;
		t->marked = 0;
	}
	n->rdpPending = false;
}

struct Glr* glrCreate(struct N64* n) {
	char error[256];
	struct Glr* g = calloc(1, sizeof(*g));
	if (!g) {
		return NULL;
	}
	g->gles = glesCreateContext(error, sizeof(error));
	g->batch = malloc((size_t) BATCH_VERTICES * VERTEX_FLOATS * 4);
	g->image = malloc(1024 * TARGET_ROWS * 4);
	g->queue = malloc(sizeof(*g->queue) * QUEUE);
	if (!g->gles || !g->batch || !g->image || !g->queue || !loadGl(g)) {
		fprintf(stderr, "GL renderer unavailable (%s), using the RDP\n", g->gles ? "GL functions missing" : error);
		glrDestroy(g);
		n->rdpGl = false;
		return NULL;
	}
	glesUse(g->gles);
	struct Gl* gl = &g->gl;
	g->vertexShader = compile(g, GL_VERTEX_SHADER, vertexSource);
	gl->GenRenderbuffers(1, &g->depth);
	gl->BindRenderbuffer(GL_RENDERBUFFER, g->depth);
	const char* scale = n->settings ? n->settings("N64_GL_SCALE") : getenv("N64_GL_SCALE");
	g->scale = scale && *scale >= '1' && *scale <= '8' ? *scale - '0' : 1;
	n->glrScale = g->scale;
	gl->RenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 1024 * g->scale, TARGET_ROWS * g->scale);
	if (g->scale > 1) {
		gl->GenTextures(1, &g->native);
		uploadTexture(g, g->native);
		gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1024, TARGET_ROWS, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		gl->GenFramebuffers(1, &g->nativeFramebuffer);
		useFramebuffer(g, g->nativeFramebuffer);
		gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g->native, 0);
	}
	gl->GenVertexArrays(1, &g->vertexArray);
	gl->BindVertexArray(g->vertexArray);
	gl->GenBuffers(1, &g->vertexBuffer);
	gl->BindBuffer(GL_ARRAY_BUFFER, g->vertexBuffer);
	gl->BufferData(GL_ARRAY_BUFFER, VERTEX_BUFFER_BYTES, NULL, GL_STREAM_DRAW);
	gl->VertexAttribPointer(0, 4, GL_FLOAT, 0, VERTEX_FLOATS * 4, (const void*) 0);
	gl->VertexAttribPointer(1, 4, GL_FLOAT, 0, VERTEX_FLOATS * 4, (const void*) 16);
	gl->VertexAttribPointer(2, 2, GL_FLOAT, 0, VERTEX_FLOATS * 4, (const void*) 32);
	gl->VertexAttribIPointer(3, 1, GL_UNSIGNED_INT, VERTEX_FLOATS * 4, (const void*) 40);
	gl->EnableVertexAttribArray(0);
	gl->EnableVertexAttribArray(1);
	gl->EnableVertexAttribArray(2);
	gl->EnableVertexAttribArray(3);
	gl->PixelStorei(GL_PACK_ALIGNMENT, 1);
	gl->PixelStorei(GL_UNPACK_ALIGNMENT, 1);
	gl->Disable(GL_DITHER);
	memset(&g->applied, 0xFF, sizeof(g->applied));
	fprintf(stderr, "GL renderer on %s\n", glesRenderer(g->gles));
	if (!g->vertexShader) {
		glrDestroy(g);
		n->rdpGl = false;
		return NULL;
	}
	return g;
}

// A save state loaded under the renderer: the RDP's generations start over, so nothing cached by
// them holds.
void glrStateLoaded(struct Glr* g) {
	memset(g->tileTexture, 0, sizeof(g->tileTexture));
	g->queueRead = g->queueWrite;
	g->fastValid = false;
	memset(&g->applied, 0xFF, sizeof(g->applied));
}

void glrForget(struct N64* n) {
	if (n->glr) {
		memset(&n->glr->applied, 0xFF, sizeof(n->glr->applied));
	}
}

void glrEnd(struct N64* n) {
	glrSync(n);
	glrDestroy(n->glr);
	n->glr = NULL;
	n->rdpGl = n->glrStop = false;
	n->rdpThreads = n->rdpThreadsAsked;
	// Compiled loads check the RDP's pages as the renderers need.
	blocksFree(n);
}

void glrDestroy(struct Glr* g) {
	if (!g) {
		return;
	}
	glesDestroy(g->gles);
	free(g->batch);
	free(g->image);
	free(g->queue);
	free(g);
}
