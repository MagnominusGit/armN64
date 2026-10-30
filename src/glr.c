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
	GL_TEXTURE_2D = 0x0DE1, GL_RGBA = 0x1908, GL_RGBA8 = 0x8058, GL_UNSIGNED_BYTE = 0x1401, GL_FLOAT = 0x1406,
	GL_FRAMEBUFFER = 0x8D40, GL_COLOR_ATTACHMENT0 = 0x8CE0, GL_DEPTH_ATTACHMENT = 0x8D00, GL_RENDERBUFFER = 0x8D41,
	GL_DEPTH_COMPONENT24 = 0x81A6, GL_FRAMEBUFFER_COMPLETE = 0x8CD5,
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

enum { TARGETS = 8, TARGET_ROWS = 480, PROGRAMS = 256, TEXTURES = 1024, TEXTURE_BUCKETS = 2048, BATCH_VERTICES = 1 << 16, VERTEX_FLOATS = 10,
       VERTEX_BUFFER_BYTES = BATCH_VERTICES * VERTEX_FLOATS * 4 * 2 };

struct Gl {
	void (*GenFramebuffers)(GLsizei, GLuint*);
	void (*BindFramebuffer)(GLenum, GLuint);
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
	int width, height;
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
	bool depthTest, depthWrite, decal, blendOn, persp;
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
		g->gl.Scissor(x, y, w, h);
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
	return GLFN(GenFramebuffers) && GLFN(BindFramebuffer) && GLFN(FramebufferTexture2D) && GLFN(FramebufferRenderbuffer) &&
	       GLFN(CheckFramebufferStatus) && GLFN(DeleteFramebuffers) && GLFN(GenRenderbuffers) && GLFN(BindRenderbuffer) &&
	       GLFN(RenderbufferStorage) && GLFN(GenTextures) && GLFN(DeleteTextures) && GLFN(BindTexture) && GLFN(ActiveTexture) &&
	       GLFN(TexImage2D) && GLFN(TexSubImage2D) && GLFN(TexParameteri) && GLFN(Viewport) && GLFN(Scissor) && GLFN(Enable) &&
	       GLFN(Disable) && GLFN(ClearColor) && GLFN(ClearDepthf) && GLFN(Clear) && GLFN(DepthFunc) && GLFN(DepthMask) &&
	       GLFN(BlendFunc) && GLFN(BlendFuncSeparate) && GLFN(BlendColor) && GLFN(PolygonOffset) && GLFN(ReadPixels) && GLFN(PixelStorei) && GLFN(CreateShader) &&
	       GLFN(ShaderSource) && GLFN(CompileShader) && GLFN(GetShaderiv) && GLFN(GetShaderInfoLog) && GLFN(DeleteShader) &&
	       GLFN(CreateProgram) && GLFN(AttachShader) && GLFN(LinkProgram) && GLFN(GetProgramiv) && GLFN(GetProgramInfoLog) &&
	       GLFN(UseProgram) && GLFN(DeleteProgram) && GLFN(GetUniformLocation) && GLFN(Uniform1i) && GLFN(Uniform1f) &&
	       GLFN(Uniform2f) && GLFN(Uniform4f) && GLFN(Uniform4iv) && GLFN(GenVertexArrays) && GLFN(BindVertexArray) &&
	       GLFN(GenBuffers) && GLFN(BindBuffer) && GLFN(BufferData) && GLFN(MapBufferRange) && GLFN(UnmapBuffer) && GLFN(VertexAttribPointer) &&
	       GLFN(EnableVertexAttribArray) && GLFN(DrawArrays) && GLFN(Flush);
#undef GLFN
}

// ---- Shaders

static const char vertexSource[] =
	"#version 300 es\n"
	"layout(location = 0) in vec4 aPos;\n"
	"layout(location = 1) in vec4 aShade;\n"
	"layout(location = 2) in vec2 aTex;\n"
	"uniform vec2 uSize;\n"
	"uniform int uPersp;\n"
	"out vec4 vShade;\n"
	"out vec3 vTex;\n"
	"void main() {\n"
	"	gl_Position = vec4(aPos.x / uSize.x * 2.0 - 1.0, aPos.y / uSize.y * 2.0 - 1.0, aPos.z * 2.0 - 1.0, 1.0);\n"
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
	"uniform vec4 uPrim, uEnv, uFog, uBlend;\n"
	"uniform float uPrimLod, uPrimDepth;\n"
	"uniform ivec4 uTile0[3], uTile1[3];\n"
	"uniform ivec4 uImage0, uImage1;\n"
	"uniform highp sampler2D uTex0, uTex1;\n"
	"out vec4 oColor;\n"
	"vec3 e8(vec3 v) { return floor(v * 255.0 + 0.5); }\n"
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
	"	return texelFetch(t, ivec2(clamp(i, 0, image.x - 1), clamp(j, 0, image.y - 1)), 0);\n"
	"}\n"
	"vec4 sampleTile(sampler2D t, ivec4 a, ivec4 b, ivec4 c, ivec4 image) {\n"
	"	vec3 top = vTex - 0.5 * dFdy(vTex);\n"
	"	vec2 st = top.xy / top.z;\n"
	"	int s = int(floor(st.x * 32.0 + 0.01)), tt = int(floor(st.y * 32.0 + 0.01));\n"
	"	int fs, ft, f;\n"
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

// modes: cycle type (bits 0-1), alpha compare (2), textured (3), shaded (4), the blender's first
// cycle and last cycle selections (8-15, 16-23), blending with memory (24), depth from prim (25),
// coverage times alpha with antialiasing (26), alpha as coverage (27), the last cycle's blend formula
// in the shader (28).
enum {
	MODE_CYCLE = 3, MODE_ALPHA_COMPARE = 4, MODE_TEXTURED = 8, MODE_SHADED = 16, MODE_BLEND = 1 << 24, MODE_PRIM_DEPTH = 1 << 25,
	MODE_CVG_ALPHA = 1 << 26, MODE_ALPHA_CVG = 1 << 27, MODE_BLEND_COLOR = 1 << 28, MODE_BLEND_MEMORY_FIRST = 1 << 29, MODE_BLEND_CONSTANT = 1 << 30,
};

static const char* rgbInput(int which, int sel) {
	static const char* common[6] = { "combined.rgb", "tex0.rgb", "tex1.rgb", "uPrim.rgb", "shade.rgb", "uEnv.rgb" };
	if (sel < 6) {
		return common[sel];
	}
	if (which == 2) {
		static const char* mul[10] = { "vec3(0.0)", "vec3(combined.a)", "vec3(tex0.a)", "vec3(tex1.a)", "vec3(uPrim.a)", "vec3(shade.a)",
		                               "vec3(uEnv.a)", "vec3(0.0)", "vec3(uPrimLod)", "vec3(0.0)" };
		return sel < 16 ? mul[sel - 6] : "vec3(0.0)";
	}
	return (which == 0 || which == 3) && sel == 6 ? "vec3(1.0)" : "vec3(0.0)";
}

static const char* alphaInput(int sel, bool multiplier) {
	static const char* common[6] = { "combined.a", "tex0.a", "tex1.a", "uPrim.a", "shade.a", "uEnv.a" };
	if (sel == 0 && multiplier) {
		return "0.0";
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
// alpha, kept within 0-255.
static void cycleSource(char* out, size_t size, uint64_t c, int cycle) {
	int a = cycle ? (c >> 37) & 15 : (c >> 52) & 15, m = cycle ? (c >> 32) & 31 : (c >> 47) & 31;
	int b = cycle ? (c >> 24) & 15 : (c >> 28) & 15, d = cycle ? (c >> 6) & 7 : (c >> 15) & 7;
	int aa = cycle ? (c >> 21) & 7 : (c >> 44) & 7, ab = cycle ? (c >> 3) & 7 : (c >> 12) & 7;
	int am = cycle ? (c >> 18) & 7 : (c >> 41) & 7, ad = cycle ? c & 7 : (c >> 9) & 7;
	char in[8][64];
	snprintf(out, size, "	combined = clamp(floor((vec4((%s - %s) * %s, (%s - %s) * %s) + vec4(%s, %s) * 256.0 + 128.0) / 256.0), 0.0, 255.0) / 255.0;\n",
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
		if (g->programs[i].key.combine == key.combine && g->programs[i].key.modes == key.modes) {
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
	strcat(source, "void main() {\n	vec4 combined = vec4(0.0);\n");
	strcat(source, key.modes & MODE_SHADED ? "	vec4 shade = vShade;\n" : "	vec4 shade = vec4(0.0);\n");
	if (key.modes & MODE_TEXTURED) {
		strcat(source, "	vec4 tex0 = sampleTile(uTex0, uTile0[0], uTile0[1], uTile0[2], uImage0);\n");
		strcat(source, cycleType == 1 ? "	vec4 tex1 = sampleTile(uTex1, uTile1[0], uTile1[1], uTile1[2], uImage1);\n" : "	vec4 tex1 = tex0;\n");
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
			snprintf(line, sizeof(line), "	{ %s oColor = vec4(%s * blendA - (1.0 - reject) * 0.49 / 255.0, mix(blendA8 / 255.0, 1.0, reject)); }\n", factors,
			         blendColorInput(last));
		} else {
			snprintf(line, sizeof(line), "	oColor = vec4(%s, %s * memoryKeep);\n", blendColorInput(last), blendAlphaInput(last >> 2));
		}
		strcat(source, line);
	} else {
		strcat(source, "	oColor = combined;\n");
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
	uploadTexture(g, t->texture);
	gl->TexSubImage2D(GL_TEXTURE_2D, 0, 0, from, (GLsizei) t->width, to - from, GL_RGBA, GL_UNSIGNED_BYTE, out);
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
		gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei) width, TARGET_ROWS, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		useFramebuffer(g, t->framebuffer);
		gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->texture, 0);
		gl->FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, g->depth);
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
	useFramebuffer(g, t->framebuffer);
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
	bool same = slot->texture && slot->width == w && slot->height == h;
	if (!slot->texture) {
		gl->GenTextures(1, &slot->texture);
	}
	uint32_t* pixels = malloc((size_t) w * h * 4);
	if (!pixels) {
		return NULL;
	}
	rdpTileImage(r, index, w, h, convert, pixels);
	++n->stats.glTextures;
	uploadTexture(g, slot->texture);
	if (same) {
		// The same size: the texels replaced in place (no new storage).
		gl->TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	} else {
		gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}
	free(pixels);
	slot->hash = key, slot->width = w, slot->height = h, slot->used = ++g->clock;
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
		gl->Viewport(0, 0, (GLsizei) s->target->width, TARGET_ROWS);
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
			int image[4] = { s->texture[k]->width, s->texture[k]->height, 0, 0 };
			if (memcmp(u->image[k], image, sizeof(image))) {
				gl->Uniform4iv(p->image[k], 1, image);
				memcpy(u->image[k], image, sizeof(image));
			}
		}
	}
	// The RDP passes what is no farther (within its depth slope), and anything on a cleared depth.
	int depth = s->depthTest ? 1 : s->depthWrite ? 2 : 0;
	if (a->depth != depth) {
		depth ? gl->Enable(GL_DEPTH_TEST) : gl->Disable(GL_DEPTH_TEST);
		if (depth) {
			gl->DepthFunc(depth == 1 ? GL_LEQUAL : GL_ALWAYS);
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
		gl->DrawArrays(GL_TRIANGLES, (GLint) (g->vertexOffset / (VERTEX_FLOATS * 4)), g->batchCount);
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
	int primitive = textured | shaded << 1 | depth << 2 | tile << 3 | persp << 6;
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
	struct ProgramKey key = { cycleType < 2 ? r->combine : 0, modes };
	s.program = program(g, key);
	if (!s.program) {
		return false;
	}
	if (textured) {
		for (int k = 0; k < (cycleType == 1 ? 2 : 1); ++k) {
			// Without filtering a cycle converts its texels from YUV.
			int index = (tile + k) & 7, w, h;
			bool filter = k == 0 ? cycleType >= 2 || ((r->otherModes >> 43) & 1) : (r->otherModes >> 42) & 1;
			s.texture[k] = texture(n, g, r, index, !filter, &w, &h);
			const struct Tile* t = &r->tile[index];
			bool bilinear = filter && ((r->otherModes >> 45) & 1) && cycleType < 2;
			int params[12] = { t->shiftS, t->shiftT, t->sl, t->tl, t->sh, t->th, t->cmS, t->cmT, t->maskS, t->maskT, bilinear, 0 };
			memcpy(s.tile[k], params, sizeof(params));
		}
	}
	colorOf(&r->prim, s.prim), colorOf(&r->env, s.env), colorOf(&r->fog, s.fog), colorOf(&r->blend, s.blend);
	s.primLod = r->primLod / 255.0f;
	s.primDepth = (float) (r->primZ >> 3) / 0x7FFF;
	s.depthTest = depth && r->zCompare;
	s.depthWrite = depth && r->zUpdate;
	s.decal = s.depthTest && r->zMode == 3;
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

static float halfFloat(uint64_t w, bool high) {
	uint32_t bits = high ? (uint32_t) (w >> 32) : (uint32_t) w;
	float f;
	memcpy(&f, &bits, 4);
	return f;
}

// A GL triangle's RDP time: its pixels inside the scissor box (the polygon clipped to it), as
// primitiveClocks counts a triangle's.
uint64_t glrTriangleClocks(const struct RDP* r, const uint64_t* w) {
	float poly[2][16][2];
	int count = 3;
	for (int k = 0; k < 3; ++k) {
		poly[0][k][0] = halfFloat(w[1 + k * 5], true);
		poly[0][k][1] = halfFloat(w[1 + k * 5], false);
	}
	const float bounds[4] = { r->scissorX0 / 4.0f, r->scissorY0 / 4.0f, r->scissorX1 / 4.0f, r->scissorY1 / 4.0f };
	int from = 0;
	for (int edge = 0; edge < 4 && count; ++edge) {
		int axis = edge & 1, out = 0;
		float limit = bounds[edge], sign = edge < 2 ? 1.0f : -1.0f;
		for (int i = 0; i < count; ++i) {
			const float* a = poly[from][i], * b = poly[from][(i + 1) % count];
			float da = (a[axis] - limit) * sign, db = (b[axis] - limit) * sign;
			if (da >= 0) {
				poly[!from][out][0] = a[0], poly[!from][out][1] = a[1], ++out;
			}
			if ((da >= 0) != (db >= 0) && out < 15) {
				float t = da / (da - db);
				poly[!from][out][0] = a[0] + (b[0] - a[0]) * t, poly[!from][out][1] = a[1] + (b[1] - a[1]) * t, ++out;
			}
		}
		count = out, from = !from;
	}
	float area = 0;
	for (int i = 0; i < count; ++i) {
		const float* a = poly[from][i], * b = poly[from][(i + 1) % count];
		area += a[0] * b[1] - b[0] * a[1];
	}
	return (uint64_t) (fabsf(area) / 2) * (r->cycleType == 1 ? 2 : 1) + 20;
}

void glrDraw(struct N64* n, struct RDP* r, const uint64_t* w, int op) {
	struct Glr* g = n->glr;
	glesUse(g->gles);
	uint64_t c = w[0];
	if (op == 0x01) {
		bool z = c & 1, tex = c & 2, shade = c & 4, persp = (c >> 16) & 1;
		float v[3][10];
		float top = 1e9f, bottom = -1e9f;
		for (int k = 0; k < 3; ++k) {
			for (int i = 0; i < 5; ++i) {
				v[k][i * 2] = halfFloat(w[1 + k * 5 + i], true);
				v[k][i * 2 + 1] = halfFloat(w[1 + k * 5 + i], false);
			}
			top = v[k][1] < top ? v[k][1] : top;
			bottom = v[k][1] > bottom ? v[k][1] : bottom;
		}
		if (!drawState(n, g, r, tex, shade, z, (c >> 8) & 7, persp, (int) floorf(top), (int) ceilf(bottom) + 1)) {
			return;
		}
		for (int k = 0; k < 3; ++k) {
			vertex(g, v[k][0], v[k][1], v[k][2], v[k][3], &v[k][4], v[k][8], v[k][9]);
		}
		return;
	}
	if (op >= 0x08 && op <= 0x0F) {
		// An RDP triangle (from a microcode run on the RSP): the RDP draws it in RDRAM; many in a
		// field and the RDP threads take over.
		if (g->rawField != n->fields) {
			g->rawField = n->fields;
			g->rawTriangles = 0;
		}
		n->glrStop |= ++g->rawTriangles > 100;
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
		if (r->cycleType == 3 && (y1 >> 2) >= TARGET_ROWS) {
			// A fill taller than a target (several images at once): in RDRAM.
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
			int sx1 = (x1 >> 2) + 1 < g->state.scissor[2] ? (x1 >> 2) + 1 : g->state.scissor[2];
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
		if (!drawState(n, g, r, false, false, false, 0, false, y0 >> 2, (y1 >> 2) + 1)) {
			return;
		}
		static const float zero[2] = { 0, 0 };
		rectangle(g, x0 / 4.0f, y0 / 4.0f, x1 / 4.0f, y1 / 4.0f, 0, zero, zero, false);
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
	float ax[2] = { x0 - cx0 - 0.5f, x1 + 1 - cx0 - 0.5f }, ay[2] = { (float) (y0 - cy0), (float) (y1 + 1 - cy0) };
	const float* along = flip ? ay : ax, * across = flip ? ax : ay;
	float ss[2] = { s0 + along[0] * dsdx, s0 + along[1] * dsdx }, ts[2] = { t0 + across[0] * dtdy, t0 + across[1] * dtdy };
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
	if (!g->gles || !g->batch || !g->image || !loadGl(g)) {
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
	gl->RenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 1024, TARGET_ROWS);
	gl->GenVertexArrays(1, &g->vertexArray);
	gl->BindVertexArray(g->vertexArray);
	gl->GenBuffers(1, &g->vertexBuffer);
	gl->BindBuffer(GL_ARRAY_BUFFER, g->vertexBuffer);
	gl->BufferData(GL_ARRAY_BUFFER, VERTEX_BUFFER_BYTES, NULL, GL_STREAM_DRAW);
	gl->VertexAttribPointer(0, 4, GL_FLOAT, 0, VERTEX_FLOATS * 4, (const void*) 0);
	gl->VertexAttribPointer(1, 4, GL_FLOAT, 0, VERTEX_FLOATS * 4, (const void*) 16);
	gl->VertexAttribPointer(2, 2, GL_FLOAT, 0, VERTEX_FLOATS * 4, (const void*) 32);
	gl->EnableVertexAttribArray(0);
	gl->EnableVertexAttribArray(1);
	gl->EnableVertexAttribArray(2);
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
	free(g);
}
