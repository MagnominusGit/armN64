#include "gles.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The few EGL and GLES names used, so building needs no EGL or GLES headers.
typedef void* EGLDisplay;
typedef void* EGLConfig;
typedef void* EGLContext;
typedef void* EGLSurface;
typedef int32_t EGLint;
typedef unsigned EGLBoolean;
typedef unsigned EGLenum;
typedef unsigned GLenum;
typedef unsigned GLuint;
typedef int GLint;
typedef int GLsizei;
typedef unsigned GLbitfield;
typedef intptr_t GLintptr;
typedef intptr_t GLsizeiptr;

enum {
	EGL_NONE = 0x3038, EGL_RENDERABLE_TYPE = 0x3040, EGL_SURFACE_TYPE = 0x3033, EGL_PBUFFER_BIT = 0x0001,
	EGL_OPENGL_ES3_BIT = 0x0040, EGL_OPENGL_ES_API = 0x30A0, EGL_CONTEXT_MAJOR_VERSION = 0x3098,
	EGL_CONTEXT_MINOR_VERSION = 0x30FB, EGL_EXTENSIONS = 0x3055, EGL_WIDTH = 0x3057, EGL_HEIGHT = 0x3056,
	EGL_PLATFORM_SURFACELESS_MESA = 0x31DD, EGL_DRAW = 0x3059, EGL_READ = 0x305A,
	GL_COMPUTE_SHADER = 0x91B9, GL_COMPILE_STATUS = 0x8B81, GL_LINK_STATUS = 0x8B82, GL_INFO_LOG_LENGTH = 0x8B84,
	GL_SHADER_STORAGE_BUFFER = 0x90D2, GL_DYNAMIC_DRAW = 0x88E8, GL_MAP_READ_BIT = 0x0001,
	GL_STREAM_READ = 0x88E1, GL_COPY_READ_BUFFER = 0x8F36, GL_COPY_WRITE_BUFFER = 0x8F37,
	GL_SHADER_STORAGE_BARRIER_BIT = 0x2000, GL_BUFFER_UPDATE_BARRIER_BIT = 0x0200, GL_RENDERER = 0x1F01,
	GL_VERSION = 0x1F02, GL_PROGRAM_BINARY_RETRIEVABLE_HINT = 0x8257, GL_PROGRAM_BINARY_LENGTH = 0x8741,
};

struct Gles {
	void* egl;
	void* gles;
	EGLDisplay display;
	EGLContext context;
	EGLSurface surface;
	// The frontend's context (glesFrontend), current already: nothing to make or switch.
	bool external;
	GLuint program, buffer[GLES_BUFFERS];
	// Programs added with glesAddProgram (the first is glesCreate's), and their uniforms.
	GLuint programs[GLES_PROGRAMS];
	GLint programUniforms[GLES_PROGRAMS];
	int programCount;
	// Where glesMap copies what it reads: memory the CPU reads cached, unlike the storage buffers.
	GLuint staging;
	size_t stagingSize;
	GLint uniforms;
	char renderer[128];

	EGLDisplay (*eglGetDisplay)(void*);
	EGLDisplay (*eglGetPlatformDisplayEXT)(EGLenum, void*, const EGLint*);
	EGLBoolean (*eglInitialize)(EGLDisplay, EGLint*, EGLint*);
	EGLBoolean (*eglTerminate)(EGLDisplay);
	EGLBoolean (*eglBindAPI)(EGLenum);
	EGLBoolean (*eglChooseConfig)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*);
	EGLContext (*eglCreateContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint*);
	EGLSurface (*eglCreatePbufferSurface)(EGLDisplay, EGLConfig, const EGLint*);
	EGLBoolean (*eglMakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
	EGLDisplay (*eglGetCurrentDisplay)(void);
	EGLSurface (*eglGetCurrentSurface)(EGLint);
	EGLContext (*eglGetCurrentContext)(void);
	EGLenum (*eglQueryAPI)(void);
	EGLBoolean (*eglDestroyContext)(EGLDisplay, EGLContext);
	EGLBoolean (*eglDestroySurface)(EGLDisplay, EGLSurface);
	const char* (*eglQueryString)(EGLDisplay, EGLint);
	void* (*eglGetProcAddress)(const char*);

	GLuint (*glCreateShader)(GLenum);
	void (*glShaderSource)(GLuint, GLsizei, const char* const*, const GLint*);
	void (*glCompileShader)(GLuint);
	void (*glGetShaderiv)(GLuint, GLenum, GLint*);
	void (*glGetShaderInfoLog)(GLuint, GLsizei, GLsizei*, char*);
	void (*glDeleteShader)(GLuint);
	GLuint (*glCreateProgram)(void);
	void (*glAttachShader)(GLuint, GLuint);
	void (*glLinkProgram)(GLuint);
	void (*glGetProgramiv)(GLuint, GLenum, GLint*);
	void (*glGetProgramInfoLog)(GLuint, GLsizei, GLsizei*, char*);
	void (*glUseProgram)(GLuint);
	void (*glDeleteProgram)(GLuint);
	GLint (*glGetUniformLocation)(GLuint, const char*);
	void (*glUniform1iv)(GLint, GLsizei, const GLint*);
	void (*glGenBuffers)(GLsizei, GLuint*);
	void (*glDeleteBuffers)(GLsizei, const GLuint*);
	void (*glBindBuffer)(GLenum, GLuint);
	void (*glBindBufferBase)(GLenum, GLuint, GLuint);
	void (*glBufferData)(GLenum, GLsizeiptr, const void*, GLenum);
	void (*glBufferSubData)(GLenum, GLintptr, GLsizeiptr, const void*);
	void* (*glMapBufferRange)(GLenum, GLintptr, GLsizeiptr, GLbitfield);
	unsigned char (*glUnmapBuffer)(GLenum);
	void (*glDispatchCompute)(GLuint, GLuint, GLuint);
	void (*glMemoryBarrier)(GLbitfield);
	void (*glFlush)(void);
	void (*glCopyBufferSubData)(GLenum, GLenum, GLintptr, GLintptr, GLsizeiptr);
	// Optional: programs kept compiled on disk.
	void (*glGetProgramBinary)(GLuint, GLsizei, GLsizei*, GLenum*, void*);
	void (*glProgramBinary)(GLuint, GLenum, const void*, GLsizei);
	void (*glProgramParameteri)(GLuint, GLenum, GLint);
	const unsigned char* (*glGetString)(GLenum);
};



// A frontend's GL context (libretro's shared hardware context), current whenever the machine runs:
// contexts made from then on are that one, with its functions.
static void* (*frontendProc)(const char*);

void glesFrontend(void* (*getProc)(const char* name)) {
	frontendProc = getProc;
}

static void* lookup(struct Gles* g, const char* name) {
	if (g->external) {
		return frontendProc(name);
	}
	void* f = g->gles ? dlsym(g->gles, name) : NULL;
	if (!f) {
		f = dlsym(g->egl, name);
	}
	if (!f && g->eglGetProcAddress) {
		f = g->eglGetProcAddress(name);
	}
	return f;
}

#define LOAD(name) (*(void**) &g->name = lookup(g, #name), g->name != NULL)

static bool loadAll(struct Gles* g) {
	bool ok = LOAD(eglGetDisplay) && LOAD(eglInitialize) && LOAD(eglTerminate) && LOAD(eglBindAPI) &&
	          LOAD(eglChooseConfig) && LOAD(eglCreateContext) && LOAD(eglCreatePbufferSurface) && LOAD(eglMakeCurrent) &&
	          LOAD(eglGetCurrentDisplay) && LOAD(eglGetCurrentSurface) && LOAD(eglGetCurrentContext) && LOAD(eglQueryAPI) &&
	          LOAD(eglDestroyContext) && LOAD(eglDestroySurface) && LOAD(eglQueryString);
	(void) LOAD(eglGetPlatformDisplayEXT);
	return ok;
}

static bool loadGl(struct Gles* g) {
	(void) (LOAD(glGetProgramBinary) && LOAD(glProgramBinary) && LOAD(glProgramParameteri));
	return LOAD(glCreateShader) && LOAD(glShaderSource) && LOAD(glCompileShader) && LOAD(glGetShaderiv) &&
	       LOAD(glGetShaderInfoLog) && LOAD(glDeleteShader) && LOAD(glCreateProgram) && LOAD(glAttachShader) &&
	       LOAD(glLinkProgram) && LOAD(glGetProgramiv) && LOAD(glGetProgramInfoLog) && LOAD(glUseProgram) &&
	       LOAD(glDeleteProgram) && LOAD(glGetUniformLocation) && LOAD(glUniform1iv) && LOAD(glGenBuffers) &&
	       LOAD(glDeleteBuffers) && LOAD(glBindBuffer) && LOAD(glBindBufferBase) && LOAD(glBufferData) &&
	       LOAD(glBufferSubData) && LOAD(glMapBufferRange) && LOAD(glUnmapBuffer) && LOAD(glDispatchCompute) &&
	       LOAD(glMemoryBarrier) && LOAD(glGetString) && LOAD(glFlush) && LOAD(glCopyBufferSubData);
}

// The context current on this thread, and how many contexts share the initialized display (EGL
// keeps one per display, so it is terminated with the last of them).
static _Thread_local struct Gles* current;
static EGLDisplay sharedDisplay;
static int displayUsers;

// What was current on this thread before one of these was made current (a frontend's context, or
// none, and the client API it was bound for: EGL keeps a current context per API), for glesYield to
// put back; and the context glesResume makes current again.
static _Thread_local struct {
	bool saved;
	EGLenum api;
	EGLDisplay display;
	EGLSurface draw, read;
	EGLContext context;
} before;
static _Thread_local struct Gles* yielded;

static void saveBefore(struct Gles* g) {
	if (!current && !before.saved) {
		before.saved = true;
		before.api = g->eglQueryAPI();
		before.display = g->eglGetCurrentDisplay();
		before.draw = g->eglGetCurrentSurface(EGL_DRAW);
		before.read = g->eglGetCurrentSurface(EGL_READ);
		before.context = g->eglGetCurrentContext();
	}
}

static void use(struct Gles* g) {
	if (!g->external && current != g) {
		saveBefore(g);
		g->eglBindAPI(EGL_OPENGL_ES_API);
		g->eglMakeCurrent(g->display, g->surface, g->surface, g->context);
		current = g;
	}
}

void glesYield(void) {
	if (current) {
		current->eglMakeCurrent(current->display, NULL, NULL, NULL);
		if (before.saved) {
			current->eglBindAPI(before.api);
			if (before.context) {
				current->eglMakeCurrent(before.display, before.draw, before.read, before.context);
			}
		}
	}
	before.saved = false;
	yielded = current;
	current = NULL;
}

void glesResume(void) {
	if (yielded) {
		use(yielded);
		yielded = NULL;
	}
}

static void release(struct Gles* g, EGLDisplay display) {
	if (display == sharedDisplay && displayUsers > 0) {
		return;
	}
	g->eglTerminate(display);
}

static bool fail(char* error, size_t size, const char* what) {
	snprintf(error, size, "%s", what);
	return false;
}

// A GLES 3.1 context on the display, current without a window (or with a 1x1 pbuffer).
static bool makeContext(struct Gles* g, EGLDisplay display) {
	g->eglBindAPI(EGL_OPENGL_ES_API);
	const EGLint attributes[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
	const EGLint pbufferAttributes[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE };
	const EGLint version[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
	EGLConfig config = NULL;
	EGLint count = 0;
	if (!g->eglChooseConfig(display, attributes, &config, 1, &count) || !count) {
		config = NULL;
	}
	EGLContext context = g->eglCreateContext(display, config, NULL, version);
	if (!context) {
		return false;
	}
	if (g->eglMakeCurrent(display, NULL, NULL, context)) {
		g->context = context;
		return true;
	}
	EGLConfig pbufferConfig = NULL;
	const EGLint size[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
	EGLSurface surface = NULL;
	if (g->eglChooseConfig(display, pbufferAttributes, &pbufferConfig, 1, &count) && count &&
	    (surface = g->eglCreatePbufferSurface(display, pbufferConfig, size)) && g->eglMakeCurrent(display, surface, surface, context)) {
		g->context = context;
		g->surface = surface;
		return true;
	}
	if (surface) {
		g->eglDestroySurface(display, surface);
	}
	g->eglDestroyContext(display, context);
	return false;
}

static bool context(struct Gles* g, char* error, size_t errorSize) {
	if (frontendProc) {
		g->external = true;
		if (!loadGl(g)) {
			return fail(error, errorSize, "incomplete GL from the frontend");
		}
		const char* renderer = (const char*) g->glGetString(GL_RENDERER);
		snprintf(g->renderer, sizeof(g->renderer), "%s", renderer ? renderer : "?");
		return true;
	}
	g->egl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!g->egl) {
		g->egl = dlopen("libEGL.so", RTLD_NOW | RTLD_LOCAL);
	}
	if (!g->egl) {
		return fail(error, errorSize, "no libEGL");
	}
	g->gles = dlopen("libGLESv2.so.2", RTLD_NOW | RTLD_LOCAL);
	*(void**) &g->eglGetProcAddress = dlsym(g->egl, "eglGetProcAddress");
	if (!loadAll(g)) {
		return fail(error, errorSize, "incomplete libEGL");
	}
	saveBefore(g);
	const char* clientExtensions = g->eglQueryString(NULL, EGL_EXTENSIONS);
	bool surfaceless = g->eglGetPlatformDisplayEXT && clientExtensions && strstr(clientExtensions, "EGL_MESA_platform_surfaceless");
	// The surfaceless platform first (no window system needed), then the default display.
	for (int attempt = surfaceless ? 0 : 1; attempt < 2 && !g->context; ++attempt) {
		EGLDisplay display = attempt == 0 ? g->eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL) : g->eglGetDisplay(NULL);
		if (!display || !g->eglInitialize(display, NULL, NULL)) {
			continue;
		}
		if (makeContext(g, display)) {
			g->display = display;
			sharedDisplay = display;
			++displayUsers;
			current = g;
		} else {
			release(g, display);
		}
	}
	if (!g->context) {
		return fail(error, errorSize, "no GLES 3.1 context");
	}
	if (!loadGl(g)) {
		return fail(error, errorSize, "incomplete GLES 3.1");
	}
	const char* renderer = (const char*) g->glGetString(GL_RENDERER);
	snprintf(g->renderer, sizeof(g->renderer), "%s", renderer ? renderer : "?");
	return true;
}

// A compute program from source, in use after; 0 when it doesn't build.
// A program just made, now the one in use and in the list.
static GLuint useNew(struct Gles* g, GLuint p) {
	g->glUseProgram(p);
	g->program = p;
	g->uniforms = g->glGetUniformLocation(p, "u");
	g->programs[g->programCount] = p;
	g->programUniforms[g->programCount++] = g->uniforms;
	return p;
}

// Compiled programs kept on disk (glesCacheDir), by their source and the driver: Mali's compiler
// takes seconds a program, every time.
static char cacheDir[4096];

void glesCacheDir(const char* dir) {
	snprintf(cacheDir, sizeof(cacheDir), "%s", dir ? dir : "");
}

static bool cachePath(struct Gles* g, const char* source, char* path, size_t size) {
	if (!cacheDir[0] || !g->glGetProgramBinary || !g->glProgramBinary || !g->glProgramParameteri) {
		return false;
	}
	const char* parts[3] = { (const char*) g->glGetString(GL_RENDERER), (const char*) g->glGetString(GL_VERSION), source };
	uint64_t h = 14695981039346656037ull;
	for (int i = 0; i < 3; ++i) {
		for (const char* c = parts[i] ? parts[i] : ""; *c; ++c) {
			h = (h ^ (uint8_t) *c) * 1099511628211ull;
		}
	}
	snprintf(path, size, "%s/%016llx.bin", cacheDir, (unsigned long long) h);
	return true;
}

static GLuint cachedProgram(struct Gles* g, const char* path) {
	FILE* f = fopen(path, "rb");
	if (!f) {
		return 0;
	}
	GLenum format = 0;
	void* binary = NULL;
	long size = fseek(f, 0, SEEK_END) == 0 ? ftell(f) - (long) sizeof(format) : -1;
	GLuint p = 0;
	if (size > 0 && fseek(f, 0, SEEK_SET) == 0 && fread(&format, sizeof(format), 1, f) == 1 && (binary = malloc((size_t) size)) &&
	    fread(binary, 1, (size_t) size, f) == (size_t) size) {
		p = g->glCreateProgram();
		g->glProgramBinary(p, format, binary, (GLsizei) size);
		GLint ok = 0;
		g->glGetProgramiv(p, GL_LINK_STATUS, &ok);
		if (!ok) {
			g->glDeleteProgram(p);
			p = 0;
		}
	}
	free(binary);
	fclose(f);
	return p;
}

static void saveProgram(struct Gles* g, GLuint p, const char* path) {
	GLint size = 0;
	g->glGetProgramiv(p, GL_PROGRAM_BINARY_LENGTH, &size);
	void* binary = size > 0 ? malloc((size_t) size) : NULL;
	GLenum format = 0;
	GLsizei length = 0;
	if (binary) {
		g->glGetProgramBinary(p, size, &length, &format, binary);
	}
	char temporary[4210];
	snprintf(temporary, sizeof(temporary), "%s.tmp", path);
	FILE* f = length > 0 ? fopen(temporary, "wb") : NULL;
	if (f) {
		bool ok = fwrite(&format, sizeof(format), 1, f) == 1 && fwrite(binary, 1, (size_t) length, f) == (size_t) length;
		if (fclose(f) == 0 && ok) {
			rename(temporary, path);
		} else {
			remove(temporary);
		}
	}
	free(binary);
}

static GLuint compileProgram(struct Gles* g, const char* source, char* error, size_t errorSize) {
	char path[4200];
	bool cached = cachePath(g, source, path, sizeof(path));
	GLuint p = cached ? cachedProgram(g, path) : 0;
	if (p) {
		return useNew(g, p);
	}
	GLuint shader = g->glCreateShader(GL_COMPUTE_SHADER);
	g->glShaderSource(shader, 1, &source, NULL);
	g->glCompileShader(shader);
	GLint ok = 0;
	g->glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		g->glGetShaderInfoLog(shader, (GLsizei) errorSize, NULL, error);
		g->glDeleteShader(shader);
		return 0;
	}
	p = g->glCreateProgram();
	if (cached) {
		g->glProgramParameteri(p, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, 1);
	}
	g->glAttachShader(p, shader);
	g->glLinkProgram(p);
	g->glDeleteShader(shader);
	g->glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok) {
		g->glGetProgramInfoLog(p, (GLsizei) errorSize, NULL, error);
		g->glDeleteProgram(p);
		return 0;
	}
	if (cached) {
		saveProgram(g, p, path);
	}
	return useNew(g, p);
}

static bool program(struct Gles* g, const char* source, char* error, size_t errorSize) {
	if (!compileProgram(g, source, error, errorSize)) {
		return false;
	}
	g->glGenBuffers(GLES_BUFFERS, g->buffer);
	for (int i = 0; i < GLES_BUFFERS; ++i) {
		g->glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buffer[i]);
		g->glBufferData(GL_SHADER_STORAGE_BUFFER, 16, NULL, GL_DYNAMIC_DRAW);
		g->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, g->buffer[i]);
	}
	return true;
}

struct Gles* glesCreateContext(char* error, size_t errorSize) {
	struct Gles* g = calloc(1, sizeof(*g));
	if (!g) {
		return NULL;
	}
	if (!context(g, error, errorSize)) {
		glesDestroy(g);
		return NULL;
	}
	return g;
}

void* glesProc(struct Gles* g, const char* name) {
	return lookup(g, name);
}

void glesUse(struct Gles* g) {
	use(g);
}

struct Gles* glesCreate(const char* source, char* error, size_t errorSize) {
	struct Gles* g = calloc(1, sizeof(*g));
	if (!g) {
		return NULL;
	}
	if (!context(g, error, errorSize) || !program(g, source, error, errorSize)) {
		glesDestroy(g);
		return NULL;
	}
	return g;
}

void glesDestroy(struct Gles* g) {
	if (!g) {
		return;
	}
	if (g->external) {
		if (g->program) {
			g->glDeleteBuffers(GLES_BUFFERS, g->buffer);
			g->glDeleteBuffers(1, &g->staging);
			for (int i = 0; i < g->programCount; ++i) {
				g->glDeleteProgram(g->programs[i]);
			}
		}
		free(g);
		return;
	}
	if (g->context) {
		use(g);
		if (g->program) {
			g->glDeleteBuffers(GLES_BUFFERS, g->buffer);
			g->glDeleteBuffers(1, &g->staging);
			for (int i = 0; i < g->programCount; ++i) {
				g->glDeleteProgram(g->programs[i]);
			}
		}
		g->eglMakeCurrent(g->display, NULL, NULL, NULL);
		g->eglDestroyContext(g->display, g->context);
		current = NULL;
		if (yielded == g) {
			yielded = NULL;
		}
		--displayUsers;
	}
	if (g->surface) {
		g->eglDestroySurface(g->display, g->surface);
	}
	if (g->display) {
		release(g, g->display);
	}
	if (g->gles) {
		dlclose(g->gles);
	}
	if (g->egl) {
		dlclose(g->egl);
	}
	free(g);
}

const char* glesRenderer(struct Gles* g) {
	return g->renderer;
}

bool glesInFrontend(void) {
	return frontendProc != NULL;
}

void glesData(struct Gles* g, int binding, const void* data, size_t size) {
	use(g);
	g->glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buffer[binding]);
	g->glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr) (size ? size : 16), data, GL_DYNAMIC_DRAW);
	g->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, g->buffer[binding]);
}

void glesSubData(struct Gles* g, int binding, size_t offset, const void* data, size_t size) {
	use(g);
	g->glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buffer[binding]);
	g->glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr) offset, (GLsizeiptr) size, data);
}

const void* glesMap(struct Gles* g, int binding, size_t offset, size_t size) {
	use(g);
	g->glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	if (!g->staging) {
		g->glGenBuffers(1, &g->staging);
	}
	g->glBindBuffer(GL_COPY_READ_BUFFER, g->buffer[binding]);
	g->glBindBuffer(GL_COPY_WRITE_BUFFER, g->staging);
	if (size > g->stagingSize) {
		g->glBufferData(GL_COPY_WRITE_BUFFER, (GLsizeiptr) size, NULL, GL_STREAM_READ);
		g->stagingSize = size;
	}
	g->glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, (GLintptr) offset, 0, (GLsizeiptr) size);
	return g->glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, (GLsizeiptr) size, GL_MAP_READ_BIT);
}

const void* glesMapRanges(struct Gles* g, const int* bindings, const size_t* offsets, const size_t* sizes, int count) {
	use(g);
	g->glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	size_t total = 0;
	for (int i = 0; i < count; ++i) {
		total += sizes[i];
	}
	if (!g->staging) {
		g->glGenBuffers(1, &g->staging);
	}
	g->glBindBuffer(GL_COPY_WRITE_BUFFER, g->staging);
	if (total > g->stagingSize) {
		g->glBufferData(GL_COPY_WRITE_BUFFER, (GLsizeiptr) total, NULL, GL_STREAM_READ);
		g->stagingSize = total;
	}
	size_t at = 0;
	for (int i = 0; i < count; ++i) {
		g->glBindBuffer(GL_COPY_READ_BUFFER, g->buffer[bindings[i]]);
		g->glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, (GLintptr) offsets[i], (GLintptr) at, (GLsizeiptr) sizes[i]);
		at += sizes[i];
	}
	return g->glMapBufferRange(GL_COPY_WRITE_BUFFER, 0, (GLsizeiptr) total, GL_MAP_READ_BIT);
}

void glesUnmap(struct Gles* g, int binding) {
	use(g);
	g->glBindBuffer(GL_COPY_WRITE_BUFFER, g->staging);
	g->glUnmapBuffer(GL_COPY_WRITE_BUFFER);
}

int glesAddProgram(struct Gles* g, const char* source, char* error, size_t errorSize) {
	use(g);
	if (g->programCount == GLES_PROGRAMS || !compileProgram(g, source, error, errorSize)) {
		return -1;
	}
	return g->programCount - 1;
}

unsigned glesBuffer(struct Gles* g, int binding) {
	return g->buffer[binding];
}

void glesSelectProgram(struct Gles* g, int index) {
	use(g);
	if (g->program != g->programs[index]) {
		g->program = g->programs[index];
		g->uniforms = g->programUniforms[index];
		g->glUseProgram(g->program);
	}
}

void glesUniforms(struct Gles* g, const int* values, int count) {
	use(g);
	g->glUniform1iv(g->uniforms, count, values);
}

void glesDispatch(struct Gles* g, unsigned x, unsigned y) {
	use(g);
	g->glDispatchCompute(x, y, 1);
	// Its writes seen by later dispatches, and kept by later uploads into the same buffer.
	g->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	// Each dispatch its own submission: the kernel's GPU hang check times submissions.
	g->glFlush();
}
