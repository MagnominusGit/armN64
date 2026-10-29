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
	EGL_PLATFORM_SURFACELESS_MESA = 0x31DD,
	GL_COMPUTE_SHADER = 0x91B9, GL_COMPILE_STATUS = 0x8B81, GL_LINK_STATUS = 0x8B82, GL_INFO_LOG_LENGTH = 0x8B84,
	GL_SHADER_STORAGE_BUFFER = 0x90D2, GL_DYNAMIC_DRAW = 0x88E8, GL_MAP_READ_BIT = 0x0001,
	GL_SHADER_STORAGE_BARRIER_BIT = 0x2000, GL_BUFFER_UPDATE_BARRIER_BIT = 0x0200, GL_RENDERER = 0x1F01,
};

struct Gles {
	void* egl;
	void* gles;
	EGLDisplay display;
	EGLContext context;
	EGLSurface surface;
	GLuint program, buffer[4];
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
	const unsigned char* (*glGetString)(GLenum);
};



static void* lookup(struct Gles* g, const char* name) {
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
	          LOAD(eglDestroyContext) && LOAD(eglDestroySurface) && LOAD(eglQueryString);
	(void) LOAD(eglGetPlatformDisplayEXT);
	return ok;
}

static bool loadGl(struct Gles* g) {
	return LOAD(glCreateShader) && LOAD(glShaderSource) && LOAD(glCompileShader) && LOAD(glGetShaderiv) &&
	       LOAD(glGetShaderInfoLog) && LOAD(glDeleteShader) && LOAD(glCreateProgram) && LOAD(glAttachShader) &&
	       LOAD(glLinkProgram) && LOAD(glGetProgramiv) && LOAD(glGetProgramInfoLog) && LOAD(glUseProgram) &&
	       LOAD(glDeleteProgram) && LOAD(glGetUniformLocation) && LOAD(glUniform1iv) && LOAD(glGenBuffers) &&
	       LOAD(glDeleteBuffers) && LOAD(glBindBuffer) && LOAD(glBindBufferBase) && LOAD(glBufferData) &&
	       LOAD(glBufferSubData) && LOAD(glMapBufferRange) && LOAD(glUnmapBuffer) && LOAD(glDispatchCompute) &&
	       LOAD(glMemoryBarrier) && LOAD(glGetString) && LOAD(glFlush);
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
		} else {
			g->eglTerminate(display);
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

static bool program(struct Gles* g, const char* source, char* error, size_t errorSize) {
	GLuint shader = g->glCreateShader(GL_COMPUTE_SHADER);
	g->glShaderSource(shader, 1, &source, NULL);
	g->glCompileShader(shader);
	GLint ok = 0;
	g->glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		g->glGetShaderInfoLog(shader, (GLsizei) errorSize, NULL, error);
		g->glDeleteShader(shader);
		return false;
	}
	g->program = g->glCreateProgram();
	g->glAttachShader(g->program, shader);
	g->glLinkProgram(g->program);
	g->glDeleteShader(shader);
	g->glGetProgramiv(g->program, GL_LINK_STATUS, &ok);
	if (!ok) {
		g->glGetProgramInfoLog(g->program, (GLsizei) errorSize, NULL, error);
		return false;
	}
	g->glUseProgram(g->program);
	g->uniforms = g->glGetUniformLocation(g->program, "u");
	g->glGenBuffers(4, g->buffer);
	for (int i = 0; i < 4; ++i) {
		g->glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buffer[i]);
		g->glBufferData(GL_SHADER_STORAGE_BUFFER, 16, NULL, GL_DYNAMIC_DRAW);
		g->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, g->buffer[i]);
	}
	return true;
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
	if (g->context) {
		if (g->program) {
			g->glDeleteBuffers(4, g->buffer);
			g->glDeleteProgram(g->program);
		}
		g->eglMakeCurrent(g->display, NULL, NULL, NULL);
		g->eglDestroyContext(g->display, g->context);
	}
	if (g->surface) {
		g->eglDestroySurface(g->display, g->surface);
	}
	if (g->display) {
		g->eglTerminate(g->display);
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

void glesData(struct Gles* g, int binding, const void* data, size_t size) {
	g->glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buffer[binding]);
	g->glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr) (size ? size : 16), data, GL_DYNAMIC_DRAW);
	g->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, g->buffer[binding]);
}

void glesSubData(struct Gles* g, int binding, size_t offset, const void* data, size_t size) {
	g->glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buffer[binding]);
	g->glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr) offset, (GLsizeiptr) size, data);
}

const void* glesMap(struct Gles* g, int binding, size_t offset, size_t size) {
	g->glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
	g->glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buffer[binding]);
	return g->glMapBufferRange(GL_SHADER_STORAGE_BUFFER, (GLintptr) offset, (GLsizeiptr) size, GL_MAP_READ_BIT);
}

void glesUnmap(struct Gles* g, int binding) {
	g->glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->buffer[binding]);
	g->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

void glesUniforms(struct Gles* g, const int* values, int count) {
	g->glUniform1iv(g->uniforms, count, values);
}

void glesDispatch(struct Gles* g, unsigned x, unsigned y) {
	g->glDispatchCompute(x, y, 1);
	g->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	// Each dispatch its own submission: the kernel's GPU hang check times submissions.
	g->glFlush();
}
