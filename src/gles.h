#pragma once

#include <stdbool.h>
#include <stddef.h>

// A GLES 3.1 compute context without a window, loaded at run time (libEGL/libGLESv2 through dlopen),
// with one compute program and storage buffers at bindings 0-4. Several can be used on one thread;
// each call makes its own context current.
struct Gles;

enum { GLES_BUFFERS = 6 };

struct Gles* glesCreate(const char* source, char* error, size_t errorSize);
// A context without a program, for code making its own GL calls: glesProc finds them, glesUse
// makes the context current first.
struct Gles* glesCreateContext(char* error, size_t errorSize);
void* glesProc(struct Gles*, const char* name);
void glesUse(struct Gles*);
// Makes current again what was current on this thread before these contexts (a frontend's own), and
// glesResume the context that was current then, for code that doesn't make its own current first.
void glesYield(void);
void glesResume(void);
// From now on, contexts are the frontend's own, current already when they are used (libretro's
// shared hardware context), with GL functions from getProc.
void glesFrontend(void* (*getProc)(const char* name));
void glesDestroy(struct Gles*);
const char* glesRenderer(struct Gles*);
// Whether contexts are the frontend's (glesFrontend).
bool glesInFrontend(void);
// Gives a buffer new storage of size bytes (data may be NULL).
void glesData(struct Gles*, int binding, const void* data, size_t size);
void glesSubData(struct Gles*, int binding, size_t offset, const void* data, size_t size);
// Maps [offset, offset + size) of a buffer for reading after the dispatches so far; NULL on failure.
const void* glesMap(struct Gles*, int binding, size_t offset, size_t size);
void glesUnmap(struct Gles*, int binding);
// Maps ranges of buffers at once, one after another (glesUnmap when done).
const void* glesMapRanges(struct Gles*, const int* bindings, const size_t* offsets, const size_t* sizes, int count);
// Another program for the same buffers (its source built like glesCreate's), in use after: its index
// for glesSelectProgram, -1 when it doesn't build or there are GLES_PROGRAMS already.
enum { GLES_PROGRAMS = 64 };
int glesAddProgram(struct Gles*, const char* source, char* error, size_t errorSize);
void glesSelectProgram(struct Gles*, int index);
// The GL name of the buffer at a binding (for a frontend sharing the context).
unsigned glesBuffer(struct Gles*, int binding);
// The program's "uniform int u[N]".
void glesUniforms(struct Gles*, const int* values, int count);
void glesDispatch(struct Gles*, unsigned x, unsigned y);
