#pragma once

#include <stdbool.h>
#include <stddef.h>

// A GLES 3.1 compute context without a window, loaded at run time (libEGL/libGLESv2 through dlopen),
// with one compute program and storage buffers at bindings 0-3.
struct Gles;

struct Gles* glesCreate(const char* source, char* error, size_t errorSize);
void glesDestroy(struct Gles*);
const char* glesRenderer(struct Gles*);
// Gives a buffer new storage of size bytes (data may be NULL).
void glesData(struct Gles*, int binding, const void* data, size_t size);
void glesSubData(struct Gles*, int binding, size_t offset, const void* data, size_t size);
// Maps [offset, offset + size) of a buffer for reading after the dispatches so far; NULL on failure.
const void* glesMap(struct Gles*, int binding, size_t offset, size_t size);
void glesUnmap(struct Gles*, int binding);
// The program's "uniform int u[N]".
void glesUniforms(struct Gles*, const int* values, int count);
void glesDispatch(struct Gles*, unsigned x, unsigned y);
