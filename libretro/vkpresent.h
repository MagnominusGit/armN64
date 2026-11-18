#pragma once

#include <stdbool.h>
#include "libretro.h"

// Pictures for a Vulkan frontend, drawn by the core's own GL context: a Vulkan image per frontend
// frame, shared with GL through a dma-buf (or, without the extensions for that, read back into a
// mapped one).

// Asks the frontend for a Vulkan context with the device extensions the sharing needs; fills hw.
// False when the frontend has none (or the core was built without Vulkan's headers).
bool vkPresentAsk(retro_environment_t environment, struct retro_hw_render_callback* hw);
// The frontend's context, made (again): true when pictures can go to it. getProc finds GL and EGL
// functions; the core's GL context is current.
bool vkPresentReset(retro_environment_t environment, void* (*getProc)(const char* name));
// The context is going: its images, and with gl the GL objects on them (the GL context current).
void vkPresentDestroy(bool gl);
// The framebuffer, in the current GL context, a width by height picture goes to this frame (its top
// row at y 0); 0 when there is none.
unsigned vkPresentTarget(int width, int height);
// Gives the frontend the picture drawn in the target, before RETRO_HW_FRAME_BUFFER_VALID is
// refreshed.
void vkPresentSubmit(void);
// Whether the GL and Vulkan images share their memory (else each picture is read back).
bool vkPresentShared(void);
