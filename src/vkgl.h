#pragma once

// The GL that the GL renderer (glr.c) and the core's picture code use, done with Vulkan: one device,
// either the frontend's or one of its own, its GL objects and state kept here. A GL name of any kind
// is only valid with the context that made it.

#include <stdbool.h>
#include <stdint.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

// The frontend's device (RetroArch's Vulkan interface), with what it asks of a core: its queue taken
// between lockQueue and unlockQueue, and the picture's image handed over with setImage.
struct VkglFrontend {
	VkInstance instance;
	VkPhysicalDevice gpu;
	VkDevice device;
	VkQueue queue;
	uint32_t queueFamily;
	PFN_vkGetInstanceProcAddr getInstanceProcAddr;
	PFN_vkGetDeviceProcAddr getDeviceProcAddr;
	void* handle;
	void (*lockQueue)(void* handle);
	void (*unlockQueue)(void* handle);
	uint32_t (*syncIndex)(void* handle);
	uint32_t (*syncIndexMask)(void* handle);
};

// A device the frontend uses as well (libretro's context negotiation): on gpu, or the best GPU when
// that's VK_NULL_HANDLE, with a queue for graphics and compute that presents to surface (or a second
// that does), and what the context draws better with that the GPU has. create makes it (the
// frontend's wrapper of vkCreateDevice, which adds what the frontend needs).
struct VkglDevice {
	VkPhysicalDevice gpu;
	VkDevice device;
	VkQueue queue, presentQueue;
	uint32_t queueFamily, presentFamily;
};

bool vkglMakeDevice(VkInstance instance, VkPhysicalDevice gpu, VkSurfaceKHR surface, PFN_vkGetInstanceProcAddr get,
                    VkDevice (*create)(VkPhysicalDevice, void*, const VkDeviceCreateInfo*), void* opaque, struct VkglDevice* out);

// Starts the context: on the frontend's device, or with frontend NULL, on a device of its own. False,
// with why in error, when Vulkan or a usable device isn't there.
bool vkglCreate(const struct VkglFrontend* frontend, char* error, size_t errorSize);
void vkglDestroy(void);
bool vkglActive(void);
// A GL function by name ("glDrawArrays"), or NULL.
void* vkglProc(const char* name);
const char* vkglRenderer(void);
// Whether draws read their own pixels in rasterization order, with no barriers.
bool vkglOrdered(void);

// The framebuffer name the frontend's picture is drawn to (with a blit), width by height pixels; then
// vkglPicture submits the work and gives the image to pass to the frontend, in layout.
unsigned vkglFrontendFramebuffer(int width, int height);
bool vkglPicture(VkImage* image, VkImageView* view, VkImageLayout* layout, VkImageViewCreateInfo* info);
