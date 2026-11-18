#include "vkpresent.h"

#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#define ARMN64_VULKAN 1
#endif
#endif

#ifndef ARMN64_VULKAN

bool vkPresentAsk(retro_environment_t environment, struct retro_hw_render_callback* hw) {
	return false;
}
bool vkPresentReset(retro_environment_t environment, void* (*getProc)(const char* name)) {
	return false;
}
void vkPresentDestroy(bool gl) {
}
unsigned vkPresentTarget(int width, int height) {
	return 0;
}
void vkPresentSubmit(void) {
}
bool vkPresentShared(void) {
	return false;
}

#else

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "libretro_vulkan.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef void* EGLDisplay;
typedef void* EGLImage;
typedef void* EGLSync;
typedef int32_t EGLint;

enum {
	GL_TEXTURE_2D = 0x0DE1, GL_FRAMEBUFFER = 0x8D40, GL_READ_FRAMEBUFFER = 0x8CA8, GL_COLOR_ATTACHMENT0 = 0x8CE0,
	GL_FRAMEBUFFER_COMPLETE = 0x8CD5, GL_RGBA = 0x1908, GL_RGBA8 = 0x8058, GL_UNSIGNED_BYTE = 0x1401, GL_PACK_ALIGNMENT = 0x0D05,
	GL_PACK_ROW_LENGTH = 0x0D02, GL_TEXTURE_BINDING_2D = 0x8069, GL_DRAW_FRAMEBUFFER_BINDING = 0x8CA6, GL_READ_FRAMEBUFFER_BINDING = 0x8CAA,
	GL_DRAW_FRAMEBUFFER = 0x8CA9,
	EGL_NONE = 0x3038, EGL_EXTENSIONS = 0x3055, EGL_WIDTH = 0x3057, EGL_HEIGHT = 0x3056, EGL_LINUX_DMA_BUF_EXT = 0x3270,
	EGL_LINUX_DRM_FOURCC_EXT = 0x3271, EGL_DMA_BUF_PLANE0_FD_EXT = 0x3272, EGL_DMA_BUF_PLANE0_OFFSET_EXT = 0x3273,
	EGL_DMA_BUF_PLANE0_PITCH_EXT = 0x3274, EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT = 0x3443, EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT = 0x3444,
	EGL_SYNC_NATIVE_FENCE_ANDROID = 0x3144,
	// R8G8B8A8 in memory, as VK_FORMAT_R8G8B8A8_UNORM and GL_RGBA8 keep it.
	DRM_FORMAT_ABGR8888 = 0x34324241,
};

static struct {
	void (*GenTextures)(int, unsigned*);
	void (*DeleteTextures)(int, const unsigned*);
	void (*BindTexture)(unsigned, unsigned);
	void (*TexImage2D)(unsigned, int, int, int, int, int, unsigned, unsigned, const void*);
	void (*GenFramebuffers)(int, unsigned*);
	void (*DeleteFramebuffers)(int, const unsigned*);
	void (*BindFramebuffer)(unsigned, unsigned);
	void (*FramebufferTexture2D)(unsigned, unsigned, unsigned, unsigned, int);
	unsigned (*CheckFramebufferStatus)(unsigned);
	void (*Flush)(void);
	void (*Finish)(void);
	void (*ReadPixels)(int, int, int, int, unsigned, unsigned, void*);
	void (*PixelStorei)(unsigned, int);
	void (*GetIntegerv)(unsigned, int*);
	EGLDisplay (*GetCurrentDisplay)(void);
	const char* (*QueryString)(EGLDisplay, EGLint);
	// Optional: the dma-buf images and native fences.
	void (*EGLImageTargetTexture2DOES)(unsigned, void*);
	EGLImage (*CreateImageKHR)(EGLDisplay, void*, unsigned, void*, const EGLint*);
	unsigned (*DestroyImageKHR)(EGLDisplay, EGLImage);
	EGLSync (*CreateSyncKHR)(EGLDisplay, unsigned, const EGLint*);
	unsigned (*DestroySyncKHR)(EGLDisplay, EGLSync);
	EGLint (*DupNativeFenceFDANDROID)(EGLDisplay, EGLSync);
} gl;
static const char* const glNames[] = { "glGenTextures", "glDeleteTextures", "glBindTexture", "glTexImage2D", "glGenFramebuffers",
	                                   "glDeleteFramebuffers", "glBindFramebuffer", "glFramebufferTexture2D", "glCheckFramebufferStatus",
	                                   "glFlush", "glFinish", "glReadPixels", "glPixelStorei", "glGetIntegerv", "eglGetCurrentDisplay",
	                                   "eglQueryString", "glEGLImageTargetTexture2DOES", "eglCreateImageKHR", "eglDestroyImageKHR",
	                                   "eglCreateSyncKHR", "eglDestroySyncKHR", "eglDupNativeFenceFDANDROID" };
enum { GL_REQUIRED = 16 };

static struct {
	PFN_vkCreateImage CreateImage;
	PFN_vkDestroyImage DestroyImage;
	PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
	PFN_vkAllocateMemory AllocateMemory;
	PFN_vkFreeMemory FreeMemory;
	PFN_vkBindImageMemory BindImageMemory;
	PFN_vkMapMemory MapMemory;
	PFN_vkGetImageSubresourceLayout GetImageSubresourceLayout;
	PFN_vkCreateImageView CreateImageView;
	PFN_vkDestroyImageView DestroyImageView;
	PFN_vkCreateSemaphore CreateSemaphore;
	PFN_vkDestroySemaphore DestroySemaphore;
	PFN_vkCreateCommandPool CreateCommandPool;
	PFN_vkDestroyCommandPool DestroyCommandPool;
	PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
	PFN_vkFreeCommandBuffers FreeCommandBuffers;
	PFN_vkBeginCommandBuffer BeginCommandBuffer;
	PFN_vkEndCommandBuffer EndCommandBuffer;
	PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
	PFN_vkQueueSubmit QueueSubmit;
	PFN_vkQueueWaitIdle QueueWaitIdle;
	// Optional: from the extensions create_device enabled.
	PFN_vkGetMemoryFdKHR GetMemoryFdKHR;
	PFN_vkImportSemaphoreFdKHR ImportSemaphoreFdKHR;
} vk;
static const char* const vkNames[] = { "vkCreateImage", "vkDestroyImage", "vkGetImageMemoryRequirements", "vkAllocateMemory", "vkFreeMemory",
	                                   "vkBindImageMemory", "vkMapMemory", "vkGetImageSubresourceLayout", "vkCreateImageView",
	                                   "vkDestroyImageView", "vkCreateSemaphore", "vkDestroySemaphore", "vkCreateCommandPool",
	                                   "vkDestroyCommandPool", "vkAllocateCommandBuffers", "vkFreeCommandBuffers", "vkBeginCommandBuffer",
	                                   "vkEndCommandBuffer", "vkCmdPipelineBarrier", "vkQueueSubmit", "vkQueueWaitIdle", "vkGetMemoryFdKHR",
	                                   "vkImportSemaphoreFdKHR" };
enum { VK_REQUIRED = 21 };

// The device extensions the sharing uses, enabled by createDevice where the GPU has them.
static const char* const wanted[] = { "VK_KHR_external_memory", "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
	                                  "VK_KHR_external_semaphore", "VK_KHR_external_semaphore_fd", "VK_KHR_dedicated_allocation",
	                                  "VK_KHR_get_memory_requirements2" };
enum { WANTED = sizeof(wanted) / sizeof(*wanted) };
static bool enabled[WANTED];

// An image per frontend frame (its sync index): the Vulkan image, and the GL texture and framebuffer on
// its memory (shared) or beside it (read back into its mapping).
enum { SLOTS = 8 };
static struct Slot {
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
	VkSemaphore semaphore;
	struct retro_vulkan_image info;
	EGLImage eglImage;
	unsigned texture, framebuffer;
	uint8_t* mapped;
	VkDeviceSize pitch;
	int width, height;
} slots[SLOTS];

static struct retro_hw_render_interface_vulkan* frontend;
static VkPhysicalDeviceMemoryProperties memoryTypes;
static VkCommandPool pool;
static EGLDisplay display;
static bool sharing, fences, modifiers;
static int drawn = -1;

static bool createDevice(struct retro_vulkan_context* context, VkInstance instance, VkPhysicalDevice gpu, VkSurfaceKHR surface,
                         PFN_vkGetInstanceProcAddr instanceProc, const char** required, unsigned requiredCount, const char** layers,
                         unsigned layerCount, const VkPhysicalDeviceFeatures* features) {
	PFN_vkEnumeratePhysicalDevices enumerate = (PFN_vkEnumeratePhysicalDevices) instanceProc(instance, "vkEnumeratePhysicalDevices");
	PFN_vkGetPhysicalDeviceQueueFamilyProperties families =
	    (PFN_vkGetPhysicalDeviceQueueFamilyProperties) instanceProc(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
	PFN_vkEnumerateDeviceExtensionProperties extensions =
	    (PFN_vkEnumerateDeviceExtensionProperties) instanceProc(instance, "vkEnumerateDeviceExtensionProperties");
	PFN_vkGetPhysicalDeviceSurfaceSupportKHR presents =
	    surface ? (PFN_vkGetPhysicalDeviceSurfaceSupportKHR) instanceProc(instance, "vkGetPhysicalDeviceSurfaceSupportKHR") : NULL;
	PFN_vkCreateDevice create = (PFN_vkCreateDevice) instanceProc(instance, "vkCreateDevice");
	PFN_vkGetDeviceProcAddr deviceProc = (PFN_vkGetDeviceProcAddr) instanceProc(instance, "vkGetDeviceProcAddr");
	if (!enumerate || !families || !extensions || !create || !deviceProc) {
		return false;
	}
	if (!gpu) {
		uint32_t count = 1;
		VkResult r = enumerate(instance, &count, &gpu);
		if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !count) {
			return false;
		}
	}
	// A queue for graphics and compute that presents to the surface.
	VkQueueFamilyProperties props[16];
	uint32_t familyCount = 16;
	families(gpu, &familyCount, props);
	uint32_t family = UINT32_MAX;
	for (uint32_t i = 0; i < familyCount && family == UINT32_MAX; ++i) {
		VkBool32 present = VK_TRUE;
		if (presents && presents(gpu, i, surface, &present) != VK_SUCCESS) {
			present = VK_FALSE;
		}
		VkQueueFlags const both = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
		if ((props[i].queueFlags & both) == both && present) {
			family = i;
		}
	}
	if (family == UINT32_MAX) {
		return false;
	}
	uint32_t available = 0;
	extensions(gpu, NULL, &available, NULL);
	VkExtensionProperties* list = calloc(available ? available : 1, sizeof(*list));
	if (!list) {
		return false;
	}
	extensions(gpu, NULL, &available, list);
	const char* names[64];
	unsigned count = 0;
	for (unsigned i = 0; i < requiredCount && count < 64; ++i) {
		names[count++] = required[i];
	}
	for (unsigned w = 0; w < WANTED; ++w) {
		enabled[w] = false;
		bool already = false;
		for (unsigned i = 0; i < count; ++i) {
			already |= !strcmp(names[i], wanted[w]);
		}
		for (uint32_t i = 0; i < available && !enabled[w]; ++i) {
			enabled[w] = !strcmp(list[i].extensionName, wanted[w]);
		}
		if (enabled[w] && !already && count < 64) {
			names[count++] = wanted[w];
		}
	}
	free(list);
	float const priority = 1.0f;
	VkDeviceQueueCreateInfo const queue = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, family, 1, &priority };
	VkDeviceCreateInfo const info = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &queue, layerCount, layers, count, names, features };
	VkDevice device = VK_NULL_HANDLE;
	if (create(gpu, &info, NULL, &device) != VK_SUCCESS) {
		memset(enabled, 0, sizeof(enabled));
		return false;
	}
	PFN_vkGetDeviceQueue getQueue = (PFN_vkGetDeviceQueue) deviceProc(device, "vkGetDeviceQueue");
	VkQueue q = VK_NULL_HANDLE;
	getQueue(device, family, 0, &q);
	context->gpu = gpu;
	context->device = device;
	context->queue = q;
	context->queue_family_index = family;
	context->presentation_queue = q;
	context->presentation_queue_family_index = family;
	return true;
}

static const struct retro_hw_render_context_negotiation_interface_vulkan negotiation = {
	.interface_type = RETRO_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE_VULKAN,
	.interface_version = 1,
	.create_device = createDevice,
};

bool vkPresentAsk(retro_environment_t environment, struct retro_hw_render_callback* hw) {
	hw->context_type = RETRO_HW_CONTEXT_VULKAN;
	hw->version_major = VK_MAKE_VERSION(1, 0, 0);
	hw->version_minor = 0;
	if (!environment(RETRO_ENVIRONMENT_SET_HW_RENDER, hw)) {
		return false;
	}
	memset(enabled, 0, sizeof(enabled));
	environment(RETRO_ENVIRONMENT_SET_HW_RENDER_CONTEXT_NEGOTIATION_INTERFACE, (void*) &negotiation);
	return true;
}

static bool has(const char* list, const char* name) {
	size_t const n = strlen(name);
	for (const char* p = list; p && (p = strstr(p, name)) != NULL; p += n) {
		if ((p == list || p[-1] == ' ') && (p[n] == ' ' || p[n] == 0)) {
			return true;
		}
	}
	return false;
}

static bool extensionOn(const char* name) {
	for (unsigned w = 0; w < WANTED; ++w) {
		if (!strcmp(wanted[w], name)) {
			return enabled[w];
		}
	}
	return false;
}

bool vkPresentReset(retro_environment_t environment, void* (*getProc)(const char* name)) {
	frontend = NULL;
	struct retro_hw_render_interface_vulkan* face = NULL;
	if (!environment(RETRO_ENVIRONMENT_GET_HW_RENDER_INTERFACE, (void*) &face) || !face || face->interface_type != RETRO_HW_RENDER_INTERFACE_VULKAN ||
	    face->interface_version < RETRO_HW_RENDER_INTERFACE_VULKAN_VERSION) {
		return false;
	}
	void** fns = (void**) &vk;
	for (size_t i = 0; i < sizeof(vkNames) / sizeof(*vkNames); ++i) {
		fns[i] = (void*) face->get_device_proc_addr(face->device, vkNames[i]);
		if (!fns[i] && i < VK_REQUIRED) {
			return false;
		}
	}
	fns = (void**) &gl;
	for (size_t i = 0; i < sizeof(glNames) / sizeof(*glNames); ++i) {
		fns[i] = getProc(glNames[i]);
		if (!fns[i] && i < GL_REQUIRED) {
			return false;
		}
	}
	PFN_vkGetPhysicalDeviceMemoryProperties memoryProperties =
	    (PFN_vkGetPhysicalDeviceMemoryProperties) face->get_instance_proc_addr(face->instance, "vkGetPhysicalDeviceMemoryProperties");
	if (!memoryProperties) {
		return false;
	}
	memoryProperties(face->gpu, &memoryTypes);
	VkCommandPoolCreateInfo const poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		                                       face->queue_index };
	if (vk.CreateCommandPool(face->device, &poolInfo, NULL, &pool) != VK_SUCCESS) {
		return false;
	}
	display = gl.GetCurrentDisplay();
	const char* egl = display ? gl.QueryString(display, EGL_EXTENSIONS) : NULL;
	sharing = has(egl, "EGL_EXT_image_dma_buf_import") && gl.EGLImageTargetTexture2DOES && gl.CreateImageKHR && gl.DestroyImageKHR &&
	          vk.GetMemoryFdKHR && extensionOn("VK_KHR_external_memory_fd") && extensionOn("VK_EXT_external_memory_dma_buf") &&
	          extensionOn("VK_KHR_external_memory");
	modifiers = has(egl, "EGL_EXT_image_dma_buf_import_modifiers");
	fences = sharing && has(egl, "EGL_ANDROID_native_fence_sync") && gl.CreateSyncKHR && gl.DestroySyncKHR && gl.DupNativeFenceFDANDROID &&
	         vk.ImportSemaphoreFdKHR && extensionOn("VK_KHR_external_semaphore_fd") && extensionOn("VK_KHR_external_semaphore");
	frontend = face;
	drawn = -1;
	return true;
}

bool vkPresentShared(void) {
	return sharing;
}

static uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want) {
	for (uint32_t i = 0; i < memoryTypes.memoryTypeCount; ++i) {
		if ((bits & (1u << i)) && (memoryTypes.memoryTypes[i].propertyFlags & want) == want) {
			return i;
		}
	}
	return UINT32_MAX;
}

// The image's layout from its first one to GENERAL, where the frontend reads it, before GL draws.
static bool toGeneral(VkImage image, VkImageLayout from) {
	VkDevice const device = frontend->device;
	VkCommandBufferAllocateInfo const allocate = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	if (vk.AllocateCommandBuffers(device, &allocate, &cmd) != VK_SUCCESS) {
		return false;
	}
	VkCommandBufferBeginInfo const begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
	vk.BeginCommandBuffer(cmd, &begin);
	VkImageMemoryBarrier const barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		                                   NULL,
		                                   from == VK_IMAGE_LAYOUT_PREINITIALIZED ? VK_ACCESS_HOST_WRITE_BIT : 0,
		                                   VK_ACCESS_SHADER_READ_BIT,
		                                   from,
		                                   VK_IMAGE_LAYOUT_GENERAL,
		                                   VK_QUEUE_FAMILY_IGNORED,
		                                   VK_QUEUE_FAMILY_IGNORED,
		                                   image,
		                                   { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
	vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL,
	                      0, NULL, 1, &barrier);
	vk.EndCommandBuffer(cmd);
	VkSubmitInfo const submit = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd, 0, NULL };
	if (frontend->lock_queue) {
		frontend->lock_queue(frontend->handle);
	}
	bool const ok = vk.QueueSubmit(frontend->queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vk.QueueWaitIdle(frontend->queue) == VK_SUCCESS;
	if (frontend->unlock_queue) {
		frontend->unlock_queue(frontend->handle);
	}
	vk.FreeCommandBuffers(device, pool, 1, &cmd);
	return ok;
}

static void freeSlot(struct Slot* s, bool withGl) {
	VkDevice const device = frontend ? frontend->device : VK_NULL_HANDLE;
	if (withGl) {
		if (s->framebuffer) {
			gl.DeleteFramebuffers(1, &s->framebuffer);
		}
		if (s->texture) {
			gl.DeleteTextures(1, &s->texture);
		}
		if (s->eglImage) {
			gl.DestroyImageKHR(display, s->eglImage);
		}
	}
	if (device) {
		if (s->view) {
			vk.DestroyImageView(device, s->view, NULL);
		}
		if (s->image) {
			vk.DestroyImage(device, s->image, NULL);
		}
		if (s->memory) {
			vk.FreeMemory(device, s->memory, NULL);
		}
		if (s->semaphore) {
			vk.DestroySemaphore(device, s->semaphore, NULL);
		}
	}
	memset(s, 0, sizeof(*s));
}

static bool makeSlot(struct Slot* s, int width, int height) {
	VkDevice const device = frontend->device;
	VkExternalMemoryImageCreateInfo const external = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, NULL,
		                                               VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
	VkImageLayout const first = sharing ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_PREINITIALIZED;
	VkImageCreateInfo const imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		                                  sharing ? &external : NULL,
		                                  0,
		                                  VK_IMAGE_TYPE_2D,
		                                  VK_FORMAT_R8G8B8A8_UNORM,
		                                  { (uint32_t) width, (uint32_t) height, 1 },
		                                  1,
		                                  1,
		                                  VK_SAMPLE_COUNT_1_BIT,
		                                  VK_IMAGE_TILING_LINEAR,
		                                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
		                                  VK_SHARING_MODE_EXCLUSIVE,
		                                  0,
		                                  NULL,
		                                  first };
	if (vk.CreateImage(device, &imageInfo, NULL, &s->image) != VK_SUCCESS) {
		s->image = VK_NULL_HANDLE;
		return false;
	}
	VkMemoryRequirements need;
	vk.GetImageMemoryRequirements(device, s->image, &need);
	uint32_t type = sharing ? memoryType(need.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
	                        : memoryType(need.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	if (type == UINT32_MAX) {
		type = memoryType(need.memoryTypeBits, sharing ? 0 : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	}
	if (type == UINT32_MAX) {
		return false;
	}
	VkMemoryDedicatedAllocateInfo const dedicated = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, s->image, VK_NULL_HANDLE };
	VkExportMemoryAllocateInfo const exported = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
		                                          extensionOn("VK_KHR_dedicated_allocation") ? &dedicated : NULL,
		                                          VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
	VkMemoryAllocateInfo const allocate = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, sharing ? &exported : NULL, need.size, type };
	if (vk.AllocateMemory(device, &allocate, NULL, &s->memory) != VK_SUCCESS) {
		s->memory = VK_NULL_HANDLE;
		return false;
	}
	if (vk.BindImageMemory(device, s->image, s->memory, 0) != VK_SUCCESS) {
		return false;
	}
	VkImageSubresource const color = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
	VkSubresourceLayout layout;
	vk.GetImageSubresourceLayout(device, s->image, &color, &layout);
	s->pitch = layout.rowPitch;
	VkImageViewCreateInfo const viewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		                                     NULL,
		                                     0,
		                                     s->image,
		                                     VK_IMAGE_VIEW_TYPE_2D,
		                                     VK_FORMAT_R8G8B8A8_UNORM,
		                                     { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_ONE },
		                                     { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
	if (vk.CreateImageView(device, &viewInfo, NULL, &s->view) != VK_SUCCESS) {
		s->view = VK_NULL_HANDLE;
		return false;
	}
	s->info.image_view = s->view;
	s->info.image_layout = VK_IMAGE_LAYOUT_GENERAL;
	s->info.create_info = viewInfo;
	if (fences) {
		VkSemaphoreCreateInfo const semaphoreInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, NULL, 0 };
		if (vk.CreateSemaphore(device, &semaphoreInfo, NULL, &s->semaphore) != VK_SUCCESS) {
			s->semaphore = VK_NULL_HANDLE;
			return false;
		}
	}
	if (!sharing) {
		void* p = NULL;
		if (vk.MapMemory(device, s->memory, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) {
			return false;
		}
		s->mapped = (uint8_t*) p + layout.offset;
	}
	if (!toGeneral(s->image, first)) {
		return false;
	}
	// The GL side: a texture on the image's memory, else one the picture is read back from.
	int texture = 0, drawFramebuffer = 0, readFramebuffer = 0;
	gl.GetIntegerv(GL_TEXTURE_BINDING_2D, &texture);
	gl.GetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFramebuffer);
	gl.GetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFramebuffer);
	gl.GenTextures(1, &s->texture);
	gl.BindTexture(GL_TEXTURE_2D, s->texture);
	bool ok = true;
	if (sharing) {
		VkMemoryGetFdInfoKHR const getFd = { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, NULL, s->memory, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
		int fd = -1;
		ok = vk.GetMemoryFdKHR(device, &getFd, &fd) == VK_SUCCESS && fd >= 0;
		if (ok) {
			EGLint attributes[] = { EGL_WIDTH,
				                    width,
				                    EGL_HEIGHT,
				                    height,
				                    EGL_LINUX_DRM_FOURCC_EXT,
				                    DRM_FORMAT_ABGR8888,
				                    EGL_DMA_BUF_PLANE0_FD_EXT,
				                    fd,
				                    EGL_DMA_BUF_PLANE0_OFFSET_EXT,
				                    (EGLint) layout.offset,
				                    EGL_DMA_BUF_PLANE0_PITCH_EXT,
				                    (EGLint) layout.rowPitch,
				                    // A linear image: no modifier, or the linear one.
				                    modifiers ? EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT : EGL_NONE,
				                    0,
				                    EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
				                    0,
				                    EGL_NONE };
			s->eglImage = gl.CreateImageKHR(display, NULL, EGL_LINUX_DMA_BUF_EXT, NULL, attributes);
			close(fd);
			ok = s->eglImage != NULL;
		}
		if (ok) {
			gl.EGLImageTargetTexture2DOES(GL_TEXTURE_2D, s->eglImage);
		}
	} else {
		gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}
	if (ok) {
		gl.GenFramebuffers(1, &s->framebuffer);
		gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, s->framebuffer);
		gl.FramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s->texture, 0);
		ok = gl.CheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
	}
	gl.BindTexture(GL_TEXTURE_2D, (unsigned) texture);
	gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, (unsigned) drawFramebuffer);
	gl.BindFramebuffer(GL_READ_FRAMEBUFFER, (unsigned) readFramebuffer);
	s->width = width;
	s->height = height;
	return ok;
}

unsigned vkPresentTarget(int width, int height) {
	drawn = -1;
	if (!frontend || width <= 0 || height <= 0) {
		return 0;
	}
	if (frontend->wait_sync_index) {
		frontend->wait_sync_index(frontend->handle);
	}
	int const index = (int) (frontend->get_sync_index(frontend->handle) % SLOTS);
	struct Slot* s = &slots[index];
	if (s->width != width || s->height != height || !s->framebuffer) {
		freeSlot(s, true);
		if (!makeSlot(s, width, height)) {
			freeSlot(s, true);
			// Without the sharing, read back.
			if (!sharing) {
				return 0;
			}
			sharing = fences = false;
			for (int i = 0; i < SLOTS; ++i) {
				freeSlot(&slots[i], true);
			}
			if (!makeSlot(s, width, height)) {
				freeSlot(s, true);
				return 0;
			}
		}
	}
	drawn = index;
	return s->framebuffer;
}

void vkPresentSubmit(void) {
	if (drawn < 0 || !frontend) {
		return;
	}
	struct Slot* s = &slots[drawn];
	drawn = -1;
	if (!sharing) {
		int readFramebuffer = 0;
		gl.GetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFramebuffer);
		gl.BindFramebuffer(GL_READ_FRAMEBUFFER, s->framebuffer);
		gl.PixelStorei(GL_PACK_ALIGNMENT, 4);
		gl.PixelStorei(GL_PACK_ROW_LENGTH, (int) (s->pitch / 4));
		gl.ReadPixels(0, 0, s->width, s->height, GL_RGBA, GL_UNSIGNED_BYTE, s->mapped);
		gl.PixelStorei(GL_PACK_ROW_LENGTH, 0);
		gl.BindFramebuffer(GL_READ_FRAMEBUFFER, (unsigned) readFramebuffer);
		frontend->set_image(frontend->handle, &s->info, 0, NULL, VK_QUEUE_FAMILY_IGNORED);
		return;
	}
	// The frontend waits for GL's drawing on the GPU, through a fence of it, else GL finishes first.
	bool fenced = false;
	if (fences) {
		const EGLint attributes[] = { EGL_NONE };
		EGLSync sync = gl.CreateSyncKHR(display, EGL_SYNC_NATIVE_FENCE_ANDROID, attributes);
		if (sync) {
			gl.Flush();
			int const fd = gl.DupNativeFenceFDANDROID(display, sync);
			gl.DestroySyncKHR(display, sync);
			if (fd >= 0) {
				VkImportSemaphoreFdInfoKHR const import = { VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR, NULL, s->semaphore,
					                                        VK_SEMAPHORE_IMPORT_TEMPORARY_BIT, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, fd };
				fenced = vk.ImportSemaphoreFdKHR(frontend->device, &import) == VK_SUCCESS;
				if (!fenced) {
					close(fd);
				}
			}
		}
	}
	if (!fenced) {
		gl.Finish();
	}
	frontend->set_image(frontend->handle, &s->info, fenced ? 1 : 0, fenced ? &s->semaphore : NULL, VK_QUEUE_FAMILY_EXTERNAL);
}

void vkPresentDestroy(bool withGl) {
	if (frontend) {
		if (frontend->lock_queue) {
			frontend->lock_queue(frontend->handle);
		}
		vk.QueueWaitIdle(frontend->queue);
		if (frontend->unlock_queue) {
			frontend->unlock_queue(frontend->handle);
		}
	}
	for (int i = 0; i < SLOTS; ++i) {
		freeSlot(&slots[i], withGl);
	}
	if (frontend && pool) {
		vk.DestroyCommandPool(frontend->device, pool, NULL);
	}
	pool = VK_NULL_HANDLE;
	frontend = NULL;
	drawn = -1;
}

#endif
