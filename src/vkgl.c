#include "vkgl.h"

// Names and shader lines are cut at their buffers, longer than any the renderer makes.
#pragma GCC diagnostic ignored "-Wformat-truncation"

#include "gles.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

// The GL subset the GL renderer and the core's picture code use, on Vulkan. GL's state is kept here
// and turned into Vulkan's when something draws: a render pass for the bound framebuffer (begun when
// a draw or clear needs it, ended by anything else), a pipeline for each program and fixed-function
// state, the program's uniforms in a uniform buffer and its textures in a descriptor set. Images stay
// in the general layout, with a full barrier between a render pass and a copy and around each copy.
// Work goes to the queue in turns of three; what GL would have done at once (a read of pixels or of
// a mapped buffer) submits and waits. The image's own pixel (GL_NV_texture_barrier, the renderer's
// sampler uMemory, when the image drawn to is the texture bound to it) is read as an input
// attachment, a by-region barrier between the draws that need what the one before wrote (none where
// the GPU reads it in rasterization order); with another texture bound, uMemory samples that, as GL
// does. Rows are as GL keeps them: a framebuffer's row 0 is GL's y = 0, the bottom, which is the
// first row in memory for both, so nothing is flipped.

typedef unsigned GLenum;
typedef unsigned GLuint;
typedef int GLint;
typedef int GLsizei;
typedef unsigned GLbitfield;
typedef intptr_t GLintptr;
typedef intptr_t GLsizeiptr;
typedef unsigned char GLboolean;

enum {
	GL_TEXTURE_2D = 0x0DE1, GL_RGBA8 = 0x8058, GL_R32F = 0x822E, GL_FLOAT = 0x1406, GL_UNSIGNED_INT = 0x1405,
	GL_FRAMEBUFFER = 0x8D40, GL_READ_FRAMEBUFFER = 0x8CA8, GL_DRAW_FRAMEBUFFER = 0x8CA9, GL_COLOR_ATTACHMENT0 = 0x8CE0,
	GL_DEPTH_ATTACHMENT = 0x8D00, GL_STENCIL_ATTACHMENT = 0x8D20, GL_DEPTH_STENCIL_ATTACHMENT = 0x821A,
	GL_FRAMEBUFFER_COMPLETE = 0x8CD5, GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT = 0x8CD7,
	GL_STENCIL_TEST = 0x0B90, GL_DEPTH_TEST = 0x0B71, GL_BLEND = 0x0BE2, GL_SCISSOR_TEST = 0x0C11, GL_POLYGON_OFFSET_FILL = 0x8037,
	GL_EXTENSIONS = 0x1F03, GL_RENDERER = 0x1F01, GL_VERSION = 0x1F02, GL_VENDOR = 0x1F00, GL_SHADING_LANGUAGE_VERSION = 0x8B8C,
	GL_MAX_TEXTURE_SIZE = 0x0D33, GL_MAX_RENDERBUFFER_SIZE = 0x84E8, GL_MAX_SAMPLES = 0x8D57,
	GL_COLOR_BUFFER_BIT = 0x4000, GL_DEPTH_BUFFER_BIT = 0x100, GL_STENCIL_BUFFER_BIT = 0x400,
	GL_NEAREST = 0x2600, GL_LINEAR = 0x2601, GL_TEXTURE_MIN_FILTER = 0x2801, GL_TEXTURE_MAG_FILTER = 0x2800,
	GL_TEXTURE_WRAP_S = 0x2802, GL_TEXTURE_WRAP_T = 0x2803, GL_CLAMP_TO_EDGE = 0x812F, GL_REPEAT = 0x2901, GL_MIRRORED_REPEAT = 0x8370,
	GL_TEXTURE0 = 0x84C0, GL_ARRAY_BUFFER = 0x8892, GL_PIXEL_PACK_BUFFER = 0x88EB, GL_PIXEL_UNPACK_BUFFER = 0x88EC,
	GL_MAP_READ_BIT = 1, GL_MAP_WRITE_BIT = 2,
	GL_VERTEX_SHADER = 0x8B31, GL_FRAGMENT_SHADER = 0x8B30, GL_COMPILE_STATUS = 0x8B81, GL_LINK_STATUS = 0x8B82,
	GL_INFO_LOG_LENGTH = 0x8B84, GL_COMPLETION_STATUS_KHR = 0x91B1,
	GL_TRIANGLES = 4, GL_TRIANGLE_STRIP = 5, GL_TRIANGLE_FAN = 6,
	GL_PACK_ALIGNMENT = 0x0D05, GL_UNPACK_ALIGNMENT = 0x0CF5, GL_UNPACK_ROW_LENGTH = 0x0CF2, GL_PACK_ROW_LENGTH = 0x0D02,
};

enum { SUBS = 3, UNITS = 12, SAMPLERS = 16, MEMBERS = 64, ATTRIBS = 8, UBO_BYTES = 4 << 20, STAGE_BYTES = 16 << 20, SETS = 4096 };
enum { LOCATION_ELEMENTS = 1024, SAMPLER_LOCATION = MEMBERS * LOCATION_ELEMENTS, SUBMIT_DRAWS = 256, SUBMIT_BARRIERS = 64, SET_CACHE = 2048 };
// Images up to 256 KB share 2 MB allocations, in power-of-two pieces from 4 KB.
enum { SLAB_MIN = 12, SLAB_MAX = 18, SLAB_BYTES = 2 << 20 };

// ---- Vulkan's functions

#define VKGL_GLOBAL(X) X(vkCreateInstance) X(vkEnumerateInstanceVersion)
#define VKGL_INSTANCE(X)                                                                                                       \
	X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceMemoryProperties) \
	X(vkGetPhysicalDeviceFormatProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr)
#define VKGL_DEVICE(X)                                                                                                         \
	X(vkGetDeviceQueue) X(vkDestroyDevice) X(vkDeviceWaitIdle) X(vkCreateCommandPool) X(vkDestroyCommandPool)                   \
	X(vkAllocateCommandBuffers) X(vkResetCommandPool) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit)            \
	X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) X(vkCreateBuffer) X(vkDestroyBuffer)                 \
	X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements)   \
	X(vkBindImageMemory) X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkInvalidateMappedMemoryRanges)                   \
	X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkDestroySampler) X(vkCreateRenderPass)                     \
	X(vkDestroyRenderPass) X(vkCreateFramebuffer) X(vkDestroyFramebuffer) X(vkCreateShaderModule) X(vkDestroyShaderModule)      \
	X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)         \
	X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) X(vkCreatePipelineCache) X(vkDestroyPipelineCache)                        \
	X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets)                   \
	X(vkUpdateDescriptorSets) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets)     \
	X(vkCmdBindVertexBuffers) X(vkCmdDraw) X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetDepthBias)                         \
	X(vkCmdSetBlendConstants) X(vkCmdSetStencilReference) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask)             \
	X(vkCmdClearAttachments) X(vkCmdPipelineBarrier) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage)      \
	X(vkCmdBlitImage) X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage)

#define VKGL_DECLARE(name) static PFN_##name name;
VKGL_GLOBAL(VKGL_DECLARE)
VKGL_INSTANCE(VKGL_DECLARE)
VKGL_DEVICE(VKGL_DECLARE)
#undef VKGL_DECLARE

// ---- Objects

struct Img {
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
	VkFormat format;
	VkImageAspectFlags aspect;
	int width, height;
	VkImageLayout layout;
	// Its memory a piece of 1 << piece bytes of slab - 1 at offset, or with slab 0 its own.
	int slab, piece;
	VkDeviceSize offset;
	// The turns whose command buffer used it, and whose uploads ahead of it wrote it.
	uint64_t used, early;
};

struct Slab {
	VkDeviceMemory memory;
	uint32_t type;
};

struct Piece {
	int slab;
	VkDeviceSize offset;
};

struct Tex {
	bool used;
	struct Img img;
	GLenum filter[2], wrap[2];
	// Bumped whenever what a descriptor of it holds changes (its image or how it's sampled).
	uint32_t version;
};

struct Rb {
	bool used;
	struct Img img;
};

struct Fbo {
	bool used;
	GLuint color, depthStencil;
	bool colorIsRb;
};

struct Buf {
	bool used;
	VkBuffer buffer;
	VkDeviceMemory memory;
	uint8_t* map;
	size_t size;
	// The submission that last used it (sub serial).
	uint64_t serial;
};

struct Attrib {
	bool enabled, integer;
	GLuint buffer;
	int size, stride;
	GLenum type;
	size_t offset;
};

struct Vao {
	bool used;
	struct Attrib attrib[ATTRIBS];
};

struct Shader {
	bool used;
	GLenum type;
	char* source;
};

struct Member {
	char name[48];
	char type[16];
	int offset, size, count;
	bool isInt;
	// A constant array's values in the source, put in the block when linked.
	const char* values;
};

struct Prog {
	bool used, linked;
	GLuint shader[2];
	char* log;
	VkShaderModule vertex, fragment;
	VkDescriptorSetLayout setLayout;
	VkPipelineLayout layout;
	struct Member member[MEMBERS];
	int members, blockSize;
	uint8_t* shadow;
	bool shadowDirty;
	char sampler[SAMPLERS][48];
	int unit[SAMPLERS];
	int samplers;
	// The sampler read as the image's own pixel (uMemory), or -1, and the block's member saying
	// whether the texture bound to it is the image drawn to (then read as the input attachment).
	int input, target;
	uint32_t inputs;
	// The uniform offset its last draw used, for the next one that changes none.
	uint32_t uboOffset;
	uint64_t uboSerial;
};

struct Pipe {
	uint8_t key[128];
	VkPipeline pipeline;
};

struct Garbage {
	int kind;
	uint64_t a, b;
};

struct SetEntry {
	uint64_t serial;
	VkDescriptorSetLayout layout;
	VkDescriptorImageInfo images[SAMPLERS + 1];
	VkDescriptorSet set;
};

struct Sub {
	VkCommandPool pool;
	// early: uploads to images cb hasn't used, run before it (outside its render passes).
	VkCommandBuffer cb, early;
	VkFence fence;
	bool recording, earlyRecording, submitted;
	uint64_t serial;
	VkDescriptorPool sets;
	struct Buf ubo, stage;
	size_t uboUsed, stageUsed;
	struct Garbage* garbage;
	int garbageCount, garbageCapacity;
	// Images given up during the turn, destroyed once it's done.
	struct Img* retired;
	int retiredCount, retiredCapacity;
	// The sets made this turn by their images (SET_CACHE of them, those of other turns unused).
	struct SetEntry* setCache;
};

struct RenderPass {
	VkFormat color, depth;
	VkRenderPass pass;
};

struct FbCache {
	VkRenderPass pass;
	VkImageView color, depth;
	int width, height;
	VkFramebuffer framebuffer;
};

struct State {
	GLuint drawFbo, readFbo, program, vao, arrayBuffer, packBuffer, unpackBuffer, renderbuffer;
	GLuint texture[UNITS];
	int unit;
	int viewport[4], scissor[4];
	bool scissorOn, depthTest, stencilTest, blend, polygonOffset;
	GLenum depthFunc;
	bool depthMask;
	GLenum blendSrc[2], blendDst[2], blendEq[2];
	float blendColor[4];
	float offsetFactor, offsetUnits;
	GLenum stencilFunc, stencilFail, stencilZfail, stencilZpass;
	int stencilRef;
	unsigned stencilMask;
	bool colorMask[4];
	float clearColor[4], clearDepth;
	int clearStencil;
};

static struct Vkgl {
	// Draws read the pixels drawn before them in rasterization order (no barriers between them).
	bool ordered;
	bool active, external;
	char renderer[256];
	struct VkglFrontend frontend;
	void* library;
	VkInstance instance;
	VkPhysicalDevice gpu;
	VkDevice device;
	VkQueue queue;
	uint32_t queueFamily;
	VkPhysicalDeviceMemoryProperties memory;
	VkPhysicalDeviceLimits limits;
	VkFormat depthFormat;
	VkPipelineCache pipelineCache;
	struct Sub sub[SUBS];
	int current;
	uint64_t serial;
	struct Tex* tex;
	struct Rb* rb;
	struct Fbo* fbo;
	struct Buf* buf;
	struct Vao* vao;
	struct Shader* shader;
	struct Prog* prog;
	int texCount, rbCount, fboCount, bufCount, vaoCount, shaderCount, progCount;
	struct Pipe* pipes;
	int pipeCount, pipeCapacity;
	struct RenderPass passes[8];
	int passCount;
	struct FbCache* fbs;
	int fbCount, fbCapacity;
	struct Slab* slabs;
	int slabCount, slabCapacity;
	struct Piece* pieces[SLAB_MAX - SLAB_MIN + 1];
	int pieceCount[SLAB_MAX - SLAB_MIN + 1], pieceCapacity[SLAB_MAX - SLAB_MIN + 1];
	VkSampler samplers[2][3][3][2];
	struct State st;
	// The render pass open, on that framebuffer, and what the command buffer has set in it.
	bool passOpen;
	GLuint passFbo;
	VkRenderPass pass;
	int passWidth, passHeight;
	VkPipeline boundPipeline;
	// What the command buffer has set within the render pass (valid since it began).
	struct {
		bool valid;
		VkViewport viewport;
		VkRect2D scissor;
		float bias[2], blend[4];
		uint32_t stencil[2];
		VkPipelineLayout layout;
		VkDescriptorSet set;
		uint32_t offset;
		VkBuffer vb[ATTRIBS];
		VkDeviceSize vo[ATTRIBS];
	} bound;
	// The last pipeline found, by its key, and the rest by a hash of theirs.
	uint8_t lastKey[128];
	VkPipeline lastPipe;
	int pipeHash[4096];
	bool barrierNeeded;
	// Draws since the last submission: the queue gets them a few hundred at a time, as GL's drivers
	// do, so the GPU works while the next are made.
	int pendingDraws, pendingBarriers;
	// Broadcom's V3D starts a job at each barrier within a pass, with memory of its own until the
	// submission is done: 4 GB in a few thousand.
	bool barrierJobs;
	// The frontend's pictures, one for each of its frames in flight, and the framebuffer for them.
	struct Img picture[4];
	int pictureIndex;
	GLuint pictureFbo, pictureTex;
	uint8_t* readback;
	struct Buf readBuffer;
	bool glslangReady;
} V;

static bool fail(char* error, size_t size, const char* what) {
	snprintf(error, size, "%s", what);
	return false;
}

// ---- Memory, buffers and images

static int memoryType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags also) {
	for (int pass = 0; pass < 2; ++pass) {
		for (uint32_t i = 0; i < V.memory.memoryTypeCount; ++i) {
			VkMemoryPropertyFlags const f = V.memory.memoryTypes[i].propertyFlags;
			if ((bits & (1u << i)) && (f & want) == want && (pass || (f & also) == also)) {
				return (int) i;
			}
		}
	}
	return -1;
}

static bool bufferMake(struct Buf* b, size_t size, VkBufferUsageFlags usage, bool cached) {
	VkBufferCreateInfo info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size ? size : 4, usage, VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
	if (vkCreateBuffer(V.device, &info, NULL, &b->buffer) != VK_SUCCESS) {
		return false;
	}
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(V.device, b->buffer, &req);
	int const type = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	                            cached ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0);
	VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, (uint32_t) type };
	if (type < 0 || vkAllocateMemory(V.device, &alloc, NULL, &b->memory) != VK_SUCCESS) {
		vkDestroyBuffer(V.device, b->buffer, NULL);
		b->buffer = VK_NULL_HANDLE;
		return false;
	}
	vkBindBufferMemory(V.device, b->buffer, b->memory, 0);
	void* p = NULL;
	vkMapMemory(V.device, b->memory, 0, VK_WHOLE_SIZE, 0, &p);
	b->map = p;
	b->size = size;
	return true;
}

static void bufferFree(struct Buf* b) {
	if (b->buffer) {
		vkDestroyBuffer(V.device, b->buffer, NULL);
		vkFreeMemory(V.device, b->memory, NULL);
	}
	b->buffer = VK_NULL_HANDLE, b->memory = VK_NULL_HANDLE, b->map = NULL, b->size = 0;
}

static bool piecePut(int size, int slab, VkDeviceSize offset) {
	int const k = size - SLAB_MIN;
	if (V.pieceCount[k] == V.pieceCapacity[k]) {
		int const n = V.pieceCapacity[k] ? V.pieceCapacity[k] * 2 : SLAB_BYTES >> size;
		struct Piece* more = realloc(V.pieces[k], (size_t) n * sizeof(*more));
		if (!more) {
			return false;
		}
		V.pieces[k] = more, V.pieceCapacity[k] = n;
	}
	V.pieces[k][V.pieceCount[k]++] = (struct Piece) { slab, offset };
	return true;
}

// A piece of 1 << size bytes of memory type, a new slab cut up for it if there's none.
static bool pieceTake(int size, uint32_t type, struct Piece* out) {
	int const k = size - SLAB_MIN;
	for (int i = V.pieceCount[k] - 1; i >= 0; --i) {
		if (V.slabs[V.pieces[k][i].slab].type == type) {
			*out = V.pieces[k][i];
			V.pieces[k][i] = V.pieces[k][--V.pieceCount[k]];
			return true;
		}
	}
	if (V.slabCount == V.slabCapacity) {
		int const n = V.slabCapacity ? V.slabCapacity * 2 : 16;
		struct Slab* more = realloc(V.slabs, (size_t) n * sizeof(*more));
		if (!more) {
			return false;
		}
		V.slabs = more, V.slabCapacity = n;
	}
	VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, SLAB_BYTES, type };
	VkDeviceMemory memory;
	if (vkAllocateMemory(V.device, &alloc, NULL, &memory) != VK_SUCCESS) {
		return false;
	}
	int const slab = V.slabCount++;
	V.slabs[slab] = (struct Slab) { memory, type };
	for (VkDeviceSize at = SLAB_BYTES; at > 1u << size;) {
		at -= 1u << size;
		piecePut(size, slab, at);
	}
	*out = (struct Piece) { slab, 0 };
	return true;
}

static void imageDestroy(struct Img* img) {
	if (img->view) {
		vkDestroyImageView(V.device, img->view, NULL);
	}
	vkDestroyImage(V.device, img->image, NULL);
	if (img->slab) {
		piecePut(img->piece, img->slab - 1, img->offset);
	} else {
		vkFreeMemory(V.device, img->memory, NULL);
	}
}

static bool imageMake(struct Img* img, VkFormat format, int width, int height) {
	bool const depth = format == V.depthFormat;
	VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
	                          (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_SAMPLED_BIT);
	if (format == VK_FORMAT_R8G8B8A8_UNORM) {
		usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
	}
	VkImageCreateInfo info = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, format,
		                       { (uint32_t) (width > 0 ? width : 1), (uint32_t) (height > 0 ? height : 1), 1 }, 1, 1,
		                       VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, usage, VK_SHARING_MODE_EXCLUSIVE, 0, NULL,
		                       VK_IMAGE_LAYOUT_UNDEFINED };
	if (vkCreateImage(V.device, &info, NULL, &img->image) != VK_SUCCESS) {
		return false;
	}
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(V.device, img->image, &req);
	int type = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
	type = type < 0 ? memoryType(req.memoryTypeBits, 0, 0) : type;
	int size = SLAB_MIN;
	while ((VkDeviceSize) 1 << size < req.size || (VkDeviceSize) 1 << size < req.alignment) {
		++size;
	}
	struct Piece piece;
	img->slab = 0, img->piece = size, img->offset = 0;
	img->used = img->early = 0;
	if (size <= SLAB_MAX && pieceTake(size, (uint32_t) type, &piece)) {
		img->memory = V.slabs[piece.slab].memory;
		img->slab = piece.slab + 1, img->offset = piece.offset;
	} else {
		VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, (uint32_t) type };
		if (vkAllocateMemory(V.device, &alloc, NULL, &img->memory) != VK_SUCCESS) {
			vkDestroyImage(V.device, img->image, NULL);
			img->image = VK_NULL_HANDLE;
			return false;
		}
	}
	vkBindImageMemory(V.device, img->image, img->memory, img->offset);
	img->aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
	VkImageViewCreateInfo view = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img->image, VK_IMAGE_VIEW_TYPE_2D, format,
		                           { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
		                           { img->aspect, 0, 1, 0, 1 } };
	img->format = format, img->width = width, img->height = height, img->layout = VK_IMAGE_LAYOUT_UNDEFINED;
	if (vkCreateImageView(V.device, &view, NULL, &img->view) != VK_SUCCESS) {
		img->view = VK_NULL_HANDLE;
		imageDestroy(img);
		img->image = VK_NULL_HANDLE;
		return false;
	}
	return true;
}

// ---- Submission

enum { GARBAGE_BUFFER, GARBAGE_FRAMEBUFFER, GARBAGE_PIPELINE };

static void garbage(int kind, uint64_t a, uint64_t b) {
	struct Sub* s = &V.sub[V.current];
	if (s->garbageCount == s->garbageCapacity) {
		int const n = s->garbageCapacity ? s->garbageCapacity * 2 : 64;
		struct Garbage* more = realloc(s->garbage, (size_t) n * sizeof(*more));
		if (!more) {
			return;
		}
		s->garbage = more, s->garbageCapacity = n;
	}
	s->garbage[s->garbageCount++] = (struct Garbage) { kind, a, b };
}

static void collect(struct Sub* s) {
	for (int i = 0; i < s->garbageCount; ++i) {
		struct Garbage* g = &s->garbage[i];
		if (g->kind == GARBAGE_BUFFER) {
			vkDestroyBuffer(V.device, (VkBuffer) (uintptr_t) g->a, NULL);
			vkFreeMemory(V.device, (VkDeviceMemory) (uintptr_t) g->b, NULL);
		} else if (g->kind == GARBAGE_FRAMEBUFFER) {
			vkDestroyFramebuffer(V.device, (VkFramebuffer) (uintptr_t) g->a, NULL);
		}
	}
	s->garbageCount = 0;
	for (int i = 0; i < s->retiredCount; ++i) {
		imageDestroy(&s->retired[i]);
	}
	s->retiredCount = 0;
}

static void imageFree(struct Img* img);

static VkCommandBuffer cmd(void) {
	struct Sub* s = &V.sub[V.current];
	if (!s->recording) {
		VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
		vkBeginCommandBuffer(s->cb, &begin);
		s->recording = true;
	}
	return s->cb;
}

// Everything before against everything after, outside a render pass.
static void barrier(VkCommandBuffer cb) {
	VkMemoryBarrier m = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m, 0, NULL, 0, NULL);
	if (cb == V.sub[V.current].cb) {
		V.barrierNeeded = false;
	}
}

static void submit(bool wait);

static void passEnd(void) {
	if (V.passOpen) {
		vkCmdEndRenderPass(V.sub[V.current].cb);
		V.passOpen = false;
		V.boundPipeline = VK_NULL_HANDLE;
		V.bound.valid = false;
		V.barrierNeeded = true;
	}
}

// The draws so far to the queue when there are enough of them, at the start of something that takes
// room in the turn's buffers (never in the middle: what it took would go with the turn before).
static void submitDue(void) {
	if ((V.pendingDraws >= SUBMIT_DRAWS && (!V.passOpen || V.passFbo != V.st.drawFbo)) || (V.barrierJobs && V.pendingBarriers >= SUBMIT_BARRIERS)) {
		submit(false);
	}
}

// The command buffer for a copy, the render pass ended and a barrier before it.
static VkCommandBuffer transfer(void) {
	passEnd();
	VkCommandBuffer cb = cmd();
	if (V.barrierNeeded) {
		barrier(cb);
	}
	V.barrierNeeded = true;
	return cb;
}

static void touch(struct Img* img) {
	img->used = V.sub[V.current].serial;
}

// The command buffer for a copy into img: the one run ahead of this turn's when it hasn't used img
// (so a render pass goes on), else transfer's.
static VkCommandBuffer transferTo(struct Img* img) {
	struct Sub* s = &V.sub[V.current];
	if (img->used == s->serial) {
		return transfer();
	}
	if (!s->earlyRecording) {
		VkCommandBufferBeginInfo begin = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
		vkBeginCommandBuffer(s->early, &begin);
		s->earlyRecording = true;
		barrier(s->early);
	} else if (img->early == s->serial) {
		barrier(s->early);
	}
	img->early = s->serial;
	return s->early;
}

static void layoutTo(VkCommandBuffer cb, struct Img* img, VkImageLayout to) {
	if (img->layout == to) {
		return;
	}
	VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT,
		                       VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, img->layout, to, VK_QUEUE_FAMILY_IGNORED,
		                       VK_QUEUE_FAMILY_IGNORED, img->image, { img->aspect, 0, 1, 0, 1 } };
	vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
	img->layout = to;
}

static void subWait(struct Sub* s) {
	if (s->submitted) {
		vkWaitForFences(V.device, 1, &s->fence, VK_TRUE, UINT64_MAX);
		vkResetFences(V.device, 1, &s->fence);
		s->submitted = false;
		collect(s);
	}
}

// The work so far to the queue; the next turn's resources ready (its last work done). With wait, the
// work submitted is done too.
static void submit(bool wait) {
	passEnd();
	struct Sub* s = &V.sub[V.current];
	if (s->recording || s->earlyRecording) {
		// The early buffer first; the turn's first command waits for it (its barrier).
		cmd();
		if (V.barrierNeeded) {
			barrier(s->cb);
		}
		VkMemoryBarrier host = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_HOST_READ_BIT };
		vkCmdPipelineBarrier(s->cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
		vkEndCommandBuffer(s->cb);
		VkCommandBuffer const buffers[2] = { s->early, s->cb };
		if (s->earlyRecording) {
			vkEndCommandBuffer(s->early);
		}
		VkSubmitInfo info = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, s->earlyRecording ? 2 : 1, s->earlyRecording ? buffers : &s->cb, 0, NULL };
		if (V.external && V.frontend.lockQueue) {
			V.frontend.lockQueue(V.frontend.handle);
		}
		vkQueueSubmit(V.queue, 1, &info, s->fence);
		V.pendingDraws = V.pendingBarriers = 0;
		if (V.external && V.frontend.unlockQueue) {
			V.frontend.unlockQueue(V.frontend.handle);
		}
		s->recording = s->earlyRecording = false;
		s->submitted = true;
		// The queue doesn't order one submission's work after the last's: the next turn's first
		// command waits on it.
		V.barrierNeeded = true;
		int const done = V.current;
		V.current = (V.current + 1) % SUBS;
		struct Sub* next = &V.sub[V.current];
		subWait(next);
		vkResetCommandPool(V.device, next->pool, 0);
		vkResetDescriptorPool(V.device, next->sets, 0);
		next->uboUsed = next->stageUsed = 0;
		next->serial = ++V.serial;
		if (wait) {
			subWait(&V.sub[done]);
		}
	} else if (wait) {
		for (int i = 0; i < SUBS; ++i) {
			subWait(&V.sub[i]);
		}
	}
}

// Waits for the work that used a buffer up to serial.
static void waitSerial(uint64_t serial) {
	if (serial >= V.sub[V.current].serial) {
		submit(true);
		return;
	}
	for (int i = 0; i < SUBS; ++i) {
		if (V.sub[i].submitted && V.sub[i].serial <= serial) {
			subWait(&V.sub[i]);
		}
	}
}

// Room for size bytes of a turn's ring buffer, aligned; the next turn's when this one's is full.
static size_t ringTake(bool ubo, size_t size, size_t align) {
	for (int tries = 0; tries < 2; ++tries) {
		struct Sub* s = &V.sub[V.current];
		size_t* used = ubo ? &s->uboUsed : &s->stageUsed;
		size_t const capacity = ubo ? s->ubo.size : s->stage.size;
		size_t const at = (*used + align - 1) / align * align;
		if (at + size <= capacity) {
			*used = at + size;
			return at;
		}
		submit(false);
	}
	return SIZE_MAX;
}

static void imageFree(struct Img* img) {
	if (!img->image) {
		return;
	}
	for (int i = 0; i < V.fbCount; ++i) {
		if (V.fbs[i].color == img->view || V.fbs[i].depth == img->view) {
			garbage(GARBAGE_FRAMEBUFFER, (uint64_t) (uintptr_t) V.fbs[i].framebuffer, 0);
			V.fbs[i--] = V.fbs[--V.fbCount];
		}
	}
	struct Sub* s = &V.sub[V.current];
	if (s->retiredCount == s->retiredCapacity) {
		int const n = s->retiredCapacity ? s->retiredCapacity * 2 : 64;
		struct Img* more = realloc(s->retired, (size_t) n * sizeof(*more));
		if (!more) {
			memset(img, 0, sizeof(*img));
			return;
		}
		s->retired = more, s->retiredCapacity = n;
	}
	s->retired[s->retiredCount++] = *img;
	memset(img, 0, sizeof(*img));
}

// A new image's layout set (general, for everything), so it can be used at once, and its pixels
// zero: Vulkan's memory comes with what it last held, GL's drivers give it cleared, and pixels the
// renderer reads back before drawing them must be the same each run.
static void imageReady(struct Img* img) {
	VkCommandBuffer cb = transferTo(img);
	layoutTo(cb, img, VK_IMAGE_LAYOUT_GENERAL);
	VkImageSubresourceRange const all = { img->aspect, 0, 1, 0, 1 };
	if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
		VkClearColorValue const zero = { { 0, 0, 0, 0 } };
		vkCmdClearColorImage(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
	} else {
		VkClearDepthStencilValue const far = { 1.0f, 0 };
		vkCmdClearDepthStencilImage(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &far, 1, &all);
	}
}

// ---- The device

static bool loadDevice(void) {
#define VKGL_LOAD(name) name = (PFN_##name) vkGetDeviceProcAddr(V.device, #name); if (!name) return false;
	VKGL_DEVICE(VKGL_LOAD)
#undef VKGL_LOAD
	return true;
}

static bool loadInstance(PFN_vkGetInstanceProcAddr get) {
#define VKGL_LOAD(name) name = (PFN_##name) get(V.instance, #name); if (!name) return false;
	VKGL_INSTANCE(VKGL_LOAD)
#undef VKGL_LOAD
	return true;
}

// What a device is made with for draws to read the pixels drawn before them in rasterization order,
// when gpu can: extension and features (order chained on it), for the device's create info.
struct Ordering {
	const char* extension;
	VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT order;
};

static VkDevice orderedDevice;

static bool ordering(VkInstance instance, PFN_vkGetInstanceProcAddr get, VkPhysicalDevice gpu, struct Ordering* o) {
	PFN_vkGetPhysicalDeviceProperties properties = (PFN_vkGetPhysicalDeviceProperties) get(instance, "vkGetPhysicalDeviceProperties");
	PFN_vkEnumerateDeviceExtensionProperties extensions = (PFN_vkEnumerateDeviceExtensionProperties) get(instance, "vkEnumerateDeviceExtensionProperties");
	if (!properties || !extensions) {
		return false;
	}
	// A Vulkan 1.0 GPU's features through VK_KHR_get_physical_device_properties2 (some Adrenos).
	VkPhysicalDeviceProperties p;
	properties(gpu, &p);
	PFN_vkGetPhysicalDeviceFeatures2 features =
	    (PFN_vkGetPhysicalDeviceFeatures2) get(instance, p.apiVersion >= VK_API_VERSION_1_1 ? "vkGetPhysicalDeviceFeatures2" : "vkGetPhysicalDeviceFeatures2KHR");
	uint32_t n = 0;
	if (!features || extensions(gpu, NULL, &n, NULL) != VK_SUCCESS) {
		return false;
	}
	VkExtensionProperties* list = calloc(n ? n : 1, sizeof(*list));
	if (!list || extensions(gpu, NULL, &n, list) != VK_SUCCESS) {
		free(list);
		return false;
	}
	o->extension = NULL;
	for (uint32_t i = 0; i < n; ++i) {
		if (!strcmp(list[i].extensionName, VK_EXT_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME)) {
			o->extension = VK_EXT_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME;
		} else if (!strcmp(list[i].extensionName, VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME) && !o->extension) {
			o->extension = VK_ARM_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_EXTENSION_NAME;
		}
	}
	free(list);
	if (!o->extension) {
		return false;
	}
	o->order = (VkPhysicalDeviceRasterizationOrderAttachmentAccessFeaturesEXT) { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_FEATURES_EXT,
		                                                                         NULL, VK_FALSE, VK_FALSE, VK_FALSE };
	VkPhysicalDeviceFeatures2 all = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &o->order, { 0 } };
	features(gpu, &all);
	o->order.pNext = NULL;
	o->order.rasterizationOrderDepthAttachmentAccess = o->order.rasterizationOrderStencilAttachmentAccess = VK_FALSE;
	return o->order.rasterizationOrderColorAttachmentAccess;
}

// The GPU a device goes on: discrete, then integrated, then any, with a graphics queue.
static VkPhysicalDevice bestGpu(VkInstance instance, PFN_vkGetInstanceProcAddr get) {
	PFN_vkEnumeratePhysicalDevices enumerate = (PFN_vkEnumeratePhysicalDevices) get(instance, "vkEnumeratePhysicalDevices");
	PFN_vkGetPhysicalDeviceProperties properties = (PFN_vkGetPhysicalDeviceProperties) get(instance, "vkGetPhysicalDeviceProperties");
	PFN_vkGetPhysicalDeviceQueueFamilyProperties families = (PFN_vkGetPhysicalDeviceQueueFamilyProperties) get(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
	VkPhysicalDevice gpus[16];
	uint32_t count = 16;
	if (!enumerate || !properties || !families || enumerate(instance, &count, gpus) < 0) {
		return VK_NULL_HANDLE;
	}
	VkPhysicalDevice best = VK_NULL_HANDLE;
	int bestRank = -1;
	for (uint32_t i = 0; i < count; ++i) {
		VkPhysicalDeviceProperties p;
		properties(gpus[i], &p);
		VkQueueFamilyProperties f[16];
		uint32_t n = 16;
		families(gpus[i], &n, f);
		for (uint32_t k = 0; k < n; ++k) {
			int const rank = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
			if ((f[k].queueFlags & VK_QUEUE_GRAPHICS_BIT) && rank > bestRank) {
				best = gpus[i], bestRank = rank;
			}
		}
	}
	return best;
}

bool vkglMakeDevice(VkInstance instance, VkPhysicalDevice gpu, VkSurfaceKHR surface, PFN_vkGetInstanceProcAddr get,
                    VkDevice (*create)(VkPhysicalDevice, void*, const VkDeviceCreateInfo*), void* opaque, struct VkglDevice* out) {
	PFN_vkGetPhysicalDeviceQueueFamilyProperties families = (PFN_vkGetPhysicalDeviceQueueFamilyProperties) get(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
	PFN_vkGetPhysicalDeviceSurfaceSupportKHR presents = (PFN_vkGetPhysicalDeviceSurfaceSupportKHR) get(instance, "vkGetPhysicalDeviceSurfaceSupportKHR");
	PFN_vkGetDeviceProcAddr deviceProc = (PFN_vkGetDeviceProcAddr) get(instance, "vkGetDeviceProcAddr");
	gpu = gpu ? gpu : bestGpu(instance, get);
	if (!gpu || !families || !deviceProc || (surface && !presents)) {
		return false;
	}
	VkQueueFamilyProperties f[16];
	uint32_t n = 16;
	families(gpu, &n, f);
	// One family for graphics, compute and the surface if there is one, else one for each.
	uint32_t graphics = UINT32_MAX, present = UINT32_MAX;
	for (uint32_t i = 0; i < n; ++i) {
		VkBool32 can = !surface;
		if (surface) {
			presents(gpu, i, surface, &can);
		}
		bool const draws = (f[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
		if (draws && can) {
			graphics = present = i;
			break;
		}
		graphics = draws && graphics == UINT32_MAX ? i : graphics;
		present = can && present == UINT32_MAX ? i : present;
	}
	if (graphics == UINT32_MAX || present == UINT32_MAX) {
		return false;
	}
	float const priority = 1.0f;
	VkDeviceQueueCreateInfo const queues[2] = { { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, graphics, 1, &priority },
		                                        { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, present, 1, &priority } };
	struct Ordering o;
	bool const ordered = ordering(instance, get, gpu, &o);
	VkDeviceCreateInfo const info = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, ordered ? &o.order : NULL, 0, present == graphics ? 1u : 2u, queues, 0, NULL,
		                              ordered ? 1u : 0u, ordered ? &o.extension : NULL, NULL };
	VkDevice const device = create(gpu, opaque, &info);
	PFN_vkGetDeviceQueue getQueue = device ? (PFN_vkGetDeviceQueue) deviceProc(device, "vkGetDeviceQueue") : NULL;
	if (!getQueue) {
		return false;
	}
	*out = (struct VkglDevice) { gpu, device, VK_NULL_HANDLE, VK_NULL_HANDLE, graphics, present };
	getQueue(device, graphics, 0, &out->queue);
	getQueue(device, present, 0, &out->presentQueue);
	orderedDevice = ordered ? device : VK_NULL_HANDLE;
	return true;
}

// An instance and device of its own: the first GPU with a graphics queue, discrete or integrated
// first.
static bool ownDevice(char* error, size_t errorSize) {
	V.library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!V.library) {
		V.library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
	}
	if (!V.library) {
		return fail(error, errorSize, "no libvulkan");
	}
	PFN_vkGetInstanceProcAddr get = (PFN_vkGetInstanceProcAddr) dlsym(V.library, "vkGetInstanceProcAddr");
	if (!get) {
		return fail(error, errorSize, "no vkGetInstanceProcAddr");
	}
	vkCreateInstance = (PFN_vkCreateInstance) get(NULL, "vkCreateInstance");
	vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion) get(NULL, "vkEnumerateInstanceVersion");
	uint32_t version = VK_API_VERSION_1_0;
	if (vkEnumerateInstanceVersion) {
		vkEnumerateInstanceVersion(&version);
	}
	VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "armN64", 1, "armN64", 1,
		                      version >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_1 : VK_API_VERSION_1_0 };
	PFN_vkEnumerateInstanceExtensionProperties instanceExtensions = (PFN_vkEnumerateInstanceExtensionProperties) get(NULL, "vkEnumerateInstanceExtensionProperties");
	const char* properties2 = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
	uint32_t found = 0, n = 0;
	VkExtensionProperties* list = instanceExtensions && instanceExtensions(NULL, &n, NULL) == VK_SUCCESS ? calloc(n ? n : 1, sizeof(*list)) : NULL;
	if (list && instanceExtensions(NULL, &n, list) == VK_SUCCESS) {
		for (uint32_t i = 0; i < n; ++i) {
			found |= !strcmp(list[i].extensionName, properties2);
		}
	}
	free(list);
	VkInstanceCreateInfo info = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, found, found ? &properties2 : NULL };
	if (!vkCreateInstance || vkCreateInstance(&info, NULL, &V.instance) != VK_SUCCESS || !loadInstance(get)) {
		return fail(error, errorSize, "no Vulkan instance");
	}
	VkPhysicalDevice gpus[16];
	uint32_t count = 16;
	vkEnumeratePhysicalDevices(V.instance, &count, gpus);
	int best = -1, bestRank = -1;
	uint32_t bestFamily = 0;
	for (uint32_t i = 0; i < count; ++i) {
		VkPhysicalDeviceProperties p;
		vkGetPhysicalDeviceProperties(gpus[i], &p);
		VkQueueFamilyProperties families[16];
		uint32_t n = 16;
		vkGetPhysicalDeviceQueueFamilyProperties(gpus[i], &n, families);
		for (uint32_t f = 0; f < n; ++f) {
			if (!(families[f].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				continue;
			}
			int const rank = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
			if (rank > bestRank) {
				best = (int) i, bestRank = rank, bestFamily = f;
			}
			break;
		}
	}
	if (best < 0) {
		return fail(error, errorSize, "no Vulkan GPU with a graphics queue");
	}
	V.gpu = gpus[best];
	V.queueFamily = bestFamily;
	float const priority = 1.0f;
	VkDeviceQueueCreateInfo queue = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, V.queueFamily, 1, &priority };
	struct Ordering o;
	V.ordered = ordering(V.instance, get, V.gpu, &o);
	VkDeviceCreateInfo dev = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, V.ordered ? &o.order : NULL, 0, 1, &queue, 0, NULL,
		                       V.ordered ? 1u : 0u, V.ordered ? &o.extension : NULL, NULL };
	if (vkCreateDevice(V.gpu, &dev, NULL, &V.device) != VK_SUCCESS || !loadDevice()) {
		return fail(error, errorSize, "no Vulkan device");
	}
	vkGetDeviceQueue(V.device, V.queueFamily, 0, &V.queue);
	return true;
}

static void freeAll(void);
static struct Img* dummyImage(void);
static struct Img dummy;
static void queueStart(void);
static void queueStop(void);
static void drain(void);

bool vkglCreate(const struct VkglFrontend* frontend, char* error, size_t errorSize) {
	if (V.active) {
		return true;
	}
	memset(&V, 0, sizeof(V));
	if (frontend) {
		V.external = true;
		V.frontend = *frontend;
		V.instance = frontend->instance, V.gpu = frontend->gpu, V.device = frontend->device, V.queue = frontend->queue;
		V.queueFamily = frontend->queueFamily;
		V.ordered = frontend->device == orderedDevice;
		if (!loadInstance(frontend->getInstanceProcAddr)) {
			return fail(error, errorSize, "Vulkan functions missing");
		}
		vkGetDeviceProcAddr = frontend->getDeviceProcAddr;
		if (!loadDevice()) {
			return fail(error, errorSize, "Vulkan functions missing");
		}
	} else if (!ownDevice(error, errorSize)) {
		freeAll();
		return false;
	}
	VkPhysicalDeviceProperties p;
	vkGetPhysicalDeviceProperties(V.gpu, &p);
	V.limits = p.limits;
	V.barrierJobs = p.vendorID == 0x14e4;
	snprintf(V.renderer, sizeof(V.renderer), "%s (Vulkan %u.%u)", p.deviceName, VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion));
	vkGetPhysicalDeviceMemoryProperties(V.gpu, &V.memory);
	VkFormatProperties fp;
	vkGetPhysicalDeviceFormatProperties(V.gpu, VK_FORMAT_D24_UNORM_S8_UINT, &fp);
	V.depthFormat = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_D32_SFLOAT_S8_UINT;
	// The pipelines made before on this GPU and driver (the cache's header checks they match).
	char path[4300];
	void* saved = NULL;
	size_t savedSize = 0;
	if (glesCacheDirectory()[0]) {
		snprintf(path, sizeof(path), "%s/vk-pipelines-%08x-%08x.bin", glesCacheDirectory(), p.vendorID, p.deviceID);
		FILE* in = fopen(path, "rb");
		if (in && !fseek(in, 0, SEEK_END)) {
			long const n = ftell(in);
			saved = n > 0 && n < (64 << 20) ? malloc((size_t) n) : NULL;
			if (saved && !fseek(in, 0, SEEK_SET) && fread(saved, 1, (size_t) n, in) == (size_t) n) {
				savedSize = (size_t) n;
			}
		}
		if (in) {
			fclose(in);
		}
	}
	VkPipelineCacheCreateInfo cache = { VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, NULL, 0, savedSize, saved };
	if (vkCreatePipelineCache(V.device, &cache, NULL, &V.pipelineCache) != VK_SUCCESS) {
		cache.initialDataSize = 0, cache.pInitialData = NULL;
		vkCreatePipelineCache(V.device, &cache, NULL, &V.pipelineCache);
	}
	free(saved);
	VkDescriptorPoolSize sizes[3] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, SETS }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, SETS * 4 },
		                              { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, SETS } };
	for (int i = 0; i < SUBS; ++i) {
		struct Sub* s = &V.sub[i];
		VkCommandPoolCreateInfo pool = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, V.queueFamily };
		VkFenceCreateInfo fence = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
		VkDescriptorPoolCreateInfo sets = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, SETS, 3, sizes };
		if (vkCreateCommandPool(V.device, &pool, NULL, &s->pool) != VK_SUCCESS || vkCreateFence(V.device, &fence, NULL, &s->fence) != VK_SUCCESS ||
		    vkCreateDescriptorPool(V.device, &sets, NULL, &s->sets) != VK_SUCCESS ||
		    !bufferMake(&s->ubo, UBO_BYTES, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, false) ||
		    !bufferMake(&s->stage, STAGE_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false)) {
			freeAll();
			return fail(error, errorSize, "Vulkan resources unavailable");
		}
		s->setCache = calloc(SET_CACHE, sizeof(*s->setCache));
		if (!s->setCache) {
			freeAll();
			return fail(error, errorSize, "Vulkan resources unavailable");
		}
		VkCommandBufferAllocateInfo cb = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
		vkAllocateCommandBuffers(V.device, &cb, &s->cb);
		vkAllocateCommandBuffers(V.device, &cb, &s->early);
	}
	V.sub[0].serial = V.serial = 1;
	V.st = (struct State) { .depthFunc = 0x201, .depthMask = true, .blendSrc = { 1, 1 }, .blendDst = { 0, 0 }, .blendEq = { 0x8006, 0x8006 },
		                    .stencilFunc = 0x207, .stencilFail = 0x1E00, .stencilZfail = 0x1E00, .stencilZpass = 0x1E00, .stencilMask = ~0u,
		                    .colorMask = { true, true, true, true }, .clearDepth = 1.0f };
	if (!V.glslangReady) {
		glslang_initialize_process();
		V.glslangReady = true;
	}
	V.active = true;
	dummyImage();
	queueStart();
	return true;
}

static void keepPipelines(void) {
	PFN_vkGetPipelineCacheData get = (PFN_vkGetPipelineCacheData) vkGetDeviceProcAddr(V.device, "vkGetPipelineCacheData");
	size_t size = 0;
	if (!glesCacheDirectory()[0] || !get || get(V.device, V.pipelineCache, &size, NULL) != VK_SUCCESS || !size) {
		return;
	}
	void* data = malloc(size);
	if (data && get(V.device, V.pipelineCache, &size, data) == VK_SUCCESS) {
		VkPhysicalDeviceProperties p;
		vkGetPhysicalDeviceProperties(V.gpu, &p);
		char path[4300], temp[4400];
		snprintf(path, sizeof(path), "%s/vk-pipelines-%08x-%08x.bin", glesCacheDirectory(), p.vendorID, p.deviceID);
		snprintf(temp, sizeof(temp), "%s.%ld.tmp", path, (long) getpid());
		FILE* out = fopen(temp, "wb");
		if (out && fwrite(data, 1, size, out) == size && !fclose(out)) {
			rename(temp, path);
		} else if (out) {
			fclose(out);
			remove(temp);
		}
	}
	free(data);
}

static void freeAll(void) {
	if (V.device) {
		vkDeviceWaitIdle(V.device);
		for (int i = 0; i < SUBS; ++i) {
			V.sub[i].submitted = false;
		}
		// The objects first (what they free goes to the turns' garbage), then the turns.
		for (int i = 0; i < V.texCount; ++i) {
			imageFree(&V.tex[i].img);
		}
		for (int i = 0; i < V.rbCount; ++i) {
			imageFree(&V.rb[i].img);
		}
		for (int i = 0; i < 4; ++i) {
			imageFree(&V.picture[i]);
		}
		imageFree(&dummy);
		for (int i = 0; i < V.bufCount; ++i) {
			bufferFree(&V.buf[i]);
		}
		bufferFree(&V.readBuffer);
		for (int i = 0; i < V.progCount; ++i) {
			struct Prog* p = &V.prog[i];
			if (p->vertex) {
				vkDestroyShaderModule(V.device, p->vertex, NULL);
				vkDestroyShaderModule(V.device, p->fragment, NULL);
			}
			if (p->layout) {
				vkDestroyPipelineLayout(V.device, p->layout, NULL);
				vkDestroyDescriptorSetLayout(V.device, p->setLayout, NULL);
			}
			free(p->shadow);
			free(p->log);
		}
		for (int i = 0; i < V.shaderCount; ++i) {
			free(V.shader[i].source);
		}
		for (int i = 0; i < V.pipeCount; ++i) {
			vkDestroyPipeline(V.device, V.pipes[i].pipeline, NULL);
		}
		for (int i = 0; i < V.fbCount; ++i) {
			vkDestroyFramebuffer(V.device, V.fbs[i].framebuffer, NULL);
		}
		for (int i = 0; i < V.passCount; ++i) {
			vkDestroyRenderPass(V.device, V.passes[i].pass, NULL);
		}
		for (int a = 0; a < 2; ++a) {
			for (int b = 0; b < 3; ++b) {
				for (int c = 0; c < 3; ++c) {
					for (int d = 0; d < 2; ++d) {
						if (V.samplers[a][b][c][d]) {
							vkDestroySampler(V.device, V.samplers[a][b][c][d], NULL);
						}
					}
				}
			}
		}
		for (int i = 0; i < SUBS; ++i) {
			struct Sub* s = &V.sub[i];
			collect(s);
			free(s->garbage);
			s->garbage = NULL, s->garbageCount = s->garbageCapacity = 0;
			free(s->retired);
			s->retired = NULL, s->retiredCount = s->retiredCapacity = 0;
			free(s->setCache);
			s->setCache = NULL;
			bufferFree(&s->ubo);
			bufferFree(&s->stage);
			if (s->sets) {
				vkDestroyDescriptorPool(V.device, s->sets, NULL);
			}
			if (s->fence) {
				vkDestroyFence(V.device, s->fence, NULL);
			}
			if (s->pool) {
				vkDestroyCommandPool(V.device, s->pool, NULL);
			}
		}
		for (int i = 0; i < V.slabCount; ++i) {
			vkFreeMemory(V.device, V.slabs[i].memory, NULL);
		}
		free(V.slabs);
		V.slabs = NULL, V.slabCount = V.slabCapacity = 0;
		for (int k = 0; k <= SLAB_MAX - SLAB_MIN; ++k) {
			free(V.pieces[k]);
			V.pieces[k] = NULL, V.pieceCount[k] = V.pieceCapacity[k] = 0;
		}
		if (V.pipelineCache) {
			keepPipelines();
			vkDestroyPipelineCache(V.device, V.pipelineCache, NULL);
		}
		if (!V.external) {
			vkDestroyDevice(V.device, NULL);
		}
	}
	if (!V.external && V.instance) {
		vkDestroyInstance(V.instance, NULL);
	}
	if (V.library) {
		dlclose(V.library);
	}
	free(V.tex), free(V.rb), free(V.fbo), free(V.buf), free(V.vao), free(V.shader), free(V.prog), free(V.pipes), free(V.fbs);
	free(V.readback);
	bool const glslang = V.glslangReady;
	memset(&V, 0, sizeof(V));
	V.glslangReady = glslang;
}

void vkglDestroy(void) {
	queueStop();
	if (V.active || V.device) {
		freeAll();
	}
}

bool vkglActive(void) {
	return V.active;
}

const char* vkglRenderer(void) {
	return V.renderer;
}

bool vkglOrdered(void) {
	return V.ordered;
}

// ---- Render passes, framebuffers and samplers

static VkRenderPass renderPass(VkFormat color, VkFormat depth) {
	for (int i = 0; i < V.passCount; ++i) {
		if (V.passes[i].color == color && V.passes[i].depth == depth) {
			return V.passes[i].pass;
		}
	}
	if (V.passCount == 8) {
		return VK_NULL_HANDLE;
	}
	VkAttachmentDescription att[2];
	VkAttachmentReference colorRef = { 0, VK_IMAGE_LAYOUT_GENERAL }, depthRef = { color ? 1u : 0u, VK_IMAGE_LAYOUT_GENERAL };
	uint32_t n = 0;
	if (color) {
		att[n++] = (VkAttachmentDescription) { 0, color, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
			                                   VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL };
	}
	if (depth) {
		att[n++] = (VkAttachmentDescription) { 0, depth, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
			                                   VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL };
	}
	// The color attachment is also the subpass's input (the image's own pixel), with a dependency on
	// itself for the barriers between draws.
	VkSubpassDescriptionFlags const flags = color && V.ordered ? VK_SUBPASS_DESCRIPTION_RASTERIZATION_ORDER_ATTACHMENT_COLOR_ACCESS_BIT_EXT : 0;
	VkSubpassDescription sub = { flags, VK_PIPELINE_BIND_POINT_GRAPHICS, color ? 1u : 0u, color ? &colorRef : NULL, color ? 1u : 0u, color ? &colorRef : NULL, NULL,
		                         depth ? &depthRef : NULL, 0, NULL };
	VkSubpassDependency deps[3] = {
		{ VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
		  VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0 },
		{ 0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
		  VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0 },
		{ 0, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		  VK_ACCESS_INPUT_ATTACHMENT_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT },
	};
	VkRenderPassCreateInfo info = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, NULL, 0, n, att, 1, &sub, color ? 3u : 2u, deps };
	VkRenderPass pass;
	if (vkCreateRenderPass(V.device, &info, NULL, &pass) != VK_SUCCESS) {
		return VK_NULL_HANDLE;
	}
	V.passes[V.passCount++] = (struct RenderPass) { color, depth, pass };
	return pass;
}

static struct Img* fboColor(struct Fbo* f) {
	if (V.pictureFbo && f == &V.fbo[V.pictureFbo]) {
		return V.picture[V.pictureIndex].image ? &V.picture[V.pictureIndex] : NULL;
	}
	if (!f->color) {
		return NULL;
	}
	if (f->colorIsRb) {
		return f->color < (GLuint) V.rbCount && V.rb[f->color].img.image ? &V.rb[f->color].img : NULL;
	}
	return f->color < (GLuint) V.texCount && V.tex[f->color].img.image ? &V.tex[f->color].img : NULL;
}

static struct Img* fboDepth(struct Fbo* f) {
	return f->depthStencil && f->depthStencil < (GLuint) V.rbCount && V.rb[f->depthStencil].img.image ? &V.rb[f->depthStencil].img : NULL;
}

static struct Fbo* fboGet(GLuint name) {
	return name && name < (GLuint) V.fboCount && V.fbo[name].used ? &V.fbo[name] : NULL;
}

// The render pass on the draw framebuffer, begun when not open on it. False when it can't be drawn to.
static bool passBegin(void) {
	if (V.passOpen && V.passFbo == V.st.drawFbo) {
		return true;
	}
	passEnd();
	struct Fbo* f = fboGet(V.st.drawFbo);
	if (!f) {
		return false;
	}
	struct Img* color = fboColor(f);
	struct Img* depth = fboDepth(f);
	if (!color && !depth) {
		return false;
	}
	VkCommandBuffer cb = cmd();
	if (V.barrierNeeded) {
		barrier(cb);
	}
	if (color) {
		touch(color);
		layoutTo(cb, color, VK_IMAGE_LAYOUT_GENERAL);
	}
	if (depth) {
		touch(depth);
		layoutTo(cb, depth, VK_IMAGE_LAYOUT_GENERAL);
	}
	int w = color ? color->width : depth->width, h = color ? color->height : depth->height;
	if (color && depth) {
		w = w < depth->width ? w : depth->width;
		h = h < depth->height ? h : depth->height;
	}
	VkRenderPass pass = renderPass(color ? color->format : VK_FORMAT_UNDEFINED, depth ? depth->format : VK_FORMAT_UNDEFINED);
	VkImageView const cv = color ? color->view : VK_NULL_HANDLE, dv = depth ? depth->view : VK_NULL_HANDLE;
	VkFramebuffer fb = VK_NULL_HANDLE;
	for (int i = 0; i < V.fbCount; ++i) {
		struct FbCache* c = &V.fbs[i];
		if (c->pass == pass && c->color == cv && c->depth == dv && c->width == w && c->height == h) {
			fb = c->framebuffer;
			break;
		}
	}
	if (!fb) {
		VkImageView views[2];
		uint32_t n = 0;
		if (color) {
			views[n++] = cv;
		}
		if (depth) {
			views[n++] = dv;
		}
		VkFramebufferCreateInfo info = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, NULL, 0, pass, n, views, (uint32_t) w, (uint32_t) h, 1 };
		if (!pass || vkCreateFramebuffer(V.device, &info, NULL, &fb) != VK_SUCCESS) {
			return false;
		}
		if (V.fbCount == V.fbCapacity) {
			int const cap = V.fbCapacity ? V.fbCapacity * 2 : 32;
			struct FbCache* more = realloc(V.fbs, (size_t) cap * sizeof(*more));
			if (!more) {
				vkDestroyFramebuffer(V.device, fb, NULL);
				return false;
			}
			V.fbs = more, V.fbCapacity = cap;
		}
		V.fbs[V.fbCount++] = (struct FbCache) { pass, cv, dv, w, h, fb };
	}
	VkRenderPassBeginInfo begin = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, NULL, pass, fb, { { 0, 0 }, { (uint32_t) w, (uint32_t) h } }, 0, NULL };
	vkCmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
	V.passOpen = true;
	V.passFbo = V.st.drawFbo;
	V.pass = pass;
	V.passWidth = w, V.passHeight = h;
	V.boundPipeline = VK_NULL_HANDLE;
	memset(&V.bound, 0, sizeof(V.bound));
	return true;
}

static VkSampler sampler(GLenum minFilter, GLenum wrapS, GLenum wrapT, GLenum magFilter) {
	int const lin = minFilter == GL_LINEAR, mag = magFilter == GL_LINEAR;
	int const s = wrapS == GL_REPEAT ? 1 : wrapS == GL_MIRRORED_REPEAT ? 2 : 0;
	int const t = wrapT == GL_REPEAT ? 1 : wrapT == GL_MIRRORED_REPEAT ? 2 : 0;
	VkSampler* slot = &V.samplers[lin][s][t][mag];
	if (!*slot) {
		static const VkSamplerAddressMode modes[3] = { VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT };
		VkSamplerCreateInfo info = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, mag ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
			                         lin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, modes[s], modes[t],
			                         VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0, VK_FALSE, 1, VK_FALSE, VK_COMPARE_OP_ALWAYS, 0, 0,
			                         VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_FALSE };
		vkCreateSampler(V.device, &info, NULL, slot);
	}
	return *slot;
}

// ---- Shaders and programs
//
// The renderer's GLSL ES 3.00 becomes Vulkan's: its loose uniforms one std140 block (set 0, binding
// 0) both stages share, laid out here so a GL uniform location is an offset into it; each sampler a
// binding after it, its texture unit given with Uniform1i; varyings given locations by name, the
// output location 0; gl_VertexID gl_VertexIndex; and GL's -1 to 1 depth Vulkan's 0 to 1.

struct Str {
	char* p;
	size_t n, cap;
};

static void put(struct Str* s, const char* text, size_t len) {
	if (s->n + len + 1 > s->cap) {
		size_t cap = s->cap ? s->cap * 2 : 4096;
		while (cap < s->n + len + 1) {
			cap *= 2;
		}
		char* more = realloc(s->p, cap);
		if (!more) {
			return;
		}
		s->p = more, s->cap = cap;
	}
	memcpy(s->p + s->n, text, len);
	s->n += len;
	s->p[s->n] = 0;
}

static void puts_(struct Str* s, const char* text) {
	put(s, text, strlen(text));
}

static const char* skipSpace(const char* p) {
	while (*p == ' ' || *p == '\t') {
		++p;
	}
	return p;
}

static bool startsWord(const char* p, const char* word) {
	size_t const n = strlen(word);
	return !strncmp(p, word, n) && (p[n] == ' ' || p[n] == '\t');
}

// A type's std140 size and alignment (arrays: each element 16-byte aligned).
static bool typeLayout(const char* type, int* size, int* align, bool* isInt) {
	static const struct {
		const char* name;
		int size, align;
		bool isInt;
	} types[] = {
		{ "float", 4, 4, false }, { "int", 4, 4, true }, { "uint", 4, 4, true }, { "bool", 4, 4, true },
		{ "vec2", 8, 8, false }, { "ivec2", 8, 8, true }, { "uvec2", 8, 8, true },
		{ "vec3", 12, 16, false }, { "ivec3", 12, 16, true }, { "uvec3", 12, 16, true },
		{ "vec4", 16, 16, false }, { "ivec4", 16, 16, true }, { "uvec4", 16, 16, true }, { "mat4", 64, 16, false },
	};
	for (size_t i = 0; i < sizeof(types) / sizeof(*types); ++i) {
		if (!strcmp(type, types[i].name)) {
			*size = types[i].size, *align = types[i].align, *isInt = types[i].isInt;
			return true;
		}
	}
	return false;
}

// A uniform line's type and names ("uniform highp ivec4 uTile[24], uImage[8];"): calls each(name,
// count) and returns the type in type; false when it isn't one.
static bool uniformLine(const char* line, char* type, size_t typeSize, char names[][48], int* counts, int* n, int max) {
	const char* p = skipSpace(line);
	if (!startsWord(p, "uniform")) {
		return false;
	}
	p = skipSpace(p + 7);
	for (int k = 0; k < 2; ++k) {
		if (startsWord(p, "highp") || startsWord(p, "mediump") || startsWord(p, "lowp")) {
			p = skipSpace(strchr(p, ' ') + 1);
		}
	}
	size_t t = 0;
	while (*p && *p != ' ' && *p != '\t' && t + 1 < typeSize) {
		type[t++] = *p++;
	}
	type[t] = 0;
	*n = 0;
	while (*p && *p != ';' && *n < max) {
		p = skipSpace(p);
		if (*p == ',') {
			p = skipSpace(p + 1);
		}
		size_t k = 0;
		while (*p && *p != ',' && *p != ';' && *p != '[' && *p != ' ' && k < 47) {
			names[*n][k++] = *p++;
		}
		names[*n][k] = 0;
		counts[*n] = 0;
		if (*p == '[') {
			counts[*n] = atoi(p + 1);
			while (*p && *p != ']') {
				++p;
			}
			if (*p) {
				++p;
			}
		}
		if (k) {
			++*n;
		}
		p = skipSpace(p);
	}
	return t > 0;
}

static void addMember(struct Prog* p, const char* name, const char* type, int count, const char* values) {
	int size, align;
	bool isInt;
	int k = 0;
	while (k < p->members && strcmp(p->member[k].name, name)) {
		++k;
	}
	if (k == p->members && k < MEMBERS && typeLayout(type, &size, &align, &isInt)) {
		struct Member* m = &p->member[p->members++];
		snprintf(m->name, sizeof(m->name), "%s", name);
		snprintf(m->type, sizeof(m->type), "%s", type);
		int const stride = count ? (size + 15) / 16 * 16 : size;
		align = count ? 16 : align;
		m->offset = (p->blockSize + align - 1) / align * align;
		m->size = count ? stride : size;
		m->count = count;
		m->isInt = isInt;
		m->values = values;
		p->blockSize = m->offset + (count ? stride * count : size);
	}
}

// A constant array ("const ivec2 rcpTable[64] = ivec2[64](...);", at the start of a line): Vulkan
// compilers may copy it to every invocation's scratch memory to index it, GL's make it uniforms.
static bool constArray(const char* line, char* type, char* name, int* count, const char** values) {
	if (strncmp(line, "const ", 6) || sscanf(line + 6, "%31s %47[A-Za-z0-9_][%d] =", type, name, count) != 3 || *count <= 0) {
		return false;
	}
	const char* v = strchr(line, '=');
	v = v ? strchr(v, '(') : NULL;
	*values = v ? v + 1 : NULL;
	return v != NULL;
}

// First pass over both stages: the uniforms (members, then samplers) and the varyings' locations.
static void gather(struct Prog* p, const char* source, char varyings[][48], int* varyingCount) {
	for (const char* line = source; line && *line;) {
		const char* end = strchr(line, '\n');
		size_t const len = end ? (size_t) (end - line) : strlen(line);
		char buf[1024];
		snprintf(buf, sizeof(buf), "%.*s", (int) (len < sizeof(buf) - 1 ? len : sizeof(buf) - 1), line);
		char type[32], names[24][48];
		int counts[24], n = 0;
		if (uniformLine(buf, type, sizeof(type), names, counts, &n, 24)) {
			for (int i = 0; i < n; ++i) {
				if (strstr(type, "sampler")) {
					int k = 0;
					while (k < p->samplers && strcmp(p->sampler[k], names[i])) {
						++k;
					}
					if (k == p->samplers && k < SAMPLERS) {
						snprintf(p->sampler[p->samplers++], 48, "%s", names[i]);
					}
					continue;
				}
				addMember(p, names[i], type, counts[i], NULL);
			}
		}
		const char* values;
		if (constArray(line, type, names[0], &counts[0], &values)) {
			addMember(p, names[0], type, counts[0], values);
		}
		const char* q = skipSpace(buf);
		bool const flat = startsWord(q, "flat");
		if (flat) {
			q = skipSpace(q + 4);
		}
		if (startsWord(q, "out") && varyings) {
			const char* name = strrchr(q, ' ');
			if (name && *varyingCount < 16) {
				snprintf(varyings[*varyingCount], 48, "%.*s", (int) strcspn(name + 1, ";"), name + 1);
				++*varyingCount;
			}
		}
		line = end ? end + 1 : NULL;
	}
	p->blockSize = (p->blockSize + 15) / 16 * 16;
}

static int varyingLocation(char varyings[][48], int count, const char* name) {
	for (int i = 0; i < count; ++i) {
		if (!strcmp(varyings[i], name)) {
			return i;
		}
	}
	return count;
}

static void putBlock(const struct Prog* p, struct Str* out) {
	puts_(out, "layout(std140, set = 0, binding = 0) uniform VkglUniforms {\n");
	for (int k = 0; k < p->members; ++k) {
		char decl[160];
		if (p->member[k].count) {
			snprintf(decl, sizeof(decl), "\thighp %s %s[%d];\n", p->member[k].type, p->member[k].name, p->member[k].count);
		} else {
			snprintf(decl, sizeof(decl), "\thighp %s %s;\n", p->member[k].type, p->member[k].name);
		}
		puts_(out, decl);
	}
	puts_(out, "};\n");
}

// A constant array's values (numbers in its source, constructors' names skipped) into the shadow.
static void putValues(struct Prog* p, struct Member* m) {
	int size = 4, align;
	bool isInt = true;
	typeLayout(m->type, &size, &align, &isInt);
	int const per = size / 4;
	int i = 0;
	for (const char* c = m->values; *c && !(c[0] == ')' && c[1] == ';') && i < m->count * per;) {
		if ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') || *c == '_') {
			while ((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') || (*c >= '0' && *c <= '9') || *c == '_') {
				++c;
			}
		} else if ((*c >= '0' && *c <= '9') || ((*c == '-' || *c == '.') && c[1] >= '0' && c[1] <= '9')) {
			char* end;
			uint8_t* at = p->shadow + m->offset + (i / per) * m->size + (i % per) * 4;
			if (isInt) {
				int32_t const v = (int32_t) strtol(c, &end, 0);
				memcpy(at, &v, 4);
			} else {
				float const v = strtof(c, &end);
				memcpy(at, &v, 4);
			}
			c = end;
			++i;
		} else {
			++c;
		}
	}
	m->values = NULL;
}

// Second pass: one stage's Vulkan GLSL.
static char* translate(const struct Prog* p, const char* source, bool vertex, char varyings[][48], int varyingCount, uint32_t* inputs) {
	struct Str out = { 0 };
	bool blockDone = false;
	for (const char* line = source; line && *line;) {
		const char* end = strchr(line, '\n');
		size_t const len = end ? (size_t) (end - line) : strlen(line);
		char buf[2048];
		snprintf(buf, sizeof(buf), "%.*s", (int) (len < sizeof(buf) - 1 ? len : sizeof(buf) - 1), line);
		const char* const here = line;
		line = end ? end + 1 : NULL;
		const char* q = skipSpace(buf);
		if (!strncmp(q, "#version", 8)) {
			puts_(&out, "#version 310 es\n");
			continue;
		}
		char type[32], names[24][48];
		int counts[24], n = 0;
		if (uniformLine(buf, type, sizeof(type), names, counts, &n, 24)) {
			if (strstr(type, "sampler")) {
				for (int i = 0; i < n; ++i) {
					int k = 0;
					while (k < p->samplers && strcmp(p->sampler[k], names[i])) {
						++k;
					}
					char decl[160];
					if (k == p->input) {
						snprintf(decl, sizeof(decl),
						         "layout(set = 0, binding = %d) uniform highp sampler2D %s;\n"
						         "layout(input_attachment_index = 0, set = 0, binding = %d) uniform highp subpassInput vkglMemory;\n",
						         1 + k, names[i], 1 + p->samplers);
					} else {
						snprintf(decl, sizeof(decl), "layout(set = 0, binding = %d) uniform highp %s %s;\n", 1 + k, type, names[i]);
					}
					puts_(&out, decl);
				}
			} else if (!blockDone && p->members) {
				putBlock(p, &out);
				blockDone = true;
			}
			continue;
		}
		const char* values;
		if (constArray(here, type, names[0], &counts[0], &values)) {
			if (!blockDone && p->members) {
				putBlock(p, &out);
				blockDone = true;
			}
			const char* close = strstr(values, ");");
			line = close ? strchr(close, '\n') : NULL;
			line = line ? line + 1 : NULL;
			continue;
		}
		bool const flat = startsWord(q, "flat");
		const char* r = flat ? skipSpace(q + 4) : q;
		if (startsWord(r, "out") || (startsWord(r, "in") && !vertex)) {
			const char* name = strrchr(r, ' ');
			char nm[48];
			snprintf(nm, sizeof(nm), "%.*s", name ? (int) strcspn(name + 1, ";") : 0, name ? name + 1 : "");
			int const location = !vertex && startsWord(r, "out") ? 0 : varyingLocation(varyings, varyingCount, nm);
			char decl[64];
			snprintf(decl, sizeof(decl), "layout(location = %d) ", location);
			puts_(&out, decl);
			puts_(&out, q);
			puts_(&out, "\n");
			continue;
		}
		if (vertex && !strncmp(q, "layout(location", 15) && strstr(q, ") in ")) {
			*inputs |= 1u << atoi(q + strcspn(q, "=") + 1);
		}
		// The rest as it is, with Vulkan's names.
		static const char fetch[] = "texelFetch(uMemory, ivec2(gl_FragCoord.xy), 0)";
		static const char either[] = "(vkglTarget != 0 ? subpassLoad(vkglMemory) : texelFetch(uMemory, ivec2(gl_FragCoord.xy), 0))";
		for (char* v = strstr(buf, fetch); v && p->input >= 0; v = strstr(v + strlen(either), fetch)) {
			char rest[2048];
			snprintf(rest, sizeof(rest), "%s%s", either, v + strlen(fetch));
			snprintf(v, sizeof(buf) - (size_t) (v - buf), "%s", rest);
		}
		for (char* v = strstr(buf, "gl_VertexID"); v; v = strstr(v + 1, "gl_VertexID")) {
			char rest[2048];
			snprintf(rest, sizeof(rest), "gl_VertexIndex%s", v + 11);
			snprintf(v, sizeof(buf) - (size_t) (v - buf), "%s", rest);
		}
		if (vertex && strstr(buf, "void main()")) {
			char* m = strstr(buf, "void main()");
			char rest[2048];
			snprintf(rest, sizeof(rest), "void vkglMain()%s", m + 11);
			snprintf(m, sizeof(buf) - (size_t) (m - buf), "%s", rest);
		}
		puts_(&out, buf);
		puts_(&out, "\n");
	}
	if (vertex) {
		puts_(&out, "void main() {\n\tvkglMain();\n\tgl_Position.z = (gl_Position.z + gl_Position.w) * 0.5;\n}\n");
	}
	return out.p;
}

// SPIR-V for a stage, or NULL with the compiler's messages in log.
static uint32_t* spirv(const char* source, bool vertex, size_t* words, char** log) {
	glslang_input_t input = {
		.language = GLSLANG_SOURCE_GLSL,
		.stage = vertex ? GLSLANG_STAGE_VERTEX : GLSLANG_STAGE_FRAGMENT,
		.client = GLSLANG_CLIENT_VULKAN,
		.client_version = GLSLANG_TARGET_VULKAN_1_0,
		.target_language = GLSLANG_TARGET_SPV,
		.target_language_version = GLSLANG_TARGET_SPV_1_0,
		.code = source,
		.default_version = 310,
		.default_profile = GLSLANG_ES_PROFILE,
		.messages = GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT,
		.resource = glslang_default_resource(),
	};
	glslang_shader_t* shader = glslang_shader_create(&input);
	uint32_t* code = NULL;
	if (!glslang_shader_preprocess(shader, &input) || !glslang_shader_parse(shader, &input)) {
		const char* msg = glslang_shader_get_info_log(shader);
		*log = strdup(msg ? msg : "compile failed");
		glslang_shader_delete(shader);
		return NULL;
	}
	glslang_program_t* program = glslang_program_create();
	glslang_program_add_shader(program, shader);
	if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
		const char* msg = glslang_program_get_info_log(program);
		*log = strdup(msg ? msg : "link failed");
	} else {
		glslang_program_SPIRV_generate(program, input.stage);
		*words = glslang_program_SPIRV_get_size(program);
		code = malloc(*words * 4);
		if (code) {
			glslang_program_SPIRV_get(program, code);
		}
	}
	glslang_program_delete(program);
	glslang_shader_delete(shader);
	return code;
}

static uint32_t wordsHash(uint32_t h, const uint32_t* words, size_t count) {
	for (size_t i = 0; i < count; ++i) {
		h = (h ^ words[i]) * 16777619u;
	}
	return h;
}

// A program's SPIR-V kept on disk by its sources' hash (the shader cache folder): the two stages'
// word counts and a hash of their words, then the words. A file that doesn't check out is left (the
// driver would take broken SPIR-V as it is).
static void cachedSpirv(const char* path, uint32_t** v, size_t* vWords, uint32_t** f, size_t* fWords) {
	FILE* in = fopen(path, "rb");
	uint32_t counts[3];
	if (!in || fread(counts, 4, 3, in) != 3 || !counts[0] || !counts[1] || counts[0] > (1u << 22) || counts[1] > (1u << 22)) {
		if (in) {
			fclose(in);
		}
		return;
	}
	*v = malloc((size_t) counts[0] * 4);
	*f = malloc((size_t) counts[1] * 4);
	if (*v && *f && fread(*v, 4, counts[0], in) == counts[0] && fread(*f, 4, counts[1], in) == counts[1] && (*v)[0] == 0x07230203u &&
	    (*f)[0] == 0x07230203u && wordsHash(wordsHash(2166136261u, *v, counts[0]), *f, counts[1]) == counts[2]) {
		*vWords = counts[0], *fWords = counts[1];
	} else {
		free(*v);
		free(*f);
		*v = *f = NULL;
	}
	fclose(in);
}

static void keepSpirv(const char* path, const uint32_t* v, size_t vWords, const uint32_t* f, size_t fWords) {
	char temp[4400];
	snprintf(temp, sizeof(temp), "%s.%ld.tmp", path, (long) getpid());
	FILE* out = fopen(temp, "wb");
	uint32_t const counts[3] = { (uint32_t) vWords, (uint32_t) fWords, wordsHash(wordsHash(2166136261u, v, vWords), f, fWords) };
	if (out && fwrite(counts, 4, 3, out) == 3 && fwrite(v, 4, vWords, out) == vWords && fwrite(f, 4, fWords, out) == fWords && !fclose(out)) {
		rename(temp, path);
	} else if (out) {
		fclose(out);
		remove(temp);
	}
}

static bool linkProgram(struct Prog* p) {
	const char* vs = p->shader[0] && p->shader[0] < (GLuint) V.shaderCount ? V.shader[p->shader[0]].source : NULL;
	const char* fs = p->shader[1] && p->shader[1] < (GLuint) V.shaderCount ? V.shader[p->shader[1]].source : NULL;
	free(p->log);
	p->log = NULL;
	if (!vs || !fs) {
		p->log = strdup("a stage is missing");
		return false;
	}
	p->members = p->samplers = p->blockSize = 0;
	char varyings[16][48];
	int varyingCount = 0;
	gather(p, vs, varyings, &varyingCount);
	gather(p, fs, NULL, NULL);
	p->input = p->target = -1;
	for (int i = 0; i < p->samplers; ++i) {
		p->input = !strcmp(p->sampler[i], "uMemory") ? i : p->input;
	}
	if (p->input >= 0 && p->members < MEMBERS) {
		struct Member* m = &p->member[p->target = p->members++];
		*m = (struct Member) { "vkglTarget", "int", p->blockSize, 4, 0, true, NULL };
		p->blockSize += 16;
	}
	p->inputs = 0;
	char* vText = translate(p, vs, true, varyings, varyingCount, &p->inputs);
	char* fText = translate(p, fs, false, varyings, varyingCount, NULL);
	size_t vWords = 0, fWords = 0;
	uint32_t* vCode = NULL;
	uint32_t* fCode = NULL;
	char path[4300] = "";
	if (vText && fText && glesCacheDirectory()[0]) {
		uint64_t h = 0xCBF29CE484222325ull;
		for (const char* c = vText; *c; ++c) {
			h = (h ^ (uint8_t) *c) * 0x100000001B3ull;
		}
		for (const char* c = fText; *c; ++c) {
			h = (h ^ (uint8_t) *c) * 0x100000001B3ull;
		}
		snprintf(path, sizeof(path), "%s/vk-%016llx.spv", glesCacheDirectory(), (unsigned long long) h);
		cachedSpirv(path, &vCode, &vWords, &fCode, &fWords);
	}
	if (!vCode) {
		vCode = vText ? spirv(vText, true, &vWords, &p->log) : NULL;
		fCode = vCode && fText ? spirv(fText, false, &fWords, &p->log) : NULL;
		if (vCode && fCode && path[0]) {
			keepSpirv(path, vCode, vWords, fCode, fWords);
		}
	}
	if (getenv("N64_VKGL_SHADERS") && (!vCode || !fCode)) {
		fprintf(stderr, "vkgl: program failed:\n%s\n---\n%s\n---\n%s\n", vText ? vText : "", fText ? fText : "", p->log ? p->log : "");
	}
	free(vText);
	free(fText);
	if (!vCode || !fCode) {
		free(vCode);
		free(fCode);
		return false;
	}
	VkShaderModuleCreateInfo vm = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, vWords * 4, vCode };
	VkShaderModuleCreateInfo fm = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, fWords * 4, fCode };
	bool ok = vkCreateShaderModule(V.device, &vm, NULL, &p->vertex) == VK_SUCCESS && vkCreateShaderModule(V.device, &fm, NULL, &p->fragment) == VK_SUCCESS;
	free(vCode);
	free(fCode);
	VkDescriptorSetLayoutBinding bindings[2 + SAMPLERS];
	uint32_t n = 0;
	bindings[n++] = (VkDescriptorSetLayoutBinding) { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
	for (int i = 0; i < p->samplers; ++i) {
		bindings[n++] = (VkDescriptorSetLayoutBinding) { (uint32_t) (1 + i), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
			                                             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
	}
	if (p->input >= 0) {
		bindings[n++] = (VkDescriptorSetLayoutBinding) { (uint32_t) (1 + p->samplers), VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
	}
	VkDescriptorSetLayoutCreateInfo sl = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, n, bindings };
	ok = ok && vkCreateDescriptorSetLayout(V.device, &sl, NULL, &p->setLayout) == VK_SUCCESS;
	VkPipelineLayoutCreateInfo pl = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &p->setLayout, 0, NULL };
	ok = ok && vkCreatePipelineLayout(V.device, &pl, NULL, &p->layout) == VK_SUCCESS;
	free(p->shadow);
	p->shadow = calloc(1, (size_t) (p->blockSize ? p->blockSize : 16));
	p->shadowDirty = true;
	for (int k = 0; k < p->members && p->shadow; ++k) {
		if (p->member[k].values) {
			putValues(p, &p->member[k]);
		}
	}
	memset(p->unit, 0, sizeof(p->unit));
	if (!ok) {
		p->log = strdup("Vulkan objects unavailable");
	}
	return ok;
}

// ---- Pipelines and drawing

struct PipeKey {
	VkRenderPass pass;
	uint32_t prog;
	uint16_t mode, depthFunc, blendSrc[2], blendDst[2], blendEq[2], stencilFunc, stencilFail, stencilZfail, stencilZpass;
	uint8_t blend, colorMask, depthTest, depthWrite, stencilTest, offset, hasColor, pad;
	uint8_t format[ATTRIBS];
	uint16_t stride[ATTRIBS];
};

_Static_assert(sizeof(struct PipeKey) <= 128, "pipeline key");

static VkBlendFactor blendFactor(GLenum f) {
	switch (f) {
	case 0: return VK_BLEND_FACTOR_ZERO;
	case 1: return VK_BLEND_FACTOR_ONE;
	case 0x300: return VK_BLEND_FACTOR_SRC_COLOR;
	case 0x301: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
	case 0x302: return VK_BLEND_FACTOR_SRC_ALPHA;
	case 0x303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	case 0x304: return VK_BLEND_FACTOR_DST_ALPHA;
	case 0x305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
	case 0x306: return VK_BLEND_FACTOR_DST_COLOR;
	case 0x307: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
	case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
	case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
	case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
	case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
	}
	return VK_BLEND_FACTOR_ONE;
}

static VkBlendOp blendOp(GLenum e) {
	return e == 0x8007 ? VK_BLEND_OP_MIN : e == 0x8008 ? VK_BLEND_OP_MAX : e == 0x800A ? VK_BLEND_OP_SUBTRACT : e == 0x800B ? VK_BLEND_OP_REVERSE_SUBTRACT : VK_BLEND_OP_ADD;
}

static VkCompareOp compareOp(GLenum f) {
	return f >= 0x200 && f <= 0x207 ? (VkCompareOp) (f - 0x200) : VK_COMPARE_OP_ALWAYS;
}

static VkStencilOp stencilOp(GLenum op) {
	switch (op) {
	case 0: return VK_STENCIL_OP_ZERO;
	case 0x1E01: return VK_STENCIL_OP_REPLACE;
	case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
	case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
	case 0x150A: return VK_STENCIL_OP_INVERT;
	}
	return VK_STENCIL_OP_KEEP;
}

static const VkFormat attribFormats[6] = { VK_FORMAT_UNDEFINED, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT,
	                                       VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32_UINT };

static uint32_t keyHash(const struct PipeKey* k) {
	const uint8_t* b = (const uint8_t*) k;
	uint32_t h = 2166136261u;
	for (size_t i = 0; i < sizeof(*k); ++i) {
		h = (h ^ b[i]) * 16777619u;
	}
	return h;
}

static VkPipeline pipeline(const struct PipeKey* k, struct Prog* p) {
	if (V.lastPipe && !memcmp(V.lastKey, k, sizeof(*k))) {
		return V.lastPipe;
	}
	uint32_t const h = keyHash(k);
	for (uint32_t i = 0; i < 4096; ++i) {
		int const at = V.pipeHash[(h + i) & 4095];
		if (!at) {
			break;
		}
		if (!memcmp(V.pipes[at - 1].key, k, sizeof(*k))) {
			memcpy(V.lastKey, k, sizeof(*k));
			V.lastPipe = V.pipes[at - 1].pipeline;
			return V.lastPipe;
		}
	}
	VkPipelineShaderStageCreateInfo stages[2] = {
		{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, p->vertex, "main", NULL },
		{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, p->fragment, "main", NULL },
	};
	VkVertexInputBindingDescription bindings[ATTRIBS];
	VkVertexInputAttributeDescription attributes[ATTRIBS];
	uint32_t n = 0;
	for (uint32_t i = 0; i < ATTRIBS; ++i) {
		if (k->format[i]) {
			bindings[n] = (VkVertexInputBindingDescription) { i, k->stride[i], VK_VERTEX_INPUT_RATE_VERTEX };
			attributes[n] = (VkVertexInputAttributeDescription) { i, i, attribFormats[k->format[i]], 0 };
			++n;
		}
	}
	VkPipelineVertexInputStateCreateInfo vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, NULL, 0, n, bindings, n, attributes };
	VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0,
		                                          k->mode == GL_TRIANGLE_STRIP ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
		                                          : k->mode == GL_TRIANGLE_FAN ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN
		                                                                       : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		                                          VK_FALSE };
	VkPipelineViewportStateCreateInfo vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL };
	VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_FALSE,
		                                          VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, k->offset, 0, 0, 0, 1.0f };
	VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 0, NULL, VK_FALSE, VK_FALSE };
	VkStencilOpState so = { stencilOp(k->stencilFail), stencilOp(k->stencilZpass), stencilOp(k->stencilZfail), compareOp(k->stencilFunc), 0xFF, 0xFF, 0 };
	VkPipelineDepthStencilStateCreateInfo ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, k->depthTest, k->depthWrite,
		                                         compareOp(k->depthFunc), VK_FALSE, k->stencilTest, so, so, 0, 1 };
	VkPipelineColorBlendAttachmentState ba = { k->blend, blendFactor(k->blendSrc[0]), blendFactor(k->blendDst[0]), blendOp(k->blendEq[0]),
		                                       blendFactor(k->blendSrc[1]), blendFactor(k->blendDst[1]), blendOp(k->blendEq[1]), k->colorMask };
	VkPipelineColorBlendStateCreateInfo cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL,
		                                       k->hasColor && V.ordered ? VK_PIPELINE_COLOR_BLEND_STATE_CREATE_RASTERIZATION_ORDER_ATTACHMENT_ACCESS_BIT_EXT : 0, VK_FALSE, VK_LOGIC_OP_COPY,
		                                       k->hasColor ? 1u : 0u, &ba, { 0, 0, 0, 0 } };
	static const VkDynamicState dynamic[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS,
		                                      VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
		                                      VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE };
	VkPipelineDynamicStateCreateInfo dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 7, dynamic };
	VkGraphicsPipelineCreateInfo info = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, NULL, 0, 2, stages, &vi, &ia, NULL, &vp, &rs, &ms, &ds, &cb,
		                                  &dy, p->layout, k->pass, 0, VK_NULL_HANDLE, -1 };
	VkPipeline pipe;
	if (vkCreateGraphicsPipelines(V.device, V.pipelineCache, 1, &info, NULL, &pipe) != VK_SUCCESS) {
		return VK_NULL_HANDLE;
	}
	if (V.pipeCount == V.pipeCapacity) {
		int const cap = V.pipeCapacity ? V.pipeCapacity * 2 : 64;
		struct Pipe* more = realloc(V.pipes, (size_t) cap * sizeof(*more));
		if (!more) {
			vkDestroyPipeline(V.device, pipe, NULL);
			return VK_NULL_HANDLE;
		}
		V.pipes = more, V.pipeCapacity = cap;
	}
	struct Pipe* e = &V.pipes[V.pipeCount++];
	memset(e->key, 0, sizeof(e->key));
	memcpy(e->key, k, sizeof(*k));
	e->pipeline = pipe;
	for (uint32_t i = 0; i < 4096; ++i) {
		if (!V.pipeHash[(h + i) & 4095]) {
			V.pipeHash[(h + i) & 4095] = V.pipeCount;
			break;
		}
	}
	memcpy(V.lastKey, k, sizeof(*k));
	V.lastPipe = pipe;
	return pipe;
}

static struct Prog* progGet(GLuint name) {
	return name && name < (GLuint) V.progCount && V.prog[name].used ? &V.prog[name] : NULL;
}

static struct Img* dummyImage(void);

// The program's descriptor set (its uniforms at *offset in this turn's ring): reused while nothing it
// holds changed. False when the turn moved on while taking room (then try again).
static bool bindings(struct Prog* p, VkDescriptorSet* set, uint32_t* offset) {
	uint64_t const serial = V.sub[V.current].serial;
	// uMemory reads the image drawn to (the input attachment) when that's the texture bound to it.
	struct Fbo* drawing = fboGet(V.st.drawFbo);
	struct Img* own = drawing ? fboColor(drawing) : NULL;
	GLuint const memory = p->input >= 0 ? V.st.texture[p->unit[p->input] >= 0 && p->unit[p->input] < UNITS ? p->unit[p->input] : 0] : 0;
	int32_t const isTarget = p->input >= 0 && own && drawing && !drawing->colorIsRb && memory && memory == drawing->color;
	if (p->target >= 0 && memcmp(p->shadow + p->member[p->target].offset, &isTarget, 4)) {
		memcpy(p->shadow + p->member[p->target].offset, &isTarget, 4);
		p->shadowDirty = true;
	}
	if (p->shadowDirty || p->uboSerial != serial) {
		size_t const at = ringTake(true, (size_t) p->blockSize ? (size_t) p->blockSize : 16, V.limits.minUniformBufferOffsetAlignment);
		if (at == SIZE_MAX || V.sub[V.current].serial != serial) {
			return false;
		}
		memcpy(V.sub[V.current].ubo.map + at, p->shadow, (size_t) p->blockSize);
		p->uboOffset = (uint32_t) at, p->uboSerial = serial, p->shadowDirty = false;
	}
	*offset = p->uboOffset;
	VkImageView const inputView = p->input >= 0 && own ? own->view : VK_NULL_HANDLE;
	VkDescriptorImageInfo images[SAMPLERS + 1];
	for (int i = 0; i < p->samplers; ++i) {
		int const unit = p->unit[i] >= 0 && p->unit[i] < UNITS ? p->unit[i] : 0;
		// The image drawn to isn't sampled as well (it is the input attachment then).
		GLuint const t = i == p->input && isTarget ? 0 : V.st.texture[unit];
		struct Tex* tex = t && t < (GLuint) V.texCount && V.tex[t].used && V.tex[t].img.image ? &V.tex[t] : NULL;
		struct Img* img = tex ? &tex->img : dummyImage();
		touch(img);
		images[i] = (VkDescriptorImageInfo) { tex ? sampler(tex->filter[0], tex->wrap[0], tex->wrap[1], tex->filter[1]) : sampler(GL_NEAREST, 0, 0, GL_NEAREST),
			                                  img->view, VK_IMAGE_LAYOUT_GENERAL };
	}
	images[p->samplers] = (VkDescriptorImageInfo) { VK_NULL_HANDLE, inputView ? inputView : dummyImage()->view, VK_IMAGE_LAYOUT_GENERAL };
	// A set made this turn with the same images is used again (views are only destroyed when their
	// turn is done, so the same view this turn is the same image).
	int const count = p->samplers + 1;
	uint64_t h = (uint64_t) (uintptr_t) p->setLayout;
	for (int i = 0; i < count; ++i) {
		h = (h ^ (uint64_t) (uintptr_t) images[i].imageView) * 0x100000001B3ull;
		h = (h ^ (uint64_t) (uintptr_t) images[i].sampler) * 0x100000001B3ull;
	}
	struct SetEntry* entry = NULL;
	for (int probe = 0; probe < 8; ++probe) {
		struct SetEntry* e = &V.sub[V.current].setCache[(h + (uint64_t) probe) & (SET_CACHE - 1)];
		if (e->serial != serial) {
			entry = e;
			break;
		}
		bool same = e->layout == p->setLayout;
		for (int i = 0; i < count && same; ++i) {
			same = e->images[i].imageView == images[i].imageView && e->images[i].sampler == images[i].sampler;
		}
		if (same) {
			*set = e->set;
			return true;
		}
	}
	VkDescriptorSetAllocateInfo alloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, V.sub[V.current].sets, 1, &p->setLayout };
	if (vkAllocateDescriptorSets(V.device, &alloc, set) != VK_SUCCESS) {
		submit(false);
		return false;
	}
	VkDescriptorBufferInfo ubo = { V.sub[V.current].ubo.buffer, 0, (VkDeviceSize) (p->blockSize ? p->blockSize : 16) };
	VkWriteDescriptorSet writes[2 + SAMPLERS];
	uint32_t n = 0;
	writes[n++] = (VkWriteDescriptorSet) { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, *set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, NULL, &ubo, NULL };
	if (p->input >= 0) {
		writes[n++] = (VkWriteDescriptorSet) { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, *set, (uint32_t) (1 + p->samplers), 0, 1,
			                                   VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, &images[p->samplers], NULL, NULL };
	}
	for (int i = 0; i < p->samplers; ++i) {
		writes[n++] = (VkWriteDescriptorSet) { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, *set, (uint32_t) (1 + i), 0, 1,
			                                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &images[i], NULL, NULL };
	}
	vkUpdateDescriptorSets(V.device, n, writes, 0, NULL);
	if (entry) {
		entry->serial = serial, entry->layout = p->setLayout, entry->set = *set;
		memcpy(entry->images, images, (size_t) count * sizeof(images[0]));
	}
	return true;
}

static void glDrawArrays_(GLenum mode, GLint first, GLsizei count) {
	struct Prog* p = progGet(V.st.program);
	if (!p || !p->linked || count <= 0) {
		return;
	}
	submitDue();
	VkDescriptorSet set = VK_NULL_HANDLE;
	uint32_t offset = 0;
	for (int tries = 0; tries < 3 && !bindings(p, &set, &offset); ++tries) {
		set = VK_NULL_HANDLE;
	}
	if (!set || !passBegin()) {
		return;
	}
	VkCommandBuffer cb = V.sub[V.current].cb;
	struct Fbo* f = fboGet(V.passFbo);
	bool const hasColor = fboColor(f) != NULL, hasDepth = fboDepth(f) != NULL;
	struct PipeKey k;
	memset(&k, 0, sizeof(k));
	k.pass = V.pass, k.prog = (uint32_t) V.st.program, k.mode = (uint16_t) mode;
	k.hasColor = hasColor;
	k.blend = hasColor && V.st.blend;
	if (k.blend) {
		for (int i = 0; i < 2; ++i) {
			k.blendSrc[i] = (uint16_t) V.st.blendSrc[i], k.blendDst[i] = (uint16_t) V.st.blendDst[i], k.blendEq[i] = (uint16_t) V.st.blendEq[i];
		}
	}
	k.colorMask = (uint8_t) (V.st.colorMask[0] | V.st.colorMask[1] << 1 | V.st.colorMask[2] << 2 | V.st.colorMask[3] << 3);
	k.depthTest = hasDepth && V.st.depthTest;
	k.depthWrite = k.depthTest && V.st.depthMask;
	k.depthFunc = k.depthTest ? (uint16_t) V.st.depthFunc : 0;
	k.stencilTest = hasDepth && V.st.stencilTest;
	if (k.stencilTest) {
		k.stencilFunc = (uint16_t) V.st.stencilFunc, k.stencilFail = (uint16_t) V.st.stencilFail;
		k.stencilZfail = (uint16_t) V.st.stencilZfail, k.stencilZpass = (uint16_t) V.st.stencilZpass;
	}
	k.offset = V.st.polygonOffset;
	struct Vao* vao = V.st.vao < (GLuint) V.vaoCount ? &V.vao[V.st.vao] : NULL;
	VkBuffer buffers[ATTRIBS];
	VkDeviceSize offsets[ATTRIBS];
	for (int i = 0; i < ATTRIBS; ++i) {
		buffers[i] = VK_NULL_HANDLE;
		if (!(p->inputs & (1u << i)) || !vao || !vao->attrib[i].enabled) {
			continue;
		}
		struct Attrib* a = &vao->attrib[i];
		struct Buf* b = a->buffer && a->buffer < (GLuint) V.bufCount ? &V.buf[a->buffer] : NULL;
		if (!b || !b->buffer) {
			continue;
		}
		k.format[i] = a->integer ? 5 : (uint8_t) (a->size >= 1 && a->size <= 4 ? a->size : 4);
		k.stride[i] = (uint16_t) (a->stride ? a->stride : (a->integer ? 4 : 4 * a->size));
		buffers[i] = b->buffer;
		offsets[i] = (VkDeviceSize) a->offset;
		b->serial = V.sub[V.current].serial;
	}
	VkPipeline pipe = pipeline(&k, p);
	if (!pipe) {
		return;
	}
	if (pipe != V.boundPipeline) {
		vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
		V.boundPipeline = pipe;
	}
	VkViewport viewport = { (float) V.st.viewport[0], (float) V.st.viewport[1], (float) V.st.viewport[2], (float) V.st.viewport[3], 0.0f, 1.0f };
	if (viewport.width <= 0 || viewport.height <= 0) {
		viewport.width = (float) V.passWidth, viewport.height = (float) V.passHeight;
	}
	if (!V.bound.valid || memcmp(&viewport, &V.bound.viewport, sizeof(viewport))) {
		vkCmdSetViewport(cb, 0, 1, &viewport);
		V.bound.viewport = viewport;
	}
	int x0 = 0, y0 = 0, x1 = V.passWidth, y1 = V.passHeight;
	if (V.st.scissorOn) {
		x0 = V.st.scissor[0] > 0 ? V.st.scissor[0] : 0, y0 = V.st.scissor[1] > 0 ? V.st.scissor[1] : 0;
		x1 = V.st.scissor[0] + V.st.scissor[2], y1 = V.st.scissor[1] + V.st.scissor[3];
		x1 = x1 < V.passWidth ? x1 : V.passWidth, y1 = y1 < V.passHeight ? y1 : V.passHeight;
	}
	if (x1 <= x0 || y1 <= y0) {
		return;
	}
	VkRect2D scissor = { { x0, y0 }, { (uint32_t) (x1 - x0), (uint32_t) (y1 - y0) } };
	bool const fresh = !V.bound.valid;
	if (fresh || memcmp(&scissor, &V.bound.scissor, sizeof(scissor))) {
		vkCmdSetScissor(cb, 0, 1, &scissor);
		V.bound.scissor = scissor;
	}
	if (fresh || V.bound.bias[0] != V.st.offsetUnits || V.bound.bias[1] != V.st.offsetFactor) {
		vkCmdSetDepthBias(cb, V.st.offsetUnits, 0.0f, V.st.offsetFactor);
		V.bound.bias[0] = V.st.offsetUnits, V.bound.bias[1] = V.st.offsetFactor;
	}
	if (fresh || memcmp(V.bound.blend, V.st.blendColor, sizeof(V.bound.blend))) {
		vkCmdSetBlendConstants(cb, V.st.blendColor);
		memcpy(V.bound.blend, V.st.blendColor, sizeof(V.bound.blend));
	}
	uint32_t const stencil[2] = { V.st.stencilMask & 0xFF, (uint32_t) V.st.stencilRef & 0xFF };
	if (fresh || memcmp(stencil, V.bound.stencil, sizeof(stencil))) {
		vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, stencil[0]);
		vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
		vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, stencil[1]);
		memcpy(V.bound.stencil, stencil, sizeof(stencil));
	}
	if (fresh || V.bound.set != set || V.bound.offset != offset || V.bound.layout != p->layout) {
		vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p->layout, 0, 1, &set, 1, &offset);
		V.bound.set = set, V.bound.offset = offset, V.bound.layout = p->layout;
	}
	for (uint32_t i = 0; i < ATTRIBS; ++i) {
		if (buffers[i] && (fresh || V.bound.vb[i] != buffers[i] || V.bound.vo[i] != offsets[i])) {
			vkCmdBindVertexBuffers(cb, i, 1, &buffers[i], &offsets[i]);
			V.bound.vb[i] = buffers[i], V.bound.vo[i] = offsets[i];
		}
	}
	V.bound.valid = true;
	vkCmdDraw(cb, (uint32_t) count, 1, (uint32_t) first, 0);
	++V.pendingDraws;
}

static void glClear_(GLbitfield mask) {
	if (!passBegin()) {
		return;
	}
	struct Fbo* f = fboGet(V.passFbo);
	VkClearAttachment att[2];
	uint32_t n = 0;
	if ((mask & GL_COLOR_BUFFER_BIT) && fboColor(f)) {
		att[n].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, att[n].colorAttachment = 0;
		memcpy(att[n].clearValue.color.float32, V.st.clearColor, sizeof(V.st.clearColor));
		++n;
	}
	VkImageAspectFlags aspects = ((mask & GL_DEPTH_BUFFER_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | ((mask & GL_STENCIL_BUFFER_BIT) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
	if (aspects && fboDepth(f)) {
		att[n].aspectMask = aspects, att[n].colorAttachment = 0;
		att[n].clearValue.depthStencil = (VkClearDepthStencilValue) { V.st.clearDepth, (uint32_t) V.st.clearStencil };
		++n;
	}
	if (!n) {
		return;
	}
	int x0 = 0, y0 = 0, x1 = V.passWidth, y1 = V.passHeight;
	if (V.st.scissorOn) {
		x0 = V.st.scissor[0] > 0 ? V.st.scissor[0] : 0, y0 = V.st.scissor[1] > 0 ? V.st.scissor[1] : 0;
		x1 = V.st.scissor[0] + V.st.scissor[2], y1 = V.st.scissor[1] + V.st.scissor[3];
		x1 = x1 < V.passWidth ? x1 : V.passWidth, y1 = y1 < V.passHeight ? y1 : V.passHeight;
	}
	if (x1 <= x0 || y1 <= y0) {
		return;
	}
	VkClearRect rect = { { { x0, y0 }, { (uint32_t) (x1 - x0), (uint32_t) (y1 - y0) } }, 0, 1 };
	vkCmdClearAttachments(V.sub[V.current].cb, n, att, 1, &rect);
}

// ---- Copies

// size bytes of data into the image's rectangle (RGBA bytes or floats, rows tight).
static void upload(struct Img* img, int x, int y, int w, int h, const void* data, size_t size, VkBuffer from, VkDeviceSize fromOffset) {
	if (w <= 0 || h <= 0 || !img->image) {
		return;
	}
	submitDue();
	VkBuffer src = from;
	VkDeviceSize at = fromOffset;
	if (!src) {
		size_t const ring = size <= STAGE_BYTES ? ringTake(false, size, 16) : SIZE_MAX;
		if (ring != SIZE_MAX) {
			memcpy(V.sub[V.current].stage.map + ring, data, size);
			src = V.sub[V.current].stage.buffer, at = ring;
		} else {
			struct Buf temp = { 0 };
			if (!bufferMake(&temp, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false)) {
				return;
			}
			memcpy(temp.map, data, size);
			garbage(GARBAGE_BUFFER, (uint64_t) (uintptr_t) temp.buffer, (uint64_t) (uintptr_t) temp.memory);
			src = temp.buffer, at = 0;
		}
	}
	VkCommandBuffer cb = transferTo(img);
	layoutTo(cb, img, VK_IMAGE_LAYOUT_GENERAL);
	VkBufferImageCopy region = { at, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { x, y, 0 }, { (uint32_t) w, (uint32_t) h, 1 } };
	vkCmdCopyBufferToImage(cb, src, img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

static struct Img* dummyImage(void) {
	if (!dummy.image && imageMake(&dummy, VK_FORMAT_R8G8B8A8_UNORM, 1, 1)) {
		imageReady(&dummy);
		uint32_t const black = 0;
		upload(&dummy, 0, 0, 1, 1, &black, 4, VK_NULL_HANDLE, 0);
	}
	return &dummy;
}

// ---- GL: objects

#define NAMES(type, array, count)                                         \
	static GLuint new_##type(void) {                                      \
		for (int i = 1; i < count; ++i) {                                 \
			if (!array[i].used) {                                         \
				memset(&array[i], 0, sizeof(array[i]));                   \
				array[i].used = true;                                     \
				return (GLuint) i;                                        \
			}                                                             \
		}                                                                 \
		int const n = count ? count * 2 : 64;                             \
		void* more = realloc(array, (size_t) n * sizeof(*array));         \
		if (!more) {                                                      \
			return 0;                                                     \
		}                                                                 \
		array = more;                                                     \
		memset(&array[count], 0, (size_t) (n - count) * sizeof(*array));  \
		int const at = count ? count : 1;                                 \
		count = n;                                                        \
		array[at].used = true;                                            \
		return (GLuint) at;                                               \
	}

NAMES(tex, V.tex, V.texCount)
NAMES(rb, V.rb, V.rbCount)
NAMES(fbo, V.fbo, V.fboCount)
NAMES(buf, V.buf, V.bufCount)
NAMES(vao, V.vao, V.vaoCount)
NAMES(shader, V.shader, V.shaderCount)
NAMES(prog, V.prog, V.progCount)
#undef NAMES

static void glGenTextures_(GLsizei n, GLuint* out) {
	for (GLsizei i = 0; i < n; ++i) {
		out[i] = new_tex();
		if (out[i]) {
			V.tex[out[i]].filter[0] = V.tex[out[i]].filter[1] = GL_LINEAR;
			V.tex[out[i]].wrap[0] = V.tex[out[i]].wrap[1] = GL_REPEAT;
		}
	}
}

static void glDeleteTextures_(GLsizei n, const GLuint* names) {
	for (GLsizei i = 0; i < n; ++i) {
		GLuint const t = names[i];
		if (t && t < (GLuint) V.texCount && V.tex[t].used) {
			passEnd();
			imageFree(&V.tex[t].img);
			V.tex[t].used = false;
			for (int u = 0; u < UNITS; ++u) {
				V.st.texture[u] = V.st.texture[u] == t ? 0 : V.st.texture[u];
			}
		}
	}
}

static void glActiveTexture_(GLenum unit) {
	int const u = (int) (unit - GL_TEXTURE0);
	V.st.unit = u >= 0 && u < UNITS ? u : 0;
}

static void glBindTexture_(GLenum target, GLuint t) {
	(void) target;
	if (V.st.texture[V.st.unit] != t) {
		V.st.texture[V.st.unit] = t;
	}
}

static struct Tex* boundTex(void) {
	GLuint const t = V.st.texture[V.st.unit];
	return t && t < (GLuint) V.texCount && V.tex[t].used ? &V.tex[t] : NULL;
}

static void glTexParameteri_(GLenum target, GLenum pname, GLint v) {
	(void) target;
	struct Tex* t = boundTex();
	if (!t) {
		return;
	}
	if (pname == GL_TEXTURE_MIN_FILTER) {
		t->filter[0] = v == GL_LINEAR ? GL_LINEAR : GL_NEAREST;
	} else if (pname == GL_TEXTURE_MAG_FILTER) {
		t->filter[1] = (GLenum) v;
	} else if (pname == GL_TEXTURE_WRAP_S) {
		t->wrap[0] = (GLenum) v;
	} else if (pname == GL_TEXTURE_WRAP_T) {
		t->wrap[1] = (GLenum) v;
	}
}

static void glTexImage2D_(GLenum target, GLint level, GLint internal, GLsizei w, GLsizei h, GLint border, GLenum format, GLenum type, const void* data) {
	(void) target, (void) border, (void) format, (void) type;
	struct Tex* t = boundTex();
	if (!t || level) {
		return;
	}
	VkFormat const f = internal == GL_R32F ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
	struct Buf* unpack = V.st.unpackBuffer && V.st.unpackBuffer < (GLuint) V.bufCount ? &V.buf[V.st.unpackBuffer] : NULL;
	if (!t->img.image || t->img.format != f || t->img.width != w || t->img.height != h) {
		if (V.passOpen && fboColor(fboGet(V.passFbo)) == &t->img) {
			passEnd();
		}
		imageFree(&t->img);
		if (!imageMake(&t->img, f, w, h)) {
			return;
		}
		if (!data && !unpack) {
			imageReady(&t->img);
		}
	}
	if (data || unpack) {
		upload(&t->img, 0, 0, w, h, data, (size_t) w * (size_t) h * 4, unpack ? unpack->buffer : VK_NULL_HANDLE, (VkDeviceSize) (uintptr_t) data);
	}
}

static void glTexSubImage2D_(GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, const void* data) {
	(void) target, (void) format, (void) type;
	struct Tex* t = boundTex();
	if (!t || level || !t->img.image) {
		return;
	}
	struct Buf* unpack = V.st.unpackBuffer && V.st.unpackBuffer < (GLuint) V.bufCount ? &V.buf[V.st.unpackBuffer] : NULL;
	if (unpack) {
		unpack->serial = V.sub[V.current].serial;
	}
	upload(&t->img, x, y, w, h, data, (size_t) w * (size_t) h * 4, unpack ? unpack->buffer : VK_NULL_HANDLE, (VkDeviceSize) (uintptr_t) data);
}

static void glGenRenderbuffers_(GLsizei n, GLuint* out) {
	for (GLsizei i = 0; i < n; ++i) {
		out[i] = new_rb();
	}
}

static void glDeleteRenderbuffers_(GLsizei n, const GLuint* names) {
	for (GLsizei i = 0; i < n; ++i) {
		GLuint const r = names[i];
		if (r && r < (GLuint) V.rbCount && V.rb[r].used) {
			passEnd();
			imageFree(&V.rb[r].img);
			V.rb[r].used = false;
		}
	}
}

static void glBindRenderbuffer_(GLenum target, GLuint r) {
	(void) target;
	V.st.renderbuffer = r;
}

static void glRenderbufferStorage_(GLenum target, GLenum internal, GLsizei w, GLsizei h) {
	(void) target;
	GLuint const r = V.st.renderbuffer;
	if (!r || r >= (GLuint) V.rbCount || !V.rb[r].used) {
		return;
	}
	struct Img* img = &V.rb[r].img;
	VkFormat const f = internal == 0x88F0 ? V.depthFormat : VK_FORMAT_R8G8B8A8_UNORM;
	if (img->image && img->format == f && img->width == w && img->height == h) {
		return;
	}
	passEnd();
	imageFree(img);
	if (imageMake(img, f, w, h)) {
		imageReady(img);
	}
}

static void glRenderbufferStorageMultisample_(GLenum target, GLsizei samples, GLenum internal, GLsizei w, GLsizei h) {
	(void) samples;
	glRenderbufferStorage_(target, internal, w, h);
}

static void glGenFramebuffers_(GLsizei n, GLuint* out) {
	for (GLsizei i = 0; i < n; ++i) {
		out[i] = new_fbo();
	}
}

static void glDeleteFramebuffers_(GLsizei n, const GLuint* names) {
	for (GLsizei i = 0; i < n; ++i) {
		GLuint const f = names[i];
		if (f && f < (GLuint) V.fboCount && V.fbo[f].used) {
			if (V.passOpen && V.passFbo == f) {
				passEnd();
			}
			V.fbo[f].used = false;
			V.st.drawFbo = V.st.drawFbo == f ? 0 : V.st.drawFbo;
			V.st.readFbo = V.st.readFbo == f ? 0 : V.st.readFbo;
		}
	}
}

static void glBindFramebuffer_(GLenum target, GLuint f) {
	if (target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER) {
		V.st.drawFbo = f;
	}
	if (target == GL_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER) {
		V.st.readFbo = f;
	}
}

static struct Fbo* targetFbo(GLenum target) {
	return fboGet(target == GL_READ_FRAMEBUFFER ? V.st.readFbo : V.st.drawFbo);
}

static void glFramebufferTexture2D_(GLenum target, GLenum attachment, GLenum textarget, GLuint t, GLint level) {
	(void) textarget, (void) level;
	struct Fbo* f = targetFbo(target);
	if (!f) {
		return;
	}
	if (V.passOpen && &V.fbo[V.passFbo] == f) {
		passEnd();
	}
	if (attachment == GL_COLOR_ATTACHMENT0) {
		f->color = t, f->colorIsRb = false;
	}
}

static void glFramebufferRenderbuffer_(GLenum target, GLenum attachment, GLenum rbtarget, GLuint r) {
	(void) rbtarget;
	struct Fbo* f = targetFbo(target);
	if (!f) {
		return;
	}
	if (V.passOpen && &V.fbo[V.passFbo] == f) {
		passEnd();
	}
	if (attachment == GL_COLOR_ATTACHMENT0) {
		f->color = r, f->colorIsRb = true;
	} else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT || attachment == GL_DEPTH_ATTACHMENT || attachment == GL_STENCIL_ATTACHMENT) {
		f->depthStencil = r;
	}
}

static GLenum glCheckFramebufferStatus_(GLenum target) {
	struct Fbo* f = targetFbo(target);
	return f && (fboColor(f) || fboDepth(f)) ? GL_FRAMEBUFFER_COMPLETE : GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
}

static void glGenBuffers_(GLsizei n, GLuint* out) {
	for (GLsizei i = 0; i < n; ++i) {
		out[i] = new_buf();
	}
}

static void glDeleteBuffers_(GLsizei n, const GLuint* names) {
	for (GLsizei i = 0; i < n; ++i) {
		GLuint const b = names[i];
		if (b && b < (GLuint) V.bufCount && V.buf[b].used) {
			if (V.buf[b].buffer) {
				garbage(GARBAGE_BUFFER, (uint64_t) (uintptr_t) V.buf[b].buffer, (uint64_t) (uintptr_t) V.buf[b].memory);
			}
			memset(&V.buf[b], 0, sizeof(V.buf[b]));
		}
	}
}

static GLuint* bufferBinding(GLenum target) {
	return target == GL_PIXEL_PACK_BUFFER ? &V.st.packBuffer : target == GL_PIXEL_UNPACK_BUFFER ? &V.st.unpackBuffer : &V.st.arrayBuffer;
}

static void glBindBuffer_(GLenum target, GLuint b) {
	*bufferBinding(target) = b;
}

static void glBufferData_(GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
	(void) usage;
	GLuint const name = *bufferBinding(target);
	if (!name || name >= (GLuint) V.bufCount || !V.buf[name].used) {
		return;
	}
	struct Buf* b = &V.buf[name];
	if (b->buffer) {
		garbage(GARBAGE_BUFFER, (uint64_t) (uintptr_t) b->buffer, (uint64_t) (uintptr_t) b->memory);
		b->buffer = VK_NULL_HANDLE;
	}
	if (!bufferMake(b, (size_t) size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	                target == GL_PIXEL_PACK_BUFFER)) {
		return;
	}
	b->serial = 0;
	if (data) {
		memcpy(b->map, data, (size_t) size);
	}
}

static void* glMapBufferRange_(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
	(void) length;
	GLuint const name = *bufferBinding(target);
	if (!name || name >= (GLuint) V.bufCount || !V.buf[name].map) {
		return NULL;
	}
	struct Buf* b = &V.buf[name];
	if (!(access & 0x20) && b->serial) {
		waitSerial(b->serial);
	}
	return b->map + offset;
}

static GLboolean glUnmapBuffer_(GLenum target) {
	(void) target;
	return 1;
}

static void glGenVertexArrays_(GLsizei n, GLuint* out) {
	for (GLsizei i = 0; i < n; ++i) {
		out[i] = new_vao();
	}
}

static void glDeleteVertexArrays_(GLsizei n, const GLuint* names) {
	for (GLsizei i = 0; i < n; ++i) {
		if (names[i] && names[i] < (GLuint) V.vaoCount) {
			V.vao[names[i]].used = false;
		}
	}
}

static void glBindVertexArray_(GLuint a) {
	V.st.vao = a;
}

static struct Attrib* attrib(GLuint index) {
	if (index >= ATTRIBS) {
		return NULL;
	}
	if (!V.st.vao) {
		if (!V.vaoCount) {
			GLuint const a = new_vao();
			(void) a;
		}
		return V.vaoCount ? &V.vao[0].attrib[index] : NULL;
	}
	return V.st.vao < (GLuint) V.vaoCount ? &V.vao[V.st.vao].attrib[index] : NULL;
}

static void glVertexAttribPointer_(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer) {
	(void) normalized;
	struct Attrib* a = attrib(index);
	if (a) {
		a->buffer = V.st.arrayBuffer, a->size = size, a->type = type, a->stride = stride, a->offset = (size_t) (uintptr_t) pointer, a->integer = false;
	}
}

static void glVertexAttribIPointer_(GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer) {
	struct Attrib* a = attrib(index);
	if (a) {
		a->buffer = V.st.arrayBuffer, a->size = size, a->type = type, a->stride = stride, a->offset = (size_t) (uintptr_t) pointer, a->integer = true;
	}
}

static void glEnableVertexAttribArray_(GLuint index) {
	struct Attrib* a = attrib(index);
	if (a) {
		a->enabled = true;
	}
}

// ---- GL: shaders and programs

static GLuint glCreateShader_(GLenum type) {
	GLuint const s = new_shader();
	if (s) {
		V.shader[s].type = type;
	}
	return s;
}

static void glShaderSource_(GLuint s, GLsizei count, const char* const* strings, const GLint* lengths) {
	if (!s || s >= (GLuint) V.shaderCount) {
		return;
	}
	struct Str text = { 0 };
	for (GLsizei i = 0; i < count; ++i) {
		put(&text, strings[i], lengths && lengths[i] >= 0 ? (size_t) lengths[i] : strlen(strings[i]));
	}
	free(V.shader[s].source);
	V.shader[s].source = text.p;
}

static void glCompileShader_(GLuint s) {
	(void) s;
}

static void glGetShaderiv_(GLuint s, GLenum pname, GLint* out) {
	(void) s;
	*out = pname == GL_COMPILE_STATUS || pname == GL_INFO_LOG_LENGTH ? 1 : 0;
}

static void glGetShaderInfoLog_(GLuint s, GLsizei max, GLsizei* length, char* log) {
	(void) s;
	if (max > 0) {
		log[0] = 0;
	}
	if (length) {
		*length = 0;
	}
}

static void glDeleteShader_(GLuint s) {
	if (s && s < (GLuint) V.shaderCount && V.shader[s].used) {
		free(V.shader[s].source);
		memset(&V.shader[s], 0, sizeof(V.shader[s]));
	}
}

static GLuint glCreateProgram_(void) {
	return new_prog();
}

static void glAttachShader_(GLuint program, GLuint s) {
	struct Prog* p = progGet(program);
	if (p && s && s < (GLuint) V.shaderCount) {
		p->shader[V.shader[s].type == GL_FRAGMENT_SHADER ? 1 : 0] = s;
	}
}

static void progRelease(struct Prog* p, GLuint name) {
	if (!p->vertex && !p->layout) {
		return;
	}
	submit(true);
	vkDeviceWaitIdle(V.device);
	for (int i = 0; i < V.pipeCount; ++i) {
		struct PipeKey k;
		memcpy(&k, V.pipes[i].key, sizeof(k));
		if (k.prog == name) {
			vkDestroyPipeline(V.device, V.pipes[i].pipeline, NULL);
			V.pipes[i--] = V.pipes[--V.pipeCount];
		}
	}
	// The lookups made again for the pipelines left.
	memset(V.pipeHash, 0, sizeof(V.pipeHash));
	for (int i = 0; i < V.pipeCount; ++i) {
		struct PipeKey k;
		memcpy(&k, V.pipes[i].key, sizeof(k));
		uint32_t const h = keyHash(&k);
		for (uint32_t j = 0; j < 4096; ++j) {
			if (!V.pipeHash[(h + j) & 4095]) {
				V.pipeHash[(h + j) & 4095] = i + 1;
				break;
			}
		}
	}
	V.lastPipe = VK_NULL_HANDLE;
	if (p->vertex) {
		vkDestroyShaderModule(V.device, p->vertex, NULL);
		vkDestroyShaderModule(V.device, p->fragment, NULL);
	}
	if (p->layout) {
		vkDestroyPipelineLayout(V.device, p->layout, NULL);
		vkDestroyDescriptorSetLayout(V.device, p->setLayout, NULL);
	}
	p->vertex = p->fragment = VK_NULL_HANDLE, p->layout = VK_NULL_HANDLE, p->setLayout = VK_NULL_HANDLE;
}

static void glLinkProgram_(GLuint program) {
	struct Prog* p = progGet(program);
	if (p) {
		progRelease(p, program);
		p->linked = linkProgram(p);
	}
}

static void glGetProgramiv_(GLuint program, GLenum pname, GLint* out) {
	struct Prog* p = progGet(program);
	if (pname == GL_LINK_STATUS) {
		*out = p && p->linked;
	} else if (pname == GL_COMPLETION_STATUS_KHR) {
		*out = 1;
	} else if (pname == GL_INFO_LOG_LENGTH) {
		*out = p && p->log ? (GLint) strlen(p->log) + 1 : 1;
	} else {
		*out = 0;
	}
}

static void glGetProgramInfoLog_(GLuint program, GLsizei max, GLsizei* length, char* log) {
	struct Prog* p = progGet(program);
	if (max <= 0) {
		return;
	}
	snprintf(log, (size_t) max, "%s", p && p->log ? p->log : "");
	if (length) {
		*length = (GLsizei) strlen(log);
	}
}

static void glUseProgram_(GLuint program) {
	V.st.program = program;
}

static void glDeleteProgram_(GLuint program) {
	struct Prog* p = progGet(program);
	if (!p) {
		return;
	}
	progRelease(p, program);
	free(p->shadow);
	free(p->log);
	memset(p, 0, sizeof(*p));
	V.st.program = V.st.program == program ? 0 : V.st.program;
}

static GLint glGetUniformLocation_(GLuint program, const char* name) {
	struct Prog* p = progGet(program);
	if (!p) {
		return -1;
	}
	// A location: the member times LOCATION_ELEMENTS plus the array element named ("uTile[3]"); a
	// sampler's, SAMPLER_LOCATION plus its index.
	char base[48];
	snprintf(base, sizeof(base), "%.*s", (int) strcspn(name, "["), name);
	const char* bracket = strchr(name, '[');
	int const element = bracket ? atoi(bracket + 1) : 0;
	for (int i = 0; i < p->members; ++i) {
		if (!strcmp(p->member[i].name, base)) {
			return element >= 0 && element < (p->member[i].count ? p->member[i].count : 1) ? i * LOCATION_ELEMENTS + element : -1;
		}
	}
	for (int i = 0; i < p->samplers; ++i) {
		if (!strcmp(p->sampler[i], base)) {
			return SAMPLER_LOCATION + i;
		}
	}
	return -1;
}

static void* uniformAt(GLint location, int* stride, int* count) {
	struct Prog* p = progGet(V.st.program);
	if (!p || location < 0 || location >= SAMPLER_LOCATION || location / LOCATION_ELEMENTS >= p->members || !p->shadow) {
		return NULL;
	}
	struct Member* m = &p->member[location / LOCATION_ELEMENTS];
	int const element = location % LOCATION_ELEMENTS;
	*stride = m->size, *count = (m->count ? m->count : 1) - element;
	p->shadowDirty = true;
	return p->shadow + m->offset + element * m->size;
}

static void glUniform1i_(GLint location, GLint v) {
	struct Prog* p = progGet(V.st.program);
	if (p && location >= SAMPLER_LOCATION && location - SAMPLER_LOCATION < p->samplers) {
		if (p->unit[location - SAMPLER_LOCATION] != v) {
			p->unit[location - SAMPLER_LOCATION] = v;
		}
		return;
	}
	int stride, count;
	void* at = uniformAt(location, &stride, &count);
	if (at) {
		memcpy(at, &v, 4);
	}
}

static void glUniform1f_(GLint location, float v) {
	int stride, count;
	void* at = uniformAt(location, &stride, &count);
	if (at) {
		memcpy(at, &v, 4);
	}
}

static void glUniform2f_(GLint location, float a, float b) {
	int stride, count;
	float* at = uniformAt(location, &stride, &count);
	if (at) {
		at[0] = a, at[1] = b;
	}
}

static void glUniform4f_(GLint location, float a, float b, float c, float d) {
	int stride, count;
	float* at = uniformAt(location, &stride, &count);
	if (at) {
		at[0] = a, at[1] = b, at[2] = c, at[3] = d;
	}
}

static void glUniform4iv_(GLint location, GLsizei n, const GLint* v) {
	int stride, count;
	uint8_t* at = uniformAt(location, &stride, &count);
	for (GLsizei i = 0; at && i < n && i < count; ++i) {
		memcpy(at + (size_t) i * (size_t) stride, v + i * 4, 16);
	}
}

// ---- GL: state

static void cap(GLenum c, bool on) {
	if (c == GL_SCISSOR_TEST) {
		V.st.scissorOn = on;
	} else if (c == GL_DEPTH_TEST) {
		V.st.depthTest = on;
	} else if (c == GL_STENCIL_TEST) {
		V.st.stencilTest = on;
	} else if (c == GL_BLEND) {
		V.st.blend = on;
	} else if (c == GL_POLYGON_OFFSET_FILL) {
		V.st.polygonOffset = on;
	}
}

static void glEnable_(GLenum c) {
	cap(c, true);
}

static void glDisable_(GLenum c) {
	cap(c, false);
}

static void glViewport_(GLint x, GLint y, GLsizei w, GLsizei h) {
	V.st.viewport[0] = x, V.st.viewport[1] = y, V.st.viewport[2] = w, V.st.viewport[3] = h;
}

static void glScissor_(GLint x, GLint y, GLsizei w, GLsizei h) {
	V.st.scissor[0] = x, V.st.scissor[1] = y, V.st.scissor[2] = w, V.st.scissor[3] = h;
}

static void glClearColor_(float r, float g, float b, float a) {
	V.st.clearColor[0] = r, V.st.clearColor[1] = g, V.st.clearColor[2] = b, V.st.clearColor[3] = a;
}

static void glClearDepthf_(float d) {
	V.st.clearDepth = d;
}

static void glClearStencil_(GLint s) {
	V.st.clearStencil = s;
}

static void glDepthFunc_(GLenum f) {
	V.st.depthFunc = f;
}

static void glDepthMask_(GLboolean on) {
	V.st.depthMask = on != 0;
}

static void glBlendFuncSeparate_(GLenum sc, GLenum dc, GLenum sa, GLenum da) {
	V.st.blendSrc[0] = sc, V.st.blendDst[0] = dc, V.st.blendSrc[1] = sa, V.st.blendDst[1] = da;
}

static void glBlendFunc_(GLenum s, GLenum d) {
	glBlendFuncSeparate_(s, d, s, d);
}

static void glBlendEquationSeparate_(GLenum c, GLenum a) {
	V.st.blendEq[0] = c, V.st.blendEq[1] = a;
}

static void glBlendColor_(float r, float g, float b, float a) {
	V.st.blendColor[0] = r, V.st.blendColor[1] = g, V.st.blendColor[2] = b, V.st.blendColor[3] = a;
}

static void glPolygonOffset_(float factor, float units) {
	V.st.offsetFactor = factor, V.st.offsetUnits = units;
}

static void glStencilFunc_(GLenum f, GLint ref, GLuint mask) {
	V.st.stencilFunc = f, V.st.stencilRef = ref, V.st.stencilMask = mask;
}

static void glStencilOp_(GLenum fail, GLenum zfail, GLenum zpass) {
	V.st.stencilFail = fail, V.st.stencilZfail = zfail, V.st.stencilZpass = zpass;
}

static void glColorMask_(GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
	V.st.colorMask[0] = r != 0, V.st.colorMask[1] = g != 0, V.st.colorMask[2] = b != 0, V.st.colorMask[3] = a != 0;
}

static const unsigned char* glGetString_(GLenum name) {
	const char* s = name == GL_RENDERER ? V.renderer : name == GL_VERSION ? "OpenGL ES 3.1 (armN64 on Vulkan)"
	              : name == GL_VENDOR ? "armN64"
	              : name == GL_SHADING_LANGUAGE_VERSION ? "OpenGL ES GLSL ES 3.10"
	              : name == GL_EXTENSIONS               ? "GL_NV_texture_barrier"
	                                                    : "";
	return (const unsigned char*) s;
}

static void glGetIntegerv_(GLenum pname, GLint* out) {
	if (pname == GL_MAX_TEXTURE_SIZE || pname == GL_MAX_RENDERBUFFER_SIZE) {
		*out = (GLint) V.limits.maxImageDimension2D;
	} else if (pname == GL_MAX_SAMPLES) {
		*out = 1;
	} else {
		*out = 0;
	}
}

static void glPixelStorei_(GLenum pname, GLint v) {
	(void) pname, (void) v;
}

static void glFlush_(void) {
	submit(false);
}

static void glFinish_(void) {
	submit(true);
}

// What the draws before wrote, for the next to read as its own pixel: within the render pass, pixel
// by pixel.
static void glTextureBarrierNV_(void) {
	if (V.passOpen && !V.ordered) {
		VkMemoryBarrier m = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_INPUT_ATTACHMENT_READ_BIT };
		vkCmdPipelineBarrier(V.sub[V.current].cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		                     VK_DEPENDENCY_BY_REGION_BIT, 1, &m, 0, NULL, 0, NULL);
		++V.pendingBarriers;
	}
}

static void glMemoryBarrier_(GLbitfield bits) {
	(void) bits;
}

static GLenum glGetError_(void) {
	return 0;
}

// ---- GL: reading and copying pixels

static void glReadPixels_(GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, void* out) {
	(void) format, (void) type;
	struct Fbo* f = fboGet(V.st.readFbo);
	struct Img* img = f ? fboColor(f) : NULL;
	if (!img || w <= 0 || h <= 0) {
		return;
	}
	size_t const size = (size_t) w * (size_t) h * 4;
	struct Buf* pack = V.st.packBuffer && V.st.packBuffer < (GLuint) V.bufCount && V.buf[V.st.packBuffer].buffer ? &V.buf[V.st.packBuffer] : NULL;
	if (!pack && V.readBuffer.size < size) {
		if (V.readBuffer.buffer) {
			garbage(GARBAGE_BUFFER, (uint64_t) (uintptr_t) V.readBuffer.buffer, (uint64_t) (uintptr_t) V.readBuffer.memory);
			V.readBuffer.buffer = VK_NULL_HANDLE;
		}
		if (!bufferMake(&V.readBuffer, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true)) {
			return;
		}
	}
	VkCommandBuffer cb = transfer();
	touch(img);
	layoutTo(cb, img, VK_IMAGE_LAYOUT_GENERAL);
	VkBufferImageCopy region = { pack ? (VkDeviceSize) (uintptr_t) out : 0, 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { x, y, 0 },
		                         { (uint32_t) w, (uint32_t) h, 1 } };
	vkCmdCopyImageToBuffer(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, pack ? pack->buffer : V.readBuffer.buffer, 1, &region);
	if (pack) {
		pack->serial = V.sub[V.current].serial;
		return;
	}
	submit(true);
	memcpy(out, V.readBuffer.map, size);
}

static void glBlitFramebuffer_(GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield mask, GLenum filter) {
	struct Fbo* r = fboGet(V.st.readFbo);
	struct Fbo* d = fboGet(V.st.drawFbo);
	if (!r || !d) {
		return;
	}
	if (mask & GL_COLOR_BUFFER_BIT) {
		struct Img* src = fboColor(r);
		struct Img* dst = fboColor(d);
		if (src && dst) {
			VkCommandBuffer cb = transfer();
			touch(src), touch(dst);
			layoutTo(cb, src, VK_IMAGE_LAYOUT_GENERAL);
			layoutTo(cb, dst, VK_IMAGE_LAYOUT_GENERAL);
			VkImageBlit blit = { { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, { { sx0, sy0, 0 }, { sx1, sy1, 1 } }, { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
				                 { { dx0, dy0, 0 }, { dx1, dy1, 1 } } };
			vkCmdBlitImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
			               filter == GL_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
		}
	}
	if (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)) {
		struct Img* src = fboDepth(r);
		struct Img* dst = fboDepth(d);
		int const w = sx1 - sx0, h = sy1 - sy0;
		if (src && dst && w > 0 && h > 0 && w == dx1 - dx0 && h == dy1 - dy0) {
			VkCommandBuffer cb = transfer();
			touch(src), touch(dst);
			VkImageCopy copy = { { src->aspect, 0, 0, 1 }, { sx0, sy0, 0 }, { dst->aspect, 0, 0, 1 }, { dx0, dy0, 0 }, { (uint32_t) w, (uint32_t) h, 1 } };
			vkCmdCopyImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
		}
	}
}

// ---- The frontend's picture

unsigned vkglFrontendFramebuffer(int width, int height) {
	drain();
	if (!V.pictureFbo) {
		V.pictureFbo = new_fbo();
		if (!V.pictureFbo) {
			return 0;
		}
	}
	uint32_t const mask = V.external && V.frontend.syncIndexMask ? V.frontend.syncIndexMask(V.frontend.handle) : 1;
	int count = 0;
	for (uint32_t m = mask; m && count < 4; m >>= 1) {
		count += (int) (m & 1);
	}
	count = count ? count : 1;
	V.pictureIndex = V.external && V.frontend.syncIndex ? (int) (V.frontend.syncIndex(V.frontend.handle) % (uint32_t) count) : 0;
	struct Img* img = &V.picture[V.pictureIndex];
	if (!img->image || img->width != width || img->height != height) {
		passEnd();
		imageFree(img);
		if (!imageMake(img, VK_FORMAT_R8G8B8A8_UNORM, width, height)) {
			return 0;
		}
		imageReady(img);
	}
	// The framebuffer's color is this frame's picture (picture slot, not a texture).
	V.fbo[V.pictureFbo].color = 0;
	V.fbo[V.pictureFbo].colorIsRb = false;
	return V.pictureFbo;
}

bool vkglPicture(VkImage* image, VkImageView* view, VkImageLayout* layout, VkImageViewCreateInfo* info) {
	drain();
	struct Img* img = &V.picture[V.pictureIndex];
	if (!img->image) {
		return false;
	}
	VkCommandBuffer cb = transfer();
	touch(img);
	layoutTo(cb, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	submit(false);
	*image = img->image, *view = img->view, *layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	*info = (VkImageViewCreateInfo) { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img->image, VK_IMAGE_VIEW_TYPE_2D, img->format,
		                              { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY },
		                              { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
	return true;
}

// ---- A thread of its own
//
// The calls that return nothing go to a ring of records that a thread of its own runs (and records
// in Vulkan) while the emulator goes on; the others wait until it has run every record before them
// and run here. Mapping the vertex buffer to write without waiting is done here as well, from the
// buffer bindings copied as they're queued (the buffer itself only changes with calls that wait).

// Records are handed over QUEUE_BATCH at a time (and all before a wait), so the thread sleeps between
// batches instead of waiting for each.
enum { QUEUE_BYTES = 16 << 20, QUEUE_BATCH = 128, GL_MAP_UNSYNCHRONIZED_BIT = 0x20 };

struct Call {
	void (*run)(const struct Call*);
	uint32_t size;
	int32_t i[10];
	float f[4];
	intptr_t p;
};

static struct {
	pthread_t thread;
	bool running;
	uint8_t* ring;
	// Bytes queued (head, the queuer's), run (tail) and handed over (published): ring positions.
	size_t head;
	int unpublished;
	_Atomic size_t tail, published;
	_Atomic bool sleeping, waiting, stop;
	pthread_mutex_t lock;
	pthread_cond_t wake, idle;
	GLuint buffers[3];
} Q;

static void relax(void) {
#if defined(__x86_64__)
	__builtin_ia32_pause();
#elif defined(__aarch64__)
	__asm__ volatile("yield");
#endif
}

static void publish(void) {
	Q.unpublished = 0;
	atomic_store(&Q.published, Q.head);
	if (atomic_load(&Q.sleeping)) {
		pthread_mutex_lock(&Q.lock);
		pthread_cond_signal(&Q.wake);
		pthread_mutex_unlock(&Q.lock);
	}
}

// Every record queued run.
static void drain(void) {
	if (!Q.running) {
		return;
	}
	publish();
	for (int spin = 0; spin < 4000; ++spin) {
		if (atomic_load(&Q.tail) == Q.head) {
			return;
		}
		relax();
	}
	pthread_mutex_lock(&Q.lock);
	atomic_store(&Q.waiting, true);
	while (atomic_load(&Q.tail) != Q.head) {
		pthread_cond_wait(&Q.idle, &Q.lock);
	}
	atomic_store(&Q.waiting, false);
	pthread_mutex_unlock(&Q.lock);
}

// A record for run with data bytes after it (an end-of-ring mark first when it doesn't fit there).
static struct Call* callNew(void (*run)(const struct Call*), size_t data) {
	size_t const size = (sizeof(struct Call) + data + 7) & ~(size_t) 7;
	size_t const room = QUEUE_BYTES - Q.head % QUEUE_BYTES;
	if (QUEUE_BYTES - (Q.head - atomic_load(&Q.tail)) < size + (room < size ? room : 0)) {
		drain();
	}
	if (room < size) {
		*(void**) (Q.ring + Q.head % QUEUE_BYTES) = NULL;
		Q.head += room;
	}
	struct Call* c = (struct Call*) (Q.ring + Q.head % QUEUE_BYTES);
	c->run = run, c->size = (uint32_t) size;
	return c;
}

static void callEnd(struct Call* c) {
	Q.head += c->size;
	if (++Q.unpublished == QUEUE_BATCH) {
		publish();
	}
}

static void* queueRun(void* unused) {
	(void) unused;
	size_t tail = 0;
	while (!atomic_load(&Q.stop) || tail != atomic_load(&Q.published)) {
		if (tail == atomic_load(&Q.published)) {
			int spin = 0;
			while (tail == atomic_load(&Q.published) && spin++ < 200) {
				relax();
			}
			pthread_mutex_lock(&Q.lock);
			atomic_store(&Q.sleeping, true);
			while (tail == atomic_load(&Q.published) && !atomic_load(&Q.stop)) {
				pthread_cond_wait(&Q.wake, &Q.lock);
			}
			atomic_store(&Q.sleeping, false);
			pthread_mutex_unlock(&Q.lock);
			continue;
		}
		struct Call* c = (struct Call*) (Q.ring + tail % QUEUE_BYTES);
		if (c->run) {
			c->run(c);
			tail += c->size;
		} else {
			tail += QUEUE_BYTES - tail % QUEUE_BYTES;
		}
		atomic_store(&Q.tail, tail);
		if (atomic_load(&Q.waiting) && tail == atomic_load(&Q.published)) {
			pthread_mutex_lock(&Q.lock);
			pthread_cond_signal(&Q.idle);
			pthread_mutex_unlock(&Q.lock);
		}
	}
	return NULL;
}

static void queueStart(void) {
	Q.ring = malloc(QUEUE_BYTES);
	Q.head = 0, Q.unpublished = 0;
	atomic_store(&Q.tail, 0);
	atomic_store(&Q.published, 0);
	atomic_store(&Q.sleeping, false);
	atomic_store(&Q.waiting, false);
	atomic_store(&Q.stop, false);
	memset(Q.buffers, 0, sizeof(Q.buffers));
	pthread_mutex_init(&Q.lock, NULL);
	pthread_cond_init(&Q.wake, NULL);
	pthread_cond_init(&Q.idle, NULL);
	Q.running = Q.ring && pthread_create(&Q.thread, NULL, queueRun, NULL) == 0;
	if (!Q.running) {
		free(Q.ring);
		Q.ring = NULL;
	}
}

static void queueStop(void) {
	if (!Q.running) {
		return;
	}
	drain();
	pthread_mutex_lock(&Q.lock);
	atomic_store(&Q.stop, true);
	pthread_cond_signal(&Q.wake);
	pthread_mutex_unlock(&Q.lock);
	pthread_join(Q.thread, NULL);
	Q.running = false;
	pthread_cond_destroy(&Q.wake);
	pthread_cond_destroy(&Q.idle);
	pthread_mutex_destroy(&Q.lock);
	free(Q.ring);
	Q.ring = NULL;
}

// A queued call: store puts its arguments in the record c, call runs it from there. Without the
// thread it runs at once from a record here.
#define QUEUED(name, params, store, call)                                                   \
	static void run_##name(const struct Call* c) {                                          \
		(void) c;                                                                           \
		call;                                                                               \
	}                                                                                       \
	static void q_##name params {                                                           \
		struct Call local, *c = Q.running ? callNew(run_##name, 0) : &local;                \
		store;                                                                              \
		if (c == &local) {                                                                  \
			run_##name(c);                                                                  \
		} else {                                                                            \
			callEnd(c);                                                                     \
		}                                                                                   \
	}

// A call that waits for the thread, then runs here.
#define WAITS(type, name, params, args) \
	static type w_##name params {       \
		drain();                        \
		return gl##name##_ args;        \
	}
#define WAITS_VOID(name, params, args) \
	static void w_##name params {      \
		drain();                       \
		gl##name##_ args;              \
	}

QUEUED(BindFramebuffer, (GLenum t, GLuint f), (c->i[0] = (int32_t) t, c->i[1] = (int32_t) f), glBindFramebuffer_((GLenum) c->i[0], (GLuint) c->i[1]))
QUEUED(BlitFramebuffer, (GLint a, GLint b, GLint d, GLint e, GLint g, GLint h, GLint j, GLint k, GLbitfield mask, GLenum filter),
       (c->i[0] = a, c->i[1] = b, c->i[2] = d, c->i[3] = e, c->i[4] = g, c->i[5] = h, c->i[6] = j, c->i[7] = k, c->i[8] = (int32_t) mask, c->i[9] = (int32_t) filter),
       glBlitFramebuffer_(c->i[0], c->i[1], c->i[2], c->i[3], c->i[4], c->i[5], c->i[6], c->i[7], (GLbitfield) c->i[8], (GLenum) c->i[9]))
QUEUED(FramebufferTexture2D, (GLenum t, GLenum a, GLenum tt, GLuint x, GLint l),
       (c->i[0] = (int32_t) t, c->i[1] = (int32_t) a, c->i[2] = (int32_t) tt, c->i[3] = (int32_t) x, c->i[4] = l),
       glFramebufferTexture2D_((GLenum) c->i[0], (GLenum) c->i[1], (GLenum) c->i[2], (GLuint) c->i[3], c->i[4]))
QUEUED(FramebufferRenderbuffer, (GLenum t, GLenum a, GLenum rt, GLuint r),
       (c->i[0] = (int32_t) t, c->i[1] = (int32_t) a, c->i[2] = (int32_t) rt, c->i[3] = (int32_t) r),
       glFramebufferRenderbuffer_((GLenum) c->i[0], (GLenum) c->i[1], (GLenum) c->i[2], (GLuint) c->i[3]))
QUEUED(BindRenderbuffer, (GLenum t, GLuint r), (c->i[0] = (int32_t) t, c->i[1] = (int32_t) r), glBindRenderbuffer_((GLenum) c->i[0], (GLuint) c->i[1]))
QUEUED(RenderbufferStorage, (GLenum t, GLenum f, GLsizei w, GLsizei h), (c->i[0] = (int32_t) t, c->i[1] = (int32_t) f, c->i[2] = w, c->i[3] = h),
       glRenderbufferStorage_((GLenum) c->i[0], (GLenum) c->i[1], c->i[2], c->i[3]))
QUEUED(RenderbufferStorageMultisample, (GLenum t, GLsizei s, GLenum f, GLsizei w, GLsizei h),
       (c->i[0] = (int32_t) t, c->i[1] = s, c->i[2] = (int32_t) f, c->i[3] = w, c->i[4] = h),
       glRenderbufferStorageMultisample_((GLenum) c->i[0], c->i[1], (GLenum) c->i[2], c->i[3], c->i[4]))
QUEUED(BindTexture, (GLenum t, GLuint x), (c->i[0] = (int32_t) t, c->i[1] = (int32_t) x), glBindTexture_((GLenum) c->i[0], (GLuint) c->i[1]))
QUEUED(ActiveTexture, (GLenum u), c->i[0] = (int32_t) u, glActiveTexture_((GLenum) c->i[0]))
QUEUED(TexParameteri, (GLenum t, GLenum n, GLint v), (c->i[0] = (int32_t) t, c->i[1] = (int32_t) n, c->i[2] = v),
       glTexParameteri_((GLenum) c->i[0], (GLenum) c->i[1], c->i[2]))
QUEUED(Viewport, (GLint x, GLint y, GLsizei w, GLsizei h), (c->i[0] = x, c->i[1] = y, c->i[2] = w, c->i[3] = h), glViewport_(c->i[0], c->i[1], c->i[2], c->i[3]))
QUEUED(Scissor, (GLint x, GLint y, GLsizei w, GLsizei h), (c->i[0] = x, c->i[1] = y, c->i[2] = w, c->i[3] = h), glScissor_(c->i[0], c->i[1], c->i[2], c->i[3]))
QUEUED(Enable, (GLenum x), c->i[0] = (int32_t) x, glEnable_((GLenum) c->i[0]))
QUEUED(Disable, (GLenum x), c->i[0] = (int32_t) x, glDisable_((GLenum) c->i[0]))
QUEUED(ClearColor, (float r, float g, float b, float a), (c->f[0] = r, c->f[1] = g, c->f[2] = b, c->f[3] = a), glClearColor_(c->f[0], c->f[1], c->f[2], c->f[3]))
QUEUED(ClearDepthf, (float d), c->f[0] = d, glClearDepthf_(c->f[0]))
QUEUED(ClearStencil, (GLint s), c->i[0] = s, glClearStencil_(c->i[0]))
QUEUED(Clear, (GLbitfield m), c->i[0] = (int32_t) m, glClear_((GLbitfield) c->i[0]))
QUEUED(DepthFunc, (GLenum x), c->i[0] = (int32_t) x, glDepthFunc_((GLenum) c->i[0]))
QUEUED(DepthMask, (GLboolean x), c->i[0] = x, glDepthMask_((GLboolean) c->i[0]))
QUEUED(BlendFunc, (GLenum s, GLenum d), (c->i[0] = (int32_t) s, c->i[1] = (int32_t) d), glBlendFunc_((GLenum) c->i[0], (GLenum) c->i[1]))
QUEUED(BlendFuncSeparate, (GLenum a, GLenum b, GLenum d, GLenum e), (c->i[0] = (int32_t) a, c->i[1] = (int32_t) b, c->i[2] = (int32_t) d, c->i[3] = (int32_t) e),
       glBlendFuncSeparate_((GLenum) c->i[0], (GLenum) c->i[1], (GLenum) c->i[2], (GLenum) c->i[3]))
QUEUED(BlendEquationSeparate, (GLenum a, GLenum b), (c->i[0] = (int32_t) a, c->i[1] = (int32_t) b), glBlendEquationSeparate_((GLenum) c->i[0], (GLenum) c->i[1]))
QUEUED(BlendColor, (float r, float g, float b, float a), (c->f[0] = r, c->f[1] = g, c->f[2] = b, c->f[3] = a), glBlendColor_(c->f[0], c->f[1], c->f[2], c->f[3]))
QUEUED(PolygonOffset, (float a, float b), (c->f[0] = a, c->f[1] = b), glPolygonOffset_(c->f[0], c->f[1]))
QUEUED(StencilFunc, (GLenum x, GLint r, GLuint m), (c->i[0] = (int32_t) x, c->i[1] = r, c->i[2] = (int32_t) m), glStencilFunc_((GLenum) c->i[0], c->i[1], (GLuint) c->i[2]))
QUEUED(StencilOp, (GLenum a, GLenum b, GLenum d), (c->i[0] = (int32_t) a, c->i[1] = (int32_t) b, c->i[2] = (int32_t) d),
       glStencilOp_((GLenum) c->i[0], (GLenum) c->i[1], (GLenum) c->i[2]))
QUEUED(ColorMask, (GLboolean r, GLboolean g, GLboolean b, GLboolean a), (c->i[0] = r, c->i[1] = g, c->i[2] = b, c->i[3] = a),
       glColorMask_((GLboolean) c->i[0], (GLboolean) c->i[1], (GLboolean) c->i[2], (GLboolean) c->i[3]))
QUEUED(PixelStorei, (GLenum n, GLint v), (c->i[0] = (int32_t) n, c->i[1] = v), glPixelStorei_((GLenum) c->i[0], c->i[1]))
QUEUED(UseProgram, (GLuint x), c->i[0] = (int32_t) x, glUseProgram_((GLuint) c->i[0]))
QUEUED(Uniform1i, (GLint l, GLint v), (c->i[0] = l, c->i[1] = v), glUniform1i_(c->i[0], c->i[1]))
QUEUED(Uniform1f, (GLint l, float v), (c->i[0] = l, c->f[0] = v), glUniform1f_(c->i[0], c->f[0]))
QUEUED(Uniform2f, (GLint l, float a, float b), (c->i[0] = l, c->f[0] = a, c->f[1] = b), glUniform2f_(c->i[0], c->f[0], c->f[1]))
QUEUED(Uniform4f, (GLint l, float a, float b, float d, float e), (c->i[0] = l, c->f[0] = a, c->f[1] = b, c->f[2] = d, c->f[3] = e),
       glUniform4f_(c->i[0], c->f[0], c->f[1], c->f[2], c->f[3]))
QUEUED(BindVertexArray, (GLuint x), c->i[0] = (int32_t) x, glBindVertexArray_((GLuint) c->i[0]))
QUEUED(VertexAttribPointer, (GLuint x, GLint s, GLenum t, GLboolean n, GLsizei st, const void* p),
       (c->i[0] = (int32_t) x, c->i[1] = s, c->i[2] = (int32_t) t, c->i[3] = n, c->i[4] = st, c->p = (intptr_t) p),
       glVertexAttribPointer_((GLuint) c->i[0], c->i[1], (GLenum) c->i[2], (GLboolean) c->i[3], c->i[4], (const void*) c->p))
QUEUED(VertexAttribIPointer, (GLuint x, GLint s, GLenum t, GLsizei st, const void* p),
       (c->i[0] = (int32_t) x, c->i[1] = s, c->i[2] = (int32_t) t, c->i[3] = st, c->p = (intptr_t) p),
       glVertexAttribIPointer_((GLuint) c->i[0], c->i[1], (GLenum) c->i[2], c->i[3], (const void*) c->p))
QUEUED(EnableVertexAttribArray, (GLuint x), c->i[0] = (int32_t) x, glEnableVertexAttribArray_((GLuint) c->i[0]))
QUEUED(DrawArrays, (GLenum m, GLint first, GLsizei n), (c->i[0] = (int32_t) m, c->i[1] = first, c->i[2] = n), glDrawArrays_((GLenum) c->i[0], c->i[1], c->i[2]))
QUEUED(Flush, (void), (void) c, glFlush_())
QUEUED(MemoryBarrier, (GLbitfield b), c->i[0] = (int32_t) b, glMemoryBarrier_((GLbitfield) c->i[0]))
QUEUED(TextureBarrierNV, (void), (void) c, glTextureBarrierNV_())

QUEUED(BindBufferRun, (GLenum t, GLuint b), (c->i[0] = (int32_t) t, c->i[1] = (int32_t) b), glBindBuffer_((GLenum) c->i[0], (GLuint) c->i[1]))
static void q_BindBuffer(GLenum target, GLuint b) {
	Q.buffers[target == GL_PIXEL_PACK_BUFFER ? 1 : target == GL_PIXEL_UNPACK_BUFFER ? 2 : 0] = b;
	q_BindBufferRun(target, b);
}

// Pixels (from memory, not a buffer) go in the record; very many wait and go at once.
static void run_TexImage2D(const struct Call* c) {
	glTexImage2D_((GLenum) c->i[0], c->i[1], c->i[2], c->i[3], c->i[4], c->i[5], (GLenum) c->i[6], (GLenum) c->i[7], c->i[8] ? (const void*) (c + 1) : (const void*) c->p);
}

static void q_TexImage2D(GLenum target, GLint level, GLint internal, GLsizei w, GLsizei h, GLint border, GLenum format, GLenum type, const void* data) {
	size_t const bytes = data && !Q.buffers[2] && w > 0 && h > 0 ? (size_t) w * (size_t) h * 4 : 0;
	if (!Q.running || bytes > QUEUE_BYTES / 4) {
		drain();
		glTexImage2D_(target, level, internal, w, h, border, format, type, data);
		return;
	}
	struct Call* c = callNew(run_TexImage2D, bytes);
	c->i[0] = (int32_t) target, c->i[1] = level, c->i[2] = internal, c->i[3] = w, c->i[4] = h, c->i[5] = border;
	c->i[6] = (int32_t) format, c->i[7] = (int32_t) type, c->i[8] = bytes != 0, c->p = (intptr_t) data;
	if (bytes) {
		memcpy(c + 1, data, bytes);
	}
	callEnd(c);
}

static void run_TexSubImage2D(const struct Call* c) {
	glTexSubImage2D_((GLenum) c->i[0], c->i[1], c->i[2], c->i[3], c->i[4], c->i[5], (GLenum) c->i[6], (GLenum) c->i[7], c->i[8] ? (const void*) (c + 1) : (const void*) c->p);
}

static void q_TexSubImage2D(GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum format, GLenum type, const void* data) {
	size_t const bytes = data && !Q.buffers[2] && w > 0 && h > 0 ? (size_t) w * (size_t) h * 4 : 0;
	if (!Q.running || bytes > QUEUE_BYTES / 4) {
		drain();
		glTexSubImage2D_(target, level, x, y, w, h, format, type, data);
		return;
	}
	struct Call* c = callNew(run_TexSubImage2D, bytes);
	c->i[0] = (int32_t) target, c->i[1] = level, c->i[2] = x, c->i[3] = y, c->i[4] = w, c->i[5] = h;
	c->i[6] = (int32_t) format, c->i[7] = (int32_t) type, c->i[8] = bytes != 0, c->p = (intptr_t) data;
	if (bytes) {
		memcpy(c + 1, data, bytes);
	}
	callEnd(c);
}

static void run_Uniform4iv(const struct Call* c) {
	glUniform4iv_(c->i[0], c->i[1], (const GLint*) (c + 1));
}

static void q_Uniform4iv(GLint location, GLsizei n, const GLint* v) {
	size_t const bytes = n > 0 ? (size_t) n * 16 : 0;
	if (!Q.running || bytes > QUEUE_BYTES / 4) {
		drain();
		glUniform4iv_(location, n, v);
		return;
	}
	struct Call* c = callNew(run_Uniform4iv, bytes);
	c->i[0] = location, c->i[1] = n;
	if (bytes) {
		memcpy(c + 1, v, bytes);
	}
	callEnd(c);
}

// Writing without waiting: the buffer's memory at once; else after the thread's done.
static void* q_MapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
	GLuint const name = Q.buffers[target == GL_PIXEL_PACK_BUFFER ? 1 : target == GL_PIXEL_UNPACK_BUFFER ? 2 : 0];
	if (Q.running && (access & GL_MAP_UNSYNCHRONIZED_BIT) && !(access & GL_MAP_READ_BIT)) {
		return name && name < (GLuint) V.bufCount && V.buf[name].map ? V.buf[name].map + offset : NULL;
	}
	drain();
	return glMapBufferRange_(target, offset, length, access);
}

WAITS_VOID(GenFramebuffers, (GLsizei n, GLuint* out), (n, out))
WAITS_VOID(DeleteFramebuffers, (GLsizei n, const GLuint* names), (n, names))
WAITS(GLenum, CheckFramebufferStatus, (GLenum t), (t))
WAITS_VOID(GenRenderbuffers, (GLsizei n, GLuint* out), (n, out))
WAITS_VOID(DeleteRenderbuffers, (GLsizei n, const GLuint* names), (n, names))
WAITS_VOID(GenTextures, (GLsizei n, GLuint* out), (n, out))
WAITS_VOID(DeleteTextures, (GLsizei n, const GLuint* names), (n, names))
WAITS(const unsigned char*, GetString, (GLenum n), (n))
WAITS_VOID(GetIntegerv, (GLenum n, GLint* out), (n, out))
WAITS_VOID(ReadPixels, (GLint x, GLint y, GLsizei w, GLsizei h, GLenum f, GLenum t, void* out), (x, y, w, h, f, t, out))
WAITS(GLuint, CreateShader, (GLenum t), (t))
WAITS_VOID(ShaderSource, (GLuint s, GLsizei n, const char* const* strings, const GLint* lengths), (s, n, strings, lengths))
WAITS_VOID(CompileShader, (GLuint s), (s))
WAITS_VOID(GetShaderiv, (GLuint s, GLenum n, GLint* out), (s, n, out))
WAITS_VOID(GetShaderInfoLog, (GLuint s, GLsizei max, GLsizei* length, char* log), (s, max, length, log))
WAITS_VOID(DeleteShader, (GLuint s), (s))
WAITS(GLuint, CreateProgram, (void), ())
WAITS_VOID(AttachShader, (GLuint p, GLuint s), (p, s))
WAITS_VOID(LinkProgram, (GLuint p), (p))
WAITS_VOID(GetProgramiv, (GLuint p, GLenum n, GLint* out), (p, n, out))
WAITS_VOID(GetProgramInfoLog, (GLuint p, GLsizei max, GLsizei* length, char* log), (p, max, length, log))
WAITS_VOID(DeleteProgram, (GLuint p), (p))
WAITS(GLint, GetUniformLocation, (GLuint p, const char* name), (p, name))
WAITS_VOID(GenVertexArrays, (GLsizei n, GLuint* out), (n, out))
WAITS_VOID(DeleteVertexArrays, (GLsizei n, const GLuint* names), (n, names))
WAITS_VOID(GenBuffers, (GLsizei n, GLuint* out), (n, out))
WAITS_VOID(DeleteBuffers, (GLsizei n, const GLuint* names), (n, names))
WAITS_VOID(BufferData, (GLenum t, GLsizeiptr size, const void* data, GLenum usage), (t, size, data, usage))
WAITS_VOID(Finish, (void), ())
WAITS(GLenum, GetError, (void), ())

// ---- GL's functions by name

void* vkglProc(const char* name) {
	static const struct {
		const char* name;
		void* f;
	} table[] = {
#define Q_(n) { "gl" #n, (void*) q_##n }
#define W_(n) { "gl" #n, (void*) w_##n }
		W_(GenFramebuffers), Q_(BindFramebuffer), Q_(BlitFramebuffer), Q_(FramebufferTexture2D), Q_(FramebufferRenderbuffer),
		W_(CheckFramebufferStatus), W_(DeleteFramebuffers), W_(GenRenderbuffers), W_(DeleteRenderbuffers), Q_(BindRenderbuffer),
		Q_(RenderbufferStorage), Q_(RenderbufferStorageMultisample), W_(GenTextures), W_(DeleteTextures), Q_(BindTexture), Q_(ActiveTexture),
		Q_(TexImage2D), Q_(TexSubImage2D), Q_(TexParameteri), Q_(Viewport), Q_(Scissor), Q_(Enable), Q_(Disable), Q_(ClearColor),
		Q_(ClearDepthf), Q_(Clear), Q_(DepthFunc), Q_(DepthMask), Q_(BlendFunc), Q_(BlendFuncSeparate), Q_(BlendEquationSeparate),
		Q_(BlendColor), Q_(PolygonOffset), Q_(StencilFunc), Q_(StencilOp), Q_(ColorMask), Q_(ClearStencil), W_(GetString), W_(GetIntegerv),
		W_(ReadPixels), Q_(PixelStorei), W_(CreateShader), W_(ShaderSource), W_(CompileShader), W_(GetShaderiv), W_(GetShaderInfoLog),
		W_(DeleteShader), W_(CreateProgram), W_(AttachShader), W_(LinkProgram), W_(GetProgramiv), W_(GetProgramInfoLog), Q_(UseProgram),
		W_(DeleteProgram), W_(GetUniformLocation), Q_(Uniform1i), Q_(Uniform1f), Q_(Uniform2f), Q_(Uniform4f), Q_(Uniform4iv),
		W_(GenVertexArrays), W_(DeleteVertexArrays), Q_(BindVertexArray), W_(GenBuffers), W_(DeleteBuffers), Q_(BindBuffer), W_(BufferData),
		Q_(MapBufferRange), { "glUnmapBuffer", (void*) glUnmapBuffer_ }, Q_(VertexAttribPointer), Q_(VertexAttribIPointer), Q_(EnableVertexAttribArray), Q_(DrawArrays),
		Q_(Flush), W_(Finish), Q_(MemoryBarrier), W_(GetError), Q_(TextureBarrierNV),
#undef Q_
#undef W_
	};
	for (size_t i = 0; i < sizeof(table) / sizeof(*table); ++i) {
		if (!strcmp(table[i].name, name)) {
			return table[i].f;
		}
	}
	return NULL;
}
