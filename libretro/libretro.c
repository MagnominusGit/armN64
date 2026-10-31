// armN64 as a libretro core: a video field each run, the VI's picture, the AI's samples at
// 44.1 kHz, controller 1, and the cartridge's save memory. The GPU renderer draws in a GL context the
// frontend shares with its own when it draws with GL (the picture is then put in its framebuffer),
// else in one of the core's own, made current only while the core runs.
#include "gles.h"
#include "n64.h"

#include "libretro.h"
#include "options.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

enum {
	OUT_RATE = 44100,
	// The save file as mupen64plus-next lays it out, so saves go between the two: the EEPROM, the four
	// Controller Paks, the SRAM, then the flash (these two in little-endian words).
	FILE_PAKS = 0x800,
	FILE_SRAM = FILE_PAKS + 4 * 0x8000,
	FILE_FLASH = FILE_SRAM + 0x8000,
	SAVE_SIZE = FILE_FLASH + 0x20000,
	// A second of the fastest AI rate, left and right, and room for it from rates down to 22 kHz.
	CAPTURE_SAMPLES = 2 * 49152,
	OUT_FRAMES = 2 * 49152,
};

static retro_environment_t environment;
static retro_video_refresh_t videoRefresh;
static retro_audio_sample_batch_t audioBatch;
static retro_input_poll_t inputPoll;
static retro_input_state_t inputState;
static retro_log_printf_t logPrintf;
static retro_set_rumble_state_t setRumble;

static struct N64* machine;
// The game as it was loaded (and the 64DD's IPL for a disk), to start again on a reset.
static uint8_t* game;
static size_t gameSize;
static uint8_t* ipl;
static size_t iplSize;
static bool disk;
static bool pal;
static uint32_t picture[640 * 480];
static int pictureWidth = 320, pictureHeight = 240;
static int16_t captured[CAPTURE_SAMPLES];
static int16_t played[OUT_FRAMES * 2];
// Where resampling is between the last frame kept from the run before and the next.
static double audioPhase;
static int16_t audioLast[2];
static uint8_t saveMemory[SAVE_SIZE];
// The frontend's GL context for the GPU renderer, and what puts the picture in its framebuffer.
static struct retro_hw_render_callback hw;
static bool hwWanted, hwReady, viInContext;
static struct {
	void (*GenTextures)(int, unsigned*);
	void (*DeleteTextures)(int, const unsigned*);
	void (*BindTexture)(unsigned, unsigned);
	void (*TexImage2D)(unsigned, int, int, int, int, int, unsigned, unsigned, const void*);
	void (*TexSubImage2D)(unsigned, int, int, int, int, int, unsigned, unsigned, const void*);
	void (*GenFramebuffers)(int, unsigned*);
	void (*DeleteFramebuffers)(int, const unsigned*);
	void (*BindFramebuffer)(unsigned, unsigned);
	void (*FramebufferTexture2D)(unsigned, unsigned, unsigned, unsigned, int);
	void (*BlitFramebuffer)(int, int, int, int, int, int, int, int, unsigned, unsigned);
	void (*Disable)(unsigned);
	void (*BindBuffer)(unsigned, unsigned);
	void (*MemoryBarrier)(unsigned);
} gl;
static unsigned presentTexture, presentFramebuffer;
static uint32_t presentRgba[640 * 480];
// RetroArch fills saveMemory from the save file after the game loads: it goes to the machine on the
// first run.
static bool saveGiven;
// RDRAM as achievements read it: in little-endian words, as mupen64plus-next keeps it (the sets are
// made on that). Kept up to date each run once the frontend has asked for it.
static uint8_t* ramView;
// What the frontend has in each port (RETRO_DEVICE_*), and the rumble each was last given.
static unsigned devices[4] = { RETRO_DEVICE_JOYPAD, RETRO_DEVICE_JOYPAD, RETRO_DEVICE_JOYPAD, RETRO_DEVICE_JOYPAD };
static bool rumbling[4];

static struct retro_core_option_v2_category optionCategories[] = {
	{ "system", "System", "How the game is emulated." },
	{ "input", "Input", "What is in each controller's slot." },
	{ NULL, NULL, NULL },
};

static struct retro_core_option_v2_definition optionDefinitions[] = {
	{ "armn64_accuracy", "Graphics (Restart)", NULL,
	  "HLE draws the game's display lists without its microcode and puts speed first. MLE and LLE run the microcode on the RSP: "
	  "LLE is the hardware cycle for cycle, MLE the middle ground that also runs hacks made on other emulators.",
	  NULL, "system", { { "hle", "HLE" }, { "mle", "MLE" }, { "lle", "LLE" }, { NULL, NULL } }, "hle" },
	{ "armn64_rdp_threads", "Renderer Threads (Restart)", NULL, "Threads beside the emulation that draw the RDP's pictures. Auto takes the cores there are, up to 3.",
	  NULL, "system", { { "auto", "Auto" }, { "0", "None" }, { "1", "1" }, { "2", "2" }, { "3", "3" }, { NULL, NULL } }, "auto" },
	{ "armn64_renderer", "Renderer (Restart)", NULL,
	  "GPU draws with OpenGL ES 3.1: HLE with the GL renderer, MLE and LLE with the RDP in compute shaders (exact, but slower than the renderer threads on most handhelds). CPU draws with the renderer threads. Auto takes the GPU for HLE and the CPU otherwise.",
	  NULL, "system", { { "auto", "Auto" }, { "gpu", "GPU" }, { "cpu", "CPU" }, { NULL, NULL } }, "auto" },
	{ "armn64_frameskip", "Frame Skip (Restart)", NULL, "Leaves out the drawing of frames that are never shown, as many games make more than the screen can show.",
	  NULL, "system", { { "disabled", "Off" }, { "enabled", "On" }, { NULL, NULL } }, "disabled" },
	{ "armn64_pak1", "Player 1 Pak", NULL, "What is in controller 1's slot: a Controller Pak for saves, or a Rumble Pak.",
	  NULL, "input", { { "memory", "Controller Pak" }, { "rumble", "Rumble Pak" }, { "none", "None" }, { NULL, NULL } }, "memory" },
	{ "armn64_pak2", "Player 2 Pak", NULL, "What is in controller 2's slot: a Controller Pak for saves, or a Rumble Pak.",
	  NULL, "input", { { "memory", "Controller Pak" }, { "rumble", "Rumble Pak" }, { "none", "None" }, { NULL, NULL } }, "memory" },
	{ "armn64_pak3", "Player 3 Pak", NULL, "What is in controller 3's slot: a Controller Pak for saves, or a Rumble Pak.",
	  NULL, "input", { { "memory", "Controller Pak" }, { "rumble", "Rumble Pak" }, { "none", "None" }, { NULL, NULL } }, "memory" },
	{ "armn64_pak4", "Player 4 Pak", NULL, "What is in controller 4's slot: a Controller Pak for saves, or a Rumble Pak.",
	  NULL, "input", { { "memory", "Controller Pak" }, { "rumble", "Rumble Pak" }, { "none", "None" }, { NULL, NULL } }, "memory" },
	{ NULL, NULL, NULL, NULL, NULL, NULL, { { NULL, NULL } }, NULL },
};

static struct retro_core_options_v2 coreOptions = { optionCategories, optionDefinitions };

static void logMessage(enum retro_log_level level, const char* message) {
	if (logPrintf) {
		logPrintf(level, "[armN64] %s\n", message);
	}
}

// The emulator's settings, from the core's options; the recompilers are always on.
static const char* settings(const char* name) {
	static char threads[4];
	if (!strcmp(name, "N64_ACCURACY")) {
		return optionValue(environment, "armn64_accuracy");
	}
	if (!strcmp(name, "N64_SLICE")) {
		return optionIs(environment, "armn64_accuracy", "hle") ? "8192" : "1024";
	}
	if (!strcmp(name, "N64_JIT") || !strcmp(name, "N64_RSPJIT")) {
		return "1";
	}
	if (!strcmp(name, "N64_RDP_GL") || !strcmp(name, "N64_RDP_GPU")) {
		return hwWanted ? "1" : "0";
	}
	// The GPU RDP's pictures stay on the GPU (the VI in its context); beside the renderer threads the
	// VI runs in the frontend's context.
	if (!strcmp(name, "N64_VI_GPU")) {
		return (hwWanted && !optionIs(environment, "armn64_accuracy", "hle")) || viInContext ? "1" : "0";
	}
	if (!strcmp(name, "N64_FRAMESKIP")) {
		return optionIs(environment, "armn64_frameskip", "enabled") ? "1" : "0";
	}
	if (!strcmp(name, "N64_RDP_THREADS")) {
		const char* value = optionValue(environment, "armn64_rdp_threads");
		if (value && strcmp(value, "auto")) {
			return value;
		}
		long cores = sysconf(_SC_NPROCESSORS_ONLN);
		snprintf(threads, sizeof(threads), "%ld", cores > 4 ? 3 : cores > 1 ? cores - 1 : 0);
		return threads;
	}
	return NULL;
}

void retro_set_environment(retro_environment_t callback) {
	environment = callback;
	struct retro_log_callback logging;
	if (callback(RETRO_ENVIRONMENT_GET_LOG_INTERFACE, &logging)) {
		logPrintf = logging.log;
	}
	optionsGive(callback, &coreOptions);
}

void retro_set_video_refresh(retro_video_refresh_t callback) {
	videoRefresh = callback;
}

void retro_set_audio_sample(retro_audio_sample_t callback) {
}

void retro_set_audio_sample_batch(retro_audio_sample_batch_t callback) {
	audioBatch = callback;
}

void retro_set_input_poll(retro_input_poll_t callback) {
	inputPoll = callback;
}

void retro_set_input_state(retro_input_state_t callback) {
	inputState = callback;
}

unsigned retro_api_version(void) {
	return RETRO_API_VERSION;
}

void retro_get_system_info(struct retro_system_info* info) {
	*info = (struct retro_system_info) { "armN64", "0.1", "n64|z64|v64|ndd", false, false };
}

void retro_get_system_av_info(struct retro_system_av_info* info) {
	info->geometry = (struct retro_game_geometry) { (unsigned) pictureWidth, (unsigned) pictureHeight, 640, 480, 4.0f / 3.0f };
	info->timing = (struct retro_system_timing) { pal ? 50.0 : 60.0, OUT_RATE };
}

void retro_init(void) {
}

void retro_deinit(void) {
}

void retro_set_controller_port_device(unsigned port, unsigned device) {
	if (port < 4) {
		devices[port] = device;
	}
}

// Bytes in 32-bit words of the other byte order (the save file's SRAM and flash, achievement memory).
static void swapWords(uint8_t* to, const uint8_t* from, size_t size) {
	for (size_t i = 0; i < size; i += 4) {
		uint32_t w;
		memcpy(&w, from + i, 4);
		w = __builtin_bswap32(w);
		memcpy(to + i, &w, 4);
	}
}

static void saveExport(void) {
	memcpy(saveMemory, machine->eeprom, sizeof(machine->eeprom));
	memcpy(saveMemory + FILE_PAKS, machine->paks, sizeof(machine->paks));
	swapWords(saveMemory + FILE_SRAM, machine->sram, sizeof(machine->sram));
	swapWords(saveMemory + FILE_FLASH, machine->flash, sizeof(machine->flash));
}

// A Controller Pak left empty (by a save file from before there were any) is formatted.
static void saveImport(void) {
	memcpy(machine->eeprom, saveMemory, sizeof(machine->eeprom));
	memcpy(machine->paks, saveMemory + FILE_PAKS, sizeof(machine->paks));
	swapWords(machine->sram, saveMemory + FILE_SRAM, sizeof(machine->sram));
	swapWords(machine->flash, saveMemory + FILE_FLASH, sizeof(machine->flash));
	for (int i = 0; i < 4; ++i) {
		const uint8_t* pak = machine->paks[i];
		size_t k = 0;
		while (k < sizeof(machine->paks[i]) && !pak[k]) {
			++k;
		}
		if (k == sizeof(machine->paks[i])) {
			pakFormat(machine->paks[i]);
		}
	}
}

static uint8_t* readFile(const char* path, size_t* size) {
	FILE* f = fopen(path, "rb");
	if (!f) {
		return NULL;
	}
	fseek(f, 0, SEEK_END);
	long length = ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t* data = length > 0 ? malloc((size_t) length) : NULL;
	if (data && fread(data, 1, (size_t) length, f) != (size_t) length) {
		free(data);
		data = NULL;
	}
	fclose(f);
	*size = data ? (size_t) length : 0;
	return data;
}

// A machine running the game loaded, from power-on.
static struct N64* startMachine(void) {
	struct N64* n = n64Create();
	if (!n) {
		return NULL;
	}
	n->settings = settings;
	if (disk ? !n64LoadDisk(n, ipl, iplSize, game, gameSize) : !n64LoadRom(n, game, gameSize)) {
		n64Destroy(n);
		return NULL;
	}
	n->audio = captured;
	n->audioCapacity = CAPTURE_SAMPLES;
	return n;
}

enum {
	GL_TEXTURE_2D = 0x0DE1, GL_RGBA = 0x1908, GL_RGBA8 = 0x8058, GL_UNSIGNED_BYTE = 0x1401, GL_READ_FRAMEBUFFER = 0x8CA8,
	GL_DRAW_FRAMEBUFFER = 0x8CA9, GL_COLOR_ATTACHMENT0 = 0x8CE0, GL_COLOR_BUFFER_BIT = 0x4000, GL_NEAREST = 0x2600,
	GL_SCISSOR_TEST = 0x0C11, GL_PIXEL_UNPACK_BUFFER = 0x88EC, GL_PIXEL_BUFFER_BARRIER_BIT = 0x80,
};

static void* hwProc(const char* name) {
	return (void*) hw.get_proc_address(name);
}

// The frontend made the context (again): the renderer starts in it, and the picture's texture.
static void hwReset(void) {
	glesFrontend(hwProc);
	void** fns = (void**) &gl;
	static const char* const names[] = { "glGenTextures", "glDeleteTextures", "glBindTexture", "glTexImage2D", "glTexSubImage2D", "glGenFramebuffers",
		                                   "glDeleteFramebuffers", "glBindFramebuffer", "glFramebufferTexture2D", "glBlitFramebuffer", "glDisable",
		                                   "glBindBuffer", "glMemoryBarrier" };
	hwReady = true;
	for (size_t i = 0; i < sizeof(names) / sizeof(*names); ++i) {
		fns[i] = hwProc(names[i]);
		hwReady &= fns[i] != NULL;
	}
	if (!hwReady) {
		logMessage(RETRO_LOG_ERROR, "the frontend's GL context lacks functions the core needs");
		return;
	}
	gl.GenTextures(1, &presentTexture);
	gl.BindTexture(GL_TEXTURE_2D, presentTexture);
	gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	gl.GenFramebuffers(1, &presentFramebuffer);
	gl.BindFramebuffer(GL_READ_FRAMEBUFFER, presentFramebuffer);
	gl.FramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, presentTexture, 0);
}

static void hwDestroy(void) {
	if (hwReady) {
		gl.DeleteFramebuffers(1, &presentFramebuffer);
		gl.DeleteTextures(1, &presentTexture);
	}
	hwReady = false;
}

// The GPU renderer in the frontend's context when it draws with GL: a context of the same kind,
// version 3.1 for GLES and 4.3 for a core profile (both run the renderer's GLSL ES shaders); a
// compatibility profile is asked for without a version.
static bool askHardwareContext(void) {
	unsigned preferred = RETRO_HW_CONTEXT_NONE;
	if (!environment(RETRO_ENVIRONMENT_GET_PREFERRED_HW_RENDER, &preferred)) {
		return false;
	}
	bool es = preferred == RETRO_HW_CONTEXT_OPENGLES2 || preferred == RETRO_HW_CONTEXT_OPENGLES3 || preferred == RETRO_HW_CONTEXT_OPENGLES_VERSION;
	if (!es && preferred != RETRO_HW_CONTEXT_OPENGL && preferred != RETRO_HW_CONTEXT_OPENGL_CORE) {
		return false;
	}
	hw = (struct retro_hw_render_callback) { 0 };
	hw.context_type = es ? RETRO_HW_CONTEXT_OPENGLES_VERSION : preferred;
	hw.version_major = es ? 3 : preferred == RETRO_HW_CONTEXT_OPENGL_CORE ? 4 : 0;
	hw.version_minor = es ? 1 : preferred == RETRO_HW_CONTEXT_OPENGL_CORE ? 3 : 0;
	hw.context_reset = hwReset;
	hw.context_destroy = hwDestroy;
	hw.bottom_left_origin = true;
	if (!environment(RETRO_ENVIRONMENT_SET_HW_RENDER, &hw)) {
		// A frontend built for GLES can still say it prefers OpenGL (RetroArch's gl driver on ROCKNIX).
		if (es) {
			return false;
		}
		hw.context_type = RETRO_HW_CONTEXT_OPENGLES_VERSION;
		hw.version_major = 3;
		hw.version_minor = 1;
		if (!environment(RETRO_ENVIRONMENT_SET_HW_RENDER, &hw)) {
			return false;
		}
	}
	environment(RETRO_ENVIRONMENT_SET_HW_SHARED_CONTEXT, NULL);
	return true;
}

// The VI's picture into the frontend's framebuffer, upside down as GL keeps it: from the GL buffer
// the GPU made it in, else from picture.
static void present(int width, int height, unsigned buffer) {
	gl.Disable(GL_SCISSOR_TEST);
	gl.BindTexture(GL_TEXTURE_2D, presentTexture);
	if (buffer) {
		gl.MemoryBarrier(GL_PIXEL_BUFFER_BARRIER_BIT);
		gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
		gl.TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		gl.BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	} else {
		for (int i = 0; i < width * height; ++i) {
			uint32_t p = picture[i];
			presentRgba[i] = (p >> 16 & 0xFF) | (p & 0xFF00) | (p & 0xFF) << 16 | 0xFF000000u;
		}
		gl.TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, presentRgba);
	}
	gl.BindFramebuffer(GL_READ_FRAMEBUFFER, presentFramebuffer);
	gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, (unsigned) hw.get_current_framebuffer());
	gl.BlitFramebuffer(0, 0, width, height, 0, height, width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glrForget(machine);
	videoRefresh(RETRO_HW_FRAME_BUFFER_VALID, (unsigned) width, (unsigned) height, 0);
}

// The GL renderer's picture from its framebuffer into the frontend's, upside down as GL keeps it.
static void presentGl(unsigned framebuffer, int x, int y, int width, int height) {
	gl.Disable(GL_SCISSOR_TEST);
	gl.BindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
	gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, (unsigned) hw.get_current_framebuffer());
	gl.BlitFramebuffer(x, y, x + width, y + height, 0, height, width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glrForget(machine);
	videoRefresh(RETRO_HW_FRAME_BUFFER_VALID, (unsigned) width, (unsigned) height, 0);
}

// Each port's controls, and what can be plugged in.
static void describeInput(void) {
	static const struct {
		unsigned device, index, id;
		const char* name;
	} controls[] = {
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_B, "A" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_Y, "B" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_L2, "Z" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_L, "L" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_R, "R" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_START, "Start" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_UP, "D-Pad Up" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_DOWN, "D-Pad Down" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_LEFT, "D-Pad Left" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_RIGHT, "D-Pad Right" },
		{ RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X, "Control Stick X" },
		{ RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y, "Control Stick Y" },
		{ RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X, "C Buttons X" },
		{ RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y, "C Buttons Y" },
	};
	enum { CONTROLS = sizeof(controls) / sizeof(*controls) };
	static struct retro_input_descriptor descriptors[4 * CONTROLS + 1];
	for (unsigned port = 0; port < 4; ++port) {
		for (unsigned i = 0; i < CONTROLS; ++i) {
			descriptors[port * CONTROLS + i] = (struct retro_input_descriptor) { port, controls[i].device, controls[i].index, controls[i].id, controls[i].name };
		}
	}
	environment(RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS, descriptors);
	static const struct retro_controller_description types[] = {
		{ "Controller", RETRO_DEVICE_JOYPAD },
		{ "N64 Mouse", RETRO_DEVICE_MOUSE },
		{ "None", RETRO_DEVICE_NONE },
	};
	static const struct retro_controller_info ports[] = {
		{ types, 3 }, { types, 3 }, { types, 3 }, { types, 3 }, { NULL, 0 },
	};
	environment(RETRO_ENVIRONMENT_SET_CONTROLLER_INFO, (void*) ports);
}

bool retro_load_game(const struct retro_game_info* info) {
	if (!info || !info->data || !info->size) {
		return false;
	}
	enum retro_pixel_format format = RETRO_PIXEL_FORMAT_XRGB8888;
	if (!environment(RETRO_ENVIRONMENT_SET_PIXEL_FORMAT, &format)) {
		return false;
	}
	game = malloc(info->size);
	if (!game) {
		return false;
	}
	// The GPU renderers, in the frontend's GL context if it has one, else in the core's own (beside a
	// Vulkan frontend, say). With the renderer threads, the VI's filters use the GL context.
	hwWanted = optionIs(environment, "armn64_renderer", "gpu") ||
	           (!optionIs(environment, "armn64_renderer", "cpu") && optionIs(environment, "armn64_accuracy", "hle"));
	bool context = askHardwareContext();
	viInContext = context && !hwWanted;
	if (hwWanted && !context) {
		unsigned preferred = RETRO_HW_CONTEXT_NONE;
		environment(RETRO_ENVIRONMENT_GET_PREFERRED_HW_RENDER, &preferred);
		hwWanted = preferred == RETRO_HW_CONTEXT_VULKAN || preferred == RETRO_HW_CONTEXT_NONE;
		hw.context_type = RETRO_HW_CONTEXT_NONE;
	}
	memcpy(game, info->data, info->size);
	gameSize = info->size;
	size_t length = info->path ? strlen(info->path) : 0;
	disk = length > 4 && !strcasecmp(info->path + length - 4, ".ndd");
	if (disk) {
		// A 64DD disk boots from the drive's IPL, 64DD_IPL.bin in the system folder.
		const char* system = NULL;
		char path[4096];
		if (environment(RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY, &system) && system) {
			snprintf(path, sizeof(path), "%s/64DD_IPL.bin", system);
			ipl = readFile(path, &iplSize);
		}
		if (!ipl) {
			logMessage(RETRO_LOG_ERROR, "a 64DD disk needs 64DD_IPL.bin in the system folder");
			return false;
		}
	}
	machine = startMachine();
	if (!machine) {
		logMessage(RETRO_LOG_ERROR, "this game can't be loaded");
		return false;
	}
	pal = machine->fieldCycles == CPU_HZ / 50;
	saveExport();
	saveGiven = false;
	struct retro_rumble_interface rumble;
	setRumble = environment(RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE, &rumble) ? rumble.set_rumble_state : NULL;
	memset(rumbling, 0, sizeof(rumbling));
	audioPhase = 0;
	memset(audioLast, 0, sizeof(audioLast));
	describeInput();
	bool achievements = true;
	environment(RETRO_ENVIRONMENT_SET_SUPPORT_ACHIEVEMENTS, &achievements);
	return true;
}

bool retro_load_game_special(unsigned type, const struct retro_game_info* info, size_t count) {
	return false;
}

void retro_unload_game(void) {
	n64Destroy(machine);
	glesYield();
	machine = NULL;
	free(game);
	free(ipl);
	free(ramView);
	game = ipl = NULL;
	ramView = NULL;
}

void retro_reset(void) {
	if (!machine) {
		return;
	}
	n64Destroy(machine);
	glesYield();
	machine = startMachine();
	if (machine) {
		saveImport();
	}
}

// An analog axis as the controller's stick reports it: about 80 each way.
static int stick(unsigned port, unsigned index, unsigned axis) {
	return inputState(port, RETRO_DEVICE_ANALOG, index, axis) * 80 / 32768;
}

// A controller as the PIF reads it: the buttons in the top half, then the stick's X and Y.
static uint32_t readController(unsigned port) {
	static const struct {
		unsigned id;
		uint32_t bit;
	} buttons[] = {
		{ RETRO_DEVICE_ID_JOYPAD_B, 1u << 31 }, { RETRO_DEVICE_ID_JOYPAD_Y, 1u << 30 }, { RETRO_DEVICE_ID_JOYPAD_L2, 1u << 29 },
		{ RETRO_DEVICE_ID_JOYPAD_START, 1u << 28 }, { RETRO_DEVICE_ID_JOYPAD_UP, 1u << 27 }, { RETRO_DEVICE_ID_JOYPAD_DOWN, 1u << 26 },
		{ RETRO_DEVICE_ID_JOYPAD_LEFT, 1u << 25 }, { RETRO_DEVICE_ID_JOYPAD_RIGHT, 1u << 24 }, { RETRO_DEVICE_ID_JOYPAD_L, 1u << 21 },
		{ RETRO_DEVICE_ID_JOYPAD_R, 1u << 20 },
	};
	uint32_t state = 0;
	for (size_t i = 0; i < sizeof(buttons) / sizeof(*buttons); ++i) {
		state |= inputState(port, RETRO_DEVICE_JOYPAD, 0, buttons[i].id) ? buttons[i].bit : 0;
	}
	// The C buttons from the right stick, pushed past halfway.
	int cx = inputState(port, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X);
	int cy = inputState(port, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y);
	state |= (cy < -16384 ? 1u << 19 : 0) | (cy > 16384 ? 1u << 18 : 0) | (cx < -16384 ? 1u << 17 : 0) | (cx > 16384 ? 1u << 16 : 0);
	int x = stick(port, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X);
	int y = -stick(port, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y);
	return state | (uint32_t) (uint8_t) x << 8 | (uint8_t) y;
}

// An N64 Mouse: its buttons where A and B are, and how far it moved since the last field.
static uint32_t readMouse(unsigned port) {
	int x = inputState(port, RETRO_DEVICE_MOUSE, 0, RETRO_DEVICE_ID_MOUSE_X);
	int y = -inputState(port, RETRO_DEVICE_MOUSE, 0, RETRO_DEVICE_ID_MOUSE_Y);
	x = x < -128 ? -128 : x > 127 ? 127 : x;
	y = y < -128 ? -128 : y > 127 ? 127 : y;
	return (inputState(port, RETRO_DEVICE_MOUSE, 0, RETRO_DEVICE_ID_MOUSE_LEFT) ? 1u << 31 : 0) |
	       (inputState(port, RETRO_DEVICE_MOUSE, 0, RETRO_DEVICE_ID_MOUSE_RIGHT) ? 1u << 30 : 0) | (uint32_t) (uint8_t) x << 8 | (uint8_t) y;
}

// The ports as the frontend has them, with the paks the options put in, and the Rumble Paks' motors
// to the frontend.
static void readPorts(void) {
	for (unsigned port = 0; port < 4; ++port) {
		unsigned const device = devices[port] & RETRO_DEVICE_MASK;
		machine->plugged[port] = device == RETRO_DEVICE_MOUSE ? PLUG_MOUSE : device == RETRO_DEVICE_NONE ? PLUG_NONE : PLUG_CONTROLLER;
		char key[16];
		snprintf(key, sizeof(key), "armn64_pak%u", port + 1);
		machine->pak[port] = optionIs(environment, key, "rumble") ? PAK_RUMBLE : optionIs(environment, key, "none") ? PAK_NONE : PAK_MEMORY;
		machine->pad[port] = machine->plugged[port] == PLUG_MOUSE ? readMouse(port) : machine->plugged[port] ? readController(port) : 0;
		bool const on = machine->pak[port] == PAK_RUMBLE && machine->rumble[port];
		if (setRumble && on != rumbling[port]) {
			setRumble(port, RETRO_RUMBLE_STRONG, on ? 0xFFFF : 0);
			rumbling[port] = on;
		}
	}
}

// The field's samples, from the AI's rate to 44.1 kHz by straight lines between them.
static void playAudio(void) {
	size_t frames = machine->audioCount / 2;
	machine->audioCount = 0;
	if (!frames && !machine->ai.queued) {
		// The AI is idle: a field of silence.
		size_t silence = OUT_RATE / (pal ? 50 : 60);
		memset(played, 0, silence * 4);
		for (size_t done = 0; done < silence;) {
			done += audioBatch(played + done * 2, silence - done);
		}
		return;
	}
	if (!frames || !machine->audioRate) {
		return;
	}
	double step = (double) machine->audioRate / OUT_RATE;
	size_t out = 0;
	// Frame -1 is the last one before this field's.
	while (audioPhase < (double) frames - 1 && out < OUT_FRAMES) {
		long i = (long) (audioPhase + 1) - 1;
		double f = audioPhase - (double) i;
		const int16_t* a = i < 0 ? audioLast : &captured[i * 2];
		const int16_t* b = &captured[(i + 1) * 2];
		played[out * 2] = (int16_t) (a[0] + (b[0] - a[0]) * f);
		played[out * 2 + 1] = (int16_t) (a[1] + (b[1] - a[1]) * f);
		++out;
		audioPhase += step;
	}
	audioPhase -= (double) frames;
	memcpy(audioLast, &captured[(frames - 1) * 2], sizeof(audioLast));
	for (size_t done = 0; done < out;) {
		done += audioBatch(played + done * 2, out - done);
	}
}

void retro_run(void) {
	if (!machine) {
		return;
	}
	if (!saveGiven) {
		saveImport();
		saveGiven = true;
	}
	bool shared = hw.context_type != RETRO_HW_CONTEXT_NONE;
	if (!shared) {
		glesResume();
	}
	inputPoll();
	readPorts();
	n64RunField(machine);
	int width = 0, height = 0;
	// The GPU RDP's picture goes from its buffer to the frontend's framebuffer without the CPU, and so
	// does the GL renderer's.
	unsigned buffer = 0, framebuffer = 0;
	int x = 0, y = 0;
	if (shared && hwReady && n64GlPicture(machine, &framebuffer, &x, &y, &width, &height) && width && height) {
		if (width != pictureWidth || height != pictureHeight) {
			pictureWidth = width, pictureHeight = height;
			struct retro_game_geometry geometry = { (unsigned) width, (unsigned) height, 640, 480, 4.0f / 3.0f };
			environment(RETRO_ENVIRONMENT_SET_GEOMETRY, &geometry);
		}
		presentGl(framebuffer, x, y, width, height);
	} else if (shared && hwReady && n64PictureStart(machine, &width, &height) && width && height && (buffer = n64PictureBuffer(machine))) {
		if (width != pictureWidth || height != pictureHeight) {
			pictureWidth = width, pictureHeight = height;
			struct retro_game_geometry geometry = { (unsigned) width, (unsigned) height, 640, 480, 4.0f / 3.0f };
			environment(RETRO_ENVIRONMENT_SET_GEOMETRY, &geometry);
		}
		present(width, height, buffer);
	} else if (n64Picture(machine, picture, &width, &height) && width && height) {
		if (width != pictureWidth || height != pictureHeight) {
			pictureWidth = width, pictureHeight = height;
			struct retro_game_geometry geometry = { (unsigned) width, (unsigned) height, 640, 480, 4.0f / 3.0f };
			environment(RETRO_ENVIRONMENT_SET_GEOMETRY, &geometry);
		}
		if (!shared) {
			videoRefresh(picture, (unsigned) width, (unsigned) height, (size_t) width * 4);
		} else if (hwReady) {
			present(width, height, 0);
		}
	} else {
		videoRefresh(NULL, (unsigned) pictureWidth, (unsigned) pictureHeight, 0);
	}
	if (!shared) {
		glesYield();
	}
	playAudio();
	if (machine->saveDirty) {
		saveExport();
		machine->saveDirty = false;
	}
	if (ramView) {
		swapWords(ramView, machine->rdram, RDRAM_SIZE);
	}
}

size_t retro_serialize_size(void) {
	return n64StateSize();
}

// Save states: the GPU's pictures come back to RDRAM, and go on loading, in the core's own GL
// context when it has one.
bool retro_serialize(void* data, size_t size) {
	if (!machine) {
		return false;
	}
	bool own = hw.context_type == RETRO_HW_CONTEXT_NONE;
	if (own) {
		glesResume();
	}
	bool saved = n64StateSave(machine, data, size);
	if (own) {
		glesYield();
	}
	return saved;
}

bool retro_unserialize(const void* data, size_t size) {
	if (!machine) {
		return false;
	}
	bool own = hw.context_type == RETRO_HW_CONTEXT_NONE;
	if (own) {
		glesResume();
	}
	bool loaded = n64StateLoad(machine, data, size);
	if (own) {
		glesYield();
	}
	// The state's saves stand (the save file takes them on the next run).
	saveGiven |= loaded;
	return loaded;
}

void retro_cheat_reset(void) {
}

void retro_cheat_set(unsigned index, bool enabled, const char* code) {
}

unsigned retro_get_region(void) {
	return pal ? RETRO_REGION_PAL : RETRO_REGION_NTSC;
}

void* retro_get_memory_data(unsigned id) {
	if (id == RETRO_MEMORY_SAVE_RAM) {
		return saveMemory;
	}
	if (id == RETRO_MEMORY_SYSTEM_RAM && machine) {
		if (!ramView && (ramView = malloc(RDRAM_SIZE))) {
			swapWords(ramView, machine->rdram, RDRAM_SIZE);
		}
		return ramView;
	}
	return NULL;
}

size_t retro_get_memory_size(unsigned id) {
	if (id == RETRO_MEMORY_SAVE_RAM) {
		return SAVE_SIZE;
	}
	if (id == RETRO_MEMORY_SYSTEM_RAM && machine) {
		return RDRAM_SIZE;
	}
	return 0;
}
