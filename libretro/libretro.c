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
	// The save file: the EEPROM, then the SRAM or flash.
	EEPROM_SIZE = sizeof(((struct N64*) 0)->eeprom),
	CART_SAVE_SIZE = sizeof(((struct N64*) 0)->save),
	SAVE_SIZE = EEPROM_SIZE + CART_SAVE_SIZE,
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
static bool hwWanted, hwReady;
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
} gl;
static unsigned presentTexture, presentFramebuffer;
static uint32_t presentRgba[640 * 480];
// RetroArch fills saveMemory from the save file after the game loads: it goes to the machine on the
// first run.
static bool saveGiven;

static struct retro_core_option_v2_category optionCategories[] = {
	{ "system", "System", "How the game is emulated." },
	{ NULL, NULL, NULL },
};

static struct retro_core_option_v2_definition optionDefinitions[] = {
	{ "armn64_accuracy", "Graphics (Restart)", NULL,
	  "High Level draws the game's display lists without its microcode and is the fastest. Compatible and Accurate run the microcode on the RSP; "
	  "Accurate keeps the hardware's own behaviour where Compatible follows what other emulators do.",
	  NULL, "system", { { "hle", "High Level" }, { "lle-compatible", "Compatible" }, { "lle-accurate", "Accurate" }, { NULL, NULL } }, "hle" },
	{ "armn64_rdp_threads", "Renderer Threads (Restart)", NULL, "Threads beside the emulation that draw the RDP's pictures. Auto takes the cores there are, up to 3.",
	  NULL, "system", { { "auto", "Auto" }, { "0", "None" }, { "1", "1" }, { "2", "2" }, { "3", "3" }, { NULL, NULL } }, "auto" },
	{ "armn64_renderer", "Renderer (Restart)", NULL,
	  "GPU draws High Level graphics with OpenGL ES 3.1 and is faster on handhelds; CPU draws them with the renderer threads. Compatible and Accurate always use the CPU.",
	  NULL, "system", { { "gpu", "GPU" }, { "cpu", "CPU" }, { NULL, NULL } }, "gpu" },
	{ "armn64_frameskip", "Frame Skip (Restart)", NULL, "Leaves out the drawing of frames that are never shown, as many games make more than the screen can show.",
	  NULL, "system", { { "disabled", "Off" }, { "enabled", "On" }, { NULL, NULL } }, "disabled" },
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
		return "1024";
	}
	if (!strcmp(name, "N64_JIT") || !strcmp(name, "N64_RSPJIT")) {
		return "1";
	}
	if (!strcmp(name, "N64_RDP_GL")) {
		return hwWanted ? "1" : "0";
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
	GL_SCISSOR_TEST = 0x0C11,
};

static void* hwProc(const char* name) {
	return (void*) hw.get_proc_address(name);
}

// The frontend made the context (again): the renderer starts in it, and the picture's texture.
static void hwReset(void) {
	glesFrontend(hwProc);
	void** fns = (void**) &gl;
	static const char* const names[] = { "glGenTextures", "glDeleteTextures", "glBindTexture", "glTexImage2D", "glTexSubImage2D", "glGenFramebuffers",
		                                   "glDeleteFramebuffers", "glBindFramebuffer", "glFramebufferTexture2D", "glBlitFramebuffer", "glDisable" };
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
		return false;
	}
	environment(RETRO_ENVIRONMENT_SET_HW_SHARED_CONTEXT, NULL);
	return true;
}

// The VI's picture into the frontend's framebuffer, upside down as GL keeps it.
static void present(int width, int height) {
	for (int i = 0; i < width * height; ++i) {
		uint32_t p = picture[i];
		presentRgba[i] = (p >> 16 & 0xFF) | (p & 0xFF00) | (p & 0xFF) << 16 | 0xFF000000u;
	}
	gl.Disable(GL_SCISSOR_TEST);
	gl.BindTexture(GL_TEXTURE_2D, presentTexture);
	gl.TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, presentRgba);
	gl.BindFramebuffer(GL_READ_FRAMEBUFFER, presentFramebuffer);
	gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, (unsigned) hw.get_current_framebuffer());
	gl.BlitFramebuffer(0, 0, width, height, 0, height, width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glrForget(machine);
	videoRefresh(RETRO_HW_FRAME_BUFFER_VALID, (unsigned) width, (unsigned) height, 0);
}

static void describeInput(void) {
	static const struct retro_input_descriptor descriptors[] = {
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_B, "A" },
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_Y, "B" },
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_L2, "Z" },
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_L, "L" },
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_R, "R" },
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_START, "Start" },
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_UP, "D-Pad Up" },
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_DOWN, "D-Pad Down" },
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_LEFT, "D-Pad Left" },
		{ 0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_RIGHT, "D-Pad Right" },
		{ 0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X, "Control Stick X" },
		{ 0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y, "Control Stick Y" },
		{ 0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X, "C Buttons X" },
		{ 0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y, "C Buttons Y" },
		{ 0, 0, 0, 0, NULL },
	};
	environment(RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS, (void*) descriptors);
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
	// The GPU renderer draws High Level graphics, in the frontend's GL context if it has one, else in
	// the core's own (beside a Vulkan frontend, say).
	hwWanted = optionIs(environment, "armn64_accuracy", "hle") && !optionIs(environment, "armn64_renderer", "cpu");
	if (hwWanted && !askHardwareContext()) {
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
	memcpy(saveMemory, machine->eeprom, EEPROM_SIZE);
	memcpy(saveMemory + EEPROM_SIZE, machine->save, CART_SAVE_SIZE);
	saveGiven = false;
	audioPhase = 0;
	memset(audioLast, 0, sizeof(audioLast));
	describeInput();
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
	game = ipl = NULL;
}

void retro_reset(void) {
	if (!machine) {
		return;
	}
	n64Destroy(machine);
	glesYield();
	machine = startMachine();
	if (machine) {
		memcpy(machine->eeprom, saveMemory, EEPROM_SIZE);
		memcpy(machine->save, saveMemory + EEPROM_SIZE, CART_SAVE_SIZE);
	}
}

static bool held(unsigned id) {
	return inputState(0, RETRO_DEVICE_JOYPAD, 0, id) != 0;
}

// An analog axis as the controller's stick reports it: about 80 each way.
static int stick(unsigned index, unsigned axis) {
	return inputState(0, RETRO_DEVICE_ANALOG, index, axis) * 80 / 32768;
}

// Controller 1 as the PIF reads it: the buttons in the top half, then the stick's X and Y.
static uint32_t readController(void) {
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
		state |= held(buttons[i].id) ? buttons[i].bit : 0;
	}
	// The C buttons from the right stick, pushed past halfway.
	int cx = inputState(0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X);
	int cy = inputState(0, RETRO_DEVICE_ANALOG, RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y);
	state |= (cy < -16384 ? 1u << 19 : 0) | (cy > 16384 ? 1u << 18 : 0) | (cx < -16384 ? 1u << 17 : 0) | (cx > 16384 ? 1u << 16 : 0);
	int x = stick(RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X);
	int y = -stick(RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y);
	return state | (uint32_t) (uint8_t) x << 8 | (uint8_t) y;
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
		memcpy(machine->eeprom, saveMemory, EEPROM_SIZE);
		memcpy(machine->save, saveMemory + EEPROM_SIZE, CART_SAVE_SIZE);
		saveGiven = true;
	}
	bool shared = hw.context_type != RETRO_HW_CONTEXT_NONE;
	if (!shared) {
		glesResume();
	}
	inputPoll();
	machine->buttons = readController();
	n64RunField(machine);
	int width = 0, height = 0;
	if (n64Picture(machine, picture, &width, &height) && width && height) {
		if (width != pictureWidth || height != pictureHeight) {
			pictureWidth = width, pictureHeight = height;
			struct retro_game_geometry geometry = { (unsigned) width, (unsigned) height, 640, 480, 4.0f / 3.0f };
			environment(RETRO_ENVIRONMENT_SET_GEOMETRY, &geometry);
		}
		if (!shared) {
			videoRefresh(picture, (unsigned) width, (unsigned) height, (size_t) width * 4);
		} else if (hwReady) {
			present(width, height);
		}
	} else {
		videoRefresh(NULL, (unsigned) pictureWidth, (unsigned) pictureHeight, 0);
	}
	if (!shared) {
		glesYield();
	}
	playAudio();
	memcpy(saveMemory, machine->eeprom, EEPROM_SIZE);
	memcpy(saveMemory + EEPROM_SIZE, machine->save, CART_SAVE_SIZE);
}

size_t retro_serialize_size(void) {
	return 0;
}

bool retro_serialize(void* data, size_t size) {
	return false;
}

bool retro_unserialize(const void* data, size_t size) {
	return false;
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
		return machine->rdram;
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
