// armN64 as a libretro core: a video field each run, the VI's picture, the AI's samples at
// 44.1 kHz, controller 1, and the cartridge's save memory. The GPU renderer draws in a GL context the
// frontend shares with its own when it draws with GL (the picture is then put in its framebuffer),
// else in one of the core's own, made current only while the core runs.
#include "gles.h"
#include "n64.h"

#include "libretro.h"
#include "options.h"
#include "paks.h"

#include <dirent.h>
#ifndef __ANDROID__
#include <glob.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
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
// GameShark codes from the frontend (retro_cheat_set).
static struct Cheats cheats;
// The game as it was loaded (and the 64DD's IPL for a disk), to start again on a reset.
static uint8_t* game;
static size_t gameSize;
static uint8_t* ipl;
static size_t iplSize;
static bool disk;
static bool pal;
static uint32_t picture[640 * 480];
static int pictureWidth = 320, pictureHeight = 240;
static float pictureAspect = 4.0f / 3.0f;
static unsigned maxScale = 1;
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
// The texture's size: 640 by 480, larger only for a picture that needs it (the GPU RDP's at its scale).
static int presentWidth, presentHeight;
static uint32_t presentRgba[640 * 480];
// RetroArch fills saveMemory from the save file after the game loads: it goes to the machine on the
// first run.
static bool saveGiven;
// RDRAM as achievements read it: in little-endian words, as mupen64plus-next keeps it (the sets are
// made on that). Once the frontend has asked for it, the pages written are brought up to date after
// each run.
static uint8_t* ramView;
// What the frontend has in each port (RETRO_DEVICE_*), and the rumble each was last given.
static unsigned devices[4] = { RETRO_DEVICE_JOYPAD, RETRO_DEVICE_JOYPAD, RETRO_DEVICE_JOYPAD, RETRO_DEVICE_JOYPAD };
static bool rumbling[4];

static struct retro_core_option_v2_category optionCategories[] = {
	{ "system", "System", "How exactly the console is emulated, how fast it runs, and its memory." },
	{ "video", "Video", "The renderer, and the picture's size, shape and edges." },
	{ "gl", "GL Renderer", "How the GL renderer filters, smooths and builds its shaders." },
	{ "input", "Input", "The stick, button layouts, two-controller play and rumble." },
	{ "paks", "Controller Paks", "What is in each controller's slot, and the Controller Pak Manager." },
	{ "advanced", "Advanced", "Settings for particular games and devices." },
	{ NULL, NULL, NULL },
};

// Keys and values match saved configurations; labels are what the menu shows. showOptions hides
// the options that do nothing with the others' values.
static struct retro_core_option_v2_definition optionDefinitions[] = {
	{ "armn64_accuracy", "Accuracy", NULL,
	  "Fastest and Fast put speed first; Balanced runs each game's own audio code; Accurate is cycle-exact. Applies on Restart.",
	  NULL, "system", { { "fle", "Fastest (FLE)" }, { "hle", "Fast (HLE)" }, { "mle", "Balanced (MLE)" }, { "lle", "Accurate (LLE)" }, { NULL, NULL } }, "hle" },
	{ "armn64_overclock", "Overclock", NULL,
	  "A faster CPU, RSP and RDP for less slowdown and shorter loading. Applies on Restart.",
	  NULL, "system", { { "1", "Off" }, { "1.5", "1.5x" }, { "2", "2x" }, { "2.5", "2.5x" }, { "3", "3x" }, { "4", "4x" }, { NULL, NULL } }, "1" },
	{ "armn64_frameskip", "Frame Skip", NULL,
	  "Skips drawing frames for full speed on slow devices. Auto: only when it falls behind. Applies on Restart.",
	  NULL, "system", { { "disabled", "Off" }, { "auto", "Auto" }, { "enabled", "Every Other Frame" }, { NULL, NULL } }, "disabled" },
	{ "armn64_expansion", "Expansion Pak", NULL,
	  "The 4 MB memory add-on most later games use. Off: as without one, for games and hacks that need that. Applies on Restart.",
	  NULL, "system", { { "enabled", "On" }, { "disabled", "Off" }, { NULL, NULL } }, "enabled" },
	{ "armn64_renderer", "Renderer", NULL,
	  "GL: fastest, HD, not exact. CPU: exact, the choice on handhelds. GPU: exact, for desktop graphics cards, also HD. Applies on Restart.",
	  NULL, "video", { { "gl", "GL (Fast, HD)" }, { "gpu", "GPU (Exact, HD)" }, { "cpu", "CPU (Exact, Native)" }, { NULL, NULL } }, "gl" },
	{ "armn64_resolution", "Resolution", NULL,
	  "GL and GPU drawing resolution. Auto fits the display with GL and is Native with GPU (exact only at Native). Applies on Restart.",
	  NULL, "video", { { "auto", "Auto" }, { "1", "Native" }, { "2", "2x" }, { "3", "3x" }, { "4", "4x" }, { "5", "5x" }, { "6", "6x" }, { "7", "7x" }, { "8", "8x" }, { NULL, NULL } }, "auto" },
	{ "armn64_widescreen", "Widescreen", NULL,
	  "16:9 and 16:10 draw more at the sides (Library graphics); things just off the 4:3 edges may be missing. Picture Only: for games' own wide modes.",
	  NULL, "video", { { "disabled", "Off" }, { "16:9", "16:9" }, { "16:10", "16:10" }, { "16:9-picture", "16:9 Picture Only" }, { "16:10-picture", "16:10 Picture Only" }, { NULL, NULL } }, "disabled" },
	{ "armn64_aspect", "Aspect Ratio", NULL,
	  "Fill 4:3 stretches the picture; TV 4:3 keeps a TV's borders; Square Pixels shows each pixel square.",
	  NULL, "video", { { "fill", "Fill 4:3" }, { "tv", "TV 4:3" }, { "square", "Square Pixels" }, { NULL, NULL } }, "fill" },
	{ "armn64_crop_overscan", "Crop Overscan", NULL,
	  "Hides the picture's edges, as a TV's frame did. Custom: each side's own amount.",
	  NULL, "video", { { "enabled", "On" }, { "disabled", "Off" }, { "custom", "Custom" }, { NULL, NULL } }, "enabled" },
	{ "armn64_crop_top", "Crop Top", NULL, "With Crop Overscan Custom: the console's pixels hidden at the top.",
	  NULL, "video", { { "0", "0" }, { "2", "2" }, { "4", "4" }, { "6", "6" }, { "8", "8" }, { "10", "10" }, { "12", "12" }, { "16", "16" }, { "20", "20" }, { "24", "24" }, { NULL, NULL } }, "8" },
	{ "armn64_crop_bottom", "Crop Bottom", NULL, "With Crop Overscan Custom: the console's pixels hidden at the bottom.",
	  NULL, "video", { { "0", "0" }, { "2", "2" }, { "4", "4" }, { "6", "6" }, { "8", "8" }, { "10", "10" }, { "12", "12" }, { "16", "16" }, { "20", "20" }, { "24", "24" }, { NULL, NULL } }, "8" },
	{ "armn64_crop_left", "Crop Left", NULL, "With Crop Overscan Custom: the console's pixels hidden at the left.",
	  NULL, "video", { { "0", "0" }, { "2", "2" }, { "4", "4" }, { "6", "6" }, { "8", "8" }, { "10", "10" }, { "12", "12" }, { "16", "16" }, { "20", "20" }, { "24", "24" }, { NULL, NULL } }, "10" },
	{ "armn64_crop_right", "Crop Right", NULL, "With Crop Overscan Custom: the console's pixels hidden at the right.",
	  NULL, "video", { { "0", "0" }, { "2", "2" }, { "4", "4" }, { "6", "6" }, { "8", "8" }, { "10", "10" }, { "12", "12" }, { "16", "16" }, { "20", "20" }, { "24", "24" }, { NULL, NULL } }, "10" },
	{ "armn64_vi_filters", "TV Filters", NULL,
	  "The console's antialiasing and dither filters, as a TV showed them. Auto: on with Balanced and Accurate.",
	  NULL, "video", { { "auto", "Auto" }, { "on", "On" }, { "off", "Off" }, { NULL, NULL } }, "auto" },
	{ "armn64_gl_filter", "Texture Filtering", NULL,
	  "Console: the N64's own three-texel filter. Smooth: four texels, softer. Sharp: unfiltered, square texels. Applies on Restart.",
	  NULL, "gl", { { "console", "Console" }, { "smooth", "Smooth" }, { "sharp", "Sharp" }, { NULL, NULL } }, "console" },
	{ "armn64_gl_msaa", "Anti-Aliasing", NULL,
	  "Smooths edges and closes hairline gaps between triangles. Costs GPU time. Applies on Restart.",
	  NULL, "gl", { { "off", "Off" }, { "2", "2x" }, { "4", "4x" }, { "8", "8x" }, { NULL, NULL } }, "off" },
	{ "armn64_gl_seal", "Seal Edges", NULL,
	  "Opaque triangles reach half a pixel further, closing cracks the background shows through. Applies on Restart.",
	  NULL, "gl", { { "disabled", "Off" }, { "enabled", "On" }, { NULL, NULL } }, "disabled" },
	{ "armn64_gl_shaders", "Shader Accuracy", NULL,
	  "Fast: simpler shaders for weak GPUs; colors and textures can differ slightly from the console. Applies on Restart.",
	  NULL, "gl", { { "accurate", "Accurate" }, { "fast", "Fast" }, { NULL, NULL } }, "accurate" },
	{ "armn64_gl_fb_effects", "Framebuffer Effects", NULL,
	  "Fast: effects drawn from the picture (blurs, screen copies) use the last one in memory. Faster on some devices; may lag or glitch.",
	  NULL, "gl", { { "exact", "Exact" }, { "fast", "Fast" }, { NULL, NULL } }, "exact" },
	{ "armn64_gl_precompile", "Shader Precompile", NULL,
	  "Builds the shaders a game used before as it starts, with a progress bar, for fewer stutters while playing.",
	  NULL, "gl", { { "disabled", "Off" }, { "enabled", "On" }, { NULL, NULL } }, "disabled" },
	{ "armn64_stick_gate", "Octagonal Stick Gate", NULL,
	  "Shapes the stick's range like the N64 controller's octagonal gate instead of a circle.",
	  NULL, "input", { { "disabled", "Off" }, { "enabled", "On" }, { NULL, NULL } }, "disabled" },
	{ "armn64_stick_deadzone", "Stick Deadzone", NULL,
	  "Ignores the stick this close to its middle, for sticks that drift; past it the full range still reaches the full tilt.",
	  NULL, "input", { { "0", "Off" }, { "5", "5%" }, { "10", "10%" }, { "15", "15%" }, { "20", "20%" }, { "25", "25%" }, { "30", "30%" }, { NULL, NULL } }, "0" },
	{ "armn64_stick_sensitivity", "Stick Sensitivity", NULL,
	  "Scales the stick, for sticks that reach further or less far than the N64's. Above 1x reaches the full tilt sooner.",
	  NULL, "input", { { "50", "0.5x" }, { "60", "0.6x" }, { "70", "0.7x" }, { "80", "0.8x" }, { "90", "0.9x" }, { "100", "1.0x" }, { "110", "1.1x" }, { "120", "1.2x" }, { "130", "1.3x" }, { "140", "1.4x" }, { "150", "1.5x" }, { "160", "1.6x" }, { "170", "1.7x" }, { "180", "1.8x" }, { "190", "1.9x" }, { "200", "2.0x" }, { NULL, NULL } }, "100" },
	{ "armn64_face_c", "C Buttons on Face Buttons", NULL,
	  "Holding R2, the face buttons press the C buttons in their places. Not with the stick swaps that use R2 or with Dual Controllers.",
	  NULL, "input", { { "disabled", "Off" }, { "enabled", "R2 + Face Buttons" }, { NULL, NULL } }, "enabled" },
	{ "armn64_control_swap", "Control Swap", NULL,
	  "D-Pad / Left Stick: swaps the two (L1 is Z). Both Sticks: either stick moves the stick. Left / Right Stick: swaps the sticks (R2 is Z).",
	  NULL, "input", { { "disabled", "Off" }, { "dpad", "D-Pad / Left Stick" }, { "both", "Both Sticks" }, { "sticks", "Left / Right Stick" }, { NULL, NULL } }, "disabled" },
	{ "armn64_swap_hotkeys", "Control Swap Hotkeys", NULL,
	  "Player 1: Start or Select + D-pad switches D-Pad / Left Stick, + R3 Left / Right Stick; L3 + R3 switches Both Sticks.",
	  NULL, "input", { { "start", "Start + D-Pad / R3, L3 + R3" }, { "select", "Select + D-Pad / R3, L3 + R3" }, { "disabled", "Off" }, { NULL, NULL } }, "start" },
	{ "armn64_dual_controller", "Dual Controllers", NULL,
	  "Player 1's controller as two N64 controllers, one per hand, for two-controller game modes. Players 2 and 3 become 3 and 4.",
	  NULL, "input", { { "disabled", "Off" }, { "enabled", "On" }, { NULL, NULL } }, "disabled" },
	{ "armn64_rumble", "Controller Rumble", NULL, "Shakes your controller when a player's Rumble Pak would (set the Pak under Controller Paks).",
	  NULL, "input", { { "disabled", "Off" }, { "enabled", "On" }, { NULL, NULL } }, "disabled" },
	{ "armn64_pak1", "Player 1 Pak", NULL,
	  "What is in controller 1's slot. Transfer Pak (on Restart): the Game Boy ROM in system/armN64/TransferPak1.",
	  NULL, "paks", { { "memory", "Controller Pak" }, { "rumble", "Rumble Pak" }, { "transfer", "Transfer Pak" }, { "none", "None" }, { NULL, NULL } }, "memory" },
	{ "armn64_pak2", "Player 2 Pak", NULL,
	  "What is in controller 2's slot. Transfer Pak (on Restart): the Game Boy ROM in system/armN64/TransferPak2.",
	  NULL, "paks", { { "memory", "Controller Pak" }, { "rumble", "Rumble Pak" }, { "transfer", "Transfer Pak" }, { "none", "None" }, { NULL, NULL } }, "memory" },
	{ "armn64_pak3", "Player 3 Pak", NULL,
	  "What is in controller 3's slot. Transfer Pak (on Restart): the Game Boy ROM in system/armN64/TransferPak3.",
	  NULL, "paks", { { "memory", "Controller Pak" }, { "rumble", "Rumble Pak" }, { "transfer", "Transfer Pak" }, { "none", "None" }, { NULL, NULL } }, "memory" },
	{ "armn64_pak4", "Player 4 Pak", NULL,
	  "What is in controller 4's slot. Transfer Pak (on Restart): the Game Boy ROM in system/armN64/TransferPak4.",
	  NULL, "paks", { { "memory", "Controller Pak" }, { "rumble", "Rumble Pak" }, { "transfer", "Transfer Pak" }, { "none", "None" }, { NULL, NULL } }, "memory" },
	{ "armn64_pak1_card", "Player 1 Controller Pak", NULL,
	  "This Game's saves notes with the game; a Shared Pak is the same in every game, like a real one.",
	  NULL, "paks", { { "game", "This Game's" }, { "1", "Shared Pak 1" }, { "2", "Shared Pak 2" }, { "3", "Shared Pak 3" }, { "4", "Shared Pak 4" },
	                  { "5", "Shared Pak 5" }, { "6", "Shared Pak 6" }, { "7", "Shared Pak 7" }, { "8", "Shared Pak 8" }, { NULL, NULL } }, "game" },
	{ "armn64_pak2_card", "Player 2 Controller Pak", NULL,
	  "This Game's saves notes with the game; a Shared Pak is the same in every game, like a real one.",
	  NULL, "paks", { { "game", "This Game's" }, { "1", "Shared Pak 1" }, { "2", "Shared Pak 2" }, { "3", "Shared Pak 3" }, { "4", "Shared Pak 4" },
	                  { "5", "Shared Pak 5" }, { "6", "Shared Pak 6" }, { "7", "Shared Pak 7" }, { "8", "Shared Pak 8" }, { NULL, NULL } }, "game" },
	{ "armn64_pak3_card", "Player 3 Controller Pak", NULL,
	  "This Game's saves notes with the game; a Shared Pak is the same in every game, like a real one.",
	  NULL, "paks", { { "game", "This Game's" }, { "1", "Shared Pak 1" }, { "2", "Shared Pak 2" }, { "3", "Shared Pak 3" }, { "4", "Shared Pak 4" },
	                  { "5", "Shared Pak 5" }, { "6", "Shared Pak 6" }, { "7", "Shared Pak 7" }, { "8", "Shared Pak 8" }, { NULL, NULL } }, "game" },
	{ "armn64_pak4_card", "Player 4 Controller Pak", NULL,
	  "This Game's saves notes with the game; a Shared Pak is the same in every game, like a real one.",
	  NULL, "paks", { { "game", "This Game's" }, { "1", "Shared Pak 1" }, { "2", "Shared Pak 2" }, { "3", "Shared Pak 3" }, { "4", "Shared Pak 4" },
	                  { "5", "Shared Pak 5" }, { "6", "Shared Pak 6" }, { "7", "Shared Pak 7" }, { "8", "Shared Pak 8" }, { NULL, NULL } }, "game" },
	{ "armn64_pak_manager", "Controller Pak Manager", NULL,
	  "Open pauses the game to copy, delete, export, import and format the paks' notes. B closes it.",
	  NULL, "paks", { { "closed", "Closed" }, { "open", "Open" }, { NULL, NULL } }, "closed" },
	{ "armn64_gfx_rsp", "Graphics Microcode", NULL,
	  "Library: the core's own copy of the graphics code, fast. Game's Own: the game's code on the RSP, exact but slower. Applies on Restart.",
	  NULL, "advanced", { { "library", "Library" }, { "rsp", "Game's Own" }, { NULL, NULL } }, "library" },
	{ "armn64_hle_timing", "Graphics Task Timing", NULL,
	  "With Library graphics. Short: graphics tasks finish sooner; some games then draw more frames than on a console. Applies on Restart.",
	  NULL, "advanced", { { "console", "Console" }, { "short", "Short" }, { NULL, NULL } }, "console" },
	{ "armn64_rsp_thread", "Threaded RSP", NULL,
	  "With Fast. Runs the RSP on a core of its own: faster for some games. Applies on Restart.",
	  NULL, "advanced", { { "disabled", "Off" }, { "enabled", "On" }, { NULL, NULL } }, "disabled" },
	{ "armn64_rdp_threads", "Renderer Threads", NULL,
	  "With the CPU renderer. Cores drawing beside the emulation; Auto takes up to 3. Applies on Restart.",
	  NULL, "advanced", { { "auto", "Auto" }, { "0", "None" }, { "1", "1" }, { "2", "2" }, { "3", "3" }, { NULL, NULL } }, "auto" },
	{ NULL, NULL, NULL, NULL, NULL, NULL, { { NULL, NULL } }, NULL },
};

static struct retro_core_options_v2 coreOptions = { optionCategories, optionDefinitions };

static void logMessage(enum retro_log_level level, const char* message) {
	if (logPrintf) {
		logPrintf(level, "[armN64] %s\n", message);
	}
}

static const char* accuracy(void) {
	const char* value = optionValue(environment, "armn64_accuracy");
	return value && (!strcmp(value, "fle") || !strcmp(value, "mle") || !strcmp(value, "lle")) ? value : "hle";
}

// HLE, or FLE built on it.
static bool highLevel(void) {
	return !strcmp(accuracy(), "hle") || !strcmp(accuracy(), "fle");
}

// The renderer the option picks (GL for a value from before Auto went).
static const char* renderer(void) {
	const char* value = optionValue(environment, "armn64_renderer");
	if (value && (!strcmp(value, "gpu") || !strcmp(value, "cpu"))) {
		return value;
	}
	return "gl";
}

static const char* overclock(void) {
	const char* value = optionValue(environment, "armn64_overclock");
	return value ? value : "1";
}

// Hides the options that do nothing with the others' values. The frontend calls it when an option
// changes; returns whether that changed which are shown.
static bool showOptions(void) {
	static bool shownSet, shown[24];
	bool const hle = highLevel(), gl = !strcmp(renderer(), "gl"), gpu = !strcmp(renderer(), "gpu");
	bool const custom = optionIs(environment, "armn64_crop_overscan", "custom");
	struct {
		const char* key;
		bool shown;
	} const options[] = {
		{ "armn64_resolution", gl || gpu },
		{ "armn64_crop_top", custom },
		{ "armn64_crop_bottom", custom },
		{ "armn64_crop_left", custom },
		{ "armn64_crop_right", custom },
		{ "armn64_gl_fb_effects", gl },
		{ "armn64_gl_filter", gl },
		{ "armn64_gl_precompile", gl },
		{ "armn64_gl_shaders", gl },
		{ "armn64_gl_msaa", gl },
		{ "armn64_gl_seal", gl },
		{ "armn64_vi_filters", !gl },
		{ "armn64_gfx_rsp", true },
		{ "armn64_hle_timing", !optionIs(environment, "armn64_gfx_rsp", "rsp") },
		{ "armn64_rsp_thread", hle },
		{ "armn64_rdp_threads", !strcmp(renderer(), "cpu") },
		{ "armn64_pak1_card", optionIs(environment, "armn64_pak1", "memory") || !optionValue(environment, "armn64_pak1") },
		{ "armn64_pak2_card", optionIs(environment, "armn64_pak2", "memory") || !optionValue(environment, "armn64_pak2") },
		{ "armn64_pak3_card", optionIs(environment, "armn64_pak3", "memory") || !optionValue(environment, "armn64_pak3") },
		{ "armn64_pak4_card", optionIs(environment, "armn64_pak4", "memory") || !optionValue(environment, "armn64_pak4") },
	};
	bool changed = false;
	for (size_t i = 0; i < sizeof(options) / sizeof(options[0]); ++i) {
		if (!shownSet || shown[i] != options[i].shown) {
			shown[i] = options[i].shown;
			struct retro_core_option_display display = { options[i].key, options[i].shown };
			environment(RETRO_ENVIRONMENT_SET_CORE_OPTIONS_DISPLAY, &display);
			changed = true;
		}
	}
	shownSet = true;
	return changed;
}

// The GL renderer's scale: the option's, or with Auto the most times a 320x240 picture fits the
// connected display (from DRM's modes), up to 8.
static int glScale(void) {
	const char* value = optionValue(environment, "armn64_resolution");
	if (value && *value >= '1' && *value <= '8') {
		return *value - '0';
	}
	static int display;
	if (!display) {
		display = 1;
#ifndef __ANDROID__
		// Android's apps can't read DRM's state (and its glob comes only with API 28).
		glob_t found;
		if (glob("/sys/class/drm/card*-*/status", 0, NULL, &found) == 0) {
			for (size_t i = 0; i < found.gl_pathc; ++i) {
				char path[512], line[64] = "";
				FILE* f = fopen(found.gl_pathv[i], "r");
				bool connected = f && fgets(line, sizeof(line), f) && !strncmp(line, "connected", 9);
				if (f) {
					fclose(f);
				}
				snprintf(path, sizeof(path), "%.*smodes", (int) (strlen(found.gl_pathv[i]) - 6), found.gl_pathv[i]);
				f = connected ? fopen(path, "r") : NULL;
				int w = 0, h = 0;
				if (f && fscanf(f, "%dx%d", &w, &h) == 2) {
					int k = w / 320 < h / 240 ? w / 320 : h / 240;
					display = k > display ? k : display;
				}
				if (f) {
					fclose(f);
				}
			}
			globfree(&found);
		}
#endif
		display = display > 8 ? 8 : display;
	}
	return display;
}

// The emulator's settings, from the core's options; the recompilers are always on.
static const char* settings(const char* name) {
	static char threads[4], scale[2];
	if (!strcmp(name, "N64_ACCURACY")) {
		return accuracy();
	}
	if (!strcmp(name, "N64_EXPANSION")) {
		return optionIs(environment, "armn64_expansion", "disabled") ? "0" : "1";
	}
	if (!strcmp(name, "N64_OVERCLOCK")) {
		return overclock();
	}
	if (!strcmp(name, "N64_GPU_SCALE")) {
		// GPU: Native unless a resolution is chosen (Auto keeps it exact).
		const char* value = optionValue(environment, "armn64_resolution");
		scale[0] = !strcmp(renderer(), "gpu") && value && *value >= '1' && *value <= '8' ? *value : '1';
		return scale;
	}
	if (!strcmp(name, "N64_GL_SCALE")) {
		scale[0] = (char) ('0' + glScale());
		return scale;
	}
	// Compiled GPU programs in system/armN64/cache.
	if (!strcmp(name, "N64_SHADER_CACHE")) {
		static char cache[4096];
		const char* system = NULL;
		if (!environment(RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY, &system) || !system) {
			return NULL;
		}
		snprintf(cache, sizeof(cache), "%s/armN64", system);
		mkdir(cache, 0755);
		snprintf(cache, sizeof(cache), "%s/armN64/cache", system);
		mkdir(cache, 0755);
		return cache;
	}
	if (!strcmp(name, "N64_SLICE")) {
		return highLevel() ? "8192" : "1024";
	}
	if (!strcmp(name, "N64_JIT") || !strcmp(name, "N64_RSPJIT")) {
		return "1";
	}
	if (!strcmp(name, "N64_RDP_GL") || !strcmp(name, "N64_RDP_GPU")) {
		return hwWanted && !strcmp(renderer(), strcmp(name, "N64_RDP_GL") ? "gpu" : "gl") ? "1" : "0";
	}
	// The GPU RDP's pictures stay on the GPU (the VI in its context); beside the renderer threads the
	// VI runs in the frontend's context.
	if (!strcmp(name, "N64_VI_GPU")) {
		return (hwWanted && !strcmp(renderer(), "gpu")) || viInContext ? "1" : "0";
	}
	if (!strcmp(name, "N64_GFX_RSP")) {
		return optionIs(environment, "armn64_gfx_rsp", "rsp") ? "1" : "0";
	}
	if (!strcmp(name, "N64_HLE_TIMING")) {
		return optionIs(environment, "armn64_hle_timing", "short") ? "short" : "console";
	}
	if (!strcmp(name, "N64_RSP_THREAD")) {
		return optionIs(environment, "armn64_rsp_thread", "enabled") ? "1" : "0";
	}
	if (!strcmp(name, "N64_GL_FB_EFFECTS")) {
		return optionIs(environment, "armn64_gl_fb_effects", "fast") ? "fast" : "exact";
	}
	if (!strcmp(name, "N64_GL_MSAA")) {
		const char* v = optionValue(environment, "armn64_gl_msaa");
		return v && strcmp(v, "off") ? v : "0";
	}
	if (!strcmp(name, "N64_WIDESCREEN")) {
		const char* v = optionValue(environment, "armn64_widescreen");
		return v && strcmp(v, "disabled") ? v : "0";
	}
	if (!strcmp(name, "N64_GL_FILTER")) {
		const char* v = optionValue(environment, "armn64_gl_filter");
		return v ? v : "console";
	}
	if (!strcmp(name, "N64_GL_SEAL")) {
		return optionIs(environment, "armn64_gl_seal", "enabled") ? "1" : "0";
	}
	if (!strcmp(name, "N64_GL_SHADERS")) {
		return optionIs(environment, "armn64_gl_shaders", "fast") ? "fast" : "accurate";
	}
	if (!strcmp(name, "N64_GL_PRECOMPILE")) {
		return optionIs(environment, "armn64_gl_precompile", "enabled") ? "2" : "0";
	}
	if (!strcmp(name, "N64_FRAMESKIP")) {
		return optionIs(environment, "armn64_frameskip", "auto") ? "2" : optionIs(environment, "armn64_frameskip", "enabled") ? "1" : "0";
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
	if (optionsGive(callback, &coreOptions)) {
		struct retro_core_options_update_display_callback display = { showOptions };
		callback(RETRO_ENVIRONMENT_SET_CORE_OPTIONS_UPDATE_DISPLAY_CALLBACK, &display);
		showOptions();
	}
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
	*info = (struct retro_system_info) { "armN64", "1.1", "n64|z64|v64|ndd", false, false };
}

void retro_get_system_av_info(struct retro_system_av_info* info) {
	// The GL renderer's pictures are up to its scale times the largest.
	maxScale = (unsigned) glScale();
	info->geometry = (struct retro_game_geometry) { (unsigned) pictureWidth, (unsigned) pictureHeight, 640 * maxScale, 480 * maxScale, pictureAspect };
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
	paksExport(machine, saveMemory + FILE_PAKS);
	swapWords(saveMemory + FILE_SRAM, machine->sram, sizeof(machine->sram));
	swapWords(saveMemory + FILE_FLASH, machine->flash, sizeof(machine->flash));
}

static void saveImport(void) {
	memcpy(machine->eeprom, saveMemory, sizeof(machine->eeprom));
	paksImport(machine, saveMemory + FILE_PAKS);
	swapWords(machine->sram, saveMemory + FILE_SRAM, sizeof(machine->sram));
	swapWords(machine->flash, saveMemory + FILE_FLASH, sizeof(machine->flash));
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
// The Game Boy cartridges in Transfer Paks: the first .gb or .gbc in system/armN64/TransferPak<port>,
// and its save RAM as the .sav of the same name there.
static char gbSavePath[4][4096];

static void transferPaksLoad(struct N64* n) {
	const char* system = NULL;
	if (!environment(RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY, &system) || !system) {
		return;
	}
	for (int port = 0; port < 4; ++port) {
		char key[16], dir[4096];
		snprintf(key, sizeof(key), "armn64_pak%d", port + 1);
		gbSavePath[port][0] = 0;
		if (!optionIs(environment, key, "transfer")) {
			continue;
		}
		snprintf(dir, sizeof(dir), "%s/armN64/TransferPak%d", system, port + 1);
		DIR* d = opendir(dir);
		struct dirent* e;
		while (d && (e = readdir(d))) {
			size_t len = strlen(e->d_name);
			if ((len > 3 && !strcasecmp(e->d_name + len - 3, ".gb")) || (len > 4 && !strcasecmp(e->d_name + len - 4, ".gbc"))) {
				char path[4096];
				int const stem = (int) (strrchr(e->d_name, '.') - e->d_name);
				if (snprintf(path, sizeof(path), "%s/%s", dir, e->d_name) >= (int) sizeof(path) ||
				    snprintf(gbSavePath[port], sizeof(gbSavePath[port]), "%s/%.*s.sav", dir, stem, e->d_name) >= (int) sizeof(gbSavePath[port])) {
					gbSavePath[port][0] = 0;
					break;
				}
				size_t size;
				uint8_t* rom = readFile(path, &size);
				if (!rom || !gbCartLoad(&n->tpak[port].cart, rom, size)) {
					gbSavePath[port][0] = 0;
				} else {
					uint8_t* sav = readFile(gbSavePath[port], &size);
					if (sav) {
						memcpy(n->tpak[port].cart.ram, sav, size < n->tpak[port].cart.ramSize ? size : n->tpak[port].cart.ramSize);
					}
					free(sav);
				}
				free(rom);
				break;
			}
		}
		if (d) {
			closedir(d);
		}
	}
}

static void transferPaksSave(void) {
	for (int port = 0; machine && port < 4; ++port) {
		struct GbCart* c = &machine->tpak[port].cart;
		if (!c->ramDirty || !gbSavePath[port][0]) {
			continue;
		}
		FILE* f = fopen(gbSavePath[port], "wb");
		if (f && fwrite(c->ram, 1, c->ramSize, f) == c->ramSize) {
			c->ramDirty = false;
		}
		if (f) {
			fclose(f);
		}
	}
}

// A 64DD disk's changes, beside the save files as <disk name>.ndd.sav.
static char diskSavePath[4096];

static void diskSave(void) {
	size_t size;
	uint8_t* changes = disk && machine && !machine->dd.diskSaved ? ddChanges(machine, &size) : NULL;
	if (!changes) {
		return;
	}
	FILE* f = fopen(diskSavePath, "wb");
	if (f && fwrite(changes, 1, size, f) == size) {
		machine->dd.diskSaved = true;
	}
	if (f) {
		fclose(f);
	}
	free(changes);
}

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
	size_t size;
	uint8_t* changes = disk && diskSavePath[0] ? readFile(diskSavePath, &size) : NULL;
	if (changes && !ddApplyChanges(n, changes, size)) {
		logMessage(RETRO_LOG_WARN, "the disk's saved changes don't fit this disk");
	}
	free(changes);
	transferPaksLoad(n);
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
	presentWidth = 640, presentHeight = 480;
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

static void setGeometry(int width, int height);

// The columns and rows hidden at each edge of a picture made at k times its size (Crop Overscan).
struct Crop {
	int left, top, right, bottom;
};

// The pixels hidden at each side of a picture made at k times its size (Crop Overscan): a thirtieth
// of the picture each side, or each side's own amount with Custom.
static struct Crop overscan(int width, int height, int k) {
	if (optionIs(environment, "armn64_crop_overscan", "disabled")) {
		return (struct Crop) { 0, 0, 0, 0 };
	}
	if (!optionIs(environment, "armn64_crop_overscan", "custom")) {
		int const x = width / k / 30 * k, y = height / k / 30 * k;
		return (struct Crop) { x, y, x, y };
	}
	static const char* const keys[4] = { "armn64_crop_left", "armn64_crop_top", "armn64_crop_right", "armn64_crop_bottom" };
	int side[4];
	for (int i = 0; i < 4; ++i) {
		const char* v = optionValue(environment, keys[i]);
		side[i] = (v ? atoi(v) : 8) * k;
	}
	// At least a pixel of the picture stays.
	side[2] = side[0] + side[2] >= width ? 0 : side[2], side[0] = side[0] >= width ? 0 : side[0];
	side[3] = side[1] + side[3] >= height ? 0 : side[3], side[1] = side[1] >= height ? 0 : side[1];
	return (struct Crop) { side[0], side[1], side[2], side[3] };
}

// The VI's picture into the frontend's framebuffer, upside down as GL keeps it: from the GL buffer
// the GPU made it in, else from picture.
static void present(int width, int height, unsigned buffer) {
	gl.Disable(GL_SCISSOR_TEST);
	gl.BindTexture(GL_TEXTURE_2D, presentTexture);
	if (width > presentWidth || height > presentHeight) {
		presentWidth = width > presentWidth ? width : presentWidth, presentHeight = height > presentHeight ? height : presentHeight;
		gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, presentWidth, presentHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}
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
	int const k = n64PictureScale(machine);
	struct Crop const c = overscan(width, height, k);
	int const w = width - c.left - c.right, h = height - c.top - c.bottom;
	setGeometry(w, h);
	gl.BindFramebuffer(GL_READ_FRAMEBUFFER, presentFramebuffer);
	gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, (unsigned) hw.get_current_framebuffer());
	gl.BlitFramebuffer(c.left, c.top, c.left + w, c.top + h, 0, h, w, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glrForget(machine);
	videoRefresh(RETRO_HW_FRAME_BUFFER_VALID, (unsigned) w, (unsigned) h, 0);
}

// The GL renderer's picture from its framebuffer into the frontend's, upside down as GL keeps it, at
// the renderer's scale (width and height are already scaled).
static void presentGl(unsigned framebuffer, int x, int y, int width, int height) {
	int const k = machine->glrScale;
	gl.Disable(GL_SCISSOR_TEST);
	struct Crop const c = overscan(width, height, k);
	int const w = width - c.left - c.right, h = height - c.top - c.bottom;
	setGeometry(w, h);
	gl.BindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
	gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, (unsigned) hw.get_current_framebuffer());
	gl.BlitFramebuffer(x * k + c.left, y * k + c.top, x * k + c.left + w, y * k + c.top + h, 0, h, w, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
	glrForget(machine);
	videoRefresh(RETRO_HW_FRAME_BUFFER_VALID, (unsigned) w, (unsigned) h, 0);
}

// A notice on screen for a few seconds.
static void notify(const char* text) {
	struct retro_message_ext ext = { text, 5000, 1, RETRO_LOG_WARN, RETRO_MESSAGE_TARGET_ALL, RETRO_MESSAGE_TYPE_NOTIFICATION, -1 };
	if (!environment(RETRO_ENVIRONMENT_SET_MESSAGE_EXT, &ext)) {
		struct retro_message message = { text, 300 };
		environment(RETRO_ENVIRONMENT_SET_MESSAGE, &message);
	}
	logMessage(RETRO_LOG_WARN, text);
}

// Frame Skip Auto: whether the frontend's audio buffer is running low (a third full or less), and
// whether a longer buffer has been asked for.
static bool audioBehind, latencySet;

static void audioBufferStatus(bool active, unsigned occupancy, bool underrunLikely) {
	audioBehind = active && (underrunLikely || occupancy <= 33);
}

// Shader Precompile's progress, as a bar where the frontend draws one.
static void showProgress(int done, int total) {
	char text[64];
	int percent = total ? done * 100 / total : 100;
	snprintf(text, sizeof(text), "Compiling shaders: %d%%", percent);
	struct retro_message_ext ext = { text, 1000, 2, RETRO_LOG_INFO, RETRO_MESSAGE_TARGET_OSD, RETRO_MESSAGE_TYPE_PROGRESS, (int8_t) percent };
	if (!environment(RETRO_ENVIRONMENT_SET_MESSAGE_EXT, &ext)) {
		struct retro_message message = { text, 60 };
		environment(RETRO_ENVIRONMENT_SET_MESSAGE, &message);
	}
}

// What HLE gave up for this game, said once: a graphics microcode it doesn't know (run on the RSP).
static bool toldRsp;

static void tellFallbacks(void) {
	char text[160];
	if (!toldRsp && machine->accuracy == ACCURACY_HLE && machine->stats.gfxRsp) {
		snprintf(text, sizeof(text), "HLE doesn't know this graphics microcode (%s): it runs on the RSP, slower.", machine->stats.gfxRspUcode);
		notify(text);
		toldRsp = true;
	}
}

// Each port's controls, and what can be plugged in.
static void describeInput(void) {
	static const struct {
		unsigned device, index, id;
		const char* name;
	} controls[] = {
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_B, "A" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_Y, "B" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_A, "C-Down" },
		{ RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_X, "C-Left" },
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
	showOptions();
	hwWanted = strcmp(renderer(), "cpu") != 0;
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
		const char* saves = NULL;
		const char* name = strrchr(info->path, '/') ? strrchr(info->path, '/') + 1 : info->path;
		if (!environment(RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY, &saves) || !saves) {
			saves = ".";
		}
		snprintf(diskSavePath, sizeof(diskSavePath), "%s/%s.sav", saves, name);
	}
	const char* saveDir = NULL;
	paksInit(environment(RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY, &saveDir) ? saveDir : NULL);
	machine = startMachine();
	if (!machine) {
		logMessage(RETRO_LOG_ERROR, "this game can't be loaded");
		return false;
	}
	pal = machine->fieldCycles == hwCycles(machine, CPU_HZ / 50);
	saveExport();
	saveGiven = false;
	toldRsp = false;
	struct retro_rumble_interface rumble;
	setRumble = environment(RETRO_ENVIRONMENT_GET_RUMBLE_INTERFACE, &rumble) ? rumble.set_rumble_state : NULL;
	memset(rumbling, 0, sizeof(rumbling));
	audioPhase = 0;
	memset(audioLast, 0, sizeof(audioLast));
	describeInput();
	struct retro_audio_buffer_status_callback bufferStatus = { audioBufferStatus };
	audioBehind = latencySet = false;
	environment(RETRO_ENVIRONMENT_SET_AUDIO_BUFFER_STATUS_CALLBACK, machine->skip.automatic ? &bufferStatus : NULL);
	bool achievements = true;
	environment(RETRO_ENVIRONMENT_SET_SUPPORT_ACHIEVEMENTS, &achievements);
	return true;
}

bool retro_load_game_special(unsigned type, const struct retro_game_info* info, size_t count) {
	return false;
}

void retro_unload_game(void) {
	if (machine) {
		paksSync(machine);
	}
	diskSave();
	transferPaksSave();
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
	diskSave();
	transferPaksSave();
	n64Destroy(machine);
	glesYield();
	machine = startMachine();
	if (machine) {
		saveImport();
	}
}

// The stick as the controller reports it from ix, iy (-32768 to 32767) times the sensitivity: about 80
// each way; or with the octagonal gate, the input's circle stretched to the gate's octagon (85
// across, corners at 69, 69) and kept inside it.
static void stick(int ix, int iy, bool gate, int* x, int* y) {
	const char* sensitivity = optionValue(environment, "armn64_stick_sensitivity");
	double const k = sensitivity ? atoi(sensitivity) / 100.0 : 1;
	double dx = ix / 32768.0, dy = iy / 32768.0;
	// The deadzone, round, and the rest of the range stretched to the full tilt.
	const char* deadzone = optionValue(environment, "armn64_stick_deadzone");
	double const dz = deadzone ? atoi(deadzone) / 100.0 : 0, length = sqrt(dx * dx + dy * dy);
	if (dz > 0) {
		double const scale = length <= dz ? 0 : (length - dz) / (1 - dz) / length;
		dx *= scale, dy *= scale;
	}
	dx *= k > 0 ? k : 1, dy *= k > 0 ? k : 1;
	if (!gate) {
		dx = dx < -1 ? -1 : dx > 1 ? 1 : dx;
		dy = dy < -1 ? -1 : dy > 1 ? 1 : dy;
		*x = (int) (dx * 80), *y = (int) (dy * 80);
		return;
	}
	double r = sqrt(dx * dx + dy * dy);
	if (r < 1e-6) {
		*x = *y = 0;
		return;
	}
	dx /= r, dy /= r;
	r = r > 1 ? 1 : r;
	// How far the octagon's edge is in this direction: the edge from (85, 0) to (69, 69) is
	// x + y * 16 / 69 = 85.
	double const ax = fabs(dx), ay = fabs(dy);
	double const edge = 85 / (ax >= ay ? ax + ay * 16 / 69 : ay + ax * 16 / 69);
	*x = (int) lround(dx * r * edge), *y = (int) lround(dy * r * edge);
}

// Control Swap's hotkeys on player 1's controller. With Start or Select + D-pad, the D-pad is held back
// from the game while that is held, and Start too: a short press alone reaches it as it's let go (for
// two fields), a long one once it's held that long.
enum { START_LONG = 20, START_PULSE = 2 };
static bool startCombo, selectCombo, selectHeld;
static int startFields, startPulse;
static bool startUsed;

static void swapHotkeys(void) {
	startCombo = optionIs(environment, "armn64_swap_hotkeys", "start") || !optionValue(environment, "armn64_swap_hotkeys");
	selectCombo = optionIs(environment, "armn64_swap_hotkeys", "select");
	bool const start = startCombo && inputState(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_START);
	selectHeld = selectCombo && inputState(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_SELECT);
	startPulse -= startPulse > 0;
	if (start) {
		++startFields;
	} else {
		if (startFields && startFields <= START_LONG && !startUsed) {
			startPulse = START_PULSE;
		}
		startFields = 0;
		startUsed = false;
	}
	bool dpad = false;
	for (unsigned id = RETRO_DEVICE_ID_JOYPAD_UP; id <= RETRO_DEVICE_ID_JOYPAD_RIGHT; ++id) {
		dpad |= inputState(0, RETRO_DEVICE_JOYPAD, 0, id) != 0;
	}
	bool const on = startCombo || selectCombo;
	static const struct {
		const char* value, * message;
	} swaps[3] = {
		{ "dpad", "Control Swap: D-Pad / Left Stick" },
		{ "both", "Control Swap: Both Sticks" },
		{ "sticks", "Control Swap: Left / Right Stick" },
	};
	bool const l3 = inputState(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_L3), r3 = inputState(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_R3);
	bool const down[3] = {
		(start || selectHeld) && dpad,
		on && l3 && r3,
		(start || selectHeld) && r3 && !l3,
	};
	static bool held[3];
	for (int i = 0; i < 3; ++i) {
		if (down[i] && !held[i]) {
			bool const now = !optionIs(environment, "armn64_control_swap", swaps[i].value);
			struct retro_variable set = { "armn64_control_swap", now ? swaps[i].value : "disabled" };
			environment(RETRO_ENVIRONMENT_SET_VARIABLE, &set);
			notify(now ? swaps[i].message : "Control Swap: Off");
		}
		held[i] = down[i];
	}
	startUsed |= (down[0] || down[2]) && start;
}

// A button as the game sees it: player 1's Start and D-pad held back while Start or Select + D-pad is
// the hotkey.
static bool pressed(unsigned port, unsigned id) {
	if (port == 0 && selectHeld && id >= RETRO_DEVICE_ID_JOYPAD_UP && id <= RETRO_DEVICE_ID_JOYPAD_RIGHT) {
		return false;
	}
	if (port == 0 && startCombo) {
		bool const start = startFields > 0;
		if (id == RETRO_DEVICE_ID_JOYPAD_START) {
			return startPulse > 0 || (start && startFields > START_LONG && !startUsed);
		}
		if (start && id >= RETRO_DEVICE_ID_JOYPAD_UP && id <= RETRO_DEVICE_ID_JOYPAD_RIGHT) {
			return false;
		}
	}
	return inputState(port, RETRO_DEVICE_JOYPAD, 0, id) != 0;
}

enum { HAND_BOTH, HAND_LEFT, HAND_RIGHT };

// A controller as the PIF reads it from a frontend port: the buttons in the top half, then the
// stick's X and Y. With Dual Controllers, the left or right hand's half of player 1's.
static uint32_t readController(unsigned port, int hand) {
	bool const swapDpad = optionIs(environment, "armn64_control_swap", "dpad"), swapShoulders = swapDpad;
	// Left / Right Stick (not with Dual Controllers, where each hand has its own): the right stick as
	// the control stick, the left as the C buttons, and Z on R2, beside it.
	bool const swapSticks = hand == HAND_BOTH && optionIs(environment, "armn64_control_swap", "sticks");
	unsigned const mainStick = swapSticks ? RETRO_DEVICE_INDEX_ANALOG_RIGHT : RETRO_DEVICE_INDEX_ANALOG_LEFT;
	unsigned const cStick = swapSticks ? RETRO_DEVICE_INDEX_ANALOG_LEFT : RETRO_DEVICE_INDEX_ANALOG_RIGHT;
	// Both Sticks: the two added together as the control stick, and no C buttons from either.
	bool const bothSticks = hand == HAND_BOTH && optionIs(environment, "armn64_control_swap", "both");
	bool const gate = optionIs(environment, "armn64_stick_gate", "enabled");
	int const z = swapShoulders ? RETRO_DEVICE_ID_JOYPAD_L : swapSticks ? RETRO_DEVICE_ID_JOYPAD_R2 : RETRO_DEVICE_ID_JOYPAD_L2;
	int const l = swapShoulders ? RETRO_DEVICE_ID_JOYPAD_L2 : RETRO_DEVICE_ID_JOYPAD_L;
	#define PRESSED(id) pressed(port, (id))
	#define AXIS(index, id) inputState(port, RETRO_DEVICE_ANALOG, (index), (id))
	uint32_t state = 0;
	int x, y;
	if (hand == HAND_RIGHT) {
		// The right hand's controller: the right stick, R2 as Z, R1, and A and X as A and B.
		state |= (PRESSED(RETRO_DEVICE_ID_JOYPAD_A) ? 1u << 31 : 0) | (PRESSED(RETRO_DEVICE_ID_JOYPAD_X) ? 1u << 30 : 0) |
		         (PRESSED(RETRO_DEVICE_ID_JOYPAD_R2) ? 1u << 29 : 0) | (PRESSED(RETRO_DEVICE_ID_JOYPAD_R) ? 1u << 20 : 0);
		stick(AXIS(RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X), AXIS(RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y), gate, &x, &y);
		return state | (uint32_t) (uint8_t) x << 8 | (uint8_t) -y;
	}
	state |= (PRESSED(RETRO_DEVICE_ID_JOYPAD_B) ? 1u << 31 : 0) | (PRESSED(RETRO_DEVICE_ID_JOYPAD_Y) ? 1u << 30 : 0) | (PRESSED(z) ? 1u << 29 : 0) |
	         (PRESSED(RETRO_DEVICE_ID_JOYPAD_START) ? 1u << 28 : 0) | (PRESSED(l) ? 1u << 21 : 0);
	if (hand == HAND_BOTH) {
		state |= PRESSED(RETRO_DEVICE_ID_JOYPAD_R) ? 1u << 20 : 0;
	}
	if (bothSticks && PRESSED(RETRO_DEVICE_ID_JOYPAD_R2)) {
		state |= 1u << 29;
	}
	if (hand == HAND_BOTH && !bothSticks) {
		// The C buttons from the other stick, pushed past halfway.
		int const cx = AXIS(cStick, RETRO_DEVICE_ID_ANALOG_X), cy = AXIS(cStick, RETRO_DEVICE_ID_ANALOG_Y);
		state |= (cy < -16384 ? 1u << 19 : 0) | (cy > 16384 ? 1u << 18 : 0) | (cx < -16384 ? 1u << 17 : 0) | (cx > 16384 ? 1u << 16 : 0);
	}
	// C Buttons on Face Buttons: with R2 held, the face buttons as the C buttons in their places, and
	// not A and B. Otherwise east and north are C-Down and C-Left.
	if (hand == HAND_BOTH && !swapSticks && !bothSticks && optionIs(environment, "armn64_face_c", "enabled") && PRESSED(RETRO_DEVICE_ID_JOYPAD_R2)) {
		state &= ~(3u << 30);
		state |= (PRESSED(RETRO_DEVICE_ID_JOYPAD_X) ? 1u << 19 : 0) | (PRESSED(RETRO_DEVICE_ID_JOYPAD_B) ? 1u << 18 : 0) |
		         (PRESSED(RETRO_DEVICE_ID_JOYPAD_Y) ? 1u << 17 : 0) | (PRESSED(RETRO_DEVICE_ID_JOYPAD_A) ? 1u << 16 : 0);
	} else if (hand == HAND_BOTH) {
		state |= (PRESSED(RETRO_DEVICE_ID_JOYPAD_A) ? 1u << 18 : 0) | (PRESSED(RETRO_DEVICE_ID_JOYPAD_X) ? 1u << 17 : 0);
	}
	// The D-pad and the stick, one as the other when swapped: the D-pad as the stick pushed all the
	// way, the stick as the D-pad pushed past halfway.
	bool const up = PRESSED(RETRO_DEVICE_ID_JOYPAD_UP), down = PRESSED(RETRO_DEVICE_ID_JOYPAD_DOWN);
	bool const left = PRESSED(RETRO_DEVICE_ID_JOYPAD_LEFT), right = PRESSED(RETRO_DEVICE_ID_JOYPAD_RIGHT);
	int lx = AXIS(mainStick, RETRO_DEVICE_ID_ANALOG_X), ly = AXIS(mainStick, RETRO_DEVICE_ID_ANALOG_Y);
	if (bothSticks) {
		lx += AXIS(RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X), ly += AXIS(RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y);
		lx = lx < -32768 ? -32768 : lx > 32767 ? 32767 : lx;
		ly = ly < -32768 ? -32768 : ly > 32767 ? 32767 : ly;
	}
	if (swapDpad) {
		state |= (ly < -16384 ? 1u << 27 : 0) | (ly > 16384 ? 1u << 26 : 0) | (lx < -16384 ? 1u << 25 : 0) | (lx > 16384 ? 1u << 24 : 0);
		stick((right - left) * 32767, (down - up) * 32767, gate, &x, &y);
	} else {
		state |= (up ? 1u << 27 : 0) | (down ? 1u << 26 : 0) | (left ? 1u << 25 : 0) | (right ? 1u << 24 : 0);
		stick(lx, ly, gate, &x, &y);
	}
	#undef PRESSED
	#undef AXIS
	return state | (uint32_t) (uint8_t) x << 8 | (uint8_t) -y;
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
	swapHotkeys();
	// With Dual Controllers, player 1 holds controllers 1 and 2, and the other players move up one.
	bool const dual = optionIs(environment, "armn64_dual_controller", "enabled");
	bool shake[4] = { false };
	for (unsigned port = 0; port < 4; ++port) {
		unsigned const from = !dual ? port : port < 2 ? 0 : port - 1;
		unsigned const device = devices[from] & RETRO_DEVICE_MASK;
		machine->plugged[port] = device == RETRO_DEVICE_MOUSE ? PLUG_MOUSE : device == RETRO_DEVICE_NONE ? PLUG_NONE : PLUG_CONTROLLER;
		char key[16];
		snprintf(key, sizeof(key), "armn64_pak%u", port + 1);
		machine->pak[port] = optionIs(environment, key, "rumble") ? PAK_RUMBLE : optionIs(environment, key, "transfer") ? PAK_TRANSFER :
		                     optionIs(environment, key, "none") ? PAK_NONE : PAK_MEMORY;
		int const hand = !dual || port > 1 ? HAND_BOTH : port ? HAND_RIGHT : HAND_LEFT;
		machine->pad[port] = machine->plugged[port] == PLUG_MOUSE ? readMouse(from) : machine->plugged[port] ? readController(from, hand) : 0;
		shake[from] |= machine->pak[port] == PAK_RUMBLE && machine->rumble[port] && optionIs(environment, "armn64_rumble", "enabled");
	}
	for (unsigned port = 0; port < 4; ++port) {
		if (setRumble && shake[port] != rumbling[port]) {
			setRumble(port, RETRO_RUMBLE_STRONG, shake[port] ? 0xFFFF : 0);
			rumbling[port] = shake[port];
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

// The picture's size and shape to the frontend when either changes.
static void setGeometry(int width, int height) {
	float aspect = optionIs(environment, "armn64_aspect", "tv")       ? n64TvAspect(machine)
	               : optionIs(environment, "armn64_aspect", "square") ? (float) width / (float) height
	                                                                  : 4.0f / 3.0f;
	// Widescreen: the picture drawn that much wider (with the built-in graphics), or a game's own
	// code drawing it so (Picture Only).
	bool const library = machine && !machine->gfxRsp;
	if (optionIs(environment, "armn64_widescreen", "16:9-picture") || (library && optionIs(environment, "armn64_widescreen", "16:9"))) {
		aspect *= 4.0f / 3;
	} else if (optionIs(environment, "armn64_widescreen", "16:10-picture") || (library && optionIs(environment, "armn64_widescreen", "16:10"))) {
		aspect *= 1.2f;
	}
	if (width != pictureWidth || height != pictureHeight || aspect != pictureAspect) {
		pictureWidth = width, pictureHeight = height, pictureAspect = aspect;
		struct retro_game_geometry geometry = { (unsigned) width, (unsigned) height, 640 * maxScale, 480 * maxScale, aspect };
		environment(RETRO_ENVIRONMENT_SET_GEOMETRY, &geometry);
	}
}

// A picture made on the CPU (0x00RRGGBB) to the frontend.
static void showPicture(int width, int height, bool shared) {
	if (!shared) {
		struct Crop const c = overscan(width, height, 1);
		int const w = width - c.left - c.right, h = height - c.top - c.bottom;
		setGeometry(w, h);
		videoRefresh(picture + (size_t) c.top * width + c.left, (unsigned) w, (unsigned) h, (size_t) width * 4);
	} else if (hwReady) {
		present(width, height, 0);
	}
}

// Each port's Controller Pak card from the options; the manager while it is open (the game paused),
// and the option back to Closed once it is.
static bool pakManagerAsked;

static bool pakManager(bool shared) {
	int cards[4];
	for (int port = 0; port < 4; ++port) {
		char key[24];
		snprintf(key, sizeof(key), "armn64_pak%d_card", port + 1);
		const char* value = optionValue(environment, key);
		cards[port] = value && value[0] >= '1' && value[0] <= '8' ? value[0] - '0' : 0;
	}
	paksPlug(machine, cards);
	bool const asked = optionIs(environment, "armn64_pak_manager", "open");
	if (asked && !pakManagerAsked) {
		paksManagerOpen();
	}
	pakManagerAsked = asked;
	uint32_t buttons = 0;
	for (unsigned id = 0; id <= RETRO_DEVICE_ID_JOYPAD_R; ++id) {
		buttons |= inputState(0, RETRO_DEVICE_JOYPAD, 0, id) ? 1u << id : 0;
	}
	int width, height;
	if (!paksManager(machine, buttons, picture, &width, &height)) {
		if (asked) {
			struct retro_variable closed = { "armn64_pak_manager", "closed" };
			environment(RETRO_ENVIRONMENT_SET_VARIABLE, &closed);
		}
		return false;
	}
	showPicture(width, height, shared);
	size_t const silence = OUT_RATE / (pal ? 50 : 60);
	memset(played, 0, silence * 4);
	for (size_t done = 0; done < silence;) {
		done += audioBatch(played + done * 2, silence - done);
	}
	return true;
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
	} else if (hwReady) {
		// The frontend drew in this context since the last frame.
		glrForget(machine);
	}
	inputPoll();
	if (pakManager(shared)) {
		if (!shared) {
			glesYield();
		}
		return;
	}
	// The game waits while the shaders it used before are made, a few a frame.
	int done, total;
	if (shared && hwReady && n64GlPrecompile(machine, 12.0, &done, &total)) {
		showProgress(done, total);
		if (done < total) {
			videoRefresh(NULL, (unsigned) pictureWidth, (unsigned) pictureHeight, 0);
			return;
		}
	}
	readPorts();
	if (machine->skip.automatic) {
		// A longer audio buffer gives the skips time to catch up.
		if (!latencySet) {
			unsigned latency = pal ? 120 : 100;
			environment(RETRO_ENVIRONMENT_SET_MINIMUM_AUDIO_LATENCY, &latency);
			latencySet = true;
		}
		machine->skip.behind = audioBehind;
	}
	machine->viFilters = optionIs(environment, "armn64_vi_filters", "on") ||
	                     (!optionIs(environment, "armn64_vi_filters", "off") && machine->accuracy != ACCURACY_HLE && !machine->rdpGl);
	cheatsApply(machine, &cheats);
	n64RunField(machine);
	int width = 0, height = 0;
	// The GPU RDP's picture goes from its buffer to the frontend's framebuffer without the CPU, and so
	// does the GL renderer's.
	unsigned buffer = 0, framebuffer = 0;
	int x = 0, y = 0;
	if (machine->glr && n64PictureSkipped(machine)) {
		videoRefresh(NULL, (unsigned) pictureWidth, (unsigned) pictureHeight, 0);
	} else if (shared && hwReady && n64GlPicture(machine, &framebuffer, &x, &y, &width, &height) && width && height) {
		width *= machine->glrScale, height *= machine->glrScale;
		presentGl(framebuffer, x, y, width, height);
	} else if (shared && hwReady && n64PictureStart(machine, &width, &height) && width && height && (buffer = n64PictureBuffer(machine))) {
		present(width, height, buffer);
	} else if (n64Picture(machine, picture, &width, &height) && width && height) {
		showPicture(width, height, shared);
	} else {
		videoRefresh(NULL, (unsigned) pictureWidth, (unsigned) pictureHeight, 0);
	}
	if (!shared) {
		glesYield();
	}
	playAudio();
	tellFallbacks();
	if (machine->saveDirty) {
		paksSync(machine);
		saveExport();
		machine->saveDirty = false;
	}
	diskSave();
	transferPaksSave();
	if (ramView) {
		for (uint32_t p = 0; p < (RDRAM_SIZE >> 12); ++p) {
			if (!machine->dirtyPages[p]) {
				swapWords(ramView + (p << 12), machine->rdram + (p << 12), 4096);
				machine->dirtyPages[p] = 0xFF;
			}
		}
	}
}

// Run-ahead and rollback states, taken many times a second, leave the GL renderer's pictures on the
// GPU.
static bool quickState(void) {
	int context = RETRO_SAVESTATE_CONTEXT_NORMAL;
	return environment(RETRO_ENVIRONMENT_GET_SAVESTATE_CONTEXT, &context) &&
	       (context == RETRO_SAVESTATE_CONTEXT_RUNAHEAD_SAME_INSTANCE || context == RETRO_SAVESTATE_CONTEXT_ROLLBACK_NETPLAY);
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
	bool saved = n64StateSave(machine, data, size, quickState());
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
	bool loaded = n64StateLoad(machine, data, size, quickState());
	if (own) {
		glesYield();
	}
	// The state's saves stand (the save file takes them on the next run).
	saveGiven |= loaded;
	return loaded;
}

void retro_cheat_reset(void) {
	cheatsClear(&cheats);
}

void retro_cheat_set(unsigned index, bool enabled, const char* code) {
	cheatsSet(&cheats, index, enabled, code);
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
			memset(machine->dirtyPages, 0xFF, sizeof(machine->dirtyPages));
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
