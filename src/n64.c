#include "n64.h"
#include "gles.h"
#include "rdp.h"
#include "vi_gpu_glsl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct N64* n64Create(void) {
	static bool tables;
	if (!tables) {
		rspInitTables();
		tables = true;
	}
	struct N64* n = calloc(1, sizeof(*n));
	if (!n) {
		return NULL;
	}
	n->rdram = calloc(1, RDRAM_SIZE);
	n->hidden = calloc(1, RDRAM_SIZE / 2);
	n->vpages = malloc(2 * VPAGES * sizeof(*n->vpages));
	if (!n->rdram || !n->hidden || !n->vpages) {
		free(n->rdram);
		free(n->hidden);
		free(n->vpages);
		free(n);
		return NULL;
	}
	// Controller 1 plugged in; Controller Paks as they come formatted; flash erased; the hardware's clock.
	n->plugged[0] = PLUG_CONTROLLER;
	n->ocNum = n->ocDen = 1;
	n->countCycles = 2;
	for (int i = 0; i < 4; ++i) {
		pakFormat(n->paks[i]);
	}
	memset(n->flash, 0xFF, sizeof(n->flash));
	// RDRAM holds noise at power-on, not zeros, and the boot code clears only what it uses. Wayne
	// Gretzky's 3D Hockey '98 (EU) sends its first audio list empty and gets by on that noise.
	uint32_t x = 0x2545F491u;
	for (uint32_t i = 0; i < RDRAM_SIZE; i += 4) {
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		put32(&n->rdram[i], x);
	}
	n->pageFlags[RDRAM_SIZE >> 12] = 0xFF;
	n->vpageFillCount = VPAGE_FILLED + 1;
	cpuTlbChanged(n);
	return n;
}

static void viKeptFree(struct ViKept* kept);
struct ViPool;
static void viPoolStop(struct ViPool* p);

void n64Destroy(struct N64* n) {
	if (n) {
		rspThreadStop(n);
		blocksFree(n);
		jitFree(n);
		rspJitFree(n);
		rdpDestroy(n);
		glesDestroy(n->viGles);
		free(n->lastPicture);
		viKeptFree(n->viKept);
		viPoolStop(n->viPool);
		free(n->isViewer);
		free(n->hle);
		free(n->hleAudio);
		for (int i = 0; i < 4; ++i) {
			gbCartFree(&n->tpak[i].cart);
		}
		free(n->rdram);
		free(n->hidden);
		free(n->vpages);
		ddFree(n);
		free(n->rom);
		free(n);
	}
}

static void start(struct N64* n);

// Cartridge images come big-endian (.z64), halfword-swapped (.v64) or word-swapped (.n64).
bool n64LoadRom(struct N64* n, const uint8_t* data, size_t size) {
	if (size < 0x1000) {
		return false;
	}
	uint8_t* rom = malloc(size);
	if (!rom) {
		return false;
	}
	uint32_t magic = be32(data);
	for (size_t i = 0; i + 3 < size; i += 4) {
		const uint8_t* s = data + i;
		uint8_t* d = rom + i;
		switch (magic) {
		case 0x37804012:
			d[0] = s[1], d[1] = s[0], d[2] = s[3], d[3] = s[2];
			break;
		case 0x40123780:
			d[0] = s[3], d[1] = s[2], d[2] = s[1], d[3] = s[0];
			break;
		default:
			memcpy(d, s, 4);
		}
	}
	free(n->rom);
	n->rom = rom;
	n->romSize = size & ~3u;
	start(n);
	return true;
}

// A setting by its environment variable's name, from the frontend's lookup when it gave one.
static const char* setting(struct N64* n, const char* name) {
	return n->settings ? n->settings(name) : getenv(name);
}

// From power-on: the PIF's boot from the cartridge, or from the 64DD's IPL without one.
static void start(struct N64* n) {
	const char* accuracy = setting(n, "N64_ACCURACY");
	// The modes' old names, lle-compatible and lle-accurate, still work.
	n->accuracy = !accuracy ? ACCURACY_LLE : !strcmp(accuracy, "mle") || !strcmp(accuracy, "lle-compatible") ? ACCURACY_MLE :
	              !strcmp(accuracy, "hle") ? ACCURACY_HLE : ACCURACY_LLE;
	// HLE draws at high level with the hardware's own timing; only mle behaves like the
	// emulators hacks were made on.
	n->compat = n->accuracy == ACCURACY_MLE;
	n->emulatorLike = n->accuracy != ACCURACY_LLE;
	n->rspPlainTiming = n->accuracy != ACCURACY_LLE;
	n->fast = n->accuracy == ACCURACY_HLE;
	const char* idleSkip = setting(n, "N64_IDLESKIP");
	n->idle.enabled = n->accuracy == ACCURACY_HLE && !(idleSkip && idleSkip[0] == '0');
	n->hleCheck = setting(n, "N64HLECHECK") != NULL;
	const char* gfxRsp = setting(n, "N64_GFX_RSP");
	n->gfxRsp = gfxRsp && gfxRsp[0] == '1';
	if (setting(n, "N64ISVIEWER") && !n->isViewer) {
		n->isViewer = calloc(1, 0x10000);
	}
	const char* mouse = setting(n, "N64_MOUSE");
	if (mouse && *mouse >= '1' && *mouse <= '4') {
		n->plugged[*mouse - '1'] = PLUG_MOUSE;
	}
	// N64_PAK: what is in the controllers' slots, a letter a port (m Controller Pak, r Rumble Pak, t
	// Transfer Pak).
	const char* paks = setting(n, "N64_PAK");
	for (int i = 0; paks && i < 4 && paks[i]; ++i) {
		n->pak[i] = paks[i] == 'm' ? PAK_MEMORY : paks[i] == 'r' ? PAK_RUMBLE : paks[i] == 't' ? PAK_TRANSFER : PAK_NONE;
	}
	n->flashMode = n->flashStatus = 0;
	if (n->compat) {
		memset(n->rdram, 0, RDRAM_SIZE);
	}
	blocksFree(n);
	cpuReset(n);
	pifBoot(n);
	n->eeprom16k = n->rom ? cartEeprom16k(n->rom) : false;
	memset(n->eeprom, 0xFF, sizeof(n->eeprom));
	// 1, 1.5, 2, 2.5, 3 or 4 times the CPU's clock; LLE is always the hardware's.
	const char* overclock = setting(n, "N64_OVERCLOCK");
	double factor = overclock && n->accuracy != ACCURACY_LLE ? strtod(overclock, NULL) : 1;
	n->ocDen = 2;
	n->ocNum = factor >= 1 && factor <= 4 ? (uint32_t) (factor * 2 + 0.5) : 2;
	if (n->ocNum % 2 == 0) {
		n->ocNum /= 2, n->ocDen = 1;
	}
	n->countCycles = 2 * n->ocNum / n->ocDen;
	n->fieldCycles = hwCycles(n, CPU_HZ / (n->cpu.gpr[20] == 0 ? 50 : 60));
	n->vi.reg[6] = n->cpu.gpr[20] == 0 ? 0x271 : 0x20D;
	n->nextVi = n->fieldCycles;
	n->eventAt = 0;
	const char* slice = setting(n, "N64_SLICE");
	n->slice = slice ? strtoull(slice, NULL, 0) : 0;
	const char* blocks = setting(n, "N64_BLOCKS");
#ifdef N64_HAS_JIT
	const char* jit = setting(n, "N64_JIT");
	const char* rspJit = setting(n, "N64_RSPJIT");
	n->useJit = n->slice && jit && jit[0] == '1';
	n->useRspJit = rspJit && rspJit[0] == '1';
#endif
	n->useBlocks = n->useJit || (n->slice && blocks && blocks[0] == '1');
	const char* rdpThreads = setting(n, "N64_RDP_THREADS");
	n->rdpThreads = rdpThreads ? atoi(rdpThreads) : 0;
	glesCacheDir(setting(n, "N64_SHADER_CACHE"));
	const char* rdpGpu = setting(n, "N64_RDP_GPU");
	n->rdpGpu = rdpGpu && rdpGpu[0] == '1';
	if (n->rdpGpu && !n->rdpThreads) {
		n->rdpThreads = 1;
	}
	const char* skip = setting(n, "N64_FRAMESKIP");
	n->skip.on = skip && skip[0] == '1';
	const char* rdpGl = setting(n, "N64_RDP_GL");
	n->rdpGl = rdpGl && rdpGl[0] == '1';
	if (n->rdpGl) {
		n->rdpThreads = 0;
		n->rdpGpu = false;
	}
	const char* viGpu = setting(n, "N64_VI_GPU");
	n->viGpu = viGpu && viGpu[0] == '1';
	const char* filters = setting(n, "N64_VI_FILTERS");
	n->viFilters = filters ? filters[0] == '1' : n->accuracy != ACCURACY_HLE && !n->rdpGl;
	rspJitFree(n);
	++n->rspCodeGen;
	// The RSP's thread, in HLE, a slice at a time. A longer window holds the RSP back further and
	// games notice (Perfect Dark draws at 60 frames a second instead of 30 with 4096 cycles in MLE);
	// MLE's 1024 are too short for the thread to gain.
	rspThreadStop(n);
	const char* rspThread = setting(n, "N64_RSP_THREAD");
	if (n->accuracy == ACCURACY_HLE && n->slice && rspThread && rspThread[0] == '1') {
		const char* window = setting(n, "N64_RSP_WINDOW");
		n->rspWindow = window ? strtoull(window, NULL, 0) : n->slice;
		n->rspWindowAt = 0;
		rspThreadStart(n);
	}
}

bool n64LoadDisk(struct N64* n, const uint8_t* ipl, size_t iplSize, const uint8_t* disk, size_t diskSize) {
	if (!ddLoad(n, ipl, iplSize, disk, diskSize)) {
		return false;
	}
	if (!n->rom) {
		start(n);
	}
	return true;
}

void n64Interrupt(struct N64* n, uint32_t line, bool set) {
	if (set) {
		n->miIntr |= line;
	} else {
		n->miIntr &= ~line;
	}
	cpuCheckInterrupts(n);
}

void rspRunTo(struct N64* n, uint64_t until) {
	while (n->rsp.clock < until && !(n->sp.status & 1)) {
		// Compiled code from the start of a block, and not from a delay slot.
		void* code;
		if (n->useRspJit && n->rsp.nextPc == ((n->sp.pc + 4) & 0xFFC) && (code = rspJitLookup(n, n->sp.pc & 0xFFC))) {
			n->rspEnter(n, code, until);
		} else {
			rspStep(n);
		}
	}
}

void rspCatchUp(struct N64* n) {
	if (n->rspThread) {
		rspThreadJoin(n);
	}
	if (n->catchingUp) {
		return;
	}
	// The RSP runs at two thirds of the CPU clock, on its own cycle count.
	uint64_t rspNow = rspCycles(n);
	if ((n->sp.status & 1) || n->sp.hleDoneAt) {
		n->rsp.clock = rspNow;
		return;
	}
	n->catchingUp = true;
	rspRunTo(n, rspNow);
	n->catchingUp = false;
}

// When the beam reaches the V_INTR line in the field from start (never if the field is shorter).
static uint64_t viIntrAt(struct N64* n, uint64_t start) {
	uint32_t lines = (n->vi.reg[6] & 0x3FF) + 1, line = n->vi.reg[3] & 0x3FF;
	return line < lines ? start + (uint64_t) line * n->fieldCycles / lines : UINT64_MAX;
}

// One video field: the CPU runs until the field ends, with the VI interrupt when the beam passes
// the V_INTR line (none in a field that doesn't reach it).
// With a slice, the CPU runs ahead of the RSP by up to that many cycles (or to the next event),
// and the RSP catches up then or when the CPU touches what it shares with it. Without one, the
// two step in lockstep.
void n64RunField(struct N64* n) {
	uint64_t start = n->nextVi - n->fieldCycles, passed = start;
	while (n->cycles < n->nextVi) {
		uint64_t intrAt = viIntrAt(n, start);
		if (intrAt >= passed && intrAt <= n->cycles) {
			n64Interrupt(n, MI_VI, true);
		}
		passed = n->cycles + 1;
		if (n->ai.queued && n->cycles >= n->ai.endsAt) {
			aiUpdate(n);
		}
		if (n->dp.syncAt && n->cycles >= n->dp.syncAt) {
			n->dp.syncAt = 0;
			n64Interrupt(n, MI_DP, true);
			if (n->dp.syncQueued) {
				n->dp.syncAt = n->dp.syncQueue[0];
				memmove(n->dp.syncQueue, n->dp.syncQueue + 1, sizeof(n->dp.syncQueue[0]) * (size_t) --n->dp.syncQueued);
			}
		}
		if (n->sp.hleDoneAt && n->cycles >= n->sp.hleDoneAt && hleWaiting(n)) {
			hleResume(n);
		} else if (n->sp.hleDoneAt && n->cycles >= n->sp.hleDoneAt) {
			// A task run at high level ends as its microcode would: halted at a break with the
			// task-done signal.
			n->sp.hleDoneAt = 0;
			n->sp.status |= 0x203;
			if (n->sp.status & 0x40) {
				n64Interrupt(n, MI_SP, true);
			}
		}
		if (n->si.busyUntil && n->cycles >= n->si.busyUntil) {
			n->si.busyUntil = 0;
			n->si.status = (n->si.status & ~1u) | 0x1000;
			n64Interrupt(n, MI_SI, true);
		}
		if (n->pi.busyUntil && n->cycles >= n->pi.busyUntil) {
			n->pi.busyUntil = 0;
			n->pi.status &= ~1u;
			n64Interrupt(n, MI_PI, true);
		}
		if (n->dd.mechaAt && n->cycles >= n->dd.mechaAt) {
			ddMechaResponse(n);
		}
		if (n->dd.bmAt && n->cycles >= n->dd.bmAt) {
			ddBmRequest(n);
		}
		uint64_t until = n->nextVi;
		if (n->slice) {
			until = n->cycles + n->slice < until ? n->cycles + n->slice : until;
			if (intrAt > n->cycles && intrAt < until) {
				until = intrAt;
			}
			n->eventAt = until;
			n64Event(n, n->ai.queued ? n->ai.endsAt : until);
			n64Event(n, n->dp.syncAt ? n->dp.syncAt : until);
			n64Event(n, n->si.busyUntil ? n->si.busyUntil : until);
			n64Event(n, n->sp.hleDoneAt ? n->sp.hleDoneAt : until);
			n64Event(n, n->pi.busyUntil ? n->pi.busyUntil : until);
			n64Event(n, n->dd.mechaAt ? n->dd.mechaAt : until);
			n64Event(n, n->dd.bmAt ? n->dd.bmAt : until);
		}
		// One cycle an instruction, plus cache fills, RCP register reads and long operations.
		if (n->useBlocks) {
			blocksRun(n);
		} else do {
			cpuStep(n);
			n->cycles += 1 + n->cpu.stall;
			n->cpu.stall = 0;
		} while (n->slice && n->cycles < n->eventAt);
		if (n->rspThread) {
			rspThreadSlice(n);
		} else {
			rspCatchUp(n);
		}
	}
	if (n->rspThread) {
		rspThreadJoin(n);
	}
	uint64_t intrAt = viIntrAt(n, start);
	if (intrAt >= passed && intrAt < n->nextVi) {
		n64Interrupt(n, MI_VI, true);
	}
	n->nextVi += n->fieldCycles;
	++n->fields;
}

// One framebuffer row for the VI from column -3 (index 0): 8-bit channels, their 5-bit values
// for the dither filter, and coverage (from the hidden bits for 16-bit pixels, the top alpha bits
// for 32-bit ones).
struct ViRow {
	uint8_t c[3][646], q[3][646], cvg[646];
};

static void viStore(struct ViRow* row, int k, int r, int g, int b, int cvg) {
	row->c[0][k] = (uint8_t) r, row->c[1][k] = (uint8_t) g, row->c[2][k] = (uint8_t) b;
	row->q[0][k] = (uint8_t) (r >> 3), row->q[1][k] = (uint8_t) (g >> 3), row->q[2][k] = (uint8_t) (b >> 3);
	row->cvg[k] = (uint8_t) cvg;
}

static void viDecodePixel(struct N64* n, bool wide, uint32_t a, struct ViRow* row, int k) {
	const uint8_t* m = &n->rdram[a];
	if (wide) {
		viStore(row, k, m[0], m[1], m[2], (m[3] >> 5) & 7);
		return;
	}
	int px = m[0] << 8 | m[1];
	int r = (px >> 11) & 31, g = (px >> 6) & 31, b = (px >> 1) & 31;
	viStore(row, k, r << 3 | r >> 2, g << 3 | g >> 2, b << 3 | b >> 2, (px & 1) << 2 | n->hidden[a >> 1]);
}

// Decodes count pixels from index start; rows that don't wrap around RDRAM take the straight loops.
static void viDecodeRow(struct N64* n, bool wide, uint32_t origin, int64_t start, int count, struct ViRow* restrict row) {
	uint32_t size = wide ? 4 : 2;
	uint32_t a0 = (origin + (uint32_t) start * size) & (RDRAM_SIZE - size);
	if (a0 + (uint32_t) count * size > RDRAM_SIZE) {
		for (int k = 0; k < count; ++k) {
			viDecodePixel(n, wide, (origin + (uint32_t) (start + k) * size) & (RDRAM_SIZE - size), row, k);
		}
		return;
	}
	const uint8_t* restrict m = &n->rdram[a0];
	if (wide) {
		for (int k = 0; k < count; ++k) {
			int r = m[k * 4], g = m[k * 4 + 1], b = m[k * 4 + 2];
			row->c[0][k] = (uint8_t) r, row->c[1][k] = (uint8_t) g, row->c[2][k] = (uint8_t) b;
			row->q[0][k] = (uint8_t) (r >> 3), row->q[1][k] = (uint8_t) (g >> 3), row->q[2][k] = (uint8_t) (b >> 3);
			row->cvg[k] = (m[k * 4 + 3] >> 5) & 7;
		}
		return;
	}
	const uint8_t* restrict hidden = &n->hidden[a0 >> 1];
	for (int k = 0; k < count; ++k) {
		int px = m[k * 2] << 8 | m[k * 2 + 1];
		int r = (px >> 11) & 31, g = (px >> 6) & 31, b = (px >> 1) & 31;
		row->c[0][k] = (uint8_t) (r << 3 | r >> 2), row->c[1][k] = (uint8_t) (g << 3 | g >> 2), row->c[2][k] = (uint8_t) (b << 3 | b >> 2);
		row->q[0][k] = (uint8_t) r, row->q[1][k] = (uint8_t) g, row->q[2][k] = (uint8_t) b;
		row->cvg[k] = (uint8_t) ((px & 1) << 2 | hidden[k]);
	}
}

static inline int viMax(int a, int b) {
	return a > b ? a : b;
}

static inline int viMin(int a, int b) {
	return a < b ? a : b;
}

// The dither filter for one channel of a row: a fully covered pixel moves a step toward each of
// its eight neighbours (5-bit values compared).
static void viDitherRow(int16_t* restrict out, const uint8_t* restrict c, const uint8_t* restrict at, const uint8_t* restrict up,
                        const uint8_t* restrict down, int count) {
	for (int x = 0; x < count; ++x) {
		int b = at[x], s = 0;
		s += (b < up[x - 1]) - (b > up[x - 1]);
		s += (b < up[x]) - (b > up[x]);
		s += (b < up[x + 1]) - (b > up[x + 1]);
		s += (b < at[x - 1]) - (b > at[x - 1]);
		s += (b < at[x + 1]) - (b > at[x + 1]);
		s += (b < down[x - 1]) - (b > down[x - 1]);
		s += (b < down[x]) - (b > down[x]);
		s += (b < down[x + 1]) - (b > down[x + 1]);
		out[x] = (int16_t) (c[x] + s);
	}
}

// Antialiasing for one channel of a row: a partly covered pixel moves toward the colour the fully
// covered pixels around it agree on, by the coverage it lacks. That colour comes from the
// penumbra of the pixel and those neighbours: the largest if the pixel has it, else the second
// largest (repeats count), and the same for the smallest.
static void viRestoreRow(int16_t* restrict out, const uint8_t* restrict at, const uint8_t* restrict atCvg, const uint8_t* restrict up,
                         const uint8_t* restrict upCvg, const uint8_t* restrict down, const uint8_t* restrict downCvg, int count) {
	for (int x = 0; x < count; ++x) {
		int c = at[x], hi1 = c, hi2 = -1, lo1 = c, lo2 = 256;
		// A neighbour that isn't fully covered counts as -1 for the largest and 256 or more for the
		// smallest, which changes neither.
#define VI_NEIGHBOUR(v, cvg) do { \
			int vh_ = (v) | -((cvg) != 7), vl_ = (v) | ((cvg) != 7) << 8; \
			hi2 = viMax(hi2, viMin(hi1, vh_)), hi1 = viMax(hi1, vh_); \
			lo2 = viMin(lo2, viMax(lo1, vl_)), lo1 = viMin(lo1, vl_); \
		} while (0)
		VI_NEIGHBOUR(up[x - 1], upCvg[x - 1]);
		VI_NEIGHBOUR(up[x + 1], upCvg[x + 1]);
		VI_NEIGHBOUR(at[x - 2], atCvg[x - 2]);
		VI_NEIGHBOUR(at[x + 2], atCvg[x + 2]);
		VI_NEIGHBOUR(down[x - 1], downCvg[x - 1]);
		VI_NEIGHBOUR(down[x + 1], downCvg[x + 1]);
#undef VI_NEIGHBOUR
		int isHi = -(c == hi1), isLo = -(c == lo1);
		int high = (hi1 & isHi) | (hi2 & ~isHi), low = (lo1 & isLo) | (lo2 & ~isLo);
		int col = low + high - (c << 1);
		out[x] = (int16_t) (((((col * (7 - atCvg[x])) + 4) >> 3) + c) & 255);
	}
}

// What the VI scans out: the framebuffer, the picture's size and the filters on.
struct ViSetup {
	uint32_t origin, stride;
	int w, h;
	float tvAspect;
	bool wide, gammaOn, divot, dither, aa;
};

// The setup, the RDRAM bytes and hidden bits the filters read, and the picture they made.
struct ViKept {
	struct ViSetup setup;
	uint32_t addr, len;
	uint8_t* bytes;
	uint32_t* picture;
	size_t room, pictureRoom;
	bool valid;
};

static void viKeptFree(struct ViKept* kept) {
	if (kept) {
		free(kept->bytes);
		free(kept->picture);
		free(kept);
	}
}

static bool viSetup(struct N64* n, struct ViSetup* s) {
	uint32_t* v = n->vi.reg;
	int type = v[0] & 3;
	s->origin = v[1] & 0xFFFFFF;
	s->stride = v[2] & 0xFFF;
	int hStart = (v[9] >> 16) & 0x3FF, hEnd = v[9] & 0x3FF;
	int vStart = (v[10] >> 16) & 0x3FF, vEnd = v[10] & 0x3FF;
	uint32_t xScale = v[12] & 0xFFF, yScale = v[13] & 0xFFF;
	// In mle accuracy, an empty window (osViBlack) shows the picture anyway in the TV type's
	// usual one, as emulators that draw the RDP's image directly do.
	if (n->emulatorLike && (hEnd <= hStart || vEnd <= vStart)) {
		bool pal = n->fieldCycles == hwCycles(n, CPU_HZ / 50);
		hStart = pal ? 0x80 : 0x6C, hEnd = pal ? 0x300 : 0x2EC;
		vStart = pal ? 0x5F : 0x25, vEnd = pal ? 0x239 : 0x1FF;
	}
	int w = (int) (((hEnd - hStart) * xScale + 512) >> 10);
	int h = (int) ((((vEnd - vStart) >> 1) * yScale + 512) >> 10);
	if (type < 2 || w <= 0 || h <= 0 || s->stride == 0) {
		return false;
	}
	// The window's shape on a TV, whose 640 dots by 480 half-lines (576 in PAL) show as 4:3.
	bool pal = n->fieldCycles == hwCycles(n, CPU_HZ / 50);
	s->tvAspect = 4.0f / 3.0f * ((float) (hEnd - hStart) / 640.0f) / ((float) (vEnd - vStart) / (pal ? 576.0f : 480.0f));
	s->w = w > 640 ? 640 : w;
	s->h = h > 480 ? 480 : h;
	s->gammaOn = v[0] & 8, s->divot = v[0] & 16, s->dither = v[0] & 0x10000, s->aa = ((v[0] >> 8) & 3) <= 1, s->wide = type == 3;
	if (!n->viFilters) {
		s->divot = s->dither = s->aa = false;
	}
	return true;
}

// The RDRAM the filters read: rows y-1 to h, columns -3 to w+2. False when it wraps around RDRAM.
static bool viRange(const struct ViSetup* s, uint32_t* addr, uint32_t* len) {
	uint32_t size = s->wide ? 4 : 2;
	int64_t first = -(int64_t) s->stride - 3, last = (int64_t) s->h * s->stride + s->w + 2;
	*addr = (s->origin + (uint32_t) first * size) & (RDRAM_SIZE - size);
	*len = (uint32_t) (last - first + 1) * size;
	return *addr + *len <= RDRAM_SIZE;
}

// Waits for the RDP commands drawing what the VI reads.
static void viWait(struct N64* n, const struct ViSetup* s) {
	uint32_t addr, len;
	if (viRange(s, &addr, &len)) {
		rdpWaitDrawn(n, addr, len);
	} else {
		rdpWait(n);
	}
}

// Starts the filters on the GPU: uploads the rows they read (y-1 to h, columns -3 to w+2) and their
// hidden bits. False when the GPU can't start or the rows wrap around RDRAM.
static bool viGpuStart(struct N64* n, const struct ViSetup* s) {
	uint32_t a0, bytes;
	struct Gles* shared = viRange(s, &a0, &bytes) && n->viSharedProgram >= 0 ? rdpGpuShare(n, a0, bytes) : NULL;
	if (shared && !n->viSharedProgram) {
		static char source[1 << 16];
		snprintf(source, sizeof(source), "#version 310 es\n#define VI_SHARED 1\n%s", strchr(viGpuSource, '\n') + 1);
		char error[512] = "";
		int index = glesAddProgram(shared, source, error, sizeof(error));
		if (index < 0) {
			fprintf(stderr, "GPU VI beside the GPU RDP unavailable (%s)\n", error);
		} else {
			glesData(shared, 4, NULL, 640 * 480 * 4);
		}
		n->viSharedProgram = index < 0 ? -1 : index + 1;
	}
	// With the GPU RDP asked for, the VI runs in its context or on the CPU, never in one of its own.
	if (n->rdpGpu && !(shared && n->viSharedProgram > 0)) {
		return false;
	}
	if (shared && n->viSharedProgram > 0) {
		glesSelectProgram(shared, n->viSharedProgram - 1);
		int u[8] = { (int) s->origin, (int) s->stride, s->wide, s->w, s->h, 0, -(int) RDRAM_SIZE,
		             s->aa | s->dither << 1 | s->divot << 2 | s->gammaOn << 3 };
		glesUniforms(shared, u, 8);
		glesDispatch(shared, (unsigned) (s->w + 15) / 16, (unsigned) (s->h + 15) / 16);
		n->pictureGles = shared, n->pictureBinding = 4, n->pictureRgba = true;
		return true;
	}
	viWait(n, s);
	if (!n->viGles) {
		char error[512];
		// In the frontend's context the picture goes to a texture there: RGBA bytes.
		static char source[1 << 16];
		snprintf(source, sizeof(source), "#version 310 es\n#define VI_RGBA 1\n%s", strchr(viGpuSource, '\n') + 1);
		n->viGles = glesCreate(glesInFrontend() ? source : viGpuSource, error, sizeof(error));
		if (!n->viGles) {
			fprintf(stderr, "GPU VI unavailable (%s), using the CPU\n", error);
			n->viGpu = false;
			return false;
		}
		fprintf(stderr, "GPU VI on %s\n", glesRenderer(n->viGles));
		glesData(n->viGles, 2, NULL, 640 * 480 * 4);
	}
	if (!viRange(s, &a0, &bytes)) {
		return false;
	}
	uint32_t begin = a0 & ~3u, end = (a0 + bytes + 3) & ~3u;
	glesData(n->viGles, 0, &n->rdram[begin], end - begin);
	uint32_t hiddenBegin = (a0 >> 1) & ~3u, hiddenEnd = (((a0 + bytes) >> 1) + 3) & ~3u;
	glesData(n->viGles, 1, s->wide ? NULL : &n->hidden[hiddenBegin], s->wide ? 0 : hiddenEnd - hiddenBegin);
	int u[8] = { (int) s->origin, (int) s->stride, s->wide, s->w, s->h, (int) begin, (int) hiddenBegin,
	             s->aa | s->dither << 1 | s->divot << 2 | s->gammaOn << 3 };
	glesUniforms(n->viGles, u, 8);
	glesDispatch(n->viGles, (unsigned) (s->w + 15) / 16, (unsigned) (s->h + 15) / 16);
	n->pictureGles = n->viGles, n->pictureBinding = 2, n->pictureRgba = glesInFrontend();
	return true;
}

unsigned n64PictureBuffer(struct N64* n) {
	return n->pictureGles && n->pictureRgba ? glesBuffer(n->pictureGles, n->pictureBinding) : 0;
}

// With the GL renderer, the VI's picture straight from the GPU when it drew it last (no VI filters,
// as HLE puts speed first): the framebuffer and the picture's first pixel there. False when RDRAM
// has the picture.
bool n64GlPicture(struct N64* n, unsigned* framebuffer, int* x, int* y, int* width, int* height) {
	struct ViSetup s;
	if (!n->glr || !viSetup(n, &s)) {
		return false;
	}
	*framebuffer = glrPicture(n, s.origin, s.stride, s.wide ? 3 : 2, s.h, x, y);
	*width = s.w;
	*height = s.h;
	return *framebuffer != 0;
}

float n64TvAspect(struct N64* n) {
	struct ViSetup s;
	return viSetup(n, &s) ? s.tvAspect : 4.0f / 3.0f;
}

bool n64PictureStart(struct N64* n, int* width, int* height) {
	struct ViSetup s;
	if (!n->viGpu || !viSetup(n, &s)) {
		*width = *height = 0;
		return false;
	}
	if (n->glr && glrShows(n, s.origin)) {
		*width = s.w;
		*height = s.h;
		return true;
	}
	// A skipped frame: the GPU still has the picture before it.
	if (n->skip.on && frameSkipped(n, s.origin) && n->viGles) {
		*width = n->lastWidth;
		*height = n->lastHeight;
		return true;
	}
	n->lastWidth = s.w, n->lastHeight = s.h;
	if (!viGpuStart(n, &s)) {
		*width = *height = 0;
		return false;
	}
	*width = s.w;
	*height = s.h;
	return true;
}

// The picture the VI scans out, scaled by its X and Y scale registers, as 0x00RRGGBB.
// The VI filters for rows y0 to y1 of the picture, as one thread makes them: rows y-1, y and y+1 (a
// ring of three), decoded from column -3 to w+2.
struct ViBand {
	struct ViRow rows[3];
	int16_t full[3][642], restored[3][642], line[3][642], divoted[3][640];
	uint8_t lineCvg[642];
};

static uint8_t viGamma[256];

static void viRows(struct N64* n, const struct ViSetup* restrict sp, uint32_t* out, int y0, int y1, struct ViBand* restrict b) {
	uint32_t origin = sp->origin, stride = sp->stride;
	bool gammaOn = sp->gammaOn, divot = sp->divot, dither = sp->dither, aa = sp->aa, wide = sp->wide;
	int w = sp->w;
	const uint8_t* gamma = viGamma;
	struct ViRow* rows = b->rows;
	int16_t (*full)[642] = b->full, (*restored)[642] = b->restored, (*line)[642] = b->line, (*divoted)[640] = b->divoted;
	uint8_t* lineCvg = b->lineCvg;
	for (int k = 0; k < 3; ++k) {
		viDecodeRow(n, wide, origin, (int64_t) (y0 + k - 1) * stride - 3, w + 6, &rows[(y0 + k) % 3]);
	}
	for (int y = y0; y < y1; ++y) {
		if (y > y0) {
			viDecodeRow(n, wide, origin, (int64_t) (y + 1) * stride - 3, w + 6, &rows[(y + 2) % 3]);
		}
		const struct ViRow* up = &rows[y % 3], * at = &rows[(y + 1) % 3], * down = &rows[(y + 2) % 3];
		// Columns -1 to w (index 0 to w+1; rows from index 2).
		int count = w + 2;
		bool partial = false;
		if (aa) {
			uint8_t all = 7;
			for (int x = 0; x < count; ++x) {
				all &= at->cvg[x + 2];
			}
			partial = all != 7;
		}
		for (int j = 0; j < 3; ++j) {
			if (dither) {
				viDitherRow(full[j], at->c[j] + 2, at->q[j] + 2, up->q[j] + 2, down->q[j] + 2, count);
			} else {
				for (int x = 0; x < count; ++x) {
					full[j][x] = at->c[j][x + 2];
				}
			}
			if (partial) {
				viRestoreRow(restored[j], at->c[j] + 2, at->cvg + 2, up->c[j] + 2, up->cvg + 2, down->c[j] + 2, down->cvg + 2, count);
			}
		}
		for (int x = 0; x < count; ++x) {
			lineCvg[x] = aa ? at->cvg[x + 2] : 7;
		}
		for (int j = 0; j < 3; ++j) {
			if (partial) {
				for (int x = 0; x < count; ++x) {
					line[j][x] = lineCvg[x] == 7 ? full[j][x] : restored[j][x];
				}
			} else {
				memcpy(line[j], full[j], sizeof(line[j][0]) * (size_t) count);
			}
		}
		// The divot filter: each channel the median of a pixel and its neighbours, unless all three
		// are fully covered.
		if (divot) {
			for (int j = 0; j < 3; ++j) {
				for (int x = 0; x < w; ++x) {
					int a = line[j][x], b = line[j][x + 1], c = line[j][x + 2];
					int median = viMax(viMin(a, b), viMin(viMax(a, b), c));
					divoted[j][x] = (int16_t) ((lineCvg[x] & lineCvg[x + 1] & lineCvg[x + 2]) != 7 ? median : b);
				}
			}
		} else {
			for (int j = 0; j < 3; ++j) {
				memcpy(divoted[j], line[j] + 1, sizeof(line[j][0]) * (size_t) w);
			}
		}
		uint32_t* o = &out[y * w];
		if (gammaOn) {
			for (int x = 0; x < w; ++x) {
				o[x] = (uint32_t) gamma[divoted[0][x]] << 16 | (uint32_t) gamma[divoted[1][x]] << 8 | gamma[divoted[2][x]];
			}
		} else {
			for (int x = 0; x < w; ++x) {
				o[x] = (uint32_t) divoted[0][x] << 16 | (uint32_t) divoted[1][x] << 8 | (uint32_t) divoted[2][x];
			}
		}
	}
}

// Threads that make bands of the picture beside the emulation's own, as many as the renderers: they
// are idle once the VI has waited for them.
#include <pthread.h>
struct ViPool {
	pthread_t threads[8];
	int count;
	pthread_mutex_t lock;
	pthread_cond_t start, done;
	uint64_t job;
	int busy;
	bool quit;
	struct N64* n;
	const struct ViSetup* setup;
	uint32_t* out;
	struct ViBand bands[9];
};

static void viBand(struct ViPool* p, int index) {
	int const parts = p->count + 1, h = p->setup->h;
	viRows(p->n, p->setup, p->out, h * index / parts, h * (index + 1) / parts, &p->bands[index]);
}

static void* viWorker(void* arg) {
	struct ViPool* p = arg;
	pthread_mutex_lock(&p->lock);
	int const index = p->busy++;
	uint64_t seen = p->job;
	pthread_cond_signal(&p->done);
	for (;;) {
		while (p->job == seen && !p->quit) {
			pthread_cond_wait(&p->start, &p->lock);
		}
		if (p->quit) {
			break;
		}
		seen = p->job;
		pthread_mutex_unlock(&p->lock);
		viBand(p, index + 1);
		pthread_mutex_lock(&p->lock);
		if (--p->busy == 0) {
			pthread_cond_signal(&p->done);
		}
	}
	pthread_mutex_unlock(&p->lock);
	return NULL;
}

static struct ViPool* viPoolStart(int count) {
	struct ViPool* p = calloc(1, sizeof(*p));
	if (!p) {
		return NULL;
	}
	pthread_mutex_init(&p->lock, NULL);
	pthread_cond_init(&p->start, NULL);
	pthread_cond_init(&p->done, NULL);
	pthread_mutex_lock(&p->lock);
	for (int i = 0; i < count && i < 8; ++i) {
		if (pthread_create(&p->threads[p->count], NULL, viWorker, p) == 0) {
			++p->count;
		}
	}
	// Each takes its index as it starts.
	while (p->busy < p->count) {
		pthread_cond_wait(&p->done, &p->lock);
	}
	p->busy = 0;
	pthread_mutex_unlock(&p->lock);
	return p;
}

static void viPoolStop(struct ViPool* p) {
	if (!p) {
		return;
	}
	pthread_mutex_lock(&p->lock);
	p->quit = true;
	pthread_cond_broadcast(&p->start);
	pthread_mutex_unlock(&p->lock);
	for (int i = 0; i < p->count; ++i) {
		pthread_join(p->threads[i], NULL);
	}
	pthread_mutex_destroy(&p->lock);
	pthread_cond_destroy(&p->start);
	pthread_cond_destroy(&p->done);
	free(p);
}

static void viPoolRun(struct N64* n, struct ViPool* p, const struct ViSetup* setup, uint32_t* out) {
	pthread_mutex_lock(&p->lock);
	p->n = n, p->setup = setup, p->out = out;
	p->busy = p->count;
	++p->job;
	pthread_cond_broadcast(&p->start);
	pthread_mutex_unlock(&p->lock);
	viBand(p, 0);
	pthread_mutex_lock(&p->lock);
	while (p->busy) {
		pthread_cond_wait(&p->done, &p->lock);
	}
	pthread_mutex_unlock(&p->lock);
}

static int viPicture(struct N64* n, uint32_t* out, int* width, int* height) {
	struct ViSetup s;
	memset(&s, 0, sizeof(s));
	if (!viSetup(n, &s)) {
		*width = *height = 0;
		return 0;
	}
	int w = s.w, h = s.h;
	*width = w;
	*height = h;
	if (n->viGpu && viGpuStart(n, &s)) {
		const uint32_t* picture = glesMap(n->pictureGles, n->pictureBinding, 0, (size_t) w * h * 4);
		if (picture) {
			for (int i = 0; i < w * h; ++i) {
				uint32_t p = picture[i];
				out[i] = n->pictureRgba ? (p & 0xFF) << 16 | (p & 0xFF00) | (p >> 16 & 0xFF) : p;
			}
			glesUnmap(n->pictureGles, n->pictureBinding);
			return w * h;
		}
	}
	viWait(n, &s);
	// Many games show a frame for two or three fields: the same setup over the same bytes makes the
	// same picture.
	uint32_t keepAddr, keepLen;
	bool const keepable = viRange(&s, &keepAddr, &keepLen);
	size_t const hiddenLen = s.wide ? 0 : keepLen / 2, keepBytes = keepLen + hiddenLen;
	struct ViKept* kept = n->viKept;
	if (keepable && kept && kept->valid && kept->addr == keepAddr && kept->len == keepLen && !memcmp(&kept->setup, &s, sizeof(s)) &&
	    !memcmp(kept->bytes, &n->rdram[keepAddr], keepLen) && !memcmp(kept->bytes + keepLen, &n->hidden[keepAddr >> 1], hiddenLen)) {
		memcpy(out, kept->picture, (size_t) w * h * 4);
		return w * h;
	}
	// With gamma on, the VI outputs the square root of each channel (scaled to 8 bits).
	if (!viGamma[255]) {
		for (int i = 0; i < 256; ++i) {
			int g = 0;
			while ((g + 1) * (g + 1) <= i * 256) {
				++g;
			}
			viGamma[i] = (uint8_t) (g > 255 ? 255 : g);
		}
	}
	if (n->rdpThreads > 0 && !n->viPool) {
		n->viPool = viPoolStart(n->rdpThreads);
	}
	if (n->viPool && h >= 32) {
		viPoolRun(n, n->viPool, &s, out);
	} else {
		static struct ViBand band;
		viRows(n, &s, out, 0, h, &band);
	}
	if (!kept && (kept = n->viKept = calloc(1, sizeof(*kept))) == NULL) {
		return w * h;
	}
	kept->valid = false;
	size_t const pixels = (size_t) w * h * 4;
	if (keepable && keepBytes > kept->room) {
		uint8_t* more = realloc(kept->bytes, keepBytes);
		kept->bytes = more ? more : kept->bytes;
		kept->room = more ? keepBytes : kept->room;
	}
	if (keepable && pixels > kept->pictureRoom) {
		uint32_t* more = realloc(kept->picture, pixels);
		kept->picture = more ? more : kept->picture;
		kept->pictureRoom = more ? pixels : kept->pictureRoom;
	}
	if (keepable && keepBytes <= kept->room && pixels <= kept->pictureRoom) {
		kept->setup = s;
		kept->addr = keepAddr;
		kept->len = keepLen;
		memcpy(kept->bytes, &n->rdram[keepAddr], keepLen);
		memcpy(kept->bytes + keepLen, &n->hidden[keepAddr >> 1], hiddenLen);
		memcpy(kept->picture, out, pixels);
		kept->valid = true;
	}
	return w * h;
}

int n64Picture(struct N64* n, uint32_t* out, int* width, int* height) {
	if (!n->skip.on) {
		return viPicture(n, out, width, height);
	}
	// A skipped frame shows the picture before it again.
	if (frameSkipped(n, n->vi.reg[1] & 0xFFFFFF) && n->lastPicture && n->lastWidth) {
		*width = n->lastWidth, *height = n->lastHeight;
		memcpy(out, n->lastPicture, (size_t) n->lastWidth * n->lastHeight * 4);
		return n->lastWidth * n->lastHeight;
	}
	int count = viPicture(n, out, width, height);
	if (!n->lastPicture) {
		n->lastPicture = malloc(640 * 480 * 4);
	}
	if (n->lastPicture && count) {
		memcpy(n->lastPicture, out, (size_t) count * 4);
		n->lastWidth = *width, n->lastHeight = *height;
	}
	return count;
}
