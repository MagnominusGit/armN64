#include "n64.h"
#include "gles.h"
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

void n64Destroy(struct N64* n) {
	if (n) {
		blocksFree(n);
		jitFree(n);
		rspJitFree(n);
		rdpDestroy(n);
		glesDestroy(n->viGles);
		free(n->hle);
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

// From power-on: the PIF's boot from the cartridge, or from the 64DD's IPL without one.
static void start(struct N64* n) {
	const char* accuracy = getenv("N64_ACCURACY");
	n->accuracy = !accuracy ? ACCURACY_LLE_ACCURATE : !strcmp(accuracy, "lle-compatible") ? ACCURACY_LLE_COMPATIBLE :
	              !strcmp(accuracy, "hle") ? ACCURACY_HLE : ACCURACY_LLE_ACCURATE;
	n->compat = n->accuracy >= ACCURACY_LLE_COMPATIBLE;
	n->hleCheck = getenv("N64HLECHECK") != NULL;
	const char* mouse = getenv("N64_MOUSE");
	n->mousePort = mouse && *mouse >= '1' && *mouse <= '4' ? *mouse - '0' : 0;
	if (n->compat) {
		memset(n->rdram, 0, RDRAM_SIZE);
	}
	blocksFree(n);
	cpuReset(n);
	pifBoot(n);
	n->eeprom16k = n->rom ? cartEeprom16k(n->rom) : false;
	memset(n->eeprom, 0xFF, sizeof(n->eeprom));
	n->fieldCycles = CPU_HZ / (n->cpu.gpr[20] == 0 ? 50 : 60);
	n->vi.reg[6] = n->cpu.gpr[20] == 0 ? 0x271 : 0x20D;
	n->nextVi = n->fieldCycles;
	n->eventAt = 0;
	const char* slice = getenv("N64_SLICE");
	n->slice = slice ? strtoull(slice, NULL, 0) : 0;
	const char* blocks = getenv("N64_BLOCKS");
#ifdef N64_HAS_JIT
	const char* jit = getenv("N64_JIT");
	const char* rspJit = getenv("N64_RSPJIT");
	n->useJit = n->slice && jit && jit[0] == '1';
	n->useRspJit = rspJit && rspJit[0] == '1';
#endif
	n->useBlocks = n->useJit || (n->slice && blocks && blocks[0] == '1');
	const char* rdpThreads = getenv("N64_RDP_THREADS");
	n->rdpThreads = rdpThreads ? atoi(rdpThreads) : 0;
	const char* rdpGpu = getenv("N64_RDP_GPU");
	n->rdpGpu = rdpGpu && rdpGpu[0] == '1';
	if (n->rdpGpu && !n->rdpThreads) {
		n->rdpThreads = 1;
	}
	const char* viGpu = getenv("N64_VI_GPU");
	n->viGpu = viGpu && viGpu[0] == '1';
	rspJitFree(n);
	++n->rspCodeGen;
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

void rspCatchUp(struct N64* n) {
	if (n->catchingUp) {
		return;
	}
	// The RSP runs at two thirds of the CPU clock, on its own cycle count.
	uint64_t rspNow = n->cycles * 2 / 3;
	if (n->sp.status & 1) {
		n->rsp.clock = rspNow;
		return;
	}
	n->catchingUp = true;
	while (n->rsp.clock < rspNow && !(n->sp.status & 1)) {
		// Compiled code from the start of a block, and not from a delay slot.
		void* code;
		if (n->useRspJit && n->rsp.nextPc == ((n->sp.pc + 4) & 0xFFC) && (code = rspJitLookup(n, n->sp.pc & 0xFFC))) {
			n->rspEnter(n, code, rspNow);
		} else {
			rspStep(n);
		}
	}
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
		rspCatchUp(n);
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
	bool wide, gammaOn, divot, dither, aa;
};

static bool viSetup(struct N64* n, struct ViSetup* s) {
	uint32_t* v = n->vi.reg;
	int type = v[0] & 3;
	s->origin = v[1] & 0xFFFFFF;
	s->stride = v[2] & 0xFFF;
	int hStart = (v[9] >> 16) & 0x3FF, hEnd = v[9] & 0x3FF;
	int vStart = (v[10] >> 16) & 0x3FF, vEnd = v[10] & 0x3FF;
	uint32_t xScale = v[12] & 0xFFF, yScale = v[13] & 0xFFF;
	// In lle-compatible accuracy, an empty window (osViBlack) shows the picture anyway in the TV type's
	// usual one, as emulators that draw the RDP's image directly do.
	if (n->compat && (hEnd <= hStart || vEnd <= vStart)) {
		bool pal = n->fieldCycles == CPU_HZ / 50;
		hStart = pal ? 0x80 : 0x6C, hEnd = pal ? 0x300 : 0x2EC;
		vStart = pal ? 0x5F : 0x25, vEnd = pal ? 0x239 : 0x1FF;
	}
	int w = (int) (((hEnd - hStart) * xScale + 512) >> 10);
	int h = (int) ((((vEnd - vStart) >> 1) * yScale + 512) >> 10);
	if (type < 2 || w <= 0 || h <= 0 || s->stride == 0) {
		return false;
	}
	s->w = w > 640 ? 640 : w;
	s->h = h > 480 ? 480 : h;
	s->gammaOn = v[0] & 8, s->divot = v[0] & 16, s->dither = v[0] & 0x10000, s->aa = ((v[0] >> 8) & 3) <= 1, s->wide = type == 3;
	return true;
}

// Starts the filters on the GPU: uploads the rows they read (y-1 to h, columns -3 to w+2) and their
// hidden bits. False when the GPU can't start or the rows wrap around RDRAM.
static bool viGpuStart(struct N64* n, const struct ViSetup* s) {
	if (!n->viGles) {
		char error[512];
		n->viGles = glesCreate(viGpuSource, error, sizeof(error));
		if (!n->viGles) {
			fprintf(stderr, "GPU VI unavailable (%s), using the CPU\n", error);
			n->viGpu = false;
			return false;
		}
		fprintf(stderr, "GPU VI on %s\n", glesRenderer(n->viGles));
		glesData(n->viGles, 2, NULL, 640 * 480 * 4);
	}
	uint32_t size = s->wide ? 4 : 2;
	int64_t first = -(int64_t) s->stride - 3, last = (int64_t) s->h * s->stride + s->w + 2;
	uint32_t a0 = (s->origin + (uint32_t) first * size) & (RDRAM_SIZE - size), bytes = (uint32_t) (last - first + 1) * size;
	if (a0 + bytes > RDRAM_SIZE) {
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
	return true;
}

bool n64PictureStart(struct N64* n, int* width, int* height) {
	rdpWait(n);
	struct ViSetup s;
	if (!n->viGpu || !viSetup(n, &s) || !viGpuStart(n, &s)) {
		*width = *height = 0;
		return false;
	}
	*width = s.w;
	*height = s.h;
	return true;
}

// The picture the VI scans out, scaled by its X and Y scale registers, as 0x00RRGGBB.
int n64Picture(struct N64* n, uint32_t* out, int* width, int* height) {
	rdpWait(n);
	struct ViSetup s;
	if (!viSetup(n, &s)) {
		*width = *height = 0;
		return 0;
	}
	int w = s.w, h = s.h;
	*width = w;
	*height = h;
	if (n->viGpu && viGpuStart(n, &s)) {
		const uint32_t* picture = glesMap(n->viGles, 2, 0, (size_t) w * h * 4);
		if (picture) {
			memcpy(out, picture, (size_t) w * h * 4);
			glesUnmap(n->viGles, 2);
			return w * h;
		}
	}
	uint32_t origin = s.origin, stride = s.stride;
	bool gammaOn = s.gammaOn, divot = s.divot, dither = s.dither, aa = s.aa, wide = s.wide;
	// With gamma on, the VI outputs the square root of each channel (scaled to 8 bits).
	static uint8_t gamma[256];
	if (!gamma[255]) {
		for (int i = 0; i < 256; ++i) {
			int g = 0;
			while ((g + 1) * (g + 1) <= i * 256) {
				++g;
			}
			gamma[i] = (uint8_t) (g > 255 ? 255 : g);
		}
	}
	// Rows y-1, y and y+1 (a ring of three), decoded from column -3 to w+2.
	static struct ViRow rows[3];
	static int16_t full[3][642], restored[3][642], line[3][642], divoted[3][640];
	static uint8_t lineCvg[642];
	for (int k = 0; k < 3; ++k) {
		viDecodeRow(n, wide, origin, (int64_t) (k - 1) * stride - 3, w + 6, &rows[k]);
	}
	for (int y = 0; y < h; ++y) {
		if (y > 0) {
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
	return w * h;
}
