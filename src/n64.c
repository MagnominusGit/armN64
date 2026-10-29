#include "n64.h"

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

// One framebuffer pixel for the VI: 8-bit channels and its coverage (from the hidden bits for
// 16-bit pixels, the top alpha bits for 32-bit ones).
struct ViPixel {
	int r, g, b, cvg;
};

static struct ViPixel viRead(struct N64* n, bool wide, uint32_t origin, int64_t i) {
	if (wide) {
		const uint8_t* p = &n->rdram[(origin + (uint32_t) i * 4) & (RDRAM_SIZE - 4)];
		return (struct ViPixel) { p[0], p[1], p[2], (p[3] >> 5) & 7 };
	}
	uint32_t a = (origin + (uint32_t) i * 2) & (RDRAM_SIZE - 2);
	uint16_t px = (uint16_t) (n->rdram[a] << 8 | n->rdram[a + 1]);
	int r = (px >> 11) & 31, g = (px >> 6) & 31, b = (px >> 1) & 31;
	return (struct ViPixel) { r << 3 | r >> 2, g << 3 | g >> 2, b << 3 | b >> 2, (px & 1) << 2 | n->hidden[a >> 1] };
}

// The second largest and second smallest of a few values (the largest and smallest when they
// repeat), as the VI's edge filter finds them.
static void penumbra(const int* v, int count, int* low, int* high) {
	int posMax = 0, posMin = 0, penMax = v[0], penMin = v[0];
	for (int i = 1; i < count; ++i) {
		if (v[i] > v[posMax]) {
			penMax = v[posMax];
			posMax = i;
		} else if (v[i] < v[posMin]) {
			penMin = v[posMin];
			posMin = i;
		}
	}
	if (penMax != v[posMax]) {
		for (int i = posMax + 1; i < count; ++i) {
			penMax = v[i] > penMax ? v[i] : penMax;
		}
	}
	if (penMin != v[posMin]) {
		for (int i = posMin + 1; i < count; ++i) {
			penMin = v[i] < penMin ? v[i] : penMin;
		}
	}
	*low = penMin;
	*high = penMax;
}

// The VI's fetch of a pixel: with antialiasing, a partly covered pixel moves toward the colour the
// fully covered pixels around it agree on, by the coverage it lacks; with the dither filter, a fully
// covered one moves a step toward each of its eight neighbours.
static struct ViPixel viFetch(struct N64* n, bool wide, uint32_t origin, int64_t i, int64_t stride, bool aa, bool dither) {
	struct ViPixel c = viRead(n, wide, origin, i);
	if (!aa) {
		c.cvg = 7;
	}
	int* ch[3] = { &c.r, &c.g, &c.b };
	if (c.cvg == 7) {
		if (dither) {
			static const int8_t dx[8] = { -1, 0, 1, -1, 0, 1, -1, 1 }, dy[8] = { -1, -1, -1, 1, 1, 1, 0, 0 };
			int base[3] = { c.r >> 3, c.g >> 3, c.b >> 3 };
			for (int k = 0; k < 8; ++k) {
				struct ViPixel o = viRead(n, wide, origin, i + dy[k] * stride + dx[k]);
				int nb[3] = { o.r >> 3, o.g >> 3, o.b >> 3 };
				for (int j = 0; j < 3; ++j) {
					*ch[j] += (base[j] < nb[j]) - (base[j] > nb[j]);
				}
			}
		}
		return c;
	}
	int values[3][7] = { { c.r }, { c.g }, { c.b } }, count = 1;
	const int64_t around[6] = { -stride - 1, -stride + 1, -2, 2, stride - 1, stride + 1 };
	for (int k = 0; k < 6; ++k) {
		struct ViPixel o = viRead(n, wide, origin, i + around[k]);
		if (o.cvg == 7) {
			values[0][count] = o.r, values[1][count] = o.g, values[2][count] = o.b;
			++count;
		}
	}
	for (int j = 0; j < 3; ++j) {
		int low, high;
		penumbra(values[j], count, &low, &high);
		int col = low + high - (*ch[j] << 1);
		*ch[j] = ((((col * (7 - c.cvg)) + 4) >> 3) + *ch[j]) & 255;
	}
	return c;
}

// The divot filter: each channel the median of a pixel and its neighbours, unless all three are
// fully covered.
static int median3(int a, int b, int c) {
	return a > b ? (b > c ? b : a > c ? c : a) : (a > c ? a : b > c ? c : b);
}

// The picture the VI scans out, scaled by its X and Y scale registers, as 0x00RRGGBB.
int n64Picture(struct N64* n, uint32_t* out, int* width, int* height) {
	rdpWait(n);
	uint32_t* v = n->vi.reg;
	int type = v[0] & 3;
	uint32_t origin = v[1] & 0xFFFFFF;
	uint32_t stride = v[2] & 0xFFF;
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
	if (type < 2 || w <= 0 || h <= 0 || stride == 0) {
		*width = *height = 0;
		return 0;
	}
	if (w > 640) {
		w = 640;
	}
	if (h > 480) {
		h = 480;
	}
	// With gamma on, the VI outputs the square root of each channel (scaled to 8 bits).
	uint8_t gamma[256];
	for (int i = 0; i < 256; ++i) {
		int s = 0;
		while ((s + 1) * (s + 1) <= i * 256) {
			++s;
		}
		gamma[i] = (uint8_t) (s > 255 ? 255 : s);
	}
	bool gammaOn = v[0] & 8, divot = v[0] & 16, dither = v[0] & 0x10000, aa = ((v[0] >> 8) & 3) <= 1, wide = type == 3;
	struct ViPixel line[642];
	for (int y = 0; y < h; ++y) {
		int64_t row = (int64_t) y * stride;
		for (int x = -1; x <= w; ++x) {
			line[x + 1] = viFetch(n, wide, origin, row + x, stride, aa, dither);
		}
		for (int x = 0; x < w; ++x) {
			struct ViPixel c = line[x + 1];
			if (divot) {
				const struct ViPixel* l = &line[x], * r = &line[x + 2];
				if ((c.cvg & l->cvg & r->cvg) != 7) {
					c.r = median3(l->r, c.r, r->r), c.g = median3(l->g, c.g, r->g), c.b = median3(l->b, c.b, r->b);
				}
			}
			uint32_t px = (uint32_t) c.r << 16 | (uint32_t) c.g << 8 | (uint32_t) c.b;
			if (gammaOn) {
				px = (uint32_t) gamma[c.r] << 16 | (uint32_t) gamma[c.g] << 8 | gamma[c.b];
			}
			out[y * w + x] = px;
		}
	}
	*width = w;
	*height = h;
	return w * h;
}
