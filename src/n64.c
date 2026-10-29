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
	n->vpages = malloc(2 * VPAGES * sizeof(*n->vpages));
	if (!n->rdram || !n->vpages) {
		free(n->rdram);
		free(n->vpages);
		free(n);
		return NULL;
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
		free(n->rdram);
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

// One video field: the CPU runs until the field ends, with the VI interrupt at the V_INTR line.
// With a slice, the CPU runs ahead of the RSP by up to that many cycles (or to the next event),
// and the RSP catches up then or when the CPU touches what it shares with it. Without one, the
// two step in lockstep.
void n64RunField(struct N64* n) {
	uint64_t start = n->nextVi - n->fieldCycles;
	uint32_t lines = (n->vi.reg[6] & 0x3FF) + 1;
	uint64_t intrAt = start + (uint64_t) (n->vi.reg[3] & 0x3FF) * n->fieldCycles / lines;
	bool raised = false;
	while (n->cycles < n->nextVi) {
		if (!raised && n->cycles >= intrAt) {
			n64Interrupt(n, MI_VI, true);
			raised = true;
		}
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
			if (!raised && intrAt < until) {
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
	if (!raised) {
		n64Interrupt(n, MI_VI, true);
	}
	n->nextVi += n->fieldCycles;
	++n->fields;
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
	bool gammaOn = v[0] & 8;
	for (int y = 0; y < h; ++y) {
		uint32_t row = origin + (uint32_t) y * stride * (type == 3 ? 4 : 2);
		for (int x = 0; x < w; ++x) {
			uint32_t c;
			if (type == 3) {
				const uint8_t* p = &n->rdram[(row + x * 4) & (RDRAM_SIZE - 4)];
				c = (uint32_t) p[0] << 16 | (uint32_t) p[1] << 8 | p[2];
			} else {
				const uint8_t* p = &n->rdram[(row + x * 2) & (RDRAM_SIZE - 2)];
				uint16_t px = (uint16_t) (p[0] << 8 | p[1]);
				uint32_t r = (px >> 11) & 31, g = (px >> 6) & 31, b = (px >> 1) & 31;
				c = (r << 3 | r >> 2) << 16 | (g << 3 | g >> 2) << 8 | (b << 3 | b >> 2);
			}
			if (gammaOn) {
				c = (uint32_t) gamma[c >> 16] << 16 | (uint32_t) gamma[(c >> 8) & 255] << 8 | gamma[c & 255];
			}
			out[y * w + x] = c;
		}
	}
	*width = w;
	*height = h;
	return w * h;
}
