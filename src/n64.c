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
	if (!n->rdram) {
		free(n);
		return NULL;
	}
	return n;
}

void n64Destroy(struct N64* n) {
	if (n) {
		rdpDestroy(n);
		free(n->rdram);
		free(n->rom);
		free(n);
	}
}

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
	cpuReset(n);
	pifBoot(n);
	n->vi.reg[6] = 0x20D;
	n->nextVi = CYCLES_PER_FIELD;
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

// One video field: the CPU runs until the field ends, with the VI interrupt at the V_INTR line.
void n64RunField(struct N64* n) {
	uint64_t start = n->nextVi - CYCLES_PER_FIELD;
	uint32_t lines = (n->vi.reg[6] & 0x3FF) + 1;
	uint64_t intrAt = start + (uint64_t) (n->vi.reg[3] & 0x3FF) * CYCLES_PER_FIELD / lines;
	bool raised = false;
	while (n->cycles < n->nextVi) {
		if (!raised && n->cycles >= intrAt) {
			n64Interrupt(n, MI_VI, true);
			raised = true;
		}
		cpuStep(n);
		++n->cycles;
		// The RSP runs at two thirds of the CPU clock.
		if (n->cycles % 3 != 0 && !(n->sp.status & 1)) {
			rspStep(n);
		}
	}
	if (!raised) {
		n64Interrupt(n, MI_VI, true);
	}
	n->nextVi += CYCLES_PER_FIELD;
	++n->fields;
}

// The picture the VI scans out, scaled by its X and Y scale registers, as 0x00RRGGBB.
int n64Picture(struct N64* n, uint32_t* out, int* width, int* height) {
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
			out[y * w + x] = c;
		}
	}
	*width = w;
	*height = h;
	return w * h;
}
