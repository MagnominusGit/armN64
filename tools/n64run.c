// Runs a cartridge headless. Usage: n64run rom fields [picture.ppm]
#include "n64.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint8_t* readFile(const char* path, size_t* size) {
	FILE* f = fopen(path, "rb");
	if (!f) {
		return NULL;
	}
	fseek(f, 0, SEEK_END);
	*size = (size_t) ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t* data = malloc(*size);
	if (data && fread(data, 1, *size, f) != *size) {
		free(data);
		data = NULL;
	}
	fclose(f);
	return data;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: %s rom fields [picture.ppm]\n", argv[0]);
		return 1;
	}
	size_t size;
	uint8_t* data = readFile(argv[1], &size);
	struct N64* n = n64Create();
	if (!data || !n || !n64LoadRom(n, data, size)) {
		fprintf(stderr, "can't load %s\n", argv[1]);
		return 1;
	}
	free(data);
	printf("CIC %d\n", n->cic);
	int fields = atoi(argv[2]);
	const char* reported = NULL;
	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for (int f = 0; f < fields; ++f) {
		n64RunField(n);
		if (n->unimplemented && n->unimplemented != reported) {
			printf("field %d: %s (pc %08x)\n", f, n->unimplemented, (uint32_t) n->cpu.curPc);
			reported = n->unimplemented;
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);
	double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
	printf("%d fields in %.2f s (%.1fx real time), pc %08x\n", fields, secs, fields / 60.0 / secs,
	       (uint32_t) n->cpu.pc);
	if (getenv("N64STATE")) {
		struct CPU* c = &n->cpu;
		printf("sp status %04x pc %03x, dp status %x, mi intr %02x mask %02x, pi status %x, vi intr %d\n",
		       n->sp.status, n->sp.pc, n->dp.status, n->miIntr, n->miMask, n->pi.status, n->vi.reg[3]);
		printf("cop0 status %08x cause %08x epc %08x\n", (uint32_t) c->cop0[CP0_STATUS],
		       (uint32_t) c->cop0[CP0_CAUSE], (uint32_t) c->cop0[CP0_EPC]);
		for (int k = -8; k < 4; ++k) {
			uint32_t a = (uint32_t) c->pc + k * 4;
			printf("%08x %08x%s\n", a, busRead32(n, a & 0x1FFFFFFF), k == 0 ? " <" : "");
		}
		for (int i = 0; i < 32; ++i) {
			printf("r%-2d %016llx%s", i, (unsigned long long) c->gpr[i], i % 4 == 3 ? "\n" : "  ");
		}
	}
	if (argc > 3) {
		static uint32_t picture[640 * 480];
		int w, h;
		n64Picture(n, picture, &w, &h);
		FILE* out = fopen(argv[3], "wb");
		if (out && w && h) {
			fprintf(out, "P6\n%d %d\n255\n", w, h);
			for (int i = 0; i < w * h; ++i) {
				uint8_t px[3] = { picture[i] >> 16, picture[i] >> 8, picture[i] };
				fwrite(px, 1, 3, out);
			}
		}
		if (out) {
			fclose(out);
		}
		printf("picture %dx%d\n", w, h);
	}
	n64Destroy(n);
	return 0;
}
