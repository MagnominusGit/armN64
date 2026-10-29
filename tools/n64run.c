// Runs a cartridge headless. Usage: n64run rom fields [picture.ppm]
#include "n64.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static void writePicture(struct N64* n, const char* path) {
	static uint32_t picture[640 * 480];
	int w, h;
	n64Picture(n, picture, &w, &h);
	FILE* out = fopen(path, "wb");
	if (!out) {
		return;
	}
	if (!w || !h) {
		w = 1;
		h = 1;
		picture[0] = 0;
	}
	fprintf(out, "P6\n%d %d\n255\n", w, h);
	for (int i = 0; i < w * h; ++i) {
		uint8_t px[3] = { picture[i] >> 16, picture[i] >> 8, picture[i] };
		fwrite(px, 1, 3, out);
	}
	fclose(out);
}

int main(int argc, char** argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: %s rom fields [picture.ppm]\n", argv[0]);
		return 1;
	}
	size_t size;
	uint8_t* data = readFile(argv[1], &size);
	struct N64* n = n64Create();
	// 64DD: N64DDIPL=the IPL ROM. A .ndd game boots from it; N64DDDISK=disk adds one to a cartridge.
	const char* iplPath = getenv("N64DDIPL");
	const char* diskPath = getenv("N64DDDISK");
	size_t len = strlen(argv[1]);
	bool diskGame = len > 4 && !strcmp(argv[1] + len - 4, ".ndd");
	if (diskGame) {
		diskPath = argv[1];
	}
	if (iplPath && n) {
		size_t iplSize, diskSize = 0;
		uint8_t* ipl = readFile(iplPath, &iplSize);
		uint8_t* disk = diskPath ? (diskGame ? data : readFile(diskPath, &diskSize)) : NULL;
		if (diskGame) {
			diskSize = size;
		}
		if (!ipl || (diskPath && !disk) || !n64LoadDisk(n, ipl, iplSize, disk, diskSize)) {
			fprintf(stderr, "can't load the 64DD IPL %s or disk %s\n", iplPath, diskPath ? diskPath : "-");
			return 1;
		}
		free(ipl);
		if (disk != data) {
			free(disk);
		}
	}
	if (!data || !n || (!diskGame && !n64LoadRom(n, data, size)) || (diskGame && !iplPath)) {
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
		// N64SHOTS=prefix N64SHOTEVERY=fields: a picture every so many fields, as prefix-<field>.ppm.
		if (getenv("N64SHOTS") && (f + 1) % atoi(getenv("N64SHOTEVERY") ? getenv("N64SHOTEVERY") : "300") == 0) {
			char path[1024];
			snprintf(path, sizeof(path), "%s-%05d.ppm", getenv("N64SHOTS"), f + 1);
			writePicture(n, path);
		}
		if (getenv("N64PCLOG") && f % 60 == 59) {
			printf("field %4d pc %08x rsp %03x%s  pi %llu (%llu KiB)  rsp tasks %llu  rdp syncs %llu  ai %llu  si %llu\n", f + 1,
			       (uint32_t) n->cpu.pc, n->sp.pc, n->sp.status & 1 ? " halted" : "", (unsigned long long) n->stats.piDmas, (unsigned long long) (n->stats.piBytes >> 10),
			       (unsigned long long) n->stats.rspTasks, (unsigned long long) n->stats.dpSyncs,
			       (unsigned long long) n->stats.aiBuffers, (unsigned long long) n->stats.siDmas);
		}
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
		printf("rsp code:");
		for (int k = -6; k < 4; ++k) {
			uint32_t a = (n->sp.pc + k * 4) & 0xFFC;
			printf("%s%03x:%08x", k == 0 ? " >" : " ", a, be32(&n->sp.mem[0x1000 + a]));
		}
		printf("\nrsp regs:");
		for (int i = 0; i < 32; ++i) {
			printf(" %d=%x", i, n->rsp.r[i]);
		}
		printf("\ndp start %06x current %06x end %06x\n", n->dp.start, n->dp.current, n->dp.end);
		printf("vi:");
		for (int i = 0; i < 14; ++i) {
			printf(" %08x", n->vi.reg[i]);
		}
		printf("\n");
		printf("cop0 status %08x cause %08x epc %08x\n", (uint32_t) c->cop0[CP0_STATUS],
		       (uint32_t) c->cop0[CP0_CAUSE], (uint32_t) c->cop0[CP0_EPC]);
		for (int k = -8; k < 4; ++k) {
			uint32_t a = (uint32_t) c->pc + k * 4, p;
			if (cpuPeekAddress(n, a, &p)) {
				printf("%08x %08x%s\n", a, busRead32(n, p), k == 0 ? " <" : "");
			}
		}
		for (int i = 0; i < 32; ++i) {
			printf("r%-2d %016llx%s", i, (unsigned long long) c->gpr[i], i % 4 == 3 ? "\n" : "  ");
		}
	}
	rdpWait(n);
	if (getenv("N64RAMDUMP")) {
		FILE* f = fopen(getenv("N64RAMDUMP"), "wb");
		if (f) {
			fwrite(n->rdram, 1, RDRAM_SIZE, f);
			fclose(f);
		}
	}
	if (argc > 3) {
		writePicture(n, argv[3]);
	}
	n64Destroy(n);
	return 0;
}
