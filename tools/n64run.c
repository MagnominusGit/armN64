// Runs a cartridge headless. Usage: n64run rom fields [picture.ppm]
#define _GNU_SOURCE
#include "n64.h"
#include "rdp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>
#include <sys/syscall.h>

// N64SAMPLE=file: the main thread's program counter every half millisecond of its CPU time, as
// 64-bit words, for a profile where perf isn't there (N64_PERFMAP=1 names the compiled blocks).
enum { SAMPLES = 1 << 22 };
static uint64_t* samples;
static volatile size_t sampleCount;

static void sample(int sig, siginfo_t* info, void* context) {
	ucontext_t* uc = context;
#if defined(__aarch64__)
	uint64_t pc = uc->uc_mcontext.pc;
#else
	uint64_t pc = (uint64_t) uc->uc_mcontext.gregs[REG_RIP];
#endif
	if (sampleCount < SAMPLES) {
		samples[sampleCount++] = pc;
	}
}

static void sampleStart(void) {
	samples = malloc(SAMPLES * sizeof(*samples));
	struct sigaction sa = { 0 };
	sa.sa_sigaction = sample;
	sa.sa_flags = SA_SIGINFO | SA_RESTART;
	sigaction(SIGPROF, &sa, NULL);
	struct sigevent ev = { 0 };
	ev.sigev_notify = SIGEV_THREAD_ID;
	ev.sigev_signo = SIGPROF;
	ev._sigev_un._tid = (pid_t) syscall(SYS_gettid);
	timer_t timer;
	timer_create(CLOCK_THREAD_CPUTIME_ID, &ev, &timer);
	struct itimerspec every = { { 0, 500000 }, { 0, 500000 } };
	timer_settime(timer, 0, &every, NULL);
}

// The samples, and beside them (path.maps) where the libraries were loaded.
static void sampleWrite(const char* path) {
	FILE* f = fopen(path, "wb");
	if (f) {
		fwrite(samples, sizeof(*samples), sampleCount, f);
		fclose(f);
	}
	char mapsPath[512];
	snprintf(mapsPath, sizeof(mapsPath), "%s.maps", path);
	FILE* in = fopen("/proc/self/maps", "r");
	FILE* out = fopen(mapsPath, "w");
	char line[512];
	while (in && out && fgets(line, sizeof(line), in)) {
		fputs(line, out);
	}
	if (in) {
		fclose(in);
	}
	if (out) {
		fclose(out);
	}
}

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
		// N64DDSAVE=path: the disk's changes (ddChanges) put back before the run, saved after it.
		size_t changesSize;
		uint8_t* changes = getenv("N64DDSAVE") ? readFile(getenv("N64DDSAVE"), &changesSize) : NULL;
		if (changes && !ddApplyChanges(n, changes, changesSize)) {
			fprintf(stderr, "the disk changes in %s don't fit\n", getenv("N64DDSAVE"));
		}
		free(changes);
	}
	if (!data || !n || (!diskGame && !n64LoadRom(n, data, size)) || (diskGame && !iplPath)) {
		fprintf(stderr, "can't load %s\n", argv[1]);
		return 1;
	}
	free(data);
	printf("CIC %d\n", n->cic);
	int fields = atoi(argv[2]);
	const char* reported = NULL;
	// N64PICS=1: a picture every field as a frontend takes it (on the GPU when it can); N64PROF=1
	// prints the wall, main-thread and picture times.
	double picSecs = 0;
	bool pics = getenv("N64PICS") != NULL;
	static uint32_t frame[640 * 480];
	struct timespec t0, t1;
	if (getenv("N64SAMPLE")) {
		sampleStart();
	}
	// N64_TPAK=Game Boy ROM: the cartridge in controller 1's Transfer Pak; N64_TPAKSAV=its save RAM,
	// loaded before the run and saved after.
	if (getenv("N64_TPAK")) {
		size_t gbSize;
		uint8_t* gb = readFile(getenv("N64_TPAK"), &gbSize);
		if (!gb || !gbCartLoad(&n->tpak[0].cart, gb, gbSize)) {
			fprintf(stderr, "can't load the Game Boy ROM %s\n", getenv("N64_TPAK"));
		}
		free(gb);
		size_t savSize;
		uint8_t* sav = getenv("N64_TPAKSAV") ? readFile(getenv("N64_TPAKSAV"), &savSize) : NULL;
		if (sav) {
			memcpy(n->tpak[0].cart.ram, sav, savSize < n->tpak[0].cart.ramSize ? savSize : n->tpak[0].cart.ramSize);
		}
		free(sav);
	}
	// N64SAVELOAD / N64SAVEDUMP=path: the saves before the run and after it, as the machine holds them
	// (the EEPROM, the four Controller Paks, the SRAM, the flash).
	struct {
		uint8_t* at;
		size_t size;
	} const saves[] = { { n->eeprom, sizeof(n->eeprom) }, { &n->paks[0][0], sizeof(n->paks) }, { n->sram, sizeof(n->sram) }, { n->flash, sizeof(n->flash) } };
	if (getenv("N64SAVELOAD")) {
		FILE* f = fopen(getenv("N64SAVELOAD"), "rb");
		for (size_t i = 0; f && i < sizeof(saves) / sizeof(*saves); ++i) {
			if (fread(saves[i].at, 1, saves[i].size, f) != saves[i].size) {
				break;
			}
		}
		if (f) {
			fclose(f);
		}
	}
	// N64INPUT="from-to:pad,...": controller 1 holds pad (buttons and stick as the PIF reads them, in
	// hex) for fields from to to.
	// N64AUDIODUMP=path: what the AI played, as 16-bit stereo samples, with its last rate printed.
	if (getenv("N64AUDIODUMP")) {
		n->audioCapacity = 2 * 48000 * (size_t) (fields / 50 + 2);
		n->audio = calloc(n->audioCapacity, sizeof(int16_t));
	}
	// N64STATEIN=path: a save state loaded before the run (made by the same build); N64STATEOUT=path:
	// one saved after it.
	if (getenv("N64STATEIN")) {
		size_t stateSize;
		uint8_t* saved = readFile(getenv("N64STATEIN"), &stateSize);
		if (!saved || !n64StateLoad(n, saved, stateSize, false)) {
			fprintf(stderr, "can't load the state %s\n", getenv("N64STATEIN"));
			return 1;
		}
		free(saved);
	}
	const char* input = getenv("N64INPUT");
	// N64STATETEST=field: a save state there, then after the run it is loaded and the rest run again;
	// RDRAM and the picture must come out the same.
	int stateAt = getenv("N64STATETEST") ? atoi(getenv("N64STATETEST")) : -1;
	bool quick = getenv("N64STATEQUICK") != NULL;
	void* state = NULL;
	bool again = false;
	uint64_t firstHash = 0;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for (int f = 0; f < fields; ++f) {
		if (f == stateAt && !again) {
			state = malloc(n64StateSize());
			if (!state || !n64StateSave(n, state, n64StateSize(), quick)) {
				fprintf(stderr, "can't save a state\n");
				return 1;
			}
		}
		if (input) {
			uint32_t pad = 0;
			for (const char* p = input; p && *p; p = strchr(p, ',') ? strchr(p, ',') + 1 : NULL) {
				int from, to;
				unsigned held;
				if (sscanf(p, "%d-%d:%x", &from, &to, &held) == 3 && f >= from && f < to) {
					pad |= held;
				}
			}
			n->pad[0] = pad;
		}
		// N64RUNAHEAD=1: each field as RetroArch's run-ahead runs it: a quick state, the field, the
		// state loaded and the field again.
		static void* ahead;
		if (getenv("N64RUNAHEAD")) {
			if (!ahead) {
				ahead = malloc(n64StateSize());
			}
			n64StateSave(n, ahead, n64StateSize(), true);
			n64RunField(n);
			n64StateLoad(n, ahead, n64StateSize(), true);
		}
		// N64DIRTYTEST=1: every page whose bytes changed in the field must have been marked written.
		static uint8_t* before;
		static long unmarked;
		bool dirtyTest = getenv("N64DIRTYTEST") != NULL;
		if (dirtyTest) {
			if (!before) {
				before = malloc(RDRAM_SIZE);
			}
			rdpWait(n);
			memcpy(before, n->rdram, RDRAM_SIZE);
			memset(n->dirtyPages, 0xFF, sizeof(n->dirtyPages));
		}
		n64RunField(n);
		if (dirtyTest) {
			rdpWait(n);
			for (uint32_t p = 0; p < (RDRAM_SIZE >> 12); ++p) {
				if (n->dirtyPages[p] && memcmp(before + (p << 12), n->rdram + (p << 12), 4096)) {
					if (unmarked++ < 10) {
						printf("field %d: page %04x changed unmarked\n", f, p);
					}
				}
			}
			if (f + 1 == fields) {
				printf("dirty test: %ld unmarked page changes\n", unmarked);
			}
		}
		// N64RUMBLELOG: when controller 1's Rumble Pak motor turns on and off.
		static bool rumbling;
		if (getenv("N64RUMBLELOG") && n->rumble[0] != rumbling) {
			rumbling = n->rumble[0];
			printf("field %d: rumble %s\n", f, rumbling ? "on" : "off");
		}
		// The game's frames: how often the VI starts showing another framebuffer.
		static uint32_t lastOrigin;
		static int flips;
		if (n->vi.reg[1] != lastOrigin) {
			lastOrigin = n->vi.reg[1];
			++flips;
		}
		if (getenv("N64FPS") && (f + 1) % 300 == 0) {
			printf("fields %d-%d: %.1f game frames a second\n", f - 298, f + 1, flips / 5.0);
			flips = 0;
		}
		if (pics) {
			struct timespec a, b;
			int pw, ph;
			clock_gettime(CLOCK_MONOTONIC, &a);
			// As the core takes it: from the GL renderer's framebuffer when the GPU drew it last.
			unsigned framebuffer;
			int px, py;
			if (!n64GlPicture(n, &framebuffer, &px, &py, &pw, &ph) && !n64PictureStart(n, &pw, &ph)) {
				n64Picture(n, frame, &pw, &ph);
			}
			clock_gettime(CLOCK_MONOTONIC, &b);
			picSecs += (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
		}
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
		if (f + 1 == fields && state) {
			rdpWait(n);
			if (n->glr) {
				glrSync(n);
			}
			uint64_t hash = 1469598103934665603ull;
			for (uint32_t i = 0; i < RDRAM_SIZE; ++i) {
				hash = (hash ^ n->rdram[i]) * 1099511628211ull;
			}
			if (!again) {
				firstHash = hash;
				again = true;
				if (!n64StateLoad(n, state, n64StateSize(), quick)) {
					fprintf(stderr, "can't load the state\n");
					return 1;
				}
				f = stateAt - 1;
			} else {
				printf("state test: RDRAM %s after loading at field %d\n", hash == firstHash ? "the same" : "DIFFERENT", stateAt);
			}
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);
	if (getenv("N64SAMPLE")) {
		sampleWrite(getenv("N64SAMPLE"));
	}
	double secs = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
	printf("%d fields in %.2f s (%.1fx real time), pc %08x\n", fields, secs, fields / 60.0 / secs,
	       (uint32_t) n->cpu.pc);
	if (getenv("N64PROF")) {
		struct timespec mc;
		clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mc);
		printf("PROF wall %.3f main_cpu %.3f pictures %.3f gl_syncs %llu gl_rows %llu gl_textures %llu gl_loads %llu idle %llu%% block_gens %u gl_batches %llu\n", secs, mc.tv_sec + mc.tv_nsec / 1e9, picSecs,
		       (unsigned long long) n->stats.glSyncs, (unsigned long long) n->stats.glRows, (unsigned long long) n->stats.glTextures,
		       (unsigned long long) n->stats.glTargetRows,
		       (unsigned long long) (n->cycles ? n->stats.idleCycles * 100 / n->cycles : 0), n->blockGen, (unsigned long long) n->stats.glBatches);
	}
	// N64GFXREPORT=1: graphics tasks run at high level and on the RSP, and the last microcode there.
	if (getenv("N64GFXREPORT")) {
		printf("GFX hle %llu rsp %llu ucode %s\n", (unsigned long long) n->stats.gfxHle, (unsigned long long) n->stats.gfxRsp,
		       n->stats.gfxRsp ? n->stats.gfxRspUcode : "-");
	}
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
	// N64GLSHOT=path: the GL renderer's picture as it drew it (at N64_GL_SCALE), as a PPM.
	unsigned glFramebuffer;
	int gx, gy, gw, gh;
	if (getenv("N64GLSHOT") && n64GlPicture(n, &glFramebuffer, &gx, &gy, &gw, &gh)) {
		int k = n->glrScale;
		uint32_t* image = malloc((size_t) gw * gh * k * k * 4);
		FILE* f = fopen(getenv("N64GLSHOT"), "wb");
		if (image && f && glrRead(n, glFramebuffer, gx, gy, gw, gh, image)) {
			fprintf(f, "P6\n%d %d\n255\n", gw * k, gh * k);
			for (int y = 0; y < gh * k; ++y) {
				for (int x = 0; x < gw * k; ++x) {
					uint32_t c = image[(size_t) y * gw * k + x];
					fputc(c & 255, f), fputc((c >> 8) & 255, f), fputc((c >> 16) & 255, f);
				}
			}
		}
		if (f) {
			fclose(f);
		}
		free(image);
	}
	rdpWait(n);
	if (getenv("N64SAVEDUMP")) {
		FILE* f = fopen(getenv("N64SAVEDUMP"), "wb");
		for (size_t i = 0; f && i < sizeof(saves) / sizeof(*saves); ++i) {
			fwrite(saves[i].at, 1, saves[i].size, f);
		}
		if (f) {
			fclose(f);
		}
		printf("save chip %s\n", n->saveKind == SAVE_SRAM ? "sram" : n->saveKind == SAVE_FLASH ? "flash" : "unused");
	}
	if (getenv("N64DDSAVE")) {
		size_t size;
		uint8_t* changes = ddChanges(n, &size);
		FILE* f = changes ? fopen(getenv("N64DDSAVE"), "wb") : NULL;
		if (f) {
			fwrite(changes, 1, size, f);
			fclose(f);
			printf("disk changes: %zu bytes\n", size);
		}
		free(changes);
	}
	if (getenv("N64_TPAKSAV") && n->tpak[0].cart.ramDirty) {
		FILE* f = fopen(getenv("N64_TPAKSAV"), "wb");
		if (f) {
			fwrite(n->tpak[0].cart.ram, 1, n->tpak[0].cart.ramSize, f);
			fclose(f);
			printf("Game Boy save RAM written\n");
		}
	}
	if (getenv("N64AUDIODUMP") && n->audio) {
		FILE* f = fopen(getenv("N64AUDIODUMP"), "wb");
		if (f) {
			fwrite(n->audio, sizeof(int16_t), n->audioCount, f);
			fclose(f);
			printf("audio: %zu samples at %u Hz\n", n->audioCount / 2, n->audioRate);
		}
	}
	if (getenv("N64STATEOUT")) {
		void* out = malloc(n64StateSize());
		FILE* f = out && n64StateSave(n, out, n64StateSize(), false) ? fopen(getenv("N64STATEOUT"), "wb") : NULL;
		if (f) {
			fwrite(out, 1, n64StateSize(), f);
			fclose(f);
		}
		free(out);
	}
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
