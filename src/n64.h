#ifndef N64_H
#define N64_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CPU_HZ 93750000
// About 50 microseconds for the PIF to take or return a 64-byte block.
#define SI_DMA_CYCLES 4608
#define RDRAM_SIZE (8 << 20)

// MI interrupt lines.
enum { MI_SP = 1, MI_SI = 2, MI_AI = 4, MI_VI = 8, MI_PI = 16, MI_DP = 32 };

// CP0 registers by number.
enum {
	CP0_INDEX = 0, CP0_RANDOM = 1, CP0_ENTRYLO0 = 2, CP0_ENTRYLO1 = 3, CP0_CONTEXT = 4,
	CP0_PAGEMASK = 5, CP0_WIRED = 6, CP0_BADVADDR = 8, CP0_COUNT = 9, CP0_ENTRYHI = 10,
	CP0_COMPARE = 11, CP0_STATUS = 12, CP0_CAUSE = 13, CP0_EPC = 14, CP0_PRID = 15,
	CP0_CONFIG = 16, CP0_LLADDR = 17, CP0_XCONTEXT = 20, CP0_TAGLO = 28, CP0_TAGHI = 29,
	CP0_ERROREPC = 30,
};

// Exception codes.
enum {
	EXC_INT = 0, EXC_MOD = 1, EXC_TLBL = 2, EXC_TLBS = 3, EXC_ADEL = 4, EXC_ADES = 5,
	EXC_SYS = 8, EXC_BP = 9, EXC_RI = 10, EXC_CPU = 11, EXC_OV = 12, EXC_TR = 13, EXC_FPE = 15,
};

struct TLBEntry {
	uint64_t entryHi;
	uint64_t entryLo[2];
	uint32_t pageMask;
};

struct CPU {
	uint64_t gpr[32];
	uint64_t hi, lo;
	uint64_t pc;
	uint64_t nextPc;
	uint64_t curPc;
	bool inDelay;
	bool nextInDelay;
	bool llBit;
	uint64_t cop0[32];
	struct TLBEntry tlb[32];
	uint64_t countBase;
	// FPU registers as raw bits; how the 32 registers pair up depends on Status.FR.
	uint64_t fpr[32];
	uint32_t fcr31;
	bool exception;
};

struct SP {
	uint8_t mem[0x2000];
	uint32_t memAddr, dramAddr, status, semaphore, pc;
};

struct RSP {
	uint32_t r[32];
	uint32_t nextPc;
	int16_t vr[32][8];
	int64_t acc[8];
	uint16_t vco, vcc;
	uint8_t vce;
	int16_t divIn;
	uint32_t divOut;
	bool divDp;
	// Pipeline timing: the RSP's own clock and when each register's pending result is ready.
	uint64_t clock;
	uint64_t scalarReady[32];
	uint64_t vectorReady[32];
	// Dual issue: the previous instruction, when it started, and whether it could still take a partner.
	uint32_t lastOp, lastPc;
	uint64_t lastStart;
	bool lastAlone;
	bool inDelay;
};

struct DP {
	uint32_t start, end, current, status;
	// When the RDP finishes the commands it has, and when its pending full-sync interrupt is due.
	uint64_t busyUntil, syncAt;
};

struct VI {
	uint32_t reg[14];
	uint32_t line;
};

struct AI {
	uint32_t dramAddr, len, control, dacrate, bitrate;
	// Up to two buffers: the one playing and one queued behind it.
	int queued;
	uint32_t queueLen[2];
	uint64_t startedAt, endsAt;
};

struct PI {
	uint32_t dramAddr, cartAddr, status;
	uint64_t busyUntil;
	uint32_t timing[8];
	uint32_t romLatch;
	bool romLatched;
};

struct SI {
	uint32_t dramAddr, status;
	uint64_t busyUntil;
};

struct RDP;

struct N64 {
	uint8_t* rdram;
	uint8_t* rom;
	size_t romSize;
	uint8_t pifRam[64];
	uint8_t save[0x20000];
	uint8_t eeprom[2048];
	bool eeprom16k;
	int cic;

	struct CPU cpu;
	struct SP sp;
	struct RSP rsp;
	struct DP dp;
	struct RDP* rdp;
	struct VI vi;
	struct AI ai;
	struct PI pi;
	struct SI si;
	uint32_t miMode, miIntr, miMask;
	uint32_t riReg[8];

	uint64_t cycles;
	uint64_t nextVi;
	// CPU cycles per video field: 60 fields a second for NTSC and MPAL, 50 for PAL.
	uint64_t fieldCycles;
	uint64_t fields;
	uint32_t buttons;
	const char* unimplemented;
	// Activity counts for diagnostics.
	struct {
		uint64_t piDmas, piBytes, rspTasks, dpSyncs, aiBuffers, siDmas;
	} stats;
};

// n64.c
struct N64* n64Create(void);
void n64Destroy(struct N64*);
bool n64LoadRom(struct N64*, const uint8_t* data, size_t size);
void n64RunField(struct N64*);
void n64Interrupt(struct N64*, uint32_t line, bool set);
void aiUpdate(struct N64*);
int n64Picture(struct N64*, uint32_t* out, int* width, int* height);

// cpu.c
void cpuReset(struct N64*);
void cpuStep(struct N64*);
void cpuException(struct N64*, int code, int coprocessor);
void cpuCheckInterrupts(struct N64*);
uint32_t cpuCount(struct N64*);
bool cpuPeekAddress(struct N64*, uint64_t vaddr, uint32_t* paddr);

// rsp.c
void rspInitTables(void);
void rspStep(struct N64*);

// fpu.c
void fpuExecute(struct N64*, uint32_t op);

// bus.c
uint8_t busRead8(struct N64*, uint32_t addr);
uint16_t busRead16(struct N64*, uint32_t addr);
uint32_t busRead32(struct N64*, uint32_t addr);
uint64_t busRead64(struct N64*, uint32_t addr);
void busWrite8(struct N64*, uint32_t addr, uint8_t value);
void busWrite16(struct N64*, uint32_t addr, uint16_t value);
void busWrite32(struct N64*, uint32_t addr, uint32_t value);
void busWrite64(struct N64*, uint32_t addr, uint64_t value);

// rdp.c
void rdpRun(struct N64*);
void rdpDestroy(struct N64*);

// carts.c
bool cartEeprom16k(const uint8_t* rom);

// pif.c
void pifBoot(struct N64*);
void pifRun(struct N64*);

static inline uint32_t be32(const uint8_t* p) {
	return (uint32_t) p[0] << 24 | (uint32_t) p[1] << 16 | (uint32_t) p[2] << 8 | p[3];
}

static inline void put32(uint8_t* p, uint32_t v) {
	p[0] = v >> 24;
	p[1] = v >> 16;
	p[2] = v >> 8;
	p[3] = v;
}

static inline uint64_t sext32(uint32_t v) {
	return (uint64_t) (int64_t) (int32_t) v;
}

#endif
