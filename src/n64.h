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
	// Timing: cache tags (only to cost misses; the data lives in memory) and cycles owed by this step.
	// D-cache tags are line addresses, DCACHE_NONE for an empty line.
	uint32_t icacheTag[512], dcacheTag[512];
	bool dcacheDirty[512];
	uint32_t stall;
	// Count when the timer was last checked, so a Compare value passed during a stall still fires.
	uint32_t lastCount;
};

struct SP {
	uint8_t mem[0x2000];
	uint32_t memAddr, dramAddr, status, semaphore, pc;
};

struct RSP {
	uint32_t r[32];
	uint32_t nextPc;
	int16_t vr[32][8] __attribute__((aligned(16)));
	// The 48-bit accumulator as its high, middle and low 16 bits.
	int16_t accH[8], accM[8], accL[8];
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
	// The 24-bit counters: RCP cycles since the clock counter was cleared (from clockBase in CPU
	// cycles), and RDP clocks with the command buffer busy, the pipeline busy and TMEM loading.
	uint64_t clockBase;
	uint32_t bufBusy, pipeBusy, tmemBusy;
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
	// When the write the latch holds is done on the bus.
	uint64_t romLatchUntil;
};

struct SI {
	uint32_t dramAddr, status;
	uint64_t busyUntil;
};

struct RDP;

// The N64_ACCURACY levels.
enum { ACCURACY_LLE_ACCURATE, ACCURACY_LLE_COMPATIBLE };

// CPU cycles for cache line fills and the write-back of a dirty data line (cpu.c and both code
// generators).
enum { ICACHE_FILL = 48, DCACHE_FILL = 40, DCACHE_WRITEBACK = 40 };

// Per 4 KB page of RDRAM: it holds compiled code, or RDP commands not yet drawn write or read it.
enum { PAGE_CODE = 1, PAGE_RDP_WRITES = 2, PAGE_RDP_READS = 4, PAGE_RDP = 6 };

// Compiled RSP code for an IMEM address, good while rspCodeGen is unchanged.
struct RspJitEntry {
	uint32_t gen;
	void* code;
};
struct RspBlock;

// Virtual 4 KB pages (vpages), and how many filled pages are listed for emptying them again.
enum { VPAGES = 1 << 20, VPAGE_FILLED = 1024 };
#define DCACHE_NONE 0xFFFFFFFFu

// Compiled code for a virtual address, good while blockGen is unchanged.
enum { JIT_TABLE_SIZE = 1 << 14 };
struct JitEntry {
	uint32_t vaddr, gen;
	void* code;
};

// The 64DD drive (dd.c): IPL ROM, disk (SDK layout) and where its LBAs are, buffers and registers.
struct DD {
	uint8_t* ipl;
	uint8_t* disk;
	size_t diskSize;
	int diskType;
	uint32_t sysOffset, idOffset;
	bool readable[16];
	uint32_t lbaOffset[0x10DC + 1];
	uint16_t physLba[0x4000];
	uint8_t c2s[0x400], ds[0x100], ms[0x40];
	uint16_t data, currentTrack, id;
	uint8_t currentSector, sectorSizeBuf, sectorSize, sectorBlock, setType, errors;
	bool requestUser, requestC2, busy, resetState, spindleStopped, headRetracted, writeProtect, mechaError;
	bool diskChanged, diskPresent, seeking;
	bool bmStart, bmReset, bmError, blockTransfer, readMode, c1Single, c1Double;
	bool irqMecha, irqBm;
	// When the command's response and the next sector are due (0 for none).
	uint64_t mechaAt, bmAt;
};

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
	// The boot code hasn't read RI_SELECT yet (see busRead32).
	bool coldBoot;
	// Dumps whose header checksum isn't the boot code's (Virtual Console extractions, prototypes) hang
	// in the boot code on a console. The boot code's own first reads of the two header words get the
	// checksum it works out (bits 0 and 1 of bootChecksumReads); the game sees the header as it is.
	uint32_t bootChecksum[2];
	uint8_t bootChecksumReads;

	uint64_t cycles;
	// The earliest pending event (AI, PI, SI, DP sync): the CPU runs no further before it is handled.
	uint64_t eventAt;
	// CPU cycles the CPU may run ahead of the RSP before it catches up.
	uint64_t slice;
	// N64_ACCURACY: lle-accurate (the default) is the hardware. lle-compatible is for hacks made on
	// emulators: RDRAM starts zeroed, misaligned CPU accesses align down instead of raising address
	// errors, unmapped null-pointer accesses read zero, PI DMA copies between the exact addresses
	// and finishes at once, and a blanked VI window still shows the picture. compat is true for it.
	int accuracy;
	bool compat;
	// The RSP is being caught up: its own accesses to the shared registers must not catch it up again.
	bool catchingUp;
	uint64_t nextVi;
	// Instruction blocks (blocks.c): one table of blocks per 4 KB RDRAM page that has any, and
	// what each page holds (PAGE_*): a write to compiled code throws it away.
	struct Block** blockPage[RDRAM_SIZE >> 12];
	// The entry past RDRAM has every flag set: virtual pages not yet looked up point there (vpages).
	uint8_t pageFlags[(RDRAM_SIZE >> 12) + 1];
	// One bit per RDRAM word that a block holds: writes elsewhere in a page with code keep its blocks.
	uint8_t codeWords[RDRAM_SIZE >> 5];
	uint32_t blockGen;
	bool useBlocks;
	// Compiled blocks (jit_x64.c): the code arena, how far compiled code may run, and the code
	// for recent addresses. blockGen also changes with the TLB.
	bool useJit;
	uint8_t* jitArena;
	size_t jitUsed;
	uint64_t jitLimit, jitFire;
	void (*jitEnter)(struct N64*, void* code);
	void* jitLeave;
	struct JitEntry jitTable[JIT_TABLE_SIZE];
	// Pages for compiled loads and stores: what to add to an address in each 4 KB virtual page for
	// the physical one, 1 << 20 entries for loads, then as many for stores. A page not looked up
	// since the TLB last changed, or not writable for stores, adds up to the page past RDRAM. The
	// pages filled in since are listed.
	uint32_t* vpages;
	uint32_t vpageFilled[VPAGE_FILLED];
	uint32_t vpageFillCount;
	// Compiled RSP blocks (rsp_jit_x64.c): every version compiled for each IMEM address, and the
	// one for IMEM as it is; rspCodeGen changes with every write to IMEM.
	bool useRspJit;
	uint8_t* rspArena;
	size_t rspUsed;
	void (*rspEnter)(struct N64*, void* code, uint64_t limit);
	void* rspLeave;
	uint32_t rspCodeGen;
	struct RspJitEntry rspTable[1024];
	struct RspBlock* rspVariants[1024];
	// RDP threads (N64_RDP_THREADS), and whether they have commands not yet drawn.
	int rdpThreads;
	// N64_RDP_GPU: the RDP's pixels drawn by a GLES 3.1 compute shader (rdp.c), falling back to the
	// renderer threads where that can't start.
	bool rdpGpu;
	bool rdpPending;
	// CPU cycles per video field: 60 fields a second for NTSC and MPAL, 50 for PAL.
	uint64_t fieldCycles;
	uint64_t fields;
	// Controller 1: buttons in the top half, stick (or with a mouse, the X and Y movement) below.
	uint32_t buttons;
	// The port (1-4) with an N64 Mouse (N64_MOUSE=port), 0 for none, and its buttons and movement in
	// the same layout.
	int mousePort;
	uint32_t mouse;
	const char* unimplemented;
	// Activity counts for diagnostics.
	struct {
		uint64_t piDmas, piBytes, rspTasks, dpSyncs, aiBuffers, siDmas;
	} stats;
	struct DD dd;
};

// n64.c
struct N64* n64Create(void);
void n64Destroy(struct N64*);
bool n64LoadRom(struct N64*, const uint8_t* data, size_t size);
void n64RunField(struct N64*);
void n64Interrupt(struct N64*, uint32_t line, bool set);
// The 64DD (dd.c): with an IPL ROM, and a disk image if given; with no cartridge it boots from the IPL.
bool n64LoadDisk(struct N64*, const uint8_t* ipl, size_t iplSize, const uint8_t* disk, size_t diskSize);
bool ddLoad(struct N64*, const uint8_t* ipl, size_t iplSize, const uint8_t* disk, size_t diskSize);
void ddFree(struct N64*);
bool ddPresent(const struct N64*);
bool ddInterrupt(const struct N64*);
uint32_t ddRead32(struct N64*, uint32_t addr);
void ddWrite32(struct N64*, uint32_t addr, uint32_t v);
uint8_t ddReadByte(struct N64*, uint32_t addr);
void ddWriteByte(struct N64*, uint32_t addr, uint8_t v);
void ddMechaResponse(struct N64*);
void ddBmRequest(struct N64*);
void aiUpdate(struct N64*);
int n64Picture(struct N64*, uint32_t* out, int* width, int* height);

// cpu.c
void cpuReset(struct N64*);
// The TLB or ASID changed: forgets the pages compiled loads and stores looked up.
void cpuTlbChanged(struct N64*);
void cpuStep(struct N64*);
void cpuStepOp(struct N64*, uint32_t op, uint32_t paddr);
bool cpuJitReady(struct N64*);
bool cpuJitStep(struct N64*, uint32_t op);
void cpuException(struct N64*, int code, int coprocessor);
void cpuCheckInterrupts(struct N64*);
uint32_t cpuCount(struct N64*);
bool cpuPeekAddress(struct N64*, uint64_t vaddr, uint32_t* paddr);

// rsp.c
void rspInitTables(void);
void rspStep(struct N64*);
void rspTimingStep(struct N64*, uint32_t op, uint32_t pc);
bool rspPairable(uint32_t first, uint32_t firstPc, uint32_t op);
bool rspIsBranch(uint32_t op);
int rspTimingRegs(uint32_t op, int* scalar, int* vector, int* nv, int* written, int* latency, int* nw);
void rspCop2(struct N64*, uint32_t op);
// rsp_vu.c
void rspVectorOp(struct RSP*, uint32_t op);
// The function for a vector instruction, given vd, vs and vt and the element's byte indices
// (rspElementMask); NULL for those rspVectorOp does itself (VRNDP, VRNDN, the divides and VMOV).
typedef void RspVectorFn(struct RSP*, int16_t* vd, const int16_t* vs, const int16_t* vt, const uint8_t* element);
RspVectorFn* rspVectorFn(uint32_t op);
const uint8_t* rspElementMask(uint32_t op);
void rspVectorLoad(struct N64*, uint32_t op);
void rspVectorStore(struct N64*, uint32_t op);
void rspCop0(struct N64*, uint32_t op);
void rspBreak(struct N64*);
// rsp_jit.c, and a code generator for the host (rsp_jit_x64.c, rsp_jit_a64.c)
void* rspJitLookup(struct N64*, uint32_t pc);
void rspJitFree(struct N64*);
void* rspCompile(struct N64*, const uint8_t* ops, uint32_t count, uint32_t start);
void rspArenaFree(struct N64*);
// blocks.c: the CPU runs from blocks of instructions fetched once, until the next event.
void blocksRun(struct N64*);
void blocksFree(struct N64*);
void blocksInvalidatePage(struct N64*, uint32_t page);
bool blockIsIdleLoop(const uint32_t* ops, uint32_t count, uint32_t vaddr);
// The cycles one pass of a poll loop takes with its cache lines in, or 0 if the block isn't one: a
// loop back to its start through loads and ALU instructions whose every pass reads the same.
uint32_t blockPollPass(const uint32_t* ops, uint32_t count, uint32_t vaddr);
// jit_x64.c
void* jitCompile(struct N64*, const uint32_t* ops, uint32_t count, uint32_t vaddr, uint32_t paddr);
void jitReset(struct N64*);
void jitFree(struct N64*);
// rdp.c: waits for RDP commands still being drawn by other threads.
void rdpWait(struct N64*);

// Whether any word of RDRAM [addr, addr + len) inside page p is part of a block.
static inline bool n64HoldsCode(struct N64* n, uint32_t p, uint32_t addr, uint32_t len) {
	uint32_t first = addr & (RDRAM_SIZE - 1), last = (addr + len - 1) & (RDRAM_SIZE - 1);
	uint32_t from = p << 10, to = from + 1023;
	uint32_t w0 = first >> 2 > from && first >> 2 <= to ? first >> 2 : from;
	uint32_t w1 = last >> 2 >= from && last >> 2 < to ? last >> 2 : to;
	for (uint32_t w = w0; w <= w1; ++w) {
		if (n->codeWords[w >> 3] & (1 << (w & 7))) {
			return true;
		}
	}
	return false;
}

// RDRAM [addr, addr + len) is about to be written: throws away blocks there, and waits for RDP
// commands that draw there or read it.
static inline void n64RdramWrite(struct N64* n, uint32_t addr, uint32_t len) {
	uint32_t first = (addr & (RDRAM_SIZE - 1)) >> 12, last = ((addr + len - 1) & (RDRAM_SIZE - 1)) >> 12;
	for (uint32_t p = first;; p = (p + 1) & ((RDRAM_SIZE >> 12) - 1)) {
		if (n->pageFlags[p] & PAGE_RDP) {
			rdpWait(n);
		}
		if ((n->pageFlags[p] & PAGE_CODE) && n64HoldsCode(n, p, addr, len)) {
			blocksInvalidatePage(n, p);
		}
		if (p == last) {
			break;
		}
	}
}

// RDRAM [addr, addr + len) is about to be read: waits for RDP commands that draw there.
static inline void n64RdramRead(struct N64* n, uint32_t addr, uint32_t len) {
	if (!n->rdpPending) {
		return;
	}
	uint32_t first = (addr & (RDRAM_SIZE - 1)) >> 12, last = ((addr + len - 1) & (RDRAM_SIZE - 1)) >> 12;
	for (uint32_t p = first;; p = (p + 1) & ((RDRAM_SIZE >> 12) - 1)) {
		if (n->pageFlags[p] & PAGE_RDP_WRITES) {
			rdpWait(n);
			return;
		}
		if (p == last) {
			break;
		}
	}
}

// The RSP runs to the CPU's time: before the CPU touches its registers, memory or MI, and once a slice.
void rspCatchUp(struct N64*);
static inline void n64Event(struct N64* n, uint64_t at) {
	if (at < n->eventAt) {
		n->eventAt = at;
	}
}

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
bool cartBootChecksum(const uint8_t* rom, size_t size, int cic, uint8_t seedByte, uint32_t out[2]);

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
