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
	// Every write to a COP0 register is latched; the unused registers read the latch. It's kept in
	// unused register 7's place.
	CP0_LATCH = 7,
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
	// COP2's moves read and write one 64-bit latch, whatever register they name.
	uint64_t cop2Latch;
	// A compiled poll loop tried to skip its passes since it was entered (a64).
	bool pollTried;
	uint64_t cop0[32];
	struct TLBEntry tlb[32];
	uint64_t countBase;
	// FPU registers as raw bits; how the 32 registers pair up depends on Status.FR.
	uint64_t fpr[32];
	uint32_t fcr31;
	bool exception;
	// The caches (cpu.c): tags, the line's physical address while it is valid, else DCACHE_NONE; the
	// tags as CACHE reads them back; dirty D-cache lines; the lines' data, in RDRAM's byte order.
	uint32_t icacheTag[512], dcacheTag[512];
	uint32_t icachePtag[512], dcachePtag[512];
	bool dcacheDirty[512];
	// The last address translated goes through the caches.
	bool cachedAccess;
	uint32_t stall;
	// Count when the timer was last checked, so a Compare value passed during a stall still fires.
	uint32_t lastCount;
	// The row each RDRAM bank has open, and when the CPU's posted uncached writes are done.
	uint32_t openRow[8];
	uint64_t writesDone;
	// Last, past the offsets compiled code reaches with an immediate.
	uint8_t icacheData[512][32], dcacheData[512][16];
};

struct SP {
	uint8_t mem[0x2000];
	uint32_t memAddr, dramAddr, status, semaphore, pc;
	// The length registers as they read: the last DMA's length (0xFF8 once done), count and skip.
	uint32_t length;
	// RCP cycles when the DMAs under way are done, and when a queued one starts (DMA full until then).
	uint64_t dmaUntil, dmaFullUntil;
	// A task hleGraphics ran: the RSP shows it running (doing nothing) until then.
	uint64_t hleDoneAt;
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
	// What pairing needs of lastOp (rspTimingDesc).
	uint16_t lastDesc;
	uint64_t lastStart;
	bool lastAlone;
	bool inDelay;
};

// DP_STATUS's pipe busy and start gclk: set when the RDP is given commands, until a full sync (ares);
// start valid: START written, not yet taken by a write to END.
enum { DP_RUNNING = 0x28, DP_START_VALID = 0x400 };

struct DP {
	uint32_t start, end, current, status;
	// A command the buffer ended inside: its words so far, the rest from the next buffer's start.
	uint64_t partial[22];
	int partialWords;
	// When the RDP finishes the commands it has, and when its pending full-sync interrupt is due.
	uint64_t busyUntil, syncAt;
	// With HLE, full syncs after the pending one (a whole task's commands arrive at once), each
	// interrupting when the RDP gets there.
	uint64_t syncQueue[8];
	int syncQueued;
	// The 24-bit counters: RCP cycles since the clock counter was cleared (at clockBase, in RCP
	// cycles), and RDP clocks with the command buffer busy, the pipeline busy and TMEM loading.
	uint64_t clockBase;
	uint32_t bufBusy, pipeBusy, tmemBusy;
};

struct VI {
	uint32_t reg[14];
	uint32_t line;
};

// N64_FRAMESKIP (a speed setting, not exact): every other frame drawn into a framebuffer the VI
// shows is left undrawn (its triangles and rectangles; modes, loads and other images still run),
// and the picture before it shown again.
struct FrameSkip {
	bool on;
	// The VI's last few framebuffers, the one being drawn, frames counted, and each buffer's last
	// frame skipped or not.
	uint32_t shown[4];
	uint32_t drawing;
	bool intoShown;
	uint64_t frames;
	bool skipping;
	uint32_t buffer[4];
	bool skipped[4];
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
	// PI_WR_LEN as it reads after a transfer.
	uint32_t writeLength;
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
	// A CPU write to PIF RAM keeps the SI's I/O busy until then.
	uint64_t ioUntil;
};

struct RDP;

// The N64_ACCURACY levels.
enum { ACCURACY_LLE_ACCURATE, ACCURACY_LLE_COMPATIBLE, ACCURACY_HLE };

// CPU cycles for cache line fills and the write-back of a dirty data line (cpu.c and both code
// generators).
enum { ICACHE_FILL = 44, DCACHE_FILL = 33, DCACHE_WRITEBACK = 40 };

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
	// RDRAM's hidden ninth bits, as the RDP keeps them: two coverage bits for each halfword.
	uint8_t* hidden;
	uint8_t* rom;
	size_t romSize;
	uint8_t pifRam[64];
	// PIF RAM holds Joybus commands (not a CIC challenge), which run again when it is read.
	bool pifJoybus, pifChallenged;
	uint8_t save[0x20000];
	uint8_t eeprom[2048];
	bool eeprom16k;
	int cic;

	struct CPU cpu;
	struct SP sp;
	struct RSP rsp;
	struct DP dp;
	struct RDP* rdp;
	struct Hle* hle;
	// N64HLECHECK=1: each graphics task also runs at high level without drawing, and the RDP commands
	// it would send are counted against those the microcode sends ([0] microcode, [1] high level).
	bool hleCheck;
	uint32_t checkCounts[2][64];
	// The first commands' first words, for finding where the two part.
	uint64_t checkSeq[2][4096];
	uint32_t checkLen[2];
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
	// N64ISVIEWER (test ROMs): the IS-Viewer's 64 KiB at 0x13FF0000, its text printed as it's written.
	uint8_t* isViewer;

	uint64_t cycles;
	// The earliest pending event (AI, PI, SI, DP sync): the CPU runs no further before it is handled.
	uint64_t eventAt;
	// CPU cycles the CPU may run ahead of the RSP before it catches up.
	uint64_t slice;
	// N64_ACCURACY: lle-accurate (the default) is the hardware. lle-compatible is for hacks made on
	// emulators: RDRAM starts zeroed, misaligned CPU accesses align down instead of raising address
	// errors, unmapped null-pointer accesses read zero, PI DMA copies between the exact addresses
	// and takes a quarter cycle a byte, SP DMA keeps the RDRAM address's low bits, and a blanked VI
	// window still shows the picture. hle is the
	// hardware with high-level graphics (hle.c) for the microcodes it knows. compat is true for
	// lle-compatible only.
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
	// N64_VI_GPU: the VI's filters run by a GLES 3.1 compute shader (n64.c), falling back to the CPU
	// where that can't start.
	bool viGpu;
	struct Gles* viGles;
	// With the GPU RDP, the VI runs in its context on its copy of RDRAM (the program's index + 1, -1
	// when it doesn't build); where the last GPU picture is.
	int viSharedProgram;
	struct Gles* pictureGles;
	int pictureBinding;
	// The picture is RGBA bytes for a texture (in the frontend's context), else 0x00RRGGBB.
	bool pictureRgba;
	// N64_RDP_GL (with hle): the GL renderer draws what the HLE makes (glr.c).
	bool rdpGl;
	struct Glr* glr;
	// Many RDP triangles in a field (a microcode left to the RSP): the GL renderer gives way to the
	// RDP threads asked for, between slices.
	int rdpThreadsAsked;
	// With the GL renderer, graphics tasks lately and those the RSP ran (microcodes the HLE doesn't
	// know): mostly those, and the RDP threads take over (the GPU would only add syncs).
	uint32_t glTasks, glRspTasks;
	bool glrStop;
	struct FrameSkip skip;
	// The last picture made, shown again for a skipped frame.
	uint32_t* lastPicture;
	// The VI's last picture, with what it was made from (n64.c): made again only when that changes.
	struct ViKept* viKept;
	// Threads that make bands of the VI's picture on the CPU (n64.c), as many as the renderers.
	struct ViPool* viPool;
	int lastWidth, lastHeight;
	// CPU cycles per video field: 60 fields a second for NTSC and MPAL, 50 for PAL.
	uint64_t fieldCycles;
	uint64_t fields;
	// Where the settings (N64_ACCURACY and the others) come from when the machine starts: the
	// environment unless a frontend gives its own lookup.
	const char* (*settings)(const char* name);
	// The samples the AI was given (left and right, 16-bit), for a frontend to play at audioRate a
	// second; none are kept until it sets audioCapacity and audio.
	int16_t* audio;
	size_t audioCount, audioCapacity;
	uint32_t audioRate;
	// Controller 1: buttons in the top half, stick (or with a mouse, the X and Y movement) below.
	uint32_t buttons;
	// The port (1-4) with an N64 Mouse (N64_MOUSE=port), 0 for none, and its buttons and movement in
	// the same layout.
	int mousePort;
	uint32_t mouse;
	const char* unimplemented;
	// Activity counts for diagnostics.
	struct {
		uint64_t piDmas, piBytes, rspTasks, dpSyncs, aiBuffers, siDmas, glSyncs, glRows, idleSlices, idleCycles, glTextures, glTargetRows;
	} stats;
	struct DD dd;
	// hle: a CPU loop that only waits is skipped to the end of each slice (blocks.c idleRun). Its
	// pass is traced in the interpreter: the loads it makes (to check they still read the same),
	// whether it touched a register or took an exception (fatal), or stored a new value.
	struct Idle {
		bool enabled, tracing, fatal, changed, on;
		uint32_t pc, recent[6], backoff;
		int near;
		// The traced pass's loads, then those of the pass being checked against them.
		int loads, passLoads;
		struct IdleLoad {
			uint32_t addr;
			int size;
			uint64_t value;
		} load[64];
	} idle;
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
// With N64_VI_GPU, starts the picture on the GPU and returns without waiting for it (a frontend
// shows it from there); false when the VI shows nothing or the GPU isn't used.
bool n64PictureStart(struct N64*, int* width, int* height);
// After n64PictureStart, the GL buffer holding the picture in RGBA bytes (with the GPU RDP, in its
// context); 0 when it is elsewhere.
unsigned n64PictureBuffer(struct N64*);

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
// Kernel mode with 32-bit addressing, the only mode compiled code runs in.
bool cpuPlainMode(const struct N64*);
uint32_t cpuDcacheMiss(struct N64*, uint32_t vaddr, uint32_t paddr);
uint32_t cpuIcacheMiss(struct N64*, uint32_t vaddr, uint32_t paddr);
uint32_t cpuCacheOp(struct N64*, uint32_t op, uint32_t vaddr);
bool cpuCodeCoherent(struct N64*, uint32_t vaddr, uint32_t paddr, uint32_t count);

// rsp.c
void rspInitTables(void);
void rspStep(struct N64*);
void rspTimingStep(struct N64*, uint32_t op, uint32_t pc);
uint16_t rspTimingDesc(uint32_t op);
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
void perfMapAdd(const void* code, size_t size, const char* kind, uint32_t address);
void blocksInvalidatePage(struct N64*, uint32_t page);
bool blockIsIdleLoop(const uint32_t* ops, uint32_t count, uint32_t vaddr);
void idleLoad(struct N64*, uint32_t paddr, int size, uint64_t value);
void idleStore(struct N64*, uint32_t paddr, int size, uint64_t value);
// The cycles one pass of a poll loop takes with its cache lines in, or 0 if the block isn't one: a
// loop back to its start through loads and ALU instructions whose every pass reads the same. For a
// branch on what its delay slot loads, reload is that register, plus 32 when the branch is taken
// on zero (else on non-zero); the passes are the same while it is taken on the register.
uint32_t blockPollPass(const uint32_t* ops, uint32_t count, uint32_t vaddr, int* reload);
// jit_x64.c
void* jitCompile(struct N64*, const uint32_t* ops, uint32_t count, uint32_t vaddr, uint32_t paddr);
void jitReset(struct N64*);
void jitFree(struct N64*);
// rdp.c: waits for RDP commands still being drawn by other threads.
void rdpWait(struct N64*);
void rdpWaitFor(struct N64*, uint32_t addr, uint32_t len);
void rdpWaitRead(struct N64*, uint32_t addr, uint32_t len);
struct Gles* rdpGpuShare(struct N64*, uint32_t addr, uint32_t len);
void rdpWaitDrawn(struct N64*, uint32_t addr, uint32_t len);
bool frameSkipped(struct N64*, uint32_t origin);

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
			rdpWaitFor(n, addr, len);
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
			rdpWaitRead(n, addr, len);
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
void fpuClearCauses(struct N64*);
bool fpuProbe(struct N64*, uint32_t op);
uint32_t fpuCycles(int fmt, int func);

// bus.c
uint8_t busRead8(struct N64*, uint32_t addr);
uint16_t busRead16(struct N64*, uint32_t addr);
uint32_t busRead32(struct N64*, uint32_t addr);
uint64_t busRead64(struct N64*, uint32_t addr);
// A CPU store of a byte, halfword or doubleword: RDRAM takes it as it is; the RCP's memories and
// registers take a word, the whole register shifted to the bytes' place (a doubleword: its upper
// word), as ares has it from hardware tests.
void busWrite8(struct N64*, uint32_t addr, uint64_t value);
void busWrite16(struct N64*, uint32_t addr, uint64_t value);
void busWrite32(struct N64*, uint32_t addr, uint32_t value);
void busWrite64(struct N64*, uint32_t addr, uint64_t value);
// MI repeat mode, set: the CPU's uncached write to RDRAM, spread as the MI does it; false when the
// mode is off.
bool busRepeatWrite(struct N64*, uint32_t addr, int size, uint64_t value);
// A cache line's burst to or from addr.
void busReadLine(struct N64*, uint32_t addr, uint8_t* out, int len);
void busWriteLine(struct N64*, uint32_t addr, const uint8_t* in, int len);

// rdp.c
void rdpRun(struct N64*);
void rdpCommands(struct N64*, const uint64_t* list, size_t count);

// hle.c
bool hleGraphics(struct N64*, bool dry);
void glrEnd(struct N64*);
// The GL state the renderer last set is unknown (the frontend drew in its context).
void glrForget(struct N64*);
// A task waiting where its display list branches to itself (the CPU still writing it); a look
// whether it can go on.
bool hleWaiting(struct N64*);
void hleResume(struct N64*);
void rdpDestroy(struct N64*);

// carts.c
bool cartEeprom16k(const uint8_t* rom);
bool cartBootChecksum(const uint8_t* rom, size_t size, int cic, uint8_t seedByte, uint32_t out[2]);

// pif.c
void pifBoot(struct N64*);
void pifRun(struct N64*);
void pifRead(struct N64*);
uint32_t pifDmaCycles(struct N64*, bool write);

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
