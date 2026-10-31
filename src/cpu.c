#include "n64.h"

#include <string.h>

#define RS ((op >> 21) & 31)
#define RT ((op >> 16) & 31)
#define RD ((op >> 11) & 31)
#define SA ((op >> 6) & 31)
#define IMM ((uint64_t) (int64_t) (int16_t) op)
#define UIMM ((uint64_t) (uint16_t) op)

enum { ST_IE = 1, ST_EXL = 2, ST_ERL = 4, ST_FR = 1 << 26, ST_CU1 = 1 << 29, ST_CU2 = 1 << 30, ST_BEV = 1 << 22 };

// Timing, in CPU cycles on top of one a instruction: cache line fills and write-backs, and reads of
// RCP registers (cache figures as in the VR4300 manual and ares; the RCP read cost is fitted to
// pdrome's instruction timing ROM, which counts loop iterations per field on hardware).
enum { RCP_READ = 22, PI_READ = 250 };

// What an uncached read costs: an RCP register, or a word over the PI bus (cartridge, 64DD; ares).
static uint32_t readCost(uint32_t paddr) {
	return paddr < RDRAM_SIZE ? 0 : paddr >= 0x05000000 && paddr < 0x1FC00000 ? PI_READ : RCP_READ;
}

static bool address(struct N64* n, uint64_t vaddr, int align, bool store, uint32_t* paddr);

// The caches (VR4300 manual, as ares has them): 512 D-cache lines of 16 bytes and 512 I-cache lines
// of 32, picked by the virtual address and tagged with the physical one (bits 12 and up; the line
// supplies the rest). Outside lle-compatible the lines hold their own data: stores stay in the
// D-cache until a line is written back, and instructions come from the I-cache. lle-compatible keeps
// memory up to date, the tags only costing the misses.
static uint32_t dcacheAddress(const struct CPU* c, uint32_t i) {
	return c->dcachePtag[i] | (i << 4 & 0xFF0);
}

static uint32_t icacheAddress(const struct CPU* c, uint32_t i) {
	return c->icachePtag[i] | (i << 5 & 0xFE0);
}

static void dcacheSet(struct CPU* c, uint32_t i, uint32_t ptag, bool valid) {
	c->dcachePtag[i] = ptag & ~0xFFFu;
	c->dcacheTag[i] = valid ? dcacheAddress(c, i) : DCACHE_NONE;
}

static void icacheSet(struct CPU* c, uint32_t i, uint32_t ptag, bool valid) {
	c->icachePtag[i] = ptag & ~0xFFFu;
	c->icacheTag[i] = valid ? icacheAddress(c, i) : DCACHE_NONE;
}

static void dcacheWriteBack(struct N64* n, uint32_t i) {
	n->cpu.stall += DCACHE_WRITEBACK;
	if (!n->compat) {
		busWriteLine(n, dcacheAddress(&n->cpu, i), n->cpu.dcacheData[i], 16);
	}
}

static void dcacheFill(struct N64* n, uint32_t i, uint32_t paddr) {
	struct CPU* c = &n->cpu;
	c->stall += DCACHE_FILL;
	dcacheSet(c, i, paddr, true);
	c->dcacheDirty[i] = false;
	if (!n->compat) {
		busReadLine(n, dcacheAddress(c, i), c->dcacheData[i], 16);
	}
}

static void icacheFill(struct N64* n, uint32_t i, uint32_t paddr) {
	struct CPU* c = &n->cpu;
	c->stall += ICACHE_FILL;
	icacheSet(c, i, paddr, true);
	if (!n->compat) {
		busReadLine(n, icacheAddress(c, i), c->icacheData[i], 32);
	}
}

// The D-cache line a cached access uses, written back if dirty and filled when it misses.
static uint32_t dcacheLine(struct N64* n, uint64_t vaddr, uint32_t paddr) {
	struct CPU* c = &n->cpu;
	uint32_t i = ((uint32_t) vaddr >> 4) & 511;
	if (c->dcacheTag[i] != (paddr & ~0xFu)) {
		if (c->dcacheTag[i] != DCACHE_NONE && c->dcacheDirty[i]) {
			dcacheWriteBack(n, i);
		}
		dcacheFill(n, i, paddr);
	}
	return i;
}

// For compiled code: a D-cache miss as dcacheLine has it; the cycles it costs. UINT32_MAX, with
// nothing done, when writing the old line back could throw compiled code away (or reach the RCP): the
// interpreter then runs the access.
uint32_t cpuDcacheMiss(struct N64* n, uint32_t vaddr, uint32_t paddr) {
	struct CPU* c = &n->cpu;
	uint32_t const i = (vaddr >> 4) & 511, back = dcacheAddress(c, i);
	if (!n->compat && c->dcacheTag[i] != DCACHE_NONE && c->dcacheDirty[i] && (back >= RDRAM_SIZE || (n->pageFlags[back >> 12] & PAGE_CODE))) {
		return UINT32_MAX;
	}
	dcacheLine(n, vaddr, paddr);
	uint32_t stall = n->cpu.stall;
	n->cpu.stall = 0;
	return stall;
}

static void cacheOp(struct N64* n, uint32_t op, uint64_t vaddr);

// For compiled code: a D-cache or secondary cache CACHE at an aligned kseg0 address (which can't
// fault); the cycles it costs, or UINT32_MAX as cpuDcacheMiss.
uint32_t cpuCacheOp(struct N64* n, uint32_t op, uint32_t vaddr) {
	struct CPU* c = &n->cpu;
	uint32_t const i = (vaddr >> 4) & 511, back = dcacheAddress(c, i);
	if (!n->compat && c->dcacheTag[i] != DCACHE_NONE && c->dcacheDirty[i] && (back >= RDRAM_SIZE || (n->pageFlags[back >> 12] & PAGE_CODE))) {
		return UINT32_MAX;
	}
	cacheOp(n, op, sext32(vaddr));
	uint32_t stall = c->stall;
	c->stall = 0;
	return stall;
}

// For compiled code: an I-cache miss; the cycles it costs.
uint32_t cpuIcacheMiss(struct N64* n, uint32_t vaddr, uint32_t paddr) {
	icacheFill(n, (vaddr >> 5) & 511, paddr);
	uint32_t stall = n->cpu.stall;
	n->cpu.stall = 0;
	return stall;
}

// Whether a block compiled from RDRAM runs what the I-cache would: no line it touches holds other
// words than memory has.
bool cpuCodeCoherent(struct N64* n, uint32_t vaddr, uint32_t paddr, uint32_t count) {
	struct CPU* c = &n->cpu;
	if (n->compat) {
		return true;
	}
	for (uint32_t p = paddr & ~31u; p < paddr + 4 * count; p += 32) {
		uint32_t i = ((vaddr + (p - paddr)) >> 5) & 511;
		if (c->icacheTag[i] == p && memcmp(c->icacheData[i], &n->rdram[p], 32)) {
			return false;
		}
	}
	return true;
}

// An instruction word at a translated address.
static uint32_t fetch(struct N64* n, uint64_t vaddr, uint32_t paddr) {
	struct CPU* c = &n->cpu;
	if (!c->cachedAccess) {
		c->stall += readCost(paddr);
		return busRead32(n, paddr);
	}
	uint32_t i = ((uint32_t) vaddr >> 5) & 511;
	if (c->icacheTag[i] != (paddr & ~31u)) {
		icacheFill(n, i, paddr);
	}
	if (n->compat) {
		return busRead32(n, paddr);
	}
	const uint8_t* w = &c->icacheData[i][paddr & 28];
	return (uint32_t) w[0] << 24 | (uint32_t) w[1] << 16 | (uint32_t) w[2] << 8 | w[3];
}

// A line's tag for CACHE Index Store Tag: compiled code holding the words it now claims runs again
// from what the line has.
static void icacheStoreTag(struct N64* n, uint32_t i) {
	struct CPU* c = &n->cpu;
	uint32_t const tl = (uint32_t) c->cop0[CP0_TAGLO];
	icacheSet(c, i, (tl >> 8 & 0xFFFFF) << 12, tl & 0x80);
	uint32_t const p = icacheAddress(c, i), page = p >> 12;
	if (c->icacheTag[i] != DCACHE_NONE && p < RDRAM_SIZE && (n->pageFlags[page] & PAGE_CODE) && n64HoldsCode(n, page, p, 32)) {
		blocksInvalidatePage(n, page);
	}
}

// CACHE: the address translated as a word load's (address errors, TLB exceptions); the line the
// virtual address picks; the hit operations only when it holds the physical one. The secondary
// cache isn't there.
static void cacheOp(struct N64* n, uint32_t op, uint64_t vaddr) {
	struct CPU* c = &n->cpu;
	uint32_t p;
	if (!address(n, vaddr, 4, false, &p)) {
		return;
	}
	uint32_t const di = ((uint32_t) vaddr >> 4) & 511, ii = ((uint32_t) vaddr >> 5) & 511;
	bool const dvalid = c->dcacheTag[di] != DCACHE_NONE, dhit = c->dcacheTag[di] == (p & ~0xFu);
	bool const ihit = c->icacheTag[ii] == (p & ~31u);
	switch ((op >> 16) & 31) {
	case 0x00:
		icacheSet(c, ii, p, false);
		break;
	case 0x04:
		c->cop0[CP0_TAGLO] = (c->icachePtag[ii] >> 12) << 8 | (c->icacheTag[ii] != DCACHE_NONE ? 0x80 : 0);
		break;
	case 0x08:
		icacheStoreTag(n, ii);
		break;
	case 0x10:
		if (ihit) {
			icacheSet(c, ii, c->icachePtag[ii], false);
		}
		break;
	case 0x14:
		icacheFill(n, ii, p);
		break;
	case 0x18:
		if (ihit) {
			c->stall += ICACHE_FILL;
			if (!n->compat) {
				busWriteLine(n, icacheAddress(c, ii), c->icacheData[ii], 32);
			}
		}
		break;
	case 0x01:
		if (dvalid) {
			if (c->dcacheDirty[di]) {
				dcacheWriteBack(n, di);
			}
			dcacheSet(c, di, p, false);
		}
		break;
	case 0x05:
		c->cop0[CP0_TAGLO] = (c->dcachePtag[di] >> 12) << 8 | (dvalid ? 0xC0 : 0);
		break;
	case 0x09: {
		uint32_t const tl = (uint32_t) c->cop0[CP0_TAGLO];
		dcacheSet(c, di, (tl >> 8 & 0xFFFFF) << 12, tl & 0x80);
		break;
	}
	case 0x0D:
		if (!dhit) {
			if (dvalid && c->dcacheDirty[di]) {
				dcacheWriteBack(n, di);
			}
			c->dcacheDirty[di] = false;
		}
		dcacheSet(c, di, p, true);
		break;
	case 0x11:
		if (dhit) {
			dcacheSet(c, di, c->dcachePtag[di], false);
		}
		break;
	case 0x15:
		if (dhit) {
			if (c->dcacheDirty[di]) {
				dcacheWriteBack(n, di);
			}
			dcacheSet(c, di, c->dcachePtag[di], false);
		}
		break;
	case 0x19:
		if (dhit && c->dcacheDirty[di]) {
			dcacheWriteBack(n, di);
			c->dcacheDirty[di] = false;
		}
		break;
	}
}

void cpuReset(struct N64* n) {
	struct CPU* c = &n->cpu;
	memset(c, 0, sizeof(*c));
	memset(c->dcacheTag, 0xFF, sizeof(c->dcacheTag));
	memset(c->icacheTag, 0xFF, sizeof(c->icacheTag));
	cpuTlbChanged(n);
	c->cop0[CP0_STATUS] = 0x34000000;
	c->cop0[CP0_CONFIG] = 0x7006E463;
	c->cop0[CP0_PRID] = 0x00000B22;
	c->cop0[CP0_RANDOM] = 31;
	c->cop0[CP0_EPC] = c->cop0[CP0_ERROREPC] = ~0ull;
	c->pc = 0xFFFFFFFFBFC00000ull;
	c->nextPc = c->pc + 4;
}

// Count runs at half the CPU clock, from the cycle countBase (a write starts it on an edge, ares).
uint32_t cpuCount(struct N64* n) {
	return (uint32_t) ((n->cycles - n->cpu.countBase) >> 1);
}

static bool cpuWide(const struct CPU* c);

void cpuException(struct N64* n, int code, int coprocessor) {
	n->idle.fatal = true;
	struct CPU* c = &n->cpu;
	uint64_t* cop0 = c->cop0;
	bool refill = (code == EXC_TLBL || code == EXC_TLBS) && c->exception;
	// The TLB refill vector, 0x080 with 64-bit addressing.
	uint32_t const refillAt = cpuWide(c) ? 0x80 : 0;
	c->exception = false;
	if (!(cop0[CP0_STATUS] & ST_EXL)) {
		cop0[CP0_EPC] = c->inDelay ? c->curPc - 4 : c->curPc;
		cop0[CP0_CAUSE] = (cop0[CP0_CAUSE] & ~0x80000000ull) | (c->inDelay ? 0x80000000ull : 0);
	} else {
		refill = false;
	}
	cop0[CP0_CAUSE] = (cop0[CP0_CAUSE] & ~0x3000007Cull) | (uint64_t) code << 2 | (uint64_t) coprocessor << 28;
	cop0[CP0_STATUS] |= ST_EXL;
	uint64_t base = cop0[CP0_STATUS] & ST_BEV ? 0xFFFFFFFFBFC00200ull : 0xFFFFFFFF80000000ull;
	c->pc = base + (refill ? refillAt : 0x180);
	c->nextPc = c->pc + 4;
	c->nextInDelay = false;
	c->inDelay = false;
}

void cpuCheckInterrupts(struct N64* n) {
	uint64_t* cop0 = n->cpu.cop0;
	if (n->miIntr & n->miMask) {
		cop0[CP0_CAUSE] |= 0x400;
	} else {
		cop0[CP0_CAUSE] &= ~0x400ull;
	}
	// The cartridge line (IP3): the 64DD's.
	if (ddInterrupt(n)) {
		cop0[CP0_CAUSE] |= 0x800;
	} else {
		cop0[CP0_CAUSE] &= ~0x800ull;
	}
}

static bool interruptPending(struct N64* n) {
	uint64_t* cop0 = n->cpu.cop0;
	uint64_t st = cop0[CP0_STATUS];
	return (st & ST_IE) && !(st & (ST_EXL | ST_ERL)) && (cop0[CP0_CAUSE] & st & 0xFF00);
}

// Virtual to physical. Returns false after raising the exception.
// A virtual page's entry in vpages when it isn't known: the page past RDRAM.
static uint32_t vpageUnknown(uint32_t page) {
	return ((RDRAM_SIZE >> 12) - page) << 12;
}

void cpuTlbChanged(struct N64* n) {
	if (n->vpageFillCount > VPAGE_FILLED) {
		for (uint32_t page = 0; page < VPAGES; ++page) {
			n->vpages[page] = n->vpages[VPAGES + page] = vpageUnknown(page);
		}
		// kseg0's RDRAM, which the TLB doesn't change.
		for (uint32_t page = 0x80000; page < 0x80000 + (RDRAM_SIZE >> 12); ++page) {
			n->vpages[page] = n->vpages[VPAGES + page] = 0x80000000u;
		}
	} else {
		for (uint32_t i = 0; i < n->vpageFillCount; ++i) {
			uint32_t page = n->vpageFilled[i];
			n->vpages[page] = n->vpages[VPAGES + page] = vpageUnknown(page);
		}
	}
	n->vpageFillCount = 0;
}

// The privilege the CPU runs at: kernel (KSU 0, or EXL or ERL set), supervisor (KSU 1) or user;
// and whether it addresses with 64 bits there (KX, SX or UX).
enum { MODE_KERNEL, MODE_SUPERVISOR, MODE_USER };

static int cpuMode(const struct CPU* c) {
	uint64_t st = c->cop0[CP0_STATUS];
	int ksu = (st >> 3) & 3;
	return st & (ST_EXL | ST_ERL) || ksu == 0 ? MODE_KERNEL : ksu == 1 ? MODE_SUPERVISOR : MODE_USER;
}

static bool cpuWide(const struct CPU* c) {
	return c->cop0[CP0_STATUS] >> (7 - cpuMode(c)) & 1;
}

// User mode with Status.RE runs little-endian: the bus sees each access's byte lanes swapped within
// the doubleword, the address xored by this (ares).
static uint32_t endianLanes(const struct CPU* c, int size) {
	if (!(c->cop0[CP0_STATUS] & (1u << 25)) || cpuMode(c) != MODE_USER) {
		return 0;
	}
	return size == 1 ? 7 : size == 2 ? 6 : size == 4 ? 4 : 0;
}

// Whether compiled code may run: it takes kernel mode's 32-bit addressing for granted.
bool cpuPlainMode(const struct N64* n) {
	return cpuMode(&n->cpu) == MODE_KERNEL && !cpuWide(&n->cpu);
}

// Where an address goes in the mode the CPU is in (the VR4300's segments): nowhere (an address
// error), through the TLB, or straight to a physical address, through the caches or not.
enum { SEG_UNUSED, SEG_MAPPED, SEG_CACHED, SEG_DIRECT };

static int segmentOf(const struct CPU* c, uint64_t vaddr, uint32_t* paddr) {
	int const mode = cpuMode(c);
	if (!cpuWide(c)) {
		uint32_t v = (uint32_t) vaddr;
		*paddr = v & 0x1FFFFFFF;
		if (v < 0x80000000u) {
			return SEG_MAPPED;
		}
		if (mode == MODE_KERNEL) {
			return v < 0xA0000000u ? SEG_CACHED : v < 0xC0000000u ? SEG_DIRECT : SEG_MAPPED;
		}
		return mode == MODE_SUPERVISOR && v >= 0xC0000000u && v < 0xE0000000u ? SEG_MAPPED : SEG_UNUSED;
	}
	*paddr = (uint32_t) vaddr;
	if (vaddr <= 0x000000FFFFFFFFFFull) {
		return SEG_MAPPED;
	}
	if (mode != MODE_USER && vaddr >= 0x4000000000000000ull && vaddr <= 0x400000FFFFFFFFFFull) {
		return SEG_MAPPED;
	}
	if (mode == MODE_SUPERVISOR) {
		return vaddr >= 0xFFFFFFFFC0000000ull && vaddr <= 0xFFFFFFFFDFFFFFFFull ? SEG_MAPPED : SEG_UNUSED;
	}
	if (mode == MODE_USER) {
		return SEG_UNUSED;
	}
	// xkphys: the cache algorithm in bits 61-59 (2 uncached), 32 bits of physical address.
	if (vaddr >= 0x8000000000000000ull && vaddr <= 0xBFFFFFFFFFFFFFFFull) {
		if (vaddr & 0x07FFFFFF00000000ull) {
			return SEG_UNUSED;
		}
		return ((vaddr >> 59) & 7) == 2 ? SEG_DIRECT : SEG_CACHED;
	}
	if (vaddr >= 0xC000000000000000ull && vaddr <= 0xC00000FF7FFFFFFFull) {
		return SEG_MAPPED;
	}
	if (vaddr >= 0xFFFFFFFF80000000ull) {
		*paddr = (uint32_t) vaddr & 0x1FFFFFFF;
		return vaddr < 0xFFFFFFFFA0000000ull ? SEG_CACHED : vaddr < 0xFFFFFFFFC0000000ull ? SEG_DIRECT : SEG_MAPPED;
	}
	return SEG_UNUSED;
}

// What an address error or TLB exception records of the address: BadVAddr, the page in EntryHi,
// and the page in Context and XContext.
static void faultAddress(struct CPU* c, uint64_t vaddr) {
	uint64_t* cop0 = c->cop0;
	cop0[CP0_BADVADDR] = vaddr;
	cop0[CP0_ENTRYHI] = (cop0[CP0_ENTRYHI] & 0xFF) | (vaddr & 0xC00000FFFFFFE000ull);
	cop0[CP0_CONTEXT] = (cop0[CP0_CONTEXT] & ~0x7FFFF0ull) | ((vaddr >> 9) & 0x7FFFF0);
	cop0[CP0_XCONTEXT] = (cop0[CP0_XCONTEXT] & ~0x1FFFFFFF0ull) | ((vaddr >> 9) & 0x7FFFFFF0) | (vaddr >> 62) << 31;
}

// The TLB entry mapping vaddr (its region, 40 bits of page and address space), or -1.
static int tlbFind(const struct CPU* c, uint64_t vaddr) {
	for (int i = 0; i < 32; ++i) {
		const struct TLBEntry* e = &c->tlb[i];
		uint64_t mask = ~(uint64_t) (e->pageMask | 0x1FFF) & 0xC00000FFFFFFFFFFull;
		if (((vaddr ^ e->entryHi) & mask) != 0) {
			continue;
		}
		bool global = e->entryLo[0] & e->entryLo[1] & 1;
		if (!global && ((e->entryHi ^ c->cop0[CP0_ENTRYHI]) & 0xFF)) {
			continue;
		}
		return i;
	}
	return -1;
}

static bool translate(struct N64* n, uint64_t vaddr, bool store, uint32_t* paddr) {
	struct CPU* c = &n->cpu;
	uint32_t v = (uint32_t) vaddr;
	int const seg = segmentOf(c, vaddr, paddr);
	if (seg == SEG_CACHED || seg == SEG_DIRECT) {
		c->cachedAccess = seg == SEG_CACHED;
		return true;
	}
	if (seg == SEG_UNUSED) {
		faultAddress(c, vaddr);
		cpuException(n, store ? EXC_ADES : EXC_ADEL, 0);
		return false;
	}
	int const i = tlbFind(c, vaddr);
	if (i < 0) {
		c->exception = true;
		faultAddress(c, vaddr);
		cpuException(n, store ? EXC_TLBS : EXC_TLBL, 0);
		return false;
	}
	struct TLBEntry* e = &c->tlb[i];
	uint32_t offsetMask = (e->pageMask | 0x1FFF) >> 1;
	uint64_t lo = e->entryLo[(v & (offsetMask + 1)) ? 1 : 0];
	if (!(lo & 2) || (store && !(lo & 4))) {
		faultAddress(c, vaddr);
		cpuException(n, !(lo & 2) ? (store ? EXC_TLBS : EXC_TLBL) : EXC_MOD, 0);
		return false;
	}
	*paddr = ((uint32_t) (lo >> 6) << 12 & ~offsetMask) | (v & offsetMask);
	c->cachedAccess = ((lo >> 3) & 7) != 2;
	// Compiled code finds the cached pages through vpages, by 32-bit kernel addresses only.
	if ((*paddr & ~0xFFFu) < RDRAM_SIZE && c->cachedAccess && cpuPlainMode(n) && (uint64_t) (int64_t) (int32_t) v == vaddr) {
		uint32_t delta = (*paddr & ~0xFFFu) - (v & ~0xFFFu);
		uint32_t page = v >> 12, writes = lo & 4 ? delta : vpageUnknown(page);
		if (n->vpages[page] != delta || n->vpages[VPAGES + page] != writes) {
			if (n->vpageFillCount < VPAGE_FILLED) {
				n->vpageFilled[n->vpageFillCount] = page;
			}
			++n->vpageFillCount;
			n->vpages[page] = delta;
			n->vpages[VPAGES + page] = writes;
		}
	}
	return true;
}

// For tools: the physical address of a virtual one, without raising exceptions or changing state.
bool cpuPeekAddress(struct N64* n, uint64_t vaddr, uint32_t* paddr) {
	struct CPU* c = &n->cpu;
	int const seg = segmentOf(c, vaddr, paddr);
	if (seg != SEG_MAPPED) {
		return seg != SEG_UNUSED;
	}
	int const i = tlbFind(c, vaddr);
	if (i < 0) {
		return false;
	}
	const struct TLBEntry* e = &c->tlb[i];
	uint32_t offsetMask = (e->pageMask | 0x1FFF) >> 1;
	uint64_t lo = e->entryLo[((uint32_t) vaddr & (offsetMask + 1)) ? 1 : 0];
	*paddr = ((uint32_t) (lo >> 6) << 12 & ~offsetMask) | ((uint32_t) vaddr & offsetMask);
	return lo & 2;
}

// In 32-bit addressing, an address must be a 32-bit one sign-extended.
static bool address(struct N64* n, uint64_t vaddr, int align, bool store, uint32_t* paddr) {
	if ((vaddr & (align - 1)) && n->compat) {
		vaddr &= ~(uint64_t) (align - 1);
	} else if ((vaddr & (align - 1)) || (!n->compat && (uint64_t) (int64_t) (int32_t) vaddr != vaddr && !cpuWide(&n->cpu))) {
		faultAddress(&n->cpu, vaddr);
		cpuException(n, store ? EXC_ADES : EXC_ADEL, 0);
		return false;
	}
	return translate(n, vaddr, store, paddr);
}

static void branch(struct CPU* c, bool taken, uint64_t target) {
	if (taken) {
		c->nextPc = target;
	}
	c->nextInDelay = true;
}

// Branch-likely skips its delay slot when not taken.
static void branchLikely(struct CPU* c, bool taken, uint64_t target) {
	if (taken) {
		c->nextPc = target;
		c->nextInDelay = true;
	} else {
		c->pc = c->nextPc;
		c->nextPc += 4;
	}
}

static void tlbWrite(struct N64* n, int i) {
	struct CPU* c = &n->cpu;
	struct TLBEntry* e = &c->tlb[i & 31];
	struct TLBEntry old = *e;
	// The mask goes by pairs of bits: one with its upper bit clear masks nothing (ares).
	uint32_t pm = (uint32_t) c->cop0[CP0_PAGEMASK] & 0x01554000;
	e->pageMask = pm | pm >> 1;
	e->entryHi = c->cop0[CP0_ENTRYHI] & ~(uint64_t) e->pageMask & ~0x1F00ull;
	uint64_t g = c->cop0[CP0_ENTRYLO0] & c->cop0[CP0_ENTRYLO1] & 1;
	// 32 bits of physical address: PFN bits 6-25.
	e->entryLo[0] = (c->cop0[CP0_ENTRYLO0] & 0x03FFFFFE) | g;
	e->entryLo[1] = (c->cop0[CP0_ENTRYLO1] & 0x03FFFFFE) | g;
	// Writing an entry as it was maps nothing differently.
	if (old.pageMask != e->pageMask || old.entryHi != e->entryHi || old.entryLo[0] != e->entryLo[0] || old.entryLo[1] != e->entryLo[1]) {
		++n->blockGen;
		cpuTlbChanged(n);
	}
}

static uint32_t randomIndex(struct N64* n) {
	uint32_t wired = n->cpu.cop0[CP0_WIRED] & 63;
	if (wired > 31) {
		return 63 - (uint32_t) (n->cycles % 64);
	}
	return 31 - (uint32_t) ((n->cycles) % (32 - wired));
}

static uint64_t cop0Read(struct N64* n, int r) {
	switch (r) {
	case CP0_COUNT:
		return cpuCount(n);
	case CP0_RANDOM:
		return randomIndex(n);
	case 21:
	case 22:
	case 23:
	case 24:
	case 25:
	case 31:
		return n->cpu.cop0[CP0_LATCH];
	}
	return n->cpu.cop0[r];
}

// The registers' writable bits as the VR4300 keeps them (nemu64-test, ares).
static void cop0Write(struct N64* n, int r, uint64_t v) {
	uint64_t* cop0 = n->cpu.cop0;
	cop0[CP0_LATCH] = v;
	switch (r) {
	case CP0_LATCH:
	case 21:
	case 22:
	case 23:
	case 24:
	case 25:
	case 31:
	case 27:
	case CP0_TAGHI:
		return;
	case CP0_CONTEXT:
		cop0[r] = (cop0[r] & 0x7FFFF0) | (v & ~0x7FFFFFull);
		return;
	case CP0_XCONTEXT:
		cop0[r] = (cop0[r] & 0x1FFFFFFF0ull) | (v & ~0x1FFFFFFFFull);
		return;
	case CP0_CONFIG:
		cop0[r] = (v & 0x0F00800F) | 0x70066460;
		return;
	case CP0_LLADDR:
		cop0[r] = (cop0[r] & ~0xFFFFFFFFull) | (uint32_t) v;
		return;
	case 18:
		cop0[r] = v & 0xFFFFFFFB;
		return;
	case 19:
		cop0[r] = v & 0xF;
		return;
	case 26:
		cop0[r] = v & 0xFF;
		return;
	case CP0_TAGLO:
		cop0[r] = (uint32_t) v;
		return;
	case CP0_COUNT:
		n->cpu.countBase = n->cycles - 2 * (uint64_t) (uint32_t) v;
		n->cpu.lastCount = (uint32_t) v;
		return;
	// The timer is checked from the Count at the write on, as each instruction checks it (compiled
	// code checks it only where blocks start).
	case CP0_COMPARE:
		cop0[r] = (uint32_t) v;
		cop0[CP0_CAUSE] &= ~0x8000ull;
		n->cpu.lastCount = cpuCount(n);
		return;
	case CP0_CAUSE:
		cop0[r] = (cop0[r] & ~0x300ull) | (v & 0x300);
		return;
	case CP0_STATUS:
		cop0[r] = v & 0xFFF7FFFF;
		return;
	case CP0_WIRED:
		cop0[r] = v & 63;
		return;
	case CP0_INDEX:
		cop0[r] = v & 0x8000003F;
		return;
	case CP0_ENTRYLO0:
	case CP0_ENTRYLO1:
		cop0[r] = v & 0x3FFFFFFF;
		return;
	case CP0_PAGEMASK:
		cop0[r] = v & 0x01FFE000;
		return;
	case CP0_ENTRYHI: {
		// Only the ASID takes part in translating.
		uint64_t asid = cop0[r] & 0xFF;
		cop0[r] = v & 0xC00000FFFFFFE0FFull;
		if ((cop0[r] & 0xFF) != asid) {
			++n->blockGen;
			cpuTlbChanged(n);
		}
		return;
	}
	case CP0_PRID:
	case CP0_RANDOM:
	case CP0_BADVADDR:
		return;
	}
	cop0[r] = v;
}

static void cop0Execute(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	// Outside kernel mode COP0 needs CU0.
	if ((c->cop0[CP0_STATUS] & 0x18) && cpuMode(c) != MODE_KERNEL && !(c->cop0[CP0_STATUS] & 1u << 28)) {
		cpuException(n, EXC_CPU, 0);
		return;
	}
	switch (RS) {
	case 0x00:
		c->gpr[RT] = sext32((uint32_t) cop0Read(n, RD));
		return;
	case 0x01:
		c->gpr[RT] = cop0Read(n, RD);
		return;
	// MTC0 writes the whole register, as DMTC0 does.
	case 0x04:
	case 0x05:
		cop0Write(n, RD, c->gpr[RT]);
		return;
	// CFC0, CTC0 and BC0 do nothing.
	case 0x02:
	case 0x06:
	case 0x08:
		return;
	}
	if (!(RS & 0x10)) {
		cpuException(n, EXC_RI, 0);
		return;
	}
	switch (op & 0x3F) {
	case 0x01: {
		struct TLBEntry* e = &c->tlb[c->cop0[CP0_INDEX] & 31];
		c->cop0[CP0_PAGEMASK] = e->pageMask;
		c->cop0[CP0_ENTRYHI] = e->entryHi & ~(uint64_t) e->pageMask;
		c->cop0[CP0_ENTRYLO0] = e->entryLo[0];
		c->cop0[CP0_ENTRYLO1] = e->entryLo[1];
		return;
	}
	case 0x02:
		tlbWrite(n, c->cop0[CP0_INDEX] & 31);
		return;
	case 0x06:
		tlbWrite(n, randomIndex(n));
		return;
	case 0x08:
		c->cop0[CP0_INDEX] = 0x80000000;
		for (int i = 0; i < 32; ++i) {
			struct TLBEntry* e = &c->tlb[i];
			uint64_t mask = ~((uint64_t) e->pageMask | 0x1FFF) & 0xC00000FFFFFFE000ull;
			bool global = e->entryLo[0] & e->entryLo[1] & 1;
			if (((e->entryHi ^ c->cop0[CP0_ENTRYHI]) & mask) == 0 &&
			    (global || ((e->entryHi ^ c->cop0[CP0_ENTRYHI]) & 0xFF) == 0)) {
				c->cop0[CP0_INDEX] = i;
				break;
			}
		}
		return;
	case 0x18:
		if (c->cop0[CP0_STATUS] & ST_ERL) {
			c->pc = c->cop0[CP0_ERROREPC];
			c->cop0[CP0_STATUS] &= ~(uint64_t) ST_ERL;
		} else {
			c->pc = c->cop0[CP0_EPC];
			c->cop0[CP0_STATUS] &= ~(uint64_t) ST_EXL;
		}
		c->nextPc = c->pc + 4;
		c->llBit = false;
		return;
	// RFE, the R3000's; the other functions do nothing (nemu64-test's emulator probes).
	case 0x10:
		cpuException(n, EXC_RI, 0);
		return;
	}
}

// In lle-compatible accuracy, an unmapped access within 64 KiB of address zero (a null pointer) reads
// zero and writes nothing instead of missing the TLB.
static bool nullAccess(struct N64* n, uint64_t vaddr) {
	uint32_t v = (uint32_t) vaddr, p;
	return n->compat && (v < 0x10000 || v >= 0xFFFF0000u) && !cpuPeekAddress(n, vaddr, &p);
}

// Big-endian bytes, as RDRAM and the cache lines hold them.
static uint64_t bytesBE(const uint8_t* b, int size) {
	uint64_t v = 0;
	for (int k = 0; k < size; ++k) {
		v = v << 8 | b[k];
	}
	return v;
}

static void putBE(uint8_t* b, int size, uint64_t v) {
	switch (size) {
	case 8:
		b[7] = (uint8_t) v, b[6] = (uint8_t) (v >> 8), b[5] = (uint8_t) (v >> 16), b[4] = (uint8_t) (v >> 24);
		v >>= 32;
		// fall through
	case 4:
		b[3] = (uint8_t) v, b[2] = (uint8_t) (v >> 8);
		v >>= 16;
		// fall through
	case 2:
		b[1] = (uint8_t) v;
		v >>= 8;
		// fall through
	default:
		b[0] = (uint8_t) v;
	}
}

static bool loadAt(struct N64* n, uint64_t vaddr, int size, uint64_t* out, uint32_t* paddr) {
	struct CPU* c = &n->cpu;
	uint32_t p = 0;
	if (nullAccess(n, vaddr)) {
		*out = 0;
		*paddr = 0;
		return true;
	}
	if (!address(n, vaddr, size, false, &p)) {
		return false;
	}
	p ^= endianLanes(c, size);
	*paddr = p;
	if (p >= RDRAM_SIZE) {
		c->stall += readCost(p);
	} else if (c->cachedAccess) {
		uint32_t i = dcacheLine(n, vaddr, p);
		if (!n->compat) {
			*out = bytesBE(&c->dcacheData[i][p & (16 - size)], size);
			if (n->idle.tracing) {
				idleLoad(n, p, size, *out);
			}
			return true;
		}
	}
	switch (size) {
	case 1:
		*out = busRead8(n, p);
		break;
	case 2:
		*out = busRead16(n, p);
		break;
	case 4:
		*out = busRead32(n, p);
		break;
	default:
		*out = busRead64(n, p);
	}
	if (n->idle.tracing) {
		idleLoad(n, p, size, *out);
	}
	return true;
}

static bool load(struct N64* n, uint64_t vaddr, int size, uint64_t* out) {
	uint32_t p;
	return loadAt(n, vaddr, size, out, &p);
}

static bool store(struct N64* n, uint64_t vaddr, int size, uint64_t v) {
	struct CPU* c = &n->cpu;
	uint32_t p;
	if (nullAccess(n, vaddr)) {
		return true;
	}
	if (!address(n, vaddr, size, true, &p)) {
		return false;
	}
	p ^= endianLanes(c, size);
	if (n->idle.tracing) {
		idleStore(n, p, size, v);
	}
	if (p < RDRAM_SIZE && c->cachedAccess) {
		uint32_t i = dcacheLine(n, vaddr, p);
		c->dcacheDirty[i] = true;
		if (!n->compat) {
			putBE(&c->dcacheData[i][p & (16 - size)], size, v);
			return true;
		}
	}
	if (!c->cachedAccess && p < 0x04000000 && busRepeatWrite(n, p, size, v)) {
		return true;
	}
	switch (size) {
	case 1:
		busWrite8(n, p, v);
		break;
	case 2:
		busWrite16(n, p, v);
		break;
	case 4:
		busWrite32(n, p, v);
		break;
	default:
		busWrite64(n, p, v);
	}
	return true;
}

static void special(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	uint64_t* r = c->gpr;
	uint64_t s = r[RS], t = r[RT];
	switch (op & 0x3F) {
	case 0x00:
		r[RD] = sext32((uint32_t) t << SA);
		return;
	case 0x02:
		r[RD] = sext32((uint32_t) t >> SA);
		return;
	case 0x03:
		r[RD] = sext32((uint32_t) ((int64_t) t >> SA));
		return;
	case 0x04:
		r[RD] = sext32((uint32_t) t << (s & 31));
		return;
	case 0x06:
		r[RD] = sext32((uint32_t) t >> (s & 31));
		return;
	case 0x07:
		r[RD] = sext32((uint32_t) ((int64_t) t >> (s & 31)));
		return;
	case 0x08:
		branch(c, true, s);
		return;
	case 0x09:
		r[RD] = c->pc + 4;
		branch(c, true, s);
		return;
	case 0x0C:
		cpuException(n, EXC_SYS, 0);
		return;
	case 0x0D:
		cpuException(n, EXC_BP, 0);
		return;
	case 0x0F:
		return;
	case 0x10:
		r[RD] = c->hi;
		return;
	case 0x11:
		c->hi = s;
		return;
	case 0x12:
		r[RD] = c->lo;
		return;
	case 0x13:
		c->lo = s;
		return;
	case 0x14:
		r[RD] = t << (s & 63);
		return;
	case 0x16:
		r[RD] = t >> (s & 63);
		return;
	case 0x17:
		r[RD] = (uint64_t) ((int64_t) t >> (s & 63));
		return;
	// The multiply/divide unit holds the pipeline; cycles fitted to the same timing ROM.
	// MULT: rs whole times rt sign-extended from 35 bits, as hardware does with operands that aren't
	// sign-extended 32-bit ones.
	case 0x18: {
		c->stall += 3;
		int64_t p = (int64_t) ((uint64_t) s * (uint64_t) ((int64_t) (t << 29) >> 29));
		c->lo = sext32((uint32_t) p);
		c->hi = sext32((uint32_t) (p >> 32));
		return;
	}
	case 0x19: {
		c->stall += 3;
		uint64_t p = (uint64_t) (uint32_t) s * (uint32_t) t;
		c->lo = sext32((uint32_t) p);
		c->hi = sext32((uint32_t) (p >> 32));
		return;
	}
	case 0x1A: {
		c->stall += 36;
		int32_t a = (int32_t) s, b = (int32_t) t;
		if (b == 0) {
			c->lo = a < 0 ? 1 : ~0ull;
			c->hi = sext32((uint32_t) a);
		} else if (a == INT32_MIN && b == -1) {
			c->lo = sext32((uint32_t) a);
			c->hi = 0;
		} else {
			c->lo = sext32((uint32_t) (a / b));
			c->hi = sext32((uint32_t) (a % b));
		}
		return;
	}
	case 0x1B: {
		c->stall += 36;
		uint32_t a = (uint32_t) s, b = (uint32_t) t;
		if (b == 0) {
			c->lo = ~0ull;
			c->hi = sext32(a);
		} else {
			c->lo = sext32(a / b);
			c->hi = sext32(a % b);
		}
		return;
	}
	case 0x1C: {
		c->stall += 6;
		__int128 p = (__int128) (int64_t) s * (int64_t) t;
		c->lo = (uint64_t) p;
		c->hi = (uint64_t) (p >> 64);
		return;
	}
	case 0x1D: {
		c->stall += 6;
		unsigned __int128 p = (unsigned __int128) s * t;
		c->lo = (uint64_t) p;
		c->hi = (uint64_t) (p >> 64);
		return;
	}
	case 0x1E: {
		c->stall += 67;
		int64_t a = (int64_t) s, b = (int64_t) t;
		if (b == 0) {
			c->lo = a < 0 ? 1 : ~0ull;
			c->hi = s;
		} else if (a == INT64_MIN && b == -1) {
			c->lo = s;
			c->hi = 0;
		} else {
			c->lo = (uint64_t) (a / b);
			c->hi = (uint64_t) (a % b);
		}
		return;
	}
	case 0x1F:
		c->stall += 67;
		if (t == 0) {
			c->lo = ~0ull;
			c->hi = s;
		} else {
			c->lo = s / t;
			c->hi = s % t;
		}
		return;
	case 0x20: {
		int32_t v;
		if (__builtin_add_overflow((int32_t) s, (int32_t) t, &v)) {
			cpuException(n, EXC_OV, 0);
		} else {
			r[RD] = sext32((uint32_t) v);
		}
		return;
	}
	case 0x21:
		r[RD] = sext32((uint32_t) (s + t));
		return;
	case 0x22: {
		int32_t v;
		if (__builtin_sub_overflow((int32_t) s, (int32_t) t, &v)) {
			cpuException(n, EXC_OV, 0);
		} else {
			r[RD] = sext32((uint32_t) v);
		}
		return;
	}
	case 0x23:
		r[RD] = sext32((uint32_t) (s - t));
		return;
	case 0x24:
		r[RD] = s & t;
		return;
	case 0x25:
		r[RD] = s | t;
		return;
	case 0x26:
		r[RD] = s ^ t;
		return;
	case 0x27:
		r[RD] = ~(s | t);
		return;
	case 0x2A:
		r[RD] = (int64_t) s < (int64_t) t;
		return;
	case 0x2B:
		r[RD] = s < t;
		return;
	case 0x2C: {
		int64_t v;
		if (__builtin_add_overflow((int64_t) s, (int64_t) t, &v)) {
			cpuException(n, EXC_OV, 0);
		} else {
			r[RD] = (uint64_t) v;
		}
		return;
	}
	case 0x2D:
		r[RD] = s + t;
		return;
	case 0x2E: {
		int64_t v;
		if (__builtin_sub_overflow((int64_t) s, (int64_t) t, &v)) {
			cpuException(n, EXC_OV, 0);
		} else {
			r[RD] = (uint64_t) v;
		}
		return;
	}
	case 0x2F:
		r[RD] = s - t;
		return;
	case 0x30:
		if ((int64_t) s >= (int64_t) t) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x31:
		if (s >= t) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x32:
		if ((int64_t) s < (int64_t) t) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x33:
		if (s < t) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x34:
		if (s == t) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x36:
		if (s != t) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x38:
		r[RD] = t << SA;
		return;
	case 0x3A:
		r[RD] = t >> SA;
		return;
	case 0x3B:
		r[RD] = (uint64_t) ((int64_t) t >> SA);
		return;
	case 0x3C:
		r[RD] = t << (SA + 32);
		return;
	case 0x3E:
		r[RD] = t >> (SA + 32);
		return;
	case 0x3F:
		r[RD] = (uint64_t) ((int64_t) t >> (SA + 32));
		return;
	}
	cpuException(n, EXC_RI, 0);
}

static void regimm(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	int64_t s = (int64_t) c->gpr[RS];
	// From the next fetch address: in a jump's delay slot, the jump's target.
	uint64_t target = c->pc + (IMM << 2);
	int64_t imm = (int64_t) IMM;
	switch (RT) {
	case 0x00:
		branch(c, s < 0, target);
		return;
	case 0x01:
		branch(c, s >= 0, target);
		return;
	case 0x02:
		branchLikely(c, s < 0, target);
		return;
	case 0x03:
		branchLikely(c, s >= 0, target);
		return;
	case 0x08:
		if (s >= imm) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x09:
		if ((uint64_t) s >= (uint64_t) imm) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x0A:
		if (s < imm) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x0B:
		if ((uint64_t) s < (uint64_t) imm) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x0C:
		if (s == imm) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x0E:
		if (s != imm) {
			cpuException(n, EXC_TR, 0);
		}
		return;
	case 0x10:
		c->gpr[31] = c->pc + 4;
		branch(c, s < 0, target);
		return;
	case 0x11:
		c->gpr[31] = c->pc + 4;
		branch(c, s >= 0, target);
		return;
	case 0x12:
		c->gpr[31] = c->pc + 4;
		branchLikely(c, s < 0, target);
		return;
	case 0x13:
		c->gpr[31] = c->pc + 4;
		branchLikely(c, s >= 0, target);
		return;
	}
	cpuException(n, EXC_RI, 0);
}

static void loadStore(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	uint64_t* r = c->gpr;
	uint64_t vaddr = r[RS] + IMM;
	uint64_t v;
	switch (op >> 26) {
	case 0x20:
		if (load(n, vaddr, 1, &v)) {
			r[RT] = (uint64_t) (int64_t) (int8_t) v;
		}
		return;
	case 0x21:
		if (load(n, vaddr, 2, &v)) {
			r[RT] = (uint64_t) (int64_t) (int16_t) v;
		}
		return;
	case 0x23:
		if (load(n, vaddr, 4, &v)) {
			r[RT] = sext32((uint32_t) v);
		}
		return;
	case 0x24:
		if (load(n, vaddr, 1, &v)) {
			r[RT] = (uint8_t) v;
		}
		return;
	case 0x25:
		if (load(n, vaddr, 2, &v)) {
			r[RT] = (uint16_t) v;
		}
		return;
	case 0x27:
		if (load(n, vaddr, 4, &v)) {
			r[RT] = (uint32_t) v;
		}
		return;
	case 0x37:
		if (load(n, vaddr, 8, &v)) {
			r[RT] = v;
		}
		return;
	// The partial loads and stores: little-endian, the byte's place in the word mirrored.
	case 0x22:
	case 0x26: {
		if (!load(n, vaddr & ~3ull, 4, &v)) {
			return;
		}
		int const off = endianLanes(c, 4) ? 3 - (int) (vaddr & 3) : (int) (vaddr & 3);
		uint32_t w = (uint32_t) v, old = (uint32_t) r[RT];
		if ((op >> 26) == 0x22) {
			int sh = off * 8;
			r[RT] = sext32((old & ~(0xFFFFFFFFu << sh)) | (w << sh));
		} else if (off == 3) {
			r[RT] = sext32(w);
		} else {
			// Part of a word: the register's upper half stays.
			int sh = (3 - off) * 8;
			r[RT] = (r[RT] & ~0xFFFFFFFFull) | ((old & ~(0xFFFFFFFFu >> sh)) | (w >> sh));
		}
		return;
	}
	case 0x1A:
	case 0x1B: {
		if (!load(n, vaddr & ~7ull, 8, &v)) {
			return;
		}
		int const off = endianLanes(c, 4) ? 7 - (int) (vaddr & 7) : (int) (vaddr & 7);
		uint64_t old = r[RT];
		if ((op >> 26) == 0x1A) {
			int sh = off * 8;
			r[RT] = (old & ~(~0ull << sh)) | (v << sh);
		} else {
			int sh = (7 - off) * 8;
			r[RT] = (old & ~(~0ull >> sh)) | (v >> sh);
		}
		return;
	}
	case 0x28:
		store(n, vaddr, 1, r[RT]);
		return;
	case 0x29:
		store(n, vaddr, 2, r[RT]);
		return;
	case 0x2B:
		store(n, vaddr, 4, r[RT]);
		return;
	case 0x3F:
		store(n, vaddr, 8, r[RT]);
		return;
	case 0x2A:
	case 0x2E: {
		// The store's exceptions, with the address as given.
		uint32_t p;
		if (!address(n, vaddr, 1, true, &p) || !load(n, vaddr & ~3ull, 4, &v)) {
			return;
		}
		int const off = endianLanes(c, 4) ? 3 - (int) (vaddr & 3) : (int) (vaddr & 3);
		uint32_t w = (uint32_t) v, t = (uint32_t) r[RT];
		if ((op >> 26) == 0x2A) {
			int sh = off * 8;
			w = (w & ~(0xFFFFFFFFu >> sh)) | (t >> sh);
		} else {
			int sh = (3 - off) * 8;
			w = (w & ~(0xFFFFFFFFu << sh)) | (t << sh);
		}
		store(n, vaddr & ~3ull, 4, w);
		return;
	}
	case 0x2C:
	case 0x2D: {
		uint32_t p;
		if (!address(n, vaddr, 1, true, &p) || !load(n, vaddr & ~7ull, 8, &v)) {
			return;
		}
		int const off = endianLanes(c, 4) ? 7 - (int) (vaddr & 7) : (int) (vaddr & 7);
		uint64_t t = r[RT];
		if ((op >> 26) == 0x2C) {
			int sh = off * 8;
			v = (v & ~(~0ull >> sh)) | (t >> sh);
		} else {
			int sh = (7 - off) * 8;
			v = (v & ~(~0ull << sh)) | (t << sh);
		}
		store(n, vaddr & ~7ull, 8, v);
		return;
	}
	case 0x30:
	case 0x34: {
		int size = (op >> 26) == 0x30 ? 4 : 8;
		uint32_t p;
		uint64_t v;
		if (!loadAt(n, vaddr, size, &v, &p)) {
			return;
		}
		r[RT] = size == 4 ? sext32((uint32_t) v) : v;
		n->idle.fatal = true;
		c->llBit = true;
		c->cop0[CP0_LLADDR] = p >> 4;
		return;
	}
	case 0x38:
	case 0x3C: {
		n->idle.fatal = true;
		int size = (op >> 26) == 0x38 ? 4 : 8;
		if (c->llBit) {
			if (!store(n, vaddr, size, r[RT])) {
				return;
			}
		}
		r[RT] = c->llBit;
		return;
	}
	}
	cpuException(n, EXC_RI, 0);
}

static void cop1Memory(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	uint64_t vaddr = c->gpr[RS] + IMM;
	bool fr = c->cop0[CP0_STATUS] & ST_FR;
	int ft = RT;
	uint64_t v;
	switch (op >> 26) {
	case 0x31:
		if (load(n, vaddr, 4, &v)) {
			if (fr) {
				c->fpr[ft] = (c->fpr[ft] & ~0xFFFFFFFFull) | (uint32_t) v;
			} else if (ft & 1) {
				c->fpr[ft & ~1] = (c->fpr[ft & ~1] & 0xFFFFFFFFull) | v << 32;
			} else {
				c->fpr[ft] = (c->fpr[ft] & ~0xFFFFFFFFull) | (uint32_t) v;
			}
		}
		return;
	case 0x35:
		if (load(n, vaddr, 8, &v)) {
			c->fpr[fr ? ft : ft & ~1] = v;
		}
		return;
	case 0x39:
		if (fr || !(ft & 1)) {
			store(n, vaddr, 4, (uint32_t) c->fpr[ft]);
		} else {
			store(n, vaddr, 4, c->fpr[ft & ~1] >> 32);
		}
		return;
	case 0x3D:
		store(n, vaddr, 8, c->fpr[fr ? ft : ft & ~1]);
		return;
	}
}

// The 64-bit instructions, reserved in user and supervisor modes with 32-bit addressing (ares).
static bool wideOp(uint32_t op) {
	switch (op >> 26) {
	case 0x00:
		switch (op & 0x3F) {
		case 0x14: case 0x16: case 0x17: case 0x1C: case 0x1D: case 0x1E: case 0x1F:
		case 0x2C: case 0x2D: case 0x2E: case 0x2F:
		case 0x38: case 0x3A: case 0x3B: case 0x3C: case 0x3E: case 0x3F:
			return true;
		}
		return false;
	case 0x18: case 0x19: case 0x1A: case 0x1B: case 0x2C: case 0x2D: case 0x34: case 0x37: case 0x3C: case 0x3F:
		return true;
	}
	return false;
}

static void execute(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	uint64_t* r = c->gpr;
	uint64_t s = r[RS];
	if ((c->cop0[CP0_STATUS] & 0x18) && cpuMode(c) != MODE_KERNEL && !cpuWide(c) && wideOp(op)) {
		cpuException(n, EXC_RI, 0);
		return;
	}
	// From the next fetch address: in a jump's delay slot, the jump's target.
	uint64_t target = c->pc + (IMM << 2);
	switch (op >> 26) {
	case 0x00:
		special(n, op);
		return;
	case 0x01:
		regimm(n, op);
		return;
	case 0x02:
	case 0x03: {
		uint64_t jump = (c->pc & ~0x0FFFFFFFull) | (uint64_t) (op & 0x03FFFFFF) << 2;
		if (op >> 26 == 0x03) {
			r[31] = c->pc + 4;
		}
		branch(c, true, jump);
		return;
	}
	case 0x04:
		branch(c, s == r[RT], target);
		return;
	case 0x05:
		branch(c, s != r[RT], target);
		return;
	case 0x06:
		branch(c, (int64_t) s <= 0, target);
		return;
	case 0x07:
		branch(c, (int64_t) s > 0, target);
		return;
	case 0x08: {
		int32_t v;
		if (__builtin_add_overflow((int32_t) s, (int32_t) IMM, &v)) {
			cpuException(n, EXC_OV, 0);
		} else {
			r[RT] = sext32((uint32_t) v);
		}
		return;
	}
	case 0x09:
		r[RT] = sext32((uint32_t) (s + IMM));
		return;
	case 0x0A:
		r[RT] = (int64_t) s < (int64_t) IMM;
		return;
	case 0x0B:
		r[RT] = s < IMM;
		return;
	case 0x0C:
		r[RT] = s & UIMM;
		return;
	case 0x0D:
		r[RT] = s | UIMM;
		return;
	case 0x0E:
		r[RT] = s ^ UIMM;
		return;
	case 0x0F:
		r[RT] = sext32((uint32_t) UIMM << 16);
		return;
	case 0x10:
		cop0Execute(n, op);
		return;
	case 0x11:
		if (!(c->cop0[CP0_STATUS] & ST_CU1)) {
			cpuException(n, EXC_CPU, 1);
			return;
		}
		// BC1 (the unimplemented operation past its four forms) starts with the FPU's causes clear.
		if (RS == 0x08) {
			fpuClearCauses(n);
			if (RT >= 4) {
				c->fcr31 |= 1u << 17;
				cpuException(n, EXC_FPE, 0);
				return;
			}
			bool cond = (c->fcr31 >> 23) & 1;
			bool taken = RT & 1 ? cond : !cond;
			if (RT & 2) {
				branchLikely(c, taken, target);
			} else {
				branch(c, taken, target);
			}
			return;
		}
		fpuExecute(n, op);
		return;
	// COP2: unusable unless enabled; its moves go through the latch, anything else is reserved.
	case 0x12:
		if (!(c->cop0[CP0_STATUS] & ST_CU2)) {
			cpuException(n, EXC_CPU, 2);
		} else if (RS == 0x00 || RS == 0x02) {
			r[RT] = sext32((uint32_t) c->cop2Latch);
		} else if (RS == 0x01) {
			r[RT] = c->cop2Latch;
		} else if (RS == 0x04 || RS == 0x05 || RS == 0x06) {
			c->cop2Latch = r[RT];
		} else {
			cpuException(n, EXC_RI, 2);
		}
		return;
	// LWC2, LDC2, SWC2 and SDC2.
	case 0x32:
	case 0x36:
	case 0x3A:
	case 0x3E:
		cpuException(n, c->cop0[CP0_STATUS] & ST_CU2 ? EXC_RI : EXC_CPU, 2);
		return;
	case 0x14:
		branchLikely(c, s == r[RT], target);
		return;
	case 0x15:
		branchLikely(c, s != r[RT], target);
		return;
	case 0x16:
		branchLikely(c, (int64_t) s <= 0, target);
		return;
	case 0x17:
		branchLikely(c, (int64_t) s > 0, target);
		return;
	case 0x18: {
		int64_t v;
		if (__builtin_add_overflow((int64_t) s, (int64_t) IMM, &v)) {
			cpuException(n, EXC_OV, 0);
		} else {
			r[RT] = (uint64_t) v;
		}
		return;
	}
	case 0x19:
		r[RT] = s + IMM;
		return;
	case 0x2F:
		cacheOp(n, op, s + IMM);
		return;
	case 0x31:
	case 0x35:
	case 0x39:
	case 0x3D:
		if (!(c->cop0[CP0_STATUS] & ST_CU1)) {
			cpuException(n, EXC_CPU, 1);
			return;
		}
		cop1Memory(n, op);
		return;
	}
	loadStore(n, op);
}

#ifdef N64_TRACE
#include <stdio.h>
#include <stdlib.h>

// N64TRACE=file N64TRACEN=count: the PC and a hash of the general registers before each instruction,
// for comparing against another emulator's trace.
static void traceInstruction(struct N64* n) {
	static FILE* file;
	static long count, limit;
	struct CPU* c = &n->cpu;
	if (!file && getenv("N64TRACE")) {
		file = fopen(getenv("N64TRACE"), "wb");
		limit = getenv("N64TRACEN") ? atol(getenv("N64TRACEN")) : 1000000;
	}
	if (!file || count >= limit) {
		return;
	}
	uint32_t rec[2] = { (uint32_t) c->curPc, 2166136261u };
	for (int i = 1; i < 32; ++i) {
		for (int b = 0; b < 8; ++b) {
			rec[1] = (rec[1] ^ (uint8_t) (c->gpr[i] >> (b * 8))) * 16777619u;
		}
	}
	fwrite(rec, 4, 2, file);
	if (++count == limit) {
		fclose(file);
	}
}
#endif

// cpuStep for an instruction already fetched from a cached RDRAM address (paddr): the same
// prologue and timing, without the fetch. If the prologue takes an interrupt, the handler's first
// instruction is fetched as usual.
void cpuStepOp(struct N64* n, uint32_t op, uint32_t paddr) {
	struct CPU* c = &n->cpu;
	uint32_t now = cpuCount(n);
	if ((uint32_t) ((uint32_t) c->cop0[CP0_COMPARE] - c->lastCount - 1) < now - c->lastCount) {
		c->cop0[CP0_CAUSE] |= 0x8000;
	}
	c->lastCount = now;
	c->inDelay = c->nextInDelay;
	c->nextInDelay = false;
	c->curPc = c->pc;
	bool const taken = interruptPending(n);
	if (taken) {
		cpuException(n, EXC_INT, 0);
		c->curPc = c->pc;
		if (!address(n, c->curPc, 4, false, &paddr)) {
			return;
		}
	} else {
		c->cachedAccess = true;
	}
	uint32_t const fetched = fetch(n, c->curPc, paddr);
	if (taken || !n->compat) {
		op = fetched;
	}
#ifdef N64_TRACE
	traceInstruction(n);
#endif
	c->pc = c->nextPc;
	c->nextPc += 4;
	execute(n, op);
	c->gpr[0] = 0;
}

// Before compiled code runs: cpuStep's timer check, and the limit the code runs to, the next event or
// the cycle at which Count reaches Compare. False when an interrupt is due, for cpuStep to take.
bool cpuJitReady(struct N64* n) {
	struct CPU* c = &n->cpu;
	if (!cpuPlainMode(n)) {
		return false;
	}
	uint32_t now = cpuCount(n);
	if ((uint32_t) ((uint32_t) c->cop0[CP0_COMPARE] - c->lastCount - 1) < now - c->lastCount) {
		c->cop0[CP0_CAUSE] |= 0x8000;
	}
	c->lastCount = now;
	if (interruptPending(n)) {
		return false;
	}
	uint32_t delta = (uint32_t) c->cop0[CP0_COMPARE] - now;
	n->jitFire = n->cycles - ((n->cycles - c->countBase) & 1) + 2 * (delta ? delta : 1ull << 32);
	n->jitLimit = n->eventAt < n->jitFire ? n->eventAt : n->jitFire;
	return true;
}

// One instruction for compiled code, which has stored curPc, pc, nextPc and the delay flags.
// Returns whether the code must stop: the program left the straight line, a write threw blocks
// away, an interrupt is due, a write to Count or Compare moved the timer's limit, or a CACHE
// instruction changed the I-cache the code checks.
// LW and SW at an aligned uncached address (the RCP's registers, mostly), as loadStore does them,
// without its decoding; false for anything else.
static bool uncachedWord(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	uint64_t vaddr = c->gpr[RS] + IMM;
	if ((op >> 26 != 0x23 && op >> 26 != 0x2B) || ((uint32_t) vaddr & 0xE0000003) != 0xA0000000 ||
	    (uint64_t) (int64_t) (int32_t) vaddr != vaddr || (n->miMode & 0x80)) {
		return false;
	}
	uint32_t p = (uint32_t) vaddr & 0x1FFFFFFF;
	if (op >> 26 == 0x23) {
		c->stall += readCost(p);
		c->gpr[RT] = sext32(busRead32(n, p));
	} else {
		busWrite32(n, p, (uint32_t) c->gpr[RT]);
	}
	return true;
}

bool cpuJitStep(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	uint64_t pc = c->pc;
	uint32_t gen = n->blockGen;
	if (!uncachedWord(n, op)) {
		execute(n, op);
	}
	c->gpr[0] = 0;
	n->cycles += 1 + c->stall;
	c->stall = 0;
	n->jitLimit = n->eventAt < n->jitFire ? n->eventAt : n->jitFire;
	bool timer = op >> 26 == 0x10 && (RS == 0x04 || RS == 0x05) && (RD == CP0_COUNT || RD == CP0_COMPARE);
	bool icache = op >> 26 == 0x2F && ((op >> 16) & 3) == 0;
	return c->pc != pc || n->blockGen != gen || interruptPending(n) || timer || icache || !cpuPlainMode(n);
}

void cpuStep(struct N64* n) {
	struct CPU* c = &n->cpu;
	uint32_t now = cpuCount(n);
	if ((uint32_t) ((uint32_t) c->cop0[CP0_COMPARE] - c->lastCount - 1) < now - c->lastCount) {
		c->cop0[CP0_CAUSE] |= 0x8000;
	}
	c->lastCount = now;
	c->inDelay = c->nextInDelay;
	c->nextInDelay = false;
	c->curPc = c->pc;
	if (interruptPending(n)) {
		cpuException(n, EXC_INT, 0);
		c->curPc = c->pc;
	}
	uint32_t p;
	if (!address(n, c->curPc, 4, false, &p)) {
		return;
	}
	p ^= endianLanes(c, 4);
	uint32_t op = fetch(n, c->curPc, p);
#ifdef N64_TRACE
	traceInstruction(n);
#endif
	c->pc = c->nextPc;
	c->nextPc += 4;
	execute(n, op);
	c->gpr[0] = 0;
}
