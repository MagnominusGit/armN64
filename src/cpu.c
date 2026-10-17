#include "n64.h"

#include <string.h>

#define RS ((op >> 21) & 31)
#define RT ((op >> 16) & 31)
#define RD ((op >> 11) & 31)
#define SA ((op >> 6) & 31)
#define IMM ((uint64_t) (int64_t) (int16_t) op)
#define UIMM ((uint64_t) (uint16_t) op)

enum { ST_IE = 1, ST_EXL = 2, ST_ERL = 4, ST_FR = 1 << 26, ST_CU1 = 1 << 29, ST_BEV = 1 << 22 };

// Timing, in CPU cycles on top of one a instruction: cache line fills and write-backs, and reads of
// RCP registers (cache figures as in the VR4300 manual and ares; the RCP read cost is fitted to
// pdrome's instruction timing ROM, which counts loop iterations per field on hardware).
enum { ICACHE_FILL = 48, DCACHE_FILL = 40, DCACHE_WRITEBACK = 40, RCP_READ = 22 };

static bool cached(uint64_t vaddr) {
	return ((uint32_t) vaddr & 0xE0000000) != 0xA0000000;
}

static void fetchTiming(struct CPU* c, uint64_t vaddr, uint32_t paddr) {
	if (!cached(vaddr)) {
		c->stall += paddr >= RDRAM_SIZE ? RCP_READ : 0;
		return;
	}
	uint32_t i = ((uint32_t) vaddr >> 5) & 511, tag = paddr >> 5 | 0x80000000u;
	if (c->icacheTag[i] != tag) {
		c->icacheTag[i] = tag;
		c->stall += ICACHE_FILL;
	}
}

static void dataTiming(struct CPU* c, uint64_t vaddr, uint32_t paddr, bool store) {
	if (paddr >= RDRAM_SIZE) {
		c->stall += store ? 0 : RCP_READ;
		return;
	}
	if (!cached(vaddr)) {
		return;
	}
	uint32_t i = ((uint32_t) vaddr >> 4) & 511, tag = paddr >> 4 | 0x80000000u;
	if (c->dcacheTag[i] != tag) {
		c->stall += DCACHE_FILL + (c->dcacheDirty[i] ? DCACHE_WRITEBACK : 0);
		c->dcacheTag[i] = tag;
		c->dcacheDirty[i] = false;
	}
	if (store) {
		c->dcacheDirty[i] = true;
	}
}

// CACHE: invalidates or writes back the line an address selects (only the timing state changes).
static void cacheOp(struct CPU* c, uint32_t op, uint64_t vaddr) {
	int which = (op >> 16) & 3, action = (op >> 18) & 7;
	if (which == 0) {
		c->icacheTag[((uint32_t) vaddr >> 5) & 511] = 0;
		return;
	}
	if (which != 1) {
		return;
	}
	uint32_t i = ((uint32_t) vaddr >> 4) & 511;
	if (action != 3 && c->dcacheDirty[i]) {
		c->stall += DCACHE_WRITEBACK;
		c->dcacheDirty[i] = false;
	}
	if (action == 0 || action == 4 || action == 5) {
		c->dcacheTag[i] = 0;
	}
}

void cpuReset(struct N64* n) {
	struct CPU* c = &n->cpu;
	memset(c, 0, sizeof(*c));
	c->cop0[CP0_STATUS] = 0x34000000;
	c->cop0[CP0_CONFIG] = 0x7006E463;
	c->cop0[CP0_PRID] = 0x00000B22;
	c->cop0[CP0_RANDOM] = 31;
	c->cop0[CP0_EPC] = c->cop0[CP0_ERROREPC] = ~0ull;
	c->pc = 0xFFFFFFFFBFC00000ull;
	c->nextPc = c->pc + 4;
}

// Count runs at half the CPU clock; it is kept as an offset from the cycle counter.
uint32_t cpuCount(struct N64* n) {
	return (uint32_t) ((n->cycles >> 1) - n->cpu.countBase);
}

void cpuException(struct N64* n, int code, int coprocessor) {
	struct CPU* c = &n->cpu;
	uint64_t* cop0 = c->cop0;
	bool refill = (code == EXC_TLBL || code == EXC_TLBS) && c->exception;
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
	c->pc = base + (refill ? 0 : 0x180);
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
}

static bool interruptPending(struct N64* n) {
	uint64_t* cop0 = n->cpu.cop0;
	uint64_t st = cop0[CP0_STATUS];
	return (st & ST_IE) && !(st & (ST_EXL | ST_ERL)) && (cop0[CP0_CAUSE] & st & 0xFF00);
}

// Virtual to physical. Returns false after raising the exception.
static bool translate(struct N64* n, uint64_t vaddr, bool store, uint32_t* paddr) {
	struct CPU* c = &n->cpu;
	uint32_t v = (uint32_t) vaddr;
	if ((v & 0xC0000000) == 0x80000000) {
		*paddr = v & 0x1FFFFFFF;
		return true;
	}
	for (int i = 0; i < 32; ++i) {
		struct TLBEntry* e = &c->tlb[i];
		uint32_t mask = e->pageMask | 0x1FFF;
		if (((v ^ (uint32_t) e->entryHi) & ~mask) != 0) {
			continue;
		}
		bool global = e->entryLo[0] & e->entryLo[1] & 1;
		if (!global && ((e->entryHi ^ c->cop0[CP0_ENTRYHI]) & 0xFF)) {
			continue;
		}
		uint32_t offsetMask = mask >> 1;
		uint64_t lo = e->entryLo[(v & (offsetMask + 1)) ? 1 : 0];
		if (!(lo & 2)) {
			goto miss;
		}
		if (store && !(lo & 4)) {
			c->cop0[CP0_BADVADDR] = vaddr;
			c->cop0[CP0_CONTEXT] = (c->cop0[CP0_CONTEXT] & ~0x7FFFF0ull) | ((v >> 9) & 0x7FFFF0);
			c->cop0[CP0_ENTRYHI] = (c->cop0[CP0_ENTRYHI] & 0xFF) | (vaddr & ~0x1FFFull);
			cpuException(n, EXC_MOD, 0);
			return false;
		}
		*paddr = ((uint32_t) (lo >> 6) << 12 & ~offsetMask) | (v & offsetMask);
		return true;
	}
	c->exception = true;
miss:
	c->cop0[CP0_BADVADDR] = vaddr;
	c->cop0[CP0_CONTEXT] = (c->cop0[CP0_CONTEXT] & ~0x7FFFF0ull) | ((v >> 9) & 0x7FFFF0);
	c->cop0[CP0_ENTRYHI] = (c->cop0[CP0_ENTRYHI] & 0xFF) | (vaddr & ~0x1FFFull);
	cpuException(n, store ? EXC_TLBS : EXC_TLBL, 0);
	return false;
}

// For tools: the physical address of a virtual one, without raising exceptions or changing state.
bool cpuPeekAddress(struct N64* n, uint64_t vaddr, uint32_t* paddr) {
	struct CPU* c = &n->cpu;
	uint32_t v = (uint32_t) vaddr;
	if ((v & 0xC0000000) == 0x80000000) {
		*paddr = v & 0x1FFFFFFF;
		return true;
	}
	for (int i = 0; i < 32; ++i) {
		struct TLBEntry* e = &c->tlb[i];
		uint32_t mask = e->pageMask | 0x1FFF;
		bool global = e->entryLo[0] & e->entryLo[1] & 1;
		if (((v ^ (uint32_t) e->entryHi) & ~mask) || (!global && ((e->entryHi ^ c->cop0[CP0_ENTRYHI]) & 0xFF))) {
			continue;
		}
		uint32_t offsetMask = mask >> 1;
		uint64_t lo = e->entryLo[(v & (offsetMask + 1)) ? 1 : 0];
		*paddr = ((uint32_t) (lo >> 6) << 12 & ~offsetMask) | (v & offsetMask);
		return lo & 2;
	}
	return false;
}

static bool address(struct N64* n, uint64_t vaddr, int align, bool store, uint32_t* paddr) {
	if (vaddr & (align - 1)) {
		n->cpu.cop0[CP0_BADVADDR] = vaddr;
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
	e->pageMask = c->cop0[CP0_PAGEMASK] & 0x01FFE000;
	e->entryHi = c->cop0[CP0_ENTRYHI] & ~(uint64_t) e->pageMask & ~0x1F00ull;
	uint64_t g = c->cop0[CP0_ENTRYLO0] & c->cop0[CP0_ENTRYLO1] & 1;
	e->entryLo[0] = (c->cop0[CP0_ENTRYLO0] & 0x3FFFFFFE) | g;
	e->entryLo[1] = (c->cop0[CP0_ENTRYLO1] & 0x3FFFFFFE) | g;
}

static uint32_t randomIndex(struct N64* n) {
	uint32_t wired = n->cpu.cop0[CP0_WIRED] & 31;
	if (wired > 31) {
		return 31;
	}
	return 31 - (uint32_t) ((n->cycles) % (32 - wired));
}

static uint64_t cop0Read(struct N64* n, int r) {
	switch (r) {
	case CP0_COUNT:
		return cpuCount(n);
	case CP0_RANDOM:
		return randomIndex(n);
	}
	return n->cpu.cop0[r];
}

static void cop0Write(struct N64* n, int r, uint64_t v) {
	uint64_t* cop0 = n->cpu.cop0;
	switch (r) {
	case CP0_COUNT:
		n->cpu.countBase = (n->cycles >> 1) - (uint32_t) v;
		n->cpu.lastCount = (uint32_t) v;
		return;
	case CP0_COMPARE:
		cop0[r] = (uint32_t) v;
		cop0[CP0_CAUSE] &= ~0x8000ull;
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
		cop0[r] = (cop0[r] & 0x80000000) | (v & 63);
		return;
	case CP0_ENTRYLO0:
	case CP0_ENTRYLO1:
		cop0[r] = v & 0x3FFFFFFF;
		return;
	case CP0_PAGEMASK:
		cop0[r] = v & 0x01FFE000;
		return;
	case CP0_ENTRYHI:
		cop0[r] = v & 0xC00000FFFFFFE0FFull;
		return;
	case CP0_PRID:
	case CP0_RANDOM:
	case CP0_BADVADDR:
		return;
	}
	cop0[r] = v;
}

static void cop0Execute(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	switch (RS) {
	case 0x00:
		c->gpr[RT] = sext32((uint32_t) cop0Read(n, RD));
		return;
	case 0x01:
		c->gpr[RT] = cop0Read(n, RD);
		return;
	case 0x04:
		cop0Write(n, RD, sext32((uint32_t) c->gpr[RT]));
		return;
	case 0x05:
		cop0Write(n, RD, c->gpr[RT]);
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
	}
	cpuException(n, EXC_RI, 0);
}

static bool load(struct N64* n, uint64_t vaddr, int size, uint64_t* out) {
	uint32_t p;
	if (!address(n, vaddr, size, false, &p)) {
		return false;
	}
	dataTiming(&n->cpu, vaddr, p, false);
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
	return true;
}

static bool store(struct N64* n, uint64_t vaddr, int size, uint64_t v) {
	uint32_t p;
	if (!address(n, vaddr, size, true, &p)) {
		return false;
	}
	dataTiming(&n->cpu, vaddr, p, true);
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
		r[RD] = c->curPc + 8;
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
	case 0x18: {
		c->stall += 3;
		int64_t p = (int64_t) (int32_t) s * (int32_t) t;
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
	uint64_t target = c->curPc + 4 + (IMM << 2);
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
		c->gpr[31] = c->curPc + 8;
		branch(c, s < 0, target);
		return;
	case 0x11:
		c->gpr[31] = c->curPc + 8;
		branch(c, s >= 0, target);
		return;
	case 0x12:
		c->gpr[31] = c->curPc + 8;
		branchLikely(c, s < 0, target);
		return;
	case 0x13:
		c->gpr[31] = c->curPc + 8;
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
	case 0x22:
	case 0x26: {
		if (!load(n, vaddr & ~3ull, 4, &v)) {
			return;
		}
		uint32_t w = (uint32_t) v, old = (uint32_t) r[RT];
		if ((op >> 26) == 0x22) {
			int sh = (vaddr & 3) * 8;
			r[RT] = sext32((old & ~(0xFFFFFFFFu << sh)) | (w << sh));
		} else {
			int sh = (3 - (vaddr & 3)) * 8;
			r[RT] = sext32((old & ~(0xFFFFFFFFu >> sh)) | (w >> sh));
		}
		return;
	}
	case 0x1A:
	case 0x1B: {
		if (!load(n, vaddr & ~7ull, 8, &v)) {
			return;
		}
		uint64_t old = r[RT];
		if ((op >> 26) == 0x1A) {
			int sh = (vaddr & 7) * 8;
			r[RT] = (old & ~(~0ull << sh)) | (v << sh);
		} else {
			int sh = (7 - (vaddr & 7)) * 8;
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
		if (!load(n, vaddr & ~3ull, 4, &v)) {
			return;
		}
		uint32_t w = (uint32_t) v, t = (uint32_t) r[RT];
		if ((op >> 26) == 0x2A) {
			int sh = (vaddr & 3) * 8;
			w = (w & ~(0xFFFFFFFFu >> sh)) | (t >> sh);
		} else {
			int sh = (3 - (vaddr & 3)) * 8;
			w = (w & ~(0xFFFFFFFFu << sh)) | (t << sh);
		}
		store(n, vaddr & ~3ull, 4, w);
		return;
	}
	case 0x2C:
	case 0x2D: {
		if (!load(n, vaddr & ~7ull, 8, &v)) {
			return;
		}
		uint64_t t = r[RT];
		if ((op >> 26) == 0x2C) {
			int sh = (vaddr & 7) * 8;
			v = (v & ~(~0ull >> sh)) | (t >> sh);
		} else {
			int sh = (7 - (vaddr & 7)) * 8;
			v = (v & ~(~0ull << sh)) | (t << sh);
		}
		store(n, vaddr & ~7ull, 8, v);
		return;
	}
	case 0x30:
	case 0x34: {
		int size = (op >> 26) == 0x30 ? 4 : 8;
		uint32_t p;
		if (!address(n, vaddr, size, false, &p)) {
			return;
		}
		r[RT] = size == 4 ? sext32(busRead32(n, p)) : busRead64(n, p);
		c->llBit = true;
		c->cop0[CP0_LLADDR] = p >> 4;
		return;
	}
	case 0x38:
	case 0x3C: {
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

static void execute(struct N64* n, uint32_t op) {
	struct CPU* c = &n->cpu;
	uint64_t* r = c->gpr;
	uint64_t s = r[RS];
	uint64_t target = c->curPc + 4 + (IMM << 2);
	switch (op >> 26) {
	case 0x00:
		special(n, op);
		return;
	case 0x01:
		regimm(n, op);
		return;
	case 0x02:
	case 0x03: {
		uint64_t jump = ((c->curPc + 4) & ~0x0FFFFFFFull) | (uint64_t) (op & 0x03FFFFFF) << 2;
		if (op >> 26 == 0x03) {
			r[31] = c->curPc + 8;
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
		if (RS == 0x08) {
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
	case 0x12:
		cpuException(n, EXC_CPU, 2);
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
		cacheOp(c, op, s + IMM);
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
	fetchTiming(c, c->curPc, p);
	uint32_t op = busRead32(n, p);
#ifdef N64_TRACE
	traceInstruction(n);
#endif
	c->pc = c->nextPc;
	c->nextPc += 4;
	execute(n, op);
	c->gpr[0] = 0;
}
