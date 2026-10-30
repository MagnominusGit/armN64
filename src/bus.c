#include "n64.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint32_t* reg(uint32_t* regs, uint32_t addr, int count) {
	unsigned i = (addr & 0xFFFFF) >> 2;
	return i < (unsigned) count ? &regs[i] : NULL;
}

// SP DMA: rows of len+1 bytes (rounded to 8), count+1 rows, skipping between rows in RDRAM.
static void spDma(struct N64* n, uint32_t lenReg, bool toRdram) {
	uint32_t len = ((lenReg & 0xFFF) | 7) + 1;
	uint32_t count = ((lenReg >> 12) & 0xFF) + 1;
	uint32_t skip = lenReg >> 20;
	uint32_t mem = n->sp.memAddr & 0x1FF8;
	uint32_t base = mem & 0x1000;
	uint32_t dram = n->sp.dramAddr & (n->compat ? 0xFFFFFF : 0xFFFFF8);
	for (uint32_t row = 0; row < count; ++row) {
		if (toRdram) {
			n64RdramWrite(n, dram, len);
		} else {
			n64RdramRead(n, dram, len);
		}
		if ((mem & 0xFFF) + len <= 0x1000 && (dram & (RDRAM_SIZE - 1)) + len <= RDRAM_SIZE) {
			// The row wraps neither memory.
			uint8_t* sp = &n->sp.mem[base | (mem & 0xFFF)];
			uint8_t* d = &n->rdram[dram & (RDRAM_SIZE - 1)];
			memcpy(toRdram ? d : sp, toRdram ? sp : d, len);
		} else for (uint32_t i = 0; i < len; ++i) {
			uint8_t* sp = &n->sp.mem[base | ((mem + i) & 0xFFF)];
			uint32_t d = (dram + i) & (RDRAM_SIZE - 1);
			if (toRdram) {
				n->rdram[d] = *sp;
			} else {
				*sp = n->rdram[d];
			}
		}
		mem = base | ((mem + len) & 0xFFF);
		dram += len + skip;
	}
	n->sp.memAddr = mem;
	n->sp.dramAddr = dram;
	if (!toRdram && base) {
		++n->rspCodeGen;
	}
}

static void spWriteStatus(struct N64* n, uint32_t v) {
	uint32_t* s = &n->sp.status;
	if ((v & 1) && (*s & 1)) {
		++n->stats.rspTasks;
		if (n->hleCheck && be32(&n->sp.mem[0xFC0]) == 1 && !(be32(&n->sp.mem[0xFC4]) & 1)) {
			// The last graphics task's commands, from its microcode and at high level.
			static uint64_t task;
			int shown = 0;
			for (int op = 0; op < 64; ++op) {
				if (n->checkCounts[0][op] != n->checkCounts[1][op] && shown++ < 12) {
					fprintf(stderr, "HLECHECK task %llu field %llu op %02x microcode %u high level %u\n", (unsigned long long) task, (unsigned long long) n->fields, op, n->checkCounts[0][op], n->checkCounts[1][op]);
				}
			}
			// The first command that differs, triangles, no-ops and pipe, load and tile syncs left out.
			uint32_t i = 0, j = 0;
			while (shown && i < n->checkLen[0] && j < n->checkLen[1]) {
				uint64_t a = n->checkSeq[0][i], b = n->checkSeq[1][j];
				int opA = (a >> 56) & 0x3F, opB = (b >> 56) & 0x3F;
				if ((opA & 0x38) == 0x08 || opA == 0x00 || (opA >= 0x26 && opA <= 0x28)) {
					++i;
					continue;
				}
				if ((opB & 0x38) == 0x08 || opB == 0x00 || (opB >= 0x26 && opB <= 0x28)) {
					++j;
					continue;
				}
				int op = opA;
				bool sync = op == 0x29, image = op >= 0x3D;
				uint64_t mask = (sync ? 0x3Full << 56 : image ? ~0xFC000000ull : ~0ull) & ~(0xC0ull << 56);
				if ((a ^ b) & mask) {
					fprintf(stderr, "HLESEQ task %llu microcode #%u %016llx (after %016llx) high level #%u %016llx (after %016llx)\n", (unsigned long long) task,
					        i, (unsigned long long) a, (unsigned long long) (i ? n->checkSeq[0][i - 1] : 0), j, (unsigned long long) b, (unsigned long long) (j ? n->checkSeq[1][j - 1] : 0));
					break;
				}
				++i, ++j;
			}
			memset(n->checkCounts, 0, sizeof(n->checkCounts));
			n->checkLen[0] = n->checkLen[1] = 0;
			hleGraphics(n, true);
			++task;
		}
		// A graphics task run at high level: the RSP shows it running until hleDoneAt.
		bool graphics = be32(&n->sp.mem[0xFC0]) == 1;
		if (n->accuracy == ACCURACY_HLE && hleGraphics(n, false)) {
			n64Event(n, n->sp.hleDoneAt);
		} else if (n->rdpGl && graphics) {
			++n->glRspTasks;
		}
		if (n->rdpGl && graphics && ++n->glTasks == 64) {
			n->glrStop |= n->glRspTasks * 2 > n->glTasks;
			n->glTasks = n->glRspTasks = 0;
		}
		*s &= ~1u;
		// N64TASKDUMP=number,file: the RSP's memories, registers and RDRAM as that task starts.
		static const char* taskDump;
		static bool looked;
		if (!looked) {
			taskDump = getenv("N64TASKDUMP");
			looked = true;
		}
		if (taskDump && n->stats.rspTasks == strtoull(taskDump, NULL, 10)) {
			FILE* f = fopen(strchr(taskDump, ',') + 1, "wb");
			fwrite(n->sp.mem, 1, sizeof(n->sp.mem), f);
			fwrite(&n->sp, 1, sizeof(n->sp), f);
			fwrite(&n->rsp, 1, sizeof(n->rsp), f);
			fwrite(n->rdram, 1, RDRAM_SIZE, f);
			fclose(f);
		}
	}
	if (v & 2) {
		*s |= 1;
	}
	if (v & 4) {
		*s &= ~2u;
	}
	if (v & 8) {
		n64Interrupt(n, MI_SP, false);
	}
	if (v & 16) {
		n64Interrupt(n, MI_SP, true);
	}
	if (v & 32) {
		*s &= ~0x20u;
	}
	if (v & 64) {
		*s |= 0x20;
	}
	if (v & 128) {
		*s &= ~0x40u;
	}
	if (v & 256) {
		*s |= 0x40;
	}
	for (int sig = 0; sig < 8; ++sig) {
		if (v & (1u << (9 + sig * 2))) {
			*s &= ~(0x80u << sig);
		}
		if (v & (1u << (10 + sig * 2))) {
			*s |= 0x80u << sig;
		}
	}
}

// Playing time of a buffer: four bytes per stereo sample at the VI clock over (dacrate + 1).
static uint64_t aiDuration(struct N64* n, uint32_t len) {
	uint64_t rate = 48681812ull / (n->ai.dacrate + 1);
	return rate ? (uint64_t) (len / 4) * CPU_HZ / rate : 0;
}

// A buffer the game queued, kept for a frontend as it is now in RDRAM.
static void aiKeep(struct N64* n, uint32_t addr, uint32_t len) {
	n->audioRate = (uint32_t) (48681812ull / (n->ai.dacrate + 1));
	for (uint32_t i = 0; i < len && n->audioCount < n->audioCapacity; i += 2) {
		n->audio[n->audioCount++] = (int16_t) (n->rdram[(addr + i) & (RDRAM_SIZE - 1)] << 8 | n->rdram[(addr + i + 1) & (RDRAM_SIZE - 1)]);
	}
}

static void aiStartNext(struct N64* n) {
	n->ai.startedAt = n->cycles;
	n->ai.endsAt = n->cycles + aiDuration(n, n->ai.queueLen[0]);
	n64Event(n, n->ai.endsAt);
	n64Interrupt(n, MI_AI, true);
}

void aiUpdate(struct N64* n) {
	while (n->ai.queued && n->cycles >= n->ai.endsAt) {
		n->ai.queueLen[0] = n->ai.queueLen[1];
		if (--n->ai.queued) {
			aiStartNext(n);
		}
	}
}

static uint32_t aiRemaining(struct N64* n) {
	if (!n->ai.queued) {
		return 0;
	}
	uint64_t total = n->ai.endsAt - n->ai.startedAt;
	uint64_t left = n->ai.endsAt > n->cycles ? n->ai.endsAt - n->cycles : 0;
	return total ? (uint32_t) ((uint64_t) n->ai.queueLen[0] * left / total) & ~7u : 0;
}

static uint8_t cartByte(struct N64* n, uint32_t c) {
	if (c >= 0x10000000 && c - 0x10000000 < n->romSize) {
		return n->rom[c - 0x10000000];
	}
	if (c >= 0x08000000 && c < 0x08020000) {
		return n->save[c - 0x08000000];
	}
	if (c >= 0x05000000 && c < 0x06400000) {
		return ddReadByte(n, c);
	}
	return 0;
}

// Cartridge to RDRAM goes through the RCP's 128-byte buffer, a block at a time, never crossing a
// 2 KiB RDRAM page. Odd blocks fetch a byte more. An unaligned RDRAM address starts the first block
// part way into the buffer, which then writes back only up to the block length counted from the
// buffer's start; the address continues 8-byte aligned (n64brew, Peter Lemon's PI tests).
static void piDmaToRdram(struct N64* n, uint32_t len) {
	if (n->compat) {
		for (uint32_t i = 0; i < len; ++i) {
			n->rdram[(n->pi.dramAddr + i) & (RDRAM_SIZE - 1)] = cartByte(n, n->pi.cartAddr + i);
		}
		n->pi.dramAddr = (n->pi.dramAddr + len) & 0xFFFFFF;
		n->pi.cartAddr += len;
		return;
	}
	uint32_t dram = n->pi.dramAddr & 0xFFFFFE, cart = n->pi.cartAddr & ~1u;
	uint8_t buf[128];
	bool first = true;
	while (len > 0) {
		uint32_t mis = first ? dram & 7 : 0;
		uint32_t pageLeft = 0x800 - (dram & 0x7FF);
		uint32_t block = len < 128 - mis ? len : 128 - mis;
		block = block < pageLeft ? block : pageLeft;
		uint32_t fetched = (block + 1) & ~1u;
		for (uint32_t i = 0; i < fetched && mis + i < 128; ++i) {
			buf[mis + i] = cartByte(n, cart + i);
		}
		uint32_t base = dram & ~7u;
		if (first) {
			uint32_t end = mis == 0 && fetched == 128 ? 128 : block;
			for (uint32_t i = mis; i < end; ++i) {
				n->rdram[(base + i) & (RDRAM_SIZE - 1)] = buf[i];
			}
			dram = (dram + block + 7) & ~7u;
		} else {
			for (uint32_t i = 0; i < fetched; ++i) {
				n->rdram[(dram + i) & (RDRAM_SIZE - 1)] = buf[i];
			}
			dram += fetched;
		}
		cart += fetched;
		len -= block;
		first = false;
	}
	n->pi.dramAddr = dram & 0xFFFFFF;
	n->pi.cartAddr = cart;
}

// Throws away code a PI DMA of len bytes into RDRAM is about to cover, and waits for the RDP there.
static void piDmaCode(struct N64* n, uint32_t len) {
	n64RdramWrite(n, n->pi.dramAddr & 0xFFFFF8, len + 16);
}

static void piDma(struct N64* n, uint32_t len, bool toRdram) {
	if (toRdram) {
		piDmaCode(n, len);
		piDmaToRdram(n, len);
	} else {
		uint32_t dram = n->pi.dramAddr & 0xFFFFFE, cart = n->pi.cartAddr & ~1u;
		n64RdramRead(n, dram, len);
		for (uint32_t i = 0; i < len; ++i) {
			uint32_t c = cart + i;
			if (c >= 0x08000000 && c < 0x08020000) {
				n->save[c - 0x08000000] = n->rdram[(dram + i) & (RDRAM_SIZE - 1)];
			} else if (c >= 0x05000000 && c < 0x06000000) {
				ddWriteByte(n, c, n->rdram[(dram + i) & (RDRAM_SIZE - 1)]);
			}
		}
		n->pi.dramAddr = dram + len;
		n->pi.cartAddr = cart + len;
	}
	++n->stats.piDmas;
	n->stats.piBytes += len;
	// Domain 1 bus timing, in RCP cycles: latency per page, pulse width and release per halfword.
	uint32_t* t = n->pi.timing;
	uint32_t page = 1u << ((t[2] & 15) + 2);
	uint64_t rcp = (uint64_t) ((len + page - 1) / page) * (t[0] + 1) + (uint64_t) ((len + 1) / 2) * ((t[1] + 1) + (t[3] + 1));
	// Emulators that hacks were made on finish a DMA in a quarter cycle a byte (lle-compatible accuracy).
	n->pi.busyUntil = n->cycles + (n->compat ? len / 4 + 1 : rcp * 3 / 2);
	n64Event(n, n->pi.busyUntil);
	n->pi.status |= 1;
}

static void siDma(struct N64* n, bool toPif) {
	++n->stats.siDmas;
	uint32_t dram = n->si.dramAddr & (RDRAM_SIZE - 1) & ~3u;
	if (toPif) {
		n64RdramRead(n, dram, 64);
		memcpy(n->pifRam, &n->rdram[dram], 64);
		pifRun(n);
	} else {
		pifRead(n);
		n64RdramWrite(n, dram, 64);
		memcpy(&n->rdram[dram], n->pifRam, 64);
	}
	// The PIF takes a while to run the commands; the interrupt comes when the transfer is done.
	n->si.status |= 1;
	n->si.busyUntil = n->cycles + SI_DMA_CYCLES;
	n64Event(n, n->si.busyUntil);
}

static uint32_t viCurrent(struct N64* n) {
	uint32_t lines = (n->vi.reg[6] & 0x3FF) + 1;
	uint64_t into = n->cycles - (n->nextVi - n->fieldCycles);
	uint32_t line = (uint32_t) (into * lines / n->fieldCycles);
	uint32_t field = (n->fields & 1) && (n->vi.reg[0] & 0x40) ? 1 : 0;
	return (line % lines & ~1u) | field;
}

// SP memory and registers, DP and MI registers: what the RSP (and the RDP it drives) can change or see.
enum { RDRAM_INIT_CYCLES = 2000000 };

static inline bool rspShared(uint32_t addr) {
	return addr >= 0x04000000 && addr < 0x04400000;
}

uint32_t busRead32(struct N64* n, uint32_t addr) {
	addr &= 0x1FFFFFFF;
	if (addr >= 0x05000000 && addr < 0x06400000) {
		return ddRead32(n, addr);
	}
	if (addr < RDRAM_SIZE) {
		if (n->pageFlags[addr >> 12] & PAGE_RDP_WRITES) {
			rdpWaitRead(n, addr & ~3u, 4);
		}
		return be32(&n->rdram[addr & ~3u]);
	}
	if (rspShared(addr)) {
		rspCatchUp(n);
	}
	if (addr >= 0x04000000 && addr < 0x04040000) {
		return be32(&n->sp.mem[addr & 0x1FFC]);
	}
	if (addr >= 0x10000000 && addr < 0x1FC00000) {
		// The PI's address counter moves past the word the CPU reads.
		n->pi.cartAddr = (addr & ~3u) + 4;
		// A write to cartridge space stays on the bus while the PI writes it: a read then returns it.
		if (n->pi.romLatched && n->cycles < n->pi.romLatchUntil) {
			n->pi.romLatched = false;
			return n->pi.romLatch;
		}
		uint32_t off = (addr - 0x10000000) & ~3u;
		if (n->bootChecksumReads && (off == 0x10 || off == 0x14) && (n->bootChecksumReads & (off == 0x10 ? 1 : 2))) {
			n->bootChecksumReads &= off == 0x10 ? 2 : 1;
			return n->bootChecksum[off == 0x14];
		}
		return off + 4 <= n->romSize ? be32(&n->rom[off]) : 0;
	}
	if (addr >= 0x1FC007C0 && addr < 0x1FC00800) {
		return be32(&n->pifRam[addr & 0x3C]);
	}
	switch (addr >> 20) {
	case 0x040:
		if (addr == 0x04080000) {
			return n->sp.pc;
		}
		switch (addr & 0x1F) {
		case 0x00:
			return n->sp.memAddr;
		case 0x04:
			return n->sp.dramAddr;
		case 0x10:
			return n->sp.status;
		case 0x14:
		case 0x18:
			return 0;
		case 0x1C: {
			uint32_t v = n->sp.semaphore;
			n->sp.semaphore = 1;
			return v;
		}
		}
		return 0;
	case 0x041:
		switch (addr & 0x1F) {
		case 0x00:
			return n->dp.start;
		case 0x04:
			return n->dp.end;
		case 0x08:
			return n->dp.current;
		case 0x0C:
			return n->dp.status | 0x80;
		case 0x10:
		{
			// The RSP reads it at its own time (OoT's boot microcode uses it after the clear). It
			// stops at its largest value (Majora's Mask's boot microcode, read every task, never
			// sees a low 7 bits of zero).
			uint64_t t = (n->catchingUp ? n->rsp.clock : n->cycles * 2 / 3) - n->dp.clockBase;
			return t > 0xFFFFFF ? 0xFFFFFF : (uint32_t) t;
		}
		case 0x14:
			return n->dp.bufBusy > 0xFFFFFF ? 0xFFFFFF : n->dp.bufBusy;
		case 0x18:
			return n->dp.pipeBusy > 0xFFFFFF ? 0xFFFFFF : n->dp.pipeBusy;
		case 0x1C:
			return n->dp.tmemBusy > 0xFFFFFF ? 0xFFFFFF : n->dp.tmemBusy;
		}
		return 0;
	case 0x043:
		switch (addr & 0xF) {
		case 0x0:
			return n->miMode;
		case 0x4:
			return 0x02020102;
		case 0x8:
			return n->miIntr;
		case 0xC:
			return n->miMask;
		}
		return 0;
	case 0x044: {
		if ((addr & 0xFF) == 0x10) {
			return viCurrent(n);
		}
		uint32_t* r = reg(n->vi.reg, addr, 14);
		return r ? *r : 0;
	}
	case 0x045:
		aiUpdate(n);
		if ((addr & 0x1F) == 0x0C) {
			uint32_t full = n->ai.queued >= 2 ? 0x80000001u : 0;
			uint32_t busy = n->ai.queued ? 0x40000000u : 0;
			return full | busy | 0x01100000;
		}
		return aiRemaining(n);
	case 0x046:
		switch (addr & 0x3F) {
		case 0x00:
			return n->pi.dramAddr;
		case 0x04:
			return n->pi.cartAddr;
		// The length registers read back 0x7F once a transfer is done (n64brew).
		case 0x08:
		case 0x0C:
			return 0x7F;
		case 0x10:
			return n->pi.status | (n->miIntr & MI_PI ? 8 : 0);
		default:
			if ((addr & 0x3F) >= 0x14 && (addr & 0x3F) < 0x34) {
				return n->pi.timing[((addr & 0x3F) - 0x14) >> 2];
			}
			return 0;
		}
	case 0x047: {
		uint32_t* r = reg(n->riReg, addr, 8);
		// The RI comes up initialised (pifBoot), so the boot code skips the RDRAM set-up it does
		// after a power-on: device numbering and current calibration of each chip. That takes
		// about 20 ms (estimated from the 6102 boot code); it's spent here, at the boot code's first
		// look at RI_SELECT. Games time their start from the Count the boot code zeroes just before
		// (BattleTanx needs the time).
		if ((addr & 0x1F) == 0x0C && n->coldBoot) {
			n->coldBoot = false;
			n->cpu.stall += RDRAM_INIT_CYCLES;
		}
		return r ? *r : 0;
	}
	case 0x048:
		switch (addr & 0x1F) {
		case 0x00:
			return n->si.dramAddr;
		case 0x18:
			return n->si.status;
		}
		return 0;
	}
	return 0;
}

void busWrite32(struct N64* n, uint32_t addr, uint32_t v) {
	addr &= 0x1FFFFFFF;
	if (addr >= 0x05000000 && addr < 0x06400000) {
		ddWrite32(n, addr, v);
		return;
	}
	if (rspShared(addr)) {
		rspCatchUp(n);
	}
	if (addr < RDRAM_SIZE) {
		if (n->pageFlags[addr >> 12]) {
			n64RdramWrite(n, addr, 4);
		}
		put32(&n->rdram[addr & ~3u], v);
		return;
	}
	if (addr >= 0x04000000 && addr < 0x04040000) {
		put32(&n->sp.mem[addr & 0x1FFC], v);
		n->rspCodeGen += (addr >> 12) & 1;
		return;
	}
	if (addr >= 0x1FC007C0 && addr < 0x1FC00800) {
		put32(&n->pifRam[addr & 0x3C], v);
		return;
	}
	switch (addr >> 20) {
	case 0x040:
		if (addr == 0x04080000) {
			n->sp.pc = v & 0xFFC;
			n->rsp.nextPc = (n->sp.pc + 4) & 0xFFC;
			return;
		}
		switch (addr & 0x1F) {
		case 0x00:
			n->sp.memAddr = v & 0x1FF8;
			break;
		case 0x04:
			n->sp.dramAddr = v & (n->compat ? 0xFFFFFF : 0xFFFFF8);
			break;
		case 0x08:
			spDma(n, v, false);
			break;
		case 0x0C:
			spDma(n, v, true);
			break;
		case 0x10:
			spWriteStatus(n, v);
			break;
		case 0x1C:
			n->sp.semaphore = 0;
			break;
		}
		return;
	case 0x041:
		switch (addr & 0x1F) {
		case 0x00:
			n->dp.start = n->dp.current = v & 0xFFFFF8;
			break;
		case 0x04:
			n->dp.end = v & 0xFFFFF8;
			rdpRun(n);
			break;
		case 0x0C:
			if (v & 1) {
				n->dp.status &= ~1u;
			}
			if (v & 2) {
				n->dp.status |= 1;
			}
			// While frozen the RDP takes no commands; it carries on with them when thawed.
			if (v & 8) {
				n->dp.status |= 2;
			}
			if (v & 4) {
				n->dp.status &= ~2u;
				rdpRun(n);
			}
			if (v & 0x40) {
				n->dp.tmemBusy = 0;
			}
			if (v & 0x80) {
				n->dp.pipeBusy = 0;
			}
			if (v & 0x100) {
				n->dp.bufBusy = 0;
			}
			if (v & 0x200) {
				n->dp.clockBase = n->catchingUp ? n->rsp.clock : n->cycles * 2 / 3;
			}
			break;
		}
		return;
	case 0x043:
		switch (addr & 0xF) {
		case 0x0:
			n->miMode = (n->miMode & ~0x7Fu) | (v & 0x7F);
			if (v & 0x80) {
				n->miMode &= ~0x80u;
			}
			if (v & 0x100) {
				n->miMode |= 0x80;
			}
			if (v & 0x800) {
				n64Interrupt(n, MI_DP, false);
			}
			break;
		case 0xC:
			for (int line = 0; line < 6; ++line) {
				if (v & (1u << (line * 2))) {
					n->miMask &= ~(1u << line);
				}
				if (v & (2u << (line * 2))) {
					n->miMask |= 1u << line;
				}
			}
			cpuCheckInterrupts(n);
			break;
		}
		return;
	case 0x044: {
		if ((addr & 0xFF) == 0x10) {
			n64Interrupt(n, MI_VI, false);
			return;
		}
		uint32_t* r = reg(n->vi.reg, addr, 14);
		if (r) {
			*r = v;
		}
		if ((addr & 0xFF) == 0x04 && n->skip.on && (v & 0xFFFFFF) != n->skip.shown[0]) {
			memmove(n->skip.shown + 1, n->skip.shown, sizeof(n->skip.shown[0]) * 3);
			n->skip.shown[0] = v & 0xFFFFFF;
		}
		return;
	}
	case 0x045:
		switch (addr & 0x1F) {
		case 0x00:
			n->ai.dramAddr = v & 0xFFFFF8;
			break;
		case 0x04:
			aiUpdate(n);
			n->ai.len = v & 0x3FFF8;
			if (n->ai.len && n->ai.queued < 2) {
				n->ai.queueLen[n->ai.queued++] = n->ai.len;
				++n->stats.aiBuffers;
				if (n->audioCapacity) {
					rdpWaitRead(n, n->ai.dramAddr, n->ai.len);
					aiKeep(n, n->ai.dramAddr, n->ai.len);
				}
				if (n->ai.queued == 1) {
					aiStartNext(n);
				}
			}
			break;
		case 0x08:
			n->ai.control = v & 1;
			break;
		case 0x0C:
			n64Interrupt(n, MI_AI, false);
			break;
		case 0x10:
			n->ai.dacrate = v & 0x3FFF;
			break;
		case 0x14:
			n->ai.bitrate = v & 0xF;
			break;
		}
		return;
	case 0x046:
		switch (addr & 0x3F) {
		case 0x00:
			n->pi.dramAddr = v & (n->compat ? 0xFFFFFF : 0xFFFFFE);
			break;
		case 0x04:
			n->pi.cartAddr = n->compat ? v : v & ~1u;
			break;
		case 0x08:
			piDma(n, (v & 0xFFFFFF) + 1, false);
			break;
		case 0x0C:
			piDma(n, (v & 0xFFFFFF) + 1, true);
			break;
		case 0x10:
			if (v & 2) {
				n64Interrupt(n, MI_PI, false);
			}
			if (v & 1) {
				n->pi.status = 0;
				n->pi.busyUntil = 0;
			}
			break;
		default:
			if ((addr & 0x3F) >= 0x14 && (addr & 0x3F) < 0x34) {
				n->pi.timing[((addr & 0x3F) - 0x14) >> 2] = v & 0xFF;
			}
		}
		return;
	case 0x047: {
		uint32_t* r = reg(n->riReg, addr, 8);
		if (r) {
			*r = v;
		}
		return;
	}
	case 0x048:
		switch (addr & 0x1F) {
		case 0x00:
			n->si.dramAddr = v & 0xFFFFFF;
			break;
		case 0x04:
			siDma(n, false);
			break;
		case 0x10:
			siDma(n, true);
			break;
		case 0x18:
			n->si.status &= ~0x1000u;
			n64Interrupt(n, MI_SI, false);
			break;
		}
		return;
	}
	if (addr >= 0x08000000 && addr < 0x08020000) {
		put32(&n->save[addr & 0x1FFFC], v);
	} else if (addr >= 0x10000000 && addr < 0x1FC00000) {
		n->pi.romLatch = v;
		n->pi.romLatched = true;
		// Domain 1 timing for one word: latency, then two halfwords of pulse and release.
		const uint32_t* t = n->pi.timing;
		n->pi.romLatchUntil = n->cycles + ((t[0] + 1) + 2 * ((t[1] + 1) + (t[3] + 1))) * 3 / 2;
	}
}

// Sub-word reads come from the word that holds them; RDRAM and SP memory take sub-word writes. On
// the cartridge bus the PI answers with the upper half of the word at (address + 2) & ~3, and a byte
// read takes a byte of that half (Peter Lemon's PI test, hardware).
static bool cartSpace(uint32_t addr) {
	addr &= 0x1FFFFFFF;
	return addr >= 0x10000000 && addr < 0x1FC00000;
}

uint8_t busRead8(struct N64* n, uint32_t addr) {
	if (cartSpace(addr)) {
		uint16_t h = busRead32(n, (addr + 2) & ~3u) >> 16;
		return addr & 1 ? h & 0xFF : h >> 8;
	}
	return busRead32(n, addr & ~3u) >> (24 - (addr & 3) * 8);
}

uint16_t busRead16(struct N64* n, uint32_t addr) {
	if (cartSpace(addr)) {
		return busRead32(n, (addr + 2) & ~3u) >> 16;
	}
	return busRead32(n, addr & ~3u) >> (16 - (addr & 2) * 8);
}

uint64_t busRead64(struct N64* n, uint32_t addr) {
	return (uint64_t) busRead32(n, addr) << 32 | busRead32(n, addr + 4);
}

static uint8_t* writable(struct N64* n, uint32_t addr) {
	addr &= 0x1FFFFFFF;
	if (addr < RDRAM_SIZE) {
		if (n->pageFlags[addr >> 12]) {
			n64RdramWrite(n, addr, 1);
		}
		return &n->rdram[addr];
	}
	if (addr >= 0x04000000 && addr < 0x04040000) {
		rspCatchUp(n);
		n->rspCodeGen += (addr >> 12) & 1;
		return &n->sp.mem[addr & 0x1FFF];
	}
	if (addr >= 0x1FC007C0 && addr < 0x1FC00800) {
		return &n->pifRam[addr & 0x3F];
	}
	return NULL;
}

void busWrite8(struct N64* n, uint32_t addr, uint8_t v) {
	uint8_t* p = writable(n, addr);
	if (p) {
		*p = v;
	} else {
		busWrite32(n, addr & ~3u, (uint32_t) v << (24 - (addr & 3) * 8));
	}
}

void busWrite16(struct N64* n, uint32_t addr, uint16_t v) {
	uint8_t* p = writable(n, addr & ~1u);
	if (p) {
		p[0] = v >> 8;
		p[1] = v;
	} else {
		busWrite32(n, addr & ~3u, (uint32_t) v << (16 - (addr & 2) * 8));
	}
}

void busWrite64(struct N64* n, uint32_t addr, uint64_t v) {
	busWrite32(n, addr, v >> 32);
	busWrite32(n, addr + 4, (uint32_t) v);
}
