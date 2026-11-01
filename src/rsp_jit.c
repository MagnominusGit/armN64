#include "n64.h"

#include <stdlib.h>
#include <string.h>

// RSP blocks for a code generator (rspCompile): every version compiled for each IMEM address, so a
// microcode that comes back finds its code again.

enum { BLOCK_MAX = 64 };

struct RspBlock {
	struct RspBlock* next;
	void* code;
	uint32_t count;
	uint8_t ops[BLOCK_MAX * 4];
};

void rspJitFree(struct N64* n) {
	for (int i = 0; i < 1024; ++i) {
		struct RspBlock* b = n->rspVariants[i];
		while (b) {
			struct RspBlock* next = b->next;
			free(b);
			b = next;
		}
		n->rspVariants[i] = NULL;
		n->rspTable[i].code = NULL;
	}
	rspArenaFree(n);
}

// From pc to a branch's delay slot, the end of IMEM or BLOCK_MAX instructions.
static uint32_t blockLength(const uint8_t* imem, uint32_t pc) {
	uint32_t count = 0;
	bool delay = false;
	while (count < BLOCK_MAX && pc + count * 4 < 0x1000) {
		uint32_t op = be32(&imem[pc + count * 4]);
		// A branch in a delay slot is left to the interpreter.
		if (delay && rspIsBranch(op)) {
			break;
		}
		++count;
		if (delay) {
			break;
		}
		delay = rspIsBranch(op);
	}
	return count;
}

// Whether IMEM holds b's instructions at pc and the block there ends where b does: after a branch,
// only with a branch next (left to the interpreter) or at the end of a block or of IMEM.
static bool blockHeld(const struct RspBlock* b, const uint8_t* imem, uint32_t pc) {
	uint32_t const end = pc + b->count * 4;
	if (memcmp(b->ops, &imem[pc], b->count * 4)) {
		return false;
	}
	return !rspIsBranch(be32(&b->ops[(b->count - 1) * 4])) || b->count == BLOCK_MAX || end >= 0x1000 || rspIsBranch(be32(&imem[end]));
}

// The code for the block at pc: from the table, a block compiled before for the same instructions,
// or a new one. NULL when the arena is full (it starts again).
void* rspJitLookup(struct N64* n, uint32_t pc) {
	struct RspJitEntry* t = &n->rspTable[pc >> 2];
	if (t->gen == n->rspCodeGen && t->code) {
		return t->code;
	}
	const uint8_t* imem = &n->sp.mem[0x1000];
	for (struct RspBlock* b = n->rspVariants[pc >> 2]; b; b = b->next) {
		if (blockHeld(b, imem, pc)) {
			*t = (struct RspJitEntry) { n->rspCodeGen, b->code };
			return b->code;
		}
	}
	uint32_t count = blockLength(imem, pc);
	struct RspBlock* b = malloc(sizeof(*b));
	if (!b) {
		return NULL;
	}
	b->count = count;
	memcpy(b->ops, &imem[pc], count * 4);
	size_t before = n->rspUsed;
	if (!(b->code = rspCompile(n, b->ops, count, pc))) {
		free(b);
		rspJitFree(n);
		return NULL;
	}
	perfMapAdd(b->code, n->rspUsed - before, "rsp", pc);
	b->next = n->rspVariants[pc >> 2];
	n->rspVariants[pc >> 2] = b;
	*t = (struct RspJitEntry) { n->rspCodeGen, b->code };
	return b->code;
}

// A delay slot that does the same every time it runs: a no-op, a load or an immediate operation
// that writes neither what it reads nor reg.
static bool sameEveryTime(uint32_t op, int reg) {
	int const code = op >> 26, rs = (op >> 21) & 31, rt = (op >> 16) & 31;
	if (!op || (code == 0x0F && rt != reg)) {
		return true;
	}
	bool const load = code == 0x20 || code == 0x21 || code == 0x23 || code == 0x24 || code == 0x25;
	return (load || (code >= 0x08 && code <= 0x0E)) && rt != reg && rt != rs;
}

// A wait for the SP's DMA: a block that reads DMA_FULL or DMA_BUSY into a register and branches back
// to its start while that isn't 0, with the read in the delay slot, or before the branch with a delay
// slot that does the same every time. 0 when the block isn't one, else the register, 32 for
// DMA_FULL and 64 when the read comes first.
uint32_t rspDmaWait(const uint8_t* ops, uint32_t count) {
	if (count != 2 && count != 3) {
		return 0;
	}
	bool const first = count == 3;
	uint32_t const read = be32(&ops[first ? 0 : 4]), branch = be32(&ops[first ? 4 : 0]);
	int const reg = (read >> 16) & 31, flag = (read >> 11) & 31;
	if ((read & 0xFFE007FF) != 0x40000000 || (flag != 5 && flag != 6) || !reg) {
		return 0;
	}
	if (branch != (0x14000000u | (uint32_t) reg << 21 | (first ? 0xFFFE : 0xFFFF)) || (first && !sameEveryTime(be32(&ops[8]), reg))) {
		return 0;
	}
	return (uint32_t) reg | (flag == 5 ? 32 : 0) | (first ? 64 : 0);
}

// Every pass of a DMA wait (rspDmaWait) but the last that the RSP makes before the clock reaches
// limit, made at once: what it reads only changes when the DMA's time comes. A pass takes a cycle an
// instruction and two for the branch (rspPlainTiming); the read is a cycle into one that reads first,
// else at its end, the first pass going by what the register holds.
void rspDmaWaitSkip(struct N64* n, uint32_t wait, uint64_t limit) {
	struct RSP* r = &n->rsp;
	bool const first = wait & 64;
	uint64_t const until = wait & 32 ? n->sp.dmaFullUntil : n->sp.dmaUntil, clock = r->clock, len = first ? 4 : 3;
	uint64_t loops;
	if (first) {
		loops = until > clock + 1 ? (until - clock - 1 + len - 1) / len : 0;
	} else {
		loops = !r->r[wait & 31] ? 0 : until > clock ? (until - clock + len - 1) / len : 1;
	}
	uint64_t const under = limit > clock ? (limit - clock + len - 1) / len - 1 : 0, passes = loops < under ? loops : under;
	if (passes) {
		r->clock = clock + passes * len;
		r->r[wait & 31] = first || r->clock < until;
	}
}
