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

// The code for the block at pc: from the table, a block compiled before for the same instructions,
// or a new one. NULL when the arena is full (it starts again).
void* rspJitLookup(struct N64* n, uint32_t pc) {
	struct RspJitEntry* t = &n->rspTable[pc >> 2];
	if (t->gen == n->rspCodeGen && t->code) {
		return t->code;
	}
	const uint8_t* imem = &n->sp.mem[0x1000];
	uint32_t count = blockLength(imem, pc);
	for (struct RspBlock* b = n->rspVariants[pc >> 2]; b; b = b->next) {
		if (b->count == count && !memcmp(b->ops, &imem[pc], count * 4)) {
			*t = (struct RspJitEntry) { n->rspCodeGen, b->code };
			return b->code;
		}
	}
	struct RspBlock* b = malloc(sizeof(*b));
	if (!b) {
		return NULL;
	}
	b->count = count;
	memcpy(b->ops, &imem[pc], count * 4);
	if (!(b->code = rspCompile(n, b->ops, count, pc))) {
		free(b);
		rspJitFree(n);
		return NULL;
	}
	b->next = n->rspVariants[pc >> 2];
	n->rspVariants[pc >> 2] = b;
	*t = (struct RspJitEntry) { n->rspCodeGen, b->code };
	return b->code;
}
