#include "n64.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

void cheatsClear(struct Cheats* c) {
	for (size_t i = 0; i < c->count; ++i) {
		free(c->list[i].words);
	}
	free(c->list);
	c->list = NULL;
	c->count = 0;
}

// A GameShark code: address and value words, 12 hex digits each with anything between them (spaces,
// "+" between codes); one that doesn't come out whole is left off.
void cheatsSet(struct Cheats* c, unsigned index, bool enabled, const char* code) {
	if (index >= c->count) {
		struct Cheat* more = realloc(c->list, (index + 1) * sizeof(*more));
		if (!more) {
			return;
		}
		memset(more + c->count, 0, (index + 1 - c->count) * sizeof(*more));
		c->list = more, c->count = index + 1;
	}
	struct Cheat* t = &c->list[index];
	free(t->words);
	t->words = NULL, t->count = 0;
	size_t digits = 0;
	for (const char* p = code; p && *p; ++p) {
		digits += isxdigit((unsigned char) *p) != 0;
	}
	if (!enabled || !digits || digits % 12 || !(t->words = malloc(digits / 12 * 2 * sizeof(uint32_t)))) {
		return;
	}
	uint64_t value = 0;
	size_t got = 0;
	for (const char* p = code; *p; ++p) {
		if (!isxdigit((unsigned char) *p)) {
			continue;
		}
		value = value << 4 | (uint64_t) (isdigit((unsigned char) *p) ? *p - '0' : tolower((unsigned char) *p) - 'a' + 10);
		if (++got % 12 == 0) {
			t->words[t->count++] = (uint32_t) (value >> 16), t->words[t->count++] = (uint32_t) value & 0xFFFF;
			value = 0;
		}
	}
}

// One code's write, only when memory doesn't hold its value already (a code that patches the game's
// code doesn't throw its compiled blocks away every frame).
static void cheatWrite(struct N64* n, uint32_t address, uint32_t value) {
	int const size = (address >> 24) & 1 ? 2 : 1;
	uint32_t const vaddr = 0x80000000u | (address & 0xFFFFFF);
	value &= size == 2 ? 0xFFFF : 0xFF;
	if (n64Peek(n, vaddr, size) != value) {
		n64Poke(n, vaddr, size, value);
	}
}

// The codes on, before each field once the game shows pictures (the VI on; a code written during the
// boot breaks it): writes (80/81, A0/A1 uncached, 88/89 GS button, F0/F1 at boot, all kept up), D0-D3
// (equal or not, 8 or 16 bits) deciding the next code, and 50 repeating it.
void cheatsApply(struct N64* n, const struct Cheats* c) {
	if (!(n->vi.reg[0] & 3)) {
		return;
	}
	for (size_t i = 0; i < c->count; ++i) {
		const uint32_t* w = c->list[i].words;
		size_t const count = c->list[i].count;
		for (size_t k = 0; k + 1 < count; k += 2) {
			uint32_t const address = w[k], value = w[k + 1];
			int const type = (int) (address >> 24);
			if (type >= 0xD0 && type <= 0xD3) {
				int const size = type & 1 ? 2 : 1;
				uint32_t const now = n64Peek(n, 0x80000000u | (address & 0xFFFFFF), size);
				bool const equal = now == (value & (size == 2 ? 0xFFFF : 0xFF));
				if (equal != (type <= 0xD1)) {
					k += 2;
				}
			} else if (type == 0x50 && k + 3 < count) {
				uint32_t const times = (address >> 8) & 0xFF, step = address & 0xFF;
				for (uint32_t t = 0; t < times; ++t) {
					cheatWrite(n, w[k + 2] + t * step, w[k + 3] + t * (uint32_t) (int16_t) value);
				}
				k += 2;
			} else if (type == 0x80 || type == 0x81 || type == 0xA0 || type == 0xA1 || type == 0x88 || type == 0x89 || type == 0xF0 || type == 0xF1) {
				cheatWrite(n, address, value);
			}
		}
	}
}
