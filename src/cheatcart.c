#include "n64.h"

#include <stdlib.h>
#include <string.h>

// A Datel cheat cartridge (GameShark, Action Replay) with a game plugged in on top: its LZ9FC17 logic
// chip as RWeick's and Jhynjhiruu's reconstruction has it (REF1329-N64-Gameshark-Clone), and its two
// SST29LE010 flash chips, one the high and one the low byte of each halfword on the cartridge bus.
// The chip answers addresses whose top byte is its base (0x10 from power-on: the game's place, so the
// console boots its header and program and the game's own boot code), and passes the rest to the game.

enum { CHEAT_FLASH = 0x40000, CHIP_WORDS = 0x20000 };

bool cheatCartLoad(struct N64* n, const uint8_t* data, size_t size) {
	uint8_t* flash = malloc(CHEAT_FLASH);
	if (!flash || size < 0x1000) {
		free(flash);
		return false;
	}
	memset(flash, 0xFF, CHEAT_FLASH);
	uint32_t const magic = be32(data);
	for (size_t i = 0; i + 3 < size && i < CHEAT_FLASH; i += 4) {
		const uint8_t* s = data + i;
		uint8_t* d = flash + i;
		switch (magic) {
		case 0x37804012:
			d[0] = s[1], d[1] = s[0], d[2] = s[3], d[3] = s[2];
			break;
		case 0x40123780:
			d[0] = s[3], d[1] = s[2], d[2] = s[1], d[3] = s[0];
			break;
		default:
			memcpy(d, s, 4);
		}
	}
	free(n->cheat.flash);
	n->cheat.flash = flash;
	cheatCartReset(n);
	return true;
}

void cheatCartFree(struct N64* n) {
	free(n->cheat.flash);
	n->cheat.flash = NULL;
}

// Power on: the chip's registers clear, its base back at the game's place.
void cheatCartReset(struct N64* n) {
	struct CheatCart* c = &n->cheat;
	c->ioBase = 0x10;
	c->direction = 0, c->data = 0;
	c->blockA = c->blockC = false;
	memset(c->stage, 0, sizeof(c->stage));
	memset(c->idMode, 0, sizeof(c->idMode));
	memset(c->loading, 0, sizeof(c->loading));
	c->unprotected = false;
}

// The address lines a1-a10 the chip scrambles into flash lines 0-9 in some sections (the order each
// takes them in, and those it inverts); the lines above go straight through.
struct Scramble {
	uint8_t to[10];
	uint16_t invert;
};

static const struct Scramble section3[2] = {
	{ { 5, 9, 8, 3, 4, 0, 6, 7, 1, 2 }, 0 },
	{ { 0, 9, 8, 3, 4, 5, 6, 7, 1, 2 }, 1 << 2 | 1 << 9 },
};
static const struct Scramble section9[2] = {
	{ { 9, 8, 0, 3, 5, 1, 6, 7, 2, 4 }, 0 },
	{ { 9, 8, 0, 3, 5, 1, 6, 7, 4, 2 }, 0 },
};
static const struct Scramble sectionB = { { 5, 2, 0, 3, 8, 4, 6, 7, 1, 9 }, 1 << 4 | 1 << 8 | 1 << 9 };

static uint32_t scramble(const struct Scramble* s, uint32_t a) {
	uint32_t out = (a >> 1) & ~0x3FFu & 0x7FFFF;
	for (int i = 0; i < 10; ++i) {
		out |= (uint32_t) (((a >> (i + 1)) ^ (s->invert >> i)) & 1) << s->to[i];
	}
	return out;
}

// The flash chips' address (a halfword's, 17 lines wired) for a halfword address in the chip's window.
static uint32_t flashAddress(const struct CheatCart* c, uint32_t a) {
	unsigned const section = (a >> 20) & 15;
	uint32_t address;
	if (section == 0xE || section == 0xF) {
		address = (a >> 20 & 1) | ((a >> 2) & 0x3FFFF) << 1;
	} else if (c->ioBase & 0x40) {
		address = (a >> 1) & 0x7FFFF;
	} else if (section == 3) {
		address = scramble(&section3[(c->ioBase >> 5) & 1], a);
	} else if (section == 9) {
		address = scramble(&section9[(c->ioBase >> 5) & 1], a);
	} else if (section == 0xB) {
		address = scramble(&sectionB, a);
	} else {
		address = (a >> 1) & 0x7FFFF;
	}
	return address & (CHIP_WORDS - 1);
}

// Which part of its window an address is: the flash, the input register, the game's boot code, or
// zeros; anything else, nothing drives the bus.
enum { PART_NONE, PART_FLASH, PART_INPUT, PART_GAME, PART_ZERO };

static int part(uint32_t a) {
	unsigned const section = (a >> 20) & 15;
	uint32_t const rel = a & 0xFFFFFF;
	if (section == 0 && rel <= 0x3F) {
		return PART_FLASH;
	}
	if (section == 0 && rel <= 0xFFF) {
		return PART_GAME;
	}
	if (section == 0 && rel <= 0x1FFFF) {
		return PART_FLASH;
	}
	if (rel >= 0x20000 && rel <= 0x2FFFFF) {
		return PART_ZERO;
	}
	// In sections E and F the flash is off for a word's second halfword, but the bus keeps the first's
	// data, from the same flash address.
	if (section == 3 || section == 9 || section == 0xB || section == 0xC || section == 0xE || section == 0xF) {
		return PART_FLASH;
	}
	if (section == 4 && !((a >> 10) & 15)) {
		return PART_INPUT;
	}
	return PART_NONE;
}

// A page load ends when the CPU stops loading it (any other access to the chip).
static void pageLoadsEnd(struct CheatCart* c) {
	c->loading[0] = c->loading[1] = false;
}

static uint8_t flashByte(const struct CheatCart* c, int chip, uint32_t address) {
	if (c->idMode[chip]) {
		// SST's maker code, then the SST29LE010's.
		return address & 1 ? 0x08 : 0xBF;
	}
	return c->flash[address * 2 + (unsigned) chip];
}

// A halfword read from the cartridge bus at a: true with the cartridge's value, or false for the game
// on top's (at a in its own space).
bool cheatCartRead16(struct N64* n, uint32_t a, uint16_t* v) {
	struct CheatCart* c = &n->cheat;
	a &= ~1u;
	// A read ends the chips' page loads: the write cycle runs, done before the next access here.
	pageLoadsEnd(c);
	if ((a >> 24) != c->ioBase) {
		if (c->blockA || c->blockC) {
			*v = 0;
			return true;
		}
		return false;
	}
	switch (part(a)) {
	case PART_FLASH: {
		uint32_t const address = flashAddress(c, a);
		*v = (uint16_t) (flashByte(c, 0, address) << 8 | flashByte(c, 1, address));
		return true;
	}
	case PART_INPUT: {
		// The I/O lines driven as outputs read back what was written; the rest (the parallel port and the
		// display's microcontroller, with nothing attached) are pulled high. The button pulls its line low.
		uint16_t const outputs = c->direction & 0x27F;
		*v = (uint16_t) (0xF800 | (c->button ? 0 : 0x400) | (c->data & outputs) | (0x3FF & ~outputs));
		return true;
	}
	case PART_GAME:
		return false;
	case PART_ZERO:
		*v = 0;
		return true;
	}
	// Nothing drives the bus: the PI reads back the address's low half.
	*v = (uint16_t) a;
	return true;
}

// One chip's part of a flash write: the SST29LE010's commands at 5555 and 2AAA, and its page writes
// (a page of 128 bytes erased, then the bytes loaded into it written) once unlocked.
static void flashWrite(struct CheatCart* c, int chip, uint32_t address, uint8_t v) {
	uint32_t const command = address & 0x7FFF;
	uint8_t* stage = &c->stage[chip];
	if (c->loading[chip] || c->unprotected) {
		uint32_t const page = address & ~0x7Fu;
		if (!c->loading[chip] || c->page[chip] != page) {
			for (uint32_t i = 0; i < 0x80; ++i) {
				c->flash[(page + i) * 2 + (unsigned) chip] = 0xFF;
			}
			c->page[chip] = page;
			c->loading[chip] = true;
		}
		c->flash[address * 2 + (unsigned) chip] = v;
		c->dirty = true;
		return;
	}
	if (v == 0xF0 && *stage != 2 && *stage != 5) {
		c->idMode[chip] = false;
		*stage = 0;
		return;
	}
	switch (*stage) {
	case 0: case 3:
		*stage = command == 0x5555 && v == 0xAA ? *stage + 1 : 0;
		break;
	case 1: case 4:
		*stage = command == 0x2AAA && v == 0x55 ? *stage + 1 : 0;
		break;
	case 2:
		*stage = 0;
		if (command != 0x5555) {
			break;
		}
		if (v == 0xA0) {
			c->loading[chip] = true;
			c->page[chip] = UINT32_MAX;
		} else if (v == 0x80) {
			*stage = 3;
		} else if (v == 0x90) {
			c->idMode[chip] = true;
		} else if (v == 0xF0) {
			c->idMode[chip] = false;
		}
		break;
	case 5:
		*stage = 0;
		if (command != 0x5555) {
			break;
		}
		if (v == 0x10) {
			for (uint32_t i = 0; i < CHIP_WORDS; ++i) {
				c->flash[i * 2 + (unsigned) chip] = 0xFF;
			}
			c->dirty = true;
		} else if (v == 0x20) {
			c->unprotected = true;
		} else if (v == 0x60) {
			c->idMode[chip] = true;
		}
		break;
	}
}

// A halfword written to the cartridge bus at a: true when the cartridge took it (its registers or
// flash), false for the game on top.
bool cheatCartWrite16(struct N64* n, uint32_t a, uint16_t v) {
	struct CheatCart* c = &n->cheat;
	a &= ~1u;
	if ((a >> 24) != c->ioBase) {
		return false;
	}
	unsigned const section = (a >> 20) & 15;
	if (section == 4) {
		switch ((a >> 9) & 31) {
		case 2:
			c->ioBase = (uint8_t) v;
			break;
		case 3:
			c->direction = v & 0x3FF;
			break;
		case 4:
			c->data = v & 0x7FF;
			break;
		case 5:
			c->blockA = v != 0;
			break;
		case 6:
			c->blockC = (v & 0x775F) && !(v & 0x08A0);
			break;
		}
		return true;
	}
	if (part(a) == PART_FLASH && !((section == 0xE || section == 0xF) && (a & 2))) {
		uint32_t const address = flashAddress(c, a);
		if (c->loading[0] && (address & ~0x7Fu) != c->page[0] && c->page[0] != UINT32_MAX) {
			pageLoadsEnd(c);
		}
		flashWrite(c, 0, address, (uint8_t) (v >> 8));
		flashWrite(c, 1, address, (uint8_t) v);
	}
	return true;
}
