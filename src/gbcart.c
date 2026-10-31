#include "n64.h"

#include <stdlib.h>
#include <string.h>

// A Game Boy cartridge as a Transfer Pak reaches it: ROM, save RAM and the mapper (none, MBC1, MBC2,
// MBC3 or MBC5) from the header, as Pan Docs describes them. MBC3's clock registers keep what was
// last written; they don't run.

enum { MBC_NONE, MBC1, MBC2, MBC3, MBC5 };

bool gbCartLoad(struct GbCart* c, const uint8_t* rom, size_t size) {
	free(c->rom);
	memset(c, 0, sizeof(*c));
	if (size < 0x8000 || !(c->rom = malloc(size))) {
		return false;
	}
	memcpy(c->rom, rom, size);
	c->romSize = size;
	uint8_t type = rom[0x147];
	c->mbc = type >= 0x01 && type <= 0x03 ? MBC1 : type == 0x05 || type == 0x06 ? MBC2 : type >= 0x0F && type <= 0x13 ? MBC3 :
	         type >= 0x19 && type <= 0x1E ? MBC5 : MBC_NONE;
	static const uint32_t ramSizes[6] = { 0, 0x800, 0x2000, 0x8000, 0x20000, 0x10000 };
	c->ramSize = c->mbc == MBC2 ? 512 : rom[0x149] < 6 ? ramSizes[rom[0x149]] : 0;
	if (c->ramSize > sizeof(c->ram)) {
		c->ramSize = sizeof(c->ram);
	}
	c->romLow = 1;
	return true;
}

void gbCartFree(struct GbCart* c) {
	free(c->rom);
	c->rom = NULL;
	c->romSize = 0;
}

static uint32_t romBank(const struct GbCart* c) {
	switch (c->mbc) {
	case MBC1:
		return (uint32_t) (c->romHigh << 5 | (c->romLow & 0x1F ? c->romLow & 0x1F : 1));
	case MBC5:
		return c->romLow | (uint32_t) (c->romHigh & 1) << 8;
	default:
		return c->romLow ? c->romLow : 1;
	}
}

static uint32_t ramAddress(const struct GbCart* c, uint16_t addr) {
	uint32_t bank = c->mbc == MBC1 ? (c->mode ? c->romHigh : 0) : c->ramBank;
	return c->ramSize ? (bank * 0x2000 + (addr - 0xA000)) % c->ramSize : 0;
}

uint8_t gbCartRead(struct GbCart* c, uint16_t addr) {
	if (!c->rom) {
		return 0xFF;
	}
	if (addr < 0x4000) {
		return c->rom[addr];
	}
	if (addr < 0x8000) {
		return c->rom[(romBank(c) * 0x4000 + (addr - 0x4000)) % c->romSize];
	}
	if (addr >= 0xA000 && addr < 0xC000 && c->ramOn) {
		if (c->mbc == MBC3 && c->ramBank >= 0x08 && c->ramBank <= 0x0C) {
			return c->rtcLatched[c->ramBank - 0x08];
		}
		if (c->mbc == MBC2) {
			return c->ram[(addr - 0xA000) & 511] | 0xF0;
		}
		return c->ramSize ? c->ram[ramAddress(c, addr)] : 0xFF;
	}
	return 0xFF;
}

void gbCartWrite(struct GbCart* c, uint16_t addr, uint8_t v) {
	if (!c->rom) {
		return;
	}
	if (addr < 0x2000) {
		if (c->mbc == MBC2 && (addr & 0x100)) {
			c->romLow = v & 0x0F;
		} else {
			c->ramOn = (v & 0x0F) == 0x0A;
		}
	} else if (addr < 0x4000) {
		if (c->mbc == MBC2) {
			if (addr & 0x100) {
				c->romLow = v & 0x0F;
			} else {
				c->ramOn = (v & 0x0F) == 0x0A;
			}
		} else if (c->mbc == MBC5) {
			if (addr < 0x3000) {
				c->romLow = v;
			} else {
				c->romHigh = v & 1;
			}
		} else if (c->mbc == MBC3) {
			c->romLow = v & 0x7F;
		} else {
			c->romLow = v & 0x1F;
		}
	} else if (addr < 0x6000) {
		if (c->mbc == MBC1) {
			c->romHigh = v & 3;
		} else {
			c->ramBank = v & 0x0F;
		}
	} else if (addr < 0x8000) {
		if (c->mbc == MBC1) {
			c->mode = v & 1;
		} else if (c->mbc == MBC3) {
			if (c->latch == 0 && v == 1) {
				memcpy(c->rtcLatched, c->rtc, sizeof(c->rtc));
			}
			c->latch = v;
		}
	} else if (addr >= 0xA000 && addr < 0xC000 && c->ramOn) {
		if (c->mbc == MBC3 && c->ramBank >= 0x08 && c->ramBank <= 0x0C) {
			c->rtc[c->ramBank - 0x08] = v;
		} else if (c->mbc == MBC2) {
			c->ram[(addr - 0xA000) & 511] = v & 0x0F;
			c->ramDirty = true;
		} else if (c->ramSize) {
			c->ram[ramAddress(c, addr)] = v;
			c->ramDirty = true;
		}
	}
}
