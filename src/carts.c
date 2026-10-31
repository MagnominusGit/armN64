// Cartridges whose save chip is a 16 Kbit EEPROM, by the two CRCs at 0x10 in the header (every other
// cartridge answers as a 4 Kbit EEPROM). The list is from mupen64plus's ROM database (GPLv2); the
// game IDs (header 0x3B) are those of the listed dumps, so revisions, betas and kiosk/LodgeNet
// versions of the same games answer the same.
#include "n64.h"

#include <string.h>

static const uint64_t eeprom16k[] = {
	0x975B7845A2505C18ull, // 77a Special Edition by Count0 (PD)
	0x514B6900B4B19881ull, // Banjo to Kazooie no Daibouken 2 (J) [!]
	0x155B7CDFF0DA7325ull, // Banjo-Tooie (A) [!]
	0xC9176D39EA4779D1ull, // Banjo-Tooie (E) (M4) [!]
	0xC2E9AA9A475D70AAull, // Banjo-Tooie (U) [!]
	0x373F58899A6CA80Aull, // Conker's Bad Fur Day (E) [!]
	0x30C7AC507704072Dull, // Conker's Bad Fur Day (U) [!]
	0x83F3931ECB72223Dull, // Cruis'n World (E) [!]
	0xDFE61153D76118E6ull, // Cruis'n World (U) [!]
	0x11936D8C6F2C4B43ull, // Donkey Kong 64 (E) [!]
	0x053C89A7A5064302ull, // Donkey Kong 64 (J) [!]
	0x0DD4ABABB5A2A91Eull, // Donkey Kong 64 (U) (Kiosk Demo) [!]
	0xEC58EABFAD7C7169ull, // Donkey Kong 64 (U) [!]
	0xB6306E99B63ED2B2ull, // Doraemon 2 - Nobita to Hikari no Shinden (J) [!]
	0xA8275140B9B056E8ull, // Doraemon 3 - Nobita no Machi SOS! (J) [!]
	0x202A8EE483F88B89ull, // Excitebike 64 (E) [!]
	0x861C3519F6091CE5ull, // Excitebike 64 (J) [!]
	0xAF754F7B1DD17381ull, // Excitebike 64 (U) (Kiosk Demo) [!]
	0x07861842A12EBC9Full, // Excitebike 64 (U) [!]
	0x1739EFBAD0B43A68ull, // Kobe Bryant in NBA Courtside (E) [!]
	0x616B84948A509210ull, // Kobe Bryant's NBA Courtside (U) [!]
	0xA197CB527520DE0Eull, // Madden Football 64 (E) [!]
	0x13836389265B3C76ull, // Madden Football 64 (U) [!]
	0xC56741600F5F453Cull, // Mario Party 3 (E) (M4) [!]
	0x0B0AB4CD7B158937ull, // Mario Party 3 (J) [!]
	0x7C3829D96E8247CEull, // Mario Party 3 (U) [!]
	0x839F3AD5406D15FAull, // Mario Tennis (E) [!]
	0x5001CF4FF30CB3BDull, // Mario Tennis (U) [!]
	0x3A6C42B51ACADA1Bull, // Mario Tennis 64 (J) [!]
	0x147E0EDB36C5B12Cull, // Neon Genesis Evangelion (J) [!]
	0xF468118CE32EE44Eull, // PD Ultraman Battle Collection 64 (J) [!]
	0xE4B08007A602FF33ull, // Perfect Dark (E) (M5) [!]
	0x96747EB4104BB243ull, // Perfect Dark (J) [!]
	0xDDF460CC3CA634C0ull, // Perfect Dark (U) (V1.0) [!]
	0x41F2B98FB458B466ull, // Perfect Dark (U) (V1.1) [!]
	0xFEE970104E94A9A0ull, // RR64 - Ridge Racer 64 (E) [!]
	0x2500267E2A7EC3CEull, // RR64 - Ridge Racer 64 (U) [!]
	0x53ED2DC406258002ull, // Star Wars Episode I - Racer (E) (M3) [!]
	0x61F5B152046122ABull, // Star Wars Episode I - Racer (J) [!]
	0x72F703986556A98Bull, // Star Wars Episode I - Racer (U) [!]
	0x2DCFCA608354B147ull, // Yoshi Story (J) [!]
	0xD3F97D496924135Bull, // Yoshi's Story (E) (M3) [!]
};

static const char gameIds16k[][4] = {
	"N3D", "N8M", "NB7", "NCW", "ND2", "NDO", "NDP", "NEP", "NEV", "NFU", "NM8",
	"NMV", "NMX", "NNB", "NNX", "NPD", "NRZ", "NUB", "NYS",
};

bool cartEeprom16k(const uint8_t* rom) {
	for (size_t i = 0; i < sizeof(gameIds16k) / sizeof(*gameIds16k); ++i) {
		if (!memcmp(rom + 0x3B, gameIds16k[i], 3)) {
			return true;
		}
	}
	uint64_t id = (uint64_t) be32(rom + 0x10) << 32 | be32(rom + 0x14);
	for (size_t i = 0; i < sizeof(eeprom16k) / sizeof(*eeprom16k); ++i) {
		if (eeprom16k[i] == id) {
			return true;
		}
	}
	return false;
}

// The boot code's checksum over the first megabyte after the header (as the boot code works it out
// for each lockout chip). False if the header has it already.
bool cartBootChecksum(const uint8_t* rom, size_t size, int cic, uint8_t seedByte, uint32_t out[2]) {
	if (size < 0x101000) {
		return false;
	}
	uint32_t seed = cic == 6103 ? 0xA3886759u : cic == 6105 ? 0xDF26F436u : cic == 6106 ? 0x1FEA617Au : seedByte * 0x5D588B65u + 1;
	uint32_t t1 = seed, t2 = seed, t3 = seed, t4 = seed, t5 = seed, t6 = seed;
	for (uint32_t i = 0x1000; i < 0x101000; i += 4) {
		uint32_t d = be32(rom + i);
		if (t6 + d < t6) {
			++t4;
		}
		t6 += d;
		t3 ^= d;
		uint32_t s = d & 31;
		uint32_t r = s ? d << s | d >> (32 - s) : d;
		t5 += r;
		t2 ^= t2 >= d ? r : t6 ^ d;
		t1 += cic == 6105 ? be32(rom + 0x40 + 0x710 + (i & 0xFF)) ^ d : t5 ^ d;
	}
	uint32_t crc1, crc2;
	if (cic == 6103) {
		crc1 = (t6 ^ t4) + t3, crc2 = (t5 ^ t2) + t1;
	} else if (cic == 6106) {
		crc1 = t6 * t4 + t3, crc2 = t5 * t2 + t1;
	} else {
		crc1 = t6 ^ t4 ^ t3, crc2 = t5 ^ t2 ^ t1;
	}
	out[0] = crc1;
	out[1] = crc2;
	return crc1 != be32(rom + 0x10) || crc2 != be32(rom + 0x14);
}

// The cartridge's SRAM (32 KB) or FlashRAM (128 KB, a Macronix MX29L1101: a new part as libultra
// tells them apart, pages of 128 bytes at their own addresses) at 0x08000000. Which one the game
// has shows in its first use: a flash command, or the SRAM reached first.
enum { FLASH_READ, FLASH_ID, FLASH_STATUS, FLASH_SECTOR_ERASE, FLASH_CHIP_ERASE, FLASH_PROGRAM };

static void flashCommand(struct N64* n, uint32_t v) {
	uint32_t const offset = (v & 0xFFFF) * 128;
	switch (v >> 24) {
	case 0x4B:
		n->flashMode = FLASH_SECTOR_ERASE;
		n->flashOffset = offset;
		break;
	case 0x3C:
		n->flashMode = FLASH_CHIP_ERASE;
		break;
	case 0x78:
		if (n->flashMode == FLASH_SECTOR_ERASE && n->flashOffset < sizeof(n->flash)) {
			uint32_t size = sizeof(n->flash) - n->flashOffset < 0x4000 ? sizeof(n->flash) - n->flashOffset : 0x4000;
			memset(&n->flash[n->flashOffset], 0xFF, size);
		} else if (n->flashMode == FLASH_CHIP_ERASE) {
			memset(n->flash, 0xFF, sizeof(n->flash));
		}
		n->flashStatus = 0x08;
		n->flashMode = FLASH_STATUS;
		n->saveDirty = true;
		break;
	case 0xA5:
		if (offset + 128 <= sizeof(n->flash)) {
			memcpy(&n->flash[offset], n->flashPage, 128);
		}
		n->flashStatus = 0x04;
		n->flashMode = FLASH_STATUS;
		n->saveDirty = true;
		break;
	case 0xB4:
		n->flashMode = FLASH_PROGRAM;
		break;
	case 0xD2:
		n->flashMode = FLASH_STATUS;
		break;
	case 0xE1:
		// Pokemon Puzzle League waits for status bit 0 after asking for the ID (mupen64plus).
		n->flashMode = FLASH_ID;
		n->flashStatus |= 1;
		break;
	case 0xF0:
		n->flashMode = FLASH_READ;
		break;
	}
}

// A byte the PI reads at offset in the save chip's space (a DMA to RDRAM).
uint8_t cartSaveByte(struct N64* n, uint32_t offset) {
	static const uint8_t flashId[8] = { 0x11, 0x11, 0x80, 0x01, 0x00, 0xC2, 0x00, 0x1D };
	if (n->saveKind == SAVE_UNKNOWN) {
		n->saveKind = SAVE_SRAM;
	}
	if (n->saveKind == SAVE_SRAM) {
		return offset < sizeof(n->sram) ? n->sram[offset] : 0;
	}
	switch (n->flashMode) {
	case FLASH_READ:
		return n->flash[offset & (sizeof(n->flash) - 1)];
	case FLASH_ID:
		return offset < 8 ? flashId[offset] : 0;
	}
	return 0;
}

// A byte the PI writes there (a DMA from RDRAM): to the SRAM, or the flash's page buffer.
void cartSaveStore(struct N64* n, uint32_t offset, uint8_t v) {
	if (n->saveKind == SAVE_UNKNOWN) {
		n->saveKind = SAVE_SRAM;
	}
	if (n->saveKind == SAVE_SRAM) {
		if (offset < sizeof(n->sram)) {
			n->sram[offset] = v;
			n->saveDirty = true;
		}
	} else if (n->flashMode == FLASH_PROGRAM && offset < sizeof(n->flashPage)) {
		n->flashPage[offset] = v;
	}
}

// A word the CPU reads there: the flash's status, or the SRAM.
uint32_t cartSaveRead32(struct N64* n, uint32_t offset) {
	if (n->saveKind == SAVE_FLASH) {
		return n->flashMode == FLASH_STATUS && offset == 0 ? n->flashStatus : 0;
	}
	return offset + 4 <= sizeof(n->sram) ? be32(&n->sram[offset]) : 0;
}

// A word the CPU writes there: a flash command (at 0x10000), its status, or the SRAM.
void cartSaveWrite32(struct N64* n, uint32_t offset, uint32_t v) {
	if (offset >= 0x10000) {
		n->saveKind = SAVE_FLASH;
		flashCommand(n, v);
	} else if (n->saveKind == SAVE_FLASH) {
		if (n->flashMode == FLASH_STATUS) {
			n->flashStatus = (uint8_t) v;
		}
	} else if (offset + 4 <= sizeof(n->sram)) {
		put32(&n->sram[offset], v);
		n->saveDirty = true;
	}
}

// A Controller Pak as libultra formats one: the ID block in block 1 of page 0 (and its copies in
// blocks 3, 4 and 6), the index table in page 1 (and its copy in page 2) with every page free, and
// no notes.
void pakFormat(uint8_t* pak) {
	memset(pak, 0, 0x8000);
	// Repaired, a random number, the serial number, the device ID (bit 0: a Controller Pak), one bank
	// and version 0, then the sums of the first 14 halfwords and of their complements.
	uint8_t id[32] = {
		0xFF, 0xFF, 0xFF, 0xFF, 0x05, 0x1A, 0x5F, 0x13, 0, 0, 0, 0, 0, 0, 0, 0,
		0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x01, 0x01, 0x00,
	};
	uint16_t sum = 0, isum = 0;
	for (int i = 0; i < 28; i += 2) {
		uint16_t h = (uint16_t) (id[i] << 8 | id[i + 1]);
		sum = (uint16_t) (sum + h);
		isum = (uint16_t) (isum + (uint16_t) ~h);
	}
	id[28] = (uint8_t) (sum >> 8), id[29] = (uint8_t) sum, id[30] = (uint8_t) (isum >> 8), id[31] = (uint8_t) isum;
	static const int blocks[4] = { 1, 3, 4, 6 };
	for (int i = 0; i < 4; ++i) {
		memcpy(&pak[blocks[i] * 32], id, 32);
	}
	// The index table: pages 0-4 the system's, the rest free (3), and byte 1 the sum of the bytes
	// from entry 5 on.
	uint8_t* table = &pak[0x100];
	uint8_t check = 0;
	for (int i = 5; i < 128; ++i) {
		table[2 * i + 1] = 3;
		check = (uint8_t) (check + 3);
	}
	table[1] = check;
	memcpy(&pak[0x200], table, 0x100);
}
