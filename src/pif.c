#include "n64.h"

#include <stdlib.h>
#include <string.h>

static uint32_t crc32(const uint8_t* data, size_t size) {
	uint32_t crc = ~0u;
	for (size_t i = 0; i < size; ++i) {
		crc ^= data[i];
		for (int b = 0; b < 8; ++b) {
			crc = crc >> 1 ^ (0xEDB88320u & -(crc & 1));
		}
	}
	return ~crc;
}

// The lockout chip is identified by the boot code it pairs with; its seed goes to the boot code.
static const struct {
	uint32_t crc;
	int cic;
	uint8_t seed;
} cics[] = {
	{ 0x6170A4A1, 6101, 0x3F }, { 0x90BB6CB5, 6102, 0x3F }, { 0x0B050EE0, 6103, 0x78 },
	{ 0x98BC2C86, 6105, 0x91 }, { 0xACC8580A, 6106, 0x85 }, { 0x009E9EA3, 7102, 0x3F },
	// A 6102-style boot code that wants the 64DD's seed (Zoinkity's translation patches).
	{ 0x0E018159, 8303, 0xDD },
};

// Leaves the machine as the PIF boot ROM does when it hands over to the cartridge's boot code.
void pifBoot(struct N64* n) {
	struct CPU* c = &n->cpu;
	// Without a cartridge, the 64DD's IPL ROM boots, its lockout chip known by the ROM's ID.
	bool disk = n->romSize < 0x1000 && ddPresent(n);
	const uint8_t* boot = disk ? n->dd.ipl : n->rom;
	uint32_t crc = n->romSize >= 0x1000 ? crc32(n->rom + 0x40, 0x1000 - 0x40) : 0;
	uint8_t seed = 0x3F;
	n->cic = 6102;
	for (size_t i = 0; i < sizeof(cics) / sizeof(*cics); ++i) {
		if (cics[i].crc == crc) {
			n->cic = cics[i].cic;
			seed = cics[i].seed;
		}
	}
	if (disk) {
		bool us = !memcmp(boot + 0x3B, "NDDE", 4), dev = !memcmp(boot + 0x3B, "NDXJ", 4);
		n->cic = us ? 8501 : dev ? 8401 : 8303;
		seed = us ? 0xDE : 0xDD;
	}
	if (!boot) {
		return;
	}
	// Only for the standard boot codes: others check what they load their own way.
	bool standard = n->cic >= 6101 && n->cic <= 7102 && n->cic != 8303;
	n->bootChecksumReads = !disk && standard && cartBootChecksum(n->rom, n->romSize, n->cic, seed, n->bootChecksum) ? 3 : 0;
	memcpy(n->sp.mem + 0x40, boot + 0x40, 0xFC0);
	// The PIF leaves a short routine in IMEM, which the 6105 boot code relies on.
	static const uint32_t imem[8] = {
		0x3C0DBFC0, 0x8DA807FC, 0x25AD07C0, 0x31080080, 0x5500FFFC, 0x3C0DBFC0, 0x8DA80024, 0x3C0BB000,
	};
	for (int i = 0; i < 8; ++i) {
		put32(&n->sp.mem[0x1000 + i * 4], imem[i]);
	}
	n->sp.status = 1;
	uint32_t dom1 = be32(boot);
	n->pi.timing[0] = dom1 & 0xFF;
	n->pi.timing[1] = (dom1 >> 8) & 0xFF;
	n->pi.timing[2] = (dom1 >> 16) & 0x0F;
	n->pi.timing[3] = (dom1 >> 20) & 0x03;
	n->vi.reg[3] = 1023;
	c->gpr[11] = 0xFFFFFFFFA4000040ull;
	c->gpr[19] = disk;
	// TV type from the cartridge's region (N64_TV=ntsc, pal or mpal picks the console's instead,
	// for dumps whose header names another region): 0 PAL, 1 NTSC, 2 MPAL (Brazil).
	const char* tv = getenv("N64_TV");
	switch (tv && !strcmp(tv, "ntsc") ? 'E' : tv && !strcmp(tv, "pal") ? 'P' : tv && !strcmp(tv, "mpal") ? 'B' : boot[0x3E]) {
	case 'D': case 'F': case 'H': case 'I': case 'L': case 'P': case 'S': case 'U': case 'W': case 'X': case 'Y':
		c->gpr[20] = 0;
		break;
	case 'B':
		c->gpr[20] = 2;
		break;
	default:
		c->gpr[20] = 1;
	}
	c->gpr[21] = 0;
	c->gpr[22] = seed;
	c->gpr[23] = 0;
	c->gpr[29] = 0xFFFFFFFFA4001FF0ull;
	c->gpr[31] = 0xFFFFFFFFA4001550ull;
	c->pc = 0xFFFFFFFFA4000040ull;
	c->nextPc = c->pc + 4;
	// CP0 as the boot ROMs leave it on hardware (Peter Lemon's COP0Register test, cold boot).
	c->cop0[CP0_STATUS] = 0x241000E0;
	c->cop0[CP0_CONFIG] = 0x7006E463;
	c->cop0[CP0_PRID] = 0x00000B22;
	c->cop0[CP0_CONTEXT] = 0x007FFFF0;
	c->cop0[CP0_XCONTEXT] = 0xFFFFFFF0;
	c->cop0[CP0_BADVADDR] = 0xFFFFFFFFFFFFFFFFull;
	c->cop0[CP0_EPC] = 0xFFFFFFFFFFFFFFFFull;
	c->cop0[CP0_ERROREPC] = 0xFFFFFFFFFFFFFFFFull;
	c->cop0[CP0_LLADDR] = 0xFFFFFFFF;
	n->coldBoot = true;
	n->riReg[0] = 0x0E;
	n->riReg[1] = 0x40;
	n->riReg[3] = 0x14;
	n->riReg[4] = 0x00063634;
	put32(&n->rdram[0x318], RDRAM_SIZE);
	put32(&n->rdram[0x3F0], RDRAM_SIZE);
	// A dump without boot code (development cartridges were loaded another way): what the 6102's
	// leaves for the game, its first MiB at the entry point and the boot values at 0x300.
	bool blank = !disk && n->romSize > 0x1000;
	for (int i = 0x40; blank && i < 0x1000; ++i) {
		blank = !n->rom[i];
	}
	if (blank) {
		uint32_t entry = be32(n->rom + 8), size = n->romSize - 0x1000 < 0x100000 ? n->romSize - 0x1000 : 0x100000;
		for (uint32_t i = 0; i < size; ++i) {
			n->rdram[((entry & 0x1FFFFFFF) + i) & (RDRAM_SIZE - 1)] = n->rom[0x1000 + i];
		}
		const uint32_t values[6] = { (uint32_t) c->gpr[20], (uint32_t) c->gpr[19], 0xB0000000, (uint32_t) c->gpr[23], (uint32_t) c->gpr[22], (uint32_t) c->gpr[21] };
		for (int i = 0; i < 6; ++i) {
			put32(&n->rdram[0x300 + i * 4], values[i]);
		}
		n->bootChecksumReads = 0;
		c->pc = (uint64_t) (int64_t) (int32_t) entry;
		c->nextPc = c->pc + 4;
	}
}

// A controller, or an N64 Mouse (type 0x0002), which reads the same way.
static void controller(const uint8_t* cmd, uint8_t* resp, int rx, bool mouse, uint32_t state) {
	switch (cmd[0]) {
	case 0x00:
	case 0xFF:
		if (rx >= 3) {
			resp[0] = mouse ? 0x00 : 0x05;
			resp[1] = mouse ? 0x02 : 0x00;
			resp[2] = mouse ? 0x00 : 0x02;
		}
		return;
	case 0x01:
		if (rx >= 4) {
			resp[0] = state >> 24;
			resp[1] = state >> 16;
			resp[2] = state >> 8;
			resp[3] = state;
		}
		return;
	}
}

// The cartridge's EEPROM on channel 4: info, then 8-byte block reads and writes.
static void eeprom(struct N64* n, const uint8_t* cmd, uint8_t* resp, int send, int rx) {
	uint32_t size = n->eeprom16k ? 2048 : 512;
	switch (cmd[0]) {
	case 0x00:
	case 0xFF:
		if (rx >= 3) {
			resp[0] = 0x00;
			resp[1] = n->eeprom16k ? 0xC0 : 0x80;
			resp[2] = 0x00;
		}
		return;
	case 0x04:
		if (send >= 2 && rx >= 8) {
			memcpy(resp, &n->eeprom[(cmd[1] * 8u) % size], 8);
		}
		return;
	case 0x05:
		if (send >= 10) {
			memcpy(&n->eeprom[(cmd[1] * 8u) % size], cmd + 2, 8);
			if (rx >= 1) {
				resp[0] = 0x00;
			}
		}
		return;
	}
}

// Answers the CIC-NUS-6105 challenge: 30 nibbles at 0x30 in, 30 nibbles out (X-Scale's decoding of
// the algorithm, as used by mupen64plus).
static void cicChallenge(uint8_t* ram) {
	static const uint8_t lut0[16] = { 4, 7, 10, 7, 14, 5, 14, 1, 12, 15, 8, 15, 6, 3, 6, 9 };
	static const uint8_t lut1[16] = { 4, 1, 10, 7, 14, 5, 14, 1, 12, 9, 8, 5, 6, 3, 12, 9 };
	const uint8_t* lut = lut0;
	uint8_t key = 0xB;
	for (int i = 0; i < 30; ++i) {
		uint8_t* b = &ram[0x30 + i / 2];
		int shift = i & 1 ? 0 : 4;
		uint8_t out = (key + 5 * ((*b >> shift) & 15)) & 15;
		*b = (uint8_t) ((*b & ~(15 << shift)) | out << shift);
		key = lut[out];
		int sign = out >> 3;
		int magnitude = (sign ? ~out : out) & 7;
		int mod = magnitude % 3 == 1 ? sign : 1 - sign;
		if (lut == lut1 && (out == 1 || out == 9)) {
			mod = 1;
		} else if (lut == lut1 && (out == 11 || out == 14)) {
			mod = 0;
		}
		lut = mod ? lut1 : lut0;
	}
	ram[63] = 0;
}

// Runs the Joybus commands in PIF RAM: per channel, a send length, a receive length, the command
// and room for the reply. Controller 1 is plugged in and the cartridge's EEPROM answers on channel 4.
void pifRun(struct N64* n) {
	uint8_t* ram = n->pifRam;
	if (ram[63] == 2) {
		cicChallenge(ram);
		return;
	}
	if (!(ram[63] & 1)) {
		return;
	}
	int channel = 0;
	int i = 0;
	while (i < 62 && channel < 5) {
		uint8_t tx = ram[i];
		if (tx == 0xFE) {
			break;
		}
		if (tx == 0xFF || tx == 0xFD) {
			++i;
			continue;
		}
		if (tx == 0) {
			++channel;
			++i;
			continue;
		}
		int send = tx & 0x3F;
		int receive = ram[i + 1] & 0x3F;
		if (i + 2 + send + receive > 63) {
			break;
		}
		uint8_t* cmd = &ram[i + 2];
		uint8_t* resp = cmd + send;
		if (n->mousePort && channel == n->mousePort - 1) {
			controller(cmd, resp, receive, true, n->mouse);
		} else if (channel == 0) {
			controller(cmd, resp, receive, false, n->buttons);
		} else if (channel == 4) {
			eeprom(n, cmd, resp, send, receive);
		} else {
			ram[i + 1] |= 0x80;
		}
		i += 2 + send + receive;
		++channel;
	}
	ram[63] &= ~1;
}
