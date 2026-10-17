#include "n64.h"

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
};

// Leaves the machine as the PIF boot ROM does when it hands over to the cartridge's boot code.
void pifBoot(struct N64* n) {
	struct CPU* c = &n->cpu;
	uint32_t crc = n->romSize >= 0x1000 ? crc32(n->rom + 0x40, 0x1000 - 0x40) : 0;
	uint8_t seed = 0x3F;
	n->cic = 6102;
	for (size_t i = 0; i < sizeof(cics) / sizeof(*cics); ++i) {
		if (cics[i].crc == crc) {
			n->cic = cics[i].cic;
			seed = cics[i].seed;
		}
	}
	memcpy(n->sp.mem + 0x40, n->rom + 0x40, 0xFC0);
	// The PIF leaves a short routine in IMEM, which the 6105 boot code relies on.
	static const uint32_t imem[8] = {
		0x3C0DBFC0, 0x8DA807FC, 0x25AD07C0, 0x31080080, 0x5500FFFC, 0x3C0DBFC0, 0x8DA80024, 0x3C0BB000,
	};
	for (int i = 0; i < 8; ++i) {
		put32(&n->sp.mem[0x1000 + i * 4], imem[i]);
	}
	n->sp.status = 1;
	uint32_t dom1 = be32(n->rom);
	n->pi.timing[0] = dom1 & 0xFF;
	n->pi.timing[1] = (dom1 >> 8) & 0xFF;
	n->pi.timing[2] = (dom1 >> 16) & 0x0F;
	n->pi.timing[3] = (dom1 >> 20) & 0x03;
	n->vi.reg[3] = 1023;
	c->gpr[11] = 0xFFFFFFFFA4000040ull;
	c->gpr[19] = 0;
	// TV type from the cartridge's region: 0 PAL, 1 NTSC, 2 MPAL (Brazil).
	switch (n->rom[0x3E]) {
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
	c->cop0[CP0_STATUS] = 0x34000000;
	c->cop0[CP0_CONFIG] = 0x0006E463;
	n->riReg[0] = 0x0E;
	n->riReg[1] = 0x40;
	n->riReg[3] = 0x14;
	n->riReg[4] = 0x00063634;
	put32(&n->rdram[0x318], RDRAM_SIZE);
	put32(&n->rdram[0x3F0], RDRAM_SIZE);
}

static void controller(struct N64* n, const uint8_t* cmd, uint8_t* resp, int rx) {
	switch (cmd[0]) {
	case 0x00:
	case 0xFF:
		if (rx >= 3) {
			resp[0] = 0x05;
			resp[1] = 0x00;
			resp[2] = 0x02;
		}
		return;
	case 0x01:
		if (rx >= 4) {
			resp[0] = n->buttons >> 24;
			resp[1] = n->buttons >> 16;
			resp[2] = n->buttons >> 8;
			resp[3] = n->buttons;
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

// Runs the Joybus commands in PIF RAM: per channel, a send length, a receive length, the command
// and room for the reply. Controller 1 is plugged in and the cartridge's EEPROM answers on channel 4.
void pifRun(struct N64* n) {
	uint8_t* ram = n->pifRam;
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
		if (channel == 0) {
			controller(n, cmd, resp, receive);
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
