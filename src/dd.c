#include "n64.h"

#include <stdlib.h>
#include <string.h>

// The 64DD disk drive: its buffers and registers on the cartridge bus (0x05000000), the IPL ROM
// (0x06000000), disks in the SDK layout (blocks in LBA order, as the .ndd dumps are) and its two
// interrupts on the CPU's cartridge line. The drive's behaviour follows ares (ISC licence), the
// disk layout tables Project64's (GPL); delays are in CPU cycles.

enum { SECTORS = 85, SIZE_LBA = 0x10DC, BLOCK0 = 0x4D08, DISKID_LBA = 14, SYSTEM_LBAS = 24 };

// Each disk type's virtual zones: the LBA each ends at, and the physical zone it is (8-15 on head 1).
static const uint16_t vzoneLba[7][16] = {
	{ 0x0124, 0x0248, 0x035A, 0x047E, 0x05A2, 0x06B4, 0x07C6, 0x08D8, 0x09EA, 0x0AB6, 0x0B82, 0x0C94, 0x0DA6, 0x0EB8, 0x0FCA, 0x10DC },
	{ 0x0124, 0x0248, 0x035A, 0x046C, 0x057E, 0x06A2, 0x07C6, 0x08D8, 0x09EA, 0x0AFC, 0x0BC8, 0x0C94, 0x0DA6, 0x0EB8, 0x0FCA, 0x10DC },
	{ 0x0124, 0x0248, 0x035A, 0x046C, 0x057E, 0x0690, 0x07A2, 0x08C6, 0x09EA, 0x0AFC, 0x0C0E, 0x0CDA, 0x0DA6, 0x0EB8, 0x0FCA, 0x10DC },
	{ 0x0124, 0x0248, 0x035A, 0x046C, 0x057E, 0x0690, 0x07A2, 0x08B4, 0x09C6, 0x0AEA, 0x0C0E, 0x0D20, 0x0DEC, 0x0EB8, 0x0FCA, 0x10DC },
	{ 0x0124, 0x0248, 0x035A, 0x046C, 0x057E, 0x0690, 0x07A2, 0x08B4, 0x09C6, 0x0AD8, 0x0BEA, 0x0D0E, 0x0E32, 0x0EFE, 0x0FCA, 0x10DC },
	{ 0x0124, 0x0248, 0x035A, 0x046C, 0x057E, 0x0690, 0x07A2, 0x086E, 0x0980, 0x0A92, 0x0BA4, 0x0CB6, 0x0DC8, 0x0EEC, 0x1010, 0x10DC },
	{ 0x0124, 0x0248, 0x035A, 0x046C, 0x057E, 0x0690, 0x07A2, 0x086E, 0x093A, 0x0A4C, 0x0B5E, 0x0C70, 0x0D82, 0x0E94, 0x0FB8, 0x10DC },
};

static const uint8_t vzonePzone[7][16] = {
	{ 0x0, 0x1, 0x2, 0x9, 0x8, 0x3, 0x4, 0x5, 0x6, 0x7, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA },
	{ 0x0, 0x1, 0x2, 0x3, 0xA, 0x9, 0x8, 0x4, 0x5, 0x6, 0x7, 0xF, 0xE, 0xD, 0xC, 0xB },
	{ 0x0, 0x1, 0x2, 0x3, 0x4, 0xB, 0xA, 0x9, 0x8, 0x5, 0x6, 0x7, 0xF, 0xE, 0xD, 0xC },
	{ 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0xC, 0xB, 0xA, 0x9, 0x8, 0x6, 0x7, 0xF, 0xE, 0xD },
	{ 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0xF, 0xE },
	{ 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0xF },
	{ 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8 },
};

// Each physical zone's first cylinder (head 1's zones counting inwards from theirs), the outer
// cylinder head 1's defects count from, and each zone's sector size (zones 8-15 are head 1's 1-8).
static const uint16_t zoneCylinder[16] = {
	0x000, 0x09E, 0x13C, 0x1D1, 0x266, 0x2FB, 0x390, 0x425, 0x091, 0x12F, 0x1C4, 0x259, 0x2EE, 0x383, 0x418, 0x48A,
};
static const uint16_t outerCylinder[8] = { 0x000, 0x09E, 0x13C, 0x1D1, 0x266, 0x2FB, 0x390, 0x425 };
static const uint16_t zoneSectorSize[9] = { 232, 216, 208, 192, 176, 160, 144, 128, 112 };

enum {
	CMD_NOP = 0x00, CMD_READ_SEEK, CMD_WRITE_SEEK, CMD_RECALIBRATE, CMD_SLEEP, CMD_START, CMD_SET_STANDBY,
	CMD_SET_SLEEP, CMD_CLEAR_CHANGE, CMD_CLEAR_RESET, CMD_READ_VERSION, CMD_SET_DISK_TYPE, CMD_REQUEST_STATUS,
	CMD_STANDBY, CMD_INDEX_LOCK_RETRY, CMD_SET_RTC_YM, CMD_SET_RTC_DH, CMD_SET_RTC_MS, CMD_GET_RTC_YM,
	CMD_GET_RTC_DH, CMD_GET_RTC_MS, CMD_SET_LED, CMD_READ_PROGRAM_VERSION = 0x1B,
};

static int pzoneOf(int type, uint32_t lba, int* vzone) {
	int v = 0;
	while (v < 15 && lba >= vzoneLba[type][v]) {
		++v;
	}
	*vzone = v;
	return vzonePzone[type][v];
}

static uint32_t blockBytes(int pzone) {
	return zoneSectorSize[pzone > 7 ? pzone - 7 : pzone] * SECTORS;
}

// An LBA's place on the disk: track | head << 12 | block << 13, defect tracks skipped as the system
// data lists them.
static uint16_t lbaPhys(const struct DD* d, uint32_t lba) {
	const uint8_t* sys = d->disk + d->sysOffset;
	int vzone, pzone = pzoneOf(d->diskType, lba, &vzone);
	uint16_t block = (lba & 3) == 0 || (lba & 3) == 3 ? 0 : 1;
	uint16_t head = pzone > 7;
	uint16_t diskZone = pzone ? pzone - 7 : 0;
	uint32_t start = vzone ? vzoneLba[d->diskType][vzone - 1] : 0;
	uint16_t track = (uint16_t) ((lba - start) >> 1);
	uint16_t zoneStart = zoneCylinder[pzone];
	if (head) {
		track = (uint16_t) -track;
		zoneStart = outerCylinder[diskZone - 1];
	}
	track = (uint16_t) (track + zoneCylinder[pzone]);
	uint16_t defect = pzone ? sys[8 + pzone - 1] : 0;
	uint16_t defects = (uint16_t) (sys[8 + pzone] - defect);
	while (defects && sys[0x20 + defect] + zoneStart <= track) {
		++track;
		++defect;
		--defects;
	}
	return (uint16_t) (track | head << 12 | block << 13);
}

// A system data copy: every sector of the block alike, and its fixed fields as they should be.
static bool systemBlockGood(const struct DD* d, uint32_t block, uint32_t size) {
	const uint8_t* b = d->disk + block * BLOCK0;
	if ((block + 1) * BLOCK0 > d->diskSize) {
		return false;
	}
	for (int s = 1; s < SECTORS; ++s) {
		if (memcmp(b, b + s * size, size)) {
			return false;
		}
	}
	if (block >= 12) {
		return true;
	}
	if (b[4] != 0x10 || (b[5] & 0xF0) != 0x10 || be32(b + 0x18) != 0xFFFFFFFF) {
		return false;
	}
	uint8_t step = block & 2 ? 0x0A : 0x0C;
	for (int i = 0; i < 16; ++i) {
		if (b[8 + i] != (uint8_t) ((i + 1) * step)) {
			return false;
		}
	}
	return true;
}

static bool layout(struct DD* d) {
	static const uint32_t candidates[4] = { 9, 8, 1, 0 };
	bool found = false;
	for (int i = 0; i < 4; ++i) {
		if (systemBlockGood(d, candidates[i] + 2, 0xC0)) {
			d->sysOffset = (candidates[i] + 2) * BLOCK0;
			found = true;
		}
	}
	if (!found) {
		for (int i = 0; i < 4; ++i) {
			if (systemBlockGood(d, candidates[i], 0xE8)) {
				d->sysOffset = candidates[i] * BLOCK0;
				found = true;
			}
		}
	}
	if (!found) {
		return false;
	}
	d->idOffset = DISKID_LBA * BLOCK0;
	for (int i = 2; i >= 1; --i) {
		if (systemBlockGood(d, DISKID_LBA + i, 0xE8)) {
			d->idOffset = (DISKID_LBA + i) * BLOCK0;
		}
	}
	// The system and ID copies a drive reads: those whose sectors all agree (and, for system data,
	// that look like it); the rest of those blocks fail as on a disk. The IPL relies on that: LBA 12
	// reads on a development disk and must not on a retail one.
	uint32_t sysSize = d->sysOffset / BLOCK0 >= 2 && (d->sysOffset / BLOCK0) % 8 >= 2 ? 0xC0 : 0xE8;
	for (uint32_t lba = 0; lba < 16; ++lba) {
		d->readable[lba] = systemBlockGood(d, lba, lba < 12 ? sysSize : 0xE8);
	}
	d->diskType = d->disk[d->sysOffset + 5] & 0x0F;
	if (d->diskType > 6) {
		return false;
	}
	memset(d->physLba, 0xFF, sizeof(d->physLba));
	uint32_t offset = 0;
	for (uint32_t lba = 0; lba < SIZE_LBA; ++lba) {
		int vzone, pzone = pzoneOf(d->diskType, lba, &vzone);
		d->lbaOffset[lba] = offset;
		uint16_t phys = lbaPhys(d, lba);
		if (d->physLba[phys & 0x3FFF] == 0xFFFF) {
			d->physLba[phys & 0x3FFF] = (uint16_t) lba;
		}
		offset += blockBytes(pzone);
	}
	d->lbaOffset[SIZE_LBA] = offset;
	return offset <= d->diskSize;
}

static void ddRaise(struct N64* n, bool mecha, bool bm) {
	struct DD* d = &n->dd;
	d->irqMecha |= mecha;
	d->irqBm |= bm;
	cpuCheckInterrupts(n);
}

static void ddLower(struct N64* n, bool mecha, bool bm) {
	struct DD* d = &n->dd;
	d->irqMecha &= !mecha;
	d->irqBm &= !bm;
	cpuCheckInterrupts(n);
}

bool ddInterrupt(const struct N64* n) {
	return n->dd.irqMecha || n->dd.irqBm;
}

static void motorStop(struct DD* d) {
	d->headRetracted = true;
	d->spindleStopped = true;
}

static void motorActive(struct DD* d) {
	d->headRetracted = false;
	d->spindleStopped = false;
}

static void schedule(struct N64* n, uint64_t* at, uint64_t delay) {
	*at = n->cycles + delay;
	n64Event(n, *at);
}

static void power(struct N64* n) {
	struct DD* d = &n->dd;
	memset(d->c2s, 0, sizeof(d->c2s));
	memset(d->ds, 0, sizeof(d->ds));
	memset(d->ms, 0, sizeof(d->ms));
	d->data = d->currentTrack = 0;
	d->currentSector = d->sectorSizeBuf = d->sectorSize = d->sectorBlock = 0;
	d->requestUser = d->requestC2 = d->busy = d->writeProtect = d->mechaError = false;
	d->bmStart = d->bmReset = d->bmError = d->blockTransfer = d->readMode = false;
	d->c1Single = d->c1Double = false;
	d->errors = 0;
	d->seeking = false;
	d->resetState = true;
	d->diskChanged = true;
	d->diskPresent = d->disk != NULL;
	d->id = d->ipl && !memcmp(d->ipl + 0x3B, "NDXJ", 4) ? 4 : 3;
	motorStop(d);
	d->mechaAt = d->bmAt = 0;
	ddLower(n, true, true);
}

// Whether the head is over the ROM area of the disk type the IPL set, which can't be written.
static bool seekTrack(struct DD* d) {
	static const uint16_t bounds[7] = { 0x09E, 0x13C, 0x1D1, 0x266, 0x2FB, 0x390, 0x425 };
	uint16_t track = d->currentTrack & 0xFFF;
	int zone = 0;
	while (zone < 7 && track >= bounds[zone]) {
		++zone;
	}
	zone += (d->currentTrack >> 12) & 1;
	return zone <= d->setType + 2;
}

static uint64_t seekCycles(struct DD* d, uint16_t to) {
	uint64_t cycles = 993000 / 2;
	if (d->spindleStopped || d->headRetracted) {
		cycles += 200700000 / 2;
	}
	uint16_t from = d->currentTrack & 0xFFF, target = to & 0xFFF;
	return cycles + (uint64_t) 19300 / 2 * (from > target ? from - target : target - from);
}

// The date and time the drive keeps, ticking with the emulated clock from the start of 2000, BCD.
static uint8_t bcd(unsigned v) {
	return (uint8_t) ((v / 10) << 4 | (v % 10));
}

static void rtcNow(const struct N64* n, uint8_t out[6]) {
	uint64_t secs = n->cycles / CPU_HZ;
	unsigned second = secs % 60, minute = secs / 60 % 60, hour = secs / 3600 % 24;
	unsigned day = (unsigned) (secs / 86400), month = 1, year = 0;
	static const uint8_t days[12] = { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
	while (day >= days[month - 1]) {
		day -= days[month - 1];
		++month;
	}
	out[0] = bcd(year);
	out[1] = bcd(month);
	out[2] = bcd(day + 1);
	out[3] = bcd(hour);
	out[4] = bcd(minute);
	out[5] = bcd(second);
}

enum { ERR_SELF_DIAGNOSTIC = 1, ERR_SERVO = 2, ERR_INDEX_GAP = 4, ERR_TIMEOUT = 8, ERR_UNDEFINED = 16, ERR_PARAM = 32 };

static void command(struct N64* n, uint16_t cmd) {
	struct DD* d = &n->dd;
	d->errors = 0;
	d->mechaError = false;
	d->writeProtect = false;
	if (d->busy) {
		return;
	}
	d->busy = true;
	uint64_t delay = 8000 / 2;
	switch (cmd) {
	case CMD_NOP:
		break;
	case CMD_READ_SEEK:
	case CMD_WRITE_SEEK:
		if (!d->diskPresent) {
			d->errors |= ERR_SELF_DIAGNOSTIC;
		} else if ((d->data >> 12) > 1 || (d->data & 0xFFF) > 0x5D7) {
			d->errors |= ERR_PARAM;
		} else {
			delay = seekCycles(d, d->data);
			d->currentTrack = (d->data & 0x1FFF) | 0x6000;
			bool rom = seekTrack(d);
			if (cmd == CMD_WRITE_SEEK) {
				d->writeProtect = rom;
			}
			motorActive(d);
			d->seeking = true;
		}
		break;
	case CMD_RECALIBRATE:
	case CMD_START:
		if (!d->diskPresent) {
			d->errors |= ERR_SELF_DIAGNOSTIC;
		} else {
			delay = seekCycles(d, 0);
			d->currentTrack = 0x6000;
			motorActive(d);
			d->seeking = true;
		}
		break;
	case CMD_SLEEP:
		if (!d->headRetracted || !d->spindleStopped) {
			delay = 83000000 / 2;
		}
		motorStop(d);
		break;
	case CMD_CLEAR_CHANGE:
		d->diskChanged = false;
		break;
	case CMD_CLEAR_RESET:
		d->diskChanged = false;
		d->resetState = false;
		break;
	case CMD_READ_VERSION:
		d->data = d->data & 1 ? 0x5300 : 0x0114;
		break;
	case CMD_SET_DISK_TYPE:
		d->setType = d->data & 15;
		break;
	case CMD_REQUEST_STATUS:
		d->data = (uint16_t) ((d->data & ~0x7Fu) | d->errors);
		break;
	case CMD_STANDBY:
		if (!d->diskPresent) {
			d->errors |= ERR_SELF_DIAGNOSTIC;
		} else {
			if (!d->headRetracted || !d->spindleStopped) {
				delay = 64000000 / 2;
			}
			d->headRetracted = true;
			d->spindleStopped = false;
		}
		break;
	case CMD_INDEX_LOCK_RETRY:
		if (!d->diskPresent) {
			d->errors |= ERR_SELF_DIAGNOSTIC;
		}
		break;
	case CMD_SET_RTC_YM:
	case CMD_SET_RTC_DH:
	case CMD_SET_RTC_MS:
	case CMD_SET_STANDBY:
	case CMD_SET_SLEEP:
	case CMD_SET_LED:
		break;
	case CMD_GET_RTC_YM:
	case CMD_GET_RTC_DH:
	case CMD_GET_RTC_MS: {
		uint8_t t[6];
		rtcNow(n, t);
		int i = (cmd - CMD_GET_RTC_YM) * 2;
		d->data = (uint16_t) (t[i] << 8 | t[i + 1]);
		break;
	}
	case CMD_READ_PROGRAM_VERSION:
		d->data = 0x0003;
		break;
	default:
		d->errors |= ERR_UNDEFINED;
	}
	if (d->errors || d->writeProtect) {
		d->mechaError = true;
	}
	schedule(n, &d->mechaAt, delay);
}

void ddMechaResponse(struct N64* n) {
	struct DD* d = &n->dd;
	d->mechaAt = 0;
	if (d->seeking) {
		d->seeking = false;
		if (d->diskPresent) {
			motorActive(d);
		} else {
			motorStop(d);
		}
	}
	d->busy = false;
	ddRaise(n, true, false);
}

// Where a sector of the block under the head is in the image, or -1 for a block the disk can't read
// (the system and ID copies other than the good ones, and anything off the layout).
static int64_t sectorOffset(const struct DD* d, uint16_t head, uint16_t track, uint16_t block, uint32_t sector, uint32_t size) {
	uint16_t lba = d->physLba[track | head << 12 | block << 13];
	if (lba == 0xFFFF) {
		return -1;
	}
	if (lba < 16 && !d->readable[lba]) {
		return -1;
	}
	uint64_t at = (uint64_t) d->lbaOffset[lba] + (uint64_t) sector * size;
	return at + size <= d->diskSize ? (int64_t) at : -1;
}

void ddBmRequest(struct N64* n) {
	struct DD* d = &n->dd;
	d->bmAt = 0;
	if (!d->bmStart) {
		ddLower(n, false, true);
		return;
	}
	d->requestUser = d->requestC2 = false;
	d->bmError = false;
	d->c1Single = d->c1Double = false;
	uint16_t block = d->currentSector >= 0x5A ? 1 : 0;
	uint32_t sector = d->currentSector % 0x5A;
	uint16_t track = d->currentTrack & 0xFFF, head = (d->currentTrack >> 12) & 1;
	uint32_t size = d->sectorSizeBuf + 1u;
	if (d->readMode) {
		if (sector < 0x55) {
			int64_t at = sectorOffset(d, head, track, block, sector, size);
			if (at < 0) {
				d->c1Single = d->c1Double = true;
				memset(d->ds, 0, size > sizeof(d->ds) ? sizeof(d->ds) : size);
			} else {
				memcpy(d->ds, d->disk + at, size > sizeof(d->ds) ? sizeof(d->ds) : size);
			}
			d->requestUser = true;
		} else if (sector == 0x58) {
			d->requestC2 = true;
			if (d->blockTransfer) {
				d->blockTransfer = false;
				d->currentSector = block ? 0 : 0x5A;
				ddRaise(n, false, true);
				return;
			}
			d->bmStart = false;
		}
		++d->currentSector;
	} else {
		if (sector < 0x55) {
			if (sector) {
				int64_t at = sectorOffset(d, head, track, block, sector - 1, size);
				if (at >= 0) {
					memcpy(d->disk + at, d->ds, size > sizeof(d->ds) ? sizeof(d->ds) : size);
				}
			}
			d->requestUser = true;
			++d->currentSector;
		} else if (d->blockTransfer) {
			int64_t at = sectorOffset(d, head, track, block, 0x54, size);
			if (at >= 0) {
				memcpy(d->disk + at, d->ds, size > sizeof(d->ds) ? sizeof(d->ds) : size);
			}
			d->blockTransfer = false;
			d->currentSector = block ? 0 : 0x5A;
			d->requestUser = true;
		} else {
			int64_t at = sectorOffset(d, head, track, block, 0x54, size);
			if (at >= 0) {
				memcpy(d->disk + at, d->ds, size > sizeof(d->ds) ? sizeof(d->ds) : size);
			}
			d->requestUser = false;
			d->bmStart = false;
		}
	}
	ddRaise(n, false, true);
}

static uint16_t readHalf(struct N64* n, unsigned index) {
	struct DD* d = &n->dd;
	switch (index) {
	case 0:
		return d->data;
	case 4: {
		uint16_t s = (uint16_t) (d->diskChanged | d->mechaError << 1 | d->writeProtect << 2 | d->headRetracted << 3 |
		                         d->spindleStopped << 4 | d->resetState << 6 | d->busy << 7 | d->diskPresent << 8 |
		                         d->irqMecha << 9 | d->irqBm << 10 | d->bmError << 11 | d->requestC2 << 12 |
		                         d->requestUser << 14);
		// Reading the status acknowledges a sector: the next comes a while later.
		if (d->irqBm) {
			schedule(n, &d->bmAt, 19000 + (d->currentTrack & 0xFFF) / 30);
			ddLower(n, false, true);
		}
		return s;
	}
	case 6:
		return d->currentTrack;
	case 8:
		return (uint16_t) (d->c1Single << 5 | d->c1Double << 6 | d->blockTransfer << 8 | d->bmError << 10 | d->bmStart << 15);
	case 10:
		return (uint16_t) (!d->diskPresent << 10);
	case 14:
		return (uint16_t) (d->currentSector << 8 | 0xC3);
	case 20:
		return d->sectorSizeBuf;
	case 24:
		return (uint16_t) (d->sectorSize | d->sectorBlock << 8);
	case 32:
		return d->id;
	}
	return 0;
}

static void writeHalf(struct N64* n, unsigned index, uint16_t v) {
	struct DD* d = &n->dd;
	switch (index) {
	case 0:
		d->data = v;
		break;
	case 4:
		command(n, v);
		break;
	case 8:
		d->bmReset |= (v >> 12) & 1;
		if (!(v & 0x1000) && d->bmReset) {
			d->bmStart = d->bmError = d->requestUser = d->requestC2 = false;
			d->bmReset = false;
			ddLower(n, false, true);
		}
		d->readMode = (v >> 14) & 1;
		d->blockTransfer = (v >> 9) & 1;
		if (v & 0x100) {
			ddLower(n, true, false);
		}
		d->currentSector = v & 0xFF;
		if ((v & 0x8000) && d->diskPresent) {
			d->bmStart = true;
			schedule(n, &d->bmAt, 25000 + (d->currentTrack & 0xFFF) / 30);
		}
		break;
	case 16:
		if (v == 0xAAAA) {
			power(n);
		}
		break;
	case 20:
		d->sectorSizeBuf = v & 0xFF;
		break;
	case 24:
		d->sectorSize = v & 0xFF;
		d->sectorBlock = v >> 8;
		break;
	}
}

bool ddPresent(const struct N64* n) {
	return n->dd.ipl != NULL;
}

// The drive's buffers and registers (0x05000000-0x050005BF) and the IPL ROM (0x06000000-0x063FFFFF),
// as the CPU and the PI see them. The registers are 16 bits, in a word's upper half.
uint32_t ddRead32(struct N64* n, uint32_t addr) {
	struct DD* d = &n->dd;
	if (!d->ipl) {
		return 0;
	}
	addr &= ~3u;
	if (addr >= 0x06000000) {
		return addr < 0x06400000 ? be32(&d->ipl[addr - 0x06000000]) : 0;
	}
	uint32_t off = addr - 0x05000000;
	if (off < 0x400) {
		return be32(&d->c2s[off]);
	}
	if (off < 0x500) {
		return be32(&d->ds[off - 0x400]);
	}
	if (off < 0x580) {
		return (uint32_t) readHalf(n, ((off - 0x500) & 0x7F) >> 1) << 16;
	}
	if (off < 0x5C0) {
		return be32(&d->ms[off - 0x580]);
	}
	return 0;
}

void ddWrite32(struct N64* n, uint32_t addr, uint32_t v) {
	struct DD* d = &n->dd;
	if (!d->ipl || addr >= 0x06000000) {
		return;
	}
	uint32_t off = (addr & ~3u) - 0x05000000;
	if (off < 0x400) {
		put32(&d->c2s[off], v);
	} else if (off < 0x500) {
		put32(&d->ds[off - 0x400], v);
	} else if (off < 0x580) {
		writeHalf(n, ((off - 0x500) & 0x7F) >> 1, (uint16_t) (v >> 16));
	} else if (off < 0x5C0) {
		put32(&d->ms[off - 0x580], v);
	}
}

uint8_t ddReadByte(struct N64* n, uint32_t addr) {
	return (uint8_t) (ddRead32(n, addr) >> (24 - (addr & 3) * 8));
}

// PI DMA into the drive's buffers.
void ddWriteByte(struct N64* n, uint32_t addr, uint8_t v) {
	struct DD* d = &n->dd;
	if (!d->ipl || addr < 0x05000000 || addr >= 0x050005C0) {
		return;
	}
	uint32_t off = addr - 0x05000000;
	if (off < 0x400) {
		d->c2s[off] = v;
	} else if (off < 0x500) {
		d->ds[off - 0x400] = v;
	} else if (off >= 0x580) {
		d->ms[off - 0x580] = v;
	}
}

static uint8_t* bigEndian(const uint8_t* data, size_t size) {
	uint8_t* out = malloc(size);
	if (!out) {
		return NULL;
	}
	uint32_t magic = be32(data);
	for (size_t i = 0; i + 3 < size; i += 4) {
		const uint8_t* s = data + i;
		uint8_t* o = out + i;
		if (magic == 0x40123780) {
			o[0] = s[3], o[1] = s[2], o[2] = s[1], o[3] = s[0];
		} else if (magic == 0x37804012) {
			o[0] = s[1], o[1] = s[0], o[2] = s[3], o[3] = s[2];
		} else {
			memcpy(o, s, 4);
		}
	}
	return out;
}

bool ddLoad(struct N64* n, const uint8_t* ipl, size_t iplSize, const uint8_t* disk, size_t diskSize) {
	struct DD* d = &n->dd;
	if (!ipl || iplSize != 0x400000) {
		return false;
	}
	free(d->ipl);
	free(d->disk);
	d->disk = NULL;
	d->ipl = bigEndian(ipl, iplSize);
	if (!d->ipl) {
		return false;
	}
	if (disk) {
		d->disk = malloc(diskSize);
		if (!d->disk) {
			return false;
		}
		memcpy(d->disk, disk, diskSize);
		d->diskSize = diskSize;
		if (!layout(d)) {
			free(d->disk);
			d->disk = NULL;
			return false;
		}
	}
	power(n);
	return true;
}

void ddFree(struct N64* n) {
	free(n->dd.ipl);
	free(n->dd.disk);
	n->dd.ipl = n->dd.disk = NULL;
}
