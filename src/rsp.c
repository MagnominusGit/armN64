#include "n64.h"

#include <string.h>

#define RS ((op >> 21) & 31)
#define RT ((op >> 16) & 31)
#define RD ((op >> 11) & 31)
#define SA ((op >> 6) & 31)
#define IMM ((uint32_t) (int32_t) (int16_t) op)

static uint8_t* dmem(struct N64* n, uint32_t addr) {
	return &n->sp.mem[addr & 0xFFF];
}

static inline uint8_t* memAt(uint8_t* mem, uint32_t swap, uint32_t addr) {
	return &mem[(addr & 0xFFF) ^ swap];
}

// A vector register as its 16 big-endian bytes.
static uint8_t vbyte(struct RSP* r, int reg, int k) {
	uint16_t v = (uint16_t) r->vr[reg][(k >> 1) & 7];
	return k & 1 ? (uint8_t) v : (uint8_t) (v >> 8);
}

static void setVbyte(struct RSP* r, int reg, int k, uint8_t b) {
	uint16_t* v = (uint16_t*) &r->vr[reg][(k >> 1) & 7];
	*v = k & 1 ? (*v & 0xFF00) | b : (uint16_t) ((*v & 0x00FF) | b << 8);
}

static uint32_t rspCop0Address(int reg) {
	return reg < 8 ? 0x04040000 + reg * 4 : 0x04100000 + (reg - 8) * 4;
}

static void branch(struct RSP* r, bool taken, uint32_t target) {
	if (taken) {
		r->nextPc = target & 0xFFC;
	}
}

// The vector loads and stores on a DMEM image whose byte a is at a ^ swap: 0 for the RSP's own
// (big-endian), 3 for one of host words (armlet.c).
static inline __attribute__((always_inline)) void vectorLoadIn(struct RSP* r, uint8_t* mem, uint32_t swap, uint32_t op) {
	int vt = RT, e = (op >> 7) & 15, kind = RD;
	int32_t offset = (int32_t) ((op & 0x7F) << 25) >> 25;
	static const int scale[12] = { 1, 2, 4, 8, 16, 16, 8, 8, 16, 16, 16, 16 };
	if (kind > 11) {
		return;
	}
	uint32_t addr = (r->r[RS] + (uint32_t) (offset * scale[kind])) & 0xFFF;
	// Whole lanes from DMEM without wrapping: LSV, LLV, LDV and LQV at even elements and in-range
	// addresses, LQV from a 16-byte boundary.
	int size = scale[kind];
	if (!swap && kind >= 1 && kind <= 4 && !(e & 1) && e + size <= 16 && addr + size <= 0x1000 && (kind < 4 || !(addr & 15))) {
		const uint8_t* m = &mem[addr];
		for (int i = 0; i < size / 2; ++i) {
			r->vr[vt][e / 2 + i] = (int16_t) (m[2 * i] << 8 | m[2 * i + 1]);
		}
		return;
	}
	switch (kind) {
	case 0:
	case 1:
	case 2:
	case 3:
		for (int i = 0; i < scale[kind] && e + i < 16; ++i) {
			setVbyte(r, vt, e + i, *memAt(mem, swap, addr + i));
		}
		return;
	case 4: {
		uint32_t end = (addr & ~15u) + 16;
		for (int i = e; i < 16 && addr + (i - e) < end; ++i) {
			setVbyte(r, vt, i, *memAt(mem, swap, addr + (i - e)));
		}
		return;
	}
	case 5: {
		int count = addr & 15;
		uint32_t start = addr & ~15u;
		for (int k = 0; k < count; ++k) {
			int i = e + 16 - count + k;
			if (i < 16) {
				setVbyte(r, vt, i, *memAt(mem, swap, start + k));
			}
		}
		return;
	}
	case 6:
	case 7: {
		int index = (int) (addr & 7) - e;
		addr &= ~7u;
		for (int i = 0; i < 8; ++i) {
			uint16_t b = *memAt(mem, swap, addr + ((index + i) & 15));
			r->vr[vt][i] = (int16_t) (kind == 6 ? b << 8 : b << 7);
		}
		return;
	}
	case 8: {
		int index = (int) (addr & 7) - e;
		addr &= ~7u;
		for (int i = 0; i < 8; ++i) {
			r->vr[vt][i] = (int16_t) (*memAt(mem, swap, addr + ((index + i * 2) & 15)) << 7);
		}
		return;
	}
	case 9: {
		int index = (int) (addr & 7) - e;
		addr &= ~7u;
		int16_t tmp[8];
		for (int i = 0; i < 4; ++i) {
			tmp[i] = (int16_t) (*memAt(mem, swap, addr + ((index + i * 4) & 15)) << 7);
			tmp[i + 4] = (int16_t) (*memAt(mem, swap, addr + ((index + i * 4 + 8) & 15)) << 7);
		}
		for (int i = e; i < e + 8 && i < 16; ++i) {
			uint16_t v = (uint16_t) tmp[i >> 1];
			setVbyte(r, vt, i, i & 1 ? (uint8_t) v : (uint8_t) (v >> 8));
		}
		return;
	}
	// LWV loads nothing: the register stays as it was (nemu64-test, hardware).
	case 10:
		return;
	case 11: {
		uint32_t begin = addr & ~7u;
		addr = begin + ((e + (addr & 8)) & 15);
		int base = vt & ~7, reg = e >> 1;
		for (int i = 0; i < 8; ++i) {
			for (int half = 0; half < 2; ++half) {
				setVbyte(r, base + reg, i * 2 + half, *memAt(mem, swap, addr));
				addr = addr + 1 == begin + 16 ? begin : addr + 1;
			}
			reg = (reg + 1) & 7;
		}
		return;
	}
	}
}

static inline __attribute__((always_inline)) void vectorStoreIn(struct RSP* r, uint8_t* mem, uint32_t swap, uint32_t op) {
	int vt = RT, e = (op >> 7) & 15, kind = RD;
	int32_t offset = (int32_t) ((op & 0x7F) << 25) >> 25;
	static const int scale[12] = { 1, 2, 4, 8, 16, 16, 8, 8, 16, 16, 16, 16 };
	if (kind > 11) {
		return;
	}
	uint32_t addr = (r->r[RS] + (uint32_t) (offset * scale[kind])) & 0xFFF;
	// Whole lanes to DMEM without wrapping, as for loads.
	int size = scale[kind];
	if (!swap && kind >= 1 && kind <= 4 && !(e & 1) && e + size <= 16 && addr + size <= 0x1000 && (kind < 4 || !(addr & 15))) {
		uint8_t* m = &mem[addr];
		for (int i = 0; i < size / 2; ++i) {
			uint16_t v = (uint16_t) r->vr[vt][e / 2 + i];
			m[2 * i] = (uint8_t) (v >> 8);
			m[2 * i + 1] = (uint8_t) v;
		}
		return;
	}
	switch (kind) {
	case 0:
	case 1:
	case 2:
	case 3:
		for (int i = 0; i < scale[kind]; ++i) {
			*memAt(mem, swap, addr + i) = vbyte(r, vt, (e + i) & 15);
		}
		return;
	case 4: {
		uint32_t end = (addr & ~15u) + 16;
		for (int i = 0; addr + i < end; ++i) {
			*memAt(mem, swap, addr + i) = vbyte(r, vt, (e + i) & 15);
		}
		return;
	}
	case 5: {
		int count = addr & 15;
		uint32_t start = addr & ~15u;
		for (int k = 0; k < count; ++k) {
			*memAt(mem, swap, start + k) = vbyte(r, vt, (e + 16 - count + k) & 15);
		}
		return;
	}
	case 6:
	case 7:
		for (int i = e; i < e + 8; ++i) {
			bool low = (i & 15) < 8;
			uint8_t b = low == (kind == 6) ? vbyte(r, vt, (i & 7) << 1) : (uint8_t) ((uint16_t) r->vr[vt][i & 7] >> 7);
			*memAt(mem, swap, addr++) = b;
		}
		return;
	case 8: {
		int index = addr & 7;
		addr &= ~7u;
		for (int i = 0; i < 8; ++i) {
			int b = e + i * 2;
			uint8_t v = (uint8_t) (vbyte(r, vt, b & 15) << 1 | vbyte(r, vt, (b + 1) & 15) >> 7);
			*memAt(mem, swap, addr + ((index + i * 2) & 15)) = v;
		}
		return;
	}
	case 9: {
		static const int8_t lanes[16][4] = {
			{ 0, 1, 2, 3 }, { 6, 7, 4, 5 }, { -1 }, { -1 }, { 1, 2, 3, 0 }, { 7, 4, 5, 6 }, { -1 }, { -1 },
			{ 4, 5, 6, 7 }, { -1 }, { -1 }, { 3, 0, 1, 2 }, { 5, 6, 7, 4 }, { -1 }, { -1 }, { 0, 1, 2, 3 },
		};
		int base = addr & 7;
		addr &= ~7u;
		for (int k = 0; k < 4; ++k) {
			int lane = lanes[e][0] < 0 ? -1 : lanes[e][k];
			uint8_t v = lane < 0 ? 0 : (uint8_t) ((uint16_t) r->vr[vt][lane] >> 7);
			*memAt(mem, swap, addr + ((base + k * 4) & 15)) = v;
		}
		return;
	}
	case 10: {
		int base = addr & 7;
		addr &= ~7u;
		for (int i = e; i < e + 16; ++i) {
			*memAt(mem, swap, addr + (base++ & 15)) = vbyte(r, vt, i & 15);
		}
		return;
	}
	case 11: {
		int start = vt & ~7;
		int element = 16 - (e & ~1);
		int base = (int) (addr & 7) - (e & ~1);
		addr &= ~7u;
		for (int reg = start; reg < start + 8; ++reg) {
			*memAt(mem, swap, addr + (base++ & 15)) = vbyte(r, reg, element++ & 15);
			*memAt(mem, swap, addr + (base++ & 15)) = vbyte(r, reg, element++ & 15);
		}
		return;
	}
	}
}

static void vectorLoad(struct N64* n, uint32_t op) {
	vectorLoadIn(&n->rsp, n->sp.mem, 0, op);
}

static void vectorStore(struct N64* n, uint32_t op) {
	vectorStoreIn(&n->rsp, n->sp.mem, 0, op);
}

void rspVectorLoadSwapped(struct RSP* r, uint8_t* mem, uint32_t op) {
	vectorLoadIn(r, mem, 3, op);
}

void rspVectorStoreSwapped(struct RSP* r, uint8_t* mem, uint32_t op) {
	vectorStoreIn(r, mem, 3, op);
}

static inline __attribute__((always_inline)) void cop2In(struct RSP* r, uint32_t op) {
	if (op & (1 << 25)) {
		rspVectorOp(r, op);
		return;
	}
	int e = (op >> 7) & 15;
	switch (RS) {
	case 0x00:
		if (RT) {
			r->r[RT] = (uint32_t) (int32_t) (int16_t) (vbyte(r, RD, e) << 8 | vbyte(r, RD, (e + 1) & 15));
		}
		return;
	// CFC2 and CTC2 name the flags by the register's low two bits; 2 and 3 are both VCE.
	case 0x02: {
		int32_t v = (RD & 3) == 0 ? (int16_t) r->vco : (RD & 3) == 1 ? (int16_t) r->vcc : r->vce;
		if (RT) {
			r->r[RT] = (uint32_t) v;
		}
		return;
	}
	case 0x04:
		setVbyte(r, RD, e, (uint8_t) (r->r[RT] >> 8));
		if (e < 15) {
			setVbyte(r, RD, e + 1, (uint8_t) r->r[RT]);
		}
		return;
	case 0x06:
		if ((RD & 3) == 0) {
			r->vco = (uint16_t) r->r[RT];
		} else if ((RD & 3) == 1) {
			r->vcc = (uint16_t) r->r[RT];
		} else {
			r->vce = (uint8_t) r->r[RT];
		}
		return;
	}
}

static void cop2(struct N64* n, uint32_t op) {
	cop2In(&n->rsp, op);
}

void rspCop2Regs(struct RSP* r, uint32_t op) {
	cop2In(r, op);
}

static uint32_t load(struct N64* n, uint32_t addr, int size) {
	uint32_t v = 0;
	for (int i = 0; i < size; ++i) {
		v = v << 8 | *dmem(n, addr + i);
	}
	return v;
}

static void storeBytes(struct N64* n, uint32_t addr, int size, uint32_t v) {
	for (int i = size - 1; i >= 0; --i) {
		*dmem(n, addr + i) = (uint8_t) v;
		v >>= 8;
	}
}

static void halt(struct N64* n) {
	audioCheckEnd(n);
	n->sp.status |= 3;
	if (n->sp.status & 0x40) {
		n64Interrupt(n, MI_SP, true);
	}
}

static uint64_t latest(uint64_t a, uint64_t b) {
	return a > b ? a : b;
}

// When an instruction can start: after the results it reads are ready. Loads, MFC0, MFC2 and CFC2
// have a 3-cycle latency; any write to a vector register has a 4-cycle latency.
static uint64_t issueTime(struct RSP* r, uint32_t op) {
	uint64_t t = r->clock;
	uint64_t* s = r->scalarReady;
	uint64_t* v = r->vectorReady;
	switch (op >> 26) {
	case 0x00:
	case 0x04:
	case 0x05:
	case 0x28:
	case 0x29:
	case 0x2B:
		return latest(t, latest(s[RS], s[RT]));
	case 0x10:
		return RS == 0x04 ? latest(t, s[RT]) : t;
	case 0x12:
		if (op & (1 << 25)) {
			return latest(t, latest(v[RD], v[RT]));
		}
		if (RS == 0x00) {
			return latest(t, v[RD]);
		}
		return RS == 0x04 || RS == 0x06 ? latest(t, s[RT]) : t;
	case 0x3A: {
		uint64_t ready = latest(t, s[RS]);
		if (RD == 11) {
			for (int i = 0; i < 8; ++i) {
				ready = latest(ready, v[(RT & ~7) + i]);
			}
			return ready;
		}
		return latest(ready, v[RT]);
	}
	default:
		return latest(t, s[RS]);
	}
}

static void markResults(struct RSP* r, uint32_t op, uint64_t start) {
	switch (op >> 26) {
	case 0x20:
	case 0x21:
	case 0x23:
	case 0x24:
	case 0x25:
	case 0x27:
		r->scalarReady[RT] = start + 3;
		return;
	case 0x10:
		if (RS == 0x00) {
			r->scalarReady[RT] = start + 3;
		}
		return;
	case 0x12:
		if (op & (1 << 25)) {
			r->vectorReady[SA] = start + 4;
		} else if (RS == 0x00 || RS == 0x02) {
			r->scalarReady[RT] = start + 3;
		} else if (RS == 0x04) {
			r->vectorReady[RD] = start + 4;
		}
		return;
	case 0x32:
		if (RD == 11) {
			for (int i = 0; i < 8; ++i) {
				r->vectorReady[(RT & ~7) + i] = start + 4;
			}
		} else {
			r->vectorReady[RT] = start + 4;
		}
		return;
	}
}

static bool isBranch(uint32_t op) {
	int code = op >> 26;
	return (code >= 0x01 && code <= 0x07) || (code == 0x00 && ((op & 0x3F) == 0x08 || (op & 0x3F) == 0x09));
}

static bool isVectorOp(uint32_t op) {
	return (op >> 26) == 0x12 && (op & (1 << 25));
}

// The vector register an instruction writes, or -1.
static int vectorWritten(uint32_t op) {
	if (isVectorOp(op)) {
		return SA;
	}
	if ((op >> 26) == 0x32) {
		return RT;
	}
	if ((op >> 26) == 0x12 && RS == 0x04) {
		return RD;
	}
	return -1;
}

// A scalar-unit and a vector-unit instruction issue together when neither is a branch or in a delay
// slot, the pair is 8-byte aligned and the second doesn't use a vector register the first writes.
static bool pairs(struct RSP* r, uint32_t op) {
	uint32_t first = r->lastOp;
	if (!r->lastAlone || r->inDelay || isBranch(first) || (r->lastPc & 4) || isVectorOp(first) == isVectorOp(op)) {
		return false;
	}
	int code = op >> 26, firstCode = first >> 26;
	if ((code == 0x12 && (RS == 0x02 || RS == 0x06)) || (firstCode == 0x12 && (((first >> 21) & 31) == 0x02 || ((first >> 21) & 31) == 0x06))) {
		return false;
	}
	int w = vectorWritten(first);
	if (w < 0) {
		return true;
	}
	if (isVectorOp(op)) {
		return w != (int) RT && w != (int) RD && w != (int) SA;
	}
	return !((code == 0x3A || code == 0x32) && w == (int) RT) && !(code == 0x12 && (int) RD == w);
}

// The timing of one instruction: when it issues, whether it pairs with the one before, and when
// its results are ready.
void rspTimingStep(struct N64* n, uint32_t op, uint32_t pc) {
	struct RSP* r = &n->rsp;
	uint64_t start = issueTime(r, op);
	if (pairs(r, op)) {
		uint64_t clock = r->clock;
		r->clock = r->lastStart;
		uint64_t ready = issueTime(r, op);
		r->clock = clock;
		if (ready <= r->lastStart) {
			start = r->lastStart;
		}
	}
	bool paired = start == r->lastStart && r->lastAlone && start < r->clock;
	// A branch also costs one bubble cycle after its delay slot.
	if (!paired) {
		r->clock = start + 1 + isBranch(op);
	}
	r->inDelay = isBranch(op);
	r->lastAlone = !paired;
	r->lastOp = op;
	r->lastDesc = rspTimingDesc(op);
	r->lastPc = pc;
	r->lastStart = start;
	markResults(r, op, start);
	r->scalarReady[0] = 0;
}

// What pairs() needs of the instruction before: bit 0 a branch, bit 1 a vector op, bit 2 CFC2 or
// CTC2, bits 8-15 the vector register it writes plus one (0 for none).
uint16_t rspTimingDesc(uint32_t op) {
	bool control = (op >> 26) == 0x12 && (RS == 0x02 || RS == 0x06);
	return (uint16_t) (isBranch(op) | isVectorOp(op) << 1 | control << 2 | (vectorWritten(op) + 1) << 8);
}

// For compiled code: whether op may pair with first (at firstPc) when first issued alone, and the
// registers the timing reads and writes, as issueTime and markResults use them.
bool rspPairable(uint32_t first, uint32_t firstPc, uint32_t op) {
	struct RSP r = { .lastOp = first, .lastPc = firstPc, .inDelay = isBranch(first), .lastAlone = true };
	return pairs(&r, op);
}

bool rspIsBranch(uint32_t op) {
	return isBranch(op);
}

int rspTimingRegs(uint32_t op, int* scalar, int* vector, int* nv, int* written, int* latency, int* nw) {
	int ns = 0;
	*nv = *nw = 0;
	switch (op >> 26) {
	case 0x00:
	case 0x04:
	case 0x05:
	case 0x28:
	case 0x29:
	case 0x2B:
		scalar[ns++] = RS;
		scalar[ns++] = RT;
		break;
	case 0x10:
		if (RS == 0x04) {
			scalar[ns++] = RT;
		}
		break;
	case 0x12:
		if (op & (1 << 25)) {
			vector[(*nv)++] = RD;
			vector[(*nv)++] = RT;
		} else if (RS == 0x00) {
			vector[(*nv)++] = RD;
		} else if (RS == 0x04 || RS == 0x06) {
			scalar[ns++] = RT;
		}
		break;
	case 0x3A:
		scalar[ns++] = RS;
		if (RD == 11) {
			for (int i = 0; i < 8; ++i) {
				vector[(*nv)++] = (RT & ~7) + i;
			}
		} else {
			vector[(*nv)++] = RT;
		}
		break;
	default:
		scalar[ns++] = RS;
	}
	// Written registers: scalar as 0-31, vector as 32-63.
	switch (op >> 26) {
	case 0x20:
	case 0x21:
	case 0x23:
	case 0x24:
	case 0x25:
	case 0x27:
		written[*nw] = RT, latency[(*nw)++] = 3;
		break;
	case 0x10:
		if (RS == 0x00) {
			written[*nw] = RT, latency[(*nw)++] = 3;
		}
		break;
	case 0x12:
		if (op & (1 << 25)) {
			written[*nw] = 32 + SA, latency[(*nw)++] = 4;
		} else if (RS == 0x00 || RS == 0x02) {
			written[*nw] = RT, latency[(*nw)++] = 3;
		} else if (RS == 0x04) {
			written[*nw] = 32 + RD, latency[(*nw)++] = 4;
		}
		break;
	case 0x32:
		if (RD == 11) {
			for (int i = 0; i < 8; ++i) {
				written[*nw] = 32 + (RT & ~7) + i, latency[(*nw)++] = 4;
			}
		} else {
			written[*nw] = 32 + RT, latency[(*nw)++] = 4;
		}
		break;
	}
	return ns;
}

// What compiled code calls for the instructions it doesn't run itself.
void rspCop2(struct N64* n, uint32_t op) {
	cop2(n, op);
}

void rspVectorLoad(struct N64* n, uint32_t op) {
	vectorLoad(n, op);
}

void rspVectorStore(struct N64* n, uint32_t op) {
	vectorStore(n, op);
}

void rspCop0(struct N64* n, uint32_t op) {
	uint32_t* g = n->rsp.r;
	if (RS == 0x00) {
		uint32_t v = busRead32(n, rspCop0Address(RD & 15));
		if (RT) {
			g[RT] = v;
		}
	} else if (RS == 0x04) {
		busWrite32(n, rspCop0Address(RD & 15), g[RT]);
	}
}

void rspBreak(struct N64* n) {
	halt(n);
	armletVerifyEnd(n);
}

void rspStep(struct N64* n) {
	struct RSP* r = &n->rsp;
	uint32_t pc = n->sp.pc & 0xFFC;
	uint32_t op = be32(&n->sp.mem[0x1000 | pc]);
	rspTimingStep(n, op, pc);
	n->sp.pc = r->nextPc;
	r->nextPc = (r->nextPc + 4) & 0xFFC;
	uint32_t* g = r->r;
	uint32_t s = g[RS], t = g[RT];
	uint32_t target = pc + 4 + (IMM << 2);
	uint32_t result = 0;
	int dest = -1;
	switch (op >> 26) {
	case 0x00:
		dest = RD;
		switch (op & 0x3F) {
		case 0x00:
			result = t << SA;
			break;
		case 0x02:
			result = t >> SA;
			break;
		case 0x03:
			result = (uint32_t) ((int32_t) t >> SA);
			break;
		case 0x04:
			result = t << (s & 31);
			break;
		case 0x06:
			result = t >> (s & 31);
			break;
		case 0x07:
			result = (uint32_t) ((int32_t) t >> (s & 31));
			break;
		case 0x08:
			branch(r, true, s);
			dest = -1;
			break;
		case 0x09:
			result = (pc + 8) & 0xFFF;
			branch(r, true, s);
			break;
		case 0x0D:
			rspBreak(n);
			dest = -1;
			break;
		case 0x20:
		case 0x21:
			result = s + t;
			break;
		case 0x22:
		case 0x23:
			result = s - t;
			break;
		case 0x24:
			result = s & t;
			break;
		case 0x25:
			result = s | t;
			break;
		case 0x26:
			result = s ^ t;
			break;
		case 0x27:
			result = ~(s | t);
			break;
		case 0x2A:
			result = (int32_t) s < (int32_t) t;
			break;
		case 0x2B:
			result = s < t;
			break;
		// The rest act as SRLV rd, rs, rs (sauraen's rsp_inval_instrs).
		default:
			result = s >> (s & 31);
		}
		break;
	case 0x01: {
		bool lt = (int32_t) s < 0;
		switch (RT) {
		case 0x00:
			branch(r, lt, target);
			break;
		case 0x01:
			branch(r, !lt, target);
			break;
		case 0x10:
			g[31] = (pc + 8) & 0xFFF;
			branch(r, lt, target);
			break;
		case 0x11:
			g[31] = (pc + 8) & 0xFFF;
			branch(r, !lt, target);
			break;
		}
		break;
	}
	case 0x02:
		branch(r, true, (op & 0x3FF) << 2);
		break;
	case 0x03:
		g[31] = (pc + 8) & 0xFFF;
		branch(r, true, (op & 0x3FF) << 2);
		break;
	case 0x04:
		branch(r, s == t, target);
		break;
	case 0x05:
		branch(r, s != t, target);
		break;
	case 0x06:
		branch(r, (int32_t) s <= 0, target);
		break;
	case 0x07:
		branch(r, (int32_t) s > 0, target);
		break;
	case 0x08:
	case 0x09:
		dest = RT, result = s + IMM;
		break;
	case 0x0A:
		dest = RT, result = (int32_t) s < (int32_t) IMM;
		break;
	case 0x0B:
		dest = RT, result = s < IMM;
		break;
	case 0x0C:
		dest = RT, result = s & (uint16_t) op;
		break;
	case 0x0D:
		dest = RT, result = s | (uint16_t) op;
		break;
	case 0x0E:
		dest = RT, result = s ^ (uint16_t) op;
		break;
	case 0x0F:
		dest = RT, result = (uint32_t) (uint16_t) op << 16;
		break;
	case 0x10:
		if (RS == 0x00) {
			dest = RT, result = busRead32(n, rspCop0Address(RD & 15));
		} else if (RS == 0x04) {
			busWrite32(n, rspCop0Address(RD & 15), t);
		}
		break;
	case 0x12:
		cop2(n, op);
		break;
	case 0x20:
		dest = RT, result = (uint32_t) (int32_t) (int8_t) load(n, s + IMM, 1);
		break;
	case 0x21:
		dest = RT, result = (uint32_t) (int32_t) (int16_t) load(n, s + IMM, 2);
		break;
	case 0x23:
	case 0x27:
		dest = RT, result = load(n, s + IMM, 4);
		break;
	case 0x24:
		dest = RT, result = load(n, s + IMM, 1);
		break;
	case 0x25:
		dest = RT, result = load(n, s + IMM, 2);
		break;
	case 0x28:
		storeBytes(n, s + IMM, 1, t);
		break;
	case 0x29:
		storeBytes(n, s + IMM, 2, t);
		break;
	case 0x2B:
		storeBytes(n, s + IMM, 4, t);
		break;
	case 0x32:
		vectorLoad(n, op);
		break;
	case 0x3A:
		vectorStore(n, op);
		break;
	}
	if (dest > 0) {
		g[dest] = result;
	}
}
