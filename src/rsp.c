#include "n64.h"

#include <string.h>

#define RS ((op >> 21) & 31)
#define RT ((op >> 16) & 31)
#define RD ((op >> 11) & 31)
#define SA ((op >> 6) & 31)
#define IMM ((uint32_t) (int32_t) (int16_t) op)

static uint16_t reciprocals[512];
static uint16_t inverseRoots[512];

void rspInitTables(void) {
	for (uint32_t i = 0; i < 512; ++i) {
		uint64_t a = i + 512;
		uint64_t b = (1ull << 34) / a;
		// The first entry would be 0x20000; the table holds 0xFFFF there.
		reciprocals[i] = i == 0 ? 0xFFFF : (uint16_t) ((b + 1) >> 8);
		a = (i + 512) >> (i & 1);
		b = 1 << 17;
		while (a * (b + 1) * (b + 1) < (1ull << 44)) {
			++b;
		}
		inverseRoots[i] = (uint16_t) (b >> 1);
	}
}

static uint8_t* dmem(struct N64* n, uint32_t addr) {
	return &n->sp.mem[addr & 0xFFF];
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

static uint32_t rcp(int32_t input) {
	int32_t mask = input >> 31;
	int32_t data = input ^ mask;
	if (input > -32768) {
		data -= mask;
	}
	if (data == 0) {
		return 0x7FFFFFFF;
	}
	if (input == -32768) {
		return 0xFFFF0000;
	}
	int shift = __builtin_clz((uint32_t) data);
	uint32_t index = (uint32_t) (((uint64_t) (uint32_t) data << shift & 0x7FC00000) >> 22);
	uint32_t result = (0x10000u | reciprocals[index]) << 14;
	return (result >> (31 - shift)) ^ (uint32_t) mask;
}

static uint32_t rsq(int32_t input) {
	int32_t mask = input >> 31;
	int32_t data = input ^ mask;
	if (input > -32768) {
		data -= mask;
	}
	if (data == 0) {
		return 0x7FFFFFFF;
	}
	if (input == -32768) {
		return 0xFFFF0000;
	}
	int shift = __builtin_clz((uint32_t) data);
	uint32_t index = (uint32_t) (((uint64_t) (uint32_t) data << shift & 0x7FC00000) >> 22);
	index = ((index | 0x200) & 0x3FE) | (shift & 1);
	uint32_t result = (0x10000u | inverseRoots[index & 0x1FF]) << 14;
	return (result >> ((31 - shift) >> 1)) ^ (uint32_t) mask;
}

static int16_t clampSigned(int64_t acc) {
	int64_t v = acc >> 16;
	if (v < -32768) {
		return -32768;
	}
	if (v > 32767) {
		return 32767;
	}
	return (int16_t) v;
}

// The low accumulator slice, saturated when the accumulator doesn't fit in 32 signed bits.
static uint16_t clampLow(int64_t acc) {
	int64_t v = acc >> 16;
	if (v < -32768) {
		return 0x0000;
	}
	if (v > 32767) {
		return 0xFFFF;
	}
	return (uint16_t) acc;
}

static uint16_t clampUnsigned(int64_t acc) {
	int64_t v = acc >> 16;
	if (v < 0) {
		return 0;
	}
	if (v > 32767) {
		return 0xFFFF;
	}
	return (uint16_t) v;
}

static int64_t sext48(int64_t v) {
	return (int64_t) ((uint64_t) v << 16) >> 16;
}

static void vectorOp(struct N64* n, uint32_t op) {
	struct RSP* r = &n->rsp;
	int e = (op >> 21) & 15;
	int vt = RT, vs = RD, vd = SA;
	int16_t t[8], s[8], d[8];
	for (int i = 0; i < 8; ++i) {
		int lane = e < 2 ? i : e < 4 ? (i & ~1) | (e & 1) : e < 8 ? (i & ~3) | (e & 3) : e & 7;
		t[i] = r->vr[vt][lane];
		s[i] = r->vr[vs][i];
	}
	int64_t* acc = r->acc;
	int func = op & 0x3F;
	switch (func) {
	case 0x00:
	case 0x01:
		for (int i = 0; i < 8; ++i) {
			acc[i] = (int64_t) s[i] * t[i] * 2 + 0x8000;
			d[i] = func == 0 ? clampSigned(acc[i]) : (int16_t) clampUnsigned(acc[i]);
		}
		break;
	case 0x02:
	case 0x0A: {
		bool high = vs & 1;
		for (int i = 0; i < 8; ++i) {
			int64_t value = high ? (int64_t) t[i] << 16 : t[i];
			if ((func == 0x02 && acc[i] >= 0) || (func == 0x0A && acc[i] < 0)) {
				acc[i] = sext48(acc[i] + value);
			}
			d[i] = clampSigned(acc[i]);
		}
		break;
	}
	case 0x03:
		for (int i = 0; i < 8; ++i) {
			int32_t p = (int32_t) s[i] * t[i];
			if (p < 0) {
				p += 31;
			}
			acc[i] = sext48((int64_t) p << 16);
			d[i] = (int16_t) (clampSigned(acc[i] >> 1) & ~0xF);
		}
		break;
	case 0x04:
		for (int i = 0; i < 8; ++i) {
			acc[i] = (int64_t) ((uint32_t) (uint16_t) s[i] * (uint16_t) t[i] >> 16);
			d[i] = (int16_t) clampLow(acc[i]);
		}
		break;
	case 0x05:
		for (int i = 0; i < 8; ++i) {
			acc[i] = (int64_t) s[i] * (uint16_t) t[i];
			d[i] = clampSigned(acc[i]);
		}
		break;
	case 0x06:
		for (int i = 0; i < 8; ++i) {
			acc[i] = (int64_t) (uint16_t) s[i] * t[i];
			d[i] = (int16_t) clampLow(acc[i]);
		}
		break;
	case 0x07:
		for (int i = 0; i < 8; ++i) {
			acc[i] = (int64_t) s[i] * t[i] << 16;
			d[i] = clampSigned(acc[i]);
		}
		break;
	case 0x08:
	case 0x09:
		for (int i = 0; i < 8; ++i) {
			acc[i] = sext48(acc[i] + (int64_t) s[i] * t[i] * 2);
			d[i] = func == 0x08 ? clampSigned(acc[i]) : (int16_t) clampUnsigned(acc[i]);
		}
		break;
	case 0x0B:
		for (int i = 0; i < 8; ++i) {
			int32_t a = (int32_t) (acc[i] >> 16);
			if (a < 0 && !(a & 0x20)) {
				a += 32;
			} else if (a >= 32 && !(a & 0x20)) {
				a -= 32;
			}
			acc[i] = sext48(((int64_t) a << 16) | (acc[i] & 0xFFFF));
			d[i] = (int16_t) (clampSigned(acc[i] >> 1) & ~0xF);
		}
		break;
	case 0x0C:
		for (int i = 0; i < 8; ++i) {
			acc[i] = sext48(acc[i] + (int64_t) ((uint32_t) (uint16_t) s[i] * (uint16_t) t[i] >> 16));
			d[i] = (int16_t) clampLow(acc[i]);
		}
		break;
	case 0x0D:
		for (int i = 0; i < 8; ++i) {
			acc[i] = sext48(acc[i] + (int64_t) s[i] * (uint16_t) t[i]);
			d[i] = clampSigned(acc[i]);
		}
		break;
	case 0x0E:
		for (int i = 0; i < 8; ++i) {
			acc[i] = sext48(acc[i] + (int64_t) (uint16_t) s[i] * t[i]);
			d[i] = (int16_t) clampLow(acc[i]);
		}
		break;
	case 0x0F:
		for (int i = 0; i < 8; ++i) {
			acc[i] = sext48(acc[i] + ((int64_t) s[i] * t[i] << 16));
			d[i] = clampSigned(acc[i]);
		}
		break;
	case 0x10:
	case 0x11:
		for (int i = 0; i < 8; ++i) {
			int carry = (r->vco >> i) & 1;
			int32_t v = func == 0x10 ? s[i] + t[i] + carry : s[i] - t[i] - carry;
			acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) v;
			d[i] = v > 32767 ? 32767 : v < -32768 ? -32768 : (int16_t) v;
		}
		r->vco = 0;
		break;
	case 0x13:
		for (int i = 0; i < 8; ++i) {
			int16_t v;
			if (s[i] < 0) {
				v = t[i] == -32768 ? 32767 : (int16_t) -t[i];
				acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) -t[i];
			} else {
				v = s[i] == 0 ? 0 : t[i];
				acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) v;
			}
			d[i] = v;
		}
		break;
	case 0x14:
	case 0x15: {
		uint16_t vco = 0;
		for (int i = 0; i < 8; ++i) {
			uint32_t a = (uint16_t) s[i], b = (uint16_t) t[i];
			uint32_t v;
			if (func == 0x14) {
				v = a + b;
				vco |= (uint16_t) ((v >> 16) & 1) << i;
			} else {
				v = a - b;
				vco |= (uint16_t) ((a < b) << i | (v != 0) << (i + 8));
			}
			acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) v;
			d[i] = (int16_t) v;
		}
		r->vco = vco;
		break;
	}
	case 0x1D:
		for (int i = 0; i < 8; ++i) {
			uint64_t a = (uint64_t) acc[i];
			d[i] = e == 8 ? (int16_t) (a >> 32) : e == 9 ? (int16_t) (a >> 16) : e == 10 ? (int16_t) a : 0;
		}
		break;
	case 0x20:
	case 0x21:
	case 0x22:
	case 0x23: {
		uint16_t vcc = 0;
		for (int i = 0; i < 8; ++i) {
			bool ne = (r->vco >> (i + 8)) & 1, c = (r->vco >> i) & 1;
			bool cond;
			switch (func) {
			case 0x20:
				cond = s[i] < t[i] || (s[i] == t[i] && ne && c);
				break;
			case 0x21:
				cond = s[i] == t[i] && !ne;
				break;
			case 0x22:
				cond = s[i] != t[i] || ne;
				break;
			default:
				cond = s[i] > t[i] || (s[i] == t[i] && !(ne && c));
			}
			int16_t v = func == 0x21 ? t[i] : cond ? s[i] : t[i];
			if (func == 0x22) {
				v = cond ? s[i] : t[i];
			}
			vcc |= (uint16_t) cond << i;
			acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) v;
			d[i] = v;
		}
		r->vcc = vcc;
		r->vco = 0;
		break;
	}
	case 0x24: {
		uint16_t vcc = 0;
		for (int i = 0; i < 8; ++i) {
			bool le = (r->vcc >> i) & 1, ge = (r->vcc >> (i + 8)) & 1;
			bool c = (r->vco >> i) & 1, ne = (r->vco >> (i + 8)) & 1, vce = (r->vce >> i) & 1;
			uint16_t a = (uint16_t) s[i], b = (uint16_t) t[i];
			int16_t v;
			if (c) {
				if (!ne) {
					uint32_t sum = (uint32_t) a + b;
					bool carry = sum > 0xFFFF;
					bool zero = (sum & 0xFFFF) == 0;
					le = vce ? zero || !carry : zero && !carry;
				}
				v = le ? (int16_t) -t[i] : s[i];
			} else {
				if (!ne) {
					ge = a >= b;
				}
				v = ge ? t[i] : s[i];
			}
			vcc |= (uint16_t) (le << i | ge << (i + 8));
			acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) v;
			d[i] = v;
		}
		r->vcc = vcc;
		r->vco = 0;
		r->vce = 0;
		break;
	}
	case 0x25: {
		uint16_t vcc = 0, vco = 0;
		uint8_t vce = 0;
		for (int i = 0; i < 8; ++i) {
			int32_t a = s[i], b = t[i];
			bool le, ge, ne, c;
			int16_t v;
			if ((a ^ b) < 0) {
				ge = b < 0;
				le = a + b <= 0;
				vce |= (uint8_t) ((a + b == -1) << i);
				ne = a + b != 0 && a + b != -1;
				c = true;
				v = le ? (int16_t) -b : (int16_t) a;
			} else {
				le = b < 0;
				ge = a - b >= 0;
				ne = a - b != 0;
				c = false;
				v = ge ? (int16_t) b : (int16_t) a;
			}
			vcc |= (uint16_t) (le << i | ge << (i + 8));
			vco |= (uint16_t) (c << i | ne << (i + 8));
			acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) v;
			d[i] = v;
		}
		r->vcc = vcc;
		r->vco = vco;
		r->vce = vce;
		break;
	}
	case 0x26: {
		uint16_t vcc = 0;
		for (int i = 0; i < 8; ++i) {
			int32_t a = s[i], b = t[i];
			bool le, ge;
			int16_t v;
			if ((a ^ b) < 0) {
				ge = b < 0;
				le = a + b + 1 <= 0;
				v = le ? (int16_t) ~b : (int16_t) a;
			} else {
				le = b < 0;
				ge = a - b >= 0;
				v = ge ? (int16_t) b : (int16_t) a;
			}
			vcc |= (uint16_t) (le << i | ge << (i + 8));
			acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) v;
			d[i] = v;
		}
		r->vcc = vcc;
		r->vco = 0;
		r->vce = 0;
		break;
	}
	case 0x27:
		for (int i = 0; i < 8; ++i) {
			d[i] = (r->vcc >> i) & 1 ? s[i] : t[i];
			acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) d[i];
		}
		r->vco = 0;
		break;
	case 0x28:
	case 0x29:
	case 0x2A:
	case 0x2B:
	case 0x2C:
	case 0x2D:
		for (int i = 0; i < 8; ++i) {
			uint16_t a = (uint16_t) s[i], b = (uint16_t) t[i], v;
			switch (func) {
			case 0x28:
				v = a & b;
				break;
			case 0x29:
				v = (uint16_t) ~(a & b);
				break;
			case 0x2A:
				v = a | b;
				break;
			case 0x2B:
				v = (uint16_t) ~(a | b);
				break;
			case 0x2C:
				v = a ^ b;
				break;
			default:
				v = (uint16_t) ~(a ^ b);
			}
			acc[i] = (acc[i] & ~0xFFFFll) | v;
			d[i] = (int16_t) v;
		}
		break;
	case 0x30:
	case 0x31:
	case 0x32:
	case 0x33:
	case 0x34:
	case 0x35:
	case 0x36: {
		int de = vs & 7;
		int16_t in = r->vr[vt][e & 7];
		memcpy(d, r->vr[vd], sizeof(d));
		for (int i = 0; i < 8; ++i) {
			acc[i] = (acc[i] & ~0xFFFFll) | (uint16_t) t[i];
		}
		if (func == 0x33) {
			d[de] = t[de];
			break;
		}
		if (func == 0x32 || func == 0x36) {
			r->divIn = in;
			r->divDp = true;
			d[de] = (int16_t) (r->divOut >> 16);
			break;
		}
		int32_t input = (func == 0x31 || func == 0x35) && r->divDp ? (int32_t) ((uint32_t) (uint16_t) r->divIn << 16 | (uint16_t) in)
		                                                         : in;
		uint32_t result = func <= 0x32 ? rcp(input) : rsq(input);
		r->divOut = result;
		r->divDp = false;
		d[de] = (int16_t) result;
		break;
	}
	case 0x37:
	case 0x3F:
		return;
	default:
		memset(d, 0, sizeof(d));
	}
	memcpy(r->vr[vd], d, sizeof(d));
}

static void vectorLoad(struct N64* n, uint32_t op) {
	struct RSP* r = &n->rsp;
	int vt = RT, e = (op >> 7) & 15, kind = RD;
	int32_t offset = (int32_t) ((op & 0x7F) << 25) >> 25;
	static const int scale[12] = { 1, 2, 4, 8, 16, 16, 8, 8, 16, 16, 16, 16 };
	if (kind > 11) {
		return;
	}
	uint32_t addr = (r->r[RS] + (uint32_t) (offset * scale[kind])) & 0xFFF;
	switch (kind) {
	case 0:
	case 1:
	case 2:
	case 3:
		for (int i = 0; i < scale[kind] && e + i < 16; ++i) {
			setVbyte(r, vt, e + i, *dmem(n, addr + i));
		}
		return;
	case 4: {
		uint32_t end = (addr & ~15u) + 16;
		for (int i = e; i < 16 && addr + (i - e) < end; ++i) {
			setVbyte(r, vt, i, *dmem(n, addr + (i - e)));
		}
		return;
	}
	case 5: {
		int count = addr & 15;
		uint32_t start = addr & ~15u;
		for (int k = 0; k < count; ++k) {
			int i = e + 16 - count + k;
			if (i < 16) {
				setVbyte(r, vt, i, *dmem(n, start + k));
			}
		}
		return;
	}
	case 6:
	case 7: {
		int index = (int) (addr & 7) - e;
		addr &= ~7u;
		for (int i = 0; i < 8; ++i) {
			uint16_t b = *dmem(n, addr + ((index + i) & 15));
			r->vr[vt][i] = (int16_t) (kind == 6 ? b << 8 : b << 7);
		}
		return;
	}
	case 8: {
		int index = (int) (addr & 7) - e;
		addr &= ~7u;
		for (int i = 0; i < 8; ++i) {
			r->vr[vt][i] = (int16_t) (*dmem(n, addr + ((index + i * 2) & 15)) << 7);
		}
		return;
	}
	case 9: {
		int index = (int) (addr & 7) - e;
		addr &= ~7u;
		int16_t tmp[8];
		for (int i = 0; i < 4; ++i) {
			tmp[i] = (int16_t) (*dmem(n, addr + ((index + i * 4) & 15)) << 7);
			tmp[i + 4] = (int16_t) (*dmem(n, addr + ((index + i * 4 + 8) & 15)) << 7);
		}
		for (int i = e; i < e + 8 && i < 16; ++i) {
			uint16_t v = (uint16_t) tmp[i >> 1];
			setVbyte(r, vt, i, i & 1 ? (uint8_t) v : (uint8_t) (v >> 8));
		}
		return;
	}
	case 10:
		for (int i = 16 - e; i < e + 16; ++i) {
			setVbyte(r, vt, i & 15, *dmem(n, addr + (i - (16 - e))));
		}
		return;
	case 11: {
		uint32_t begin = addr & ~7u;
		addr = begin + ((e + (addr & 8)) & 15);
		int base = vt & ~7, reg = e >> 1;
		for (int i = 0; i < 8; ++i) {
			for (int half = 0; half < 2; ++half) {
				setVbyte(r, base + reg, i * 2 + half, *dmem(n, addr));
				addr = addr + 1 == begin + 16 ? begin : addr + 1;
			}
			reg = (reg + 1) & 7;
		}
		return;
	}
	}
}

static void vectorStore(struct N64* n, uint32_t op) {
	struct RSP* r = &n->rsp;
	int vt = RT, e = (op >> 7) & 15, kind = RD;
	int32_t offset = (int32_t) ((op & 0x7F) << 25) >> 25;
	static const int scale[12] = { 1, 2, 4, 8, 16, 16, 8, 8, 16, 16, 16, 16 };
	if (kind > 11) {
		return;
	}
	uint32_t addr = (r->r[RS] + (uint32_t) (offset * scale[kind])) & 0xFFF;
	switch (kind) {
	case 0:
	case 1:
	case 2:
	case 3:
		for (int i = 0; i < scale[kind]; ++i) {
			*dmem(n, addr + i) = vbyte(r, vt, (e + i) & 15);
		}
		return;
	case 4: {
		uint32_t end = (addr & ~15u) + 16;
		for (int i = 0; addr + i < end; ++i) {
			*dmem(n, addr + i) = vbyte(r, vt, (e + i) & 15);
		}
		return;
	}
	case 5: {
		int count = addr & 15;
		uint32_t start = addr & ~15u;
		for (int k = 0; k < count; ++k) {
			*dmem(n, start + k) = vbyte(r, vt, (e + 16 - count + k) & 15);
		}
		return;
	}
	case 6:
	case 7:
		for (int i = e; i < e + 8; ++i) {
			bool low = (i & 15) < 8;
			uint8_t b = low == (kind == 6) ? vbyte(r, vt, (i & 7) << 1) : (uint8_t) ((uint16_t) r->vr[vt][i & 7] >> 7);
			*dmem(n, addr++) = b;
		}
		return;
	case 8: {
		int index = addr & 7;
		addr &= ~7u;
		for (int i = 0; i < 8; ++i) {
			int b = e + i * 2;
			uint8_t v = (uint8_t) (vbyte(r, vt, b & 15) << 1 | vbyte(r, vt, (b + 1) & 15) >> 7);
			*dmem(n, addr + ((index + i * 2) & 15)) = v;
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
			*dmem(n, addr + ((base + k * 4) & 15)) = v;
		}
		return;
	}
	case 10: {
		int base = addr & 7;
		addr &= ~7u;
		for (int i = e; i < e + 16; ++i) {
			*dmem(n, addr + (base++ & 15)) = vbyte(r, vt, i & 15);
		}
		return;
	}
	case 11: {
		int start = vt & ~7;
		int element = 16 - (e & ~1);
		int base = (int) (addr & 7) - (e & ~1);
		addr &= ~7u;
		for (int reg = start; reg < start + 8; ++reg) {
			*dmem(n, addr + (base++ & 15)) = vbyte(r, reg, element++ & 15);
			*dmem(n, addr + (base++ & 15)) = vbyte(r, reg, element++ & 15);
		}
		return;
	}
	}
}

static void cop2(struct N64* n, uint32_t op) {
	struct RSP* r = &n->rsp;
	if (op & (1 << 25)) {
		vectorOp(n, op);
		return;
	}
	int e = (op >> 7) & 15;
	switch (RS) {
	case 0x00:
		if (RT) {
			r->r[RT] = (uint32_t) (int32_t) (int16_t) (vbyte(r, RD, e) << 8 | vbyte(r, RD, (e + 1) & 15));
		}
		return;
	case 0x02: {
		int32_t v = RD == 0 ? (int16_t) r->vco : RD == 1 ? (int16_t) r->vcc : r->vce;
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
		if (RD == 0) {
			r->vco = (uint16_t) r->r[RT];
		} else if (RD == 1) {
			r->vcc = (uint16_t) r->r[RT];
		} else {
			r->vce = (uint8_t) r->r[RT];
		}
		return;
	}
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

void rspStep(struct N64* n) {
	struct RSP* r = &n->rsp;
	uint32_t pc = n->sp.pc & 0xFFC;
	uint32_t op = be32(&n->sp.mem[0x1000 | pc]);
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
	r->lastPc = pc;
	r->lastStart = start;
	markResults(r, op, start);
	r->scalarReady[0] = 0;
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
			result = pc + 8;
			branch(r, true, s);
			break;
		case 0x0D:
			halt(n);
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
		default:
			dest = -1;
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
			g[31] = pc + 8;
			branch(r, lt, target);
			break;
		case 0x11:
			g[31] = pc + 8;
			branch(r, !lt, target);
			break;
		}
		break;
	}
	case 0x02:
		branch(r, true, (op & 0x3FF) << 2);
		break;
	case 0x03:
		g[31] = pc + 8;
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
