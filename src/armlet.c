// arMLEt's task-level native graphics (src/armlet/, from mupen64plus-rsp-armlet by beebono,
// GPL-2.0-or-later): a graphics task whose microcode build has a front-end there runs as C code
// that leaves what the microcode would (DMEM, IMEM, the RDRAM it writes, its RDP lists, the SP and
// DP registers), instead of on the RSP.
//
// This is the framework the front-ends run in (arMLEt's hle_core.c and hle_rsp.cpp, for armN64):
// the task runs on images of DMEM and IMEM in arMLEt's layout (host words) and of the SP and DP
// registers; its RDRAM writes go to RDRAM with an undo log, and they and its RDP register writes are
// logged. A task the front-end gives up on is undone and runs on the RSP. A finished one is undone
// too, then its log is replayed in order through the registers, as the RSP would have made them.
#include "n64.h"

#include "hle.h"
#include "gfx_ucode.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- The builds (each front-end registers its own in the section hle_ucodes)

static const struct hle_ucode* const sentinel __attribute__((used, section("hle_ucodes"))) = NULL;
extern const struct hle_ucode* const __start_hle_ucodes[];
extern const struct hle_ucode* const __stop_hle_ucodes[];

const struct hle_ucode* armlet_hle_next(const struct hle_ucode* prev) {
	const struct hle_ucode* const* p = __start_hle_ucodes;
	if (prev) {
		while (p < __stop_hle_ucodes && *p != prev) {
			++p;
		}
		if (p < __stop_hle_ucodes) {
			++p;
		}
	}
	for (; p < __stop_hle_ucodes; ++p) {
		if (*p) {
			return *p;
		}
	}
	return NULL;
}

const struct hle_ucode* armlet_hle_find(uint64_t id) {
	for (const struct hle_ucode* const* p = __start_hle_ucodes; p < __stop_hle_ucodes; ++p) {
		if (*p && (*p)->id == id) {
			return *p;
		}
	}
	return NULL;
}

uint64_t armlet_ucode_hash(const unsigned char* rdram, uint32_t addr, uint32_t bytes) {
	uint64_t h = 0xcbf29ce484222325ull;
	for (uint32_t i = 0; i < bytes; i += 4) {
		h = (h ^ hle_ram32(rdram, addr + i)) * 0x100000001b3ull;
	}
	return h;
}

bool armlet_gfx_allowed(const struct armlet_config* cfg, uint64_t id) {
	(void) cfg, (void) id;
	return true;
}

void armlet_log(int level, const char* fmt, ...) {
	static int on = -1;
	if (on < 0) {
		on = getenv("N64ARMLETLOG") != NULL;
	}
	if (on || level == M64MSG_ERROR) {
		va_list a;
		va_start(a, fmt);
		vfprintf(stderr, fmt, a);
		va_end(a);
		fputc('\n', stderr);
	}
}

// The RSP's reciprocal ROM, which the front-ends' fixed-point code reads (hle_fx.h declares it):
// rsp_vu.c's.
uint16_t rsp_reciprocal_rom[1024];

// armN64's RSP doesn't stop a task at an idle poll (the fork's policy is off).
int prsp_poll_idle_limit(void) {
	return 0;
}

void prsp_poll_idle_mark(const uint32_t* sr) {
	(void) sr;
}

// ---- The task's world

// What the task logs for the commit: RDRAM written (bytes in the arena, big-endian), and RDP register
// writes (with an XBUS list's bytes from DMEM).
enum { REC_RAM, REC_REG };

struct Rec {
	uint8_t kind, reg;
	uint32_t addr, len, value;
	size_t at;
};

// RDRAM as it was before the task wrote it.
struct Undo {
	uint32_t addr, len;
	size_t at;
};

static struct {
	struct N64* n;
	struct armlet_ctx ctx;
	uint32_t dmem[1024], imem[1024];
	// The SP and DP registers by MTC0 number; START waiting for END; the SP interrupt.
	uint32_t regs[16], startStatus;
	uint32_t* cr[16];
	bool startValid;
	uint32_t irq;
	bool imemLoaded;
	// The interpreter's register file (scalars, vector unit).
	struct RSP rsp;
	struct Rec* recs;
	size_t recCount, recCap;
	uint8_t* arena;
	size_t arenaUsed, arenaCap;
	struct Undo* undo;
	size_t undoCount, undoCap;
	uint8_t* old;
	size_t oldUsed, oldCap;
	const char* why;
	unsigned endPc;
	bool broke;
	uint32_t listBytes;
	// The RDP commands the task made, for the verify (concatenated lists, big-endian).
	uint8_t* stream;
	size_t streamUsed, streamCap;
	// The last microcode text identified.
	uint32_t textAddr;
	uint64_t textId;
	bool textKnown;
	uint8_t text[0x1000];
} A;

static void* grow(void* p, size_t* cap, size_t need, size_t size) {
	if (need <= *cap) {
		return p;
	}
	size_t c = *cap ? *cap : 1024;
	while (c < need) {
		c *= 2;
	}
	p = realloc(p, c * size);
	if (!p) {
		abort();
	}
	*cap = c;
	return p;
}

static struct Rec* rec(int kind) {
	A.recs = grow(A.recs, &A.recCap, A.recCount + 1, sizeof(*A.recs));
	struct Rec* r = &A.recs[A.recCount++];
	memset(r, 0, sizeof(*r));
	r->kind = (uint8_t) kind;
	return r;
}

static size_t arenaPut(const uint8_t* bytes, uint32_t len) {
	A.arena = grow(A.arena, &A.arenaCap, A.arenaUsed + len, 1);
	memcpy(A.arena + A.arenaUsed, bytes, len);
	A.arenaUsed += len;
	return A.arenaUsed - len;
}

static void streamPut(const uint8_t* bytes, uint32_t len) {
	A.stream = grow(A.stream, &A.streamCap, A.streamUsed + len, 1);
	memcpy(A.stream + A.streamUsed, bytes, len);
	A.streamUsed += len;
}

static uint8_t dmemByte(unsigned a) {
	return ((const uint8_t*) A.dmem)[(a & 0xFFF) ^ 3];
}

// RDRAM [addr, addr + len) (inside RDRAM) written by the task: the old bytes kept, the new logged.
static void ramWrite(uint32_t addr, const uint8_t* bytes, uint32_t len) {
	struct N64* n = A.n;
	if (!len || addr >= RDRAM_SIZE) {
		return;
	}
	len = addr + len > RDRAM_SIZE ? RDRAM_SIZE - addr : len;
	n64RdramWrite(n, addr, len);
	A.undo = grow(A.undo, &A.undoCap, A.undoCount + 1, sizeof(*A.undo));
	A.old = grow(A.old, &A.oldCap, A.oldUsed + len, 1);
	A.undo[A.undoCount++] = (struct Undo) { addr, len, A.oldUsed };
	memcpy(A.old + A.oldUsed, n->rdram + addr, len);
	A.oldUsed += len;
	memcpy(n->rdram + addr, bytes, len);
	struct Rec* r = rec(REC_RAM);
	r->addr = addr, r->len = len;
	r->at = arenaPut(bytes, len);
}

static void undoAll(void) {
	for (size_t i = A.undoCount; i-- > 0;) {
		struct Undo* u = &A.undo[i];
		n64RdramWrite(A.n, u->addr, u->len);
		memcpy(A.n->rdram + u->addr, A.old + u->at, u->len);
	}
	A.undoCount = A.oldUsed = 0;
}

int hle_decline(struct hle_task* t, const char* why) {
	(void) t;
	if (!A.why) {
		A.why = why;
	}
	return HLE_DECLINE;
}

// An SP DMA on the images, as bus.c's spDma makes it (the registers left as it leaves them).
static void dma(uint32_t lenReg, bool toRdram) {
	struct N64* n = A.n;
	uint32_t len = ((lenReg & 0xFFF) | 7) + 1, count = ((lenReg >> 12) & 0xFF) + 1, skip = lenReg >> 20;
	uint32_t mem = A.regs[0] & 0x1FF8, base = mem & 0x1000, dram = A.regs[1] & 0xFFFFF8;
	uint8_t* img = (uint8_t*) (base ? A.imem : A.dmem);
	uint8_t row[0x1000];
	for (uint32_t k = 0; k < count; ++k) {
		if (toRdram) {
			for (uint32_t i = 0; i < len; ++i) {
				row[i] = img[((mem + i) & 0xFFF) ^ 3];
			}
			// A row that runs past RDRAM's end wraps (its bytes beyond RDRAM are dropped).
			uint32_t d = dram & 0xFFFFFF, first = d + len > RDRAM_SIZE ? (d < RDRAM_SIZE ? RDRAM_SIZE - d : 0) : len;
			ramWrite(d, row, first);
		} else {
			n64RdramRead(n, dram & (RDRAM_SIZE - 1), len);
			for (uint32_t i = 0; i < len; ++i) {
				uint32_t d = (dram + i) & 0xFFFFFF;
				img[((mem + i) & 0xFFF) ^ 3] = d < RDRAM_SIZE ? n->rdram[d] : 0;
			}
			A.imemLoaded |= base != 0;
		}
		mem = base | ((mem + len) & 0xFFF);
		dram += len + skip;
	}
	A.regs[0] = mem;
	A.regs[1] = dram;
	A.regs[2] = A.regs[3] = 0xFF8 | (lenReg & 0xFFF00000u);
}

void hle_dma_read(struct hle_task* t, unsigned memAddr, unsigned dramAddr, uint32_t len) {
	hle_mtc0(t, 0, memAddr);
	hle_mtc0(t, 1, dramAddr);
	hle_mtc0(t, 2, len);
}

void hle_dma_write(struct hle_task* t, unsigned memAddr, unsigned dramAddr, uint32_t len) {
	hle_mtc0(t, 0, memAddr);
	hle_mtc0(t, 1, dramAddr);
	hle_mtc0(t, 3, len);
}

void hle_rdram_write(struct hle_task* t, uint32_t dram, const uint32_t* words, uint32_t nbytes) {
	(void) t;
	uint8_t bytes[0x1000];
	dram &= 0x7FFFF8;
	for (uint32_t done = 0; done < nbytes;) {
		uint32_t len = nbytes - done > sizeof(bytes) ? (uint32_t) sizeof(bytes) : nbytes - done;
		for (uint32_t i = 0; i < len; i += 4) {
			uint32_t w = words[(done + i) >> 2];
			bytes[i] = (uint8_t) (w >> 24), bytes[i + 1] = (uint8_t) (w >> 16), bytes[i + 2] = (uint8_t) (w >> 8), bytes[i + 3] = (uint8_t) w;
		}
		ramWrite(dram + done, bytes, len);
		done += len;
	}
}

// SP_STATUS written: its clear and set bits as bus.c takes them. True when it halts the RSP.
static bool spStatus(uint32_t v) {
	uint32_t* s = &A.regs[4];
	uint32_t const clear = v & ~(v >> 1) & 0x00AAAAA9u, set = v & ~(v << 1) & 0x01555552u;
	v = clear | set | (v & 4);
	*s = v & 1 ? *s & ~1u : *s;
	*s = v & 2 ? *s | 1 : *s;
	*s = v & 4 ? *s & ~2u : *s;
	A.irq = v & 8 ? A.irq & ~1u : v & 16 ? A.irq | 1 : A.irq;
	*s = v & 32 ? *s & ~0x20u : v & 64 ? *s | 0x20 : *s;
	*s = v & 128 ? *s & ~0x40u : v & 256 ? *s | 0x40 : *s;
	for (int sig = 0; sig < 8; ++sig) {
		*s = v & (1u << (9 + sig * 2)) ? *s & ~(0x80u << sig) : *s;
		*s = v & (1u << (10 + sig * 2)) ? *s | 0x80u << sig : *s;
	}
	return (v & 2) != 0;
}

int hle_mtc0(struct hle_task* t, unsigned rd, uint32_t value) {
	(void) t;
	switch (rd & 15) {
	case 0:
		A.regs[0] = value & 0x1FF8;
		break;
	case 1:
		A.regs[1] = value & 0xFFFFF8;
		break;
	case 2:
	case 3:
		dma(value, (rd & 15) == 3);
		break;
	case 4:
		return spStatus(value);
	case 7:
		A.regs[7] = 0;
		break;
	case 8: {
		struct Rec* r = rec(REC_REG);
		r->reg = 8, r->value = value;
		if (!A.startValid) {
			A.regs[8] = value & 0xFFFFF8;
			A.startValid = true;
		}
		break;
	}
	case 9: {
		struct Rec* r = rec(REC_REG);
		r->reg = 9, r->value = value;
		A.regs[9] = value & 0xFFFFF8;
		if (A.startValid) {
			A.regs[10] = A.regs[8];
			A.startValid = false;
		}
		// The RDP takes the list at once: CURRENT reaches END. An XBUS list is in DMEM, which the
		// microcode goes on to overwrite: its bytes are kept for the replay.
		uint32_t cur = A.regs[10], end = A.regs[9];
		if (end > cur && end - cur <= (A.regs[11] & 1 ? 0x1000u : RDRAM_SIZE)) {
			uint32_t len = end - cur;
			A.listBytes += len;
			if (A.regs[11] & 1) {
				uint8_t bytes[0x1000];
				for (uint32_t i = 0; i < len; ++i) {
					bytes[i] = dmemByte(cur + i);
				}
				r->addr = cur, r->len = len;
				r->at = arenaPut(bytes, len);
				streamPut(bytes, len);
			} else if (cur + len <= RDRAM_SIZE) {
				streamPut(A.n->rdram + cur, len);
			}
		}
		A.regs[10] = end;
		break;
	}
	case 11: {
		struct Rec* r = rec(REC_REG);
		r->reg = 11, r->value = value;
		uint32_t* s = &A.regs[11];
		*s = value & 1 ? *s & ~1u : *s;
		*s = value & 2 ? *s | 1 : *s;
		*s = value & 8 ? *s | 2 : *s;
		*s = value & 4 ? *s & ~2u : *s;
		break;
	}
	}
	return 0;
}

int hle_poll_status(struct hle_task* t, unsigned rt, uint32_t* value) {
	(void) t, (void) rt;
	*value = A.regs[4];
	return 0;
}

void hle_cpu_init(struct hle_task* t, struct hle_cpu* c, unsigned pc, unsigned maxSteps) {
	(void) t;
	c->r = A.rsp.r;
	c->pc = pc & 0xFFC;
	c->steps = 0;
	c->max_steps = maxSteps;
	c->stop_pc = 0x1000;
}

void hle_break(struct hle_task* t, unsigned endPc) {
	(void) t;
	A.regs[4] |= 3;
	if (A.regs[4] & 0x40) {
		A.irq |= 1;
	}
	A.endPc = endPc & 0xFFC;
	A.broke = true;
}

// armN64's RSP never stops a task at an idle poll: no task is cut.
void hle_cut(struct hle_task* t, unsigned endPc) {
	(void) t, (void) endPc;
}

void hle_cut_live(struct hle_task* t, unsigned lo0, unsigned hi0, unsigned lo1, unsigned hi1, uint32_t sregs) {
	(void) t, (void) lo0, (void) hi0, (void) lo1, (void) hi1, (void) sregs;
}

// Only registered builds run: there is no gate of command kinds and no build under validation.
int hle_cmd_slow(struct hle_task* t, unsigned kind) {
	(void) t, (void) kind;
	return 0;
}

int hle_vtx_kinds_slow(struct hle_task* t, uint32_t flagsOr, int wneg) {
	(void) t, (void) flagsOr, (void) wneg;
	return 0;
}

void hle_bind(struct hle_task* t, const struct hle_ucode* u) {
	(void) u;
	t->gate = NULL;
}

int hle_build_usable(const struct hle_task* t, const struct hle_ucode* u) {
	(void) t, (void) u;
	return 1;
}

int hle_exact(void) {
	return 0;
}

void hle_dev_mark(size_t m[2]) {
	m[0] = A.recCount, m[1] = A.arenaUsed;
}

void hle_dev_rewind(const size_t m[2]) {
	A.recCount = m[0], A.arenaUsed = m[1];
}

void armlet_hle_poison(const struct hle_ucode* uc) {
	(void) uc;
}

// ---- The interpreter (arMLEt's hle_rsp.cpp): the microcode itself on the images, the vector unit
// and its loads and stores armN64's own (rsp_vu.c, rsp.c), the SP and DP registers and the DMA
// through the calls above.

static int interpDecline(struct hle_task* t, const char* why, uint32_t w, unsigned pc) {
	static char text[8][96];
	static unsigned k;
	char* s = text[k++ & 7];
	snprintf(s, sizeof(text[0]), "%s %08x at %03x", why, w, pc);
	return hle_decline(t, s);
}

int hle_rsp_run(struct hle_task* t, struct hle_cpu* c) {
	uint32_t* r = c->r;
	uint8_t* dm = (uint8_t*) t->dmem;
	bool pending = false, ran = false;
	unsigned ptarget = 0;
	r[0] = 0;
	for (;;) {
		if (!pending && ran && c->pc == c->stop_pc) {
			return HLE_STOP;
		}
		if (++c->steps > c->max_steps) {
			return interpDecline(t, "interpreter step limit", 0, c->pc);
		}
		ran = true;
		uint32_t const w = t->imem[(c->pc >> 2) & 1023];
		unsigned const op = w >> 26, rs = (w >> 21) & 31, rt = (w >> 16) & 31, rd = (w >> 11) & 31, sa = (w >> 6) & 31;
		int32_t const simm = (int16_t) w;
		uint32_t const uimm = w & 0xFFFF;
		unsigned const next = (c->pc + 4) & 0xFFC;
		bool branch = false, taken = false;
		unsigned target = 0;
#define SET(n, v) do { if (n) r[n] = (v); } while (0)
#define BR(cond) do { branch = true; taken = (cond); target = (c->pc + 4 + ((unsigned) simm << 2)) & 0xFFC; } while (0)
#define DM8(a) dm[((a) & 0xFFF) ^ 3]
		if ((w >> 25) == 0x25) {
			rspVectorOp(&A.rsp, w);
		} else {
			switch (op) {
			case 0:
				switch (w & 63) {
				case 0: SET(rd, r[rt] << sa); break;
				case 2: SET(rd, r[rt] >> sa); break;
				case 3: SET(rd, (uint32_t) ((int32_t) r[rt] >> sa)); break;
				case 4: SET(rd, r[rt] << (r[rs] & 31)); break;
				case 6: SET(rd, r[rt] >> (r[rs] & 31)); break;
				case 7: SET(rd, (uint32_t) ((int32_t) r[rt] >> (r[rs] & 31))); break;
				case 8: branch = taken = true; target = r[rs] & 0xFFC; break;
				case 9: branch = taken = true; target = r[rs] & 0xFFC; SET(rd, (c->pc + 8) & 0xFFC); break;
				case 13:
					if (pending) {
						return interpDecline(t, "BREAK in a delay slot", w, c->pc);
					}
					hle_break(t, next);
					return HLE_OK;
				case 32: case 33: SET(rd, r[rs] + r[rt]); break;
				case 34: case 35: SET(rd, r[rs] - r[rt]); break;
				case 36: SET(rd, r[rs] & r[rt]); break;
				case 37: SET(rd, r[rs] | r[rt]); break;
				case 38: SET(rd, r[rs] ^ r[rt]); break;
				case 39: SET(rd, ~(r[rs] | r[rt])); break;
				case 42: SET(rd, (int32_t) r[rs] < (int32_t) r[rt]); break;
				case 43: SET(rd, r[rs] < r[rt]); break;
				default: return interpDecline(t, "unsupported SPECIAL instruction", w, c->pc);
				}
				break;
			case 1:
				switch (rt) {
				case 0: BR((int32_t) r[rs] < 0); break;
				case 1: BR((int32_t) r[rs] >= 0); break;
				case 16: { bool tk = (int32_t) r[rs] < 0; SET(31, (c->pc + 8) & 0xFFC); BR(tk); break; }
				case 17: { bool tk = (int32_t) r[rs] >= 0; SET(31, (c->pc + 8) & 0xFFC); BR(tk); break; }
				default: return interpDecline(t, "unsupported REGIMM instruction", w, c->pc);
				}
				break;
			case 2: branch = taken = true; target = (w & 0x3FF) << 2; break;
			case 3: branch = taken = true; target = (w & 0x3FF) << 2; SET(31, (c->pc + 8) & 0xFFC); break;
			case 4: BR(r[rs] == r[rt]); break;
			case 5: BR(r[rs] != r[rt]); break;
			case 6: BR((int32_t) r[rs] <= 0); break;
			case 7: BR((int32_t) r[rs] > 0); break;
			case 8: case 9: SET(rt, r[rs] + (uint32_t) simm); break;
			case 10: SET(rt, (int32_t) r[rs] < simm); break;
			case 11: SET(rt, r[rs] < (uint32_t) simm); break;
			case 12: SET(rt, r[rs] & uimm); break;
			case 13: SET(rt, r[rs] | uimm); break;
			case 14: SET(rt, r[rs] ^ uimm); break;
			case 15: SET(rt, uimm << 16); break;
			case 16:
				if (rs == 0) {
					uint32_t v;
					if ((rd & 15) == 4) {
						hle_poll_status(t, rt, &v);
					} else if ((rd & 15) == 7) {
						v = A.regs[7];
						A.regs[7] = 1;
					} else {
						v = *t->cr[rd & 15];
					}
					SET(rt, v);
				} else if (rs == 4) {
					if (hle_mtc0(t, rd, r[rt])) {
						return hle_decline(t, "an SP register write halted the RSP");
					}
				} else {
					return interpDecline(t, "unsupported COP0 instruction", w, c->pc);
				}
				break;
			case 18:
				if (rs == 0 || rs == 2 || rs == 4 || rs == 6) {
					rspCop2Regs(&A.rsp, w);
				} else {
					return interpDecline(t, "unsupported COP2 instruction", w, c->pc);
				}
				break;
			case 32: SET(rt, (uint32_t) (int32_t) (int8_t) DM8(r[rs] + simm)); break;
			case 33: { unsigned a = r[rs] + simm; SET(rt, (uint32_t) (int32_t) (int16_t) (DM8(a) << 8 | DM8(a + 1))); break; }
			case 35: { unsigned a = r[rs] + simm; SET(rt, (uint32_t) DM8(a) << 24 | DM8(a + 1) << 16 | DM8(a + 2) << 8 | DM8(a + 3)); break; }
			case 36: SET(rt, DM8(r[rs] + simm)); break;
			case 37: { unsigned a = r[rs] + simm; SET(rt, (uint32_t) (DM8(a) << 8 | DM8(a + 1))); break; }
			case 40: DM8(r[rs] + simm) = (uint8_t) r[rt]; break;
			case 41: { unsigned a = r[rs] + simm; DM8(a) = (uint8_t) (r[rt] >> 8); DM8(a + 1) = (uint8_t) r[rt]; break; }
			case 43: { unsigned a = r[rs] + simm; for (int k = 0; k < 4; ++k) DM8(a + k) = (uint8_t) (r[rt] >> (24 - 8 * k)); break; }
			case 50:
				rspVectorLoadSwapped(&A.rsp, dm, w);
				break;
			case 58:
				rspVectorStoreSwapped(&A.rsp, dm, w);
				break;
			default:
				return interpDecline(t, "unsupported instruction", w, c->pc);
			}
		}
#undef SET
#undef BR
#undef DM8
		r[0] = 0;
		if (pending) {
			if (branch) {
				return interpDecline(t, "branch in a delay slot", w, c->pc);
			}
			c->pc = ptarget;
			pending = false;
			continue;
		}
		if (branch) {
			pending = true;
			ptarget = taken ? target : (c->pc + 8) & 0xFFC;
		}
		c->pc = next;
	}
}

int hle_interp_task(struct hle_task* t) {
	struct hle_cpu c;
	hle_cpu_init(t, &c, t->start_pc, 1u << 24);
	return hle_rsp_run(t, &c);
}

// ---- Running a task

static uint32_t be32At(const uint8_t* p) {
	return (uint32_t) p[0] << 24 | (uint32_t) p[1] << 16 | (uint32_t) p[2] << 8 | p[3];
}

// The build of the task's microcode (the FNV-1a-64 of its 4 KiB text), or NULL.
static const struct hle_ucode* taskBuild(struct N64* n) {
	uint32_t addr = be32At(&n->sp.mem[0xFD0]) & (RDRAM_SIZE - 4);
	uint8_t text[0x1000];
	for (uint32_t i = 0; i < sizeof(text); ++i) {
		text[i] = n->rdram[(addr + i) & (RDRAM_SIZE - 1)];
	}
	if (!A.textKnown || A.textAddr != addr || memcmp(A.text, text, sizeof(text))) {
		memcpy(A.text, text, sizeof(text));
		A.textAddr = addr, A.textKnown = true;
		A.textId = armlet_ucode_hash(text, 0, sizeof(text));
	}
	return armlet_hle_find(A.textId);
}

static void imagesFrom(struct N64* n) {
	for (int i = 0; i < 1024; ++i) {
		A.dmem[i] = be32At(&n->sp.mem[i * 4]);
		A.imem[i] = be32At(&n->sp.mem[0x1000 + i * 4]);
	}
}

static void imagesTo(struct N64* n) {
	for (int i = 0; i < 1024; ++i) {
		for (int k = 0; k < 4; ++k) {
			n->sp.mem[i * 4 + k] = (uint8_t) (A.dmem[i] >> (24 - 8 * k));
		}
	}
	if (A.imemLoaded) {
		for (int i = 0; i < 1024; ++i) {
			for (int k = 0; k < 4; ++k) {
				n->sp.mem[0x1000 + i * 4 + k] = (uint8_t) (A.imem[i] >> (24 - 8 * k));
			}
		}
		++n->rspCodeGen;
	}
}

// The task's log through the registers in order, from RDRAM as it was before the task.
static void replay(struct N64* n) {
	for (size_t i = 0; i < A.recCount; ++i) {
		const struct Rec* r = &A.recs[i];
		if (r->kind == REC_RAM) {
			n64RdramWrite(n, r->addr, r->len);
			memcpy(n->rdram + r->addr, A.arena + r->at, r->len);
			continue;
		}
		uint32_t const addr = r->reg == 8 ? 0x04100000 : r->reg == 9 ? 0x04100004 : 0x0410000C;
		if (r->len) {
			// An XBUS list: its bytes back in DMEM while the RDP takes it.
			uint8_t keep[0x1000];
			for (uint32_t k = 0; k < r->len; ++k) {
				keep[k] = n->sp.mem[(r->addr + k) & 0xFFF];
				n->sp.mem[(r->addr + k) & 0xFFF] = A.arena[r->at + k];
			}
			busWrite32(n, addr, r->value);
			for (uint32_t k = 0; k < r->len; ++k) {
				n->sp.mem[(r->addr + k) & 0xFFF] = keep[k];
			}
		} else {
			busWrite32(n, addr, r->value);
		}
	}
}

// The task's registers: the scalars and vector unit, and the SP and DP registers by MTC0 number.
static void worldFrom(struct N64* n, struct hle_task* t, const struct hle_ucode* uc) {
	imagesFrom(n);
	A.rsp = n->rsp;
	A.regs[0] = n->sp.memAddr, A.regs[1] = n->sp.dramAddr, A.regs[2] = A.regs[3] = n->sp.length;
	A.regs[4] = A.startStatus = n->sp.status, A.regs[5] = A.regs[6] = 0, A.regs[7] = n->sp.semaphore;
	A.regs[8] = n->dp.start, A.regs[9] = n->dp.end, A.regs[10] = n->dp.current;
	A.regs[11] = n->dp.status & ~(uint32_t) DP_START_VALID;
	A.regs[12] = A.regs[13] = A.regs[14] = A.regs[15] = 0;
	A.startValid = (n->dp.status & DP_START_VALID) != 0;
	A.irq = 0;
	A.imemLoaded = false;
	for (int i = 0; i < 16; ++i) {
		A.cr[i] = &A.regs[i];
	}
	A.recCount = A.arenaUsed = A.undoCount = A.oldUsed = A.streamUsed = 0;
	A.why = NULL;
	A.broke = false;
	A.listBytes = 0;
	memset(t, 0, sizeof(*t));
	t->dmem = A.dmem, t->imem = A.imem, t->rdram = n->rdram, t->cr = A.cr, t->irq = &A.irq;
	t->uc = uc, t->ctx = &A.ctx, t->start_pc = 0, t->log = 1;
}

// The finished task's effects for real: its log replayed, the images and registers.
static void commit(struct N64* n) {
	replay(n);
	imagesTo(n);
	memcpy(n->rsp.r, A.rsp.r, sizeof(n->rsp.r));
	memcpy(n->rsp.vr, A.rsp.vr, sizeof(n->rsp.vr));
	memcpy(n->rsp.accH, A.rsp.accH, sizeof(n->rsp.accH));
	memcpy(n->rsp.accM, A.rsp.accM, sizeof(n->rsp.accM));
	memcpy(n->rsp.accL, A.rsp.accL, sizeof(n->rsp.accL));
	n->rsp.vco = A.rsp.vco, n->rsp.vcc = A.rsp.vcc, n->rsp.vce = A.rsp.vce;
	n->rsp.divIn = A.rsp.divIn, n->rsp.divOut = A.rsp.divOut, n->rsp.divDp = A.rsp.divDp;
	n->sp.memAddr = A.regs[0], n->sp.dramAddr = A.regs[1], n->sp.length = A.regs[3];
	n->sp.semaphore = A.regs[7];
	n->sp.pc = A.endPc;
	n->rsp.nextPc = (A.endPc + 4) & 0xFFC;
	// The RSP shows the task running for about the time the microcode takes, then halts at its
	// break with the status it left.
	n->sp.hleDoneAt = n->cycles + 4096 + 8 * (uint64_t) A.listBytes;
	n->sp.doneSet = (A.regs[4] & ~A.startStatus) | 3;
	n->sp.doneClear = A.startStatus & ~A.regs[4];
	n->sp.doneIrq = (A.irq & 1) != 0;
}

static uint32_t be32At(const uint8_t* p);

// ---- Verify (N64ARMLETVERIFY=1): the task on the front-end, undone, then on the RSP; their RDP
// lists and DMEM compared (the build's dead ranges left out).

static uint8_t* rspStream;
static size_t rspStreamUsed, rspStreamCap;
static void (*savedTap)(struct N64*, const uint64_t*, int);

static void verifyTap(struct N64* n, const uint64_t* w, int count) {
	(void) n;
	if (!count) {
		return;
	}
	rspStream = grow(rspStream, &rspStreamCap, rspStreamUsed + (size_t) count * 8, 1);
	for (int i = 0; i < count; ++i) {
		for (int k = 0; k < 8; ++k) {
			rspStream[rspStreamUsed++] = (uint8_t) (w[i] >> (56 - 8 * k));
		}
	}
}

static bool deadByte(const struct hle_ucode* uc, unsigned a) {
	for (unsigned i = 0; i < uc->n_dead; ++i) {
		if (a >= uc->dead[i].lo && a < uc->dead[i].hi) {
			return true;
		}
	}
	return false;
}

// The commands' words compared, the second word of no-ops and syncs left out (what the microcode
// leaves there isn't the same on every path).
static long streamsDiffer(const uint8_t* a, size_t na, const uint8_t* b, size_t nb) {
	size_t const n = na < nb ? na : nb;
	for (size_t i = 0; i + 8 <= n; i += 8) {
		int const op = a[i] & 0x3F;
		bool const ignoreW1 = op == 0x00 || (op >= 0x26 && op <= 0x29);
		if (memcmp(a + i, b + i, ignoreW1 ? 1 : 8)) {
			return (long) i;
		}
	}
	return na == nb ? -1 : (long) n;
}

// What the front-end made of the task the RSP now runs.
static struct {
	bool pending, yielded;
	const struct hle_ucode* uc;
	uint32_t dmem[1024];
	uint8_t* stream;
	size_t len;
	uint64_t tasks, mismatches, skipped;
} V;

static void verifyStart(struct N64* n, const struct hle_ucode* uc) {
	V.uc = uc;
	memcpy(V.dmem, A.dmem, sizeof(V.dmem));
	free(V.stream);
	V.stream = malloc(A.streamUsed + 1);
	V.len = A.streamUsed;
	memcpy(V.stream, A.stream, V.len);
	savedTap = n->rdpTap;
	n->rdpTap = verifyTap;
	rspStreamUsed = 0;
	V.pending = true;
	V.yielded = false;
}

// The RSP reached a break: the task it ran compared with the front-end's.
void armletVerifyEnd(struct N64* n) {
	// A yield (SIG1) isn't the task's end: it goes on when the CPU resumes it, after the other tasks
	// that run meanwhile (audio).
	if (V.pending && (n->sp.status & 0x100)) {
		V.yielded = true;
	}
	if (!V.pending || (n->sp.status & 0x100) || be32At(&n->sp.mem[0xFC0]) != 1) {
		return;
	}
	V.pending = false;
	n->rdpTap = savedTap;
	++V.tasks;
	if (V.yielded) {
		++V.skipped;
		return;
	}
	char why[160] = "";
	long const at = streamsDiffer(V.stream, V.len, rspStream, rspStreamUsed);
	if (at >= 0) {
		snprintf(why, sizeof(why), "RDP lists differ at byte %ld of %zu/%zu", at, V.len, rspStreamUsed);
	} else {
		for (unsigned a = 0; a < 0x1000; ++a) {
			uint8_t const mine = ((const uint8_t*) V.dmem)[a ^ 3];
			if (!deadByte(V.uc, a) && mine != n->sp.mem[a]) {
				snprintf(why, sizeof(why), "DMEM differs at %03x (native %02x, RSP %02x)", a, mine, n->sp.mem[a]);
				break;
			}
		}
	}
	if (why[0] && V.mismatches++ < 50) {
		fprintf(stderr, "ARMLET verify task %llu (%s): %s\n", (unsigned long long) V.tasks, V.uc->name, why);
	}
	if (!(V.tasks & 63)) {
		fprintf(stderr, "ARMLET verify: %llu tasks, %llu differ, %llu yielded on the RSP (not compared)\n", (unsigned long long) V.tasks,
		        (unsigned long long) V.mismatches, (unsigned long long) V.skipped);
	}
}

// A fresh graphics task (SP_PC 0) of a build with a front-end runs on it, once the CPU's write
// that starts the RSP has been taken: true when it did (the RSP then shows it running until
// hleDoneAt).
bool armletTask(struct N64* n) {
	static int verifying = -1;
	if (verifying < 0) {
		verifying = getenv("N64ARMLETVERIFY") != NULL;
	}
	if ((n->sp.pc & 0xFFC) || V.pending) {
		return false;
	}
	if (!rsp_reciprocal_rom[0]) {
		rspReciprocalRom(rsp_reciprocal_rom);
	}
	const struct hle_ucode* uc = taskBuild(n);
	if (!uc || !uc->fam || !uc->fam->run) {
		return false;
	}
	A.n = n;
	// A yielded task resumed, or a list in DMEM, only where the front-end has the path.
	if ((be32At(&n->sp.mem[0xFC4]) & 1) && !uc->fam->yield_ok) {
		return false;
	}
	if ((n->dp.status & 1) && !(uc->flags & HLE_F_XBUS_OK)) {
		return false;
	}
	struct hle_task t;
	worldFrom(n, &t, uc);
	hle_bind(&t, uc);
	int rc = uc->fam->run(&t, uc);
	if (rc == HLE_OK && !A.broke) {
		A.why = "the front-end returned without ending the task";
		rc = HLE_DECLINE;
	}
	undoAll();
	if (rc != HLE_OK) {
		++n->stats.gfxArmletDeclined;
		snprintf(n->stats.gfxArmletWhy, sizeof(n->stats.gfxArmletWhy), "%s", A.why ? A.why : "(no reason given)");
		return false;
	}
	++n->stats.gfxArmlet;
	if (verifying) {
		// The RSP runs the task, its lists tapped, and armletVerifyEnd compares at its break.
		verifyStart(n, uc);
		return false;
	}
	commit(n);
	return true;
}
