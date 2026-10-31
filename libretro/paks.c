// Controller Paks: the game's own (four, in its save file) and the shared cards, which port has which,
// and the manager screen. A pak's notes follow libultra's layout: the index of 128 pages at 0x100
// (each a page's next: 1 ends a note, 3 is free; a copy at 0x200, and byte 1 the sum of the user
// pages' entries), the 16 notes' entries at 0x300 (game code, publisher, first page, extension and
// name in the N64's character set), and the pages from 5 on.
#include "paks.h"

#include "libretro.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "stb_truetype.h"
#pragma GCC diagnostic pop

#include "paks_font.inc"

enum { PAK_SIZE = 0x8000, PAGES = 128, FIRST_PAGE = 5, NOTES = 16, PAKS = 4 + PAK_CARDS };

static uint8_t gamePaks[4][PAK_SIZE];
static uint8_t cards[PAK_CARDS][PAK_SIZE];
static bool cardLoaded[PAK_CARDS];
static int plugged[4];
static char cardDir[4096];

void paksInit(const char* dir) {
	snprintf(cardDir, sizeof(cardDir), "%s/armN64/ControllerPak", dir ? dir : ".");
	memset(cardLoaded, 0, sizeof(cardLoaded));
	memset(plugged, 0, sizeof(plugged));
}

static void makeDirs(const char* path) {
	char p[4200];
	snprintf(p, sizeof(p), "%s", path);
	for (char* c = p + 1; *c; ++c) {
		if (*c == '/') {
			*c = 0;
			mkdir(p, 0755);
			*c = '/';
		}
	}
	mkdir(p, 0755);
}

static void cardPath(int card, char* out, size_t size) {
	snprintf(out, size, "%s/Card %d.mpk", cardDir, card + 1);
}

static bool empty(const uint8_t* pak) {
	for (size_t i = 0; i < PAK_SIZE; ++i) {
		if (pak[i]) {
			return false;
		}
	}
	return true;
}

static uint8_t* card(int i) {
	if (!cardLoaded[i]) {
		char path[4200];
		cardPath(i, path, sizeof(path));
		FILE* f = fopen(path, "rb");
		size_t got = f ? fread(cards[i], 1, PAK_SIZE, f) : 0;
		if (f) {
			fclose(f);
		}
		if (got != PAK_SIZE) {
			pakFormat(cards[i]);
		}
		cardLoaded[i] = true;
	}
	return cards[i];
}

static void cardSave(int i) {
	char path[4200];
	makeDirs(cardDir);
	cardPath(i, path, sizeof(path));
	FILE* f = fopen(path, "wb");
	if (f) {
		fwrite(cards[i], 1, PAK_SIZE, f);
		fclose(f);
	}
}

// A port's pak as kept: the game's own or a card.
static uint8_t* source(int port) {
	return plugged[port] ? card(plugged[port] - 1) : gamePaks[port];
}

// The kept pak changed by the manager: into the machine for the ports that have it, and saved.
static void kept(struct N64* n, const uint8_t* pak) {
	for (int port = 0; port < 4; ++port) {
		if (source(port) == pak) {
			memcpy(n->paks[port], pak, PAK_SIZE);
		}
	}
	for (int i = 0; i < PAK_CARDS; ++i) {
		if (cards[i] == pak) {
			cardSave(i);
		}
	}
	n->saveDirty = true;
}

void paksImport(struct N64* n, const uint8_t* file) {
	memcpy(gamePaks, file, sizeof(gamePaks));
	// A pak left empty (by a save file from before there were any) is formatted.
	for (int port = 0; port < 4; ++port) {
		if (empty(gamePaks[port])) {
			pakFormat(gamePaks[port]);
		}
		memcpy(n->paks[port], source(port), PAK_SIZE);
	}
}

void paksSync(struct N64* n) {
	for (int port = 0; port < 4; ++port) {
		uint8_t* to = source(port);
		if (memcmp(to, n->paks[port], PAK_SIZE)) {
			memcpy(to, n->paks[port], PAK_SIZE);
			if (plugged[port]) {
				cardSave(plugged[port] - 1);
			}
		}
	}
}

void paksExport(struct N64* n, uint8_t* file) {
	for (int port = 0; port < 4; ++port) {
		if (!plugged[port]) {
			memcpy(gamePaks[port], n->paks[port], PAK_SIZE);
		}
	}
	memcpy(file, gamePaks, sizeof(gamePaks));
}

void paksPlug(struct N64* n, const int wanted[4]) {
	for (int port = 0; port < 4; ++port) {
		int const c = wanted[port] < 0 || wanted[port] > PAK_CARDS ? 0 : wanted[port];
		if (c != plugged[port]) {
			paksSync(n);
			plugged[port] = c;
			memcpy(n->paks[port], source(port), PAK_SIZE);
		}
	}
}

// ---- Notes

static int pageNext(const uint8_t* pak, int page) {
	return pak[0x100 + page * 2] << 8 | pak[0x101 + page * 2];
}

static void pageSet(uint8_t* pak, int page, int next) {
	pak[0x100 + page * 2] = (uint8_t) (next >> 8);
	pak[0x101 + page * 2] = (uint8_t) next;
}

// The index's sum, and its copy.
static void indexDone(uint8_t* pak) {
	uint8_t sum = 0;
	for (int i = FIRST_PAGE * 2; i < PAGES * 2; ++i) {
		sum = (uint8_t) (sum + pak[0x100 + i]);
	}
	pak[0x101] = sum;
	memcpy(&pak[0x200], &pak[0x100], 0x100);
}

static int freePages(const uint8_t* pak) {
	int count = 0;
	for (int p = FIRST_PAGE; p < PAGES; ++p) {
		count += pageNext(pak, p) == 3;
	}
	return count;
}

static uint8_t* entry(uint8_t* pak, int note) {
	return &pak[0x300 + note * 32];
}

static int firstPage(const uint8_t* e) {
	return e[6] << 8 | e[7];
}

static bool used(uint8_t* pak, int note) {
	const uint8_t* e = entry(pak, note);
	int const start = firstPage(e);
	return start >= FIRST_PAGE && start < PAGES && (e[0] | e[1] | e[2] | e[3] | e[4] | e[5]);
}

// A note's pages in order (at most PAGES); 0 when its chain is broken.
static int notePages(uint8_t* pak, int note, int* pages) {
	int count = 0;
	for (int p = firstPage(entry(pak, note)); count < PAGES; p = pageNext(pak, p)) {
		if (p < FIRST_PAGE || p >= PAGES) {
			return 0;
		}
		pages[count++] = p;
		if (pageNext(pak, p) == 1) {
			return count;
		}
	}
	return 0;
}

// A note's name (and extension) in ASCII from the N64's characters (others as '?').
static char noteChar(int c) {
	static const char symbols[] = "!\"#'*+,-./:=?@";
	return c == 0 || c == 0x0F ? ' ' : c >= 0x10 && c <= 0x19 ? (char) ('0' + c - 0x10) : c >= 0x1A && c <= 0x33 ? (char) ('A' + c - 0x1A) :
	       c >= 0x34 && c <= 0x41 ? symbols[c - 0x34] : '?';
}

static void noteName(const uint8_t* e, char* out, size_t size) {
	char text[24];
	int k = 0;
	for (int i = 0; i < 16; ++i) {
		text[k++] = noteChar(e[16 + i]);
	}
	while (k && text[k - 1] == ' ') {
		--k;
	}
	if (e[12]) {
		text[k++] = '.';
		for (int i = 0; i < 4 && e[12 + i]; ++i) {
			text[k++] = noteChar(e[12 + i]);
		}
	}
	text[k] = 0;
	snprintf(out, size, "%s", k ? text : "(no name)");
}

static bool sameNote(const uint8_t* a, const uint8_t* b) {
	return !memcmp(a, b, 6) && !memcmp(a + 12, b + 12, 20);
}

// Puts a note (its entry, its pages' bytes) on a pak: false with why.
static bool notePut(uint8_t* pak, const uint8_t* e, const uint8_t* data, int count, const char** why) {
	int slot = -1;
	for (int i = 0; i < NOTES; ++i) {
		if (used(pak, i) && sameNote(entry(pak, i), e)) {
			*why = "That pak already has this note";
			return false;
		}
		if (!used(pak, i) && slot < 0) {
			slot = i;
		}
	}
	if (slot < 0) {
		*why = "That pak has no free note (16 at most)";
		return false;
	}
	if (count < 1 || freePages(pak) < count) {
		*why = "Not enough free pages on that pak";
		return false;
	}
	int pages[PAGES], k = 0;
	for (int p = FIRST_PAGE; p < PAGES && k < count; ++p) {
		if (pageNext(pak, p) == 3) {
			pages[k++] = p;
		}
	}
	for (int i = 0; i < count; ++i) {
		memcpy(&pak[pages[i] * 256], data + i * 256, 256);
		pageSet(pak, pages[i], i + 1 < count ? pages[i + 1] : 1);
	}
	indexDone(pak);
	uint8_t* to = entry(pak, slot);
	memcpy(to, e, 32);
	to[6] = (uint8_t) (pages[0] >> 8), to[7] = (uint8_t) pages[0];
	return true;
}

static void noteDelete(uint8_t* pak, int note) {
	int pages[PAGES];
	int const count = notePages(pak, note, pages);
	for (int i = 0; i < count; ++i) {
		pageSet(pak, pages[i], 3);
	}
	indexDone(pak);
	memset(entry(pak, note), 0, 32);
}

// ---- The screen

enum { WIDTH = 640, HEIGHT = 480, ROW = 26, ROWS = 11, FILES = 64 };
enum { MODE_LIST, MODE_DELETE, MODE_FORMAT, MODE_IMPORT };

static struct {
	bool open, fontReady;
	int mode, focus;
	int pak[2], row[2];
	char status[128];
	uint32_t last;
	int held;
	char files[FILES][256];
	int fileCount, fileRow;
	stbtt_bakedchar large[95], small[95];
	uint8_t largeAtlas[512 * 256], smallAtlas[512 * 256];
} m = { .pak = { 0, 4 } };

static uint8_t* pakData(int i) {
	return i < 4 ? gamePaks[i] : card(i - 4);
}

static void pakLabel(int i, char* out, size_t size) {
	if (i < 4) {
		snprintf(out, size, "Player %d's game pak%s", i + 1, plugged[i] ? "" : " (in use)");
		return;
	}
	int port = -1;
	for (int k = 0; k < 4 && port < 0; ++k) {
		port = plugged[k] == i - 3 ? k : -1;
	}
	char where[24] = "";
	if (port >= 0) {
		snprintf(where, sizeof(where), " (Player %d)", port + 1);
	}
	snprintf(out, size, "Shared card %d%s", i - 3, where);
}

// The notes a pak has, in table order.
static int notesOf(uint8_t* pak, int* list) {
	int count = 0;
	for (int i = 0; i < NOTES; ++i) {
		if (used(pak, i)) {
			list[count++] = i;
		}
	}
	return count;
}

static void fill(uint32_t* pic, int x, int y, int w, int h, uint32_t color) {
	for (int j = y < 0 ? 0 : y; j < y + h && j < HEIGHT; ++j) {
		for (int i = x < 0 ? 0 : x; i < x + w && i < WIDTH; ++i) {
			pic[j * WIDTH + i] = color;
		}
	}
}

static int text(uint32_t* pic, int x, int y, bool large, uint32_t color, const char* s) {
	const stbtt_bakedchar* chars = large ? m.large : m.small;
	const uint8_t* atlas = large ? m.largeAtlas : m.smallAtlas;
	float fx = (float) x, fy = (float) y;
	for (; *s; ++s) {
		int c = (unsigned char) *s;
		if (c < 32 || c > 126) {
			c = '?';
		}
		stbtt_aligned_quad q;
		stbtt_GetBakedQuad(chars, 512, 256, c - 32, &fx, &fy, &q, 1);
		for (int j = (int) q.y0; j < (int) q.y1; ++j) {
			for (int i = (int) q.x0; i < (int) q.x1; ++i) {
				if (i < 0 || j < 0 || i >= WIDTH || j >= HEIGHT) {
					continue;
				}
				int const a = atlas[(int) (q.t0 * 256 + (float) (j - (int) q.y0)) * 512 + (int) (q.s0 * 512 + (float) (i - (int) q.x0))];
				uint32_t* p = &pic[j * WIDTH + i];
				uint32_t r = ((*p >> 16 & 255) * (255 - a) + (color >> 16 & 255) * a) / 255;
				uint32_t g = ((*p >> 8 & 255) * (255 - a) + (color >> 8 & 255) * a) / 255;
				uint32_t b = ((*p & 255) * (255 - a) + (color & 255) * a) / 255;
				*p = r << 16 | g << 8 | b;
			}
		}
	}
	return (int) fx - x;
}

static int textWidth(bool large, const char* s) {
	const stbtt_bakedchar* chars = large ? m.large : m.small;
	float w = 0;
	for (; *s; ++s) {
		int c = (unsigned char) *s;
		w += chars[(c < 32 || c > 126 ? '?' : c) - 32].xadvance;
	}
	return (int) w;
}

static void say(const char* s) {
	snprintf(m.status, sizeof(m.status), "%s", s);
}

void paksManagerOpen(void) {
	m.open = true;
	m.mode = MODE_LIST;
	m.status[0] = 0;
	m.last = ~0u;
}

static void notesDir(char* out, size_t size) {
	snprintf(out, size, "%s/notes", cardDir);
}

static void listFiles(void) {
	char dir[4200];
	notesDir(dir, sizeof(dir));
	m.fileCount = 0, m.fileRow = 0;
	DIR* d = opendir(dir);
	struct dirent* e;
	while (d && (e = readdir(d)) && m.fileCount < FILES) {
		size_t const len = strlen(e->d_name);
		if (len > 5 && !strcmp(e->d_name + len - 5, ".note")) {
			snprintf(m.files[m.fileCount++], sizeof(m.files[0]), "%s", e->d_name);
		}
	}
	if (d) {
		closedir(d);
	}
}

static void exportNote(uint8_t* pak, int note) {
	int pages[PAGES];
	int const count = notePages(pak, note, pages);
	if (!count) {
		say("This note's pages are broken: not exported");
		return;
	}
	const uint8_t* e = entry(pak, note);
	char name[40], dir[4200], path[4600];
	noteName(e, name, sizeof(name));
	for (char* c = name; *c; ++c) {
		*c = *c == '/' || *c == '\\' || *c == ':' || *c == '"' || *c == '?' || *c == '*' ? '_' : *c;
	}
	notesDir(dir, sizeof(dir));
	makeDirs(dir);
	char code[5] = { (char) e[0], (char) e[1], (char) e[2], (char) e[3], 0 };
	for (int i = 0; i < 4; ++i) {
		code[i] = code[i] >= 'A' && code[i] <= 'Z' ? code[i] : code[i] >= '0' && code[i] <= '9' ? code[i] : '_';
	}
	snprintf(path, sizeof(path), "%s/%s %s.note", dir, code, name);
	FILE* f = fopen(path, "wb");
	if (!f) {
		say("Can't write the note file");
		return;
	}
	fwrite(e, 1, 32, f);
	for (int i = 0; i < count; ++i) {
		fwrite(&pak[pages[i] * 256], 1, 256, f);
	}
	fclose(f);
	snprintf(m.status, sizeof(m.status), "Exported to ControllerPak/notes/%s %s.note", code, name);
}

static void importNote(struct N64* n, uint8_t* pak, const char* file) {
	char dir[4200], path[4600];
	notesDir(dir, sizeof(dir));
	snprintf(path, sizeof(path), "%s/%s", dir, file);
	FILE* f = fopen(path, "rb");
	static uint8_t data[32 + PAGES * 256];
	size_t const size = f ? fread(data, 1, sizeof(data), f) : 0;
	if (f) {
		fclose(f);
	}
	if (size < 32 + 256 || (size - 32) % 256) {
		say("That file isn't a note (32 bytes, then whole pages)");
		return;
	}
	const char* why = NULL;
	if (!notePut(pak, data, data + 32, (int) ((size - 32) / 256), &why)) {
		say(why);
		return;
	}
	kept(n, pak);
	say("Imported");
}

static void act(struct N64* n, uint32_t pressed) {
	enum { B = 1 << 0, Y = 1 << 1, SELECT = 1 << 2, START = 1 << 3, UP = 1 << 4, DOWN = 1 << 5, LEFT = 1 << 6, RIGHT = 1 << 7, A = 1 << 8, X = 1 << 9,
		   L = 1 << 10, R = 1 << 11 };
	int const f = m.focus;
	uint8_t* pak = pakData(m.pak[f]);
	int list[NOTES];
	int const count = notesOf(pak, list);
	if (m.row[f] >= count) {
		m.row[f] = count ? count - 1 : 0;
	}
	if (m.mode == MODE_DELETE || m.mode == MODE_FORMAT) {
		if (pressed & A) {
			if (m.mode == MODE_DELETE && count) {
				noteDelete(pak, list[m.row[f]]);
				say("Deleted");
			} else if (m.mode == MODE_FORMAT) {
				pakFormat(pak);
				say("Formatted");
			}
			kept(n, pak);
			m.mode = MODE_LIST;
		} else if (pressed & B) {
			m.mode = MODE_LIST;
			say("");
		}
		return;
	}
	if (m.mode == MODE_IMPORT) {
		if ((pressed & UP) && m.fileRow > 0) {
			--m.fileRow;
		} else if ((pressed & DOWN) && m.fileRow + 1 < m.fileCount) {
			++m.fileRow;
		} else if ((pressed & A) && m.fileCount) {
			importNote(n, pak, m.files[m.fileRow]);
			m.mode = MODE_LIST;
		} else if (pressed & B) {
			m.mode = MODE_LIST;
		}
		return;
	}
	if (pressed & B) {
		m.open = false;
	} else if ((pressed & UP) && m.row[f] > 0) {
		--m.row[f];
	} else if ((pressed & DOWN) && m.row[f] + 1 < count) {
		++m.row[f];
	} else if (pressed & (LEFT | RIGHT)) {
		m.focus = pressed & LEFT ? 0 : 1;
	} else if (pressed & (L | R)) {
		m.pak[f] = (m.pak[f] + (pressed & L ? PAKS - 1 : 1)) % PAKS;
		m.row[f] = 0;
		say("");
	} else if ((pressed & A) && count) {
		uint8_t* other = pakData(m.pak[1 - f]);
		int pages[PAGES];
		int const n2 = notePages(pak, list[m.row[f]], pages);
		static uint8_t data[PAGES * 256];
		for (int i = 0; i < n2; ++i) {
			memcpy(&data[i * 256], &pak[pages[i] * 256], 256);
		}
		const char* why = NULL;
		if (other == pak) {
			say("Choose another pak on the other side first");
		} else if (!n2) {
			say("This note's pages are broken: not copied");
		} else if (!notePut(other, entry(pak, list[m.row[f]]), data, n2, &why)) {
			say(why);
		} else {
			kept(n, other);
			say("Copied");
		}
	} else if ((pressed & X) && count) {
		m.mode = MODE_DELETE;
	} else if ((pressed & Y) && count) {
		exportNote(pak, list[m.row[f]]);
	} else if (pressed & SELECT) {
		listFiles();
		if (m.fileCount) {
			m.mode = MODE_IMPORT;
		} else {
			say("No .note files in saves/armN64/ControllerPak/notes");
		}
	} else if (pressed & START) {
		m.mode = MODE_FORMAT;
	}
}

static void drawPane(uint32_t* pic, int side) {
	int const x = side ? 328 : 16, w = 296;
	bool const focused = m.focus == side && m.mode != MODE_IMPORT;
	uint8_t* pak = pakData(m.pak[side]);
	fill(pic, x, 52, w, 352, focused ? 0x1E2A3A : 0x181C22);
	if (focused) {
		fill(pic, x, 52, w, 2, 0x4A90E2);
	}
	char label[64], line[96];
	pakLabel(m.pak[side], label, sizeof(label));
	snprintf(line, sizeof(line), "%s%s%s", focused ? "< " : "", label, focused ? " >" : "");
	text(pic, x + (w - textWidth(false, line)) / 2, 76, false, focused ? 0xFFFFFF : 0xA0A8B4, line);
	int list[NOTES];
	int const count = notesOf(pak, list);
	if (!count) {
		text(pic, x + 12, 116, false, 0x707884, "No notes");
	}
	int const first = m.row[side] >= ROWS ? m.row[side] - ROWS + 1 : 0;
	for (int k = first; k < count && k < first + ROWS; ++k) {
		int const y = 92 + (k - first) * ROW;
		uint8_t* e = entry(pak, list[k]);
		int pages[PAGES];
		if (k == m.row[side] && focused) {
			fill(pic, x + 4, y, w - 8, ROW - 2, 0x2E4A6E);
		}
		char name[40], info[32];
		noteName(e, name, sizeof(name));
		snprintf(info, sizeof(info), "%c%c%c%c %d p", e[0] >= 32 && e[0] < 127 ? e[0] : '?', e[1] >= 32 && e[1] < 127 ? e[1] : '?',
		         e[2] >= 32 && e[2] < 127 ? e[2] : '?', e[3] >= 32 && e[3] < 127 ? e[3] : '?', notePages(pak, list[k], pages));
		text(pic, x + 10, y + 18, false, 0xE8ECF0, name);
		text(pic, x + w - 10 - textWidth(false, info), y + 18, false, 0x8C96A4, info);
	}
	snprintf(line, sizeof(line), "%d of %d pages free", freePages(pak), PAGES - FIRST_PAGE);
	text(pic, x + 12, 394, false, 0x8C96A4, line);
}

bool paksManager(struct N64* n, uint32_t buttons, uint32_t* pic, int* width, int* height) {
	if (!m.open) {
		return false;
	}
	if (!m.fontReady) {
		stbtt_BakeFontBitmap(fontInterMedium, 0, 26.0f, m.largeAtlas, 512, 256, 32, 95, m.large);
		stbtt_BakeFontBitmap(fontInterMedium, 0, 17.0f, m.smallAtlas, 512, 256, 32, 95, m.small);
		m.fontReady = true;
	}
	if (m.last == ~0u) {
		// Opened: what the game wrote so far, and nothing taken for a press until all are let go.
		paksSync(n);
		m.last = buttons;
	}
	// A press, and a held direction again every few fields.
	uint32_t pressed = buttons & ~m.last;
	uint32_t const dirs = buttons & 0xF0;
	m.held = dirs && dirs == (m.last & 0xF0) ? m.held + 1 : 0;
	if (m.held > 18 && m.held % 4 == 0) {
		pressed |= dirs;
	}
	m.last = buttons;
	if (pressed) {
		act(n, pressed);
	}
	fill(pic, 0, 0, WIDTH, HEIGHT, 0x0E1116);
	text(pic, 16, 36, true, 0xFFFFFF, "Controller Pak Manager");
	drawPane(pic, 0);
	drawPane(pic, 1);
	const char* help = "A copy across   X delete   Y export   SELECT import   START format   L/R pak   B close";
	if (m.mode == MODE_DELETE) {
		help = "Delete this note?   A yes   B no";
	} else if (m.mode == MODE_FORMAT) {
		help = "Format this pak and lose every note on it?   A yes   B no";
	} else if (m.mode == MODE_IMPORT) {
		help = "Import into the selected pak:   A import   B cancel";
		fill(pic, 96, 70, 448, 330, 0x1E2A3A);
		fill(pic, 96, 70, 448, 2, 0x4A90E2);
		int const first = m.fileRow >= 11 ? m.fileRow - 10 : 0;
		for (int k = first; k < m.fileCount && k < first + 11; ++k) {
			int const y = 82 + (k - first) * ROW;
			if (k == m.fileRow) {
				fill(pic, 100, y, 440, ROW - 2, 0x2E4A6E);
			}
			text(pic, 110, y + 18, false, 0xE8ECF0, m.files[k]);
		}
	}
	text(pic, 16, 432, false, m.mode == MODE_LIST ? 0x9FD18B : 0xF2C46D, m.status[0] ? m.status : " ");
	text(pic, 16, 462, false, 0xA0A8B4, help);
	*width = WIDTH, *height = HEIGHT;
	return m.open;
}
