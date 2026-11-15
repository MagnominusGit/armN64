/* Copyright (c) 2026 Magnominus
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Licensed under the GNU General Public License, version 2 or (at your option) any later version: see LICENSE. */

// The games beside a cheat cartridge, to put one on top of it: listed, most recently played first, and
// read, out of a .zip too.

#define _GNU_SOURCE
#include "games.h"

// Paths are cut at 4096 bytes, longer than any folder RetroArch uses.
#pragma GCC diagnostic ignored "-Wformat-truncation"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

#ifdef ARMN64_ZIP
#include <zlib.h>
#endif

// A frontend names a game in an archive "game.zip#game.z64": the archive is the file.
void gameFile(const char* path, char* out, size_t size) {
	snprintf(out, size, "%s", path);
	char* hash = strchr(out, '#');
	if (hash) {
		*hash = '\0';
	}
}

void gameName(const char* path, char* out, size_t size) {
	char file[GAME_PATH];
	gameFile(path, file, sizeof(file));
	const char* slash = strrchr(file, '/');
	snprintf(out, size, "%s", slash ? slash + 1 : file);
	char* dot = strrchr(out, '.');
	if (dot && dot != out) {
		*dot = '\0';
	}
}

static bool hasExtension(const char* name, const char* const* extensions) {
	const char* dot = strrchr(name, '.');
	for (size_t i = 0; dot && extensions[i]; ++i) {
		if (!strcasecmp(dot + 1, extensions[i])) {
			return true;
		}
	}
	return false;
}

bool gameIsCheatCart(const char* path) {
	static const char* const names[] = { "GameShark", "Game Shark", "Action Replay", "Equalizer", "Game Buster" };
	for (size_t i = 0; i < sizeof(names) / sizeof(*names); ++i) {
		if (strcasestr(path, names[i])) {
			return true;
		}
	}
	return false;
}

struct Ranked {
	struct ListedGame game;
	long long when;
};

static int byWhen(const void* a, const void* b) {
	const struct Ranked* x = a;
	const struct Ranked* y = b;
	if (x->when != y->when) {
		return x->when < y->when ? 1 : -1;
	}
	return strcasecmp(x->game.label, y->game.label);
}

size_t gamesBeside(const char* loadedPath, const char* saveDirectory, struct ListedGame* out, size_t max) {
	static const char* const extensions[] = { "z64", "n64", "v64",
#ifdef ARMN64_ZIP
		"zip",
#endif
		NULL };
	char folder[GAME_PATH];
	gameFile(loadedPath, folder, sizeof(folder));
	char* slash = strrchr(folder, '/');
	if (slash) {
		*slash = '\0';
	} else {
		snprintf(folder, sizeof(folder), ".");
	}
	DIR* d = opendir(folder);
	if (!d) {
		return 0;
	}
	size_t count = 0, capacity = 0;
	struct Ranked* all = NULL;
	for (struct dirent* e; (e = readdir(d));) {
		if (e->d_name[0] == '.' || !hasExtension(e->d_name, extensions) || gameIsCheatCart(e->d_name)) {
			continue;
		}
		if (count == capacity) {
			capacity = capacity ? capacity * 2 : 64;
			struct Ranked* more = realloc(all, capacity * sizeof(*all));
			if (!more) {
				break;
			}
			all = more;
		}
		struct Ranked* r = &all[count];
		char path[GAME_PATH], label[GAME_PATH], save[GAME_PATH * 2];
		snprintf(path, sizeof(path), "%s/%s", folder, e->d_name);
		gameName(path, label, sizeof(label));
		snprintf(save, sizeof(save), "%s/%s.srm", saveDirectory ? saveDirectory : folder, label);
		struct stat info;
		r->when = stat(save, &info) ? 0 : (long long) info.st_mtime;
		r->game.path = strdup(path);
		r->game.label = strdup(label);
		if (r->game.path && r->game.label) {
			++count;
		} else {
			free(r->game.path);
			free(r->game.label);
		}
	}
	closedir(d);
	if (count) {
		qsort(all, count, sizeof(*all), byWhen);
	}
	size_t const kept = count < max ? count : max;
	for (size_t i = 0; i < count; ++i) {
		if (i < kept) {
			out[i] = all[i].game;
		} else {
			free(all[i].game.path);
			free(all[i].game.label);
		}
	}
	free(all);
	return kept;
}

void gamesFree(struct ListedGame* games, size_t count) {
	for (size_t i = 0; i < count; ++i) {
		free(games[i].path);
		free(games[i].label);
	}
}

static uint8_t* readWhole(const char* path, size_t* size) {
	*size = 0;
	FILE* f = fopen(path, "rb");
	if (!f) {
		return NULL;
	}
	fseek(f, 0, SEEK_END);
	long const length = ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t* data = length > 0 ? malloc((size_t) length) : NULL;
	if (data && fread(data, 1, (size_t) length, f) == (size_t) length) {
		*size = (size_t) length;
	} else {
		free(data);
		data = NULL;
	}
	fclose(f);
	return data;
}

#ifdef ARMN64_ZIP
static uint32_t le32(const uint8_t* p) {
	return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24;
}

static uint16_t le16(const uint8_t* p) {
	return (uint16_t) (p[0] | p[1] << 8);
}

// The first N64 game in a .zip, stored or deflated, by its central directory.
static uint8_t* fromZip(const uint8_t* zip, size_t zipSize, size_t* size) {
	static const char* const games[] = { "z64", "n64", "v64", NULL };
	size_t end = zipSize >= 22 ? zipSize - 22 : 0;
	while (end > 0 && le32(zip + end) != 0x06054B50) {
		--end;
	}
	if (zipSize < 22 || le32(zip + end) != 0x06054B50) {
		return NULL;
	}
	uint32_t at = le32(zip + end + 16);
	unsigned const entries = le16(zip + end + 10);
	for (unsigned i = 0; i < entries && at + 46 <= zipSize && le32(zip + at) == 0x02014B50; ++i) {
		uint16_t const method = le16(zip + at + 10), nameLength = le16(zip + at + 28);
		uint32_t const packed = le32(zip + at + 20), unpacked = le32(zip + at + 24), local = le32(zip + at + 42);
		char name[512];
		snprintf(name, sizeof(name), "%.*s", nameLength < 511 ? (int) nameLength : 511, (const char*) zip + at + 46);
		at += 46 + nameLength + le16(zip + at + 30) + le16(zip + at + 32);
		if (!hasExtension(name, games) || local + 30 > zipSize) {
			continue;
		}
		uint32_t const data = local + 30 + le16(zip + local + 26) + le16(zip + local + 28);
		if (data + (uint64_t) packed > zipSize || (method != 0 && method != 8)) {
			return NULL;
		}
		uint8_t* out = malloc(unpacked ? unpacked : 1);
		if (!out) {
			return NULL;
		}
		if (method == 0) {
			memcpy(out, zip + data, unpacked);
		} else {
			z_stream s = { 0 };
			s.next_in = (Bytef*) (zip + data), s.avail_in = packed;
			s.next_out = out, s.avail_out = unpacked;
			if (inflateInit2(&s, -MAX_WBITS) != Z_OK) {
				free(out);
				return NULL;
			}
			int const result = inflate(&s, Z_FINISH);
			inflateEnd(&s);
			if (result != Z_STREAM_END || s.total_out != unpacked) {
				free(out);
				return NULL;
			}
		}
		*size = unpacked;
		return out;
	}
	return NULL;
}
#endif

void* gameRead(const char* path, size_t* size) {
	char file[GAME_PATH];
	gameFile(path, file, sizeof(file));
	uint8_t* data = readWhole(file, size);
#ifdef ARMN64_ZIP
	static const char* const zips[] = { "zip", NULL };
	if (data && hasExtension(file, zips)) {
		size_t const zipSize = *size;
		uint8_t* game = fromZip(data, zipSize, size);
		free(data);
		return game;
	}
#endif
	return data;
}
