/* Copyright (c) 2026 Magnominus
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Licensed under the GNU General Public License, version 2 or (at your option) any later version: see LICENSE. */
#ifndef ARMN64_LIBRETRO_GAMES_H
#define ARMN64_LIBRETRO_GAMES_H

#include <stdbool.h>
#include <stddef.h>

enum { GAME_PATH = 4096 };

struct ListedGame {
	// The file, and its name as shown: the file name without its extension.
	char* path;
	char* label;
};

// Whether a file is a cheat cartridge's (a GameShark or Action Replay), by its name.
bool gameIsCheatCart(const char* path);
// The N64 games in the loaded file's folder but cheat cartridges, those whose save in saveDirectory
// (NULL: the folder) was written last first, then by name; at most max. Free with gamesFree.
size_t gamesBeside(const char* loadedPath, const char* saveDirectory, struct ListedGame* out, size_t max);
void gamesFree(struct ListedGame* games, size_t count);
// Reads a game, out of a .zip (its first .z64, .n64 or .v64) when built with zlib. NULL if it can't.
void* gameRead(const char* path, size_t* size);
// A game's file name without its folder or extension, as its save is named; and the file itself (a
// frontend names a game in an archive "game.zip#game.z64").
void gameName(const char* path, char* out, size_t size);
void gameFile(const char* path, char* out, size_t size);

#endif
