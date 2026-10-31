// Controller Paks for the libretro core (paks.c): each port's pak is the game's own (kept in its save
// file) or one of the shared cards (32 KiB .mpk files in saves/armN64/ControllerPak, carried from game
// to game like a real pak), and a manager screen for the notes on them.
#pragma once

#include "n64.h"

#include <stdbool.h>
#include <stdint.h>

enum { PAK_CARDS = 8 };

// Where the shared cards are kept: dir/armN64/ControllerPak.
void paksInit(const char* dir);
// The game's own paks from the save file's four, and back.
void paksImport(struct N64* n, const uint8_t* file);
void paksExport(struct N64* n, uint8_t* file);
// What each port has: 0 the game's own pak, 1 to PAK_CARDS a shared card.
void paksPlug(struct N64* n, const int cards[4]);
// After the game wrote to a pak: into the game's paks or the cards; changed cards to their files.
void paksSync(struct N64* n);
// The manager screen for a field: buttons are RetroPad bits (1 << RETRO_DEVICE_ID_JOYPAD_*); draws
// width x height pixels (0x00RRGGBB) into picture. False once it is closed.
bool paksManager(struct N64* n, uint32_t buttons, uint32_t* picture, int* width, int* height);
void paksManagerOpen(void);
