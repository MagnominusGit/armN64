/* Copyright (c) 2026 Magnominus
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Licensed under the GNU General Public License, version 2 or (at your option) any later version: see LICENSE. */
#ifndef ARMULATORS_OPTIONS_H
#define ARMULATORS_OPTIONS_H

// A core's options: given to the frontend as definitions it can show and hide, or to an older one as
// plain variables ("Label; default|other|..."), and their values read back.

#include "libretro.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// A value an option can have and the number a core works with for it.
struct OptionChoice {
	const char* value;
	int number;
};

// Gives the frontend the options. Returns true when it took them as definitions (version 2), false
// when it took them as variables.
bool optionsGive(retro_environment_t environment, struct retro_core_options_v2* options);
// The option's value, NULL when the frontend has none.
const char* optionValue(retro_environment_t environment, const char* key);
bool optionIs(retro_environment_t environment, const char* key, const char* wanted);
// The number for the option's value, or fallback when the value isn't one of the choices.
int optionChoice(retro_environment_t environment, const char* key, const struct OptionChoice* choices, size_t count, int fallback);

#ifdef __cplusplus
}
#endif

#endif
