/* Copyright (c) 2026 Magnominus
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// What armN64 adds to the shared app (app-hooks.h has what each hook is for).

#include "app-hooks.h"
#include "rc_consoles.h"

#include <string.h>

// The GPU renderers draw in the frontend's GL context, which the app doesn't share yet: the CPU
// renderer, and no row for it.
static const char* optionRead(const struct Option* option) {
	return !strcmp(option->key, "armn64_renderer") ? "cpu" : NULL;
}

static bool optionShown(const struct Option* option, bool shown) {
	return shown && strcmp(option->key, "armn64_renderer");
}

static uint32_t cheevosConsole(const uint8_t* data, size_t size) {
	return RC_CONSOLE_NINTENDO_64;
}

const struct AppHooks appHooks = {
	.optionRead = optionRead,
	.optionShown = optionShown,
	.cheevosConsole = cheevosConsole,
};
