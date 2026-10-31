/* Copyright (c) 2026 Magnominus
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// What armN64 adds to the shared app (app-hooks.h has what each hook is for).

#include "app-hooks.h"
#include "rc_consoles.h"

static uint32_t cheevosConsole(const uint8_t* data, size_t size) {
	return RC_CONSOLE_NINTENDO_64;
}

const struct AppHooks appHooks = {
	.cheevosConsole = cheevosConsole,
};
