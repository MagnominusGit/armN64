/* Copyright (c) 2026 Magnominus
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Licensed under the GNU General Public License, version 2 or (at your option) any later version: see LICENSE. */
#include "options.h"

#include <stdio.h>
#include <string.h>

enum { VARIABLES_MOST = 64 };

// Older frontends take each option as "Label; value|value", the default first, made here from the
// definitions.
static void setVariables(retro_environment_t environment, const struct retro_core_option_v2_definition* definitions) {
	static struct retro_variable variables[VARIABLES_MOST];
	static char text[VARIABLES_MOST][1024];
	int n = 0;
	for (; definitions->key && n < VARIABLES_MOST - 1; ++definitions, ++n) {
		size_t at = (size_t) snprintf(text[n], sizeof(text[n]), "%s; %s", definitions->desc, definitions->default_value);
		for (int i = 0; definitions->values[i].value && at < sizeof(text[n]); ++i) {
			if (strcmp(definitions->values[i].value, definitions->default_value)) {
				at += (size_t) snprintf(text[n] + at, sizeof(text[n]) - at, "|%s", definitions->values[i].value);
			}
		}
		variables[n] = (struct retro_variable) { definitions->key, text[n] };
	}
	variables[n] = (struct retro_variable) { NULL, NULL };
	environment(RETRO_ENVIRONMENT_SET_VARIABLES, variables);
}

bool optionsGive(retro_environment_t environment, struct retro_core_options_v2* options) {
	unsigned version = 0;
	if (environment(RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION, &version) && version >= 2) {
		environment(RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2, options);
		return true;
	}
	setVariables(environment, options->definitions);
	return false;
}

const char* optionValue(retro_environment_t environment, const char* key) {
	struct retro_variable variable = { key, NULL };
	return environment(RETRO_ENVIRONMENT_GET_VARIABLE, &variable) ? variable.value : NULL;
}

bool optionIs(retro_environment_t environment, const char* key, const char* wanted) {
	const char* value = optionValue(environment, key);
	return value && !strcmp(value, wanted);
}

int optionChoice(retro_environment_t environment, const char* key, const struct OptionChoice* choices, size_t count, int fallback) {
	const char* value = optionValue(environment, key);
	for (size_t i = 0; value && i < count; ++i) {
		if (!strcmp(value, choices[i].value)) {
			return choices[i].number;
		}
	}
	return fallback;
}
