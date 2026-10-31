// armN64's stand-in for arMLEt's plugin context (src/armlet.c): what the imported front-ends use of
// it, a log and the configuration (none: every registered build may run).
#ifndef ARMLET_H
#define ARMLET_H

#include <stdbool.h>
#include <stdint.h>

enum { M64MSG_ERROR = 1, M64MSG_WARNING, M64MSG_INFO, M64MSG_STATUS, M64MSG_VERBOSE };

struct armlet_config {
	int unused;
};

struct armlet_ctx {
	struct armlet_config cfg;
};

void armlet_log(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));

#endif
