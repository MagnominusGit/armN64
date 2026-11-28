/* Copyright (c) 2026 Magnominus
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Licensed under the GNU General Public License, version 2 or (at your option) any later version: see LICENSE. */
#include "n64.h"
#include "rdp.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

// Save states, taken between fields: struct N64 but for compiled code's tables, then the RDRAM the
// console has (4 MiB without the Expansion Pak) and its hidden bits four to a byte, the RDP's state
// and the HLE renderer's and audio HLE's (zeros where there is none yet). Loading keeps the running
// machine's host pointers, caches and settings, throws its compiled code away, and the renderer
// threads, GPU RDP and GL renderer start again from the loaded RDP. Compiled code stays where the
// state's code is the same. Only the build that made a state loads it, with the Expansion Pak as it
// was: the header holds the sizes. Quick states (run-ahead and rollback, many a second) leave the GL
// renderer's pictures on the GPU and the renderer as it is: the machine is exact, a picture not yet
// given back to RDRAM may not be. They leave out RDRAM's hidden bits, which only the RDP's own
// drawing and the VI's filters read, when the GL renderer draws.

enum { STATE_VERSION = 2, HAS_RDP = 1, HAS_HLE = 2, HAS_AUDIO = 4 };

struct StateHeader {
	char magic[4];
	uint32_t version, machine, rdp, hle, audio, has, ram;
};

// The parts of struct N64 left out, the running machine's kept: compiled code's tables, and the HLE
// check's counts (N64HLECHECK).
#define PART(first, last) { offsetof(struct N64, first), offsetof(struct N64, last) + sizeof(((struct N64*) 0)->last) }
static const struct {
	size_t from, to;
} notSaved[] = { PART(checkCounts, checkLen), PART(blockPage, blockPage), PART(codeWords, codeWords),
	             PART(jitTable, jitTable), PART(vpageFilled, vpageFilled), PART(rspTable, rspVariants) };
#define PARTS (sizeof(notSaved) / sizeof(*notSaved))

static size_t machineSize(void) {
	size_t size = sizeof(struct N64);
	for (size_t i = 0; i < PARTS; ++i) {
		size -= notSaved[i].to - notSaved[i].from;
	}
	return size;
}

size_t n64StateSize(const struct N64* n) {
	uint32_t const ram = n ? n->ramSize : RDRAM_SIZE;
	return sizeof(struct StateHeader) + machineSize() + ram + ram / 8 + sizeof(struct RDP) + hleStateSize() + audioHleStateSize();
}

static void header(struct StateHeader* h, uint32_t has, uint32_t ram) {
	*h = (struct StateHeader) { { 'A', 'N', '6', '4' }, STATE_VERSION, sizeof(struct N64), sizeof(struct RDP), (uint32_t) hleStateSize(),
		                        (uint32_t) audioHleStateSize(), has, ram };
}

// Copies size bytes from from (or zeros) to *at and moves on.
static void put(uint8_t** at, const void* from, size_t size) {
	from ? memcpy(*at, from, size) : memset(*at, 0, size);
	*at += size;
}

// The hidden bits, a byte (0 to 3) for each halfword, four to a byte in the state.
static void packHidden(uint8_t* to, const uint8_t* from, size_t size) {
	for (size_t i = 0; i < size; ++i, from += 4) {
		to[i] = (uint8_t) (from[0] | from[1] << 2 | from[2] << 4 | from[3] << 6);
	}
}

static void unpackHidden(uint8_t* to, const uint8_t* from, size_t size) {
	for (size_t i = 0; i < size; ++i, to += 4) {
		to[0] = from[i] & 3, to[1] = from[i] >> 2 & 3, to[2] = from[i] >> 4 & 3, to[3] = from[i] >> 6;
	}
}

static bool zeros(const uint8_t* p, size_t size) {
	uint64_t any = 0;
	for (size_t i = 0; i < size; i += 8) {
		uint64_t w;
		memcpy(&w, p + i, 8);
		any |= w;
	}
	return !any;
}

// Without the Expansion Pak the state has no RDRAM above 4 MiB, memory the console doesn't have then,
// which only the RDP, the VI and the HLE renderers still reach (a game made for 4 MiB doesn't point
// them there): anything they wrote there goes as the state is taken too, so the game goes on the
// same from the state as without it.
static void clearAbove(struct N64* n, uint32_t ram) {
	if (!zeros(n->rdram + ram, RDRAM_SIZE - ram)) {
		n64RdramWrite(n, ram, RDRAM_SIZE - ram);
		memset(n->rdram + ram, 0, RDRAM_SIZE - ram);
	}
	if (!zeros(n->hidden + ram / 2, (RDRAM_SIZE - ram) / 2)) {
		memset(n->hidden + ram / 2, 0, (RDRAM_SIZE - ram) / 2);
	}
}

bool n64StateSave(struct N64* n, void* data, size_t size, bool quick) {
	if (size < n64StateSize(n)) {
		return false;
	}
	if (n->rspThread) {
		rspThreadHleJoin(n);
		rspThreadJoin(n);
	}
	// What the renderers and the GPU drew goes to RDRAM first.
	bool const keepGl = quick && n->glr;
	if (!keepGl) {
		rdpWait(n);
	}
	uint32_t const ram = n->ramSize;
	if (ram < RDRAM_SIZE) {
		clearAbove(n, ram);
	}
	uint8_t* at = data;
	struct StateHeader h;
	header(&h, (n->rdp ? HAS_RDP : 0) | (n->hle ? HAS_HLE : 0) | (n->hleAudio ? HAS_AUDIO : 0), ram);
	put(&at, &h, sizeof(h));
	size_t from = 0;
	for (size_t i = 0; i < PARTS; ++i) {
		put(&at, (const uint8_t*) n + from, notSaved[i].from - from);
		from = notSaved[i].to;
	}
	put(&at, (const uint8_t*) n + from, sizeof(*n) - from);
	put(&at, n->rdram, ram);
	if (keepGl) {
		put(&at, NULL, ram / 8);
	} else {
		packHidden(at, n->hidden, ram / 8);
		at += ram / 8;
	}
	struct RDP* rdp = n->rdp ? malloc(sizeof(*rdp)) : NULL;
	if (rdp) {
		rdpStateGet(n, rdp);
	}
	put(&at, rdp, sizeof(struct RDP));
	free(rdp);
	put(&at, n->hle, hleStateSize());
	put(&at, n->hleAudio, audioHleStateSize());
	return true;
}

bool n64StateLoad(struct N64* n, const void* data, size_t size, bool quick) {
	struct StateHeader h, want;
	if (size < n64StateSize(n)) {
		return false;
	}
	memcpy(&h, data, sizeof(h));
	header(&want, h.has, n->ramSize);
	if (memcmp(&h, &want, sizeof(h))) {
		return false;
	}
	struct N64* was = malloc(sizeof(*was));
	if (!was) {
		return false;
	}
	bool const keepGl = quick && n->glr;
	if (!keepGl) {
		rdpWait(n);
	}
	memcpy(was, n, sizeof(*was));
	const uint8_t* at = (const uint8_t*) data + sizeof(h);
	size_t from = 0;
	for (size_t i = 0; i < PARTS; ++i) {
		memcpy((uint8_t*) n + from, at, notSaved[i].from - from);
		at += notSaved[i].from - from;
		from = notSaved[i].to;
	}
	memcpy((uint8_t*) n + from, at, sizeof(*n) - from);
	at += sizeof(*n) - from;
	bool const keptData = !n->compat && !n->fast;

	// The running machine's memory, code, renderers, settings and input.
	n->rdram = was->rdram, n->hidden = was->hidden, n->rom = was->rom, n->romSize = was->romSize;
	n->cheat.flash = was->cheat.flash;
	n->rdp = was->rdp, n->hle = was->hle, n->hleAudio = was->hleAudio, n->isViewer = was->isViewer;
	n->hleCheck = was->hleCheck, n->gfxRsp = was->gfxRsp, n->hleShortTasks = was->hleShortTasks, n->fle = was->fle, n->slice = was->slice, n->accuracy = was->accuracy, n->compat = was->compat, n->opCycles = was->opCycles, n->emulatorLike = was->emulatorLike, n->rspPlainTiming = was->rspPlainTiming, n->fast = was->fast;
	n->ocWantNum = was->ocWantNum, n->ocWantDen = was->ocWantDen;
	// The GL renderer kept, its pages stay flagged for giving back what it drew.
	for (uint32_t p = 0; p < (RDRAM_SIZE >> 12); ++p) {
		n->pageFlags[p] = was->pageFlags[p] & (keepGl ? 0xFF : PAGE_CODE);
	}
	// Pending work is this machine's renderers', never the state's (another machine's, with netplay).
	n->rdpPending = keepGl && was->rdpPending;
	n->pageFlags[RDRAM_SIZE >> 12] = 0xFF;
	// The same TLB maps the same: the code for each address and the pages found stand.
	bool const sameTlb = !memcmp(n->cpu.tlb, was->cpu.tlb, sizeof(n->cpu.tlb));
	n->blockGen = was->blockGen + !sameTlb;
	n->useBlocks = was->useBlocks, n->useJit = was->useJit;
	n->jitArena = was->jitArena, n->jitUsed = was->jitUsed, n->jitEnter = was->jitEnter, n->jitLeave = was->jitLeave;
	n->jitRedispatch = was->jitRedispatch;
	n->links = was->links, n->linkCount = was->linkCount, n->linkCap = was->linkCap, n->linkGen = was->linkGen;
	n->vpages = was->vpages;
	n->useRspJit = was->useRspJit, n->rspArena = was->rspArena, n->rspUsed = was->rspUsed;
	n->rspEnter = was->rspEnter, n->rspLeave = was->rspLeave, n->rspCodeGen = was->rspCodeGen + 1;
	n->rdpTap = was->rdpTap, n->rdpThreads = was->rdpThreads, n->rdpGpu = was->rdpGpu, n->rdpGl = was->rdpGl;
	n->glr = was->glr, n->glrScale = was->glrScale;
	n->rspThread = was->rspThread, n->rspWindow = was->rspWindow;
	n->viGpu = was->viGpu, n->viFilters = was->viFilters, n->viGles = was->viGles;
	// The GPU RDP starts again (rdpStateLoad): the VI's program goes into its new context, and a
	// picture still in the old one isn't shown.
	n->viSharedProgram = was->viSharedProgram < 0 ? -1 : 0;
	n->pictureGles = was->pictureBinding == 4 ? NULL : was->pictureGles, n->pictureBinding = was->pictureBinding, n->pictureRgba = was->pictureRgba;
	n->skip.on = was->skip.on;
	n->skip.automatic = was->skip.automatic;
	n->lastPicture = was->lastPicture, n->viKept = was->viKept, n->viPool = was->viPool;
	n->lastWidth = was->lastWidth, n->lastHeight = was->lastHeight, n->settings = was->settings;
	n->audio = was->audio, n->audioCapacity = was->audioCapacity, n->audioCount = 0;
	memcpy(n->plugged, was->plugged, sizeof(n->plugged));
	memcpy(n->pak, was->pak, sizeof(n->pak));
	memcpy(n->pad, was->pad, sizeof(n->pad));
	for (int i = 0; i < 4; ++i) {
		n->tpak[i].cart.rom = was->tpak[i].cart.rom, n->tpak[i].cart.romSize = was->tpak[i].cart.romSize;
	}
	n->unimplemented = was->unimplemented, n->stats = was->stats;
	n->dd.ipl = was->dd.ipl, n->dd.disk = was->dd.disk, n->dd.diskSize = was->dd.diskSize;
	n->idle.enabled = was->idle.enabled;
	n->catchingUp = false;
	n->saveDirty = true;
	n->vpageFillCount = was->vpageFillCount;
	free(was);
	fpuHostRound(&n->cpu, (int) (n->cpu.fcr31 & 3));
	if (!sameTlb) {
		cpuTlbChanged(n);
	}
	// Code that differs in the state's RDRAM is compiled again (above the RDRAM the console has, the
	// state's is zeros).
	static const uint8_t none[4096];
	uint32_t const ram = n->ramSize;
	for (uint32_t p = 0; p < (RDRAM_SIZE >> 12); ++p) {
		if ((n->pageFlags[p] & PAGE_CODE) && memcmp(n->rdram + (p << 12), p < (ram >> 12) ? at + (p << 12) : none, 4096)) {
			blocksInvalidatePage(n, p);
		}
	}
	memcpy(n->rdram, at, ram);
	memset(n->rdram + ram, 0, RDRAM_SIZE - ram);
	memset(n->dirtyPages, 0, sizeof(n->dirtyPages));
	cpuCachesFromState(n, keptData);
	at += ram;
	if (!keepGl) {
		unpackHidden(n->hidden, at, ram / 8);
		memset(n->hidden + ram / 2, 0, (RDRAM_SIZE - ram) / 2);
	}
	at += ram / 8;
	rdpStateLoad(n, h.has & HAS_RDP ? (const struct RDP*) at : NULL, quick);
	at += sizeof(struct RDP);
	if ((h.has & HAS_HLE) && (n->hle || (n->hle = calloc(1, hleSize())))) {
		memcpy(n->hle, at, hleStateSize());
	}
	at += hleStateSize();
	if ((h.has & HAS_AUDIO) && (n->hleAudio || (n->hleAudio = malloc(audioHleStateSize())))) {
		memcpy(n->hleAudio, at, audioHleStateSize());
	}
	return true;
}
