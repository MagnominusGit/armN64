#include "n64.h"
#include "rdp.h"

#include <stdlib.h>
#include <string.h>

// Save states, taken between fields: struct N64 as one block, then RDRAM and its hidden bits, the
// RDP's state and the HLE renderer's and audio HLE's (zeros where there is none yet). Loading keeps
// the running machine's host pointers, caches and settings, throws its compiled code away, and the
// renderer threads, GPU RDP and GL renderer start again from the loaded RDP. Compiled code stays
// where the state's code is the same. Only the build that
// made a state loads it: the header holds the sizes. Quick states (run-ahead and rollback, many a
// second) leave the GL renderer's pictures on the GPU and the renderer as it is: the machine is
// exact, a picture not yet given back to RDRAM may not be. They leave out RDRAM's hidden bits, which
// only the RDP's own drawing and the VI's filters read, when the GL renderer draws.

enum { STATE_VERSION = 1, HAS_RDP = 1, HAS_HLE = 2, HAS_AUDIO = 4 };

struct StateHeader {
	char magic[4];
	uint32_t version, machine, rdp, hle, audio, has, pad;
};

size_t n64StateSize(void) {
	return sizeof(struct StateHeader) + sizeof(struct N64) + RDRAM_SIZE + RDRAM_SIZE / 2 + sizeof(struct RDP) + hleStateSize() + audioHleStateSize();
}

static void header(struct StateHeader* h, uint32_t has) {
	*h = (struct StateHeader) { { 'A', 'N', '6', '4' }, STATE_VERSION, sizeof(struct N64), sizeof(struct RDP), (uint32_t) hleStateSize(),
		                        (uint32_t) audioHleStateSize(), has, 0 };
}

// Copies size bytes from from (or zeros) to *at and moves on.
static void put(uint8_t** at, const void* from, size_t size) {
	from ? memcpy(*at, from, size) : memset(*at, 0, size);
	*at += size;
}

bool n64StateSave(struct N64* n, void* data, size_t size, bool quick) {
	if (size < n64StateSize()) {
		return false;
	}
	// What the renderers and the GPU drew goes to RDRAM first.
	bool const keepGl = quick && n->glr;
	if (!keepGl) {
		rdpWait(n);
	}
	uint8_t* at = data;
	struct StateHeader h;
	header(&h, (n->rdp ? HAS_RDP : 0) | (n->hle ? HAS_HLE : 0) | (n->hleAudio ? HAS_AUDIO : 0));
	put(&at, &h, sizeof(h));
	put(&at, n, sizeof(*n));
	put(&at, n->rdram, RDRAM_SIZE);
	put(&at, keepGl ? NULL : n->hidden, RDRAM_SIZE / 2);
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
	if (size < n64StateSize()) {
		return false;
	}
	memcpy(&h, data, sizeof(h));
	header(&want, h.has);
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
	memcpy(n, at, sizeof(*n));
	at += sizeof(*n);

	// The running machine's memory, code, renderers, settings and input.
	n->rdram = was->rdram, n->hidden = was->hidden, n->rom = was->rom, n->romSize = was->romSize;
	n->rdp = was->rdp, n->hle = was->hle, n->hleAudio = was->hleAudio, n->isViewer = was->isViewer;
	n->hleCheck = was->hleCheck, n->slice = was->slice, n->accuracy = was->accuracy, n->compat = was->compat, n->fast = was->fast;
	memcpy(n->blockPage, was->blockPage, sizeof(n->blockPage));
	memcpy(n->codeWords, was->codeWords, sizeof(n->codeWords));
	// The GL renderer kept, its pages stay flagged for giving back what it drew.
	for (uint32_t p = 0; p < (RDRAM_SIZE >> 12); ++p) {
		n->pageFlags[p] = was->pageFlags[p] & (keepGl ? 0xFF : PAGE_CODE);
	}
	if (keepGl) {
		n->rdpPending = was->rdpPending;
	}
	n->pageFlags[RDRAM_SIZE >> 12] = 0xFF;
	// The same TLB maps the same: the code for each address and the pages found stand.
	bool const sameTlb = !memcmp(n->cpu.tlb, was->cpu.tlb, sizeof(n->cpu.tlb));
	n->blockGen = was->blockGen + !sameTlb;
	n->useBlocks = was->useBlocks, n->useJit = was->useJit;
	n->jitArena = was->jitArena, n->jitUsed = was->jitUsed, n->jitEnter = was->jitEnter, n->jitLeave = was->jitLeave;
	memcpy(n->jitTable, was->jitTable, sizeof(n->jitTable));
	n->links = was->links, n->linkCount = was->linkCount, n->linkCap = was->linkCap, n->linkGen = was->linkGen;
	n->vpages = was->vpages;
	memcpy(n->vpageFilled, was->vpageFilled, sizeof(n->vpageFilled));
	n->useRspJit = was->useRspJit, n->rspArena = was->rspArena, n->rspUsed = was->rspUsed;
	n->rspEnter = was->rspEnter, n->rspLeave = was->rspLeave, n->rspCodeGen = was->rspCodeGen + 1;
	memcpy(n->rspTable, was->rspTable, sizeof(n->rspTable));
	memcpy(n->rspVariants, was->rspVariants, sizeof(n->rspVariants));
	n->rdpThreads = was->rdpThreads, n->rdpThreadsAsked = was->rdpThreadsAsked, n->rdpGpu = was->rdpGpu, n->rdpGl = was->rdpGl;
	n->glr = was->glr, n->glrScale = was->glrScale, n->glrStop = was->glrStop, n->glrStopReason = was->glrStopReason, n->glTasks = was->glTasks, n->glRspTasks = was->glRspTasks;
	n->viGpu = was->viGpu, n->viFilters = was->viFilters, n->viGles = was->viGles, n->viSharedProgram = was->viSharedProgram;
	n->pictureGles = was->pictureGles, n->pictureBinding = was->pictureBinding, n->pictureRgba = was->pictureRgba;
	n->skip.on = was->skip.on;
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
	if (!sameTlb) {
		cpuTlbChanged(n);
	}
	// Code that differs in the state's RDRAM is compiled again.
	for (uint32_t p = 0; p < (RDRAM_SIZE >> 12); ++p) {
		if ((n->pageFlags[p] & PAGE_CODE) && memcmp(n->rdram + (p << 12), at + (p << 12), 4096)) {
			blocksInvalidatePage(n, p);
		}
	}
	memcpy(n->rdram, at, RDRAM_SIZE);
	memset(n->dirtyPages, 0, sizeof(n->dirtyPages));
	at += RDRAM_SIZE;
	if (!keepGl) {
		memcpy(n->hidden, at, RDRAM_SIZE / 2);
	}
	at += RDRAM_SIZE / 2;
	rdpStateLoad(n, h.has & HAS_RDP ? (const struct RDP*) at : NULL, quick);
	at += sizeof(struct RDP);
	if ((h.has & HAS_HLE) && (n->hle || (n->hle = malloc(hleStateSize())))) {
		memcpy(n->hle, at, hleStateSize());
	}
	at += hleStateSize();
	if ((h.has & HAS_AUDIO) && (n->hleAudio || (n->hleAudio = malloc(audioHleStateSize())))) {
		memcpy(n->hleAudio, at, audioHleStateSize());
	}
	return true;
}
