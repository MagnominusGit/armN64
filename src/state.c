#include "n64.h"
#include "rdp.h"

#include <stdlib.h>
#include <string.h>

// Save states, taken between fields: struct N64 as one block, then RDRAM and its hidden bits, the
// RDP's state and the HLE renderer's and audio HLE's (zeros where there is none yet). Loading keeps
// the running machine's host pointers, caches and settings, throws its compiled code away, and the
// renderer threads, GPU RDP and GL renderer start again from the loaded RDP. Only the build that
// made a state loads it: the header holds the sizes.

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

bool n64StateSave(struct N64* n, void* data, size_t size) {
	if (size < n64StateSize()) {
		return false;
	}
	// What the renderers and the GPU drew goes to RDRAM first.
	rdpWait(n);
	if (n->glr) {
		glrSync(n);
	}
	uint8_t* at = data;
	struct StateHeader h;
	header(&h, (n->rdp ? HAS_RDP : 0) | (n->hle ? HAS_HLE : 0) | (n->hleAudio ? HAS_AUDIO : 0));
	put(&at, &h, sizeof(h));
	put(&at, n, sizeof(*n));
	put(&at, n->rdram, RDRAM_SIZE);
	put(&at, n->hidden, RDRAM_SIZE / 2);
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

bool n64StateLoad(struct N64* n, const void* data, size_t size) {
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
	rdpWait(n);
	blocksFree(n);
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
	memset(n->pageFlags, 0, sizeof(n->pageFlags) - 1);
	n->pageFlags[RDRAM_SIZE >> 12] = 0xFF;
	n->blockGen = was->blockGen + 1;
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
	n->glr = was->glr, n->glrStop = was->glrStop, n->glTasks = was->glTasks, n->glRspTasks = was->glRspTasks;
	n->viGpu = was->viGpu, n->viGles = was->viGles, n->viSharedProgram = was->viSharedProgram;
	n->pictureGles = was->pictureGles, n->pictureBinding = was->pictureBinding, n->pictureRgba = was->pictureRgba;
	n->skip.on = was->skip.on;
	n->lastPicture = was->lastPicture, n->viKept = was->viKept, n->viPool = was->viPool;
	n->lastWidth = was->lastWidth, n->lastHeight = was->lastHeight, n->settings = was->settings;
	n->audio = was->audio, n->audioCapacity = was->audioCapacity, n->audioCount = 0;
	memcpy(n->plugged, was->plugged, sizeof(n->plugged));
	memcpy(n->pak, was->pak, sizeof(n->pak));
	memcpy(n->pad, was->pad, sizeof(n->pad));
	n->unimplemented = was->unimplemented, n->stats = was->stats;
	n->dd.ipl = was->dd.ipl, n->dd.disk = was->dd.disk, n->dd.diskSize = was->dd.diskSize;
	n->idle.enabled = was->idle.enabled;
	n->catchingUp = false;
	n->saveDirty = true;
	free(was);
	// Every page the TLB mapped is looked up again.
	n->vpageFillCount = VPAGE_FILLED + 1;
	cpuTlbChanged(n);

	memcpy(n->rdram, at, RDRAM_SIZE);
	at += RDRAM_SIZE;
	memcpy(n->hidden, at, RDRAM_SIZE / 2);
	at += RDRAM_SIZE / 2;
	rdpStateLoad(n, h.has & HAS_RDP ? (const struct RDP*) at : NULL);
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
