#include "n64.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

// The RSP on a thread of its own (N64_RSP_THREAD, hle). At the end of each window of CPU
// time the RSP is caught up to it on the thread while the CPU runs on; the CPU waits for it when it
// touches the RSP's registers or memory, and at the end of the next window. What the RSP does to
// the rest of the machine (its interrupt, the RDP's commands) happens then, at those points in the
// emulated time, so a run doesn't depend on how the threads are scheduled. What only the main thread
// can do (waiting for the RDP, throwing away compiled CPU code) the RSP asks for and waits.

enum { IDLE, RUN, ASK, QUIT };
enum { SPIN = 4000 };

struct RspThread {
	pthread_t thread;
	pthread_mutex_t lock;
	pthread_cond_t wake, done;
	atomic_int state;
	atomic_bool sleeping, waiting;
	uint64_t until;
	// The RDRAM pages a DMA asks the main thread about.
	uint32_t askAddr, askLen;
	bool askWrite;
	// The window's SP interrupt changes, in order, and the RDP's command words.
	bool interrupts[8];
	int interruptCount;
	uint64_t* words;
	size_t wordCount, wordSpace;
};

static _Thread_local bool onThread;

static void relax(void) {
#if defined(__x86_64__) || defined(__i386__)
	__builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
	__asm__ volatile("yield");
#endif
}

static void set(struct RspThread* t, int state, atomic_bool* sleeper) {
	atomic_store(&t->state, state);
	if (atomic_load(sleeper)) {
		pthread_mutex_lock(&t->lock);
		pthread_cond_broadcast(sleeper == &t->sleeping ? &t->wake : &t->done);
		pthread_mutex_unlock(&t->lock);
	}
}

// Until the state is no longer from: spinning a while, then asleep.
static int await(struct RspThread* t, int from, atomic_bool* sleeper) {
	for (int i = 0; i < SPIN; ++i) {
		int s = atomic_load_explicit(&t->state, memory_order_acquire);
		if (s != from) {
			return s;
		}
		relax();
	}
	pthread_mutex_lock(&t->lock);
	atomic_store(sleeper, true);
	int s;
	while ((s = atomic_load(&t->state)) == from) {
		pthread_cond_wait(sleeper == &t->sleeping ? &t->wake : &t->done, &t->lock);
	}
	atomic_store(sleeper, false);
	pthread_mutex_unlock(&t->lock);
	return s;
}

static void* run(void* arg) {
	struct N64* n = arg;
	struct RspThread* t = n->rspThread;
	onThread = true;
	while (await(t, IDLE, &t->sleeping) == RUN) {
		n->catchingUp = true;
		rspRunTo(n, t->until);
		n->catchingUp = false;
		set(t, IDLE, &t->waiting);
	}
	return NULL;
}

void rspThreadStart(struct N64* n) {
	struct RspThread* t = calloc(1, sizeof(*t));
	if (!t) {
		return;
	}
	pthread_mutex_init(&t->lock, NULL);
	pthread_cond_init(&t->wake, NULL);
	pthread_cond_init(&t->done, NULL);
	n->rspThread = t;
	if (pthread_create(&t->thread, NULL, run, n)) {
		n->rspThread = NULL;
		free(t);
	}
}

void rspThreadStop(struct N64* n) {
	struct RspThread* t = n->rspThread;
	if (!t) {
		return;
	}
	rspThreadJoin(n);
	set(t, QUIT, &t->sleeping);
	pthread_join(t->thread, NULL);
	pthread_mutex_destroy(&t->lock);
	pthread_cond_destroy(&t->wake);
	pthread_cond_destroy(&t->done);
	free(t->words);
	free(t);
	n->rspThread = NULL;
}

static void answer(struct N64* n, struct RspThread* t) {
	if (t->askWrite) {
		n64RdramWrite(n, t->askAddr, t->askLen);
	} else {
		n64RdramRead(n, t->askAddr, t->askLen);
	}
	set(t, RUN, &t->sleeping);
}

// The window done: its interrupts and the RDP's commands now.
void rspThreadJoin(struct N64* n) {
	struct RspThread* t = n->rspThread;
	if (onThread) {
		return;
	}
	while (await(t, RUN, &t->waiting) == ASK) {
		answer(n, t);
	}
	for (int i = 0; i < t->interruptCount; ++i) {
		n64Interrupt(n, MI_SP, t->interrupts[i]);
	}
	t->interruptCount = 0;
	if (t->wordCount) {
		rdpFeed(n, t->words, t->wordCount);
		t->wordCount = 0;
	}
}

void rspThreadSlice(struct N64* n) {
	struct RspThread* t = n->rspThread;
	if (atomic_load_explicit(&t->state, memory_order_acquire) == ASK) {
		answer(n, t);
	}
	if (n->cycles < n->rspWindowAt) {
		return;
	}
	n->rspWindowAt = n->cycles + n->rspWindow;
	rspThreadJoin(n);
	uint64_t const now = rspCycles(n);
	if ((n->sp.status & 1) || n->sp.hleDoneAt) {
		n->rsp.clock = now;
		return;
	}
	if (n->rsp.clock < now) {
		t->until = now;
		set(t, RUN, &t->sleeping);
	}
}

void spInterrupt(struct N64* n, bool on) {
	struct RspThread* t = n->rspThread;
	if (!onThread) {
		n64Interrupt(n, MI_SP, on);
		return;
	}
	if (t->interruptCount == 8) {
		memmove(t->interrupts, t->interrupts + 1, 7);
		--t->interruptCount;
	}
	t->interrupts[t->interruptCount++] = on;
}

// The RSP's DMA reaches RDRAM pages: on the thread, the main thread does what touches the RDP or
// compiled code.
void spDmaPages(struct N64* n, uint32_t addr, uint32_t len, bool write) {
	if (!onThread) {
		write ? n64RdramWrite(n, addr, len) : n64RdramRead(n, addr, len);
		return;
	}
	uint32_t const first = (addr & (RDRAM_SIZE - 1)) >> 12, last = ((addr + len - 1) & (RDRAM_SIZE - 1)) >> 12;
	uint8_t const need = write ? PAGE_RDP | PAGE_CODE : n->rdpPending ? PAGE_RDP_WRITES : 0;
	for (uint32_t p = first;; p = (p + 1) & ((RDRAM_SIZE >> 12) - 1)) {
		if (n->pageFlags[p] & need) {
			struct RspThread* t = n->rspThread;
			t->askAddr = addr, t->askLen = len, t->askWrite = write;
			set(t, ASK, &t->waiting);
			await(t, ASK, &t->sleeping);
			return;
		}
		if (write) {
			n->dirtyPages[p] = 0;
		}
		if (p == last) {
			break;
		}
	}
}

// DP_END written, or the RDP thawed: on the thread the commands are taken now and run at the join.
void rdpKick(struct N64* n) {
	struct RspThread* t = n->rspThread;
	if (!onThread) {
		rdpRun(n);
		return;
	}
	struct DP* d = &n->dp;
	if (d->status & 2) {
		return;
	}
	d->status |= DP_RUNNING;
	for (; d->current < d->end; d->current += 8) {
		if (t->wordCount == t->wordSpace) {
			size_t space = t->wordSpace ? t->wordSpace * 2 : 4096;
			uint64_t* words = realloc(t->words, space * sizeof(*words));
			if (!words) {
				return;
			}
			t->words = words, t->wordSpace = space;
		}
		const uint8_t* p = d->status & 1 ? &n->sp.mem[d->current & 0xFF8] : &n->rdram[d->current & (RDRAM_SIZE - 8)];
		t->words[t->wordCount++] = (uint64_t) be32(p) << 32 | be32(p + 4);
	}
}
