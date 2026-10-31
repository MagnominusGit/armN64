/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *   Mupen64plus-rsp-hle - hle.c (audio task detection), armN64 glue      *
 *   Mupen64Plus homepage: https://mupen64plus.org/                        *
 *   Copyright (C) 2012 Bobby Smiles                                       *
 *   Copyright (C) 2009 Richard Goedeken                                   *
 *   Copyright (C) 2002 Hacktarux                                          *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.          *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* armN64: the audio tasks of mupen64plus-rsp-hle for HLE accuracy, which puts speed first. A task
 * whose microcode isn't one of these is left to the RSP. */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../n64.h"
#include "alist.h"
#include "common.h"
#include "hle_external.h"
#include "hle_internal.h"
#include "memory.h"
#include "ucodes.h"

static ucode_func_t try_audio_task_detection(struct hle_t* hle)
{
    /* identify audio ucode by using the content of ucode_data */
    uint32_t ucode_data = *dmem_u32(hle, TASK_UCODE_DATA);
    uint32_t v;

    if (dram_r32(hle, ucode_data) == 0x00000001) {
        if (dram_r32(hle, ucode_data + 0x30) == 0xf0000f00) {
            v = dram_r32(hle, ucode_data + 0x28);
            switch(v)
            {
            case 0x1e24138c: /* audio ABI (most common) */
                return &alist_process_audio;
            case 0x1dc8138c: /* GoldenEye */
                return &alist_process_audio_ge;
            case 0x1e3c1390: /* BlastCorp, DiddyKongRacing */
                return &alist_process_audio_bc;
            }
        } else {
            v = dram_r32(hle, ucode_data + 0x10);
            switch(v)
            {
            case 0x11181350: /* MarioKart, WaveRace (E) */
                return &alist_process_nead_mk;
            case 0x111812e0: /* StarFox (J) */
                return &alist_process_nead_sfj;
            case 0x110412ac: /* WaveRace (J RevB) */
                return &alist_process_nead_wrjb;
            case 0x110412cc: /* StarFox/LylatWars (except J) */
                return &alist_process_nead_sf;
            case 0x1cd01250: /* FZeroX */
                return &alist_process_nead_fz;
            case 0x1f08122c: /* YoshisStory */
                return &alist_process_nead_ys;
            case 0x1f38122c: /* 1080 Snowboarding */
                return &alist_process_nead_1080;
            case 0x1f681230: /* Zelda OoT / Zelda MM (J, J RevA) */
                return &alist_process_nead_oot;
            case 0x1f801250: /* Zelda MM (except J, J RevA, E Beta), PokemonStadium 2 */
                return &alist_process_nead_mm;
            case 0x109411f8: /* Zelda MM (E Beta) */
                return &alist_process_nead_mmb;
            case 0x1eac11b8: /* AnimalCrossing */
                return &alist_process_nead_ac;
            case 0x00010010: /* MusyX v2 (IndianaJones, BattleForNaboo) */
                return &musyx_v2_task;
            /* armN64: Mario Artist Talent Studio and FZeroX Expansion run on the RSP. */
            }
        }
    } else {
        v = dram_r32(hle, ucode_data + 0x10);
        switch(v)
        {
        case 0x00000001: /* MusyX v1
            RogueSquadron, ResidentEvil2, PolarisSnoCross,
            TheWorldIsNotEnough, RugratsInParis, NBAShowTime,
            HydroThunder, Tarzan, GauntletLegend, Rush2049 */
            return &musyx_v1_task;
        case 0x0000127c: /* naudio (many games) */
            return &alist_process_naudio;
        case 0x00001280: /* BanjoKazooie */
            return &alist_process_naudio_bk;
        case 0x1c58126c: /* DonkeyKong */
            return &alist_process_naudio_dk;
        case 0x1ae8143c: /* BanjoTooie, JetForceGemini, MickeySpeedWayUSA, PerfectDark */
            return &alist_process_naudio_mp3;
        case 0x1ab0140c: /* ConkerBadFurDay */
            return &alist_process_naudio_cbfd;
        }
    }

    return NULL;
}

/* The audio task in DMEM (type 2) run on dram when its microcode is known; false leaves it alone. */
static bool audioHleRun(struct N64* n, uint8_t* dram)
{
    if (be32(&n->sp.mem[0xFC0]) != 2) {
        return false;
    }
    if (!n->hleAudio && !(n->hleAudio = calloc(1, sizeof(struct hle_t)))) {
        return false;
    }
    struct hle_t* hle = n->hleAudio;
    hle->dram = dram;
    hle->dram_mask = RDRAM_SIZE - 1;
    hle->dram_dirty = n->dirtyPages;
    hle->dmem = hle->dmem_buffer;
    hle->user_defined = n;
    for (unsigned i = 0; i < 0x1000; ++i) {
        hle->dmem[i ^ S8] = n->sp.mem[i];
    }
    ucode_func_t run = try_audio_task_detection(hle);
    if (!run) {
        return false;
    }
    run(hle);
    return true;
}

/* The audio task in DMEM run here, done a little later; false leaves it to the RSP. */
bool audioHle(struct N64* n)
{
    if (!audioHleRun(n, n->rdram)) {
        return false;
    }
    ++n->stats.audioHleTasks;
    n->sp.hleDoneAt = n->cycles + 4096;
    return true;
}

/* N64AUDIOCHECK=1 (with the RSP running audio): each audio task also run here on a copy of RDRAM
 * as it starts, and what it wrote compared with what the microcode wrote when the RSP halts. */
static uint8_t *checkBefore, *checkHle;
static uint64_t checkTask;

void audioCheckStart(struct N64* n)
{
    if (!getenv("N64AUDIOCHECK") || checkBefore || be32(&n->sp.mem[0xFC0]) != 2) {
        return;
    }
    checkBefore = malloc(RDRAM_SIZE);
    checkHle = malloc(RDRAM_SIZE);
    memcpy(checkBefore, n->rdram, RDRAM_SIZE);
    memcpy(checkHle, n->rdram, RDRAM_SIZE);
    if (!audioHleRun(n, checkHle)) {
        printf("AUDIOCHECK task %llu: microcode not known\n", (unsigned long long) ++checkTask);
        free(checkBefore);
        free(checkHle);
        checkBefore = checkHle = NULL;
    }
}

void audioCheckEnd(struct N64* n)
{
    if (!checkBefore) {
        return;
    }
    uint32_t wrong = 0, missing = 0, first = 0;
    for (uint32_t i = 0; i < RDRAM_SIZE; ++i) {
        if (checkHle[i] != checkBefore[i] && checkHle[i] != n->rdram[i]) {
            if (!wrong++) {
                first = i;
            }
        } else if (checkHle[i] == checkBefore[i] && n->rdram[i] != checkBefore[i]) {
            ++missing;
        }
    }
    printf("AUDIOCHECK task %llu (rsp task %llu): %u bytes wrong (first %06x hle %02x rsp %02x), %u written only by the rsp\n",
           (unsigned long long) ++checkTask, (unsigned long long) n->stats.rspTasks, wrong, first, checkHle[first], n->rdram[first], missing);
    free(checkBefore);
    free(checkHle);
    checkBefore = checkHle = NULL;
}

void rsp_break(struct hle_t* hle, unsigned int setbits)
{
}

int HleForwardTask(void* user_defined)
{
    return 0;
}

void HleVerboseMessage(void* user_defined, const char *message, ...)
{
}

void HleInfoMessage(void* user_defined, const char *message, ...)
{
}

void HleWarnMessage(void* user_defined, const char *message, ...)
{
}

void HleErrorMessage(void* user_defined, const char *message, ...)
{
    va_list args;
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    fputc('\n', stderr);
}

size_t audioHleStateSize(void)
{
    return sizeof(struct hle_t);
}
