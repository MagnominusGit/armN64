/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *   Mupen64plus-rsp-hle - memory.h                                        *
 *   Mupen64Plus homepage: https://mupen64plus.org/                        *
 *   Copyright (C) 2014 Bobby Smiles                                       *
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

#ifndef MEMORY_H
#define MEMORY_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "common.h"
#include "hle_internal.h"

#ifdef M64P_BIG_ENDIAN
#define S 0
#define S16 0
#define S8 0
#else
#define S 1
#define S16 2
#define S8 3
#endif

enum {
    TASK_TYPE               = 0xfc0,
    TASK_FLAGS              = 0xfc4,
    TASK_UCODE_BOOT         = 0xfc8,
    TASK_UCODE_BOOT_SIZE    = 0xfcc,
    TASK_UCODE              = 0xfd0,
    TASK_UCODE_SIZE         = 0xfd4,
    TASK_UCODE_DATA         = 0xfd8,
    TASK_UCODE_DATA_SIZE    = 0xfdc,
    TASK_DRAM_STACK         = 0xfe0,
    TASK_DRAM_STACK_SIZE    = 0xfe4,
    TASK_OUTPUT_BUFF        = 0xfe8,
    TASK_OUTPUT_BUFF_SIZE   = 0xfec,
    TASK_DATA_PTR           = 0xff0,
    TASK_DATA_SIZE          = 0xff4,
    TASK_YIELD_DATA_PTR     = 0xff8,
    TASK_YIELD_DATA_SIZE    = 0xffc
};

static inline unsigned int align(unsigned int x, unsigned amount)
{
    --amount;
    return (x + amount) & ~amount;
}

static inline uint8_t* u8(const unsigned char* buffer, unsigned address)
{
    return (uint8_t*)(buffer + (address ^ S8));
}

static inline uint16_t* u16(const unsigned char* buffer, unsigned address)
{
    assert((address & 1) == 0);
    return (uint16_t*)(buffer + (address ^ S16));
}

static inline uint32_t* u32(const unsigned char* buffer, unsigned address)
{
    assert((address & 3) == 0);
    return (uint32_t*)(buffer + address);
}

void load_u8 (uint8_t*  dst, const unsigned char* buffer, unsigned address, size_t count);
void load_u16(uint16_t* dst, const unsigned char* buffer, unsigned address, size_t count);
void load_u32(uint32_t* dst, const unsigned char* buffer, unsigned address, size_t count);
void store_u8 (unsigned char* buffer, unsigned address, const uint8_t*  src, size_t count);
void store_u16(unsigned char* buffer, unsigned address, const uint16_t* src, size_t count);
void store_u32(unsigned char* buffer, unsigned address, const uint32_t* src, size_t count);


/* convenient functions for DMEM access */
static inline uint8_t* dmem_u8(struct hle_t* hle, uint16_t address)
{
    return u8(hle->dmem, address & 0xfff);
}

static inline uint16_t* dmem_u16(struct hle_t* hle, uint16_t address)
{
    return u16(hle->dmem, address & 0xfff);
}

static inline uint32_t* dmem_u32(struct hle_t* hle, uint16_t address)
{
    return u32(hle->dmem, address & 0xfff);
}

static inline void dmem_load_u8(struct hle_t* hle, uint8_t* dst, uint16_t address, size_t count)
{
    load_u8(dst, hle->dmem, address & 0xfff, count);
}

static inline void dmem_load_u16(struct hle_t* hle, uint16_t* dst, uint16_t address, size_t count)
{
    load_u16(dst, hle->dmem, address & 0xfff, count);
}

static inline void dmem_load_u32(struct hle_t* hle, uint32_t* dst, uint16_t address, size_t count)
{
    load_u32(dst, hle->dmem, address & 0xfff, count);
}

static inline void dmem_store_u8(struct hle_t* hle, const uint8_t* src, uint16_t address, size_t count)
{
    store_u8(hle->dmem, address & 0xfff, src, count);
}

static inline void dmem_store_u16(struct hle_t* hle, const uint16_t* src, uint16_t address, size_t count)
{
    store_u16(hle->dmem, address & 0xfff, src, count);
}

static inline void dmem_store_u32(struct hle_t* hle, const uint32_t* src, uint16_t address, size_t count)
{
    store_u32(hle->dmem, address & 0xfff, src, count);
}

/* convenient functions DRAM access
 * armN64: RDRAM is big-endian bytes (not host-order words), so DRAM goes through accessors. */
static inline uint8_t dram_r8(struct hle_t* hle, uint32_t address)
{
    return hle->dram[address & hle->dram_mask];
}

static inline uint16_t dram_r16(struct hle_t* hle, uint32_t address)
{
    return (uint16_t)(dram_r8(hle, address) << 8 | dram_r8(hle, address + 1));
}

static inline uint32_t dram_r32(struct hle_t* hle, uint32_t address)
{
    return (uint32_t)dram_r16(hle, address) << 16 | dram_r16(hle, address + 2);
}

static inline void dram_w8(struct hle_t* hle, uint32_t address, uint8_t value)
{
    hle->dram[address & hle->dram_mask] = value;
    hle->dram_dirty[(address & hle->dram_mask) >> 12] = 0;
}

static inline void dram_w16(struct hle_t* hle, uint32_t address, uint16_t value)
{
    dram_w8(hle, address, (uint8_t)(value >> 8));
    dram_w8(hle, address + 1, (uint8_t)value);
}

static inline void dram_w32(struct hle_t* hle, uint32_t address, uint32_t value)
{
    dram_w16(hle, address, (uint16_t)(value >> 16));
    dram_w16(hle, address + 2, (uint16_t)value);
}

/* armN64: DMA between DRAM and a buffer laid out as DMEM (host-order words), byte for byte. */
static inline void dram_to_buffer(struct hle_t* hle, unsigned char* buffer, unsigned offset, uint32_t address, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        buffer[(offset + i) ^ S8] = dram_r8(hle, address + (uint32_t)i);
}

static inline void buffer_to_dram(struct hle_t* hle, uint32_t address, const unsigned char* buffer, unsigned offset, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dram_w8(hle, address + (uint32_t)i, buffer[(offset + i) ^ S8]);
}

/* armN64: halfwords as mupen64plus's word-swapped DRAM lays them out, element i being halfword i ^ 1:
 * what memcpy between DRAM and a local array gives there. */
static inline void dram_load_swapped16(struct hle_t* hle, int16_t* dst, uint32_t address, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = (int16_t)dram_r16(hle, address + 2 * (uint32_t)(i ^ 1));
}

static inline void dram_store_swapped16(struct hle_t* hle, uint32_t address, const int16_t* src, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dram_w16(hle, address + 2 * (uint32_t)(i ^ 1), (uint16_t)src[i]);
}

static inline void dram_load_u8(struct hle_t* hle, uint8_t* dst, uint32_t address, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = dram_r8(hle, address + (uint32_t)i);
}

static inline void dram_load_u16(struct hle_t* hle, uint16_t* dst, uint32_t address, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = dram_r16(hle, address + 2 * (uint32_t)i);
}

static inline void dram_load_u32(struct hle_t* hle, uint32_t* dst, uint32_t address, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dst[i] = dram_r32(hle, address + 4 * (uint32_t)i);
}

static inline void dram_store_u8(struct hle_t* hle, const uint8_t* src, uint32_t address, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dram_w8(hle, address + (uint32_t)i, src[i]);
}

static inline void dram_store_u16(struct hle_t* hle, const uint16_t* src, uint32_t address, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dram_w16(hle, address + 2 * (uint32_t)i, src[i]);
}

static inline void dram_store_u32(struct hle_t* hle, const uint32_t* src, uint32_t address, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        dram_w32(hle, address + 4 * (uint32_t)i, src[i]);
}

#endif

