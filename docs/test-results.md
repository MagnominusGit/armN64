# Test ROM results

State at commit 9cdb2a8 (2026-10-17). Reference emulator: parallel-n64 with the cxd4 RSP and the Angrylion RDP.

## Suites

| Suite | ROMs | Result |
|---|---|---|
| Dillon's n64-tests (CPU) | 26 | 26 pass |
| pdrome n64-asm-timing, CPU timing (NTSC) | 1 (42 instructions) | every count within ~0.5% of hardware; the test needs exact equality, so it still prints FAIL |
| Peter Lemon N64 (expected-result PNGs) | 408 (371 with a PNG) | see below |
| Peter Lemon COP0Register, Exceptions/Trap | 2 | pass |
| Peter Lemon PI DMA alignment (4 ROMs) | 4 | all data checks pass; one interrupt-status check fails |
| Public-domain test and homebrew collection, rasky, meeq | 188 | all give a picture; 23 of them give none on the reference |

## pdrome CPU timing ROM

The ROM counts loop iterations between two scanlines (a loop of four 1-cycle instructions and one VI register read). Hardware counts are in the ROM.

| Instruction | Hardware | armN64 |
|---|---|---|
| ADD (and other 1-cycle ops) | 0xDB1C | 0xDB2C |
| DIV / DIVU | 0x5E71 | 0x5E4B |
| DDIV / DDIVU | 0x3EEC | 0x3EF5 |
| MULT / MULTU | 0xC344 | 0xC541 |
| DMULT / DMULTU | 0xB3B7 | 0xB353 |

Flat CPI 2 (the previous model) gave 0x4B222 for ADD.

## Peter Lemon, against his expected screenshots

The score is the mean difference of 64x48 thumbnails (0-255) between our picture and the PNG; the reference is scored the same way. Our pictures are 640x474 for 640x480 modes, which inflates the score on text screens, so the comparison with the reference matters more than the number.

| Folder | ROMs | as close as the reference or closer |
|---|---|---|
| RDP | 127 | 94 |
| CPUTest | 111 | 49 (most of the rest are text screens that pass; score skewed by the 474-line picture) |
| RSPTest | 56 | 25 (same) |
| RSP | 13 | 11 |
| CP1 | 12 | 12 |
| Compress | 12 | 10 |
| Video | 14 | 7 |
| others | 26 | 20 |

60 ROMs moved much closer after the RDP changes in 9cdb2a8 (1 slightly worse, a video's frame timing). The decode demos (I4, I8, GRB12/15/24) went from black to matching.

## Still failing

- LoadTLUT from 8- and 32-bit images: needs the full TMEM loading pipeline.
- Coverage and anti-aliasing (AlphaCoverage): not implemented.
- Reserved RSP vector opcodes (VADDB, VSUT, ...): cxd4 fails them too.
- YUV and LZ video demos: partly.
- PI interrupt status after clear (last line of the PI DMA test).
- COP0 Cause.CE starts at 3 on hardware; ours starts at 0.
- Homebrew that only ran on emulators: 1964 Demo (unaligned word stores), Dexanoid (FPU use with CU1 off).

## Not run

- Sources-only suites (n64-systemtest, Thar0's CPU/RDP timing tests, hacktarux's systembench, sauraen's RSP invalid instructions, rasky's systemcrash/corruption tests): they need the libdragon toolchain to build.
- Nintendo's NUS test cartridge.

## Games (first 30 s, batches of 10)

Batches 1-16 (160 ROMs) and the hard 10, compared with the reference at 5 s steps. With this commit the hard 10 all render; batch 3 is 7/10 (AeroFighters EU/USA title logo drawn huge, AeroGauge USA black), BattleTanx USA still hangs at boot (it needs the PIF boot time before IPL3, not modelled).
