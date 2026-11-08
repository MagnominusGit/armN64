# armN64

A lean, low-level Nintendo 64 emulator for x86-64 and ARM handhelds.

**Status: 1.1.** Accurate is cycle-exact; Fast, the default, puts speed first. Please report problems
in Issues.

## Rules

- **Generalizability.** Every part emulates hardware. There is no game-specific microcode, no
  per-game timing value and no per-game graphics fix. The RSP always runs the game's own
  microcode, and the RDP is emulated from its command stream.
- The only per-game data allowed is what a cartridge can't report about itself, such as its save
  chip, in the same way as armGBA's override list.
- Exact where the game can see it (interrupt order, DMA and VI timing, RSP behaviour), simplified
  only where it can't.

## Targets

- 10x real time on one core of the mini PC (Ryzen 7 8745H), as the stand-in for full speed on an
  RK3566 (Cortex-A55).
- Every game boots to a picture within 30 seconds and shows no visible defects compared with the
  accurate references (ares, and paraLLEl N64 with the cxd4 RSP and the Angrylion RDP).

## Plan

1. Portable interpreters for the VR4300, RSP and RDP, plus the RCP interfaces, checked against
   test ROMs and the references. These stay as the exact reference for everything after.
2. Recompilers for the VR4300 and RSP on x86-64 and AArch64 (NEON for the RSP vector unit).
3. Threads: the RDP, and the RSP where it is safe, beside the CPU, synchronised at bus events.
4. RDP on the GPU for devices that can afford it, and a lighter path for those that can't.

## Layout

- `src/` the emulator: `n64.c` (machine and scheduler), `cpu.c` (VR4300), `bus.c` (memory map
  and RCP registers), `pif.c` (boot and controllers).
- `tools/n64run.c` runs a ROM headless and writes the last picture.
- `libretro/` the libretro core (`armn64_libretro`): a field each run, the VI's picture, the AI's
  samples at 44.1 kHz, controller 1 and the cartridge's save memory (EEPROM, then SRAM or flash).
- `armulators/` the code shared with armGBA (submodule): the code generators, the libretro
  core's helpers and the standalone app.
- `app.cmake` and `app/` armN64's standalone app: Armulators' app with the core linked in, its
  names, systems, keys and icons, and its hooks. The GPU renderers draw in a GL context the app
  makes for the core, as RetroArch does.

The app, and its AppImages and APK (into `build-appimage/` and `build-android/`):

    git submodule update --init
    cmake -S . -B build -G Ninja -DARMN64_APP=ON
    ninja -C build armn64-app
    armulators/package/appimage/build.sh && armulators/package/appimage/package.sh
    armulators/package/android/build.sh

## Core options

The options' help texts are short; the details are here.

**System**
- **Accuracy.** Fastest: Fast with shortcuts (looser CPU timing); some effects may differ. Fast: the
  built-in audio microcodes and loose timing. Balanced: each game's own audio microcode and closer
  timing. Accurate: cycle for cycle. Graphics follow Graphics Microcode in every mode.
- **Frame Skip.** Leaves frames undrawn and shows the one before again. Auto skips only when the
  device falls behind (the frontend's audio buffer runs low); Every Other Frame always does.
- **Expansion Pak.** Off makes games see 4 MB, as on a console without one.

**Video**
- **Renderer.** GL is the fastest and can draw above the N64's resolution, but isn't exact. CPU and
  GPU draw exactly as the console does at its own resolution: CPU on the processor (full speed on
  handhelds), GPU on the graphics chip (for desktop graphics cards; below full speed on handhelds).
  GPU can also draw higher (Resolution): close to exact, with the native picture kept for what
  games read back.
- **Widescreen.** 16:9 and 16:10 draw more of the scene at the sides with Library graphics; games may
  leave out things just off the 4:3 edges, and full-screen pictures stay stretched. Picture Only
  shows the picture that wide and changes nothing else, for games with their own wide modes.
- **Crop Overscan.** On hides a thirtieth of the picture at each side. Custom takes each side's
  amount in the console's pixels (scaled with the resolution).

**GL Renderer**
- **Texture Filtering.** Console: the N64's three-texel filter. Smooth: four texels, as PC games
  filter. Sharp: no filtering. Textures a game draws unfiltered stay so. With Shader Accuracy Fast
  the GPU filters with four texels anyway, so Console and Smooth look alike.
- **Framebuffer Effects.** Exact: effects that draw from the picture (blurs, screen copies) wait for
  the GPU to hand it back. Fast: they use the last one in memory.

**Input**

The bottom face button is A, the left one B, the right one C-Down and the top one C-Left.
- **C Buttons on Face Buttons.** On by default. With R2 held, the face buttons press the C buttons
  in the same places (top C-Up, left C-Left, bottom C-Down, right C-Right) instead of A and B. Not
  with Left / Right Stick, Both Sticks or Dual Controllers, which use R2.
- **Control Swap.** D-Pad / Left Stick: the D-pad moves the control stick, the left stick presses
  the D-pad, L1 presses Z and L2 presses L. Both Sticks: the two sticks added together move the
  control stick (no C buttons on the right stick), and L2 or R2 presses Z. Left / Right Stick: the
  right stick moves the control stick, the left stick presses the C buttons, and R2 presses Z.
- **Control Swap Hotkeys.** On player 1's controller, Start (or Select) held with any D-pad
  direction switches D-Pad / Left Stick; the game doesn't see the D-pad meanwhile, and a quick
  press of Start alone still reaches it as you let go. Start (or Select) with R3 switches
  Left / Right Stick; L3 and R3 together switch Both Sticks. Each turns its swap on, or off if on.
- **Dual Controllers.** Player 1's controller is two N64 controllers, one in each hand. Controller
  1: the left stick, D-pad, L1, L2, Start, and B/Y as A/B. Controller 2: the right stick, R2 as Z,
  R1, and A/X as A/B. Players 2 and 3 become controllers 3 and 4.

**Advanced**
- **Graphics Microcode.** Library draws with the core's own copy of the graphics microcodes, at any
  Accuracy. Game's Own runs the game's microcode on the RSP: exact and slower, for a game Library
  draws wrong.
- **Graphics Task Timing.** With Library. Console: tasks take as long as the game's microcode would,
  so games keep the console's frame rate. Short: they finish sooner, and some games then draw more
  frames than on a console.

Licensed under the GNU General Public License, version 2 or (at your option) any later version (`LICENSE`).
Parts come from other projects under their own licences, kept in their files:

- The RDP's rasterization, dither and texture sampling follow paraLLEl-RDP (Copyright (c) 2020
  Themaister, MIT licence).
- Some HLE microcode details (F3DAM's texture coordinates) follow GLideN64 (Sergey Lipskiy and
  contributors, GPL-2.0), so a build of armN64 as a whole is under GPL-2.0.
- The audio HLE is mupen64plus-rsp-hle's (GPL-2.0-or-later).
- The RDP's command times (`src/rdp_timing.h`) and the model around them come from Nekokabu's ares
  fork (Copyright (c) 2004-2025 ares team, Near et al, ISC licence).
