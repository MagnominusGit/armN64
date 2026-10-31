# armN64

A lean, low-level Nintendo 64 emulator for x86-64 and ARM handhelds.

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

Licensed under the GNU General Public License, version 3 (`LICENSE`).
