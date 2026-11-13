# armN64

A lean, low-level Nintendo 64 libretro core, with its own standalone app, for x86-64 and ARM
handhelds.

Every part of armN64 emulates hardware. There is no game-specific microcode, no per-game timing value
and no per-game graphics fix: the RSP can always run the game's own microcode, and the RDP is
emulated from its command stream. The only per-game data is what a cartridge can't report about
itself, such as its save chip. It is exact where a game can see it (interrupt order, DMA and VI
timing, RSP behaviour) and simplified only where it can't.

Its four Accuracy modes run from Accurate, cycle for cycle, to Fastest. Fast, the default, puts speed
first. Please report problems in Issues.

## Legal

- **Your own games.** armN64 comes with no games and doesn't download any. Play only games you own,
  dumped from your own cartridges or bought legally. armN64 doesn't support or promote piracy.
- **Not Nintendo.** armN64 is an independent project, not affiliated with or endorsed by Nintendo.
  Nintendo 64, 64DD, Controller Pak, Rumble Pak and Transfer Pak are trademarks of Nintendo, and the
  64DD's IPL is Nintendo's copyright.

## What it runs

- Nintendo 64 cartridges (`.z64`, `.n64`, `.v64`), also inside `.zip` and `.7z` archives, with their
  EEPROM, SRAM or Flash saves.
- 64DD disks (`.ndd`), with the drive's IPL as `64DD_IPL.bin` in the frontend's system folder. A
  disk's changes are kept with the saves, as `<disk name>.ndd.sav`.
- Up to four controllers, each with a Controller Pak, Rumble Pak or Transfer Pak. A Transfer Pak
  holds the Game Boy ROM in `system/armN64/TransferPak1` (to `4`), with its save beside it.
- GameShark cheats, RetroAchievements (with memory as mupen64plus-next lays it out, which the sets
  are made on), save states, rewind and run-ahead.

## Controls

| Button | N64 |
| --- | --- |
| Bottom, left face buttons | A, B |
| Right, top face buttons | C-Down, C-Left |
| R2 held, with the face buttons | The C buttons in their places: top C-Up, left C-Left, bottom C-Down, right C-Right (C Buttons on Face Buttons) |
| Right stick | The C buttons |
| Left stick | The control stick |
| D-pad, Start | D-pad, Start |
| L, R, L2 | L, R, Z |
| Start (or Select) + D-pad, + R3; L3 + R3 | Switch a Control Swap on or off (Control Swap Hotkeys) |

## Options

The menu shows only the options that do something with the others' values: the GL Renderer's with
GL, Resolution with GL or GPU, the custom crops with Crop Overscan Custom, and so on.

| Category | Option | Values (default first) |
| --- | --- | --- |
| System | Accuracy (applies on restart) | Fast (HLE), Fastest (FLE), Balanced (MLE), Accurate (LLE) |
| | Overclock (applies on restart) | Off, 1.5x, 2x, 2.5x, 3x, 4x |
| | Frame Skip (applies on restart) | Off, Auto, Every Other Frame |
| | Expansion Pak (applies on restart) | On, Off |
| Video | Renderer (applies on restart) | GL (Fast, HD), GPU (Exact, HD), CPU (Exact, Native) |
| | Resolution (GL and GPU; applies on restart) | 2x, Auto, Native, 3x to 8x |
| | Widescreen | Off, 16:9, 16:10, 16:9 Picture Only, 16:10 Picture Only |
| | Aspect Ratio | Fill 4:3, TV 4:3, Square Pixels |
| | Crop Overscan | On, Off, Custom |
| | Crop Top, Bottom, Left, Right (with Custom) | 0 to 24 of the console's pixels (8, 8, 10, 10 by default) |
| | TV Filters (CPU and GPU) | Auto (on with Balanced and Accurate), On, Off |
| GL Renderer | Texture Filtering | Console, Smooth, Sharp |
| | Anti-Aliasing | Off, 2x, 4x, 8x |
| | Seal Edges | Off, On |
| | Shader Accuracy | Accurate, Fast |
| | Framebuffer Effects | Exact, Fast |
| | Shader Precompile | On, Off |
| Input | Octagonal Stick Gate | Off, On |
| | Stick Deadzone | Off, 5% to 30% |
| | Stick Sensitivity | 1.0x, 0.5x to 2.0x |
| | C Buttons on Face Buttons | R2 + Face Buttons, Off |
| | Control Swap | Off, D-Pad / Left Stick, Both Sticks, Left / Right Stick |
| | Control Swap Hotkeys | Start + D-Pad / R3, L3 + R3; Select + D-Pad / R3, L3 + R3; Off |
| | Dual Controllers | Off, On |
| | Controller Rumble | Off, On |
| Controller Paks | Player 1 to 4 Pak | Controller Pak, Rumble Pak, Transfer Pak, None |
| | Player 1 to 4 Controller Pak (with a Controller Pak) | This Game's, Shared Pak 1 to 8 |
| | Controller Pak Manager | Closed, Open |
| Advanced | Graphics Microcode (applies on restart) | Library, Game's Own |
| | Graphics Task Timing (with Library) | Console, Short |
| | Threaded RSP (with Fast) | Off, On |
| | Renderer Threads (with CPU) | Auto (up to 3), None, 1, 2, 3 |

### System

- **Accuracy.** Fastest: Fast with shortcuts (looser CPU timing); some effects may differ. Fast: the
  built-in audio microcodes and loose timing. Balanced: each game's own audio microcode and closer
  timing. Accurate: cycle for cycle. Graphics follow Graphics Microcode in every mode.
- **Frame Skip.** Leaves frames undrawn and shows the one before again. Auto skips only when the
  device falls behind (the frontend's audio buffer runs low); Every Other Frame always does.
- **Expansion Pak.** Off makes games see 4 MB, as on a console without one.

### Video

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

### GL Renderer

- **Texture Filtering.** Console: the N64's three-texel filter. Smooth: four texels, as PC games
  filter. Sharp: no filtering. Textures a game draws unfiltered stay so. With Shader Accuracy Fast
  the GPU filters with four texels anyway, so Console and Smooth look alike.
- **Framebuffer Effects.** Exact: effects that draw from the picture (blurs, screen copies) wait for
  the GPU to hand it back. Fast: they use the last one in memory.

### Input

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

### Controller Paks

A Controller Pak is This Game's, kept with the game's save, or one of eight Shared Paks that every
game sees, as a real pak moves between games. The Controller Pak Manager pauses the game to copy,
delete, export, import and format the paks' notes; B closes it.

### Advanced

- **Graphics Microcode.** Library draws with the core's own copy of the graphics microcodes, at any
  Accuracy. Game's Own runs the game's microcode on the RSP: exact and slower, for a game Library
  draws wrong.
- **Graphics Task Timing.** With Library. Console: tasks take as long as the game's microcode would,
  so games keep the console's frame rate. Short: they finish sooner, and some games then draw more
  frames than on a console.

## Building

    cmake -S . -B build -G Ninja
    ninja -C build armn64_libretro

This makes `armn64_libretro.so`; `libretro/armn64_libretro.info` goes in RetroArch's info directory.
The recompilers have backends for x86-64 and AArch64; other CPUs run the interpreters. Each release
carries the core for x86-64 and 64-bit ARM Linux, with the AppImages and the APK. `n64run` is the
headless test tool (see Diagnostics).

## Standalone app

`armn64` runs games in a window without RetroArch. It is Armulators' app (the `armulators`
submodule), which armGBA and armNES share; armN64 gives it its names, systems, keys and icons
(`app.cmake`) and its hooks (`app/`). The GPU renderers draw in a GL context the app makes for the
core, as RetroArch does.

    git submodule update --init
    cmake -S . -B build -G Ninja -DARMN64_APP=ON
    ninja -C build armn64-app

`armulators/package/appimage/build.sh`, then `package.sh` beside it, make AppImages for x86-64 and
64-bit ARM Linux in a Debian 11 container (Docker), with SDL2 built in.
`armulators/package/android/build.sh` makes the Android APK.

Run `armn64 game.z64` (or a `.n64`, `.v64`, `.ndd`, `.zip` or `.7z` file), or drop a game on the
window. Esc, a controller's Hotkey + X or a right-click opens the menu, where all of these can be
changed. The keyboard's keys are mupen64plus's:

| Keyboard | N64 |
| --- | --- |
| Arrow keys | Control stick |
| I, K, J, L | C-Up, C-Down, C-Left, C-Right |
| W, S, A, D | D-pad |
| Left Shift, Left Ctrl | A, B |
| Enter | Start |
| X, C, Z | L, R, Z |

The app's own keys (Tab fast-forward, R rewind, P pause, F2 restart, F5 and F7 states, F12
screenshot and so on) and controller hotkeys are those of armGBA's and armNES's app; see armGBA's
README. Its settings are in `~/.config/armn64`, with the 64DD IPL and Transfer Pak ROMs in its
`system` folder, and its saves, states and screenshots in `~/.local/share/armn64`. On ROCKNIX it
follows ROCKNIX's other standalone emulators' folders.

## Accuracy

- **Accurate** (LLE) runs the VR4300, the RSP and the RDP cycle for cycle, with the game's own audio
  microcode; it's the reference the other modes are checked against. In every mode the graphics
  follow Graphics Microcode: the core's Library, or the game's own on the RSP.
- **Balanced** (MLE) runs each game's own audio microcode with closer timing, and the graphics as
  Graphics Microcode says.
- **Fast** (HLE), the default, uses the core's own audio microcodes and looser timing.
- **Fastest** (FLE) takes shortcuts in the CPU's timing on top of Fast; some effects may differ.
- The CPU and GPU renderers draw exactly what the RDP would, at the console's resolution. The GL
  renderer is faster and can draw above it, but isn't exact.
- Save states are deterministic, so rewind and run-ahead stay in step.

## Testing

- **Test ROMs:** `docs/test-results.md` has the suites and their results: Dillon's CPU tests (26 of
  26), the CPU timing ROM, Peter Lemon's suites against his expected screenshots, and a collection
  of homebrew and public-domain test programs, all compared with parallel-n64 using the cxd4 RSP and
  the Angrylion RDP.
- **Games:** all of my games are run and checked: the recompilers against the interpreters, the ARM
  builds against the x86-64 ones, the GPU renderer against the CPU one, and the pictures against
  the accurate references (ares, and parallel-n64 with cxd4 and Angrylion).

## Performance

Frames per second on a Raspberry Pi 5 (Cortex-A76, 2.4 GHz), unthrottled, through the libretro API,
with the CPU renderer, which is exact (the GL renderer is faster), over the first 30 seconds from
power-on with no input. Full speed is 60. The games are four well-known commercial ones: a 3D
platformer, an action adventure, a kart racer and a first-person shooter.

| Game | Fastest | Fast | Balanced | Accurate |
| --- | --- | --- | --- | --- |
| 3D platformer | 220 | 190 | 142 | 140 |
| Action adventure | 183 | 164 | 87 | 86 |
| Kart racer | 286 | 253 | 153 | 152 |
| First-person shooter | 394 | 509 | 245 | 241 |

## How it works

**Interpreters first.** Portable interpreters for the VR4300, the RSP and the RDP, with the RCP's
interfaces, are the exact reference everything faster is checked against.

**Recompilers.** The VR4300 and the RSP are compiled to x86-64 or AArch64, with NEON or SSE for the
RSP's vector unit, and give the interpreters' results.

**Threads.** The RDP draws on other cores beside the CPU, kept in step at bus events (Renderer
Threads), and with Fast the RSP can run on one of its own (Threaded RSP).

**Renderers.** The CPU renderer is the exact RDP in software; the GPU renderer runs the same model
on the graphics chip; the GL renderer draws the games' display lists with its own shaders, for
speed and higher resolutions.

## Diagnostics

`n64run rom fields [picture.ppm]` runs a game headless and writes the last picture. It takes, among
others (see `tools/n64run.c`):

| Variable | Effect |
| --- | --- |
| `N64_ACCURACY` | The Accuracy mode (`fle`, `hle`, `mle`, `lle`) |
| `N64INPUT="from-to:pad,..."` | Controller 1's buttons and stick over a range of fields |
| `N64_PAK` | What is in the controllers' slots, a letter a port |
| `N64_TPAK=<rom>`, `N64_TPAKSAV=<file>` | The Game Boy cartridge in controller 1's Transfer Pak, and its save |
| `N64DDIPL`, `N64DDDISK`, `N64DDSAVE` | The 64DD's IPL, disk and saved changes |
| `N64CHEAT=<code>` | GameShark codes, joined by `+` |
| `N64SAVELOAD`, `N64SAVEDUMP` | The saves before and after the run |
| `N64STATEIN`, `N64STATEOUT`, `N64STATETEST` | Load a state, save one, or check that one replays |
| `N64AUDIODUMP=<file>` | The sound as 16-bit stereo samples |
| `N64SHOTS=<prefix>` with `N64SHOTFIELDS` or `N64SHOTEVERY` | Pictures at chosen fields |
| `N64_PERFMAP=1` | Compiled blocks named in `/tmp/perf-<pid>.map` for `perf` |

## Credits

Code taken from other projects, each under its own license (see License):

- **paraLLEl-RDP** (Themaister, MIT): how the RDP's rasterization, dither and texture sampling work.
- **GLideN64** (Sergey Lipskiy and contributors, GPL-2.0): some HLE microcode details (F3DAM's
  texture coordinates).
- **mupen64plus-rsp-hle** (GPL-2.0-or-later): the audio HLE, in `src/audio_hle/`.
- **ares**, as forked by Nekokabu (ares team, Near et al, ISC): the RDP's command times
  (`src/rdp_timing.h`) and the model around them.
- **Inter** (The Inter Project Authors, OFL-1.1): the Controller Pak Manager's font, in `shared/`.

Read for how the hardware behaves, without code taken: **Angrylion's RDP**, the **cxd4** RSP and
**n64brew**. Test ROMs: Dillon's n64-tests, pdrome's timing ROM, Peter Lemon's suites, and homebrew
by rasky, meeq and others.

## AI assistance

armN64 was written with heavy use of an AI coding assistant, Anthropic's Claude (through Claude
Code). Most of the code, comments and documentation were drafted by it at Magnominus's direction. He
chose what to build and decides what goes in. Each change was tested on his PCs and handhelds first,
and the accuracy and speed claims here come from those tests, not from the assistant's word.

## License

Licensed under the GNU General Public License, version 2 or (at your option) any later version
(`LICENSE`). GLideN64's code is GPL-2.0 only, so a build of armN64 as a whole is under GPL-2.0.
Parts from other projects keep their own notices in their files (see Credits); `src/audio_hle/`
has its own `LICENSES`.
