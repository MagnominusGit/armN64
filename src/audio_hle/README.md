The audio microcodes run at high level for HLE accuracy, from mupen64plus-rsp-hle (GPL-2.0-or-later;
copyright notices in each file, the project's license in LICENSES), as libretro's mupen64plus-next
bundles it at f275caf4b2bfa1e6d1c51636746ea793f3d80320.

Changed for armN64: RDRAM is big-endian bytes, so DRAM goes through the accessors in memory.h
(dram_r*/dram_w*, and dram_load_swapped16 where mupen64plus copied raw word-swapped memory); the
task's DMEM is copied into hle_t as host-order words; hle_internal.h keeps only what audio uses;
audio_hle.c holds the task detection from hle.c and the glue. Graphics, JPEG and other tasks are
not here (hle.c in src/ does graphics).

N64AUDIOCHECK=1 with mle or lle accuracy compares each audio task run here with the microcode's.
