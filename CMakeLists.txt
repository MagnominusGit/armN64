cmake_minimum_required(VERSION 3.13)
project(armN64 C)

set(CMAKE_C_STANDARD 11)
# The libretro core is a shared library built from the same emulator.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
# No fused multiply-adds: the FPU and RDP must round the same on every host.
add_compile_options(-Wall -Wextra -Wno-unused-parameter -ffp-contract=off -fno-math-errno)
# GCC 10's SLP vectorizer copies struct N64's misaligned arrays with aligned SSE moves (state loads crash).
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 11)
	add_compile_options(-fno-tree-slp-vectorize)
endif()
# Off where the toolchain picks the CPU (a distribution's package for each device).
option(ARMN64_TUNE "Build for x86-64-v2 or the Cortex-A55" ON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
	if(ARMN64_TUNE)
		add_compile_options(-march=x86-64-v2)
	endif()
	set(JIT_SOURCES src/jit_x64.c src/rsp_jit.c src/rsp_jit_x64.c)
	add_compile_definitions(N64_HAS_JIT)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND EXISTS ${CMAKE_SOURCE_DIR}/src/jit_a64.c)
	if(ARMN64_TUNE)
		add_compile_options(-march=armv8.2-a -mtune=cortex-a55)
	endif()
	set(JIT_SOURCES src/jit_a64.c src/rsp_jit.c src/rsp_jit_a64.c)
	add_compile_definitions(N64_HAS_JIT)
else()
	set(JIT_SOURCES src/jit_none.c)
endif()

add_library(n64 STATIC src/n64.c src/blocks.c ${JIT_SOURCES} src/cpu.c src/fpu.c src/fpu_fast.c src/bus.c src/pif.c src/rsp.c src/rsp_vu.c src/rspthread.c src/rdp.c src/hle.c src/gles.c src/glr.c src/carts.c src/gbcart.c src/dd.c src/state.c src/cheats.c
	src/audio_hle/audio_hle.c src/audio_hle/alist.c src/audio_hle/alist_audio.c src/audio_hle/alist_naudio.c src/audio_hle/alist_nead.c
	src/audio_hle/audio.c src/audio_hle/memory.c src/audio_hle/mp3.c src/audio_hle/musyx.c src/audio_hle/re2.c src/audio_hle/hvqm.c)
# The audio HLE stores words into halfword arrays through casts, as mupen64plus does.
set_source_files_properties(src/audio_hle/alist.c src/audio_hle/alist_audio.c src/audio_hle/alist_naudio.c src/audio_hle/alist_nead.c
	src/audio_hle/audio.c src/audio_hle/memory.c src/audio_hle/mp3.c src/audio_hle/musyx.c src/audio_hle/re2.c src/audio_hle/hvqm.c src/audio_hle/audio_hle.c
	PROPERTIES COMPILE_OPTIONS -fno-strict-aliasing)
# The GPU shaders (RDP and VI), as C strings.
foreach(GLSL rdp_gpu vi_gpu)
	set(GLSL_FILE ${CMAKE_SOURCE_DIR}/src/${GLSL}.glsl)
	set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GLSL_FILE})
	file(READ ${GLSL_FILE} SHADER)
	string(REPLACE "\\" "\\\\" SHADER "${SHADER}")
	string(REPLACE "\"" "\\\"" SHADER "${SHADER}")
	string(REPLACE "\n" "\\n\"\n\"" SHADER "${SHADER}")
	string(REGEX REPLACE "_gpu$" "GpuSource" NAME ${GLSL})
	file(WRITE ${CMAKE_BINARY_DIR}/generated/${GLSL}_glsl.h "static const char ${NAME}[] =\n\"${SHADER}\";\n")
endforeach()
target_include_directories(n64 PUBLIC src PRIVATE ${CMAKE_BINARY_DIR}/generated)
find_package(Threads REQUIRED)
# What the core shares with Armulators, copied into shared/ (tools/check-shared.sh compares): the
# code generators, the libretro core's helpers and the Pak manager's font.
target_include_directories(n64 PRIVATE shared/emit)
target_link_libraries(n64 m Threads::Threads ${CMAKE_DL_LIBS})

add_executable(n64run tools/n64run.c)
target_link_libraries(n64run n64)

# The libretro core.
# The Controller Pak manager's font (Inter, OFL) as a C array.
set(PAKS_FONT ${CMAKE_CURRENT_SOURCE_DIR}/shared/fonts/Inter-Medium.otf)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/paks_font.inc
	COMMAND ${CMAKE_COMMAND} -DFONTS=${PAKS_FONT} -DOUT=${CMAKE_CURRENT_BINARY_DIR}/paks_font.inc -P ${CMAKE_CURRENT_SOURCE_DIR}/shared/fonts/embed.cmake
	DEPENDS ${PAKS_FONT} ${CMAKE_CURRENT_SOURCE_DIR}/shared/fonts/embed.cmake VERBATIM)
set(PAKS_SOURCES libretro/paks.c ${CMAKE_CURRENT_BINARY_DIR}/paks_font.inc)
set(PAKS_INCLUDES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/shared)

add_library(armn64_libretro SHARED libretro/libretro.c shared/libretro/options.c ${PAKS_SOURCES})
target_include_directories(armn64_libretro PRIVATE ${PAKS_INCLUDES} shared/libretro)
target_link_libraries(armn64_libretro n64)
set_target_properties(armn64_libretro PROPERTIES PREFIX "")
target_link_options(armn64_libretro PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/shared/libretro/link.T -Wl,--no-undefined)

# armn64, the standalone app: Armulators' (armulators/app, the only part that needs the submodule)
# with the core linked in, named by app.cmake, with its own rules from app/.
option(ARMN64_APP "Build armn64, the standalone SDL2 app" OFF)
if(ARMN64_APP)
	add_subdirectory(armulators)
	armulators_app(armn64-app DESCRIPTOR ${CMAKE_CURRENT_SOURCE_DIR}/app.cmake
		SOURCES libretro/libretro.c ${PAKS_SOURCES} app/hooks.c INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/libretro ${PAKS_INCLUDES} LIBS n64)
endif()
