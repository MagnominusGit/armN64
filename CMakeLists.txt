cmake_minimum_required(VERSION 3.13)
project(armN64 C)

set(CMAKE_C_STANDARD 11)
# The libretro core is a shared library built from the same emulator.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
# No fused multiply-adds: the FPU and RDP must round the same on every host.
add_compile_options(-Wall -Wextra -Wno-unused-parameter -ffp-contract=off)
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

add_library(n64 STATIC src/n64.c src/blocks.c ${JIT_SOURCES} src/cpu.c src/fpu.c src/fpu_fast.c src/bus.c src/pif.c src/rsp.c src/rsp_vu.c src/rdp.c src/hle.c src/gles.c src/glr.c src/carts.c src/dd.c
	src/audio_hle/audio_hle.c src/audio_hle/alist.c src/audio_hle/alist_audio.c src/audio_hle/alist_naudio.c src/audio_hle/alist_nead.c
	src/audio_hle/audio.c src/audio_hle/memory.c src/audio_hle/mp3.c src/audio_hle/musyx.c)
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
# Code shared with armGBA: the code generators, and the libretro core's helpers.
add_subdirectory(armulators)
target_link_libraries(n64 m Threads::Threads ${CMAKE_DL_LIBS} armulators-emit)

add_executable(n64run tools/n64run.c)
target_link_libraries(n64run n64)

# The libretro core.
add_library(armn64_libretro SHARED libretro/libretro.c)
target_link_libraries(armn64_libretro n64 armulators-libretro)
set_target_properties(armn64_libretro PROPERTIES PREFIX "")
target_link_options(armn64_libretro PRIVATE -Wl,--version-script=${ARMULATORS_LINK_SCRIPT} -Wl,--no-undefined)

# armn64, the standalone app: Armulators' (armulators/app) with the core linked in, named by
# app.cmake, with its own rules from app/.
option(ARMN64_APP "Build armn64, the standalone SDL2 app" OFF)
if(ARMN64_APP)
	armulators_app(armn64-app DESCRIPTOR ${CMAKE_CURRENT_SOURCE_DIR}/app.cmake
		SOURCES libretro/libretro.c app/hooks.c INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/libretro LIBS n64)
endif()
