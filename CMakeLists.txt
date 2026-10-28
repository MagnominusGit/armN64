cmake_minimum_required(VERSION 3.13)
project(arch-N64 C)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

add_library(n64 STATIC src/n64.c src/blocks.c src/jit_x64.c src/rsp_jit_x64.c src/cpu.c src/fpu.c src/bus.c src/pif.c src/rsp.c src/rdp.c src/carts.c)
target_include_directories(n64 PUBLIC src)
target_link_libraries(n64 m)

add_executable(n64run tools/n64run.c)
target_link_libraries(n64run n64)
