# glslang (Khronos), compiling the Vulkan renderer's shaders: its GLSL front end and SPIR-V back end
# built from ARMN64_GLSLANG_DIR's sources (16.5.0), with no CMake of its own (it asks for a newer
# CMake than the builds have).
set(GLSLANG ${ARMN64_GLSLANG_DIR})
file(WRITE ${CMAKE_BINARY_DIR}/glslang-info/glslang/build_info.h
	"#ifndef GLSLANG_BUILD_INFO\n#define GLSLANG_BUILD_INFO\n"
	"#define GLSLANG_VERSION_MAJOR 16\n#define GLSLANG_VERSION_MINOR 5\n#define GLSLANG_VERSION_PATCH 0\n#define GLSLANG_VERSION_FLAVOR \"\"\n"
	"#define GLSLANG_VERSION_GREATER_THAN(major, minor, patch) ((GLSLANG_VERSION_MAJOR) > (major) || ((major) == GLSLANG_VERSION_MAJOR && ((GLSLANG_VERSION_MINOR) > (minor) || ((minor) == GLSLANG_VERSION_MINOR && (GLSLANG_VERSION_PATCH) > (patch)))))\n"
	"#define GLSLANG_VERSION_GREATER_OR_EQUAL_TO(major, minor, patch) ((GLSLANG_VERSION_MAJOR) > (major) || ((major) == GLSLANG_VERSION_MAJOR && ((GLSLANG_VERSION_MINOR) > (minor) || ((minor) == GLSLANG_VERSION_MINOR && (GLSLANG_VERSION_PATCH >= (patch))))))\n"
	"#define GLSLANG_VERSION_LESS_THAN(major, minor, patch) ((GLSLANG_VERSION_MAJOR) < (major) || ((major) == GLSLANG_VERSION_MAJOR && ((GLSLANG_VERSION_MINOR) < (minor) || ((minor) == GLSLANG_VERSION_MINOR && (GLSLANG_VERSION_PATCH) < (patch)))))\n"
	"#define GLSLANG_VERSION_LESS_OR_EQUAL_TO(major, minor, patch) ((GLSLANG_VERSION_MAJOR) < (major) || ((major) == GLSLANG_VERSION_MAJOR && ((GLSLANG_VERSION_MINOR) < (minor) || ((minor) == GLSLANG_VERSION_MINOR && (GLSLANG_VERSION_PATCH <= (patch))))))\n"
	"#endif\n")
set(GLSLANG_SOURCES
	glslang/CInterface/glslang_c_interface.cpp
	glslang/GenericCodeGen/CodeGen.cpp glslang/GenericCodeGen/Link.cpp
	glslang/MachineIndependent/glslang_tab.cpp glslang/MachineIndependent/attribute.cpp glslang/MachineIndependent/Constant.cpp
	glslang/MachineIndependent/iomapper.cpp glslang/MachineIndependent/InfoSink.cpp glslang/MachineIndependent/Initialize.cpp
	glslang/MachineIndependent/IntermTraverse.cpp glslang/MachineIndependent/Intermediate.cpp glslang/MachineIndependent/ParseContextBase.cpp
	glslang/MachineIndependent/ParseHelper.cpp glslang/MachineIndependent/PoolAlloc.cpp glslang/MachineIndependent/RemoveTree.cpp
	glslang/MachineIndependent/Scan.cpp glslang/MachineIndependent/ShaderLang.cpp glslang/MachineIndependent/SpirvIntrinsics.cpp
	glslang/MachineIndependent/SymbolTable.cpp glslang/MachineIndependent/Versions.cpp glslang/MachineIndependent/intermOut.cpp
	glslang/MachineIndependent/limits.cpp glslang/MachineIndependent/linkValidate.cpp glslang/MachineIndependent/parseConst.cpp
	glslang/MachineIndependent/reflection.cpp glslang/MachineIndependent/preprocessor/Pp.cpp glslang/MachineIndependent/preprocessor/PpAtom.cpp
	glslang/MachineIndependent/preprocessor/PpContext.cpp glslang/MachineIndependent/preprocessor/PpScanner.cpp
	glslang/MachineIndependent/preprocessor/PpTokens.cpp glslang/MachineIndependent/propagateNoContraction.cpp
	glslang/OSDependent/Unix/ossource.cpp
	glslang/ResourceLimits/ResourceLimits.cpp glslang/ResourceLimits/resource_limits_c.cpp
	SPIRV/GlslangToSpv.cpp SPIRV/InReadableOrder.cpp SPIRV/Logger.cpp SPIRV/SpvBuilder.cpp SPIRV/SpvPostProcess.cpp SPIRV/doc.cpp
	SPIRV/SpvTools.cpp SPIRV/disassemble.cpp SPIRV/CInterface/spirv_c_interface.cpp)
list(TRANSFORM GLSLANG_SOURCES PREPEND ${GLSLANG}/)
add_library(glslang STATIC ${GLSLANG_SOURCES})
set_target_properties(glslang PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_VISIBILITY_PRESET hidden)
target_include_directories(glslang PUBLIC ${GLSLANG} ${CMAKE_BINARY_DIR}/glslang-info)
target_compile_definitions(glslang PUBLIC ENABLE_SPIRV GLSLANG_OSINCLUDE_UNIX ENABLE_OPT=0)
target_compile_options(glslang PRIVATE -w)
target_link_libraries(glslang Threads::Threads)
