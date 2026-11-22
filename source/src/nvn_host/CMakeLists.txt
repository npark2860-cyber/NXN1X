# SPDX-License-Identifier: GPL-2.0-or-later
# NVN host layer (NXN1X docs NVN_HOST_BRIDGE.md §5): the native host's NVN implementation (dm/gfx/nvn) and its Windows Vulkan backend
# (dm/windows/gfx_vk_*.c), built from the host tree beside this source tree (NXN1X/host/dm) into one C static library.
# It is compiled the way the host's own build compiles it (dm/windows/build.py: GNU C, blocks, the Darwin compatibility headers), not with NXN1X's
# warning set and Windows definitions (UNICODE, WIN32_LEAN_AND_MEAN, ...), which this directory clears. The host runtime it needs is dm/nxn1x
# (nvnh_rt.c, nvnh_posix.c); dm/host, posix_win.c (its fault handler, TLS callback and CRT setup) and the host's window are not built.

set(NVN_HOST_DM "${PROJECT_SOURCE_DIR}/../host/dm" CACHE PATH "The host tree's dm folder (NXN1X/host/dm)")
set(NVN_HOST_GAME "01004A4010FEA000" CACHE STRING "Game profile the host layer is built for (dm/games/<id>/game.tsv; Bayonetta 3)")
get_filename_component(NVN_HOST_DM "${NVN_HOST_DM}" ABSOLUTE)
if (NOT EXISTS "${NVN_HOST_DM}/gfx/nvn/hle_nvn.c")
    message(FATAL_ERROR "NVN host layer: ${NVN_HOST_DM} is not the host's dm folder")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
enable_language(C ASM)

# Not NXN1X's directory-wide definitions and warnings: the host is built with its own (build.py CFLAGS)
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "" COMPILE_OPTIONS "")

set(NVN_HOST_GEN "${CMAKE_CURRENT_BINARY_DIR}/gen")
file(MAKE_DIRECTORY "${NVN_HOST_GEN}/${NVN_HOST_GAME}")
# The game profile header, as the host's build makes it (build.py: games/game.py header)
add_custom_command(
    OUTPUT "${NVN_HOST_GEN}/${NVN_HOST_GAME}/game_profile.h"
    COMMAND "${Python3_EXECUTABLE}" "${NVN_HOST_DM}/games/game.py" header "${NVN_HOST_GAME}" "${NVN_HOST_GEN}/${NVN_HOST_GAME}/game_profile.h"
    DEPENDS "${NVN_HOST_DM}/games/game.py" "${NVN_HOST_DM}/games/${NVN_HOST_GAME}/game.tsv"
    VERBATIM)
# Mach-O assembly to COFF, as build.py does (windows/tools/darwin2coff.py)
set(NVN_HOST_ASM)
foreach(s gfx/nvn/nvn_entry.S gfx/nvn/nvn_thunks.S)
    get_filename_component(stem "${s}" NAME_WE)
    set(out "${NVN_HOST_GEN}/${stem}.coff.S")
    add_custom_command(
        OUTPUT "${out}"
        COMMAND "${Python3_EXECUTABLE}" "${NVN_HOST_DM}/windows/tools/darwin2coff.py" "${NVN_HOST_DM}/${s}" "${out}"
        DEPENDS "${NVN_HOST_DM}/${s}" "${NVN_HOST_DM}/windows/tools/darwin2coff.py"
        VERBATIM)
    list(APPEND NVN_HOST_ASM "${out}")
endforeach()

set(NVN_HOST_C
    gfx/nvn/hle_nvn.c gfx/nvn/nvn_dirty.c gfx/nvn/nvn_tex.c gfx/nvn/nvn_capture.c
    gfx/tex/nvn_deswizzle.c gfx/tex/nvn_fmt_mtl.c
    windows/gfx_vk_core.c windows/gfx_vk_exec.c windows/gfx_vk_present.c windows/gfx_vk_shader.c windows/gfx_vk_tex.c windows/gfx_vk_win_glue.c
    windows/nvn_memory_observe.c windows/nvn_memory_content.c windows/nvn_memory_transfer.c
    windows/compat/darwin_compat.c windows/compat/sha256.c windows/compat/sections.c windows/compat/dirent_win.c
    nxn1x/nvnh_rt.c nxn1x/nvnh_posix.c)
list(TRANSFORM NVN_HOST_C PREPEND "${NVN_HOST_DM}/")

add_library(nvn_host STATIC ${NVN_HOST_C} ${NVN_HOST_ASM} "${NVN_HOST_DM}/nxn1x/nvnh_memory_guard.cpp" "${NVN_HOST_GEN}/${NVN_HOST_GAME}/game_profile.h")
target_include_directories(nvn_host PUBLIC "${NVN_HOST_DM}/nxn1x")
target_include_directories(nvn_host PRIVATE
    "$<$<COMPILE_LANGUAGE:C>:${NVN_HOST_DM}/windows/compat/include>" "${NVN_HOST_DM}/host" "${NVN_HOST_DM}/gfx/nvn" "${NVN_HOST_DM}/gfx/tex" "${NVN_HOST_DM}/gfx/nvn/include"
    "${NVN_HOST_DM}/third_party/vulkan-headers/include" "${NVN_HOST_GEN}/${NVN_HOST_GAME}" "${NVN_HOST_DM}/windows" "${NVN_HOST_DM}/windows/vkgen"
    "${PROJECT_SOURCE_DIR}/src")   # the shader lane's C header video_core/nvn/nvn_shader_bridge.h (gfx_vk.h)
target_compile_definitions(nvn_host PRIVATE _GNU_SOURCE _UCRT _FILE_OFFSET_BITS=64 DM_WINDOWS=1)
# x18 is the TEB on Windows: the target's ABI keeps the compiler off it (build.py). The flags are build.py's CFLAGS, on the C sources only
set_source_files_properties(${NVN_HOST_C} PROPERTIES COMPILE_OPTIONS
    "-include;${NVN_HOST_DM}/nxn1x/nvnh_rename.h;-include;${NVN_HOST_DM}/windows/compat/include/darwin_compat.h;-O2;-fblocks;-fno-omit-frame-pointer;-fno-strict-aliasing;-Wall;-Wno-unused-function;-Wno-unused-variable;-Wno-unused-but-set-variable;-Wno-format;-Wno-deprecated-declarations;-Wno-gnu-folding-constant;-Werror=implicit-function-declaration;-Werror=return-type;-Werror=incompatible-pointer-types;-Werror=int-conversion")
set_target_properties(nvn_host PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
find_package(Threads REQUIRED)
target_link_libraries(nvn_host PUBLIC Threads::Threads bcrypt winmm psapi advapi32 user32 gdi32 ole32 ntdll kernel32)

# Retained presentation ABI; no F5 strict activation or lock rewriting.
set_property(SOURCE "${NVN_HOST_DM}/windows/gfx_vk_present.c" APPEND PROPERTY COMPILE_OPTIONS
    -include "${NVN_HOST_DM}/nxn1x/nvnh_memory_guard.h")
