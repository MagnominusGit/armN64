# What the standalone app (Armulators' app/) is for this core: its names, Android package, look and the
# games it opens. CMake writes them into app-info.h for the C code.
set(APP_ID armn64)
set(APP_NAME armN64)
set(APP_WORDMARK ARMN64)
set(APP_ACCENT Green)
set(APP_JAVA_PACKAGE org.armn64.app)
set(APP_JAVA_CLASS Armn64)
# Its version, and how it's built: the CMake option that turns it on, its target, and the packages'
# other CMake arguments.
set(APP_VERSION 0.1)
set(APP_BUILD_OPTION ARMN64_APP)
set(APP_BUILD_TARGET armn64-app)
# The packages run on any 64-bit ARM or x86-64 machine, not only the Cortex-A55's kind.
set(APP_BUILD_ARGS -DARMN64_TUNE=OFF)
# For the desktop: what it is, and the files it opens.
set(APP_COMMENT "Nintendo 64 emulator")
set(APP_MIME application/x-n64-rom application/zip)
# The console's screen: the window is a multiple of it before a game says its own.
set(APP_WINDOW 320 240)
# The game list's files (archives too), and what's looked for inside an archive.
set(APP_EXTENSIONS z64 n64 v64 ndd zip 7z)
set(APP_ARCHIVED z64 n64 v64 ndd)
# The systems: each one's key (its folder setting is <key>_folder), name, and the folder names it's
# found by in the Games folder (its key and these, any case).
set(APP_SYSTEMS n64)
set(APP_SYSTEM_n64 "Nintendo 64" nintendo64 "nintendo 64" 64dd)
# Help texts that name the console's buttons and what the app draws over its picture.
set(APP_HELP_SWAP "Off: the bottom button is the N64's A and the left one its B, as in RetroArch. On: A and B swap, and X and Y.")
set(APP_HELP_SHADERS "The filter.")
