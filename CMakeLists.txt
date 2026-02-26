cmake_minimum_required(VERSION 3.20)
if(APPLE)
    project(streetscope_cpp LANGUAGES CXX OBJCXX)
else()
    project(streetscope_cpp LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Sanitizer mode ---
#
# auto   → Per-target ASan on test binaries only (macOS default, see tests/cpp/)
# ON     → Global ASan+UBSan on all targets (Docker).
#           LSan (leak detection) is built into ASan on Linux and runs
#           automatically for C++ tests. Disabled for Python tests via
#           ASAN_OPTIONS=detect_leaks=0 (Python's arena allocator causes
#           false positives). macOS Apple Clang does not support LSan.
# thread → Global TSan on all targets (Docker). Detects data races,
#           deadlocks, and lock-order inversions. Mutually exclusive
#           with ASan — cannot combine them in the same build.
# OFF    → No sanitizers. Used for Valgrind builds (Valgrind is also
#           incompatible with ASan/TSan instrumented binaries).
#
set(SANITIZE "auto" CACHE STRING "Sanitizer mode: auto, ON, thread, OFF")

# --- Compiler flags ---
# Warnings applied globally (including third-party code like GoogleTest).
# -Werror is applied only to our targets via the streetscope_warnings
# interface library — third-party code we don't control should not
# break our build when they introduce a new warning.
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g)
else()
    add_compile_options(-O2)
endif()

# Interface library carrying -Werror and sanitizer flags for our targets only.
# Third-party code (OpenCV, GoogleTest) does NOT link this, so it builds
# without sanitizers — same object files reused across asan/tsan/valgrind modes.
# Usage: target_link_libraries(<our_target> PRIVATE streetscope_warnings)
add_library(streetscope_warnings INTERFACE)
target_compile_options(streetscope_warnings INTERFACE -Werror)

if(SANITIZE STREQUAL "ON")
    target_compile_options(streetscope_warnings INTERFACE
        -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(streetscope_warnings INTERFACE
        -fsanitize=address,undefined)
elseif(SANITIZE STREQUAL "thread")
    target_compile_options(streetscope_warnings INTERFACE
        -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(streetscope_warnings INTERFACE
        -fsanitize=thread)
endif()

# --- OpenCV (via FetchContent, Apple only) ---
# All OpenCV-dependent code (inference, decoder) is macOS-only.
# Skipping on Linux avoids a ~10 min build in Docker sanitizer runs.
if(APPLE)
    include(FetchContent)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_java OFF CACHE BOOL "" FORCE)
    # Only build the modules we link: core, imgproc (skips dnn, video, calib3d, etc.)
    set(BUILD_LIST "core,imgproc" CACHE STRING "" FORCE)
    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG        4.13.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(opencv)
endif()

# --- Include directories ---
include_directories(${CMAKE_SOURCE_DIR}/include)

# --- Subdirectories ---
add_subdirectory(src/cpp/simd)
add_subdirectory(src/cpp/plan_executor)
add_subdirectory(src/cpp/objcpp)
add_subdirectory(src/cpp/frame_pipeline)
enable_testing()
add_subdirectory(tests/cpp)
add_subdirectory(src/cpp/bindings)
