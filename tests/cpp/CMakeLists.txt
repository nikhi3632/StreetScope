include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_simd
    test_accumulator.cpp
    test_subtractor.cpp
    test_isp_correction.cpp
)
target_link_libraries(test_simd PRIVATE streetscope_simd_core streetscope_warnings GTest::gtest_main)
set_target_properties(test_simd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Per-target ASan+UBSan in Debug when SANITIZE=auto (macOS default).
# Only instruments the test binary, not the library or pybind11 module.
# LSan is not available on macOS (Apple Clang); on Linux, use docker-asan
# which enables global ASan with LSan for C++ tests.
if(SANITIZE STREQUAL "auto" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_simd PRIVATE -fsanitize=address,undefined)
    target_link_options(test_simd PRIVATE -fsanitize=address,undefined)
endif()

include(GoogleTest)
gtest_discover_tests(test_simd)
