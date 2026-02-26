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
    test_pipeline.cpp
    test_tone_map.cpp
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

add_executable(test_frame_pipeline
    test_ring_buffer.cpp
)
target_link_libraries(test_frame_pipeline PRIVATE streetscope_frame streetscope_warnings GTest::gtest_main)
set_target_properties(test_frame_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
if(SANITIZE STREQUAL "auto" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_frame_pipeline PRIVATE -fsanitize=address,undefined)
    target_link_options(test_frame_pipeline PRIVATE -fsanitize=address,undefined)
endif()

if(APPLE AND TARGET streetscope_frame_loop)
    add_executable(test_frame_loop
        test_frame_loop.cpp
    )
    target_link_libraries(test_frame_loop PRIVATE
        streetscope_frame_loop streetscope_warnings GTest::gtest_main)
    set_target_properties(test_frame_loop PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    if(SANITIZE STREQUAL "auto" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(test_frame_loop PRIVATE -fsanitize=address,undefined)
        target_link_options(test_frame_loop PRIVATE -fsanitize=address,undefined)
    endif()
endif()

if(APPLE AND TARGET streetscope_metal)
    add_executable(test_metal_tone_map
        test_metal_tone_map.mm
    )
    target_link_libraries(test_metal_tone_map PRIVATE
        streetscope_metal streetscope_simd_core streetscope_warnings
        GTest::gtest_main
        "-framework CoreML"
        "-framework Metal"
        "-framework CoreVideo"
        "-framework Foundation"
    )
    set_target_properties(test_metal_tone_map PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

if(APPLE AND TARGET streetscope_inference)
    add_executable(test_inference
        test_detection.cpp
        test_letterbox.cpp
        test_nms.cpp
    )
    target_link_libraries(test_inference PRIVATE
        streetscope_inference streetscope_warnings GTest::gtest_main)
    set_target_properties(test_inference PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    if(SANITIZE STREQUAL "auto" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(test_inference PRIVATE -fsanitize=address,undefined)
        target_link_options(test_inference PRIVATE -fsanitize=address,undefined)
    endif()
endif()

include(GoogleTest)
gtest_discover_tests(test_simd)
gtest_discover_tests(test_frame_pipeline)
if(TARGET test_frame_loop)
    gtest_discover_tests(test_frame_loop)
endif()
if(TARGET test_metal_tone_map)
    gtest_discover_tests(test_metal_tone_map
        DISCOVERY_TIMEOUT 30
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
if(TARGET test_inference)
    gtest_discover_tests(test_inference DISCOVERY_TIMEOUT 30)
endif()
