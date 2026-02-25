add_library(streetscope_simd_core STATIC
    accumulator.cpp
    subtractor.cpp
    isp_correction.cpp
    pipeline.cpp
    tone_map.cpp
)
target_link_libraries(streetscope_simd_core PRIVATE streetscope_warnings)
