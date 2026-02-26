add_library(streetscope_plan_executor STATIC
    plan_executor.cpp
)

target_include_directories(streetscope_plan_executor PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(streetscope_plan_executor
    PUBLIC streetscope_simd_core
    PRIVATE streetscope_warnings
)
