execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_CMAKE_RESULT
)
if(NOT PYBIND11_CMAKE_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 not found: ensure 'pip install pybind11' in your environment")
endif()
find_package(pybind11 REQUIRED PATHS ${PYBIND11_CMAKE_DIR})

pybind11_add_module(streetscope_simd simd_module.cpp)
target_link_libraries(streetscope_simd PRIVATE streetscope_simd_core streetscope_warnings)
set_target_properties(streetscope_simd PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
