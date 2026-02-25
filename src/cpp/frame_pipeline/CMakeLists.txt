if(APPLE)
    add_library(streetscope_frame_loop STATIC
        frame_loop.cpp
    )
    target_include_directories(streetscope_frame_loop PUBLIC
        ${CMAKE_SOURCE_DIR}/src/cpp/objcpp
    )
    target_link_libraries(streetscope_frame_loop PUBLIC
        streetscope_simd_core
        streetscope_decoder
    )
    target_link_libraries(streetscope_frame_loop PRIVATE streetscope_warnings)
endif()
