if(APPLE)
    enable_language(OBJCXX)

    add_library(streetscope_decoder STATIC
        videotoolbox_decoder.mm
    )
    target_include_directories(streetscope_decoder PUBLIC
        ${CMAKE_SOURCE_DIR}/src/cpp/objcpp
    )
    target_link_libraries(streetscope_decoder PRIVATE streetscope_warnings)
    target_link_libraries(streetscope_decoder PUBLIC
        "-framework AVFoundation"
        "-framework CoreMedia"
        "-framework CoreVideo"
        "-framework Foundation"
        "-framework QuartzCore"
    )
    target_compile_options(streetscope_decoder PRIVATE -fobjc-arc)
endif()
