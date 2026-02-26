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
        streetscope_frame
        "-framework AVFoundation"
        "-framework CoreMedia"
        "-framework CoreVideo"
        "-framework Foundation"
        "-framework QuartzCore"
    )
    target_compile_options(streetscope_decoder PRIVATE -fobjc-arc)

    add_library(streetscope_inference STATIC
        detection.cpp
        letterbox.cpp
        nms.cpp
        coreml_detector.mm
    )
    target_include_directories(streetscope_inference PUBLIC
        ${CMAKE_SOURCE_DIR}/src/cpp/objcpp
    )
    target_include_directories(streetscope_inference SYSTEM PRIVATE
        ${opencv_SOURCE_DIR}/modules/core/include
        ${opencv_SOURCE_DIR}/modules/imgproc/include
        ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(streetscope_inference PRIVATE streetscope_warnings)
    target_link_libraries(streetscope_inference PUBLIC
        streetscope_frame
        opencv_core opencv_imgproc
        "-framework Accelerate"
        "-framework CoreML"
        "-framework CoreVideo"
        "-framework Foundation"
    )
    target_compile_options(streetscope_inference PRIVATE -fobjc-arc)

    add_library(streetscope_metal STATIC
        metal_tone_mapper.mm
    )
    target_include_directories(streetscope_metal PUBLIC
        ${CMAKE_SOURCE_DIR}/src/cpp/objcpp
    )
    target_link_libraries(streetscope_metal PRIVATE streetscope_warnings)
    target_link_libraries(streetscope_metal PUBLIC
        streetscope_frame
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework CoreVideo"
        "-framework Foundation"
    )
    target_compile_options(streetscope_metal PRIVATE -fobjc-arc)
endif()
