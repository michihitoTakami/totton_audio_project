add_library(rpi_pcm_bridge
    AlsaCapture.cpp
    Options.cpp
    TcpClient.cpp
    logging.cpp
    HwParamsMonitor.cpp
    RtpOptions.cpp
    RtpPipelineBuilder.cpp
)

target_include_directories(rpi_pcm_bridge
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(rpi_pcm_bridge
    PUBLIC
        ALSA::ALSA
        Threads::Threads
)

target_compile_options(rpi_pcm_bridge
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

add_executable(rpi_pcm_bridge_app
    main.cpp
)

target_link_libraries(rpi_pcm_bridge_app
    PRIVATE
        rpi_pcm_bridge
)

target_compile_options(rpi_pcm_bridge_app
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

set_target_properties(rpi_pcm_bridge_app PROPERTIES OUTPUT_NAME "rpi_pcm_bridge")

add_executable(rpi_rtp_sender
    rtp_main.cpp
)

target_link_libraries(rpi_rtp_sender
    PRIVATE
        rpi_pcm_bridge
        Threads::Threads
)

target_compile_options(rpi_rtp_sender
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)
