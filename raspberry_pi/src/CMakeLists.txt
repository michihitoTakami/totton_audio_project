add_library(rpi_pcm_bridge
    AlsaCapture.cpp
    Options.cpp
    TcpClient.cpp
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
