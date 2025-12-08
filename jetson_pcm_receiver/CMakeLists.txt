cmake_minimum_required(VERSION 3.18)
project(jetson_pcm_receiver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(JETSON_PCM_ENABLE_WARNINGS "Enable warning flags for jetson-pcm-receiver" ON)

include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(ALSA REQUIRED alsa)
pkg_check_modules(ZMQ REQUIRED libzmq)

# Disable bundled dependency tests to avoid phantom CTest entries
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)

include(CheckIncludeFileCXX)
include(CheckCXXSymbolExists)
include(FetchContent)
include(CTest)

check_include_file_cxx("sys/socket.h" HAVE_SYS_SOCKET_H)
check_include_file_cxx("netdb.h" HAVE_NETDB_H)
check_cxx_symbol_exists(getaddrinfo "sys/types.h;sys/socket.h;netdb.h" HAVE_GETADDRINFO)
check_cxx_symbol_exists(socket "sys/types.h;sys/socket.h" HAVE_SOCKET_SYM)

if(NOT HAVE_SYS_SOCKET_H OR NOT HAVE_NETDB_H OR NOT HAVE_GETADDRINFO OR NOT HAVE_SOCKET_SYM)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(STATUS "Forcing BSD socket availability on Linux (headers should exist).")
        set(HAVE_SYS_SOCKET_H TRUE)
        set(HAVE_NETDB_H TRUE)
        set(HAVE_GETADDRINFO TRUE)
        set(HAVE_SOCKET_SYM TRUE)
    else()
        message(WARNING "BSD socket APIs were not detected; build may fail on this platform.")
    endif()
endif()

add_library(jetson_pcm_receiver_lib
    src/app_options.cpp
    src/logging.cpp
    src/tcp_server.cpp
    src/alsa_playback.cpp
    src/xrun_detector.cpp
    src/pcm_stream_handler.cpp
    src/pcm_header.cpp
    src/status_tracker.cpp
    src/zmq_status_server.cpp
    src/output_device.cpp
    ../src/daemon/control/zmq_server.cpp
)

add_executable(jetson_pcm_receiver
    src/main.cpp
)

target_include_directories(jetson_pcm_receiver_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${ALSA_INCLUDE_DIRS}
        ${ZMQ_INCLUDE_DIRS}
)

target_link_libraries(jetson_pcm_receiver_lib
    PUBLIC
        ${ALSA_LIBRARIES}
        Threads::Threads
        ${ZMQ_LIBRARIES}
        cppzmq
        nlohmann_json::nlohmann_json
)

target_link_libraries(jetson_pcm_receiver PRIVATE jetson_pcm_receiver_lib)

target_compile_options(jetson_pcm_receiver_lib PRIVATE
    $<$<BOOL:${JETSON_PCM_ENABLE_WARNINGS}>:-Wall>
    $<$<BOOL:${JETSON_PCM_ENABLE_WARNINGS}>:-Wextra>
    $<$<BOOL:${JETSON_PCM_ENABLE_WARNINGS}>:-Wpedantic>
)

if(JETSON_PCM_ENABLE_WARNINGS)
    target_compile_options(jetson_pcm_receiver PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

FetchContent_Declare(
    cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG v4.10.0
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(cppzmq nlohmann_json)

message(STATUS "ALSA include dirs: ${ALSA_INCLUDE_DIRS}")
message(STATUS "ALSA libraries: ${ALSA_LIBRARIES}")

# =========================
# Tests
# =========================
if(BUILD_TESTING)
    enable_testing()

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(jetson_pcm_receiver_tests
        tests/test_app_options.cpp
        tests/test_pcm_header.cpp
        tests/test_tcp_server.cpp
        tests/test_pcm_stream_handler.cpp
        tests/test_alsa_playback.cpp
        tests/test_zmq_status_server.cpp
        tests/test_output_device.cpp
        tests/test_xrun_detector.cpp
    )

    target_link_libraries(jetson_pcm_receiver_tests
        PRIVATE
            jetson_pcm_receiver_lib
            GTest::gtest_main
    )

    target_include_directories(jetson_pcm_receiver_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    add_test(NAME jetson_pcm_receiver_tests COMMAND jetson_pcm_receiver_tests)
endif()
