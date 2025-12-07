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

include(CheckIncludeFileCXX)
include(CheckCXXSymbolExists)
include(FetchContent)

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
    src/tcp_server.cpp
    src/alsa_playback.cpp
    src/pcm_stream_handler.cpp
    src/pcm_header.cpp
)

add_executable(jetson_pcm_receiver
    src/main.cpp
)

target_include_directories(jetson_pcm_receiver_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ALSA_INCLUDE_DIRS}
)

target_link_libraries(jetson_pcm_receiver_lib
    PUBLIC
        ${ALSA_LIBRARIES}
        Threads::Threads
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

message(STATUS "ALSA include dirs: ${ALSA_INCLUDE_DIRS}")
message(STATUS "ALSA libraries: ${ALSA_LIBRARIES}")

# =========================
# Tests
# =========================
enable_testing()

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(jetson_pcm_receiver_tests
    tests/test_pcm_header.cpp
    tests/test_tcp_server.cpp
    tests/test_pcm_stream_handler.cpp
    tests/test_alsa_playback.cpp
)

target_link_libraries(jetson_pcm_receiver_tests
    PRIVATE
        jetson_pcm_receiver_lib
        GTest::gtest_main
)

target_include_directories(jetson_pcm_receiver_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_test(NAME jetson_pcm_receiver_tests COMMAND jetson_pcm_receiver_tests)
