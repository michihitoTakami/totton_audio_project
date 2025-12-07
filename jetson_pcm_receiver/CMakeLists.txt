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

include(CheckIncludeFile)
include(CheckSymbolExists)

check_include_file("sys/socket.h" HAVE_SYS_SOCKET_H)
check_include_file("netdb.h" HAVE_NETDB_H)
check_symbol_exists(getaddrinfo "sys/types.h;sys/socket.h;netdb.h" HAVE_GETADDRINFO)
check_symbol_exists(socket "sys/types.h;sys/socket.h" HAVE_SOCKET_SYM)

if(NOT HAVE_SYS_SOCKET_H OR NOT HAVE_NETDB_H OR NOT HAVE_GETADDRINFO OR NOT HAVE_SOCKET_SYM)
    message(FATAL_ERROR "BSD socket APIs are unavailable. Ensure networking headers and libc provide getaddrinfo/socket.")
endif()

add_executable(jetson_pcm_receiver
    src/main.cpp
    src/tcp_server.cpp
    src/alsa_playback.cpp
    src/pcm_stream_handler.cpp
)

target_include_directories(jetson_pcm_receiver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ALSA_INCLUDE_DIRS}
)

target_link_libraries(jetson_pcm_receiver PRIVATE
    ${ALSA_LIBRARIES}
    Threads::Threads
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

