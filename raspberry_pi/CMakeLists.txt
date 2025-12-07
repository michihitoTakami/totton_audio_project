cmake_minimum_required(VERSION 3.18)

project(raspberry_pi_pcm_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckSymbolExists)
check_symbol_exists(socket "sys/socket.h" HAVE_SOCKET)
if(NOT HAVE_SOCKET)
    message(FATAL_ERROR "POSIX sockets are required but were not found.")
endif()

find_package(ALSA REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_subdirectory(src)

include(CTest)
if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    include(GoogleTest)

    add_executable(rpi_capture_tests
        tests/test_cli_options.cpp
        tests/test_alsa_capture.cpp
    )

    target_link_libraries(rpi_capture_tests
        PRIVATE
            GTest::gtest_main
            rpi_pcm_bridge
            ALSA::ALSA
    )

    target_include_directories(rpi_capture_tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    gtest_discover_tests(rpi_capture_tests)
endif()
