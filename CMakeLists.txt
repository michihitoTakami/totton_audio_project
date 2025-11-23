cmake_minimum_required(VERSION 3.18)
project(GPUAudioUpsampler LANGUAGES CXX CUDA)

# Standard install directories (bin, lib, etc.)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Export compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)
pkg_check_modules(ALSA REQUIRED alsa)

# ZeroMQ for IPC communication
pkg_check_modules(ZMQ REQUIRED libzmq)

# Fetch dependencies
include(FetchContent)

# cppzmq (C++ bindings for ZeroMQ - header-only)
FetchContent_Declare(
    cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG v4.10.0
)
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# nlohmann_json (header-only JSON library)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

# GoogleTest (testing framework)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# Prevent GoogleTest from overriding compiler/linker options
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlohmann_json googletest cppzmq)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data/coefficients
    ${SNDFILE_INCLUDE_DIRS}
    ${PIPEWIRE_INCLUDE_DIRS}
    ${ALSA_INCLUDE_DIRS}
)

# Core library (static) - Phase 2 GPU upsampling engine
add_library(gpu_upsampler_core STATIC
    src/audio_io.cpp
    src/convolution_engine.cu
)

# CUDA architecture (RTX 2070 Super = Compute Capability 7.5)
set_target_properties(gpu_upsampler_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries for core
target_link_libraries(gpu_upsampler_core PUBLIC
    CUDA::cudart
    CUDA::cufft
    CUDA::nvml
    ${SNDFILE_LIBRARIES}
)

# Command-line executable
add_executable(gpu_upsampler
    src/main.cpp
    src/config_loader.cpp
)

# Link against core library
target_link_libraries(gpu_upsampler
    gpu_upsampler_core
    nlohmann_json::nlohmann_json
)

# Compiler flags for core library
target_compile_options(gpu_upsampler_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Compiler flags for executable
target_compile_options(gpu_upsampler PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# PipeWire Daemon executable
add_executable(gpu_upsampler_daemon src/pipewire_daemon.cpp)

# Link against core library and PipeWire
target_link_libraries(gpu_upsampler_daemon
    gpu_upsampler_core
    ${PIPEWIRE_LIBRARIES}
)

# Compiler flags for daemon
target_compile_options(gpu_upsampler_daemon PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# ALSA Direct Output Daemon executable
add_executable(gpu_upsampler_alsa
    src/alsa_daemon.cpp
    src/config_loader.cpp
    src/eq_parser.cpp
    src/eq_to_fir.cpp
)

# Link against core library, PipeWire, ALSA, JSON, and ZeroMQ
target_link_libraries(gpu_upsampler_alsa
    gpu_upsampler_core
    ${PIPEWIRE_LIBRARIES}
    ${ALSA_LIBRARIES}
    nlohmann_json::nlohmann_json
    ${ZMQ_LIBRARIES}
)
target_include_directories(gpu_upsampler_alsa PRIVATE
    ${ZMQ_INCLUDE_DIRS}
    ${cppzmq_SOURCE_DIR}
)

# Compiler flags for ALSA daemon
target_compile_options(gpu_upsampler_alsa PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# EQ Test executable
add_executable(test_eq
    src/test_eq.cpp
    src/eq_parser.cpp
    src/eq_to_fir.cpp
)

target_link_libraries(test_eq
    gpu_upsampler_core
)

target_compile_options(test_eq PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# ZeroMQ Communication Library (CPU-only, no GPU required)
add_library(zeromq_interface STATIC
    src/zeromq_interface.cpp
)
target_link_libraries(zeromq_interface PUBLIC
    nlohmann_json::nlohmann_json
    ${ZMQ_LIBRARIES}
)
target_include_directories(zeromq_interface PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ZMQ_INCLUDE_DIRS}
    ${cppzmq_SOURCE_DIR}
)
target_compile_options(zeromq_interface PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# DAC Capability Library (CPU-only, ALSA required)
add_library(dac_capability STATIC
    src/dac_capability.cpp
)
target_link_libraries(dac_capability PUBLIC
    ${ALSA_LIBRARIES}
)
target_include_directories(dac_capability PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ALSA_INCLUDE_DIRS}
)
target_compile_options(dac_capability PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Auto-Negotiation Library (CPU-only)
add_library(auto_negotiation STATIC
    src/auto_negotiation.cpp
)
target_link_libraries(auto_negotiation PUBLIC
    dac_capability
)
target_include_directories(auto_negotiation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(auto_negotiation PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Soft Mute Library (CPU-only, for glitch-free audio transitions)
add_library(soft_mute STATIC
    src/soft_mute.cpp
)
target_include_directories(soft_mute PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(soft_mute PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Installation for command-line tool and library
install(TARGETS gpu_upsampler DESTINATION bin)
install(TARGETS gpu_upsampler_daemon DESTINATION bin)
install(TARGETS gpu_upsampler_alsa DESTINATION bin)
install(TARGETS gpu_upsampler_core DESTINATION lib)
install(TARGETS zeromq_interface DESTINATION lib)
install(TARGETS dac_capability DESTINATION lib)
install(TARGETS auto_negotiation DESTINATION lib)
install(TARGETS soft_mute DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/gpu_upsampler)

# Install data files (filter coefficients and default config)
# These are required at runtime and referenced by relative paths
set(GPU_UPSAMPLER_DATADIR "${CMAKE_INSTALL_FULL_DATADIR}/gpu_upsampler")
install(DIRECTORY data/coefficients
        DESTINATION "${CMAKE_INSTALL_DATADIR}/gpu_upsampler/data"
        FILES_MATCHING PATTERN "*.bin" PATTERN "*.json")
install(FILES config.json
        DESTINATION "${CMAKE_INSTALL_DATADIR}/gpu_upsampler"
        OPTIONAL)  # Optional: user may not have config.json yet

# Install systemd user service file
# Default location: /usr/lib/systemd/user/ (system-wide for all users)
# User can override with -DSYSTEMD_USER_UNIT_DIR=...
if(NOT DEFINED SYSTEMD_USER_UNIT_DIR)
    # Try pkg-config first
    execute_process(
        COMMAND pkg-config --variable=systemduserunitdir systemd
        OUTPUT_VARIABLE SYSTEMD_USER_UNIT_DIR_PKG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(SYSTEMD_USER_UNIT_DIR_PKG)
        set(SYSTEMD_USER_UNIT_DIR "${SYSTEMD_USER_UNIT_DIR_PKG}")
    else()
        set(SYSTEMD_USER_UNIT_DIR "${CMAKE_INSTALL_PREFIX}/lib/systemd/user")
    endif()
endif()

# Generate service file with correct install paths
configure_file(
    systemd/gpu-upsampler.service.in
    ${CMAKE_BINARY_DIR}/gpu-upsampler.service
    @ONLY
)
install(FILES ${CMAKE_BINARY_DIR}/gpu-upsampler.service
        DESTINATION "${SYSTEMD_USER_UNIT_DIR}")

# ============================================================
# Testing
# ============================================================
enable_testing()
include(GoogleTest)

# CPU-only unit tests (no GPU required)
# Tests: RateFamily, EQ Parser, EQ to FIR, Config Loader
add_executable(cpu_tests
    tests/cpp/test_rate_family.cpp
    tests/cpp/test_eq_parser.cpp
    tests/cpp/test_eq_to_fir.cpp
    tests/cpp/test_config_loader.cpp
    # Source files needed for tests (not in gpu_upsampler_core)
    src/eq_parser.cpp
    src/eq_to_fir.cpp
    src/config_loader.cpp
)
target_link_libraries(cpu_tests
    GTest::gtest_main
    gpu_upsampler_core
    nlohmann_json::nlohmann_json
)
target_include_directories(cpu_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(cpu_tests)

# GPU integration tests (requires CUDA GPU)
add_executable(gpu_tests
    tests/gpu/test_convolution_engine.cu
)
target_link_libraries(gpu_tests
    GTest::gtest_main
    gpu_upsampler_core
)
target_include_directories(gpu_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
set_target_properties(gpu_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)
gtest_discover_tests(gpu_tests)

# ZeroMQ communication tests (CPU-only)
add_executable(zmq_tests
    tests/cpp/test_zeromq_interface.cpp
)
target_link_libraries(zmq_tests
    GTest::gtest_main
    zeromq_interface
)
gtest_discover_tests(zmq_tests)

# Auto-Negotiation tests (CPU-only, no ALSA hardware required)
add_executable(auto_negotiation_tests
    tests/cpp/test_auto_negotiation.cpp
)
target_link_libraries(auto_negotiation_tests
    GTest::gtest_main
    auto_negotiation
    gpu_upsampler_core
)
target_include_directories(auto_negotiation_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(auto_negotiation_tests)

# Soft Mute tests (CPU-only)
add_executable(soft_mute_tests
    tests/cpp/test_soft_mute.cpp
)
target_link_libraries(soft_mute_tests
    GTest::gtest_main
    soft_mute
)
target_include_directories(soft_mute_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(soft_mute_tests)
