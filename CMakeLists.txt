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
pkg_check_modules(ALSA REQUIRED alsa)

# ZeroMQ for IPC communication
pkg_check_modules(ZMQ REQUIRED libzmq)

# systemd for daemon notification (optional, for Type=notify support)
# Service file Type is automatically set based on libsystemd availability
pkg_check_modules(SYSTEMD libsystemd)
if(SYSTEMD_FOUND)
    set(HAVE_SYSTEMD TRUE)
    message(STATUS "libsystemd found - systemd notification enabled (Type=notify)")
else()
    set(HAVE_SYSTEMD FALSE)
    message(STATUS "libsystemd not found - systemd notification disabled (Type=simple)")
    message(STATUS "  To enable: sudo apt-get install libsystemd-dev && cmake -B build")
endif()

# NVML for GPU utilization monitoring (optional)
# On minimal Jetson environments or some container setups, NVML may not be available
if(TARGET CUDA::nvml)
    set(HAVE_NVML TRUE)
    message(STATUS "NVML found - GPU utilization monitoring enabled")
else()
    set(HAVE_NVML FALSE)
    message(STATUS "NVML not found - GPU utilization monitoring disabled (stub)")
endif()

# Detect platform: Jetson vs PC
# Jetson Orin Nano = SM 8.7 (Ampere), RTX 2070 Super = SM 7.5 (Turing)
if(EXISTS "/etc/nv_tegra_release")
    set(GPU_CUDA_ARCH "87")
    message(STATUS "Detected Jetson platform - CUDA arch ${GPU_CUDA_ARCH}")
else()
    set(GPU_CUDA_ARCH "75")
    message(STATUS "Detected PC platform - CUDA arch ${GPU_CUDA_ARCH}")
endif()

option(GPU_UPSAMPLER_USE_FLOAT64 "Build GPU upsampler with float64 pipeline" OFF)
if(GPU_UPSAMPLER_USE_FLOAT64)
    add_compile_definitions(GPU_UPSAMPLER_USE_FLOAT64)
    message(STATUS "GPU upsampler precision: float64 (double)")
else()
    message(STATUS "GPU upsampler precision: float32")
endif()

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

# spdlog (structured logging library)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
)

FetchContent_MakeAvailable(nlohmann_json googletest cppzmq spdlog)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data/coefficients
    ${SNDFILE_INCLUDE_DIRS}
    ${ALSA_INCLUDE_DIRS}
)

# Core library (static) - Phase 2 GPU upsampling engine
# Split into multiple files for maintainability (Issue #104)
add_library(gpu_upsampler_core STATIC
    src/audio/audio_io.cpp
    src/gpu/convolution_kernels.cu
    src/gpu/cuda_utils.cu
    src/gpu/partition_plan.cpp
    src/gpu/gpu_upsampler_core.cu
    src/gpu/gpu_upsampler_multi_rate.cu
    src/gpu/gpu_upsampler_streaming.cu
    src/gpu/gpu_upsampler_eq.cu
    src/gpu/four_channel_fir.cu
    src/audio/phase_alignment.cpp
    src/hrtf/woodworth_model.cpp
)

# CUDA architecture (auto-detected: Jetson=87, PC=75)
set_target_properties(gpu_upsampler_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${GPU_CUDA_ARCH}"
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries for core
target_link_libraries(gpu_upsampler_core PUBLIC
    CUDA::cudart
    CUDA::cufft
    ${SNDFILE_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Conditionally link NVML if available
if(HAVE_NVML)
    target_link_libraries(gpu_upsampler_core PUBLIC CUDA::nvml)
    target_compile_definitions(gpu_upsampler_core PUBLIC HAVE_NVML)
endif()

# Command-line executable
add_executable(gpu_upsampler
    src/main.cpp
)

# Link against core library and eq_support
target_link_libraries(gpu_upsampler
    gpu_upsampler_core
    eq_support
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

# ALSA Direct Output Daemon executable
add_executable(gpu_upsampler_alsa
    src/alsa_daemon.cpp
    src/core/partition_runtime_utils.cpp
    src/daemon/app/process_resources.cpp
    src/daemon/audio/crossfeed_manager.cpp
    src/daemon/audio/upsampler_builder.cpp
    src/daemon/audio_pipeline/audio_pipeline.cpp
    src/daemon/audio_pipeline/filter_manager.cpp
    src/daemon/audio_pipeline/headroom_controller.cpp
    src/daemon/audio_pipeline/rate_switcher.cpp
    src/daemon/audio_pipeline/soft_mute_runner.cpp
    src/daemon/audio_pipeline/switch_actions.cpp
    src/daemon/input/loopback_capture.cpp
    src/daemon/input/i2s_capture.cpp
    src/daemon/control/handlers/handler_registry.cpp
    src/daemon/core/pid_lock.cpp
    src/daemon/pcm/dac_manager.cpp
    src/daemon/control/control_plane.cpp
    src/daemon/output/alsa_output.cpp
    src/daemon/output/alsa_pcm_controller.cpp
    src/daemon/output/alsa_output_thread.cpp
    src/daemon/output/playback_buffer_manager.cpp
    src/daemon/output/playback_buffer_access.cpp
    src/daemon/control/zmq_server.cpp
    src/daemon/metrics/stats_file.cpp
    src/daemon/metrics/runtime_stats.cpp
    src/daemon/shutdown_manager.cpp
)

target_include_directories(gpu_upsampler_alsa PRIVATE src)

# Link against core library, eq_support, ALSA, ZeroMQ, SoftMute, FallbackManager, and Logging
target_link_libraries(gpu_upsampler_alsa
    gpu_upsampler_core
    eq_support
    soft_mute
    filter_headroom
    graceful_shutdown
    fallback_manager
    zeromq_interface
    dac_capability
    logging
    ${ALSA_LIBRARIES}
    ${ZMQ_LIBRARIES}
)

# Conditionally link libsystemd for Type=notify support
if(HAVE_SYSTEMD)
    target_link_libraries(gpu_upsampler_alsa ${SYSTEMD_LIBRARIES})
    target_compile_definitions(gpu_upsampler_alsa PRIVATE HAVE_SYSTEMD)
    target_include_directories(gpu_upsampler_alsa PRIVATE ${SYSTEMD_INCLUDE_DIRS})
endif()

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
)

target_link_libraries(test_eq
    gpu_upsampler_core
    eq_support
)

target_compile_options(test_eq PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Error Codes Library (used by zeromq_interface and other components)
add_library(error_codes STATIC
    src/core/error_codes.cpp
)
target_include_directories(error_codes PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(error_codes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Logging Library (Issue #43 - structured logging with spdlog)
add_library(logging STATIC
    src/logging/logger.cpp
    src/logging/metrics.cpp
)
target_link_libraries(logging PUBLIC
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)
target_include_directories(logging PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(logging PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)
# Conditionally add NVML support for GPU metrics
if(HAVE_NVML)
    target_link_libraries(logging PUBLIC CUDA::nvml)
    target_compile_definitions(logging PUBLIC HAVE_NVML)
endif()

# Ensure spdlog headers propagate to targets that include logger.h.
target_link_libraries(gpu_upsampler_core PUBLIC logging)

# ZeroMQ Communication Library (CPU-only, no GPU required)
add_library(zeromq_interface STATIC
    src/network/zeromq_interface.cpp
    src/core/base64.cpp
)
target_link_libraries(zeromq_interface PUBLIC
    error_codes
    logging
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
    src/io/dac_capability.cpp
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
    src/audio/auto_negotiation.cpp
)
target_link_libraries(auto_negotiation PUBLIC
    dac_capability
)
target_include_directories(auto_negotiation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_options(auto_negotiation PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Soft Mute Library (CPU-only, for glitch-free audio transitions)
add_library(soft_mute STATIC
    src/audio/soft_mute.cpp
)
target_include_directories(soft_mute PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(soft_mute PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Filter headroom helpers (CPU-only)
add_library(filter_headroom STATIC
    src/audio/filter_headroom.cpp
)
target_include_directories(filter_headroom PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(filter_headroom PUBLIC
    nlohmann_json::nlohmann_json
    logging
)
target_compile_options(filter_headroom PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Graceful Shutdown Library (CPU-only, for async-signal-safe shutdown)
add_library(graceful_shutdown STATIC
    src/graceful_shutdown.cpp
)
target_include_directories(graceful_shutdown PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(graceful_shutdown PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Fallback Manager Library (CPU-only, Issue #139)
add_library(fallback_manager STATIC
    src/audio/fallback_manager.cpp
)
target_link_libraries(fallback_manager PUBLIC
    logging
)
target_include_directories(fallback_manager PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(fallback_manager PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# De-limiter inference backend abstraction (CPU-only, Issue #1017)
add_library(delimiter_inference STATIC
    src/delimiter/inference_backend.cpp
)
target_link_libraries(delimiter_inference PUBLIC
    logging
)
target_include_directories(delimiter_inference PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(delimiter_inference PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# EQ Support Library (CPU-only, Issue #107)
# Consolidates EQ parsing, FIR generation, and config loading
# No GPU dependencies - can be used independently of gpu_upsampler_core
add_library(eq_support STATIC
    src/audio/eq_parser.cpp
    src/audio/eq_to_fir.cpp
    src/core/config_loader.cpp
)
target_link_libraries(eq_support PUBLIC
    nlohmann_json::nlohmann_json
)
target_link_libraries(eq_support PUBLIC logging)
target_include_directories(eq_support PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(eq_support PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
)

# Installation for command-line tool and library
install(TARGETS gpu_upsampler DESTINATION bin)
install(TARGETS gpu_upsampler_alsa DESTINATION bin)
install(TARGETS gpu_upsampler_core DESTINATION lib)
install(TARGETS zeromq_interface DESTINATION lib)
install(TARGETS dac_capability DESTINATION lib)
install(TARGETS auto_negotiation DESTINATION lib)
install(TARGETS soft_mute DESTINATION lib)
install(TARGETS logging DESTINATION lib)
install(TARGETS eq_support DESTINATION lib)
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

# Set systemd service type based on libsystemd availability
if(HAVE_SYSTEMD)
    set(SYSTEMD_SERVICE_TYPE "notify")
    set(SYSTEMD_NOTIFY_ACCESS "NotifyAccess=main")
    set(SYSTEMD_WATCHDOG_SEC "# Watchdog: Daemon sends WATCHDOG=1 every 100ms, systemd expects it within 5s\nWatchdogSec=5")
else()
    set(SYSTEMD_SERVICE_TYPE "simple")
    set(SYSTEMD_NOTIFY_ACCESS "# NotifyAccess not needed for Type=simple")
    set(SYSTEMD_WATCHDOG_SEC "# Watchdog disabled (libsystemd not available at build time)")
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
    tests/cpp/audio/test_rate_family.cpp
    tests/cpp/audio/test_eq_parser.cpp
    tests/cpp/audio/test_eq_to_fir.cpp
    tests/cpp/core/test_config_loader.cpp
    tests/cpp/delimiter/test_inference_backend.cpp
    tests/cpp/e2e/test_partition_plan.cpp
    tests/cpp/audio/test_phase_alignment.cpp
    tests/cpp/audio/test_playback_buffer_threshold.cpp
    tests/cpp/daemon/test_alsa_write_loop.cpp
    tests/cpp/daemon/test_runtime_stats.cpp
    tests/cpp/daemon/test_stream_buffer_sizing.cpp
    tests/cpp/daemon/test_playback_buffer_access.cpp
    tests/cpp/daemon/test_switch_actions.cpp
    tests/cpp/daemon/test_i2s_capture_config.cpp
    tests/cpp/audio/test_audio_pipeline.cpp
    tests/cpp/core/test_precision_traits.cpp
    src/daemon/audio/crossfeed_manager.cpp
    src/daemon/audio_pipeline/audio_pipeline.cpp
    src/daemon/audio_pipeline/filter_manager.cpp
    src/daemon/audio_pipeline/headroom_controller.cpp
    src/daemon/audio_pipeline/rate_switcher.cpp
    src/daemon/audio_pipeline/soft_mute_runner.cpp
    src/daemon/audio_pipeline/switch_actions.cpp
    src/daemon/control/handlers/handler_registry.cpp
    src/daemon/input/loopback_capture.cpp
    src/daemon/input/i2s_capture.cpp
    src/daemon/metrics/stats_file.cpp
    src/daemon/metrics/runtime_stats.cpp
    src/daemon/pcm/dac_manager.cpp
    src/daemon/output/alsa_output.cpp
    src/daemon/output/playback_buffer_manager.cpp
    src/daemon/output/playback_buffer_access.cpp
)
target_link_libraries(cpu_tests
    GTest::gtest_main
    gpu_upsampler_core
    eq_support
    logging
    filter_headroom
    fallback_manager
    delimiter_inference
    soft_mute
    dac_capability
    ${ALSA_LIBRARIES}
)
target_include_directories(cpu_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
gtest_discover_tests(cpu_tests)

# GPU integration tests (requires CUDA GPU)
add_executable(gpu_tests
    tests/gpu/test_convolution_engine.cu
    tests/gpu/test_four_channel_fir.cu
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
    CUDA_ARCHITECTURES "${GPU_CUDA_ARCH}"
)
gtest_discover_tests(gpu_tests)

# ZeroMQ communication tests (CPU-only)
add_executable(zmq_tests
    tests/cpp/network/test_zeromq_interface.cpp
)
target_link_libraries(zmq_tests
    GTest::gtest_main
    zeromq_interface
)
gtest_discover_tests(zmq_tests)

# Base64 encoding/decoding tests (CPU-only)
add_executable(base64_tests
    tests/cpp/core/test_base64.cpp
)
target_link_libraries(base64_tests
    GTest::gtest_main
    zeromq_interface
)
gtest_discover_tests(base64_tests)

add_executable(zmq_server_tests
    tests/cpp/network/test_zmq_server.cpp
    src/daemon/control/zmq_server.cpp
)
target_link_libraries(zmq_server_tests
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    logging
    ${ZMQ_LIBRARIES}
)
target_include_directories(zmq_server_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    src
    ${ZMQ_INCLUDE_DIRS}
    ${cppzmq_SOURCE_DIR}
)
gtest_discover_tests(zmq_server_tests)

# Error codes tests (CPU-only)
add_executable(error_codes_tests
    tests/cpp/core/test_error_codes.cpp
)
target_link_libraries(error_codes_tests
    GTest::gtest_main
    error_codes
    zeromq_interface
)
target_include_directories(error_codes_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(error_codes_tests)

# Auto-Negotiation tests (CPU-only, no ALSA hardware required)
add_executable(auto_negotiation_tests
    tests/cpp/audio/test_auto_negotiation.cpp
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

# Rate Switch E2E tests (Issue #221)
add_executable(rate_switch_e2e_tests
    tests/cpp/e2e/test_rate_switch_e2e.cpp
)
target_link_libraries(rate_switch_e2e_tests
    GTest::gtest_main
    auto_negotiation
)
target_include_directories(rate_switch_e2e_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(rate_switch_e2e_tests)

# Soft Mute tests (CPU-only)
add_executable(soft_mute_tests
    tests/cpp/audio/test_soft_mute.cpp
)
target_link_libraries(soft_mute_tests
    GTest::gtest_main
    soft_mute
)
target_include_directories(soft_mute_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(soft_mute_tests)

# Graceful Shutdown tests (CPU-only, Issue #215)
add_executable(graceful_shutdown_tests
    tests/cpp/audio/test_graceful_shutdown.cpp
)
target_link_libraries(graceful_shutdown_tests
    GTest::gtest_main
    graceful_shutdown
)
target_include_directories(graceful_shutdown_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(graceful_shutdown_tests)

# Fallback Manager tests (CPU-only, Issue #139)
add_executable(fallback_manager_tests
    tests/cpp/audio/test_fallback_manager.cpp
)
target_link_libraries(fallback_manager_tests
    GTest::gtest_main
    fallback_manager
    logging
)
target_include_directories(fallback_manager_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(fallback_manager_tests)

# Audio common utilities tests (AudioRingBuffer, AudioUtils)
add_executable(audio_common_tests
    tests/cpp/audio/test_audio_common.cpp
)
target_link_libraries(audio_common_tests
    GTest::gtest_main
)
target_include_directories(audio_common_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(audio_common_tests)

# Stall detector unit tests
add_executable(audio_input_stall_tests
    tests/cpp/e2e/test_audio_input_stall.cpp
)
target_link_libraries(audio_input_stall_tests
    GTest::gtest_main
)
target_include_directories(audio_input_stall_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
gtest_discover_tests(audio_input_stall_tests)
