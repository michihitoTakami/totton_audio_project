cmake_minimum_required(VERSION 3.18)
project(GPUAudioUpsampler LANGUAGES CXX CUDA)

# Standard install directories (bin, lib, etc.)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)
pkg_check_modules(ALSA REQUIRED alsa)

# Fetch nlohmann_json (header-only JSON library)
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

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

# Link against core library, PipeWire, ALSA, and JSON
target_link_libraries(gpu_upsampler_alsa
    gpu_upsampler_core
    ${PIPEWIRE_LIBRARIES}
    ${ALSA_LIBRARIES}
    nlohmann_json::nlohmann_json
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

# Installation for command-line tool and library
install(TARGETS gpu_upsampler DESTINATION bin)
install(TARGETS gpu_upsampler_daemon DESTINATION bin)
install(TARGETS gpu_upsampler_alsa DESTINATION bin)
install(TARGETS gpu_upsampler_core DESTINATION lib)
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
