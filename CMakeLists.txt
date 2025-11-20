cmake_minimum_required(VERSION 3.18)
project(GPUAudioUpsampler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data/coefficients
    ${SNDFILE_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/audio_io.cpp
    src/convolution_engine.cu
)

# Executable
add_executable(gpu_upsampler ${SOURCES})

# CUDA architecture (RTX 2070 Super = Compute Capability 7.5)
set_target_properties(gpu_upsampler PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

# Link libraries
target_link_libraries(gpu_upsampler
    CUDA::cudart
    CUDA::cufft
    ${SNDFILE_LIBRARIES}
)

# Compiler flags
target_compile_options(gpu_upsampler PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Installation
install(TARGETS gpu_upsampler DESTINATION bin)
