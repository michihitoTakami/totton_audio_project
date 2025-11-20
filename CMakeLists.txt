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
pkg_check_modules(LV2 REQUIRED lv2)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data/coefficients
    ${SNDFILE_INCLUDE_DIRS}
    ${LV2_INCLUDE_DIRS}
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
add_executable(gpu_upsampler src/main.cpp)

# Link against core library
target_link_libraries(gpu_upsampler
    gpu_upsampler_core
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

# LV2 Plugin (shared library)
add_library(gpu_upsampler_lv2 MODULE
    src/lv2_plugin/gpu_upsampler_lv2.cpp
)

# LV2 plugin properties
set_target_properties(gpu_upsampler_lv2 PROPERTIES
    PREFIX ""
    OUTPUT_NAME "gpu_upsampler"
    SUFFIX ".so"
)

# Link LV2 plugin against core library
target_link_libraries(gpu_upsampler_lv2
    gpu_upsampler_core
    ${LV2_LIBRARIES}
)

# Compiler flags for LV2 plugin
target_compile_options(gpu_upsampler_lv2 PRIVATE
    -Wall -Wextra -O3 -fPIC
)

# LV2 plugin installation
# Standard LV2 bundle structure: ~/.lv2/gpu_upsampler.lv2/
set(LV2_INSTALL_DIR "$ENV{HOME}/.lv2/gpu_upsampler.lv2")
install(TARGETS gpu_upsampler_lv2 DESTINATION ${LV2_INSTALL_DIR})
install(FILES
    src/lv2_plugin/manifest.ttl
    src/lv2_plugin/gpu_upsampler.ttl
    DESTINATION ${LV2_INSTALL_DIR}
)

# Installation for command-line tool and library
install(TARGETS gpu_upsampler DESTINATION bin)
install(TARGETS gpu_upsampler_core DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/gpu_upsampler)
