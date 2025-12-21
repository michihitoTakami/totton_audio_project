cmake_minimum_required(VERSION 3.18)

include(FetchContent)

set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG 12.3.1
)
FetchContent_MakeAvailable(glslang)

find_package(Vulkan REQUIRED)

add_library(vkfft INTERFACE)
target_include_directories(vkfft INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party/vkfft
    ${CMAKE_SOURCE_DIR}/third_party/vkfft/vkFFT
    ${CMAKE_SOURCE_DIR}/third_party/vkfft/benchmark_scripts/vkFFT_scripts/include
)
target_compile_definitions(vkfft INTERFACE VKFFT_BACKEND=0 VK_API_VERSION=11)

add_executable(vkfft_minimal
    vkfft_minimal.cpp
    ${CMAKE_SOURCE_DIR}/third_party/vkfft/benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp
)

target_include_directories(vkfft_minimal PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/vkfft/benchmark_scripts/vkFFT_scripts/include
    ${CMAKE_SOURCE_DIR}/third_party/vkfft/benchmark_scripts
    $<TARGET_PROPERTY:glslang,INTERFACE_INCLUDE_DIRECTORIES>
    ${glslang_SOURCE_DIR}
    ${glslang_SOURCE_DIR}/glslang
    ${CMAKE_BINARY_DIR}/_deps/glslang-src
    ${CMAKE_BINARY_DIR}/_deps/glslang-src/glslang
    ${CMAKE_BINARY_DIR}/_deps/glslang-src/glslang/Include
)

target_compile_definitions(vkfft_minimal PRIVATE VKFFT_BACKEND=0 VK_API_VERSION=11)

target_link_libraries(vkfft_minimal PRIVATE
    vkfft
    Vulkan::Vulkan
    glslang
    OGLCompiler
    OSDependent
    glslang-default-resource-limits
    SPIRV
)
