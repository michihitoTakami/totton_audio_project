cmake_minimum_required(VERSION 3.18)

include(FetchContent)

set(VKFFT_GIT_TAG "v1.3.4" CACHE STRING "VkFFT git tag/commit to fetch")

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

FetchContent_Declare(
    vkfft
    GIT_REPOSITORY https://github.com/DTolm/VkFFT.git
    GIT_TAG ${VKFFT_GIT_TAG}
)
FetchContent_GetProperties(vkfft)
if(NOT vkfft_POPULATED)
    FetchContent_Populate(vkfft)
endif()

find_package(Vulkan REQUIRED)

add_executable(vkfft_minimal
    vkfft_minimal.cpp
)

target_include_directories(vkfft_minimal PRIVATE
    ${vkfft_SOURCE_DIR}/vkFFT
    ${glslang_SOURCE_DIR}/glslang/Include
    ${glslang_SOURCE_DIR}/glslang/Public
)

target_compile_definitions(vkfft_minimal PRIVATE VKFFT_BACKEND=0 VK_API_VERSION=11)

target_link_libraries(vkfft_minimal PRIVATE
    Vulkan::Vulkan
    glslang
    OGLCompiler
    OSDependent
    glslang-default-resource-limits
    SPIRV
)
