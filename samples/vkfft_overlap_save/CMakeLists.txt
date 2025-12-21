cmake_minimum_required(VERSION 3.18)

include(FetchContent)

set(VKFFT_GIT_TAG "v1.3.4" CACHE STRING "VkFFT git tag/commit to fetch")

# glslang は他サンプルと共有できるようガードを付ける
if(NOT TARGET glslang)
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
endif()

if(NOT DEFINED vkfft_SOURCE_DIR)
    FetchContent_Declare(
        vkfft
        GIT_REPOSITORY https://github.com/DTolm/VkFFT.git
        GIT_TAG ${VKFFT_GIT_TAG}
    )
    FetchContent_GetProperties(vkfft)
    if(NOT vkfft_POPULATED)
        FetchContent_Populate(vkfft)
    endif()
endif()

find_package(Vulkan REQUIRED)

add_library(vulkan_overlap_save STATIC
    vulkan_overlap_save.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/audio_io.cpp
)

target_include_directories(vulkan_overlap_save PRIVATE
    ${vkfft_SOURCE_DIR}/vkFFT
    ${glslang_SOURCE_DIR}/glslang/Include
    ${glslang_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${SNDFILE_INCLUDE_DIRS}
)

target_compile_definitions(vulkan_overlap_save PRIVATE VKFFT_BACKEND=0 VK_API_VERSION=11)

target_link_libraries(vulkan_overlap_save PRIVATE
    Vulkan::Vulkan
    glslang
    OGLCompiler
    OSDependent
    glslang-default-resource-limits
    SPIRV
    nlohmann_json::nlohmann_json
    logging
    ${SNDFILE_LIBRARIES}
)

target_compile_options(vulkan_overlap_save PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
)

add_executable(vulkan_overlap_save_tool
    vulkan_overlap_save_tool.cpp
)

target_link_libraries(vulkan_overlap_save_tool PRIVATE
    vulkan_overlap_save
    logging
)

add_executable(vulkan_overlap_save_tests
    vulkan_overlap_save_tests.cpp
)

target_link_libraries(vulkan_overlap_save_tests PRIVATE
    vulkan_overlap_save
    GTest::gtest_main
)

target_compile_options(vulkan_overlap_save_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
)

gtest_discover_tests(vulkan_overlap_save_tests)
