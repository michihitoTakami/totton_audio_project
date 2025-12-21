add_executable(vkfft_minimal
    vkfft_minimal.cpp
)

target_include_directories(vkfft_minimal PRIVATE
    ${glslang_SOURCE_DIR}/glslang/Include
)

target_compile_definitions(vkfft_minimal PRIVATE VKFFT_BACKEND=0 VK_API_VERSION=11)

target_link_libraries(vkfft_minimal PRIVATE
    vulkan_shader_libs
)
