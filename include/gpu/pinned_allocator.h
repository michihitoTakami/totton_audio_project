#ifndef GPU_PINNED_ALLOCATOR_H
#define GPU_PINNED_ALLOCATOR_H

#include <cuda_runtime_api.h>

#include <cstddef>
#include <new>
#include <vector>

namespace ConvolutionEngine {

/**
 * @brief Allocator that obtains CUDA pinned (page-locked) host memory.
 *
 * Jetsonではホスト↔デバイス間コピーがボトルネックになりやすいため、streaming経路で
 * 使用するバッファを常にcudaHostAlloc()で確保する。RTXなどのPC GPUでも互換性があり、
 * 未対応環境ではstd::bad_allocを投げてフォールバックさせる。
 */
template <typename T>
class CudaPinnedAllocator {
   public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = CudaPinnedAllocator<U>;
    };

    CudaPinnedAllocator() noexcept = default;

    template <typename U>
    CudaPinnedAllocator(const CudaPinnedAllocator<U>&) noexcept {}

    [[nodiscard]] pointer allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }

        void* ptr = nullptr;
        const cudaError_t err =
            cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocDefault | cudaHostAllocPortable);
        if (err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        if (p) {
            cudaFreeHost(p);
        }
    }

    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type;
};

template <typename T>
using CudaPinnedVector = std::vector<T, CudaPinnedAllocator<T>>;

}  // namespace ConvolutionEngine

#endif  // GPU_PINNED_ALLOCATOR_H

