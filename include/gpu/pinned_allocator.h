#ifndef GPU_PINNED_ALLOCATOR_H
#define GPU_PINNED_ALLOCATOR_H

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <new>
#include <unordered_set>
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

        const size_type bytes = n * sizeof(T);
        void* ptr = nullptr;

        if (PinnedAllocationRegistry::pinnedEnabled().load(std::memory_order_acquire)) {
            const cudaError_t err =
                cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault | cudaHostAllocPortable);
            if (err == cudaSuccess) {
                PinnedAllocationRegistry::registerPinned(ptr);
                return static_cast<pointer>(ptr);
            }

            // Disable future attempts after first failure (likely unsupported environment)
            PinnedAllocationRegistry::pinnedEnabled().store(false, std::memory_order_release);
            if (!PinnedAllocationRegistry::hasLoggedFailure()) {
                std::fprintf(stderr,
                             "Warning: cudaHostAlloc failed (%s); falling back to pageable memory.\n",
                             cudaGetErrorString(err));
                PinnedAllocationRegistry::setLoggedFailure();
            }
        }

        ptr = ::operator new(bytes);
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        if (!p) {
            return;
        }

        if (PinnedAllocationRegistry::unregisterPinned(p)) {
            cudaFreeHost(p);
        } else {
            ::operator delete(p);
        }
    }

    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type;
};

template <typename T>
using CudaPinnedVector = std::vector<T, CudaPinnedAllocator<T>>;

namespace PinnedAllocationRegistry {
namespace detail {
inline std::unordered_set<void*>& pinnedSet() {
    static std::unordered_set<void*> set;
    return set;
}

inline std::mutex& pinnedMutex() {
    static std::mutex mtx;
    return mtx;
}
}  // namespace detail

inline void registerPinned(void* ptr) {
    std::lock_guard<std::mutex> lock(detail::pinnedMutex());
    detail::pinnedSet().insert(ptr);
}

inline bool unregisterPinned(void* ptr) {
    std::lock_guard<std::mutex> lock(detail::pinnedMutex());
    auto& set = detail::pinnedSet();
    auto it = set.find(ptr);
    if (it != set.end()) {
        set.erase(it);
        return true;
    }
    return false;
}

inline std::atomic<bool>& pinnedEnabled() {
    static std::atomic<bool> enabled{true};
    return enabled;
}

inline bool& loggedFailureFlag() {
    static bool logged = false;
    return logged;
}

inline bool hasLoggedFailure() {
    return loggedFailureFlag();
}

inline void setLoggedFailure() {
    loggedFailureFlag() = true;
}
}  // namespace PinnedAllocationRegistry

}  // namespace ConvolutionEngine

#endif  // GPU_PINNED_ALLOCATOR_H

