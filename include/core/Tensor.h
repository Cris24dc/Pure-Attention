#pragma once

// libs
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace core {
    // custom free funct smart pointer
    struct CudaDeallocator {
        void operator()(void* ptr) const {
            if (ptr) {
                cudaFree(ptr);
            }
        }
    };

    // alias
    using float32_t = float;

    // Tensor class
    class Tensor : public std::enable_shared_from_this<Tensor> {
    private:
        std::unique_ptr<float32_t, CudaDeallocator> data_ptr;
        std::unique_ptr<float32_t, CudaDeallocator> gradient_ptr;

        std::vector<int32_t> shape;
        size_t size;
        bool has_gradient; 

    public:
        Tensor(const std::vector<int>& shape, bool has_gardient = false);
        ~Tensor() = default;

        // Utility methods
        static std::shared_ptr<Tensor> zeros(); // inits a tensor full of 0s on device
        static std::shared_ptr<Tensor> rand_normal(); // inits a tensor full of normal distributed numbers on device

        // Cuda methods
        void to_device(const std::vector<float32_t>& host_data);
        std::vector<float32_t> to_host();
    };
};