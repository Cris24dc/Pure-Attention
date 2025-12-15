#pragma once

// libs
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace core {
    using float32_t = float;

    // forward declaration
    struct Function;

    struct CudaDeallocator {
        void operator()(void *ptr) const {
            if (ptr) {
                cudaFree(ptr);
            }
        }
    };

    class Tensor : public std::enable_shared_from_this<Tensor> {
    private:
        std::unique_ptr<float32_t, CudaDeallocator> data_ptr;
        std::unique_ptr<float32_t, CudaDeallocator> gradient_ptr;

        std::vector<uint32_t> shape;
        size_t size;

        bool requires_gradient;
        bool is_leaf;

        std::shared_ptr<Function> grad_function;

    public:
        explicit Tensor(const std::vector<uint32_t>& shape, bool requires_gardient = false, bool is_leaf = true);
        ~Tensor() = default;

        void to_device(const std::vector<float32_t> &host_data) const;
        std::vector<float32_t> to_host() const;
        std::vector<float32_t> grad_to_host() const;

        std::vector<uint32_t> get_shape() const;
        float32_t *get_data_ptr() const;

        bool requires_grad() const { return requires_gradient; }
        float32_t *get_gradient_ptr() const;
        void set_grad_fn(std::shared_ptr<Function> fn);

        void backward(bool seed_with_ones = true) const;

        friend struct Function;
    };
};
