// headers
#include <core/Tensor.h>
#include <core/Context.h>
#include <core/Functional.h>

// libs
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace core {
    Tensor::Tensor(const std::vector<uint32_t>& shape, const bool requires_gradient, const bool is_leaf) :
    shape(shape), 
    requires_gradient(requires_gradient),
    is_leaf(is_leaf),
    grad_function(nullptr)
    {
        size = 1;
        for (const uint32_t s : shape) {
            size *= s;
        }

        float32_t* raw_ptr_data = nullptr;
        size_t bytes = size * sizeof(float32_t);

        cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&raw_ptr_data), bytes);

        if (err != cudaSuccess) {
            const std::string error_msg = "CUDA Error: " + std::string(cudaGetErrorString(err));
            throw std::runtime_error(error_msg);
        }

        data_ptr.reset(raw_ptr_data);

        if (requires_gradient) {
            float32_t* raw_ptr_grad = nullptr;
            bytes = size * sizeof(float32_t);
    
            err = cudaMalloc(reinterpret_cast<void **>(&raw_ptr_grad), bytes);
    
            if (err != cudaSuccess) {
                const std::string error_msg = "CUDA Error: " + std::string(cudaGetErrorString(err));
                throw std::runtime_error(error_msg);
            }
            pop_grad_zeros(this);
    
            gradient_ptr.reset(raw_ptr_grad);
        }
    }

    void Tensor::to_device(const std::vector<float32_t>& host_data) const {
        if (host_data.size() != size) {
            throw std::length_error("Size mismatch: CPU array has different size to GPU tensor.");
        }

        if (const cudaError_t err = cudaMemcpyAsync(data_ptr.get(), host_data.data(), size * sizeof(float32_t), cudaMemcpyHostToDevice, CudaContext::getStream()); err != cudaSuccess) {
            throw std::runtime_error("To GPU error (Memcpy H2D)");
        }
    }

    std::vector<float32_t> Tensor::to_host() const {
        std::vector<float32_t> host_result(size);

        if (const cudaError_t err = cudaMemcpyAsync(host_result.data(), data_ptr.get(), size * sizeof(float32_t), cudaMemcpyDeviceToHost, CudaContext::getStream()); err != cudaSuccess) {
            throw std::runtime_error("To CPU error (Memcpy D2H)");
        }
        
        return host_result;
    }

    std::vector<uint32_t> Tensor::get_shape() const {
        return shape;
    };

    float32_t *Tensor::get_data_ptr() const {
        return data_ptr.get();
    };

    float32_t *Tensor::get_gradient_ptr() const {
        if (!gradient_ptr)
            return nullptr;
        return gradient_ptr.get();
    }

    void Tensor::set_grad_fn(std::shared_ptr<Function> fn) {
        grad_function=fn;
        is_leaf=false;
    }

    void Tensor::backward() {
        if (!requires_gradient) {
            throw std::runtime_error("Called backward() on a tensor that does not require gradients.");
        }

        pop_grad_ones(this);

        if (grad_function)
            grad_function->apply_backward();

    }
};