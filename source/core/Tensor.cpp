// headers
#include <core/Tensor.h>

// libs
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace core {
    Tensor::Tensor(const std::vector<int>& shape, bool has_gradient) : 
    shape(shape), 
    has_gradient(has_gradient) 
    {
        size = 1;
        for (int s : shape) {
            size *= s;
        }

        // data
        float32_t* raw_ptr_data = nullptr;
        size_t bytes = size * sizeof(float32_t);

        cudaError_t err = cudaMalloc((void**)&raw_ptr_data, bytes);

        if (err != cudaSuccess) {
            std::string error_msg = "CUDA Error: " + std::string(cudaGetErrorString(err));
            throw std::runtime_error(error_msg);
        }

        data_ptr.reset(raw_ptr_data);


        // gradient
        float32_t* raw_ptr_grad = nullptr;
        size_t bytes = size * sizeof(float32_t);

        cudaError_t err = cudaMalloc((void**)&raw_ptr_grad, bytes);

        if (err != cudaSuccess) {
            std::string error_msg = "CUDA Error: " + std::string(cudaGetErrorString(err));
            throw std::runtime_error(error_msg);
        }

        gradient_ptr.reset(raw_ptr_grad);
    }

    void Tensor::to_device(const std::vector<float32_t>& host_data) {
        if (host_data.size() != size) {
            throw std::length_error("Size mismatch: CPU array has different size to GPU tensor.");
        }

        cudaError_t err = cudaMemcpy(data_ptr.get(), host_data.data(), size * sizeof(float32_t), cudaMemcpyHostToDevice);
        
        if (err != cudaSuccess) {
            throw std::runtime_error("To GPU error (Memcpy H2D)");
        }
    }

    std::vector<float32_t> Tensor::to_host() {
        std::vector<float32_t> host_result(size);

        cudaError_t err = cudaMemcpy(host_result.data(), data_ptr.get(), size * sizeof(float32_t), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            throw std::runtime_error("To CPU error (Memcpy D2H)");
        }
        
        return host_result;
    }
};
