#pragma once

// libs
#include <memory>

#include <core/Tensor.h>

namespace core {
    void matmul(
        const std::shared_ptr<Tensor>& A, 
        const std::shared_ptr<Tensor>& B,
        std::shared_ptr<Tensor>& C, 
        const cudaStream_t& stream);

    void matadd(
        const std::shared_ptr<Tensor>& A, 
        const std::shared_ptr<Tensor>& X,
        std::shared_ptr<Tensor>& B,
        const cudaStream_t& stream);

    std::shared_ptr<Tensor> relu(
        const std::shared_ptr<Tensor> &In,
        const cudaStream_t& stream);

    void pop_data_zeros(
        const std::shared_ptr<Tensor> &A,
        const cudaStream_t& stream);

    void pop_grad_zeros(
        const std::shared_ptr<Tensor> &A,
        const cudaStream_t& stream);

    void pop_grad_zeros(
        Tensor *A,
        const cudaStream_t& stream);

    void pop_grad_ones(
        const std::shared_ptr<Tensor> &A,
        const cudaStream_t& stream);

    void pop_grad_ones(
        Tensor *A,
        const cudaStream_t& stream);

    void pop_data_normal(
        const std::shared_ptr<Tensor> &A, 
        float std_dev, 
        const cudaStream_t& stream);

    std::shared_ptr<Tensor> mse_loss(
        const std::shared_ptr<Tensor>& preds, 
        const std::shared_ptr<Tensor>& targets,
        const cudaStream_t& stream);

    void split(
        const std::shared_ptr<Tensor>& A,
        uint32_t num_parts,
        int dim,
        std::vector<std::shared_ptr<Tensor>>& parts,
        const cudaStream_t& stream);

    void reshape(
        const std::shared_ptr<Tensor>& input,
        const std::vector<uint32_t>& new_shape,
        std::shared_ptr<Tensor>& output,
        const cudaStream_t& stream);

    void flash_attention(
        const std::shared_ptr<Tensor>& Q,
        const std::shared_ptr<Tensor>& K,
        const std::shared_ptr<Tensor>& V,
        std::shared_ptr<Tensor>& O,
        const cudaStream_t& stream);


    std::shared_ptr<Tensor> layer_norm(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& gamma,
        const std::shared_ptr<Tensor>& beta,
        float epsilon,
        const cudaStream_t& stream);
};
