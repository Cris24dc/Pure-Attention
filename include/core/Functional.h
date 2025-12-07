#pragma once

// libs
#include <memory>

namespace core {
    void matmul(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B,
        std::shared_ptr<Tensor>& C, const cudaStream_t& stream );
    void matadd(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& X,
        std::shared_ptr<Tensor>& B,const cudaStream_t& stream);
    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor> &In,
        const cudaStream_t& stream );
    void pop_data_zeros(const std::shared_ptr<Tensor> &A,
        const cudaStream_t& stream );
    void pop_grad_zeros(const std::shared_ptr<Tensor> &A,
        const cudaStream_t& stream );
    void pop_grad_zeros(Tensor *A,
        const cudaStream_t& stream );
    void pop_grad_ones(const std::shared_ptr<Tensor> &A,
        const cudaStream_t& stream );
    void pop_grad_ones(Tensor *A,
        const cudaStream_t& stream );
    void pop_data_normal(const std::shared_ptr<Tensor> &A,
        const cudaStream_t& stream );
    std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& preds, const std::shared_ptr<Tensor>& targets,
        const cudaStream_t& stream );
};
