#pragma once

// libs
#include <memory>

namespace core {
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> &A, const std::shared_ptr<Tensor> &B);
    std::shared_ptr<Tensor> matadd(const std::shared_ptr<Tensor> &A, const std::shared_ptr<Tensor> &X);
    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor> &In);
    void pop_data_zeros(const std::shared_ptr<Tensor> &A);
    void pop_grad_zeros(const std::shared_ptr<Tensor> &A);
    void pop_grad_zeros(Tensor *A);
    void pop_grad_ones(const std::shared_ptr<Tensor> &A);
    void pop_grad_ones(Tensor *A);
    void pop_data_normal(const std::shared_ptr<Tensor> &A);
    std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& preds, const std::shared_ptr<Tensor>& targets);
};
