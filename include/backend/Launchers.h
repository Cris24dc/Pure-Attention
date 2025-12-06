#pragma once

// libs
#include <cuda_runtime.h>
#include <chrono>

// macro
#define TILE_WIDTH 32
using float32_t=float;

void launch_matmul_tiled(float *A, float *B, float *C, int M, int N, int K, cudaStream_t stream = 0);
void launch_matadd_tiled(float *A, float *X, float *B, int M, int N, cudaStream_t stream = 0);
void launch_ReLU_tiled(float *In, float *Out, int M, int N, cudaStream_t stream = nullptr);
void launch_zero_population(float *A, int M, int N, cudaStream_t stream = nullptr);
void launch_ones_population(float *A, int M, int N, cudaStream_t stream = nullptr);
void launch_normal_population(float *A, int M, int N, cudaStream_t stream = nullptr);

void launch_matmul_grad_X(const float* grad_Y_out, const float* W_in, float* grad_X_in,
                          int M, int N,int K, cudaStream_t stream);
void launch_matmul_grad_W(const float32_t *X_in, const float32_t *grad_Y_out, float32_t *grad_W_in,
                          int M, int N, int K, cudaStream_t stream);
void launch_tensor_add_grad(const float* src, float* dst, int size, cudaStream_t stream);
void launch_sum_rows_grad(const float* src, float* dst, int M, int N, cudaStream_t stream);
void launch_relu_backward(const float* grad_out, const float* input_data, float* grad_in, int size, cudaStream_t stream);
void launch_mse_backward(const float* preds, const float* targets, const float* grad_loss, float* grad_preds, int N, cudaStream_t stream);
void launch_mse_forward(const float* preds, const float* targets, float* loss_out, int N, cudaStream_t stream) ;
void launch_adam_step(
    float32_t* params,
    const float32_t* grads,
    float32_t* m,
    float32_t* v,
    int size,
    float32_t beta1,
    float32_t beta2,
    float32_t epsilon,
    float32_t lr,
    int step,
    cudaStream_t stream
);