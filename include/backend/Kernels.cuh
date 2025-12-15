#pragma once

// libs
#include <curand_kernel.h>
#include <cstdint>

// macro
#define TILE_WIDTH 32
using float32_t = float;

__global__ void matmul_kernel_tiled(
    const float32_t *A, 
    const float32_t *B, 
    float32_t *C, 
    const int32_t M, 
    const int32_t N, 
    const int32_t K
);

__global__ void matadd_kernel_tiled(
    const float32_t *A, 
    const float32_t *X, 
    float32_t *B, 
    const int32_t M, 
    const int32_t N
);

__global__ void ReLU_kernel_tiled(
    const float *In, 
    float *Out, 
    const u_int32_t M, 
    const uint32_t N
);

__global__ void populate_normal(
    float32_t *A, 
    uint32_t M, 
    uint32_t N, 
    float32_t std_dev,
    const uint64_t seed
);

__global__ void matmul_backward_X_kernel(
    const float32_t* grad_Y_out,
    const float32_t* W_in,
    float32_t* grad_X_in,
    const uint32_t M, 
    const uint32_t N, 
    const uint32_t K
);

__global__ void matmul_backward_W_kernel(
    const float32_t *X_in,
    const float32_t *grad_Y_out,
    float32_t *grad_W_in,
    const u_int32_t M, 
    const u_int32_t N, 
    const uint32_t K
);

__global__ void tensor_add_grad_kernel(
    const float32_t* src, 
    float32_t* dst, 
    int32_t size
);

__global__ void sum_rows_grad_kernel(
    const float32_t* src, 
    float32_t* dst, 
    int32_t M, 
    int32_t N
);

__global__ void relu_backward_kernel(
    const float32_t* grad_out, 
    const float32_t* input_data, 
    float32_t* grad_in, 
    int32_t size
);

__global__ void mse_forward_kernel(
    const float32_t *preds, 
    const float32_t *targets, 
    float32_t *loss_out, 
    const uint32_t N
);

__global__ void mse_div_kernel(
    float32_t *loss_out,
    const uint32_t N
);

__global__ void mse_backward_kernel(
    const float32_t *predictions,
    const float32_t *targets,
    const float32_t *grad_loss_scalar,
    float32_t *grad_predictions,
    const uint32_t N
);

__global__ void adam_step_kernel(
    float32_t* params,
    const float32_t* grads,
    float32_t* m,
    float32_t* v,
    const uint32_t size,
    const float32_t beta1,
    const float32_t beta2,
    const float32_t epsilon,
    const float32_t alpha,
    const float32_t beta1_corr,
    const float32_t beta2_corr
);
