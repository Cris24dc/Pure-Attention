#pragma once

// libs
#include <curand_kernel.h>

// macro
#define TILE_WIDTH 32
using float32_t = float;


__global__ void matmul_kernel_tiled(const float32_t *A, const float32_t *B, float32_t *C, int32_t M, int32_t N, int32_t K);
__global__ void matadd_kernel_tiled(const float32_t *A, const float32_t *X, float32_t *B, const int32_t M, const int32_t N);
__global__ void ReLU_kernel_tiled(const float32_t *In, float32_t *Out, int32_t M, int32_t N);
__global__ void populate_normal(float32_t *A, int32_t M, int32_t N, unsigned long long seed);

__global__ void matmul_backward_X_kernel(
    const float32_t* __restrict__ grad_Y_out,
    const float32_t* __restrict__ W_in,
    float32_t* __restrict__ grad_X_in,
    int32_t M, int32_t N, int32_t K);

__global__ void matmul_backward_B_kernel(
    const float32_t* __restrict__ A,
    const float32_t* __restrict__ grad_C,
    float32_t* __restrict__ grad_B,
    int32_t M, int32_t N, int32_t K);
__global__ void tensor_add_grad_kernel(const float32_t* src, float32_t* dst, int32_t size) ;
__global__ void sum_rows_grad_kernel(const float32_t* src, float32_t* dst, int32_t M, int32_t N) ;
__global__ void relu_backward_kernel(const float32_t* grad_out, const float32_t* input_data, float32_t* grad_in, int32_t size) ;
