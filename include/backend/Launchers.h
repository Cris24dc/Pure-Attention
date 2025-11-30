#pragma once

// libs
#include <cuda_runtime.h>
#include <chrono>

// macro
#define TILE_WIDTH 32

void launch_matmul_tiled(float *A, float *B, float *C, int M, int N, int K, cudaStream_t stream = 0);
void launch_matadd_tiled(float *A, float *X, float *B, int M, int N, cudaStream_t stream = 0);
void launch_ReLU_tiled(float *In, float *Out, int M, int N, cudaStream_t stream = nullptr);
void launch_zero_population(float *A, int M, int N, cudaStream_t stream = nullptr);
void launch_ones_population(float *A, int M, int N, cudaStream_t stream = nullptr);
void launch_normal_population(float *A, int M, int N, cudaStream_t stream = nullptr);

void launch_matmul_grad_X(const float* grad_Y_out, const float* W_in, float* grad_X_in,
                          const int M, const int N, const int K, cudaStream_t stream);
void launch_matmul_grad_W(const float* A, const float* grad_C, float* grad_B,
                          int M, int N, int K, cudaStream_t stream);
void launch_tensor_add_grad(const float* src, float* dst, int size, cudaStream_t stream);
void launch_sum_rows_grad(const float* src, float* dst, int M, int N, cudaStream_t stream);
void launch_relu_backward(const float* grad_out, const float* input_data, float* grad_in, int size, cudaStream_t stream) ;