#include <backend/Kernels.cuh>
#include <cuda_runtime.h>

// Warp reduction helper
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}




__global__ void layer_norm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ gamma,
    float* __restrict__ grad_input,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    uint32_t M,
    uint32_t N
) {
    const uint32_t row = blockIdx.x;
    if (row >= M) return;
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t num_warps = blockDim.x / 32;
    
    __shared__ float shared_sum1[32];
    __shared__ float shared_sum2[32];
    
    const float* grad_out_row = grad_output + row * N;
    const float* input_row = input + row * N;
    float* grad_in_row = grad_input ? (grad_input + row * N) : nullptr;
    
    float row_mean = mean[row];
    float row_rstd = rstd[row];
    
    if (grad_gamma || grad_beta) {
        for (uint32_t i = tid; i < N; i += blockDim.x) {
            float x_normalized = (input_row[i] - row_mean) * row_rstd;
            float grad_out_val = grad_out_row[i];
            
            if (grad_gamma) {
                atomicAdd(&grad_gamma[i], grad_out_val * x_normalized);
            }
            if (grad_beta) {
                atomicAdd(&grad_beta[i], grad_out_val);
            }
        }
    }
    
    if (grad_in_row) {
        float thread_sum1 = 0.0f;
        float thread_sum2 = 0.0f;
        
        for (uint32_t i = tid; i < N; i += blockDim.x) {
            float x_normalized = (input_row[i] - row_mean) * row_rstd;
            float grad_weighted = grad_out_row[i] * gamma[i];
            thread_sum1 += grad_weighted;
            thread_sum2 += grad_weighted * x_normalized;
        }
        
        float warp_sum1 = warp_reduce_sum(thread_sum1);
        float warp_sum2 = warp_reduce_sum(thread_sum2);
        
        if (lane_id == 0) {
            shared_sum1[warp_id] = warp_sum1;
            shared_sum2[warp_id] = warp_sum2;
        }
        __syncthreads();
        
        if (warp_id == 0) {
            float val1 = (lane_id < num_warps) ? shared_sum1[lane_id] : 0.0f;
            float val2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
            
            val1 = warp_reduce_sum(val1);
            val2 = warp_reduce_sum(val2);
            
            if (lane_id == 0) {
                shared_sum1[0] = val1;
                shared_sum2[0] = val2;
            }
        }
        __syncthreads();
        
        float sum_grad = shared_sum1[0];
        float sum_grad_x = shared_sum2[0];
        
        float scale = 1.0f / N;
        for (uint32_t i = tid; i < N; i += blockDim.x) {
            float x_normalized = (input_row[i] - row_mean) * row_rstd;
            float grad_weighted = grad_out_row[i] * gamma[i];
            
            float grad_val = row_rstd * (grad_weighted - scale * sum_grad - scale * x_normalized * sum_grad_x);
            grad_in_row[i] = grad_val;
        }
    }
}