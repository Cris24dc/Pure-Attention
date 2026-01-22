#include <backend/Kernels.cuh>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void layer_norm_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    uint32_t M,
    uint32_t N,
    float epsilon
) {
    const uint32_t row = blockIdx.x;
    if (row >= M) return;
    
    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    const uint32_t num_warps = blockDim.x / 32;
    
    __shared__ float shared_sum[32];
    __shared__ float shared_sq_sum[32];
    
    const float* input_row = input + row * N;
    float* output_row = output + row * N;
    
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    
    for (uint32_t i = tid; i < N; i += blockDim.x) {
        float val = input_row[i];
        thread_sum += val;
        thread_sq_sum += val * val;
    }
    
    float warp_sum = warp_reduce_sum(thread_sum);
    float warp_sq_sum = warp_reduce_sum(thread_sq_sum);
    
    if (lane_id == 0) {
        shared_sum[warp_id] = warp_sum;
        shared_sq_sum[warp_id] = warp_sq_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float val_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        float val_sq = (lane_id < num_warps) ? shared_sq_sum[lane_id] : 0.0f;
        
        val_sum = warp_reduce_sum(val_sum);
        val_sq = warp_reduce_sum(val_sq);
        
        if (lane_id == 0) {
            float row_mean = val_sum / N;
            float row_var = (val_sq / N) - (row_mean * row_mean);
            float row_rstd = rsqrtf(row_var + epsilon);
            
            mean[row] = row_mean;
            rstd[row] = row_rstd;
            
            shared_sum[0] = row_mean;
            shared_sq_sum[0] = row_rstd;
        }
    }
    __syncthreads();
    
    float row_mean = shared_sum[0];
    float row_rstd = shared_sq_sum[0];
    
    for (uint32_t i = tid; i < N; i += blockDim.x) {
        float normalized = (input_row[i] - row_mean) * row_rstd;
        output_row[i] = gamma[i] * normalized + beta[i];
    }
}