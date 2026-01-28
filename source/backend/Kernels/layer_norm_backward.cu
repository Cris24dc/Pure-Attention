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




// Kernel 1: Parameter Gradients (Gamma/Beta) - Grid Stride Loop Reduction
__global__ void ln_backward_param_grad_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    uint32_t M,
    uint32_t N
) {
    const uint32_t tid = threadIdx.x;
    const uint32_t col = tid;
    
    // Each block processes a subset of columns (N). For standard LN, N < 1024, so 1 block handles 1 row fully.
    // If N > 1024, we would need tilling, but standard LN implies N <= 4096 usu. 
    // Here we assume N <= blockDim.x (usually 256 or 512 or 1024). 
    // If N > blockDim.x, we loop.
    
    // We want P permanent blocks to iterate over M rows.
    // Each thread accumulates the sum for its specific column 'col'.
    
    // Grid-stride loop over batch dimension M
    for (uint32_t c = tid; c < N; c += blockDim.x) {
        float local_sum_gamma = 0.0f;
        float local_sum_beta = 0.0f;

        for (uint32_t row_idx = blockIdx.x; row_idx < M; row_idx += gridDim.x) {
            const float* grad_out_row = grad_output + row_idx * N;
            const float* input_row = input + row_idx * N;
            
            float row_mean = mean[row_idx];
            float row_rstd = rstd[row_idx];
            
            float dout = grad_out_row[c];
            float x_val = input_row[c];
            float x_norm = (x_val - row_mean) * row_rstd;
            
            // Accumulate
            local_sum_gamma += dout * x_norm;
            local_sum_beta += dout;
        }

        // Now we have partial sums for this block.
        // We atomically add them to the global buffer.
        // Since gridDim.x is small (e.g. 128), this is only ~128 atomicAdds per column.
        
        if (grad_gamma) atomicAdd(&grad_gamma[c], local_sum_gamma);
        if (grad_beta)  atomicAdd(&grad_beta[c], local_sum_beta);
    }
}

// Kernel 2: Input Gradients - Row Independent
// Standard implementation using recomputation of row-wise statistics.
__global__ void ln_backward_input_grad_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ gamma,
    float* __restrict__ grad_input,
    uint32_t M,
    uint32_t N
) {
    const uint32_t row = blockIdx.x;
    if (row >= M) return;
    
    const uint32_t tid = threadIdx.x;
    
    // Warp Reduction Helper locally defined or used from util
    
    const float* grad_out_row = grad_output + row * N;
    const float* input_row = input + row * N;
    float* grad_in_row = grad_input + row * N;
    
    float row_mean = mean[row];
    float row_rstd = rstd[row];
    
    float thread_sum1 = 0.0f; // sum(dout * gamma)
    float thread_sum2 = 0.0f; // sum(dout * gamma * x_norm)
    
    for (uint32_t i = tid; i < N; i += blockDim.x) {
        float x_norm = (input_row[i] - row_mean) * row_rstd;
        float g = gamma[i];
        float dout = grad_out_row[i];
        
        float term = dout * g;
        
        thread_sum1 += term;
        thread_sum2 += term * x_norm;
    }
    
    // Block Reduction
    static __shared__ float s_sum1;
    static __shared__ float s_sum2;
    
    float warp_sum1 = warp_reduce_sum(thread_sum1);
    float warp_sum2 = warp_reduce_sum(thread_sum2);
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&s_sum1, warp_sum1); // Small contention (max 32 warps per block), negligible
        atomicAdd(&s_sum2, warp_sum2);
    }
    
    // Alternative: Use proper block reduce to avoid even these small atomics, 
    // but for <1024 threads, shared atomics are extremely fast.
    // However, since we want "expert CUDA", let's use a cleaner approach if possible.
    // But for simplicity of this specific refactor, this is fine as it's block-local shared mem.
    
    // Reset shared memory
    if (tid == 0) {
        s_sum1 = 0.0f; 
        s_sum2 = 0.0f;
    }
    __syncthreads();
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&s_sum1, warp_sum1);
        atomicAdd(&s_sum2, warp_sum2);
    }
    __syncthreads();
    
    float sum_gamma_dout = s_sum1;
    float sum_gamma_dout_xnorm = s_sum2;
    
    // Compute Gradient Input
    float inv_N = 1.0f / N;
    
    for (uint32_t i = tid; i < N; i += blockDim.x) {
        float x_norm = (input_row[i] - row_mean) * row_rstd;
        float g = gamma[i];
        float dout = grad_out_row[i];
        
        
        float term1 = N * dout * g;
        float term2 = sum_gamma_dout;
        float term3 = x_norm * sum_gamma_dout_xnorm;
        
        float val = (term1 - term2 - term3) * row_rstd * inv_N;
        
        grad_in_row[i] = val;
    }
}