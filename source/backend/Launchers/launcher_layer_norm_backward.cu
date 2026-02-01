#include <backend/Launchers.cuh>
#include <backend/Kernels.cuh>


void launch_layer_norm_backward(
    const float* grad_output,
    const float* input,
    const float* mean,
    const float* rstd,
    const float* gamma,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    uint32_t M,
    uint32_t N,
    cudaStream_t stream
) {
    // 1. Parameter Gradients (Gamma/Beta) - if needed
    if (grad_gamma || grad_beta) {
        // Use fixed grid size for reduction to minimize atomic contention
        // 128 blocks is usually enough to saturate the GPU without causing massive contention
        int grid_size_params = 128; 
        
        // Ensure grid isn't larger than M (though unlikely)
        if (grid_size_params > M) grid_size_params = M;
        
        // Block size: N threads (one per column), capped at max threads
        int block_size_params = (N < 1024) ? ((N + 31) / 32 * 32) : 1024;
        
        ln_backward_param_grad_kernel<<<grid_size_params, block_size_params, 0, stream>>>(
            grad_output, input, mean, rstd, 
            grad_gamma, grad_beta, 
            M, N
        );
    }

    // 2. Input Gradients - if needed
    if (grad_input) {
        // Standard 1D grid layout over M rows
        int block_size_input = (N < 1024) ? ((N + 31) / 32 * 32) : 1024;
        int grid_size_input = M; 
        
        ln_backward_input_grad_kernel<<<grid_size_input, block_size_input, 0, stream>>>(
            grad_output, input, mean, rstd, gamma,
            grad_input, M, N
        );
    }
}