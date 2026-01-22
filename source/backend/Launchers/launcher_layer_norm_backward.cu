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
    const int threads = 256;
    const int blocks = M;
    
    layer_norm_backward_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, input, mean, rstd, gamma,
        grad_input, grad_gamma, grad_beta, M, N
    );
}