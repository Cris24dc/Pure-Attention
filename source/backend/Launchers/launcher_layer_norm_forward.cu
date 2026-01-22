#include <backend/Launchers.cuh>
#include <backend/Kernels.cuh>

void launch_layer_norm_forward(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    float* mean,
    float* rstd,
    uint32_t M,
    uint32_t N,
    float epsilon,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = M;
    
    layer_norm_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, gamma, beta, output, mean, rstd, M, N, epsilon
    );
}