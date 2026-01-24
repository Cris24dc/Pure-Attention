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
    static int minGridSize = 0;
    static int maxBlockSize = 0;
    if (maxBlockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, layer_norm_backward_kernel, 0, 0);
    }
    
    // Don't launch more threads than N
    int blockSize = (N < maxBlockSize) ? ((N + 31) / 32 * 32) : maxBlockSize;
    int grids = M;
    layer_norm_backward_kernel<<<grids, blockSize, 0, stream>>>(
        grad_output, input, mean, rstd, gamma,
        grad_input, grad_gamma, grad_beta, M, N
    );
}