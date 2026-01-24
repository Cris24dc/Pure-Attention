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
    static int minGridSize = 0;
    static int maxBlockSize = 0;
    if (maxBlockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, layer_norm_forward_kernel, 0, 0);
    }
    
    // Don't launch more threads than N
    int blockSize = (N < maxBlockSize) ? ((N + 31) / 32 * 32) : maxBlockSize;
    int grids = M;
    layer_norm_forward_kernel<<<grids, blockSize, 0, stream>>>(
        input, gamma, beta, output, mean, rstd, M, N, epsilon
    );
}