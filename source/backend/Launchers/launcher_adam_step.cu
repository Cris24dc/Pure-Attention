#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>

void launch_adam_step(
    float32_t* params,
    const float32_t* grads,
    float32_t* m,
    float32_t* v,
    int size,
    float32_t beta1,
    float32_t beta2,
    float32_t epsilon,
    float32_t lr,
    const float32_t beta1_corr,
    const float32_t beta2_corr,
    cudaStream_t stream)
{
    static int minGridSize = 0;
    static int blockSize = 0;
    if (blockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, adam_step_kernel, 0, 0);
    }
    int grids = (size + blockSize - 1) / blockSize;
    adam_step_kernel<<<grids, blockSize, 0, stream>>>(params, grads, m, v, size, beta1, beta2, epsilon, lr, beta1_corr, beta2_corr);
}


