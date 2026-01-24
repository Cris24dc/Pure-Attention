#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>

void launch_mse_forward(const float* preds, const float* targets, float* loss_out, int N, cudaStream_t stream) {
    cudaGetLastError();

    static int minGridSize = 0;
    static int blockSize = 0;
    if (blockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mse_forward_kernel, 0, 0);
    }
    int grids = (N + blockSize - 1) / blockSize;

    cudaError_t err_memset = cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
    if (err_memset != cudaSuccess) {
        printf("CUDA Error in MSE Memset: %s\n", cudaGetErrorString(err_memset));
        return;
    }

    mse_forward_kernel<<<grids, blockSize, 0, stream>>>(preds, targets, loss_out, N);

    cudaError_t err_launch = cudaGetLastError();
    if (err_launch != cudaSuccess) {
        printf("CUDA Error in MSE Kernel Launch: %s\n", cudaGetErrorString(err_launch));
    }

    mse_div_kernel<<<1, 1, 0, stream>>>(loss_out, N);
}