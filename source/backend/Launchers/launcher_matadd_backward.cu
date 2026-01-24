#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>

void launch_tensor_add_grad(const float32_t* src, float32_t* dst, int size, cudaStream_t stream) {
    static int minGridSize = 0;
    static int blockSize = 0;
    if (blockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, tensor_add_grad_kernel, 0, 0);
    }
    int grids = (size + blockSize - 1) / blockSize;
    tensor_add_grad_kernel<<<grids, blockSize, 0, stream>>>(src, dst, size);
}

void launch_sum_rows_grad(const float32_t* src, float32_t* dst, int M, int N, cudaStream_t stream) {
    static int minGridSize = 0;
    static int blockSize = 0;
    if (blockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sum_rows_grad_kernel, 0, 0);
    }
    int grids = (N + blockSize - 1) / blockSize;
    sum_rows_grad_kernel<<<grids, blockSize, 0, stream>>>(src, dst, M, N);
}