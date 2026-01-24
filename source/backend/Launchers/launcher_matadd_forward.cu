#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>

void launch_matadd_tiled(float32_t *A, float32_t *X, float32_t *B, int M, int N, cudaStream_t stream) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matadd_kernel_tiled<<<grid, block, 0, stream>>>(A, X, B, M, N);
}
