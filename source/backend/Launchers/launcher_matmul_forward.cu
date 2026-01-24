#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>


void launch_matmul_tiled(float32_t *A, float32_t *B, float32_t *C, int M, int N, int K, cudaStream_t stream) {
    static bool cacheConfigured = false;
    if (!cacheConfigured) {
        cudaFuncSetCacheConfig(matmul_kernel_tiled, cudaFuncCachePreferL1);
        cacheConfigured = true;
    }
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_kernel_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
