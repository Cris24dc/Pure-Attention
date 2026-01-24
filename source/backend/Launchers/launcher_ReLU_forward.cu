#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>

void launch_ReLU_tiled(float32_t *In, float32_t *Out, int M, int N, cudaStream_t stream) {
    static int minGridSize = 0;
    static int maxBlockSize = 0;
    
    if (maxBlockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, ReLU_kernel_tiled, 0, 0);
    }

    int dimx = 32;
    int dimy = maxBlockSize / dimx;

    dim3 block(dimx, dimy);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    ReLU_kernel_tiled<<<grid, block, 0, stream>>>(In, Out, M, N);
}