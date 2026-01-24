#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>

void launch_matmul_grad_X(const float32_t* grad_Y_out, const float32_t* W_in, float32_t* grad_X_in,
                          const int M, const int N, const int K, cudaStream_t stream) {

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matmul_backward_X_kernel<<<grid, block, 0, stream>>>(grad_Y_out, W_in, grad_X_in, M, N, K);
}

void launch_matmul_grad_W(const float32_t *X_in, const float32_t *grad_Y_out, float32_t *grad_W_in,
                          const int M, const int N, const int K, cudaStream_t stream) {

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((K + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matmul_backward_W_kernel<<<grid, block, 0, stream>>>(X_in, grad_Y_out, grad_W_in, M, N, K);
}
