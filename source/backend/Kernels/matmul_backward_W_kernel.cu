// headers
#include <backend/Kernels.cuh>

// grad_W = X.T * gard_Y
__global__ void matmul_backward_W_kernel(
    const float32_t *X_in,          // M x N (batch x in)
    const float32_t *grad_Y_out,    // M x K (batch x out)
    float32_t *grad_W_in,           // N x K (in x out)
    const u_int32_t M, const u_int32_t N, const uint32_t K)
{
    float32_t __shared__ s_X_in[TILE_WIDTH][TILE_WIDTH];
    float32_t __shared__ s_grad_Y[TILE_WIDTH][TILE_WIDTH+1];

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;

    const uint32_t global_col_gr_W = blockIdx.x * TILE_WIDTH + tx;
    const uint32_t global_row_gr_W = blockIdx.y * TILE_WIDTH + ty;

    const uint32_t tile_num_reduction = (M+TILE_WIDTH-1)/TILE_WIDTH;
    float32_t sum = 0.0f;

    for (uint32_t i = 0; i < tile_num_reduction; i+=1) {

        const uint32_t global_row_X = i * TILE_WIDTH + ty;
        const uint32_t global_col_X = blockIdx.y * TILE_WIDTH + tx;

        const uint32_t global_row_Y = i * TILE_WIDTH + ty;
        const uint32_t global_col_Y = blockIdx.x * TILE_WIDTH + tx;

        if (global_row_X < M && global_col_X < N) {
            s_X_in[tx][ty] = X_in[global_row_X * N + global_col_X];
        } else {
            s_X_in[tx][ty] = 0.0f;
        }

        if (global_row_Y < M && global_col_Y < K) {
            s_grad_Y[ty][tx] = grad_Y_out[global_row_Y * K + global_col_Y];
        } else {
            s_grad_Y[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (uint32_t j = 0; j < TILE_WIDTH; j+=1) {
            sum += s_X_in[ty][j] * s_grad_Y[j][tx];
        }

        __syncthreads();
    }

    if (global_row_gr_W < N && global_col_gr_W < K) {
        grad_W_in[global_row_gr_W * K + global_col_gr_W] += sum;
    }
}
