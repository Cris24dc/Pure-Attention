// headers
#include <backend/Kernels.cuh>

// grad_x_in=grad_y_out*W.T
__global__ void matmul_backward_X_kernel(
    const float32_t *grad_Y_out, // M x K (batch x out)
    const float32_t * W_in, // N x K (out x in)
    float32_t *grad_X_in, // M x N  (batch x in)
    const uint32_t M, const uint32_t N, const uint32_t K)
{
    float32_t __shared__ s_grad_Y[TILE_WIDTH][TILE_WIDTH];
    float32_t __shared__ s_W_in[TILE_WIDTH][TILE_WIDTH+1];

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;

    const uint32_t global_col_gr_X = blockIdx.x * TILE_WIDTH + tx;
    const uint32_t global_row_gr_X = blockIdx.y * TILE_WIDTH + ty;

    const uint32_t tile_num_reduction = (K+TILE_WIDTH-1)/TILE_WIDTH;
    float32_t sum = 0.0f;

    for (uint32_t i = 0; i < tile_num_reduction; i+=1) {
        const uint32_t global_row_Y = global_row_gr_X;
        const uint32_t global_col_Y = i * TILE_WIDTH + tx;

        const uint32_t global_row_WT = blockIdx.x * TILE_WIDTH + ty;
        const uint32_t global_col_WT = i * TILE_WIDTH + tx;

        if (global_row_Y < M && global_col_Y < K) {
            s_grad_Y[ty][tx] = grad_Y_out[global_row_Y * K + global_col_Y];
        } else {
            s_grad_Y[ty][tx] = 0.0f;
        }

        if (global_row_WT < N && global_col_WT < K) {
            s_W_in[tx][ty] = W_in[global_row_WT * K + global_col_WT];
        } else {
            s_W_in[tx][ty] = 0.0f;
        }

        __syncthreads();

        for (uint32_t j = 0; j < TILE_WIDTH; j+=1) {
            sum += s_grad_Y[ty][j] * s_W_in[j][tx];
        }

        __syncthreads();
    }

    if (global_col_gr_X < N && global_row_gr_X < M) {
        grad_X_in[global_row_gr_X * N + global_col_gr_X]+=sum;
    }
}
