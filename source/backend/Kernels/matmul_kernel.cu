// headers
#include <backend/Kernels.cuh>

__global__ void matmul_kernel_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;

    const uint32_t global_row = TILE_WIDTH * blockIdx.y + ty;
    const uint32_t global_col = TILE_WIDTH * blockIdx.x + tx;

    const uint32_t tile_num_reduction=(K+TILE_WIDTH-1)/TILE_WIDTH;
    float32_t val = 0.0f;

    for(int m = 0; m < tile_num_reduction; m += 1) {
        const uint32_t global_read_row_A = global_row;
        const uint32_t global_read_col_A = m * TILE_WIDTH + tx;

        const uint32_t global_read_row_B = m * TILE_WIDTH + ty;
        const uint32_t global_read_col_B = global_col;
        
        if(global_read_col_A < K && global_read_row_A < M) {
            s_A[ty][tx] = A[global_read_row_A * K + global_read_col_A];
        }
        else {
            s_A[ty][tx] = 0.0f;
        }

        if(global_read_col_B < N && global_read_row_B < K) {
            s_B[ty][tx] = B[global_read_row_B * N + global_read_col_B];
        }
        else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            val += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    } 

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = val;
    }
}
