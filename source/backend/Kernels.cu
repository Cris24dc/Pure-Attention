// headers
#include <backend/Kernels.cuh>
#include <c++/13/cstdint>

__global__ void matmul_kernel_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;

    const uint32_t global_row = TILE_WIDTH * blockIdx.y + ty;
    const uint32_t global_col = TILE_WIDTH * blockIdx.x + tx;

    const uint32_t tile_num_reduction=(N+TILE_WIDTH-1)/TILE_WIDTH;
    float32_t val = 0.0f;

    for(int m = 0; m < tile_num_reduction; m += 1) {
        const uint32_t global_read_row_A = global_row;
        const uint32_t global_read_col_A = m * TILE_WIDTH + tx;

        const uint32_t global_read_row_B = m * TILE_WIDTH + ty;
        const uint32_t global_read_col_B = global_col;
        
        if(global_read_col_A < N && global_read_row_A < M) {
            s_A[ty][tx] = A[global_read_row_A * N + global_read_col_A];
        }
        else {
            s_A[ty][tx] = 0.0f;
        }

        if(global_read_col_B < K && global_read_row_B < N) {
            s_B[ty][tx] = B[global_read_row_B * K + global_read_col_B];
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

    if (global_row < M && global_col < K) {
        C[global_row * K + global_col] = val;
    }
}

__global__ void matadd_kernel_tiled(const float *A, const float *X, float *B, const int M, const int N) {
    const uint32_t global_col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (global_row < M && global_col < N) {
        const uint32_t index = global_row * N + global_col;
        B[index] = A[index] + X[global_col];
    }
}

__global__ void populate_normal(float32_t *A, uint32_t M, uint32_t N, const uint64_t seed) {
    const uint32_t tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t total = M * N;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid0, 0, &state);

    for (uint32_t i = tid0; i < total; i += stride) {
        A[i] = curand_normal(&state);
    }
}

__global__ void ReLU_kernel_tiled(const float *In, float *Out, const u_int32_t M, const uint32_t N) {
    const uint32_t global_col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (global_row < M && global_col < N) {
        uint32_t index = global_row * N + global_col;
        Out[index] = fmaxf(0.0f, In[index]);
    }
}

// gradientul fata de stratul anterior (inputul X al stratului curent)
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

        const uint32_t global_row_WT = blockIdx.x * TILE_WIDTH + ty; // ty parcurge N
        const uint32_t global_col_WT = i * TILE_WIDTH + tx;          // tx pe K

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
        grad_X_in[global_row_gr_X * N + global_col_gr_X]=sum;
    }
}

__global__ void matmul_backward_B_kernel(
    const float* __restrict__ A,
    const float* __restrict__ grad_C,
    float* __restrict__ grad_B,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < K) {
        float sum = 0.0f;

        for (int m = 0; m < M; ++m) {
            float val_A = A[m * N + row];
            float val_grad = grad_C[m * K + col];

            sum += val_A * val_grad;
        }
                grad_B[row * K + col] += sum;
    }
}

__global__ void tensor_add_grad_kernel(const float* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] += src[idx];
    }
}

__global__ void sum_rows_grad_kernel(const float* src, float* dst, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int row = 0; row < M; ++row) {
            sum += src[row * N + col];
        }
        dst[col] += sum;
    }
}

__global__ void relu_backward_kernel(const float* grad_out, const float* input_data, float* grad_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float mask = (input_data[idx] > 0.0f) ? 1.0f : 0.0f;
        grad_in[idx] += grad_out[idx] * mask;
    }
}
