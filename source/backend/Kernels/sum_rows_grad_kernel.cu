// headers
#include <backend/Kernels.cuh>

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
