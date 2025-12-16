// headers
#include <backend/Kernels.cuh>

__global__ void ReLU_kernel_tiled(const float *In, float *Out, const u_int32_t M, const uint32_t N) {
    const uint32_t global_col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t global_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (global_row < M && global_col < N) {
        uint32_t index = global_row * N + global_col;
        Out[index] = fmaxf(0.0f, In[index]);
    }
}
