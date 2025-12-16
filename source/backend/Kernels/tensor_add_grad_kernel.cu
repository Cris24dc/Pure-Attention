// headers
#include <backend/Kernels.cuh>

__global__ void tensor_add_grad_kernel(const float* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] += src[idx];
    }
}
