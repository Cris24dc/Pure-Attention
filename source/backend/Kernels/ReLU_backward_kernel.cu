// headers
#include <backend/Kernels.cuh>

__global__ void relu_backward_kernel(const float* grad_out, const float* input_data, float* grad_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float mask = (input_data[idx] > 0.0f) ? 1.0f : 0.0f;
        grad_in[idx] += grad_out[idx] * mask;
    }
}
