// headers
#include <backend/Kernels.cuh>

__global__ void mse_forward_kernel(const float32_t *preds, const float32_t *targets, float32_t *loss_out, const uint32_t N) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        const float32_t diff = preds[idx] - targets[idx];
        const float32_t sq_diff = diff * diff;

        atomicAdd(loss_out, sq_diff);
    }
}
