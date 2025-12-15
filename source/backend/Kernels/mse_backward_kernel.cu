// headers
#include <backend/Kernels.cuh>

__global__ void mse_backward_kernel(
    const float32_t *predictions,
    const float32_t *targets,
    const float32_t *grad_loss_scalar,
    float32_t *grad_predictions,
    const uint32_t N)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        const float32_t diff = predictions[idx] - targets[idx];
        const float32_t factor = 2.0f / static_cast<float>(N);
        const float32_t local_grad = factor * diff * grad_loss_scalar[0];
        grad_predictions[idx] = local_grad;
    }
}
