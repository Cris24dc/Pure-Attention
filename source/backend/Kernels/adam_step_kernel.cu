// headers
#include <backend/Kernels.cuh>

__global__ void adam_step_kernel(
    float32_t* params,
    const float32_t* grads,
    float32_t* m,
    float32_t* v,
    const uint32_t size,
    const float32_t beta1,
    const float32_t beta2,
    const float32_t epsilon,
    const float32_t alpha,
    const float32_t beta1_corr,
    const float32_t beta2_corr) 
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const float32_t g = grads[idx];
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

        const float32_t m_hat = m[idx] / beta1_corr;
        const float32_t v_hat = v[idx] / beta2_corr;

        params[idx] -= alpha * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}
