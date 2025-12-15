// headers
#include <backend/Kernels.cuh>

__global__ void populate_normal(float32_t *A, uint32_t M, uint32_t N, float32_t std_dev, const uint64_t seed) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    const uint32_t total = M * N;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    for (uint32_t i = tid; i < total; i += stride) {
        A[i] = curand_normal(&state) * std_dev; 
    }
}
