#include <backend/Launchers.cuh>
#include <backend/Kernels.cuh>

void launch_flash_backward_optimized(
    float* Q, float* K, float* V, float* O, float* dO, float* L_vec,
    float* dQ, float* dK, float* dV,
    int N, int H, int L, int E,
    cudaStream_t stream
) {
    int D = E / H;
    float sm_scale = 1.0f / sqrtf((float)D);

    float* d_Delta;
    cudaMallocAsync(&d_Delta, N * H * L * sizeof(float), stream);

    int total_threads = N * H * L;
    int blocks_delta = (total_threads + 255) / 256;
    compute_delta_kernel<<<blocks_delta, 256, 0, stream>>>(dO, O, d_Delta, N, H, L, D);

    cudaMemsetAsync(dQ, 0, N * L * E * sizeof(float), stream);
    cudaMemsetAsync(dK, 0, N * L * E * sizeof(float), stream);
    cudaMemsetAsync(dV, 0, N * L * E * sizeof(float), stream);

    const int Br = 16;
    const int Bc = 32;

    dim3 grid((L + Br - 1) / Br, 1, N * H);
    dim3 block(32);
    size_t smem_size = (2 * Br * D + 2 * Bc * D) * sizeof(float);

    if (D == 64) {
        flash_attn_backward_kernel<16, 32, 64><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, dO, L_vec, d_Delta, dQ, dK, dV, H, 1, E, L, sm_scale
        );
    } else if (D == 32) {
         flash_attn_backward_kernel<16, 32, 32><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, dO, L_vec, d_Delta, dQ, dK, dV, H, 1, E, L, sm_scale
        );
    }

    cudaFreeAsync(d_Delta, stream);
}