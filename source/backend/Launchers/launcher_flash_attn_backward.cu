#include <backend/Launchers.cuh>
#include <backend/Kernels.cuh>

void launch_flash_backward(
    float* Q, float* K, float* V, float* O, float* dO, float* L_vec,
    float* dQ, float* dK, float* dV,
    int N, int H, int L, int E,
    cudaStream_t stream
) {
    int D = E / H;
    float sm_scale = 1.0f / sqrtf((float)D);

    float* d_Delta;
    cudaMallocAsync(&d_Delta, N * H * L * sizeof(float), stream);
    int threads = 256;
    int blocks = (N * H * L + threads - 1) / threads;
    compute_delta_kernel<<<blocks, threads, 0, stream>>>(dO, O, d_Delta, N, H, L, D);

    cudaMemsetAsync(dQ, 0, N * H * L * D * sizeof(float), stream);

    const int Bc = 32;
    const int Br = 16;
    const int PAD = 8;

    dim3 grid((L + Bc - 1) / Bc, N * H);
    dim3 block(64);


    size_t smem_size = (4 * Bc * (D + PAD) + 2 * Br * (D + PAD)) * sizeof(float);

    int stride_batch = L * H * D;
    int stride_seq = H * D;
    int stride_head = D;

    if (D == 64) {
        flash_attn_backward_kernel<64, 16, 64><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, dO, L_vec, d_Delta, dQ, dK, dV, 
            stride_batch, stride_head, stride_seq, L, sm_scale
        );
    }
    else if (D == 32) {
        flash_attn_backward_kernel<32, 16, 32><<<grid, block, smem_size, stream>>>(
           Q, K, V, O, dO, L_vec, d_Delta, dQ, dK, dV, 
           stride_batch, stride_head, stride_seq, L, sm_scale
       );
    }

    cudaFreeAsync(d_Delta, stream);
}