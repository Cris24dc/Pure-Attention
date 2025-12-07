#include <backend/Launchers.cuh>
#include <backend/Kernels.cuh>


void launch_flash_attention(
    const std::shared_ptr<core::Tensor>& Q,
    const std::shared_ptr<core::Tensor>& K,
    const std::shared_ptr<core::Tensor>& V,
    std::shared_ptr<core::Tensor>& O,
    cudaStream_t stream
) {
    const int N = Q->get_shape()[0];
    const int L = Q->get_shape()[1];
    const int E = Q->get_shape()[2];
    const int H = 8;
    const int D = E / H;

    // Constants for Tiling
    const int Br = 16;
    const int Bc = 32;

    float scale = 1.0f / sqrtf((float)D);

    int stride_batch = L * E;
    int stride_head = D;
    int stride_seq = E;

    dim3 block(D / 4, Br);
    dim3 grid(N * H, (L + Br - 1) / Br);

    size_t smem_size = (Br * D + Bc * D + Bc * D) * sizeof(float);

    if (D == 64) {
        flash_attention_kernel<16, 32, 64><<<grid, block, smem_size, stream>>>(
            Q->get_data_ptr(), K->get_data_ptr(), V->get_data_ptr(), O->get_data_ptr(),
            N, H, L, scale, stride_batch, stride_head, stride_seq
        );
    } else if (D == 32) {
         flash_attention_kernel<16, 32, 32><<<grid, block, smem_size, stream>>>(
            Q->get_data_ptr(), K->get_data_ptr(), V->get_data_ptr(), O->get_data_ptr(),
            N, H, L, scale, stride_batch, stride_head, stride_seq
        );
    }
}