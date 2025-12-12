#include <backend/Launchers.cuh>
#include <backend/Kernels.cuh>


void launch_flash_attention(
    const std::shared_ptr<core::Tensor>& Q,
    const std::shared_ptr<core::Tensor>& K,
    const std::shared_ptr<core::Tensor>& V,
    std::shared_ptr<core::Tensor>& O,
    std::shared_ptr<core::Tensor>& L_cache, // UPDATED: Necesar pentru Backward Pass (Line 2 & 15 din paper)
    cudaStream_t stream
) {
    const int N = Q->get_shape()[0];
    const int L = Q->get_shape()[1];
    const int E = Q->get_shape()[2];
    const int H = 8;
    const int D = E / H;

    // N = batch_size
    // L = sequence len
    // E = embeddings dim (model size)
    // H = number of heads
    // D = head dimension

    constexpr int B_r = 16;
    constexpr int B_c = 32;

    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    const int stride_batch = L * E;
    const int stride_head = D;
    const int stride_seq = E;


    // as the original paper mentions in pseudocode that Q, K and V matrices has Nxd dimension
    // take in consideration that it reefers to only one head, so our notation is LxD
    // which is logic when we think that we want to put attention probabilities on heads of our phrase
    // in addition the paper mentions that we divide Q into L/B_r blocks we consider block size B_r x D / 4
    // due to sram low capacity and vectorized memory access respectively

    // 1: Divide Q into Tr = ceil(N/Br) blocks... and divide K, V into Tc = ceil(N/Bc) blocks...
    dim3 block(D / 4, B_r);
    dim3 grid(N * H, (L + B_r - 1) / B_r);

    // we also collapse N and H dimensions in only one because all (batch, head) paris are independent
    // the scope of the grid s second dimension is to cover the whole dimension of the L fractured in

    // total = (grid.x) * (grid.y * block.y) * (block.x * 4)
    // total =  (N * H) * ( ceil(L) )    * (D / 4 * 4)

    size_t smem_size = (B_r * D + B_c * D + B_c * D) * sizeof(float);

    // 2: Divide the output O ... into Tr blocks... and divide the logsumexp L into Tr blocks Li...

    if (D == 64) {
        flash_attention_kernel<16, 32, 64><<<grid, block, smem_size, stream>>>(
            Q->get_data_ptr(), K->get_data_ptr(), V->get_data_ptr(), O->get_data_ptr(),
            L_cache->get_data_ptr(),
            N, H, L, scale, stride_batch, stride_head, stride_seq
        );
    } else if (D == 32) {
         flash_attention_kernel<16, 32, 32><<<grid, block, smem_size, stream>>>(
            Q->get_data_ptr(), K->get_data_ptr(), V->get_data_ptr(), O->get_data_ptr(),
            L_cache->get_data_ptr(),
            N, H, L, scale, stride_batch, stride_head, stride_seq
        );
    }
}