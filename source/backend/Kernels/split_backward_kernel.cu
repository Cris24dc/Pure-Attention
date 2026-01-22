#include <backend/Kernels.cuh>


__global__ void concat_last_dim_kernel(
    float32_t** inputs,
    float32_t* output,
    uint32_t num_splits,
    uint32_t inner_size,
    uint32_t split_size,
    uint32_t total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int row = idx / inner_size;
    int col = idx % inner_size;

    int split_idx = col / split_size;
    int col_in_split = col % split_size;

    atomicAdd(output + idx, inputs[split_idx][row * split_size + col_in_split]);
}