#include <backend/Launchers.cuh>
#include <backend/Kernels.cuh>
#include <vector>


void launch_concat_backward(
    std::vector<float*>& input_grads,
    float* output_grad,
    uint32_t num_splits, 
    uint32_t inner_size, 
    uint32_t split_size, 
    uint32_t total_elements, 
    cudaStream_t stream)
{
    float** d_in_ptrs;
    cudaMallocAsync(&d_in_ptrs, num_splits * sizeof(float*), stream);
    cudaMemcpyAsync(d_in_ptrs, input_grads.data(), num_splits * sizeof(float*), cudaMemcpyHostToDevice, stream);

    static int minGridSize = 0;
    static int blockSize = 0;
    if (blockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, concat_last_dim_kernel, 0, 0);
    }
    int grids = (total_elements + blockSize - 1) / blockSize;

    concat_last_dim_kernel<<<grids, blockSize, 0, stream>>>(
        d_in_ptrs,
        output_grad,
        num_splits,
        inner_size,
        split_size,
        total_elements
    );

    cudaFreeAsync(d_in_ptrs, stream);
}
