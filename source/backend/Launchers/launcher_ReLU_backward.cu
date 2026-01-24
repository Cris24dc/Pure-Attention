#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>

void launch_relu_backward(const float32_t* grad_out, const float32_t* input_data, float32_t* grad_in, int size,
    cudaStream_t stream) {
    static int minGridSize = 0;
    static int blockSize = 0;
    if (blockSize == 0) {
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, relu_backward_kernel, 0, 0);
    }
    int grids = (size + blockSize - 1) / blockSize;

    relu_backward_kernel<<<grids, blockSize, 0, stream>>>(grad_out, input_data, grad_in, size);
}
