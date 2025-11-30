// headers
#include <backend/Launchers.h>
#include <backend/Kernels.cuh>
#include <algorithm>
 
using float32_t = float;

void launch_matmul_tiled(float32_t *A, float32_t *B, float32_t *C, int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_kernel_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_matadd_tiled(float32_t *A, float32_t *X, float32_t *B, int M, int N, cudaStream_t stream) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matadd_kernel_tiled<<<grid, block, 0, stream>>>(A, X, B, M, N);
}

void launch_zero_population(float32_t *A, int M, int N, cudaStream_t stream){
    size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
    cudaMemsetAsync(A, 0, total * sizeof(float32_t), stream);

    cudaStreamSynchronize(stream);
}

void launch_ones_population(float32_t *A, int M, int N, cudaStream_t stream){
    size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);
    cudaMemsetAsync(A, 1, total * sizeof(float32_t), stream);

    cudaStreamSynchronize(stream);
}

void launch_normal_population(float32_t *A, int M, int N, cudaStream_t stream){
    size_t total = static_cast<size_t>(M) * static_cast<size_t>(N);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp device_props{};
    cudaGetDeviceProperties(&device_props, device);

    int threads = std::min(256, device_props.maxThreadsPerBlock);
    size_t blocks = (total + threads - 1) / threads;

    int activeBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm,
        populate_normal,    
        threads,
        0                    
    );

    size_t max_blocks = static_cast<size_t>(device_props.multiProcessorCount) * static_cast<size_t>(std::max(1, activeBlocksPerSm));
    if (blocks > max_blocks) blocks = max_blocks;

    unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );

    // seed=42;
    populate_normal<<<blocks, threads, 0, stream>>>(A, M, N, seed);
}

void launch_ReLU_tiled(float32_t *In, float32_t *Out, int M, int N, cudaStream_t stream) {
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    ReLU_kernel_tiled<<<grid, block, 0, stream>>>(In, Out, M, N);
}


void launch_matmul_grad_X(const float32_t* grad_Y_out, const float32_t* W_in, float32_t* grad_X_in,
                          const int M, const int N, const int K, cudaStream_t stream) {

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matmul_backward_X_kernel<<<grid, block, 0, stream>>>(grad_Y_out, W_in, grad_X_in, M, N, K);
}

void launch_matmul_grad_W(const float32_t* A, const float32_t* grad_C, float32_t* grad_B,
                          int M, int N, int K, cudaStream_t stream) {

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((K + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matmul_backward_B_kernel<<<grid, block, 0, stream>>>(A, grad_C, grad_B, M, N, K);
}

void launch_tensor_add_grad(const float32_t* src, float32_t* dst, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    tensor_add_grad_kernel<<<blocks, threads, 0, stream>>>(src, dst, size);
}

void launch_sum_rows_grad(const float32_t* src, float32_t* dst, int M, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sum_rows_grad_kernel<<<blocks, threads, 0, stream>>>(src, dst, M, N);
}

void launch_relu_backward(const float32_t* grad_out, const float32_t* input_data, float32_t* grad_in, int size,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads, 0, stream>>>(grad_out, input_data, grad_in, size);
}