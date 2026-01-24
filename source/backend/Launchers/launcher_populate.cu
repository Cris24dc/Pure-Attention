#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>

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

void launch_normal_population(float32_t *A, int M, int N, float32_t std_dev, cudaStream_t stream) {
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
    populate_normal<<<blocks, threads, 0, stream>>>(A, M, N, std_dev, seed);
}