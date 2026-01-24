#include <backend/Kernels.cuh>
#include <backend/Launchers.cuh>

void launch_mse_backward(
    const float* preds,
    const float* targets,
    const float* grad_loss,
    float* grad_preds,
    int N,
    cudaStream_t stream)
{
    size_t total = static_cast<size_t>(N);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp device_props{};
    cudaGetDeviceProperties(&device_props, device);

    int threads = std::min(256, device_props.maxThreadsPerBlock);
    size_t blocks = (total + threads - 1) / threads;

    mse_backward_kernel<<<blocks, threads, 0, stream>>>(
        preds, targets, grad_loss, grad_preds, N
    );
}