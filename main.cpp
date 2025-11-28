#include <memory>
#include <iostream>
#include <vector>
#include "include/layers/Linear.h"
#include "include/layers/ReLU.h"
#include "include/core/Tensor.h"
#include "include/core/Functional.h"
#include "include/core/Context.h"
#include <cuda_runtime.h>

int main() {
    using namespace layers;
    using namespace core;
    
    // cudaStream_t CudaContext::current_stream = 0; // Cannot define static member inside function
    const uint32_t B = 2;  // redus pentru debugging
    const uint32_t IN = 4;
    const uint32_t H = 3;
    const uint32_t OUT = 2;

    // Hardcoded input values
    std::vector<float> input_data = {
        // Batch 0
        0.5f, -0.3f, 1.2f, -0.8f,
        // Batch 1
        0.1f, 0.9f, -0.5f, 0.7f
    };

    // Create input tensor and fill with hardcoded values
    auto input = std::make_shared<Tensor>(std::vector<uint32_t>{B, IN});
    
    input->to_device(input_data);

    // Create layers (cu seed pentru reproducibilitate)
    Linear l1(IN, H);
    ReLU relu;
    Linear l2(H, OUT);

    // Forward pass
    auto h = l1.forward(input);
    auto a = relu.forward(h);
    auto y = l2.forward(a);

    // Print results
    auto h_host = h->to_host();
    auto a_host = a->to_host();
    auto y_host = y->to_host();

    std::cout << "Input:\n";
    for (size_t i = 0; i < input_data.size(); i++) {
        std::cout << input_data[i] << " ";
        if ((i + 1) % IN == 0) std::cout << "\n";
    }

    std::cout << "\nAfter L1 (h):\n";
    for (size_t i = 0; i < h_host.size(); i++) {
        std::cout << h_host[i] << " ";
        if ((i + 1) % H == 0) std::cout << "\n";
    }

    std::cout << "\nAfter ReLU (a):\n";
    for (size_t i = 0; i < a_host.size(); i++) {
        std::cout << a_host[i] << " ";
        if ((i + 1) % H == 0) std::cout << "\n";
    }

    std::cout << "\nFinal Output (y):\n";
    for (size_t i = 0; i < y_host.size(); i++) {
        std::cout << y_host[i] << " ";
        if ((i + 1) % OUT == 0) std::cout << "\n";
    }

    // Print first 10 weights from l1 for verification
    std::cout << "\nFirst 10 weights from l1:\n";
    auto w1 = l1.weight->to_host();
    for (int i = 0; i < std::min(10, (int)w1.size()); i++) {
        std::cout << w1[i] << " ";
    }
    std::cout << "\n";

    return 0;
}