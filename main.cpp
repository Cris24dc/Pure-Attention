// #include <memory>
// #include <iostream>
// #include <vector>
// #include <layers/Linear.h>
// #include <layers/ReLU.h>
// #include <core/Tensor.h>
// #include <core/Functional.h>
// #include <core/Context.h>
// #include <cuda_runtime.h>
//
// int main() {
//     using namespace layers;
//     using namespace core;
//
//     // cudaStream_t CudaContext::current_stream = 0; // Cannot define static member inside function
//     const uint32_t B = 3;  // redus pentru debugging
//     const uint32_t IN = 5;
//     const uint32_t H = 6;
//     const uint32_t OUT = 3;
//
//     // Hardcoded input values
//     std::vector<float> input_data = {
//         // Batch 0
//         0.5f, -0.3f, 100.2f, -0.89587f, 0.72678f,
//         // Batch 1
//         0.1f, 0.9f, -0.5f, 0.759687f, 0.256784f,
//         // Batch 2
//         0.32f, 0.13f, -0.9f, -0.01f, 0.3687f
//     };
//
//     // Create input tensor and fill with hardcoded values
//     auto input = std::make_shared<Tensor>(std::vector<uint32_t>{B, IN});
//
//     input->to_device(input_data);
//
//     // Create layers (cu seed pentru reproducibilitate)
//     Linear l1(IN, H);
//     ReLU relu;
//     Linear l2(H, OUT);
//
//     // Forward pass
//     auto h = l1.forward(input);
//     auto a = relu.forward(h);
//     auto y = l2.forward(a);
//
//     // Print results
//     auto h_host = h->to_host();
//     auto a_host = a->to_host();
//     auto y_host = y->to_host();
//
//     std::cout << "Input:\n";
//     for (size_t i = 0; i < input_data.size(); i++) {
//         std::cout << input_data[i] << " ";
//         if ((i + 1) % IN == 0) std::cout << "\n";
//     }
//
//     std::cout << "\nAfter L1 (h):\n";
//     for (size_t i = 0; i < h_host.size(); i++) {
//         std::cout << h_host[i] << " ";
//         if ((i + 1) % H == 0) std::cout << "\n";
//     }
//
//     std::cout << "\nAfter ReLU (a):\n";
//     for (size_t i = 0; i < a_host.size(); i++) {
//         std::cout << a_host[i] << " ";
//         if ((i + 1) % H == 0) std::cout << "\n";
//     }
//
//     std::cout << "\nFinal Output (y):\n";
//     for (size_t i = 0; i < y_host.size(); i++) {
//         std::cout << y_host[i] << " ";
//         if ((i + 1) % OUT == 0) std::cout << "\n";
//     }
//
//     std::cout << "\n--- Starting Backward Pass (Loss = Sum(y)) ---\n";
//
//
//     y->backward();
//
//     cudaDeviceSynchronize();
//
//     std::vector<float> w1_grad_host(IN * H);
//     cudaMemcpy(w1_grad_host.data(), l1.weight->get_gradient_ptr(), w1_grad_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
//
//     std::cout << "\nGradients for L1 Weights (First 10):\n";
//     for (int i = 0; i < std::min(10, (int)w1_grad_host.size()); i++) {
//         std::cout << w1_grad_host[i] << " ";
//     }
//     std::cout << "\n";
//
//     std::vector<float> w2_grad_host(H * OUT);
//     cudaMemcpy(w2_grad_host.data(), l2.weight->get_gradient_ptr(), w2_grad_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
//
//     std::cout << "Gradients for L2 Weights (First 10):\n";
//     for (int i = 0; i < std::min(10, (int)w2_grad_host.size()); i++) {
//         std::cout << w2_grad_host[i] << " ";
//     }
//     std::cout << "\n";
//
//     return 0;
// }
//
