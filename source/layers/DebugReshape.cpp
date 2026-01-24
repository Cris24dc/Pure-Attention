#include <layers/DebugReshape.h>
#include <core/Functional.h>
#include <core/Context.h>
#include <core/Autograd.h>
#include <backend/Launchers.cuh>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace layers {

    struct DebugReshapeFunction : public core::Function {
        std::shared_ptr<core::Tensor> Input;
        std::weak_ptr<core::Tensor> Output;
        std::vector<uint32_t> original_shape;

        DebugReshapeFunction(std::shared_ptr<core::Tensor> in, std::shared_ptr<core::Tensor> out)
            : Input(in), Output(out) {
            original_shape = in->get_shape();
        }

        void apply_backward() override {
            auto out_ptr = Output.lock();
            if (!out_ptr) return;

            if (Input->requires_grad()) {
                const cudaStream_t stream = CudaContext::getStream();
                
                size_t total_elements = 1;
                for (auto s : Input->get_shape()) total_elements *= s;
                
                std::vector<float> grad_before = Input->grad_to_host();
                 if (grad_before.empty()) {
                   grad_before.resize(total_elements, 0.0f);
               }

                uint32_t size = 1;
                for (auto s : out_ptr->get_shape()) size *= s;

                launch_tensor_add_grad(out_ptr->get_gradient_ptr(), Input->get_gradient_ptr(), size, stream);
                
                cudaStreamSynchronize(stream);
                std::vector<float> grad_after = Input->grad_to_host();
                std::vector<float> grad_output = out_ptr->grad_to_host();

                std::cout << "[DebugReshape] Verifying Backward Pass..." << std::endl;
                
                bool success = true;
                int errors = 0;
                for(size_t i = 0; i < total_elements; ++i) {
                    float accum = grad_after[i] - grad_before[i];
                    float expected = grad_output[i]; 
                    
                    if (std::abs(accum - expected) > 1e-5) {
                        if (errors < 5) {
                            std::cout << "  FAIL Backward: Index " << i << " Expected Added=" << expected 
                                      << " Actual Added=" << accum 
                                      << " (Before=" << grad_before[i] << ", After=" << grad_after[i] << ")" << std::endl;
                        }
                        success = false;
                        errors++;
                    }
                }

                if (success) {
                    std::cout << "[DebugReshape] Backward Verification Passed!" << std::endl;
                } else {
                    std::cout << "[DebugReshape] Backward Verification FAILED with " << errors << " mismatches." << std::endl;
                }

                Input->backward(false);
            }
        }
    };

    DebugReshape::DebugReshape(std::vector<uint32_t> target_shape) : target_shape(target_shape) {}

    std::shared_ptr<core::Tensor> DebugReshape::forward(const std::shared_ptr<core::Tensor> &input) {
        std::shared_ptr<core::Tensor> output;
        
        core::reshape(input, target_shape, output, CudaContext::getStream());

        std::vector<float> h_in = input->to_host();
        std::vector<float> h_out = output->to_host();
        
        bool success = true;
        if (h_in.size() != h_out.size()) {
            std::cout << "[DebugReshape] FAIL Forward: Size mismatch!" << std::endl;
            success = false;
        } else {
            for(size_t i=0; i<h_in.size(); ++i) {
                if (std::abs(h_in[i] - h_out[i]) > 1e-5) {
                    std::cout << "[DebugReshape] FAIL Forward: Data mismatch at index " << i << std::endl;
                    success = false;
                    break;
                }
            }
        }
        
        if (success) {
            std::cout << "[DebugReshape] Forward Verification Passed!" << std::endl;
        }

        if (input->requires_grad()) {
            auto node = std::make_shared<DebugReshapeFunction>(input, output);
            output->set_grad_fn(node);
        }

        return output;
    }

}
