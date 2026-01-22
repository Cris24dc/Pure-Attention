#include <layers/DebugSplit.h>
#include <core/Functional.h>
#include <core/Context.h>
#include <core/Autograd.h>
#include <backend/Launchers.cuh>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace layers {

    // Helper class for backward verification
    struct DebugSplitFunction : public core::Function {
        std::shared_ptr<core::Tensor> Input;
        std::vector<std::weak_ptr<core::Tensor>> Outputs;

        DebugSplitFunction(std::shared_ptr<core::Tensor> in, std::vector<std::shared_ptr<core::Tensor>> outs)
            : Input(in), Outputs(outs.begin(), outs.end()) {}

        void apply_backward() override {
            if (!Input || !Input->requires_grad()) return;

            std::vector<float*> grad_ptrs;
            std::vector<float*> temp_buffers;
            std::vector<std::vector<float>> h_output_grads;

            const cudaStream_t stream = CudaContext::getStream();

            auto input_shape = Input->get_shape();
            uint32_t dim = input_shape.size() - 1;
            uint32_t inner_size = input_shape[dim];
            uint32_t num_splits = Outputs.size();
            uint32_t split_size = inner_size / num_splits;
            
            uint32_t total_input_elements = 1;
            for(auto s : input_shape) total_input_elements *= s;
            uint32_t split_part_elements = total_input_elements / num_splits;

            for (auto& weak_out : Outputs) {
                auto out = weak_out.lock();
                if (out) {
                    grad_ptrs.push_back(out->get_gradient_ptr());
                    h_output_grads.push_back(out->grad_to_host());
                } else {
                    float* zero_ptr;
                    cudaMallocAsync(&zero_ptr, split_part_elements * sizeof(float), stream);
                    cudaMemsetAsync(zero_ptr, 0, split_part_elements * sizeof(float), stream);
                    grad_ptrs.push_back(zero_ptr);
                    temp_buffers.push_back(zero_ptr);
                    
                    std::vector<float> zeros(split_part_elements, 0.0f);
                    h_output_grads.push_back(zeros);
                }
            }

           std::vector<float> grad_before = Input->grad_to_host();
           if (grad_before.empty()) {
               grad_before.resize(total_input_elements, 0.0f);
           }
            launch_concat_backward(
                grad_ptrs,
                Input->get_gradient_ptr(),
                num_splits,
                inner_size,
                split_size,
                total_input_elements,
                stream
            );

            for (float* ptr : temp_buffers) {
                cudaFreeAsync(ptr, stream);
            }
            cudaStreamSynchronize(stream); // Wait for kernel
            std::vector<float> grad_after = Input->grad_to_host();

            std::cout << "[DebugSplit] Verifying Backward Pass..." << std::endl;
            
            bool success = true;
            int errors_found = 0;
            
            std::vector<int> in_strides(input_shape.size());
            int s = 1;
            for (int i = input_shape.size() - 1; i >= 0; --i) {
                in_strides[i] = s;
                s *= input_shape[i];
            }

            for (int i = 0; i < total_input_elements; ++i) {
                float val_accumulated = grad_after[i] - grad_before[i];

                std::vector<int> current_indices(input_shape.size());
                for (int d = 0; d < input_shape.size(); ++d) {
                    current_indices[d] = (i / in_strides[d]) % input_shape[d];
                }

                int idx_at_split = current_indices[dim];
                int part_idx = idx_at_split / split_size;
                int local_idx_at_split = idx_at_split % split_size;

                int out_offset = 0;
                int out_stride_accum = 1;
                for (int d = input_shape.size() - 1; d >= 0; --d) {
                    int idx_val = current_indices[d];
                    if (d == dim) {
                        idx_val = local_idx_at_split;
                    }
                    int dim_sz = (d == dim) ? split_size : input_shape[d];
                    out_offset += idx_val * out_stride_accum;
                    out_stride_accum *= dim_sz;
                }
                
                float val_expected = h_output_grads[part_idx][out_offset];

                if (std::abs(val_accumulated - val_expected) > 1e-5) {
                    if (errors_found < 5) {
                        std::cout << "  FAIL Backward: Index " << i << " Expected=" << val_expected 
                                  << " Actual (Accum)=" << val_accumulated << std::endl;
                    }
                    success = false;
                    errors_found++;
                }
            }

             if (success) {
                std::cout << "[DebugSplit] Backward Verification Passed!" << std::endl;
            } else {
                std::cout << "[DebugSplit] Backward Verification FAILED with " << errors_found << " mismatches." << std::endl;
            }

            Input->backward(false);
        }
    };

    DebugSplit::DebugSplit(uint32_t num_parts, int dim) : num_parts(num_parts), dim(dim) {}

    std::shared_ptr<core::Tensor> DebugSplit::forward(const std::shared_ptr<core::Tensor> &input) {
        throw std::runtime_error("DebugSplit returns a list of Tensors. In C++, call forward_impl. In Python, this is bound to forward.");
    }

    std::vector<std::shared_ptr<core::Tensor>> DebugSplit::forward_impl(const std::shared_ptr<core::Tensor> &input) {
        std::vector<std::shared_ptr<core::Tensor>> outputs;
        
        core::split(input, num_parts, dim, outputs, CudaContext::getStream());

        std::vector<float> h_input = input->to_host();
        std::vector<uint32_t> shape = input->get_shape();

        int split_dim = dim;
        if (split_dim < 0) split_dim += shape.size();

        uint32_t split_dim_size = shape[split_dim];
        uint32_t part_size = split_dim_size / num_parts;

        std::vector<std::vector<float>> h_outputs;
        for (auto& out : outputs) {
            h_outputs.push_back(out->to_host());
        }

        std::vector<int> in_strides(shape.size());
        int s = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            in_strides[i] = s;
            s *= shape[i];
        }

        bool success = true;
        int errors_found = 0;
        int flattened_size = h_input.size();

        for (int i = 0; i < flattened_size; ++i) {
            float val_in = h_input[i];

            std::vector<int> current_indices(shape.size());
            int temp = i;
            for (int d = 0; d < shape.size(); ++d) {
                current_indices[d] = (i / in_strides[d]) % shape[d];
            }

            int idx_at_split = current_indices[split_dim];
            int part_idx = idx_at_split / part_size;
            int local_idx_at_split = idx_at_split % part_size;

            int out_offset = 0;
            int out_stride_accum = 1;
            for (int d = shape.size() - 1; d >= 0; --d) {
               int idx_val = current_indices[d];
               if (d == split_dim) {
                   idx_val = local_idx_at_split;
               }
               int dim_sz = (d == split_dim) ? part_size : shape[d];
               
               out_offset += idx_val * out_stride_accum;
               out_stride_accum *= dim_sz;
            }

            float val_out = h_outputs[part_idx][out_offset];

            if (std::abs(val_in - val_out) > 1e-5) {
                if (errors_found < 10) {
                    std::cout << "[DebugSplit] FAIL: Index " << i << " (Part " << part_idx 
                              << ", LocalOff " << out_offset << ") Input=" 
                              << val_in << " Output=" << val_out << std::endl;
                }
                success = false;
                errors_found++;
            }
        }

        if (success) {
            std::cout << "[DebugSplit] Verification Passed! processed " << flattened_size << " elements." << std::endl;
        } else {
            std::cout << "[DebugSplit] Verification FAILED with " << errors_found << " mismatches." << std::endl;
        }

        if (input->requires_grad()) {
            auto node = std::make_shared<DebugSplitFunction>(input, outputs);
            for (auto& part : outputs) {
                part->set_grad_fn(node);
            }
        }

        return outputs;
    }

}
