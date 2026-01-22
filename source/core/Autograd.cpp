#include <core/Tensor.h>
#include <backend/Launchers.cuh>
#include <core/Context.h>
#include <core/Autograd.h>
#include <cstdio>


namespace core {
    MatMulFunction::MatMulFunction(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w, std::shared_ptr<Tensor> y)
        : X_input(x), W_input(w), Y_output(y) {}

    void MatMulFunction::apply_backward()  {
        const uint32_t M = X_input->get_shape()[0];
        const uint32_t N = X_input->get_shape()[1];
        const uint32_t K = W_input->get_shape()[1];

        const auto grad_out_ptr = Y_output.lock()->get_gradient_ptr();

        if (X_input->requires_grad()) {
            launch_matmul_grad_X(grad_out_ptr,W_input->get_data_ptr(), X_input->get_gradient_ptr(),M, N, K,
                CudaContext::getStream());
            X_input->backward(false);
        }

        if (W_input->requires_grad()) {
            launch_matmul_grad_W(X_input->get_data_ptr(),grad_out_ptr,W_input->get_gradient_ptr(), M, N, K,
                CudaContext::getStream());
            W_input->backward(false);
        }
    }

    AddFunction::AddFunction(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias, std::shared_ptr<Tensor> y)
        : X_input(x), bias_input(bias), Y_output(y) {}

    void AddFunction::apply_backward() {
        auto out_ptr = Y_output.lock();
        if (!out_ptr) return;

        const auto grad_out_ptr = out_ptr->get_gradient_ptr();
        uint32_t M = out_ptr->get_shape()[0];
        uint32_t N = out_ptr->get_shape()[1];
        uint32_t size = M * N;

        if (X_input->requires_grad()) {
            launch_tensor_add_grad(grad_out_ptr,X_input->get_gradient_ptr(),size, CudaContext::getStream());
            X_input->backward(false);
        }

        if (bias_input->requires_grad()) {
            bool is_bias = (bias_input->get_shape()[0] == 1 && M > 1);

            if (is_bias)
                launch_sum_rows_grad(grad_out_ptr,bias_input->get_gradient_ptr(), M, N, CudaContext::getStream());
            else
                launch_tensor_add_grad(grad_out_ptr,bias_input->get_gradient_ptr(),size,CudaContext::getStream());

            bias_input->backward(false);
        }
    }

    ReLUFunction::ReLUFunction(std::shared_ptr<Tensor> in, std::shared_ptr<Tensor> out)
        : Input(in), Output(out) {}

    void ReLUFunction::apply_backward()  {
        auto out_ptr = Output.lock();
        if (!out_ptr) return;

        uint32_t size = 1;
        for (auto s : Input->get_shape()) {
            size *= s;
        }

        if (Input->requires_grad()) {
            launch_relu_backward(
                out_ptr->get_gradient_ptr(), 
                Input->get_data_ptr(), 
                Input->get_gradient_ptr(),
                size, 
                CudaContext::getStream()
            );
            Input->backward(false);
        }
    }

    MSEFunction::MSEFunction(
        const std::shared_ptr<Tensor>& preds,
        const std::shared_ptr<Tensor>& targs,
        const std::shared_ptr<Tensor>& out)
        : predictions(preds), targets(targs), output_loss(out) {}

    void MSEFunction::apply_backward()  {
        auto loss_ptr = output_loss.lock();
        if (!loss_ptr) return;

        uint32_t N = 1;
        for(auto s : predictions->get_shape()) N *= s;

        if (predictions->requires_grad()) {
            launch_mse_backward(
                predictions->get_data_ptr(),
                targets->get_data_ptr(),
                loss_ptr->get_gradient_ptr(),
                predictions->get_gradient_ptr(),
                N,
                CudaContext::getStream()
            );

            predictions->backward(false);
        }
    }


    SplitFunction::SplitFunction(std::shared_ptr<Tensor> in, std::vector<std::shared_ptr<Tensor>> outs)
        : Input(in), Outputs(outs.begin(), outs.end()) {}

    void SplitFunction::apply_backward() {
        if (!Input) return;

        if (Input->requires_grad()) {
            std::vector<float*> grad_ptrs;
            std::vector<float*> temp_buffers;
            
            const cudaStream_t stream = CudaContext::getStream();

            auto input_shape = Input->get_shape();
            uint32_t dim = input_shape.size() - 1;
            uint32_t inner_size = input_shape[dim];
            uint32_t num_splits = Outputs.size();
            uint32_t split_size = inner_size / num_splits;
            
            uint32_t total_input_elements = 1;
            for(auto s : input_shape) total_input_elements *= s;
            uint32_t split_part_elements = total_input_elements / num_splits;

            #ifdef DEBUG_SPLIT
            printf("Split backward: total=%u, splits=%u, split_size=%u, part_elements=%u\n",
                total_input_elements, num_splits, split_size, split_part_elements);
            #endif

            for (auto& weak_out : Outputs) {
                auto out = weak_out.lock();
                if (out) {
                    uint32_t out_elements = 1;
                    for(auto s : out->get_shape()) out_elements *= s;
                    
                    if (out_elements != split_part_elements) {
                        fprintf(stderr, "ERROR: Split output size mismatch: %u != %u\n", 
                                out_elements, split_part_elements);
                    }
                    
                    grad_ptrs.push_back(out->get_gradient_ptr());
                } else {
                    float* zero_ptr;
                    cudaMallocAsync(&zero_ptr, split_part_elements * sizeof(float), stream);
                    cudaMemsetAsync(zero_ptr, 0, split_part_elements * sizeof(float), stream);
                    grad_ptrs.push_back(zero_ptr);
                    temp_buffers.push_back(zero_ptr);
                }
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

            Input->backward(false);
        }
    }


    ReshapeFunction::ReshapeFunction(std::shared_ptr<Tensor> in, std::shared_ptr<Tensor> out)
        : Input(in), Output(out) {
        original_shape = in->get_shape();
    }

    void ReshapeFunction::apply_backward() {
        auto out_ptr = Output.lock();
        if (!out_ptr) return;

        if (Input->requires_grad()) {
            uint32_t size = 1;
            for (auto s : out_ptr->get_shape()) size *= s;

            launch_tensor_add_grad(out_ptr->get_gradient_ptr(), Input->get_gradient_ptr(), size, CudaContext::getStream());
            
            cudaStreamSynchronize(CudaContext::getStream());

            Input->backward(false);
        }
    }
  
  
    FlashAttentionFunction::FlashAttentionFunction(
        const std::shared_ptr<Tensor>& q,
        const std::shared_ptr<Tensor>& k,
        const std::shared_ptr<Tensor>& v,
        const std::shared_ptr<Tensor>& o,
        const std::shared_ptr<Tensor>& lcache)
        : Q_input(q), K_input(k), V_input(v), O_output(o), L_cache(lcache) {}

    void FlashAttentionFunction::apply_backward() {
        auto out_ptr = O_output.lock();
        if (!out_ptr) return;

        const int N = Q_input->get_shape()[0];
        const int L = Q_input->get_shape()[1];
        const int E = Q_input->get_shape()[2];
        const int H = 8;
        const int D = E / H;

        const cudaStream_t stream = CudaContext::getStream();

        float *temp_dQ = nullptr, *temp_dK = nullptr, *temp_dV = nullptr;
        size_t bytes = (size_t)N * H * L * D * sizeof(float);

        if (Q_input->requires_grad()) {
            cudaMallocAsync(&temp_dQ, bytes, stream);
            cudaMemsetAsync(temp_dQ, 0, bytes, stream);
        }
        if (K_input->requires_grad()) {
            cudaMallocAsync(&temp_dK, bytes, stream);
            cudaMemsetAsync(temp_dK, 0, bytes, stream);
        }
        if (V_input->requires_grad()) {
            cudaMallocAsync(&temp_dV, bytes, stream);
            cudaMemsetAsync(temp_dV, 0, bytes, stream);
        }

        launch_flash_backward(
            Q_input->get_data_ptr(), K_input->get_data_ptr(), V_input->get_data_ptr(),
            out_ptr->get_data_ptr(), out_ptr->get_gradient_ptr(), L_cache->get_data_ptr(),
            temp_dQ, temp_dK, temp_dV,
            N, H, L, E,
            stream
        );

        if (Q_input->requires_grad()) {
            launch_tensor_add_grad(temp_dQ, Q_input->get_gradient_ptr(), N * H * L * D, stream);
            cudaFreeAsync(temp_dQ, stream);
            Q_input->backward(false);
        }

        if (K_input->requires_grad()) {
            launch_tensor_add_grad(temp_dK, K_input->get_gradient_ptr(), N * H * L * D, stream);
            cudaFreeAsync(temp_dK, stream);
            K_input->backward(false);
        }

        if (V_input->requires_grad()) {
            launch_tensor_add_grad(temp_dV, V_input->get_gradient_ptr(), N * H * L * D, stream);
            cudaFreeAsync(temp_dV, stream);
            V_input->backward(false);
        }
    }


    LayerNormFunction::LayerNormFunction(
        const std::shared_ptr<Tensor>& inp,
        const std::shared_ptr<Tensor>& g,
        const std::shared_ptr<Tensor>& b,
        const std::shared_ptr<Tensor>& out,
        const std::shared_ptr<Tensor>& m,
        const std::shared_ptr<Tensor>& r,
        uint32_t m_size,
        uint32_t n_size)
        : input(inp), gamma(g), beta(b), output(out), 
          mean(m), rstd(r), M(m_size), N(n_size) {}
    
    void LayerNormFunction::apply_backward() {
        auto out_ptr = output.lock();
        if (!out_ptr) return;
        
        const cudaStream_t stream = CudaContext::getStream();
        
        float* grad_output = out_ptr->get_gradient_ptr();
        float* grad_input = input->requires_grad() ? input->get_gradient_ptr() : nullptr;
        float* grad_gamma = gamma->requires_grad() ? gamma->get_gradient_ptr() : nullptr;
        float* grad_beta = beta->requires_grad() ? beta->get_gradient_ptr() : nullptr;
        
        launch_layer_norm_backward(
            grad_output,
            input->get_data_ptr(),
            mean->get_data_ptr(),
            rstd->get_data_ptr(),
            gamma->get_data_ptr(),
            grad_input,
            grad_gamma,
            grad_beta,
            M, N, stream
        );
        
        if (input->requires_grad()) {
            input->backward(false);
        }
        if (gamma->requires_grad()) {
            gamma->backward(false);
        }
        if (beta->requires_grad()) {
            beta->backward(false);
        }
    }

}
