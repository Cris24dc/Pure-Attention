#include <core/Tensor.h>
#include <backend/Launchers.cuh>
#include <core/Context.h>
#include <core/Autograd.h>


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

        uint32_t M = Input->get_shape()[0];
        uint32_t N = Input->get_shape()[1];
        uint32_t size = M * N;

        if (Input->requires_grad()) {
            launch_relu_backward(out_ptr->get_gradient_ptr(), Input->get_data_ptr(), Input->get_gradient_ptr(),
                size, CudaContext::getStream());
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
        : Input(in), Outputs(outs) {}

    void SplitFunction::apply_backward() {
        auto input_ptr = Input.lock();
        if (!input_ptr) return;

        if (input_ptr->requires_grad()) {
            std::vector<float*> grad_ptrs;
            
            for (auto& out : Outputs) {
                grad_ptrs.push_back(out->get_gradient_ptr());
            }

            uint32_t dim = input_ptr->get_shape().size() - 1;
            uint32_t num_splits = Outputs.size();
            uint32_t inner_size = input_ptr->get_shape()[dim];
            uint32_t split_size = inner_size / num_splits;
            
            uint32_t total_elements = 1;
            for(auto s : input_ptr->get_shape()) total_elements *= s;

            launch_concat_backward(
                grad_ptrs,
                input_ptr->get_gradient_ptr(),
                num_splits,
                inner_size,
                split_size,
                total_elements,
                CudaContext::getStream()
            );

            input_ptr->backward(false);
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

            launch_tensor_add_grad(
                out_ptr->get_gradient_ptr(),
                Input->get_gradient_ptr(),
                size,
                CudaContext::getStream()
            );

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

        float* dQ = Q_input->get_gradient_ptr();
        float* dK = K_input->get_gradient_ptr();
        float* dV = V_input->get_gradient_ptr();

        bool alloc_dQ = false, alloc_dK = false, alloc_dV = false;

        size_t bytes = (size_t)N * H * L * D * sizeof(float);

        if (Q_input->requires_grad() && !dQ) {
            cudaMallocAsync(&dQ, bytes, stream);
            alloc_dQ = true;
        }
        if (K_input->requires_grad() && !dK) {
            cudaMallocAsync(&dK, bytes, stream);
            alloc_dK = true;
        }
        if (V_input->requires_grad() && !dV) {
            cudaMallocAsync(&dV, bytes, stream);
            alloc_dV = true;
        }

        // call backward launcher
        launch_flash_backward(
            Q_input->get_data_ptr(), K_input->get_data_ptr(), V_input->get_data_ptr(),
            out_ptr->get_data_ptr(), out_ptr->get_gradient_ptr(), L_cache->get_data_ptr(),
            dQ, dK, dV,
            N, H, L, E,
            stream
        );

        // propagate gradients
        if (Q_input->requires_grad()) {
            if (alloc_dQ) {
                // copy temporary dQ into tensor gradient
                cudaMemcpyAsync(Q_input->get_gradient_ptr(), dQ, bytes, cudaMemcpyDeviceToDevice, stream);
                cudaFreeAsync(dQ, stream);
            }
            Q_input->backward(false);
        }

        if (K_input->requires_grad()) {
            if (alloc_dK) {
                cudaMemcpyAsync(K_input->get_gradient_ptr(), dK, bytes, cudaMemcpyDeviceToDevice, stream);
                cudaFreeAsync(dK, stream);
            }
            K_input->backward(false);
        }

        if (V_input->requires_grad()) {
            if (alloc_dV) {
                cudaMemcpyAsync(V_input->get_gradient_ptr(), dV, bytes, cudaMemcpyDeviceToDevice, stream);
                cudaFreeAsync(dV, stream);
            }
            V_input->backward(false);
        }
    }

}
