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
            X_input->backward();
        }

        if (W_input->requires_grad()) {
            launch_matmul_grad_W(X_input->get_data_ptr(),grad_out_ptr,W_input->get_gradient_ptr(), M, N, K,
                CudaContext::getStream());
            W_input->backward();
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
            X_input->backward();
        }

        if (bias_input->requires_grad()) {
            bool is_bias = (bias_input->get_shape()[0] == 1 && M > 1);

            if (is_bias)
                launch_sum_rows_grad(grad_out_ptr,bias_input->get_gradient_ptr(), M, N, CudaContext::getStream());
            else
                launch_tensor_add_grad(grad_out_ptr,bias_input->get_gradient_ptr(),size,CudaContext::getStream());

            bias_input->backward();
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
            Input->backward();
        }
    }


    MSEFunction::MSEFunction(const std::shared_ptr<Tensor>& preds,
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

            predictions->backward();
        }
    }


    FlashAttentionFunction::FlashAttentionFunction(
        std::shared_ptr<Tensor> q,
        std::shared_ptr<Tensor> k,
        std::shared_ptr<Tensor> v,
        std::shared_ptr<Tensor> out,
        std::shared_ptr<Tensor> l_vec)
        : Q_input(q), K_input(k), V_input(v), Output(out), L_vec(l_vec) {}

    void FlashAttentionFunction::apply_backward() {
        // 1. Verificăm dacă tensorul de output mai există
        auto out_ptr = Output.lock();
        if (!out_ptr) return;

        // 2. Extragem dimensiunile
        // Shape: [N, L, E]
        const uint32_t N = Q_input->get_shape()[0];
        const uint32_t L = Q_input->get_shape()[1];
        const uint32_t E = Q_input->get_shape()[2];
        const uint32_t H = 8; // Hardcoded conform contextului (sau salvat in membru)

        // 3. Obținem gradientul care vine de sus (dO)
        float* dO = out_ptr->get_gradient_ptr();

        // 4. Pregătim pointerii pentru gradienții input-urilor (dQ, dK, dV)
        // Dacă un tensor nu necesită gradient, get_gradient_ptr() ar putea fi null/nealocat.
        // Kernelul fused calculează totul simultan, deci ideal ar fi să avem bufferi validi.
        // Aici presupunem că dacă requires_grad e true, pointerul e valid.

        float* dQ = Q_input->requires_grad() ? Q_input->get_gradient_ptr() : nullptr;
        float* dK = K_input->requires_grad() ? K_input->get_gradient_ptr() : nullptr;
        float* dV = V_input->requires_grad() ? V_input->get_gradient_ptr() : nullptr;

        // Lansăm kernelul doar dacă avem cel puțin un gradient de calculat
        if (dQ || dK || dV) {
            // NOTĂ: Asigură-te că funcția launch_flash_backward_optimized este declarată în Launchers.h
            // și acceptă parametrii raw pointers, așa cum am definit-o anterior.

            // Pentru siguranță (dacă kernelul nu are null checks), într-un engine robust
            // am aloca un buffer temporar "dummy" pentru pointerii null, dar aici îi pasăm direct.

            launch_flash_backward_optimized(
                Q_input->get_data_ptr(),
                K_input->get_data_ptr(),
                V_input->get_data_ptr(),
                out_ptr->get_data_ptr(), // O (Output-ul din forward)
                dO,                      // dO (Gradientul output-ului)
                L_vec->get_data_ptr(),   // L (Statistici LogSumExp)
                dQ,                      // dQ (Output gradient)
                dK,                      // dK (Output gradient)
                dV,                      // dV (Output gradient)
                N, H, L, E,
                CudaContext::getStream()
            );
        }

        // 5. Propagăm semnalul de backward în graf
        // (Apelăm recursiv backward pe părinți)
        if (Q_input->requires_grad()) Q_input->backward();
        if (K_input->requires_grad()) K_input->backward();
        if (V_input->requires_grad()) V_input->backward();

        // L_vec nu necesită backward deoarece este un buffer intermediar de statistici, nu un parametru.
    };
}




