#include <core/Tensor.h>
#include <backend/Launchers.h>
#include <core/Context.h>
#include <core/OpsNodes.h>


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


    AddFunction::AddFunction(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> &c)
            : Input_A(a), Input_B(b), Output_C(c) {}

    void AddFunction::apply_backward() {
        auto out_ptr = Output_C.lock();
        if (!out_ptr) return;

        auto grad_out_ptr = out_ptr->get_gradient_ptr();
        uint32_t M = out_ptr->get_shape()[0];
        uint32_t N = out_ptr->get_shape()[1];
        uint32_t size = M * N;

        if (Input_A->requires_grad()) {
            launch_tensor_add_grad(grad_out_ptr,Input_A->get_gradient_ptr(),size, CudaContext::getStream());
            Input_A->backward();
        }

        if (Input_B->requires_grad()) {
            bool is_bias = (Input_B->get_shape()[0] == 1 && M > 1);

            if (is_bias)
                launch_sum_rows_grad(grad_out_ptr,Input_B->get_gradient_ptr(), M, N, CudaContext::getStream());
            else
                launch_tensor_add_grad(grad_out_ptr,Input_B->get_gradient_ptr(),size,CudaContext::getStream());

            Input_B->backward();
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


};



