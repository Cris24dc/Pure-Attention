// headers
#include <core/Tensor.h>
#include <backend/Launchers.cuh>
#include <core/Context.h>
#include <core/Autograd.h>

// libs
#include <memory>

namespace core {
    void matmul(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B,
        std::shared_ptr<Tensor>& C, const cudaStream_t& stream = CudaContext::getStream()) {
        const uint32_t M = A->get_shape()[0];
        const uint32_t N = A->get_shape()[1];
        const uint32_t K = B->get_shape()[1];

        bool needs_grad = A->requires_grad() || B->requires_grad();

        C = std::make_shared<Tensor>(std::vector<uint32_t>{M, K}, needs_grad, false);

        launch_matmul_tiled(
            A->get_data_ptr(), 
            B->get_data_ptr(), 
            C->get_data_ptr(), 
            M, N, K,
            stream
        );

        if (needs_grad) {
            auto node = std::make_shared<MatMulFunction>(A, B, C);
            C->set_grad_fn(node);
        }

    }
    
    void matadd(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& X,
        std::shared_ptr<Tensor>& B,const cudaStream_t& stream = CudaContext::getStream()) {
        uint32_t M = A->get_shape()[0];
        uint32_t N = A->get_shape()[1];

        bool needs_grad = A->requires_grad() || X->requires_grad();

        B = std::make_shared<Tensor>(std::vector<uint32_t>{M, N}, needs_grad, false);

        launch_matadd_tiled(
            A->get_data_ptr(), 
            X->get_data_ptr(), 
            B->get_data_ptr(), 
            M, N,
            stream
        );

        if (needs_grad) {
            auto node = std::make_shared<AddFunction>(A, X, B);
            B->set_grad_fn(node);
        }
    }

    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& In,
        const cudaStream_t& stream = CudaContext::getStream()) {
        uint32_t M = In->get_shape()[0];
        uint32_t N = In->get_shape()[1];

        bool needs_grad = In->requires_grad();

        auto Out = std::make_shared<Tensor>(std::vector<uint32_t>{M, N},needs_grad,false);

        launch_ReLU_tiled(
            In->get_data_ptr(), 
            Out->get_data_ptr(),
            M, N,
            stream
        );

        if (needs_grad) {
            auto node = std::make_shared<ReLUFunction>(In, Out);
            Out->set_grad_fn(node);
        }

        return Out;
    }

    void pop_data_zeros(const std::shared_ptr<Tensor>& A,
        const cudaStream_t& stream = CudaContext::getStream()){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        if (N == 0)
            N = 1;

        launch_zero_population(A->get_data_ptr(), M, N, stream);
    }

    void pop_grad_zeros(const std::shared_ptr<Tensor>& A, const cudaStream_t& stream = CudaContext::getStream()){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        if (N == 0)
            N = 1;

        launch_zero_population(A->get_gradient_ptr(), M, N, stream);
    }

    void pop_grad_zeros(Tensor *A, const cudaStream_t& stream = CudaContext::getStream()){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        if (N == 0)
            N = 1;


        launch_zero_population(A->get_gradient_ptr(), M, N, stream);
    }

    void pop_grad_ones(const std::shared_ptr<Tensor>& A,
        const cudaStream_t& stream = CudaContext::getStream()){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        if (N == 0)
            N = 1;

        launch_ones_population(A->get_gradient_ptr(), M, N, stream);
    }

    void pop_grad_ones(Tensor *A,
        const cudaStream_t& stream = CudaContext::getStream()){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        if (N == 0)
            N = 1;

        launch_ones_population(A->get_gradient_ptr(), M, N, stream);
    }

    void pop_data_normal(const std::shared_ptr<Tensor>& A, float std_dev,
        const cudaStream_t& stream = CudaContext::getStream()){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        if (N == 0)
            N = 1;

        launch_normal_population(A->get_data_ptr(), M, N, std_dev, stream);
    }


    std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& preds, const std::shared_ptr<Tensor>& targets,
        const cudaStream_t& stream = CudaContext::getStream()) {
        uint32_t N = 1;
        for (auto s : preds->get_shape()) N *= s;

        bool needs_grad = preds->requires_grad();

        auto loss = std::make_shared<Tensor>(std::vector<uint32_t>{1}, needs_grad, false);

        launch_mse_forward(
            preds->get_data_ptr(),
            targets->get_data_ptr(),
            loss->get_data_ptr(),
            N,
            stream
        );

        if (needs_grad) {
            auto node = std::make_shared<MSEFunction>(preds, targets, loss);
            loss->set_grad_fn(node);
        }

        return loss;
    }
};
