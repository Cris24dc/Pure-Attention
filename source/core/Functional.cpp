// headers
#include <core/Tensor.h>
#include <backend/Launchers.h>
#include <core/Context.h>
#include <core/OpsNodes.h>

// libs
#include <memory>

namespace core {
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B) {
        const uint32_t M = A->get_shape()[0];
        const uint32_t N = A->get_shape()[1];
        const uint32_t K = B->get_shape()[1];

        bool needs_grad = A->requires_grad() || B->requires_grad();

        auto C = std::make_shared<Tensor>(std::vector<uint32_t>{M, K}, needs_grad, false);

        launch_matmul_tiled(
            A->get_data_ptr(), 
            B->get_data_ptr(), 
            C->get_data_ptr(), 
            M, N, K,
            CudaContext::getStream()
        );

        if (needs_grad) {
            auto node = std::make_shared<MatMulFunction>(A, B, C);
            C->set_grad_fn(node);
        }

        return C;
    }
    
    std::shared_ptr<Tensor> matadd(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& X) {
        uint32_t M = A->get_shape()[0];
        uint32_t N = A->get_shape()[1];

        bool needs_grad = A->requires_grad() || X->requires_grad();

        auto B = std::make_shared<Tensor>(std::vector<uint32_t>{M, N}, needs_grad, false);

        launch_matadd_tiled(
            A->get_data_ptr(), 
            X->get_data_ptr(), 
            B->get_data_ptr(), 
            M, N,
            CudaContext::getStream()
        );

        if (needs_grad) {
            auto node = std::make_shared<AddFunction>(A, X, B);
            B->set_grad_fn(node);
        }

        return B;
    }

    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& In) {
        uint32_t M = In->get_shape()[0];
        uint32_t N = In->get_shape()[1];

        bool needs_grad = In->requires_grad();

        auto Out = std::make_shared<Tensor>(std::vector<uint32_t>{M, N},needs_grad,false);

        launch_ReLU_tiled(
            In->get_data_ptr(), 
            Out->get_data_ptr(),
            M, N,
            CudaContext::getStream()
        );

        if (needs_grad) {
            auto node = std::make_shared<ReLUFunction>(In, Out);
            Out->set_grad_fn(node);
        }

        return Out;
    }

    void pop_data_zeros(const std::shared_ptr<Tensor>& A){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        launch_zero_population(A->get_data_ptr(), M, N, CudaContext::getStream());
    }

    void pop_grad_zeros(const std::shared_ptr<Tensor>& A){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        launch_zero_population(A->get_gradient_ptr(), M, N, CudaContext::getStream());
    }

    void pop_grad_zeros(Tensor *A){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        launch_zero_population(A->get_gradient_ptr(), M, N, CudaContext::getStream());
    }

    void pop_grad_ones(const std::shared_ptr<Tensor>& A){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        launch_ones_population(A->get_gradient_ptr(), M, N, CudaContext::getStream());
    }

    void pop_grad_ones(Tensor *A){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        launch_ones_population(A->get_gradient_ptr(), M, N, CudaContext::getStream());
    }

    void pop_data_normal(const std::shared_ptr<Tensor>& A){
        int M = A->get_shape()[0];
        int N = A->get_shape()[1];

        launch_normal_population(A->get_data_ptr(), M, N, CudaContext::getStream());
    }
};
