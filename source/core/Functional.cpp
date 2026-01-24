// headers
#include <core/Tensor.h>
#include <backend/Launchers.cuh>
#include <core/Context.h>
#include <core/Autograd.h>

// libs
#include <memory>
#include <stdexcept>

namespace core {
    void matmul(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B,
        std::shared_ptr<Tensor>& C, const cudaStream_t& stream = CudaContext::getStream()) {

        const std::vector<uint32_t>& a_shape = A->get_shape();
        const std::vector<uint32_t>& b_shape = B->get_shape();

        if (b_shape.size() != 2) {
            throw std::runtime_error("Matmul: B (weights) must be 2D");
        }

        uint32_t K = a_shape.back();
        if (K != b_shape[0]) {
            throw std::runtime_error("Matmul: Dimension mismatch A[..., K] and B[K, N]");
        }
        uint32_t N = b_shape[1];

        uint32_t M = 1;
        for (size_t i = 0; i < a_shape.size() - 1; ++i) {
            M *= a_shape[i];
        }
        
        std::vector<uint32_t> c_shape = a_shape;
        c_shape.back() = N;

        bool needs_grad = A->requires_grad() || B->requires_grad();

        C = std::make_shared<Tensor>(c_shape, needs_grad, false);

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
        
        const std::vector<uint32_t>& a_shape = A->get_shape();
        const std::vector<uint32_t>& x_shape = X->get_shape();

        uint32_t N = a_shape.back();
        uint32_t M = 1;
        for (size_t i = 0; i < a_shape.size() - 1; ++i) {
            M *= a_shape[i];
        }

        bool needs_grad = A->requires_grad() || X->requires_grad();

        B = std::make_shared<Tensor>(a_shape, needs_grad, false);

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
        
        const std::vector<uint32_t>& shape = In->get_shape();
        
        uint32_t N = shape.back();
        uint32_t M = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            M *= shape[i];
        }

        bool needs_grad = In->requires_grad();

        auto Out = std::make_shared<Tensor>(shape, needs_grad, false);

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

    void split(const std::shared_ptr<Tensor>& A, uint32_t num_parts,
        int dim, std::vector<std::shared_ptr<Tensor>>& parts,
        const cudaStream_t& stream = CudaContext::getStream()) 
    {
        std::vector<uint32_t> shape = A->get_shape();

        uint32_t num_elements = 1;
        for (auto s : shape) num_elements *= s;

        
        while (dim < 0) dim += shape.size();

        if (dim != shape.size() - 1) {
            throw std::runtime_error("Only split on last dimension is currently supported!");
        }
        
        if (shape[dim] % num_parts){
            throw std::runtime_error("Split dimension must be divisible by num_parts");
        }else{
            uint32_t part_size = shape[dim] / num_parts;
            shape[dim] = part_size;
            
            for (uint32_t i = 0; i < num_parts; i+=1){
                parts.push_back(std::make_shared<Tensor>(shape, A->requires_grad(), false));
            }
        }

    std::vector<float*> out_ptrs;
    for(auto& t : parts) out_ptrs.push_back(t->get_data_ptr());

    launch_split_forward(
        A->get_data_ptr(),
        out_ptrs,
        num_parts,
        shape[dim] * num_parts,
        shape[dim],
        num_elements,
        stream
    );

    if (A->requires_grad()) {
        auto node = std::make_shared<SplitFunction>(A, parts);
        for (auto& part : parts) {
            part->set_grad_fn(node);
        }
    }

    }

    void reshape(const std::shared_ptr<Tensor>& input, const std::vector<uint32_t>& new_shape,
        std::shared_ptr<Tensor>& output, const cudaStream_t& stream = CudaContext::getStream()) {
        
        uint32_t in_elements = 1;
        for (auto s : input->get_shape()) in_elements *= s;
        
        uint32_t out_elements = 1;
        for (auto s : new_shape) out_elements *= s;

        if (in_elements != out_elements) {
            throw std::runtime_error("Reshape element count mismatch");
        }

        bool needs_grad = input->requires_grad();
        output = std::make_shared<Tensor>(new_shape, needs_grad, false);

        cudaMemcpyAsync(output->get_data_ptr(), input->get_data_ptr(), 
            in_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        if (needs_grad) {
            auto node = std::make_shared<ReshapeFunction>(input, output);
            output->set_grad_fn(node);
         }
    }
  
    void flash_attention(const std::shared_ptr<Tensor>& Q, const std::shared_ptr<Tensor>& K,
        const std::shared_ptr<Tensor>& V, std::shared_ptr<Tensor>& O,
        const cudaStream_t& stream = CudaContext::getStream()) {
        
        const int N = Q->get_shape()[0];
        const int L = Q->get_shape()[1];
        
        const int H = 8; 
        const int B_r = 16;
        uint32_t Tr = (L + B_r - 1) / B_r;

        bool needs_grad = Q->requires_grad() || K->requires_grad() || V->requires_grad();

        O = std::make_shared<Tensor>(Q->get_shape(), needs_grad, false);
        
        auto L_cache = std::make_shared<Tensor>(std::vector<uint32_t>{(uint32_t)N, (uint32_t)H, Tr}, false, false);

        launch_flash_attention(
            Q, K, V, O,
            L_cache,
            stream
        );

        if (needs_grad) {
            auto node = std::make_shared<FlashAttentionFunction>(Q, K, V, O, L_cache);
            O->set_grad_fn(node);
        }
    }




    std::shared_ptr<Tensor> layer_norm(
        const std::shared_ptr<Tensor>& input,
        const std::shared_ptr<Tensor>& gamma,
        const std::shared_ptr<Tensor>& beta,
        float epsilon,
        const cudaStream_t& stream) 
    {
    auto shape = input->get_shape();
    uint32_t M = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        M *= shape[i];
    }
    uint32_t N = shape[shape.size() - 1];

    auto output = std::make_shared<Tensor>(shape, input->requires_grad() || gamma->requires_grad() || beta->requires_grad());
    auto mean = std::make_shared<Tensor>(std::vector<uint32_t>{M}, false);
    auto rstd = std::make_shared<Tensor>(std::vector<uint32_t>{M}, false);

    launch_layer_norm_forward(
        input->get_data_ptr(),
        gamma->get_data_ptr(),
        beta->get_data_ptr(),
        output->get_data_ptr(),
        mean->get_data_ptr(),
        rstd->get_data_ptr(),
        M, N, epsilon,
        stream
    );

    if (output->requires_grad()) {
        if (gamma->requires_grad()) {
            cudaMemsetAsync(gamma->get_gradient_ptr(), 0, N * sizeof(float), stream);
        }
        if (beta->requires_grad()) {
            cudaMemsetAsync(beta->get_gradient_ptr(), 0, N * sizeof(float), stream);
        }
        
        auto grad_fn = std::make_shared<LayerNormFunction>(
            input, gamma, beta, output, mean, rstd, M, N
        );
        output->set_grad_fn(grad_fn);
    }

    return output;
}
};
