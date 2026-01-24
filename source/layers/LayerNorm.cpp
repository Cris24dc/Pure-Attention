#include <layers/LayerNorm.h>
#include <core/Functional.h>
#include <core/Context.h>
#include <vector>

namespace layers {
    LayerNorm::LayerNorm(uint32_t normalized_shape, float epsilon) 
        : normalized_shape(normalized_shape), epsilon(epsilon) {
        
        gamma = std::make_shared<core::Tensor>(
            std::vector<uint32_t>{1, normalized_shape}, true);
        beta = std::make_shared<core::Tensor>(
            std::vector<uint32_t>{1, normalized_shape}, true);
        
        std::vector<float> gamma_init(normalized_shape, 1.0f);
        std::vector<float> beta_init(normalized_shape, 0.0f);
        
        gamma->to_device(gamma_init);
        beta->to_device(beta_init);
    }
    
    std::shared_ptr<core::Tensor> LayerNorm::forward(const std::shared_ptr<core::Tensor> &input) {
        const cudaStream_t& stream = CudaContext::getStream();
        return core::layer_norm(input, gamma, beta, epsilon, stream);
    }
    
    std::vector<std::shared_ptr<core::Tensor>> LayerNorm::parameters() {
        return {gamma, beta};
    }
}