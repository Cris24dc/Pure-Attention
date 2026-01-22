#pragma once

#include <layers/Module.h>
#include <core/Tensor.h>
#include <memory>
#include <vector>

namespace layers {
    class LayerNorm : public Module {
    private:
        std::shared_ptr<core::Tensor> gamma;
        std::shared_ptr<core::Tensor> beta;
        uint32_t normalized_shape;
        float epsilon;
        
    public:
        LayerNorm(uint32_t normalized_shape, float epsilon = 1e-5f);
        
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &input) override;
        std::vector<std::shared_ptr<core::Tensor>> parameters() override;
        
        std::shared_ptr<core::Tensor> get_gamma() const { return gamma; }
        std::shared_ptr<core::Tensor> get_beta() const { return beta; }
    };
}