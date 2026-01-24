#pragma once

#include <layers/Module.h>
#include <core/Tensor.h>
#include <vector>
#include <memory>

namespace layers {
    class DebugReshape : public Module {
    public:
        DebugReshape(std::vector<uint32_t> target_shape);
        
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &input) override;

        using Module::parameters;

    private:
        std::vector<uint32_t> target_shape;
    };
}
