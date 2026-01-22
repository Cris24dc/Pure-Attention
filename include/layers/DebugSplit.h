#pragma once

#include <layers/Module.h>
#include <core/Tensor.h>
#include <vector>
#include <memory>

namespace layers {
    class DebugSplit : public Module {
    public:
        DebugSplit(uint32_t num_parts, int dim);
        
        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &input) override;

        std::vector<std::shared_ptr<core::Tensor>> forward_impl(const std::shared_ptr<core::Tensor> &input);

        using Module::parameters;

    private:
        uint32_t num_parts;
        int dim;
    };
}
