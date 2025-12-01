#pragma once

#include <layers/Module.h>
#include <core/Functional.h>
#include <stdexcept>

namespace layers {
    class MSELoss : public Module {
    public:
        MSELoss() = default;

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &input) override;

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &prediction, const std::shared_ptr<core::Tensor> &target) ;

        std::vector<std::shared_ptr<core::Tensor>> parameters() override ;
    };
}