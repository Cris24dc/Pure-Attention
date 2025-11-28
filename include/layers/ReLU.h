#pragma once

#include "Module.h"
#include "Functional.h"

namespace core {
    class Tensor;
};

namespace layers {
    class ReLU : public Module {
    public:
        ReLU() = default;

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& In) override {
            return relu(In);
        };

        std::vector<std::shared_ptr<core::Tensor>> parameters() override {
            return {};
        }
    };
};