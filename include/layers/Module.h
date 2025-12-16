#pragma once

// headers
#include <core/Tensor.h>

// libs
#include <memory>
#include <vector>

namespace layers {
    class Module {
    public:
        virtual ~Module() = default;

        virtual std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &input) = 0;
        virtual std::vector<std::shared_ptr<core::Tensor>> parameters() {
            return {}; 
        }
    };
};
