#pragma once

// headers
#include <core/Tensor.h>

// libs
#include <memory>

namespace loss {
    class Loss {
    public:
        virtual ~Loss() = default;

        virtual std::shared_ptr<core::Tensor> forward(
            const std::shared_ptr<core::Tensor> &prediction, 
            const std::shared_ptr<core::Tensor> &target
        ) = 0;
    };
}
