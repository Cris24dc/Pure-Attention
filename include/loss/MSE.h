#pragma once

// headers
#include <loss/Loss.h>

// libs
#include <stdexcept>

namespace loss {
    class MSE : public Loss {
    public:
        MSE() = default;

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &prediction, const std::shared_ptr<core::Tensor> &target) ;
    };
}
