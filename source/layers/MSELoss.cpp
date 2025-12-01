#include <layers/MSELoss.h>
#include <core/Tensor.h>
#include <core/Functional.h>

namespace layers {
    std::shared_ptr<core::Tensor> MSELoss::forward(const std::shared_ptr<core::Tensor> &input) {
        throw std::runtime_error("MSELoss cannot be called with 1 argument. Use (prediction, target).");
    }

    std::shared_ptr<core::Tensor> MSELoss::forward(const std::shared_ptr<core::Tensor> &prediction, const std::shared_ptr<core::Tensor> &target) {
        return core::mse_loss(prediction, target);
    }

    std::vector<std::shared_ptr<core::Tensor>> MSELoss::parameters()  {
        return {};
    }

}