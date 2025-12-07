#include <layers/MSELoss.h>
#include <core/Tensor.h>
#include <core/Functional.h>
#include <core/Context.h>

namespace layers {
    std::shared_ptr<core::Tensor> MSELoss::forward(const std::shared_ptr<core::Tensor> &input) {
        throw std::runtime_error("MSELoss cannot be called with 1 argument. Use (prediction, target).");
    }

    std::shared_ptr<core::Tensor> MSELoss::forward(const std::shared_ptr<core::Tensor> &prediction, const std::shared_ptr<core::Tensor> &target) {
        const cudaStream_t& stream = CudaContext::getStream();
        return core::mse_loss(prediction, target, stream);
    }

    std::vector<std::shared_ptr<core::Tensor>> MSELoss::parameters()  {
        return {};
    }

}