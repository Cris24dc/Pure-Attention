// headers
#include <loss/MSE.h>
#include <core/Tensor.h>
#include <core/Functional.h>
#include <core/Context.h>

namespace loss {
    std::shared_ptr<core::Tensor> MSE::forward(const std::shared_ptr<core::Tensor> &prediction, const std::shared_ptr<core::Tensor> &target) {
        const cudaStream_t& stream = CudaContext::getStream();
        return core::mse_loss(prediction, target, stream);
    }
}
