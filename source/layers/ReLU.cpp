// headers
#include <layers/ReLU.h>
#include <core/Context.h>

namespace layers {
    std::shared_ptr<core::Tensor> ReLU::forward(const std::shared_ptr<core::Tensor> &In) {
        const cudaStream_t& stream = CudaContext::getStream();
        return relu(In,stream);
    };

    std::vector<std::shared_ptr<core::Tensor>> ReLU::parameters() {
        return {};
    }
};
