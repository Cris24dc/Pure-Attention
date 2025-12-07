// headers
#include <layers/Linear.h>
#include <core/Tensor.h>
#include <core/Functional.h>
#include <core/Context.h>

namespace layers {
    Linear::Linear(const uint32_t in_channels, const uint32_t out_channels) {
        // x: batch x in_channels
        // W: in_channels x out_channels
        // b: 1 x out_channels
        // y = x @ W + b -> y: batch x out_channels    
        weight = std::make_shared<core::Tensor>(std::vector<uint32_t>{in_channels,out_channels},true);
        bias = std::make_shared<core::Tensor>(std::vector<uint32_t>{1,out_channels},true);

        const cudaStream_t& stream = CudaContext::getStream();
        pop_data_normal(weight,stream);
        pop_data_zeros(bias,stream);
    }

    std::shared_ptr<core::Tensor> Linear::forward(const std::shared_ptr<core::Tensor> &input) {
        std::shared_ptr<core::Tensor> XW, Y;
        const cudaStream_t& stream = CudaContext::getStream();

        matmul(input,weight,XW,stream);
        matadd(XW, bias, Y,stream);
        return Y;
    }

    std::vector<std::shared_ptr<core::Tensor>> Linear::parameters() {
        return {weight, bias};
    }
}
