// headers
#include <layers/Linear.h>
#include <core/Tensor.h>
#include <core/Functional.h>

namespace layers {
    Linear::Linear(const uint32_t in_channels, const uint32_t out_channels) {
        // x: batch x in_channels
        // W: in_channels x out_channels
        // b: 1 x out_channels
        // y = x @ W + b -> y: batch x out_channels    
        weight = std::make_shared<core::Tensor>(std::vector<uint32_t>{in_channels,out_channels},true);
        bias = std::make_shared<core::Tensor>(std::vector<uint32_t>{1,out_channels},true);

        pop_data_normal(weight);
        pop_data_zeros(bias);
    }

    std::shared_ptr<core::Tensor> Linear::forward(const std::shared_ptr<core::Tensor> &input) {
        return matadd(matmul(input, weight), bias);
    }

    std::vector<std::shared_ptr<core::Tensor>> Linear::parameters() {
        return {weight, bias};
    }
}
