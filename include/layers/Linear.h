#pragma once 

#include "Module.h"
#include <memory>
#include <vector>


namespace layers {

    class Linear : public Module{
    private:
        std::shared_ptr<core::Tensor> weight;
        std::shared_ptr<core::Tensor> bias;

    public:
        Linear(uint32_t in_channels, uint32_t out_channels);

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor>& input) override;
        std::vector<std::shared_ptr<core::Tensor>> parameters() override;
    };
};

