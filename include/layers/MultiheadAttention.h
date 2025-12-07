#pragma once

#include <layers/Module.h>
#include <core/Tensor.h>

#include <memory>
#include <vector>

namespace layers {
    class MultiheadAttention : public Module {
    private:
        uint32_t model_dimension;
        uint32_t heads_number;

        std::shared_ptr<core::Tensor> Q_weight;
        std::shared_ptr<core::Tensor> Q_bias;
        std::shared_ptr<core::Tensor> K_weight;
        std::shared_ptr<core::Tensor> K_bias;
        std::shared_ptr<core::Tensor> V_weight;
        std::shared_ptr<core::Tensor> V_bias;

        // void query_flow(const uint32_t thread_id, const std::shared_ptr<core::Tensor> &Q,
        //     std::shared_ptr<core::Tensor> &Q_proj, const uint32_t& N, const uint32_t &L,
        //     const uint32_t &E, const uint32_t&H) const;
        // void key_flow(uint32_t thread_id, const std::shared_ptr<core::Tensor> &K,
        //     std::shared_ptr<core::Tensor> &K_proj, const uint32_t& N, const uint32_t &L,
        //     const uint32_t &E, const uint32_t&H) const;
        // void value_flow(uint32_t thread_id, const std::shared_ptr<core::Tensor> &V,
        //     std::shared_ptr<core::Tensor> &V_proj, const uint32_t& N, const uint32_t &L,
        //     const uint32_t &E, const uint32_t&H) const;


    public:
        MultiheadAttention(const uint32_t model_dimension, const uint32_t heads_number);

        std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &input) override;
        std::shared_ptr<core::Tensor> forward (
            const std::shared_ptr<core::Tensor> &query,
            const std::shared_ptr<core::Tensor> &key,
            const std::shared_ptr<core::Tensor> &value,
            const std::shared_ptr<core::Tensor> &mask);

        std::vector<std::shared_ptr<core::Tensor>> parameters() override;
    };
}