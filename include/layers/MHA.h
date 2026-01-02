// #pragma once

// #include <layers/Module.h>
// #include <core/Tensor.h>


// namespace layers{
//     class MultiheadAttention : public Module {
//     private:
//         std::shared_ptr<core::Tensor> weight;
//         std::shared_ptr<core::Tensor> bias;

//         const uint32_t embed_dim;
//         const uint32_t num_heads;

//     public:
//         MultiheadAttention(uint32_t embed_dim, uint32_t num_heads);

//         std::shared_ptr<core::Tensor> forward(const std::shared_ptr<core::Tensor> &input) override;
//         std::shared_ptr<core::Tensor> MultiheadAttention::forward(const std::shared_ptr<core::Tensor> &query,const std::shared_ptr<core::Tensor> &key,const std::shared_ptr<core::Tensor> &value);
//         std::vector<std::shared_ptr<core::Tensor>> parameters() override;

//     };
// }
