//headers
#include <layers/MultiheadAttention.h>
#include <core/Tensor.h>
#include <core/Functional.h>

namespace layers {
    MultiheadAttention::MultiheadAttention(const uint32_t model_dimension, const uint32_t heads_number):
    model_dimension(model_dimension), heads_number(heads_number) {}

    std::shared_ptr<core::Tensor> MultiheadAttention::forward(const std::shared_ptr<core::Tensor> &input) {
        throw std::runtime_error("MultiheadAttention cannot be called with 1 argument. Use (query, key, value, mask).");
    }

    std::shared_ptr<core::Tensor> forward (
        const std::shared_ptr<core::Tensor> &query,
        const std::shared_ptr<core::Tensor> &key,
        const std::shared_ptr<core::Tensor> &value,
        const std::shared_ptr<core::Tensor> &mask)
    {
        return query;
    }

    std::vector<std::shared_ptr<core::Tensor>> MultiheadAttention::parameters() {
        std::vector<std::shared_ptr<core::Tensor>> result;

        return result;
    }
}
