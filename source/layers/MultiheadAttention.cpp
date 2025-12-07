//headers
#include <layers/MultiheadAttention.h>
#include <core/Tensor.h>
#include <core/Functional.h>

//libs
#include <thread>

namespace layers {
    MultiheadAttention::MultiheadAttention(const uint32_t model_dimension, const uint32_t heads_number):
    model_dimension(model_dimension), heads_number(heads_number) {}

    std::shared_ptr<core::Tensor> MultiheadAttention::forward(const std::shared_ptr<core::Tensor> &input) {
        throw std::runtime_error("MultiheadAttention cannot be called with 1 argument. Use (query, key, value, mask).");
    }



    std::shared_ptr<core::Tensor> MultiheadAttention::forward (
        const std::shared_ptr<core::Tensor> &query,     // N x L x E
        const std::shared_ptr<core::Tensor> &key,       // N x L x E
        const std::shared_ptr<core::Tensor> &value,     // N x L x E
        const std::shared_ptr<core::Tensor> &mask)
    {
        std::vector<uint32_t> q_input_shape = query->get_shape();
        std::vector<uint32_t> k_input_shape = key->get_shape();
        std::vector<uint32_t> v_input_shape = value->get_shape();

        uint32_t N,L,E;
        uint32_t H = heads_number;

        const bool valid_shapes = (q_input_shape.size() == 3) && (k_input_shape.size() == 3) && (v_input_shape.size() == 3);

        if (valid_shapes) {
            const bool valid_N_dim = (q_input_shape[0] == k_input_shape[0]) && (v_input_shape[0] == k_input_shape[0]);
            const bool valid_L_dim = (q_input_shape[1] == k_input_shape[1]) && (v_input_shape[1] == k_input_shape[1]);
            const bool valid_E_dim = (q_input_shape[2] == k_input_shape[2]) && (v_input_shape[2] == k_input_shape[2]);

            if (valid_N_dim && valid_L_dim && valid_E_dim) {
                N = q_input_shape[0], L = q_input_shape[1], E = q_input_shape[2];
            }else{
                throw std::runtime_error("MultiheadAttention shapes are not valid.");
            }
        } else {
            throw std::runtime_error("MultiheadAttention shapes are not valid.");
        }




        return nullptr;
    }

    std::vector<std::shared_ptr<core::Tensor>> MultiheadAttention::parameters() {
        std::vector<std::shared_ptr<core::Tensor>> result;

        return result;
    }
}
