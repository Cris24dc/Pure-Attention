#include <layers/MHA.h>
#include <core/Tensor.h>
#include <core/Functional.h>
#include <core/Context.h>
#include <cmath>
#include <vector>

namespace layers {

    MultiheadAttention::MultiheadAttention(uint32_t embed_dim, uint32_t num_heads) :
    embed_dim(embed_dim),
    num_heads(num_heads) 
    {
        uint32_t out_channels = 3 * embed_dim;
        weight = std::make_shared<core::Tensor>(std::vector<uint32_t>{embed_dim, out_channels}, true);
        bias = std::make_shared<core::Tensor>(std::vector<uint32_t>{1, out_channels}, true);

        const cudaStream_t& stream = CudaContext::getStream();
        float std_dev = std::sqrt(2.0f / static_cast<float>(embed_dim));
        
        pop_data_normal(weight, std_dev, stream);
        pop_data_zeros(bias, stream);
    }

    std::shared_ptr<core::Tensor> MultiheadAttention::forward(const std::shared_ptr<core::Tensor> &input) {
        std::shared_ptr<core::Tensor> XW, proj_output;
        const cudaStream_t& stream = CudaContext::getStream();
        uint32_t head_dim = embed_dim / num_heads;

        matmul(input, weight, XW, stream);
        matadd(XW, bias, proj_output, stream);
        
        std::vector<std::shared_ptr<core::Tensor>> qkv;
        split(proj_output, 3, -1, qkv, stream);

        std::shared_ptr<core::Tensor> attn_output;
        flash_attention(qkv[0], qkv[1], qkv[2], attn_output, num_heads, stream);

        std::shared_ptr<core::Tensor> final_output;
        // reshape(attn_output, input_shape, final_output, stream);
        final_output = attn_output;

        return final_output;
    }

    std::shared_ptr<core::Tensor> MultiheadAttention::forward(const std::shared_ptr<core::Tensor> &query, const std::shared_ptr<core::Tensor> &key, const std::shared_ptr<core::Tensor> &value) {
        const cudaStream_t& stream = CudaContext::getStream();
        uint32_t head_dim = embed_dim / num_heads;
        
        std::vector<std::shared_ptr<core::Tensor>> w_parts, b_parts;
        split(weight, 3, 1, w_parts, stream);
        split(bias, 3, 1, b_parts, stream);

        std::shared_ptr<core::Tensor> tmp_q, tmp_k, tmp_v;
        std::shared_ptr<core::Tensor> Q_proj, K_proj, V_proj;

        matmul(query, w_parts[0], tmp_q, stream);
        matadd(tmp_q, b_parts[0], Q_proj, stream);

        matmul(key, w_parts[1], tmp_k, stream);
        matadd(tmp_k, b_parts[1], K_proj, stream);

        matmul(value, w_parts[2], tmp_v, stream);
        matadd(tmp_v, b_parts[2], V_proj, stream);

        std::shared_ptr<core::Tensor> attn_output;
        flash_attention(Q_proj, K_proj, V_proj, attn_output, num_heads, stream);

        std::shared_ptr<core::Tensor> final_output;
        // reshape(attn_output, q_shape, final_output, stream);
        final_output = attn_output;

        return final_output;
    }

    std::vector<std::shared_ptr<core::Tensor>> MultiheadAttention::parameters() {
        return {weight, bias};
    }
}