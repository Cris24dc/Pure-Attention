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
        uint32_t in_proj_out = 3 * embed_dim;
        in_proj_weight = std::make_shared<core::Tensor>(std::vector<uint32_t>{embed_dim, in_proj_out}, true);
        in_proj_bias = std::make_shared<core::Tensor>(std::vector<uint32_t>{1, in_proj_out}, true);

        out_proj_weight = std::make_shared<core::Tensor>(std::vector<uint32_t>{embed_dim, embed_dim}, true);
        out_proj_bias = std::make_shared<core::Tensor>(std::vector<uint32_t>{1, embed_dim}, true);

        const cudaStream_t& stream = CudaContext::getStream();
        float std_dev = std::sqrt(2.0f / static_cast<float>(embed_dim));
        
        pop_data_normal(in_proj_weight, std_dev, stream);
        pop_data_zeros(in_proj_bias, stream);
        pop_data_normal(out_proj_weight, std_dev, stream);
        pop_data_zeros(out_proj_bias, stream);
    }

    std::shared_ptr<core::Tensor> MultiheadAttention::forward(const std::shared_ptr<core::Tensor> &input) {
        const cudaStream_t& stream = CudaContext::getStream();

        std::shared_ptr<core::Tensor> XW, proj_output;
        matmul(input, in_proj_weight, XW, stream);
        matadd(XW, in_proj_bias, proj_output, stream);
        
        std::vector<std::shared_ptr<core::Tensor>> qkv;
        split(proj_output, 3, -1, qkv, stream);

        std::shared_ptr<core::Tensor> attn_output;
        flash_attention(qkv[0], qkv[1], qkv[2], attn_output, num_heads, stream);

        std::shared_ptr<core::Tensor> out_proj, final_output;
        matmul(attn_output, out_proj_weight, out_proj, stream);
        matadd(out_proj, out_proj_bias, final_output, stream);

        return final_output;
    }

    std::shared_ptr<core::Tensor> MultiheadAttention::forward(const std::shared_ptr<core::Tensor> &query, const std::shared_ptr<core::Tensor> &key, const std::shared_ptr<core::Tensor> &value) {
        const cudaStream_t& stream = CudaContext::getStream();
        
        std::vector<std::shared_ptr<core::Tensor>> w_parts, b_parts;
        split(in_proj_weight, 3, 1, w_parts, stream);
        split(in_proj_bias, 3, 1, b_parts, stream);

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

        std::shared_ptr<core::Tensor> out_proj, final_output;
        matmul(attn_output, out_proj_weight, out_proj, stream);
        matadd(out_proj, out_proj_bias, final_output, stream);

        return final_output;
    }

    std::vector<std::shared_ptr<core::Tensor>> MultiheadAttention::parameters() {
        return {in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias};
    }
}