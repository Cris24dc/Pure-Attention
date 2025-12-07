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

    void MultiheadAttention::query_flow(const uint32_t thread_id, const std::shared_ptr<core::Tensor> &Q,
        std::shared_ptr<core::Tensor> &Q_proj, const uint32_t& N, const uint32_t &L,
        const uint32_t &E, const uint32_t&H) const
    {
        // Q: N x L x E
        // Q_weight: E x E
        // Q_proj: N x L x E
        cudaStream_t stream;
        int priority_high, priority_low;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority_high);

        pop_data_normal(Q_weight,stream);
        pop_data_zeros(Q_bias,stream);

        cudaStreamSynchronize(stream);

        std::shared_ptr<core::Tensor> XW_q;

        matmul(Q,Q_weight,XW_q);
        matadd(XW_q,Q_bias,Q_proj);

        // Q_Proj: N x L x E -> N x L x H x Dk          spilt

        // Q_Proj: N x L x H x Dk -> N x H x L x Dk     transpose

        // cudaFreeAsync(data,stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    void MultiheadAttention::key_flow(uint32_t thread_id, const std::shared_ptr<core::Tensor> &K,
        std::shared_ptr<core::Tensor> &K_proj, const uint32_t& N, const uint32_t &L,
        const uint32_t &E, const uint32_t&H) const {
        // K: N x L x E
        // K_weight: E x E
        // K_proj: N x L x E
        cudaStream_t stream;
        int priority_high, priority_low;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority_high);


        pop_data_normal(K_weight,stream);
        pop_data_zeros(K_bias,stream);

        cudaStreamSynchronize(stream);

        std::shared_ptr<core::Tensor> XW_k;

        matmul(K,K_weight,XW_k);
        matadd(XW_k,K_bias,K_proj);

        // K_Proj: N x L x E -> N x L x H x Dk          spilt

        // K_Proj: N x L x H x Dk -> N x H x L x Dk     transpose

        // cudaFreeAsync(data,stream);
        cudaStreamDestroy(stream);
    }

    void MultiheadAttention::value_flow(uint32_t thread_id, const std::shared_ptr<core::Tensor> &V,
        std::shared_ptr<core::Tensor> &V_proj, const uint32_t& N, const uint32_t &L,
        const uint32_t &E, const uint32_t&H) const {
        // V: N x L x E
        // V_weight: E x E
        // V_proj: N x L x E
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        pop_data_normal(V_weight,stream);
        pop_data_zeros(V_bias,stream);

        cudaStreamSynchronize(stream);

        std::shared_ptr<core::Tensor> XW_v;

        matmul(V,K_weight,XW_v);
        matadd(XW_v,K_bias,V_proj);

        // V_Proj: N x L x E -> N x L x H x Dk          spilt

        // V_Proj: N x L x H x Dk -> N x H x L x Dk     transpose

        // cudaFreeAsync(data,stream);
        cudaStreamDestroy(stream);
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

        core::Tensor Q_proj({N,L,E},true,false);
        core::Tensor K_proj({N,L,E},true,false);
        core::Tensor V_proj({N,L,E},true,false);

        std::vector<std::jthread> threads;

        threads.emplace_back(&MultiheadAttention::query_flow,this,0,query,std::ref(Q_proj),N,L,E,H);
        threads.emplace_back(&MultiheadAttention::key_flow,this,1,key,std::ref(K_proj),N,L,E,H);
        threads.emplace_back(&MultiheadAttention::value_flow,this,2,value,std::ref(V_proj),N,L,E,H);

        threads[0].join();
        threads[1].join();




        return nullptr;
    }

    std::vector<std::shared_ptr<core::Tensor>> MultiheadAttention::parameters() {
        std::vector<std::shared_ptr<core::Tensor>> result;

        return result;
    }
}
