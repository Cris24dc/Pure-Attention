#include <backend/Kernels.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include "core/Tensor.h"

#define MAX_HEAD_DIM 64

__device__ __forceinline__ float4 load_float4(const float* ptr, const uint32_t idx) {
    return reinterpret_cast<const float4*>(ptr)[idx];
}

__device__ __forceinline__ void store_float4(float* ptr, const uint32_t idx, const float4 val) {
    reinterpret_cast<float4*>(ptr)[idx] = val;
}

template <int B_r, int B_c, int D>
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L_cache,
    const int N,
    const int H,
    const int L,
    const float softmax_scale,
    const int stride_batch,
    const int stride_head,
    const int stride_seq
) {
    extern __shared__ float sram[];

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t bx = blockIdx.x;
    const uint32_t by = blockIdx.y;

    const uint32_t q_block_idx = by;
    const uint32_t kv_block_count = (L + B_c - 1) / B_c;

    const uint32_t batch_idx = bx / H;
    const uint32_t head_idx = bx % H;

    const uint32_t q_offset_base = batch_idx * stride_batch + head_idx * stride_head;
    const uint32_t k_offset_base = batch_idx * stride_batch + head_idx * stride_head;
    const uint32_t v_offset_base = batch_idx * stride_batch + head_idx * stride_head;
    const uint32_t o_offset_base = batch_idx * stride_batch + head_idx * stride_head;

    float* sramQ = sram;
    float* sramK = sram + B_r * D;
    float* sramV = sram + B_r * D + B_c * D;

    float acc[D] = {0.0f};
    float l = 0.0f;
    float m = -CUDART_INF_F;

    int q_global_offset = q_offset_base + (q_block_idx * B_r * stride_seq);

    #pragma unroll
    for (uint32_t i = ty; i < B_r; i += blockDim.y) {
        if (q_block_idx * B_r + i < L) {
            #pragma unroll
            for (uint32_t j = tx; j < D / 4; j += blockDim.x) {
                float4 loaded = load_float4(Q + q_global_offset + i * stride_seq, j);
                reinterpret_cast<float4*>(&sramQ[i * D])[j] = loaded;
            }
        } else {
             #pragma unroll
             for (uint32_t j = tx; j < D / 4; j += blockDim.x) {
                reinterpret_cast<float4*>(&sramQ[i * D])[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
             }
        }
    }

    __syncthreads();

    for (int j = 0; j < kv_block_count; j+=1) {
        const int k_global_offset = k_offset_base + (j * B_c * stride_seq);
        const int v_global_offset = v_offset_base + (j * B_c * stride_seq);

        #pragma unroll
        for (uint32_t i = ty; i < B_c; i += blockDim.y) {
            if (j * B_c + i < L) {
                #pragma unroll
                for (uint32_t x = tx; x < D / 4; x += blockDim.x) {
                    const float4 val_k = load_float4(K + k_global_offset + i * stride_seq, x);
                    reinterpret_cast<float4*>(&sramK[i * D])[x] = val_k;

                    const float4 val_v = load_float4(V + v_global_offset + i * stride_seq, x);
                    reinterpret_cast<float4*>(&sramV[i * D])[x] = val_v;
                }
            } else {
                #pragma unroll
                for (uint32_t x = tx; x < D / 4; x += blockDim.x) {
                    reinterpret_cast<float4*>(&sramK[i * D])[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    reinterpret_cast<float4*>(&sramV[i * D])[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        uint32_t row = ty;
        if (row < B_r && (q_block_idx * B_r + row) < L) {
            const float32_t row_max_prev = m;
            const float32_t row_l_prev = l;

            float32_t row_max_curr = -CUDART_INF_F;

            float32_t scores[B_c];

            // 8: On chip, compute S_ij = Q_i * K_j^T
            // in other words here we compute q * k.T raw attention score, scale it with sqrt(d_head)
            // mask padding zone with -inf, and track block max in row_max_curr
            #pragma unroll
            for (int k = 0; k < B_c; k+=1) {
                float32_t score = 0.0f;
                #pragma unroll
                for (int x = 0; x < D; x+=1) {
                    score += sramQ[row * D + x] * sramK[k * D + x];
                }
                score *= softmax_scale;

                if (j * B_c + k >= L) score = -CUDART_INF_F;

                scores[k] = score;
                row_max_curr = fmaxf(row_max_curr, score);
            }

            // 9: On chip, compute m_ij = max(m_i, rowmax(S_ij))
            float32_t new_max = fmaxf(row_max_prev, row_max_curr);
            float32_t exp_diff_prev = __expf(row_max_prev - new_max);
            float32_t row_sum_exp = 0.0f;

            // 9: On chip, compute P_ij = exp(S_ij - m_ij) (pointwise)
            // atp scores[] stores unnormalized probabilities because we do not need raw scores anymore
            // we cumulate in row_sum_expt all this unnormalized probabilities because we need it as
            // softmax s numerator
            #pragma unroll
            for (int k = 0; k < B_c; k+=1) {
                float32_t p = __expf(scores[k] - new_max);
                scores[k] = p;
                row_sum_exp += p;
            }

            // 9: On chip, compute l_ij = e^(m_prev - m_new) * l_prev + rowsum(P_ij)
            // update denominator using scaling factor based on max value after considering the block in scope
            const float32_t new_l = (row_l_prev * exp_diff_prev) + row_sum_exp;

            // 10: On chip, compute O_i = diag(e^(m_prev - m_new)) * O_i + P_ij * V_j
            #pragma unroll
            for (int x = 0; x < D; x+=1) {
                float32_t pv_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < B_c; k+=1) {
                     pv_sum += scores[k] * sramV[k * D + x];
                }
                acc[x] = (acc[x] * exp_diff_prev) + pv_sum;
            }

            l = new_l;
            m = new_max;
        }
        __syncthreads();
    }


    if (const uint32_t row = ty; row < B_r && (q_block_idx * B_r + row) < L) {
        // 12: On chip, compute O_i = diag(l_i)^-1 * O_i
        float32_t inv_l = 1.0f / (l + 1e-6f);
        const int out_idx = o_offset_base + (q_block_idx * B_r + row) * stride_seq;

        // 14: Write O_i to HBM as the i-th block of O.
        #pragma unroll
        for (int x = 0; x < D; x+=1) {
            O[out_idx + x] = acc[x] * inv_l;
        }

        // 13: On chip, compute L_i = m_i + log(l_i).
        // 15: Write L_i to HBM as the i-th block of L.
        const int l_idx = (batch_idx * H + head_idx) * L + (q_block_idx * B_r + row);
        L_cache[l_idx] = m + __logf(l);
    }
}

template __global__ void flash_attention_kernel<16, 32, 32>(
    const float*, const float*, const float*, float*, float*,
    int, int, int, float, int, int, int
);

template __global__ void flash_attention_kernel<16, 32, 64>(
    const float*, const float*, const float*, float*, float*,
    int, int, int, float, int, int, int
);