#include <backend/Kernels.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <algorithm>

#define MAX_HEAD_DIM 64

__device__ __forceinline__ float4 load_float4(const float* ptr, int idx) {
    return reinterpret_cast<const float4*>(ptr)[idx];
}

__device__ __forceinline__ void store_float4(float* ptr, int idx, float4 val) {
    reinterpret_cast<float4*>(ptr)[idx] = val;
}

template <int Br, int Bc, int D>
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int N,
    const int H,
    const int L,
    const float softmax_scale,
    const int stride_batch,
    const int stride_head,
    const int stride_seq
) {
    extern __shared__ float sram[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int q_block_idx = by;
    int kv_block_count = (L + Bc - 1) / Bc;

    int batch_idx = bx / H;
    int head_idx = bx % H;

    int q_offset_base = batch_idx * stride_batch + head_idx * stride_head;
    int k_offset_base = batch_idx * stride_batch + head_idx * stride_head;
    int v_offset_base = batch_idx * stride_batch + head_idx * stride_head;
    int o_offset_base = batch_idx * stride_batch + head_idx * stride_head;

    float* sQ = sram;
    float* sK = sram + Br * D;
    float* sV = sram + Br * D + Bc * D;

    float acc[D] = {0.0f};
    float l = 0.0f;
    float m = -CUDART_INF_F;

    int q_global_offset = q_offset_base + (q_block_idx * Br * stride_seq);

    #pragma unroll
    for (int i = ty; i < Br; i += blockDim.y) {
        if (q_block_idx * Br + i < L) {
            #pragma unroll
            for (int j = tx; j < D / 4; j += blockDim.x) {
                float4 loaded = load_float4(Q + q_global_offset + i * stride_seq, j);
                reinterpret_cast<float4*>(&sQ[i * D])[j] = loaded;
            }
        } else {
             #pragma unroll
             for (int j = tx; j < D / 4; j += blockDim.x) {
                reinterpret_cast<float4*>(&sQ[i * D])[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
             }
        }
    }

    __syncthreads();

    for (int j = 0; j < kv_block_count; ++j) {
        int k_global_offset = k_offset_base + (j * Bc * stride_seq);
        int v_global_offset = v_offset_base + (j * Bc * stride_seq);

        #pragma unroll
        for (int i = ty; i < Bc; i += blockDim.y) {
            if (j * Bc + i < L) {
                #pragma unroll
                for (int x = tx; x < D / 4; x += blockDim.x) {
                    float4 val_k = load_float4(K + k_global_offset + i * stride_seq, x);
                    reinterpret_cast<float4*>(&sK[i * D])[x] = val_k;

                    float4 val_v = load_float4(V + v_global_offset + i * stride_seq, x);
                    reinterpret_cast<float4*>(&sV[i * D])[x] = val_v;
                }
            } else {
                #pragma unroll
                for (int x = tx; x < D / 4; x += blockDim.x) {
                    reinterpret_cast<float4*>(&sK[i * D])[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    reinterpret_cast<float4*>(&sV[i * D])[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        int row = ty;
        if (row < Br && (q_block_idx * Br + row) < L) {
            float row_m_prev = m;
            float row_l_prev = l;

            float row_m_curr = -CUDART_INF_F;
            float row_l_curr = 0.0f;

            float scores[Bc];

            #pragma unroll
            for (int k = 0; k < Bc; ++k) {
                float score = 0.0f;
                #pragma unroll
                for (int x = 0; x < D; ++x) {
                    score += sQ[row * D + x] * sK[k * D + x];
                }
                score *= softmax_scale;

                if (j * Bc + k >= L) score = -CUDART_INF_F;

                scores[k] = score;
                row_m_curr = fmaxf(row_m_curr, score);
            }

            float new_m = fmaxf(row_m_prev, row_m_curr);

            float exp_diff_prev = __expf(row_m_prev - new_m);
            float exp_diff_curr = __expf(row_m_curr - new_m);

            float row_sum_exp = 0.0f;
            #pragma unroll
            for (int k = 0; k < Bc; ++k) {
                float p = __expf(scores[k] - new_m);
                scores[k] = p;
                row_sum_exp += p;
            }

            float new_l = (row_l_prev * exp_diff_prev) + row_sum_exp;

            #pragma unroll
            for (int x = 0; x < D; ++x) {
                float pv_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < Bc; ++k) {
                     pv_sum += scores[k] * sV[k * D + x];
                }
                acc[x] = (acc[x] * exp_diff_prev) + pv_sum;
            }

            l = new_l;
            m = new_m;
        }
        __syncthreads();
    }

    int row = ty;
    if (row < Br && (q_block_idx * Br + row) < L) {
        float inv_l = 1.0f / (l + 1e-6f);
        int out_idx = o_offset_base + (q_block_idx * Br + row) * stride_seq;

        #pragma unroll
        for (int x = 0; x < D; ++x) {
            O[out_idx + x] = acc[x] * inv_l;
        }
    }
}

template __global__ void flash_attention_kernel<16, 32, 32>(
    const float*, const float*, const float*, float*,
    int, int, int, float, int, int, int
);

template __global__ void flash_attention_kernel<16, 32, 64>(
    const float*, const float*, const float*, float*,
    int, int, int, float, int, int, int
);