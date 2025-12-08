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

template <int B_r, int B_c, int D>
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,    // N x L x E
    const float* __restrict__ K,    // N x L x E
    const float* __restrict__ V,    // N x L x E
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

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int q_block_idx = by;
    const int kv_block_count = (L + B_c - 1) / B_c;

    const int batch_idx = bx / H;
    const int head_idx = bx % H;

    const int q_offset_base = batch_idx * stride_batch + head_idx * stride_head;
    const int k_offset_base = batch_idx * stride_batch + head_idx * stride_head;
    const int v_offset_base = batch_idx * stride_batch + head_idx * stride_head;
    const int o_offset_base = batch_idx * stride_batch + head_idx * stride_head;

    float* sramQ = sram;
    float* sramK = sram + B_r * D;
    float* sramV = sram + B_r * D + B_c * D;

    float acc[D] = {0.0f};
    float l = 0.0f;
    float m = -CUDART_INF_F;

    int q_global_offset = q_offset_base + (q_block_idx * B_r * stride_seq);

    //
    #pragma unroll
    for (int i = ty; i < B_r; i += blockDim.y) {
        if (q_block_idx * B_r + i < L) {
            #pragma unroll
            for (int j = tx; j < D / 4; j += blockDim.x) {
                float4 loaded = load_float4(Q + q_global_offset + i * stride_seq, j);
                reinterpret_cast<float4*>(&sramQ[i * D])[j] = loaded;
            }
        } else {
             #pragma unroll
             for (int j = tx; j < D / 4; j += blockDim.x) {
                reinterpret_cast<float4*>(&sramQ[i * D])[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
             }
        }
    }

    __syncthreads();

    for (int j = 0; j < kv_block_count; ++j) {
        int k_global_offset = k_offset_base + (j * B_c * stride_seq);
        int v_global_offset = v_offset_base + (j * B_c * stride_seq);

        #pragma unroll
        for (int i = ty; i < B_c; i += blockDim.y) {
            if (j * B_c + i < L) {
                #pragma unroll
                for (int x = tx; x < D / 4; x += blockDim.x) {
                    float4 val_k = load_float4(K + k_global_offset + i * stride_seq, x);
                    reinterpret_cast<float4*>(&sramK[i * D])[x] = val_k;

                    float4 val_v = load_float4(V + v_global_offset + i * stride_seq, x);
                    reinterpret_cast<float4*>(&sramV[i * D])[x] = val_v;
                }
            } else {
                #pragma unroll
                for (int x = tx; x < D / 4; x += blockDim.x) {
                    reinterpret_cast<float4*>(&sramK[i * D])[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    reinterpret_cast<float4*>(&sramV[i * D])[x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        int row = ty;
        if (row < B_r && (q_block_idx * B_r + row) < L) {
            float row_m_prev = m;
            float row_l_prev = l;

            float row_m_curr = -CUDART_INF_F;
            float row_l_curr = 0.0f;

            float scores[B_c];

            #pragma unroll
            for (int k = 0; k < B_c; ++k) {
                float score = 0.0f;
                #pragma unroll
                for (int x = 0; x < D; ++x) {
                    score += sramQ[row * D + x] * sramK[k * D + x];
                }
                score *= softmax_scale;

                if (j * B_c + k >= L) score = -CUDART_INF_F;

                scores[k] = score;
                row_m_curr = fmaxf(row_m_curr, score);
            }

            float new_m = fmaxf(row_m_prev, row_m_curr);

            float exp_diff_prev = __expf(row_m_prev - new_m);
            float exp_diff_curr = __expf(row_m_curr - new_m);

            float row_sum_exp = 0.0f;
            #pragma unroll
            for (int k = 0; k < B_c; ++k) {
                float p = __expf(scores[k] - new_m);
                scores[k] = p;
                row_sum_exp += p;
            }

            float new_l = (row_l_prev * exp_diff_prev) + row_sum_exp;

            #pragma unroll
            for (int x = 0; x < D; ++x) {
                float pv_sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < B_c; ++k) {
                     pv_sum += scores[k] * sramV[k * D + x];
                }
                acc[x] = (acc[x] * exp_diff_prev) + pv_sum;
            }

            l = new_l;
            m = new_m;
        }
        __syncthreads();
    }

    if (const int row = ty; row < B_r && (q_block_idx * B_r + row) < L) {
        float inv_l = 1.0f / (l + 1e-6f);
        const int out_idx = o_offset_base + (q_block_idx * B_r + row) * stride_seq;

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