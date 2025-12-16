#include <backend/Kernels.cuh>


#define WARP_SIZE 32
#define MAX_THREADS 256

__device__ __forceinline__ float4 load_float4(const float* ptr, int idx) {
    return reinterpret_cast<const float4*>(ptr)[idx];
}

__device__ __forceinline__ void store_float4(float* ptr, int idx, float4 val) {
    reinterpret_cast<float4*>(ptr)[idx] = val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__global__ void compute_delta_kernel(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ Delta,
    const int N, const int H, const int L, const int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H * L) return;

    float sum = 0.0f;
    int offset = idx * D;

    for (int i = 0; i < D; ++i) {
        sum += dO[offset + i] * O[offset + i];
    }
    Delta[idx] = sum;
}

template <int Bc, int Br, int D>
__global__ void __launch_bounds__(256) flash_attn_backward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ L_vec,
    const float* __restrict__ Delta,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    const int stride_batch,
    const int stride_head,
    const int stride_seq,
    const int L,
    const float sm_scale
) {
    constexpr int PAD = 8;
    constexpr int PADDED_D = D + PAD;

    extern __shared__ float sram[];

    float* sK  = sram;
    float* sV  = sram + Bc * PADDED_D;
    float* sdK = sram + 2 * Bc * PADDED_D;
    float* sdV = sram + 3 * Bc * PADDED_D;

    float* sQ  = sram + 4 * Bc * PADDED_D;
    float* sdO = sram + 4 * Bc * PADDED_D + Br * PADDED_D;

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int k_start_idx = bx * Bc;
    int offset_base = by * L * D;
    int vector_base = by * L;

    for (int i = tx; i < Bc * PADDED_D; i += blockDim.x) {
        if (i < Bc * PADDED_D) { sdK[i] = 0.0f; }
        if (i < Bc * PADDED_D) { sdV[i] = 0.0f; }
    }
    __syncthreads();

    if (k_start_idx < L) {
        #pragma unroll
        for (int i = 0; i < Bc; ++i) {
            if (k_start_idx + i < L) {
                #pragma unroll
                for (int x = tx * 4; x < D; x += blockDim.x * 4) {
                    int global_idx = (k_start_idx + i) * D + x;
                    float4 val_k = load_float4(K + offset_base, global_idx / 4);
                    float4 val_v = load_float4(V + offset_base, global_idx / 4);

                    int sram_idx = i * PADDED_D + x;
                    sK[sram_idx + 0] = val_k.x;
                    sK[sram_idx + 1] = val_k.y;
                    sK[sram_idx + 2] = val_k.z;
                    sK[sram_idx + 3] = val_k.w;

                    sV[sram_idx + 0] = val_v.x;
                    sV[sram_idx + 1] = val_v.y;
                    sV[sram_idx + 2] = val_v.z;
                    sV[sram_idx + 3] = val_v.w;
                }
            }
        }
    }
    __syncthreads();
    int num_q_blocks = (L + Br - 1) / Br;

    for (int i = 0; i < num_q_blocks; ++i) {
        int q_start_idx = i * Br;

        #pragma unroll
        for (int row = 0; row < Br; ++row) {
            if (q_start_idx + row < L) {
                #pragma unroll
                for (int x = tx * 4; x < D; x += blockDim.x * 4) {
                    int global_idx = (q_start_idx + row) * D + x;
                    float4 val_q = load_float4(Q + offset_base, global_idx / 4);
                    float4 val_do = load_float4(dO + offset_base, global_idx / 4);

                    int sram_idx = row * PADDED_D + x;
                    sQ[sram_idx + 0] = val_q.x;
                    sQ[sram_idx + 1] = val_q.y;
                    sQ[sram_idx + 2] = val_q.z;
                    sQ[sram_idx + 3] = val_q.w;

                    sdO[sram_idx + 0] = val_do.x;
                    sdO[sram_idx + 1] = val_do.y;
                    sdO[sram_idx + 2] = val_do.z;
                    sdO[sram_idx + 3] = val_do.w;
                }
            }
        }
        __syncthreads();

        for (int q_row = 0; q_row < Br; ++q_row) {
            int global_q = q_start_idx + q_row;
            if (global_q >= L) continue;

            float l_val = L_vec[vector_base + global_q];
            float delta_val = Delta[vector_base + global_q];

            for (int k_idx = tx; k_idx < Bc; k_idx += blockDim.x) {

                float score = 0.0f;
                #pragma unroll
                for (int x = 0; x < D; ++x) {
                    score += sQ[q_row * PADDED_D + x] * sK[k_idx * PADDED_D + x];
                }

                float p = __expf(score * sm_scale - l_val);

                float dP_val = 0.0f;
                #pragma unroll
                for (int x = 0; x < D; ++x) {
                    dP_val += sdO[q_row * PADDED_D + x] * sV[k_idx * PADDED_D + x];
                }

                float dS = p * (dP_val - delta_val) * sm_scale;

                #pragma unroll
                for (int x = 0; x < D; ++x) {
                    float my_val = dS * sK[k_idx * PADDED_D + x];
                    float warp_sum = warp_reduce_sum(my_val);

                    if ((threadIdx.x % WARP_SIZE) == 0) {
                        atomicAdd(&dQ[offset_base + global_q * D + x], warp_sum);
                    }
                }

                #pragma unroll
                for (int x = 0; x < D; ++x) {
                    sdK[k_idx * PADDED_D + x] += dS * sQ[q_row * PADDED_D + x];
                    sdV[k_idx * PADDED_D + x] += p  * sdO[q_row * PADDED_D + x];
                }
            }
        }
        __syncthreads();
    }

    if (k_start_idx < L) {
        #pragma unroll
        for (int i = 0; i < Bc; ++i) {
            if (k_start_idx + i < L) {
                #pragma unroll
                for (int x = tx * 4; x < D; x += blockDim.x * 4) {
                    int global_idx = (k_start_idx + i) * D + x;
                    int sram_idx = i * PADDED_D + x;

                    float4 val_dk;
                    val_dk.x = sdK[sram_idx + 0];
                    val_dk.y = sdK[sram_idx + 1];
                    val_dk.z = sdK[sram_idx + 2];
                    val_dk.w = sdK[sram_idx + 3];

                    float4 val_dv;
                    val_dv.x = sdV[sram_idx + 0];
                    val_dv.y = sdV[sram_idx + 1];
                    val_dv.z = sdV[sram_idx + 2];
                    val_dv.w = sdV[sram_idx + 3];

                    store_float4(dK + offset_base, global_idx / 4, val_dk);
                    store_float4(dV + offset_base, global_idx / 4, val_dv);
                }
            }
        }
    }
}



template __global__ void __launch_bounds__(256) flash_attn_backward_kernel<64, 16, 64>(
    const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*,
    int, int, int, int, float
);

template __global__ void __launch_bounds__(256) flash_attn_backward_kernel<64, 16, 32>(
    const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*,
    int, int, int, int, float
);