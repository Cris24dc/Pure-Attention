#include <backend/Kernels.cuh>


#define MAX_THREADS 256

__device__ __forceinline__ float4 load_float4(const float* ptr, int idx) {
    return reinterpret_cast<const float4*>(ptr)[idx];
}

__device__ __forceinline__ void store_float4(float* ptr, int idx, float4 val) {
    reinterpret_cast<float4*>(ptr)[idx] = val;
}

__global__ void compute_delta_kernel(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ Delta,
    const int N, const int H, const int L, const int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H * L) return;

    int batch_head_idx = idx / L;
    int seq_idx = idx % L;

    float sum = 0.0f;
    int offset = idx * D;

    for (int i = 0; i < D; ++i) {
        sum += dO[offset + i] * O[offset + i];
    }
    Delta[idx] = sum;
}

template <int Br, int Bc, int D>
__global__ void __launch_bounds__(128) flash_attn_backward_kernel(
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
    extern __shared__ float sram[];
    float* sQ = sram;
    float* sdO = sram + Br * D;
    float* sK = sram + 2 * Br * D;
    float* sV = sram + 2 * Br * D + Bc * D;

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int q_start = bx * Br;

    int batch_idx = bz / stride_batch;
    int head_idx = bz % stride_batch;

    int offset_base = bz * L * D;
    int delta_base = bz * L;

    float dQ_acc[D] = {0.0f};

    if (q_start < L) {
        #pragma unroll
        for (int i = 0; i < Br; ++i) {
            if (q_start + i < L) {
                #pragma unroll
                for (int x = tx; x < D / 4; x += blockDim.x) {
                    int idx = (q_start + i) * D + x * 4;
                    float4 val_q = load_float4(Q + offset_base, idx / 4);
                    float4 val_do = load_float4(dO + offset_base, idx / 4);
                    reinterpret_cast<float4*>(&sQ[i * D])[x] = val_q;
                    reinterpret_cast<float4*>(&sdO[i * D])[x] = val_do;
                }
            }
        }
    }
    __syncthreads();

    int num_kv_blocks = (L + Bc - 1) / Bc;

    for (int j = 0; j < num_kv_blocks; ++j) {
        int k_start = j * Bc;

        #pragma unroll
        for (int i = 0; i < Bc; ++i) {
            if (k_start + i < L) {
                #pragma unroll
                for (int x = tx; x < D / 4; x += blockDim.x) {
                    int idx = (k_start + i) * D + x * 4;
                    float4 val_k = load_float4(K + offset_base, idx / 4);
                    float4 val_v = load_float4(V + offset_base, idx / 4);
                    reinterpret_cast<float4*>(&sK[i * D])[x] = val_k;
                    reinterpret_cast<float4*>(&sV[i * D])[x] = val_v;
                }
            }
        }
        __syncthreads();

        for (int i = 0; i < Br; ++i) {
            int row = q_start + i;
            if (row >= L) break;

            float l_val = L_vec[delta_base + row];
            float delta_val = Delta[delta_base + row];

            for (int k = 0; k < Bc; ++k) {
                int col = k_start + k;
                if (col >= L) break;

                float score = 0.0f;
                #pragma unroll
                for (int x = 0; x < D; ++x) {
                    score += sQ[i * D + x] * sK[k * D + x];
                }

                float p = __expf(score * sm_scale - l_val);

                float dP_val = 0.0f;
                #pragma unroll
                for (int x = 0; x < D; ++x) {
                    dP_val += sdO[i * D + x] * sV[k * D + x];
                }

                float dS = p * (dP_val - delta_val) * sm_scale;

                #pragma unroll
                for (int x = 0; x < D; ++x) {
                    dQ_acc[x] += dS * sK[k * D + x];

                    float dK_val = dS * sQ[i * D + x];
                    atomicAdd(&dK[offset_base + col * D + x], dK_val);

                    float dV_val = p * sdO[i * D + x];
                    atomicAdd(&dV[offset_base + col * D + x], dV_val);
                }
            }
        }
        __syncthreads();
    }

    if (q_start < L) {
        for (int i = 0; i < Br; ++i) {
            if (q_start + i < L) {
                 #pragma unroll
                 for (int x = tx; x < D; x += blockDim.x) {
                     dQ[offset_base + (q_start + i) * D + x] = dQ_acc[x];
                 }
            }
        }
    }
}



template __global__ void flash_attn_backward_kernel<16, 32, 64>(
    const float* __restrict__, // Q
    const float* __restrict__, // K
    const float* __restrict__, // V
    const float* __restrict__, // O
    const float* __restrict__, // dO
    const float* __restrict__, // L_vec
    const float* __restrict__, // Delta
    float* __restrict__,       // dQ
    float* __restrict__,       // dK
    float* __restrict__,       // dV
    int,                       // stride_batch
    int,                       // stride_head
    int,                       // stride_seq
    int,                       // L
    float                      // sm_scale
);

template __global__ void flash_attn_backward_kernel<16, 32, 32>(
    const float* __restrict__,
    const float* __restrict__,
    const float* __restrict__,
    const float* __restrict__,
    const float* __restrict__,
    const float* __restrict__,
    const float* __restrict__,
    float* __restrict__,
    float* __restrict__,
    float* __restrict__,
    int,
    int,
    int,
    int,
    float
);