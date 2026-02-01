#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <cuda_runtime.h>
#include <string>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

#define TILE_DIM 32

#define COARSE_TILE_DIM 64
#define COARSE_FACTOR 4

#define BM 64
#define BN 64
#define BK 16

#define TM 4
#define TN 4

void cpuMatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verifyResult(const float* cpuC, const float* gpuC, int M, int N) {
    int errors = 0;
    int size = M * N;
    float tolerance = 1e-2; 

    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(cpuC[i] - gpuC[i]);
        if (diff > tolerance) {
            float rel_err = diff / (std::fabs(cpuC[i]) + 1e-5);
            if (rel_err > 0.01) {
                if (errors < 5) {
                    int r = i / N;
                    int c = i % N;
                    printf("Error at [%d][%d]: CPU=%.4f, GPU=%.4f, Diff=%.4f\n", 
                           r, c, cpuC[i], gpuC[i], diff);
                }
                errors++;
            }
        }
    }
    if (errors > 0) {
        printf("Total errors: %d / %d\n", errors, size);
        return false;
    }
    return true;
}

__global__ void naiveMatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void tiledMatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float sum = 0.0f;

    for (int k_phase = 0; k_phase < K; k_phase += TILE_DIM) {
        if (row < M && (k_phase + tx) < K)
            s_A[ty][tx] = A[row * K + (k_phase + tx)];
        else
            s_A[ty][tx] = 0.0f;

        if (col < N && (k_phase + ty) < K)
            s_B[ty][tx] = B[(k_phase + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void vectorizedMatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float sum = 0.0f;

    const float4* A_vec = reinterpret_cast<const float4*>(A);
    int K_vec = K / 4; 

    for (int k_phase = 0; k_phase < K; k_phase += TILE_DIM) {
        int k_phase_vec = k_phase / 4;

        if (tx < (TILE_DIM / 4)) {
            if (row < M && (k_phase_vec + tx) < K_vec) {
                float4 temp = A_vec[row * K_vec + k_phase_vec + tx];
                s_A[ty][tx * 4 + 0] = temp.x;
                s_A[ty][tx * 4 + 1] = temp.y;
                s_A[ty][tx * 4 + 2] = temp.z;
                s_A[ty][tx * 4 + 3] = temp.w;
            } else {
                s_A[ty][tx * 4 + 0] = 0.0f;
                s_A[ty][tx * 4 + 1] = 0.0f;
                s_A[ty][tx * 4 + 2] = 0.0f;
                s_A[ty][tx * 4 + 3] = 0.0f;
            }
        }

        if (col < N && (k_phase + ty) < K)
            s_B[ty][tx] = B[(k_phase + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void coarseMatMul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float s_A[COARSE_TILE_DIM][COARSE_TILE_DIM];
    __shared__ float s_B[COARSE_TILE_DIM][COARSE_TILE_DIM];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int rowStart = by * COARSE_TILE_DIM;
    int colStart = bx * COARSE_TILE_DIM;

    float results[COARSE_FACTOR][COARSE_FACTOR];
    for(int i=0; i<COARSE_FACTOR; i++) 
        for(int j=0; j<COARSE_FACTOR; j++) 
            results[i][j] = 0.0f;

    for (int k_phase = 0; k_phase < K; k_phase += COARSE_TILE_DIM) {
        int tid = ty * blockDim.x + tx;
        
        for (int i = 0; i < 16; i++) {
            int t_idx = tid + i * 256; 
            int s_r = t_idx / COARSE_TILE_DIM;
            int s_c = t_idx % COARSE_TILE_DIM;

            int global_r_A = rowStart + s_r;
            int global_c_A = k_phase + s_c;
            
            if (global_r_A < M && global_c_A < K)
                s_A[s_r][s_c] = A[global_r_A * K + global_c_A];
            else
                s_A[s_r][s_c] = 0.0f;

            int global_r_B = k_phase + s_r;
            int global_c_B = colStart + s_c;

            if (global_r_B < K && global_c_B < N)
                s_B[s_r][s_c] = B[global_r_B * N + global_c_B];
            else
                s_B[s_r][s_c] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < COARSE_TILE_DIM; ++k) {
            for (int i = 0; i < COARSE_FACTOR; ++i) {
                for (int j = 0; j < COARSE_FACTOR; ++j) {
                    results[i][j] += s_A[ty * COARSE_FACTOR + i][k] * s_B[k][tx * COARSE_FACTOR + j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < COARSE_FACTOR; ++i) {
        for (int j = 0; j < COARSE_FACTOR; ++j) {
            int global_r = rowStart + ty * COARSE_FACTOR + i;
            int global_c = colStart + tx * COARSE_FACTOR + j;
            
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = results[i][j];
            }
        }
    }
}

__global__ void __launch_bounds__(256) ultimateMatMul(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y; 
    int tid = ty * blockDim.x + tx;

    float threadResults[TM][TN] = {0.0f};

    float reg_A[TM];
    float reg_B[TN];

    const float4* A_vec = reinterpret_cast<const float4*>(A);
    const float4* B_vec = reinterpret_cast<const float4*>(B);

    int rowA_start = by * BM;
    int colB_start = bx * BN;

    int K_vec = K / 4; 
    int N_vec = N / 4;

    for (int k = 0; k < K; k += BK) {
        int l_row_A = tid / 4;
        int l_col_A = tid % 4;

        int global_r_A = rowA_start + l_row_A;
        int global_c_A = (k / 4) + l_col_A;

        if (global_r_A < M && global_c_A < K_vec) {
             float4 loaded = A_vec[global_r_A * K_vec + global_c_A];
             s_A[l_row_A][l_col_A * 4 + 0] = loaded.x;
             s_A[l_row_A][l_col_A * 4 + 1] = loaded.y;
             s_A[l_row_A][l_col_A * 4 + 2] = loaded.z;
             s_A[l_row_A][l_col_A * 4 + 3] = loaded.w;
        } else {
             s_A[l_row_A][l_col_A * 4 + 0] = 0.0f;
             s_A[l_row_A][l_col_A * 4 + 1] = 0.0f;
             s_A[l_row_A][l_col_A * 4 + 2] = 0.0f;
             s_A[l_row_A][l_col_A * 4 + 3] = 0.0f;
        }
        int l_row_B = tid / 16;
        int l_col_B = tid % 16;

        int global_r_B = k + l_row_B;
        int global_c_B = (colB_start / 4) + l_col_B;

        if (global_r_B < K && global_c_B < N_vec) {
            float4 loaded = B_vec[global_r_B * N_vec + global_c_B];
            s_B[l_row_B][l_col_B * 4 + 0] = loaded.x;
            s_B[l_row_B][l_col_B * 4 + 1] = loaded.y;
            s_B[l_row_B][l_col_B * 4 + 2] = loaded.z;
            s_B[l_row_B][l_col_B * 4 + 3] = loaded.w;
        } else {
            s_B[l_row_B][l_col_B * 4 + 0] = 0.0f;
            s_B[l_row_B][l_col_B * 4 + 1] = 0.0f;
            s_B[l_row_B][l_col_B * 4 + 2] = 0.0f;
            s_B[l_row_B][l_col_B * 4 + 3] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k_idx = 0; k_idx < BK; ++k_idx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = s_A[ty * TM + i][k_idx];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[j] = s_B[k_idx][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_r = rowA_start + ty * TM + i;
            int global_c = colB_start + tx * TN + j;
            
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = threadResults[i][j];
            }
        }
    }
}

struct KernelInfo {
    std::string name;
    std::function<void(const float*, const float*, float*, int, int, int)> launcher;
};

float benchmarkKernel(std::function<void(const float*, const float*, float*, int, int, int)> kernelLaunch, 
                      const float* d_A, const float* d_B, float* d_C, 
                      int M, int N, int K) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    kernelLaunch(d_A, d_B, d_C, M, N, K); 
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i = 0; i < 10; i++) { 
        kernelLaunch(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 10.0f;
}

int main() {
    std::vector<int> sizes;

    for (int s = 128; s <= 1024; s += 128) {
        sizes.push_back(s);
    }

    for (int s = 1536; s <= 4096; s += 512) {
        sizes.push_back(s);
    }

    sizes.push_back(6144);
    sizes.push_back(8192);
    sizes.push_back(12288);
    sizes.push_back(16384);
    
    std::vector<KernelInfo> kernels = {
        {"0_Naive", [](const float* A, const float* B, float* C, int M, int N, int K) {
            dim3 block(32, 32);
            dim3 grid((N + 31) / 32, (M + 31) / 32);
            naiveMatMul<<<grid, block>>>(A, B, C, M, N, K);
        }},
        {"1_Tiled", [](const float* A, const float* B, float* C, int M, int N, int K) {
            dim3 block(TILE_DIM, TILE_DIM);
            dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
            tiledMatMul<<<grid, block>>>(A, B, C, M, N, K);
        }},
        {"2_Vectorized_A", [](const float* A, const float* B, float* C, int M, int N, int K) {
            dim3 block(TILE_DIM, TILE_DIM);
            dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
            vectorizedMatMul<<<grid, block>>>(A, B, C, M, N, K);
        }},
        {"3_Coarse", [](const float* A, const float* B, float* C, int M, int N, int K) {
            dim3 block(16, 16);
            dim3 grid((N + 63) / 64, (M + 63) / 64);
            coarseMatMul<<<grid, block>>>(A, B, C, M, N, K);
        }},
        {"4_Ultimate_Optimized", [](const float* A, const float* B, float* C, int M, int N, int K) {
            dim3 block(16, 16); 
            dim3 grid((N + 63) / 64, (M + 63) / 64);
            ultimateMatMul<<<grid, block>>>(A, B, C, M, N, K);
        }}
    };

    printf("Kernel,MatrixSize,TimeMS,GFLOPS\n");

    for (int size : sizes) {
        int M = size, N = size, K = size;
        size_t bytes = M * N * sizeof(float);
        
        std::vector<float> h_A(M * K);
        std::vector<float> h_B(K * N);
        std::vector<float> h_C_cpu(M * N);
        std::vector<float> h_C_gpu(M * N);

        for(int i=0; i<M*K; i++) h_A[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f;
        for(int i=0; i<K*N; i++) h_B[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f;

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

        bool check = (size <= 512); 
        if(check) cpuMatMul(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

        for (auto& k : kernels) {
            CHECK_CUDA(cudaMemset(d_C, 0, bytes));
            float avgTimeMs = benchmarkKernel(k.launcher, d_A, d_B, d_C, M, N, K);
            
            if(check) {
                CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
                if (!verifyResult(h_C_cpu.data(), h_C_gpu.data(), M, N)) {
                    fprintf(stderr, "FAIL: %s at size %d\n", k.name.c_str(), size);
                }
            }

            double gflops = (2.0 * M * N * K) / (avgTimeMs * 1e-3 * 1e9);
            printf("%s,%d,%.4f,%.4f\n", k.name.c_str(), size, avgTimeMs, gflops);
        }

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }

    return 0;
}
