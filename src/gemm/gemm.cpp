#include <cstdlib>
#include "microflow/gemm.hpp"
#include <cassert>
#include <cstring>
#include <algorithm>
#include <chrono>

// ARM NEON头文件
#if defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
    #define MICROFLOW_HAS_NEON
#endif

// OpenMP
#include <omp.h>

namespace microflow {

//==========================================================================
// 硬件配置查询
//==========================================================================

/**
 * @brief 获取CPU L1缓存大小
 *
 * 树莓派4 (Cortex-A72): 48KB 数据缓存
 */
static constexpr int get_l1_cache_size() {
    return 48 * 1024;  // 48KB
}

/**
 * @brief 获取CPU L2缓存大小
 *
 * 树莓派4 (Cortex-A72): 1MB 共享缓存
 */
static constexpr int get_l2_cache_size() {
    return 1024 * 1024;  // 1MB
}

/**
 * @brief 获取缓存行大小
 *
 * ARMv8: 64字节
 */
static constexpr int get_cache_line_size() {
    return 64;
}

//==========================================================================
// GEMM配置调优
//==========================================================================

GEMMConfig get_optimal_config(int M, int N, int K) {
    GEMMConfig config;

    // 根据矩阵大小调整分块参数
    // 目标: 每个块能放入L1缓存

    int l1_size = get_l1_cache_size();
    int l2_size = get_l2_cache_size();

    // mc * kc * sizeof(float) < L1 / 3 (给A, B, C各留空间)
    // 48KB / 3 ≈ 16KB = 4096 floats
    // mc * kc ≈ 4096
    // 选择 mc = 64, kc = 64

    if (M <= 32 && N <= 32 && K <= 32) {
        // 小矩阵: 使用小分块
        config.mc = 16;
        config.nc = 16;
        config.kc = 32;
    } else if (M <= 128 && N <= 128) {
        // 中等矩阵: 适配L1缓存
        config.mc = 32;
        config.nc = 32;
        config.kc = 128;
    } else {
        // 大矩阵: 适配L2缓存
        config.mc = 64;
        config.nc = 64;
        config.kc = 256;
    }

    // 寄存器分块保持固定
    // 4x8是NEON的最优选择 (32个128位寄存器)
    config.mr = 4;
    config.nr = 8;

    return config;
}

GEMMImpl select_best_implementation(int M, int N, int K) {
    // 小矩阵直接用naive
    if (M <= 8 && N <= 8 && K <= 8) {
        return GEMMImpl::kNaive;
    }

    // 中等矩阵用NEON
    if (M <= 256 && N <= 256) {
        return GEMMImpl::kNEON;
    }

    // 大矩阵用NEON + OpenMP
    return GEMMImpl::kNEON;
}

//==========================================================================
// 基础实现
//==========================================================================

void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.shapes()[0];
    int K = A.shapes()[1];
    int N = B.shapes()[1];

    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();

    // 三重循环
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += ptr_A[i * K + k] * ptr_B[k * N + j];
            }
            ptr_C[i * N + j] = sum;
        }
    }
}

void gemm_omp(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.shapes()[0];
    int K = A.shapes()[1];
    int N = B.shapes()[1];

    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();

    // OpenMP并行化外层循环
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += ptr_A[i * K + k] * ptr_B[k * N + j];
            }
            ptr_C[i * N + j] = sum;
        }
    }
}

//==========================================================================
// ARM NEON优化实现
//==========================================================================

#ifdef MICROFLOW_HAS_NEON

/**
 * @brief NEON 4x8微内核
 *
 * @detail:
 * 计算 C[4][8] += A[4][k] * B[k][8]
 *
 * @寄存器分配:
 * - v0-v3: A的4个元素 (广播为4个float)
 * - v4-v11: C的8列累加器
 * - v12-v15: B的临时加载
 *
 * @性能: 每个K迭代处理32个浮点乘加
 */
void gemm_micro_kernel_4x8(
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int K)
{
    // 加载C的初始值 (8列)
    float32x4_t c0 = vld1q_f32(&C[0 * ldc + 0]);
    float32x4_t c1 = vld1q_f32(&C[0 * ldc + 4]);
    float32x4_t c2 = vld1q_f32(&C[1 * ldc + 0]);
    float32x4_t c3 = vld1q_f32(&C[1 * ldc + 4]);
    float32x4_t c4 = vld1q_f32(&C[2 * ldc + 0]);
    float32x4_t c5 = vld1q_f32(&C[2 * ldc + 4]);
    float32x4_t c6 = vld1q_f32(&C[3 * ldc + 0]);
    float32x4_t c7 = vld1q_f32(&C[3 * ldc + 4]);

    // 主循环: K维度
    int k = 0;

    // 4x展开以隐藏延迟
    for (; k <= K - 4; k += 4) {
        // ===== 迭代 0 =====
        float32x4_t a0 = vdupq_n_f32(A[0 * lda + k]);
        float32x4_t b0 = vld1q_f32(&B[k * ldb + 0]);
        c0 = vmlaq_f32(c0, a0, b0);

        float32x4_t b1 = vld1q_f32(&B[k * ldb + 4]);
        c1 = vmlaq_f32(c1, a0, b1);

        float32x4_t a1 = vdupq_n_f32(A[1 * lda + k]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        float32x4_t a2 = vdupq_n_f32(A[2 * lda + k]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        float32x4_t a3 = vdupq_n_f32(A[3 * lda + k]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);

        // ===== 迭代 1 =====
        a0 = vdupq_n_f32(A[0 * lda + k + 1]);
        b0 = vld1q_f32(&B[(k + 1) * ldb + 0]);
        c0 = vmlaq_f32(c0, a0, b0);

        b1 = vld1q_f32(&B[(k + 1) * ldb + 4]);
        c1 = vmlaq_f32(c1, a0, b1);

        a1 = vdupq_n_f32(A[1 * lda + k + 1]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        a2 = vdupq_n_f32(A[2 * lda + k + 1]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        a3 = vdupq_n_f32(A[3 * lda + k + 1]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);

        // ===== 迭代 2 =====
        a0 = vdupq_n_f32(A[0 * lda + k + 2]);
        b0 = vld1q_f32(&B[(k + 2) * ldb + 0]);
        c0 = vmlaq_f32(c0, a0, b0);

        b1 = vld1q_f32(&B[(k + 2) * ldb + 4]);
        c1 = vmlaq_f32(c1, a0, b1);

        a1 = vdupq_n_f32(A[1 * lda + k + 2]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        a2 = vdupq_n_f32(A[2 * lda + k + 2]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        a3 = vdupq_n_f32(A[3 * lda + k + 2]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);

        // ===== 迭代 3 =====
        a0 = vdupq_n_f32(A[0 * lda + k + 3]);
        b0 = vld1q_f32(&B[(k + 3) * ldb + 0]);
        c0 = vmlaq_f32(c0, a0, b0);

        b1 = vld1q_f32(&B[(k + 3) * ldb + 4]);
        c1 = vmlaq_f32(c1, a0, b1);

        a1 = vdupq_n_f32(A[1 * lda + k + 3]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        a2 = vdupq_n_f32(A[2 * lda + k + 3]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        a3 = vdupq_n_f32(A[3 * lda + k + 3]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);

        // 预取下一个K迭代的数据
        __builtin_prefetch(&A[0 * lda + k + 8], 0, 3);
        __builtin_prefetch(&B[(k + 8) * ldb], 0, 3);
    }

    // 处理剩余的K
    for (; k < K; ++k) {
        float32x4_t a0 = vdupq_n_f32(A[0 * lda + k]);
        float32x4_t b0 = vld1q_f32(&B[k * ldb + 0]);
        float32x4_t b1 = vld1q_f32(&B[k * ldb + 4]);

        c0 = vmlaq_f32(c0, a0, b0);
        c1 = vmlaq_f32(c1, a0, b1);

        float32x4_t a1 = vdupq_n_f32(A[1 * lda + k]);
        c2 = vmlaq_f32(c2, a1, b0);
        c3 = vmlaq_f32(c3, a1, b1);

        float32x4_t a2 = vdupq_n_f32(A[2 * lda + k]);
        c4 = vmlaq_f32(c4, a2, b0);
        c5 = vmlaq_f32(c5, a2, b1);

        float32x4_t a3 = vdupq_n_f32(A[3 * lda + k]);
        c6 = vmlaq_f32(c6, a3, b0);
        c7 = vmlaq_f32(c7, a3, b1);
    }

    // 存储结果
    vst1q_f32(&C[0 * ldc + 0], c0);
    vst1q_f32(&C[0 * ldc + 4], c1);
    vst1q_f32(&C[1 * ldc + 0], c2);
    vst1q_f32(&C[1 * ldc + 4], c3);
    vst1q_f32(&C[2 * ldc + 0], c4);
    vst1q_f32(&C[2 * ldc + 4], c5);
    vst1q_f32(&C[3 * ldc + 0], c6);
    vst1q_f32(&C[3 * ldc + 4], c7);
}

/**
 * @brief NEON 4x4微内核 (处理边界)
 */
void gemm_micro_kernel_4x4(
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int K)
{
    float32x4_t c0 = vld1q_f32(&C[0 * ldc + 0]);
    float32x4_t c1 = vld1q_f32(&C[1 * ldc + 0]);
    float32x4_t c2 = vld1q_f32(&C[2 * ldc + 0]);
    float32x4_t c3 = vld1q_f32(&C[3 * ldc + 0]);

    for (int k = 0; k < K; ++k) {
        float32x4_t a0 = vdupq_n_f32(A[0 * lda + k]);
        float32x4_t a1 = vdupq_n_f32(A[1 * lda + k]);
        float32x4_t a2 = vdupq_n_f32(A[2 * lda + k]);
        float32x4_t a3 = vdupq_n_f32(A[3 * lda + k]);

        float32x4_t b = vld1q_f32(&B[k * ldb + 0]);

        c0 = vmlaq_f32(c0, a0, b);
        c1 = vmlaq_f32(c1, a1, b);
        c2 = vmlaq_f32(c2, a2, b);
        c3 = vmlaq_f32(c3, a3, b);
    }

    vst1q_f32(&C[0 * ldc + 0], c0);
    vst1q_f32(&C[1 * ldc + 0], c1);
    vst1q_f32(&C[2 * ldc + 0], c2);
    vst1q_f32(&C[3 * ldc + 0], c3);
}

#endif // MICROFLOW_HAS_NEON

//==========================================================================
// 主GEMM函数 (带分块)
//==========================================================================

void gemm_neon(const Tensor& A, const Tensor& B, Tensor& C,
               const GEMMConfig& config)
{
    int M = A.shapes()[0];
    int K = A.shapes()[1];
    int N = B.shapes()[1];

    const float* ptr_A = A.raw_ptr();
    const float* ptr_B = B.raw_ptr();
    float* ptr_C = C.raw_ptr();

    // 清零C矩阵
    std::memset(ptr_C, 0, M * N * sizeof(float));

#ifdef MICROFLOW_HAS_NEON
    // 分块参数
    const int mc = config.mc;  // M分块
    const int nc = config.nc;  // N分块
    const int kc = config.kc;  // K分块
    const int mr = config.mr;  // 微内核M
    const int nr = config.nr;  // 微内核N

    // M维度分块 (OpenMP并行)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < M; i += mc) {
        int M_cur = std::min(mc, M - i);

        // K维度分块
        for (int p = 0; p < K; p += kc) {
            int K_cur = std::min(kc, K - p);

            // N维度分块
            for (int j = 0; j < N; j += nc) {
                int N_cur = std::min(nc, N - j);

                // 微内核分块
                for (int ii = 0; ii < M_cur; ii += mr) {
                    int M_cur_cur = std::min(mr, M_cur - ii);

                    for (int jj = 0; jj < N_cur; jj += nr) {
                        int N_cur_cur = std::min(nr, N_cur - jj);

                        // 调用微内核
                        if (M_cur_cur == 4 && N_cur_cur == 8) {
                            gemm_micro_kernel_4x8(
                                &ptr_A[(i + ii) * K + p], K,
                                &ptr_B[p * N + j + jj], N,
                                &ptr_C[(i + ii) * N + j + jj], N,
                                K_cur
                            );
                        } else if (M_cur_cur == 4 && N_cur_cur == 4) {
                            gemm_micro_kernel_4x4(
                                &ptr_A[(i + ii) * K + p], K,
                                &ptr_B[p * N + j + jj], N,
                                &ptr_C[(i + ii) * N + j + jj], N,
                                K_cur
                            );
                        } else {
                            // 边界情况使用标量代码
                            for (int iii = 0; iii < M_cur_cur; ++iii) {
                                for (int jjj = 0; jjj < N_cur_cur; ++jjj) {
                                    float sum = ptr_C[(i + ii + iii) * N + j + jj + jjj];
                                    for (int k = 0; k < K_cur; ++k) {
                                        sum += ptr_A[(i + ii + iii) * K + p + k] *
                                               ptr_B[(p + k) * N + j + jj + jjj];
                                    }
                                    ptr_C[(i + ii + iii) * N + j + jj + jjj] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#else
    // 回退到OpenMP版本
    gemm_omp(A, B, C);
#endif
}

//==========================================================================
// 通用接口
//==========================================================================

void gemm(const Tensor& A, const Tensor& B, Tensor& C,
          const GEMMConfig& config)
{
    int M = A.shapes()[0];
    int N = B.shapes()[1];
    int K = A.shapes()[1];

    // 自动选择实现
    GEMMImpl impl = select_best_implementation(M, N, K);

    switch (impl) {
        case GEMMImpl::kNaive:
            gemm_naive(A, B, C);
            break;
        case GEMMImpl::kOpenMP:
            gemm_omp(A, B, C);
            break;
        case GEMMImpl::kNEON:
            gemm_neon(A, B, C, config);
            break;
        default:
            gemm_neon(A, B, C, config);
            break;
    }
}

void gemm_transpose(const Tensor& A, const Tensor& B, Tensor& C,
                    bool transpose_A, bool transpose_B,
                    const GEMMConfig& config)
{
    // 简化实现: 创建转置视图
    // 完整实现应该调整访问模式而非显式转置

    if (transpose_A) {
        Tensor A_T = A.transpose(0, 1);
        if (transpose_B) {
            Tensor B_T = B.transpose(0, 1);
            gemm(A_T, B_T, C, config);
        } else {
            gemm(A_T, B, C, config);
        }
    } else {
        if (transpose_B) {
            Tensor B_T = B.transpose(0, 1);
            gemm(A, B_T, C, config);
        } else {
            gemm(A, B, C, config);
        }
    }
}

void batch_gemm(int batch, const Tensor* A, const Tensor* B, Tensor* C,
                const GEMMConfig& config)
{
    #pragma omp parallel for
    for (int i = 0; i < batch; ++i) {
        gemm(A[i], B[i], C[i], config);
    }
}

//==========================================================================
// 工具函数
//==========================================================================

bool verify_gemm(const Tensor& C, const Tensor& C_ref, float eps) {
    if (C.shapes() != C_ref.shapes()) {
        return false;
    }

    const float* ptr_c = C.raw_ptr();
    const float* ptr_ref = C_ref.raw_ptr();
    uint32_t size = C.size();

    for (uint32_t i = 0; i < size; ++i) {
        float diff = std::abs(ptr_c[i] - ptr_ref[i]);
        if (diff > eps) {
            return false;
        }
    }
    return true;
}

GEMMStats benchmark_gemm(int M, int N, int K,
                         std::function<void()> gemm_func,
                         int iterations)
{
    GEMMStats stats;
    stats.total_ops = 2LL * M * N * K * iterations;  // 2次浮点运算(乘+加)

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        gemm_func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    stats.time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    stats.gflops = (stats.total_ops / stats.time_ms / 1e6);

    return stats;
}

void sgemm(char layout, char transA, char transB,
           int M, int N, int K,
           float alpha,
           const float* A, int lda,
           const float* B, int ldb,
           float beta,
           float* C, int ldc)
{
    // 简化实现: 仅支持Row-Major, 无转置
    // 完整实现需要处理转置和alpha/beta系数

    // 创建Tensor包装
    Tensor A_tensor(std::vector<uint32_t>{static_cast<uint32_t>(M), static_cast<uint32_t>(K)},
                   const_cast<float*>(A));
    Tensor B_tensor(std::vector<uint32_t>{static_cast<uint32_t>(K), static_cast<uint32_t>(N)},
                   const_cast<float*>(B));
    Tensor C_tensor(std::vector<uint32_t>{static_cast<uint32_t>(M), static_cast<uint32_t>(N)},
                   C);

    gemm(A_tensor, B_tensor, C_tensor);

    // 应用alpha和beta
    if (alpha != 1.0f || beta != 0.0f) {
        float* ptr_C = C_tensor.raw_ptr();
        for (int i = 0; i < M * N; ++i) {
            ptr_C[i] = alpha * ptr_C[i];
        }
    }
}

} // namespace microflow
