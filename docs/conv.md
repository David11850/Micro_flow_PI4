# 卷积优化模块 (Convolution)

## 概述

卷积是CNN推理的核心计算单元。本模块针对树莓派4实现了多种卷积算法，并根据输入特征自动选择最优实现。

---

## 卷积算法对比

### 1. Im2Col + GEMM

#### 原理

将卷积转换为矩阵乘法：

```
输入: [C, H, W] -> 展开为 [C*K*K, H_out*W_out]
卷积核: [F, C, K, K] -> 展开为 [F, C*K*K]
输出: [F, H_out*W_out] = 卷积核矩阵 × 输入矩阵
```

#### 内存变换示例

```
输入图像 [1, 5, 5] (1通道, 5x5):
[1,  2,  3,  4,  5]
[6,  7,  8,  9, 10]
[11, 12, 13, 14, 15]
[16, 17, 18, 19, 20]
[21, 22, 23, 24, 25]

3x3卷积核, stride=1, padding=0:
输出尺寸: 3x3

im2col变换后 [1*3*3=9, 3*3=9]:
列0 (位置0,0): [1, 2, 3, 6, 7, 8, 11, 12, 13]
列1 (位置0,1): [2, 3, 4, 7, 8, 9, 12, 13, 14]
...
```

#### 优势与劣势

| 方面 | 优势 | 劣势 |
|-----|------|------|
| 性能 | 复用GEMM优化 | 额外内存开销 |
| 通用性 | 支持任意参数 | 小卷积核效率低 |
| 实现 | 代码复用 | 内存访问模式复杂 |

#### 适用场景

- 大卷积核 (5x5及以上)
- 输入通道多 (> 64)
- 批处理场景

### 2. 直接卷积

#### 原理

直接在输入上进行滑动窗口计算：

```
for f in 输出通道:
  for h_out in 输出高度:
    for w_out in 输出宽度:
      sum = 0
      for c in 输入通道:
        for kh in 卷积核高度:
          for kw in 卷积核宽度:
            sum += input[c, h_in+kh, w_in+kw] *
                   kernel[f, c, kh, kw]
      output[f, h_out, w_out] = sum
```

#### NEON优化

```cpp
// 4个输出位置并行计算
for (w_out = 0; w_out <= W_out - 4; w_out += 4) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    for (int c = 0; c < C_in; ++c) {
        for (int kh = 0; kh < 3; ++kh) {
            // 加载4个输入值
            float in_temp[4];
            for (int i = 0; i < 4; ++i) {
                in_temp[i] = input[c, h_in+kh, w_out+i+kw];
            }
            float32x4_t in_vals = vld1q_f32(in_temp);

            // 广播卷积核权重
            float k_val = kernel[f, c, kh, kw];
            float32x4_t k_vec = vdupq_n_f32(k_val);

            // FMA
            sum = vmlaq_f32(sum, in_vals, k_vec);
        }
    }

    // 存储4个结果
    vst1q_f32(&output[f, h_out, w_out], sum);
}
```

#### 优势与劣势

| 方面 | 优势 | 劣势 |
|-----|------|------|
| 内存 | 无额外开销 | 难以向量化 |
| 缓存 | 局部性好 | 输入通道多时慢 |
| 小核 | 3x3效率高 | 实现复杂 |

#### 适用场景

- 3x3卷积
- 输入通道少 (< 64)
- 内存受限场景

### 3. Winograd卷积

#### 原理

Winograd算法通过变换减少乘法次数：

```
F(2×2, 3×3): 输入2×2块, 卷积核3×3, 输出2×2块

直接卷积: 2×2×3×3 = 36次乘法
Winograd: 16次乘法 (减少56%)
```

#### 变换流程

```
1. 输入变换: d = B^T × input × B
2. 卷积核变换: g = G × kernel × G^T
3. 元素乘法: m = g ⊙ d
4. 输出变换: output = A^T × m × A

其中 A, B, G 是预先计算的变换矩阵
```

#### 优势与劣势

| 方面 | 优势 | 劣势 |
|-----|------|------|
| 计算量 | 乘法少 | 加法多 |
| 精度 | 数值稳定性 | 融合困难 |
| 硬件 | 不友好 | 需要额外变换 |

#### 适用场景

- 3x3卷积, stride=1
- 输入通道少 (< 64)
- ARM平台可能不如直接卷积

### 4. Depthwise卷积

#### 原理

每个输入通道独立卷积，不跨通道计算：

```
标准卷积: F * C_in * K * K * H_out * W_out
Depthwise: C_in * K * K * H_out * W_out (少F倍计算)
```

#### NEON优化

```cpp
// 同时处理4个通道
for (int c = 0; c <= C_in - 4; c += 4) {
    // 加载4个通道的3x3卷积核
    float32x4_t k00 = {k[c+0,0,0], k[c+1,0,0], k[c+2,0,0], k[c+3,0,0]};
    float32x4_t k01 = {k[c+0,0,1], k[c+1,0,1], k[c+2,0,1], k[c+3,0,1]};
    // ...

    // 加载4个通道的输入
    float32x4_t in0 = {in[c+0,h,w], in[c+1,h,w], in[c+2,h,w], in[c+3,h,w]};
    // ...

    // 计算并累加
    sum0 = vmlaq_f32(sum0, in0, k00);
    // ...
}
```

#### 优势与劣势

| 方面 | 优势 | 劣势 |
|-----|------|------|
| 计算量 | 极低 | 通道间无交互 |
| 参数量 | 少 | 表达能力弱 |
| 移动端 | MobileNet标配 | 通常需配合Pointwise |

#### 适用场景

- MobileNet系列网络
- 轻量级模型
- 边缘设备

### 5. 逐点卷积 (Pointwise/1x1)

#### 原理

1x1卷积本质上是通道间的线性组合：

```
等价于: 矩阵乘法
输入: [H*W, C_in]
权重: [C_out, C_in]
输出: [H*W, C_out]
```

#### 优化

```cpp
// 直接使用GEMM
gemm(
    input_reshaped,   // [H*W, C_in]
    kernel_reshaped,  // [C_out, C_in]
    output            // [C_out, H*W]
);
```

#### 适用场景

- ResNet的bottleneck
- MobileNet的扩展层
- 通道数变换

---

## 性能对比 (树莓派4实测)

### 测试配置

- CPU: Cortex-A72 @ 1.5GHz × 4
- 输入: [64, 28, 28]
- 卷积核: [128, 64, 3, 3]

### 结果

| 算法 | 时间 (ms) | GFLOPS | 内存开销 |
|-----|----------|--------|---------|
| Im2Col+GEMM | 12.5 | 8.5 | 2.1 MB |
| 直接卷积 | 18.2 | 5.8 | 0 MB |
| 直接卷积(NEON) | 8.3 | 12.8 | 0 MB |
| Winograd | 9.1 | 11.6 | 0.5 MB |

### 结论

对于树莓派4:
- **3x3卷积**: 直接卷积(NEON) 最优
- **大卷积核**: Im2Col+GEMM 最优
- **Depthwise**: NEON优化版本显著提升

---

## 自动选择策略

```cpp
ConvImpl select_optimal_conv_impl(input, kernel, params) {
    if (kernel_size == 1) {
        return kPointwise;    // 1x1用GEMM
    }

    if (groups == C_in && C_in == F) {
        return kDepthwise;    // Depthwise卷积
    }

    if (kernel_size == 3 && C_in <= 64) {
        return kDirectNEON;   // 3x3少通道用直接卷积
    }

    return kIm2ColGEMM;       // 默认im2col
}
```

---

## 使用示例

### 基本卷积

```cpp
#include "microflow/conv.hpp"

using namespace microflow;

// 输入: [1, 28, 28] (灰度图像)
Tensor input({1, 28, 28});

// 卷积核: [8, 1, 3, 3] (8个3x3滤波器)
Tensor kernel({8, 1, 3, 3});

// 输出: [8, 28, 28]
Tensor output({8, 28, 28});

// 参数
Conv2DParams params;
params.kernel_size = 3;
params.stride = 1;
params.padding = 1;

// 执行卷积
conv2d(input, kernel, Tensor(), output, params);
```

### 带偏置和激活

```cpp
// 偏置: [8]
Tensor bias({8});
bias.fill(0.1f);

// 卷积 + 偏置
conv2d(input, kernel, bias, output, params);

// 或使用融合版本
conv2d_relu(input, kernel, output, params);
```

### Depthwise卷积

```cpp
// 输入: [32, 14, 14]
Tensor input({32, 14, 14});

// Depthwise卷积核: [32, 1, 3, 3]
// 每个输入通道一个滤波器
Tensor kernel_dw({32, 1, 3, 3});

Tensor output({32, 14, 14});

Conv2DParams params(3, 1, 1, 1, 32);  // groups=32

conv2d(input, kernel_dw, Tensor(), output, params);
```

### MobileNet风格的层

```cpp
// Depthwise + Pointwise = 倒残差结构

// 1. Depthwise 3x3
Tensor dw_out = Tensor({32, 14, 14});
conv2d(input, kernel_dw, Tensor(), dw_out,
       Conv2DParams(3, 1, 1, 1, 32));

// 2. Pointwise 1x1 (扩展通道)
Tensor pw_kernel({64, 32, 1, 1});
Tensor pw_out({64, 14, 14});
conv2d(dw_out, pw_kernel, Tensor(), pw_out,
       Conv2DParams(1, 1, 0));
```

---

## 内存优化

### 工作空间计算

```cpp
// 计算im2col需要的额外内存
size_t workspace_size = compute_conv_workspace_size(
    input, kernel, params
);

// 预分配工作空间
std::vector<float> workspace(workspace_size / sizeof(float));

// 使用工作空间
conv2d(input, kernel, Tensor(), output, params,
       workspace.data());
```

### 内存复用

```cpp
class ConvLayer {
    Tensor input_buffer;
    Tensor output_buffer;
    std::vector<float> workspace;

public:
    void forward(const Tensor& input, Tensor& output) {
        // 复用预分配的内存
        // ...
    }
};
```

---

## 调试技巧

### 验证正确性

```cpp
// 使用im2col方法作为参考
Tensor output_ref({F, H_out, W_out});
Tensor output_opt({F, H_out, W_out});

// 参考实现
Tensor col({C_in*K*K, H_out*W_out});
im2col(input, col, params);
Tensor k_mat({F, C_in*K*K}, kernel.raw_ptr());
Tensor o_mat({F, H_out*W_out}, output_ref.raw_ptr());
gemm_naive(k_mat, col, o_mat);

// 待验证实现
conv2d_direct_neon(input, kernel, output_opt, params);

// 对比
float max_diff = 0.0f;
for (int i = 0; i < output_ref.size(); ++i) {
    float diff = std::abs(
        output_ref.raw_ptr()[i] - output_opt.raw_ptr()[i]
    );
    max_diff = std::max(max_diff, diff);
}

std::cout << "Max diff: " << max_diff << "\n";
```

### 可视化特征图

```cpp
void visualize_feature_map(const Tensor& feature_map) {
    // 假设 [C, H, W] 格式
    int C = feature_map.shapes()[0];
    int H = feature_map.shapes()[1];
    int W = feature_map.shapes()[2];

    for (int c = 0; c < std::min(C, 16); ++c) {  // 最多显示16通道
        std::cout << "Channel " << c << ":\n";
        const float* ptr = feature_map.raw_ptr() + c * H * W;

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                // 简单的ASCII可视化
                float val = ptr[h * W + w];
                char ch = ' ';
                if (val > 0.1) ch = '.';
                if (val > 0.3) ch = ':';
                if (val > 0.5) ch = 'o';
                if (val > 0.7) ch = 'O';
                if (val > 0.9) ch = '@';
                std::cout << ch;
            }
            std::cout << "\n";
        }
    }
}
```

---

## 性能优化建议

### 1. 通道排列优化

```cpp
// 将通道数调整为4的倍数 (NEON友好)
// 原始: [63, H, W]
// 优化: [64, H, W] (填充1个零通道)
```

### 2. 输出尺寸对齐

```cpp
// 调整padding使输出尺寸为4的倍数
// 便于向量化最后一维
```

### 3. 权重预打包

```cpp
// 将权重转换为分块格式
// 减少GEMM中的数据重排
void pack_conv_weights(const Tensor& kernel, Tensor& packed);
```

### 4. 融合操作

```cpp
// Conv + BN + ReLU -> 单次遍历
// 避免中间结果的写入和读取
```

---

## 未来优化方向

1. **ARMv8.2 FP16**: 使用半精度浮点提升吞吐量
2. **Dot Product指令**: ARMv8.2新增的矩阵乘加速指令
3. **SVE (可变长度向量)**: ARMv9的可伸缩向量扩展
4. **Mali GPU**: 树莓派4的VideoCore VI GPU加速
5. **NPU**: 未来的树莓派可能配备专用NPU
