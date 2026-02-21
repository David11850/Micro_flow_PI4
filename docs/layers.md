# 层操作模块 (Layers)

## 概述

本模块实现了深度学习中常用的层操作，包括激活函数、池化、归一化等，全部针对树莓派4进行了优化。

---

## 激活函数

### ReLU (Rectified Linear Unit)

#### 数学定义

```
f(x) = max(0, x)
```

#### NEON优化实现

```cpp
void relu(Tensor& input) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

    float32x4_t zero = vdupq_n_f32(0.0f);

    // 4x展开处理
    for (int i = 0; i <= size - 16; i += 16) {
        float32x4_t v0 = vld1q_f32(&ptr[i + 0]);
        float32x4_t v1 = vld1q_f32(&ptr[i + 4]);
        float32x4_t v2 = vld1q_f32(&ptr[i + 8]);
        float32x4_t v3 = vld1q_f32(&ptr[i + 12]);

        // 向量化max操作
        v0 = vmaxq_f32(v0, zero);
        v1 = vmaxq_f32(v1, zero);
        v2 = vmaxq_f32(v2, zero);
        v3 = vmaxq_f32(v3, zero);

        vst1q_f32(&ptr[i + 0], v0);
        vst1q_f32(&ptr[i + 4], v1);
        vst1q_f32(&ptr[i + 8], v2);
        vst1q_f32(&ptr[i + 12], v3);
    }
}
```

#### 性能对比

| 实现 | 1M元素耗时 | 带宽 |
|-----|-----------|------|
| 标量OMP | 2.1ms | 1.9GB/s |
| NEON优化 | 0.5ms | 8.0GB/s |

**加速比**: 4.2x

### ReLU6

```
f(x) = min(max(0, x), 6)
```

**用途**: MobileNetV2/V3，量化友好

**NEON实现**:
```cpp
float32x4_t zero = vdupq_n_f32(0.0f);
float32x4_t six = vdupq_n_f32(6.0f);

float32x4_t v = vld1q_f32(&ptr[i]);
v = vminq_f32(v, six);   // 上限6
v = vmaxq_f32(v, zero);  // 下限0
```

### Leaky ReLU

```
f(x) = x >= 0 ? x : alpha * x
```

**NEON无分支实现**:
```cpp
float32x4_t alpha_vec = vdupq_n_f32(alpha);
float32x4_t v = vld1q_f32(&ptr[i]);

// 计算负值情况
float32x4_t neg = vmulq_f32(v, alpha_vec);

// 生成mask (v < 0)
float32x4_t mask = vcltq_f32(v, zero);

// 选择: mask ? neg : v
v = vbslq_f32(mask, neg, v);
```

### GELU (Gaussian Error Linear Unit)

**用途**: Transformer (BERT, GPT)

**近似公式**:
```
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**性能**: 比精确实现快10倍，误差<0.001

---

## 池化层

### 最大池化 (MaxPool2d)

#### 算法

```
for each output position (h, w):
    max_val = -∞
    for kh in 0..kernel_size-1:
        for kw in 0..kernel_size-1:
            val = input[h*stride+kh, w*stride+kw]
            if val > max_val:
                max_val = val
    output[h, w] = max_val
```

#### 优化策略

1. **2x2池化特殊化**: 展开内层循环
2. **3x3池化特殊化**: 使用9元素比较
3. **OpenMP并行**: 通道间独立

#### 性能 (树莓派4)

| 配置 | 输入尺寸 | 输出尺寸 | 耗时 |
|-----|---------|---------|------|
| 2x2, s=2 | [64,56,56] | [64,28,28] | 1.8ms |
| 3x3, s=2 | [64,56,56] | [64,27,27] | 2.3ms |
| 全局 | [64,7,7] | [64,1,1] | 0.3ms |

### 平均池化 (AvgPool2d)

```
output[h, w] = mean(input[h*stride:h*stride+k, w*stride:w*stride+k])
```

**优化**: 延迟除法，最后除以kernel_size²

### 自适应池化

```
adaptive_avg_pool2d(input, [H_out, W_out])
```

**应用**: ResNet分类头，将任意尺寸特征图池化到固定大小

---

## 归一化层

### BatchNorm

#### 推理模式

```
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

#### 优化: 与卷积融合

```
原始:
  conv_output = conv(input, weight)
  bn_output = batch_norm(conv_output)
  relu_output = relu(bn_output)

融合:
  weight' = weight * gamma / sqrt(var + eps)
  bias' = (bias - mean) / sqrt(var + eps) * gamma + beta

  output = relu(conv(input, weight') + bias')
```

**性能**: 节省2次内存遍历

### LayerNorm

```
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

**区别**: 对每个样本归一化，而非批次

**应用**: Transformer、RNN

### GroupNorm

```
# 将通道分组，每组独立归一化
groups = 8  # 常见设置
for g in 0..groups-1:
    channels = g * (C//groups) : (g+1) * (C//groups)
    y[channels] = layer_norm(x[channels])
```

**优势**: 不依赖batch大小，适合小batch训练

---

## 全连接层

### Linear

```
Y = X * W^T + bias
```

**实现**: 直接调用优化的GEMM

### Linear + ReLU 融合

```cpp
void linear_relu(input, weight, bias, output) {
    linear(input, weight, bias, output);
    relu(output);  // 就地操作
}
```

---

## 上采样和下采样

### 双线性插值上采样

```
for each output position (h, w):
    h_in = h / scale
    w_in = w / scale

    h0 = floor(h_in), h1 = h0 + 1
    w0 = floor(w_in), w1 = w0 + 1

    dh = h_in - h0
    dw = w_in - w0

    v00 = input[h0, w0], v01 = input[h0, w1]
    v10 = input[h1, w0], v11 = input[h1, w1]

    output[h, w] = (v00*(1-dw) + v01*dw) * (1-dh) +
                   (v10*(1-dw) + v11*dw) * dh
```

**应用**: 分割网络 (FCN, U-Net)

### 最近邻上采样

```
output[h, w] = input[h/scale, w/scale]
```

**优势**: 无计算，快3倍

---

## 使用示例

### 基本激活

```cpp
Tensor x({128, 28, 28});

// ReLU
relu(x);

// ReLU6
relu6(x);

// Leaky ReLU
leaky_relu(x, 0.01f);
```

### 池化

```cpp
Tensor input({64, 56, 56});
Tensor output({64, 28, 28});

// 2x2最大池化, stride=2
max_pool2d(input, output, 2, 2);

// 3x3平均池化, stride=2, padding=1
Tensor output2({64, 28, 28});
avg_pool2d(input, output2, 3, 2, 1);

// 全局平均池化
Tensor glob({64, 1, 1});
global_avg_pool2d(input, glob);
```

### BatchNorm融合

```cpp
// 方式1: 分开调用
conv2d(input, kernel, Tensor(), feature, params);
batch_norm(feature, mean, var, gamma, beta);
relu(feature);

// 方式2: 融合版本 (推荐)
conv2d_bn_relu(input, kernel, mean, var, gamma, beta,
               feature, params);
```

### 线性层

```cpp
Tensor input({1, 784});      // [batch, features_in]
Tensor weight({10, 784});    // [features_out, features_in]
Tensor bias({10});
Tensor output({1, 10});      // [batch, features_out]

linear(input, weight, bias, output);

// 或带ReLU
linear_relu(input, weight, bias, output);
```

### Softmax

```cpp
Tensor logits({1, 1000});    // 分类分数
softmax(logits, 1);          // 沿维度1计算

// 结果是概率分布
float* probs = logits.raw_ptr();
```

---

## 性能优化建议

### 1. 融合操作

```cpp
// ❌ 不好
conv2d(input, kernel, temp, params);
batch_norm(temp, ...);
relu(temp);

// ✅ 好
conv2d_bn_relu(input, kernel, ..., temp, params);
```

### 2. 就地操作

```cpp
// ❌ 不好
Tensor output = relu(input.clone());

// ✅ 好
Tensor output = input;  // 共享数据
relu(output);           // 就地修改
```

### 3. 预分配输出

```cpp
// ❌ 不好: 每次分配
for (int i = 0; i < 100; ++i) {
    Tensor out = max_pool(input);
}

// ✅ 好: 复用内存
Tensor out({64, 28, 28});
for (int i = 0; i < 100; ++i) {
    max_pool2d(input, out, 2, 2);
}
```

---

## 性能基准 (树莓派4)

| 操作 | 输入尺寸 | 耗时 | 吞吐量 |
|-----|---------|------|--------|
| ReLU | 1M 元素 | 0.5ms | 2 GFLOPS |
| MaxPool 2x2 | [64,56,56] | 1.8ms | - |
| AvgPool 3x3 | [64,56,56] | 2.1ms | - |
| GlobalAvgPool | [64,7,7] | 0.3ms | - |
| Softmax | [1,1000] | 0.05ms | - |
| Linear 784->10 | [1,784] | 0.08ms | - |

---

## 调试技巧

### 可视化激活分布

```cpp
void print_activation_stats(const Tensor& x) {
    const float* ptr = x.raw_ptr();
    uint32_t size = x.size();

    float sum = 0, sum_sq = 0;
    float min_val = ptr[0], max_val = ptr[0];
    int zeros = 0;

    for (uint32_t i = 0; i < size; ++i) {
        sum += ptr[i];
        sum_sq += ptr[i] * ptr[i];
        min_val = std::min(min_val, ptr[i]);
        max_val = std::max(max_val, ptr[i]);
        if (ptr[i] == 0) ++zeros;
    }

    float mean = sum / size;
    float var = (sum_sq / size) - (mean * mean);

    std::cout << "Mean: " << mean << "\n";
    std::cout << "Std: " << std::sqrt(var) << "\n";
    std::cout << "Min: " << min_val << ", Max: " << max_val << "\n";
    std::cout << "Sparsity: " << (100.0 * zeros / size) << "%\n";
}
```

### 检测数值问题

```cpp
bool has_nan_or_inf(const Tensor& x) {
    const float* ptr = x.raw_ptr();
    for (uint32_t i = 0; i < x.size(); ++i) {
        if (std::isnan(ptr[i]) || std::isinf(ptr[i])) {
            return true;
        }
    }
    return false;
}
```

---

## 未来优化方向

1. **SIMD查表**: Sigmoid/Tanh等复杂函数
2. **混合精度**: FP16计算 + FP32累加
3. **批量融合**: 多个操作融合为单个kernel
4. **动态调度**: 根据输入大小选择最优算法
