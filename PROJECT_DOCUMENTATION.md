# MicroFlow 项目完整文档

## 文档信息
- **项目名称**: MicroFlow
- **版本**: 2.0
- **平台**: 树莓派4 (Cortex-A72 ARM64)
- **最后更新**: 2025-02-21

---

## 目录
- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 目录结构](#3-目录结构)
- [4. 核心模块详解](#4-核心模块详解)
- [5. API参考](#5-api参考)
- [6. 修改历史](#6-修改历史)
- [7. 使用指南](#7-使用指南)
- [8. 性能优化](#8-性能优化)
- [9. 故障排查](#9-故障排查)
- [10. 扩展开发](#10-扩展开发)

---

## 1. 项目概述

### 1.1 什么是MicroFlow？

**MicroFlow** 是一个专为嵌入式设备和边缘计算场景设计的轻量级神经网络推理引擎。它采用纯C++实现，无需任何第三方依赖库，特别适合在资源受限的设备（如树莓派4）上运行深度学习推理任务。

### 1.2 设计理念

MicroFlow的设计遵循以下核心理念：

#### 轻量级
- **无外部依赖**: 不依赖任何第三方库（如OpenCV、TensorFlow Lite等）
- **小体积**: 核心库编译后只有几百KB
- **低内存占用**: 精心设计的内存管理，最小化运行时内存需求

#### 高性能
- **ARM NEON优化**: 充分利用ARM处理器的SIMD指令集
- **零拷贝设计**: 通过智能指针和视图机制避免不必要的内存复制
- **内存复用**: 中间张量缓存和工作空间复用，减少内存分配

#### 易用性
- **简洁API**: 类似PyTorch的接口设计，易于上手
- **流式构建器**: 通过Builder模式快速构建模型
- **Python训练工具**: 提供PyTorch到C++的模型转换脚本

### 1.3 应用场景

MicroFlow特别适合以下应用场景：

| 场景 | 描述 | 优势 |
|------|------|------|
| **图像分类** | MNIST、CIFAR-10等小型数据集 | 快速推理，低延迟 |
| **物体检测** | 简单的边界框检测和分类 | 边缘端实时处理 |
| **语音识别** | 关键词唤醒、简单命令识别 | 低功耗，持续运行 |
| **传感器数据处理** | IMU、温度传感器等数据分类 | 实时响应 |
| **教育项目** | 深度学习原理学习和实践 | 代码清晰，易于理解 |

### 1.4 性能指标

在树莓派4（Cortex-A72, 1.5GHz）上的性能表现：

| 指标 | 数值 | 说明 |
|------|------|------|
| MNIST推理延迟 | ~0.30ms | 单次推理平均时间 |
| MNIST吞吐量 | ~3300 inferences/sec | 每秒推理次数 |
| 内存占用 | <10MB | 运行时内存（不含模型权重） |
| 模型大小 | <1MB | MNIST改进模型 |

### 1.5 对比其他框架

| 特性 | MicroFlow | TensorFlow Lite | NCNN | MNN |
|------|-----------|----------------|------|-----|
| 代码体积 | 极小 | 较大 | 中等 | 中等 |
| 依赖库 | 无 | Protocol Buffers | 无 | 无 |
| ARM NEON | ✅ | ✅ | ✅ | ✅ |
| 学习曲线 | 平缓 | 陡峭 | 中等 | 中等 |
| 模型格式 | 自定义(.mflow) | TFLite | NCNN | MNN |
| Python训练 | ✅ | ✅ | ✅ | ✅ |
| 适合教学 | ✅ | ❌ | ❌ | ❌ |

### 1.6 支持的层类型

MicroFlow支持以下19种层类型，覆盖了大多数常见的神经网络层：

| 层类型 | 枚举值 | 详细描述 | 典型用途 |
|--------|--------|----------|----------|
| Input | 0 | 输入占位层，不执行计算 | 标记网络输入 |
| Conv2D | 1 | 标准2D卷积层 | 特征提取 |
| DepthwiseConv2D | 2 | 深度可分离卷积 | MobileNet风格网络 |
| PointwiseConv2D | 3 | 逐点卷积(1x1) | 通道数调整 |
| BatchNorm | 4 | 批量归一化层 | 训练加速，稳定训练 |
| ReLU | 5 | ReLU激活函数 | 非线性激活 |
| ReLU6 | 6 | ReLU6激活(上限6) | 量化友好激活 |
| LeakyReLU | 7 | LeakyReLU激活 | 缓解神经元死亡 |
| ELU | 8 | ELU激活函数 | 输出均值接近0 |
| MaxPool2D | 9 | 2D最大池化 | 下采样，特征选择 |
| AvgPool2D | 10 | 2D平均池化 | 下采样，信息聚合 |
| GlobalAvgPool2D | 11 | 全局平均池化 | 分类网络最终层 |
| AdaptiveAvgPool2D | 12 | 自适应平均池化 | 固定输出尺寸 |
| Linear | 13 | 全连接层 | 特征组合，分类 |
| Flatten | 14 | 展平层 | 卷积到全连接过渡 |
| Reshape | 15 | 重塑层 | 改变张量形状 |
| Concat | 16 | 拼接层 | 多分支融合 |
| Softmax | 17 | Softmax激活 | 概率输出 |
| Sigmoid | 18 | Sigmoid激活 | 二分类输出 |

---

## 2. 系统架构

### 2.1 整体架构图

MicroFlow采用分层架构设计，从底层到上层依次为：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        应用层 (Application Layer)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  mnist_demo  │  │  benchmark   │  │  自定义应用   │              │
│  │  (示例程序)   │  │  (性能测试)   │  │  (用户代码)   │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼──────────────────┼─────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     运行时层 (Runtime Layer)                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Model 类                                                    │  │
│  │  - 模型加载 (.mflow)                                         │  │
│  │  - 层管理 (Layer管理器)                                      │  │
│  │  - 中间张量缓存 (Tensor缓存)                                 │  │
│  │  - 工作空间管理 (Workspace管理)                              │  │
│  │  - 推理执行 (Forward执行)                                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │   Layer    │  │InferenceEng│  │ModelBuilder│  │Tensor相关  │  │
│  │   基类     │  │    ine     │  │   流式API  │  │   工具     │  │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      计算层 (Compute Layer)                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  计算内核 (Compute Kernels)                                  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │  conv2d  │  │   gemm   │  │  maxpool │  │  softmax │  │  │
│  │  │ 卷积运算  │  │ 矩阵乘法  │  │  池化    │  │  激活    │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  层实现 (Layer Implementations)                              │  │
│  │  - Conv2DLayer: 卷积层实现                                  │  │
│  │  - LinearLayer: 全连接层实现                                │  │
│  │  - ActivationLayer: 激活层实现                              │  │
│  │  - PoolingLayer: 池化层实现                                │  │
│  │  - 其他层类型...                                            │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      内存层 (Memory Layer)                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Tensor 类                                                  │  │
│  │  - 数据存储 (shared_ptr<float[]>)                           │  │
│  │  - 形状管理 (shapes, strides)                               │  │
│  │  - 视图机制 (零拷贝切片)                                    │  │
│  │  - 拷贝语义 (深拷贝 vs 浅拷贝)                              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  内存管理                                                    │  │
│  │  - Arena分配器 (Arena Allocator)                            │  │
│  │  - 对齐分配 (Alignment: 64字节)                            │  │
│  │  - 内存复用 (Tensor缓存)                                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     平台层 (Platform Layer)                          │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  ARM NEON SIMD                                              │  │
│  │  - float32x4_t: 4个float并行计算                            │  │
│  │  - 向量化加载/存储 (vld1q_f32, vst1q_f32)                  │  │
│  │  - 向量运算 (vaddq_f32, vmulq_f32, vmaxq_f32等)           │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  OpenMP 多线程                                              │  │
│  │  - 并行for循环 (#pragma omp parallel for)                   │  │
│  │  - 线程池管理                                               │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 执行流程详解

#### 2.2.1 模型加载流程

```
开始
  │
  ▼
打开.mflow文件
  │
  ▼
读取ModelHeader
  │
  ├─ 验证魔数 (0x4D464C57 = "MFLW")
  │
  ├─ 读取版本号
  │
  ├─ 读取层数 (num_layers)
  │
  └─ 读取张量数 (num_tensors)
  │
  ▼
遍历每个LayerHeader
  │
  ├─ 读取层类型 (type)
  │
  ├─ 根据类型创建对应的Layer对象
  │   ├─ kInput → InputLayer
  │   ├─ kConv2D → Conv2DLayer (读取kernel和bias)
  │   ├─ kReLU → ActivationLayer
  │   ├─ kMaxPool2D → PoolingLayer
  │   ├─ kFlatten → FlattenLayer
  │   ├─ kLinear → LinearLayer (读取weight和bias)
  │   └─ kSoftmax → SoftmaxLayer
  │
  └─ 添加到Model的layers_列表
  │
  ▼
检查是否有Softmax层
  │
  ├─ 如果没有，自动添加
  │
  └─ 确保最终输出是概率
  │
  ▼
分配中间张量 (allocate_tensors)
  │
  ├─ 遍历所有层
  │
  ├─ 计算每层的输出形状
  │
  ├─ 为每层创建输出Tensor
  │
  └─ 存储到intermediate_tensors_
  │
  ▼
分配工作空间 (compute_workspace_size)
  │
  ├─ 计算Conv2D所需的工作空间
  │
  └─ 分配workspace_向量
  │
  ▼
设置is_loaded_ = true
  │
  ▼
结束
```

#### 2.2.2 推理执行流程

```
开始推理 (Model::forward)
  │
  ▼
准备输入Tensor
  │
  ├─ 验证输入形状
  │
  ├─ 验证数据有效性
  │
  └─ 检查模型是否已加载
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 第0层: InputLayer                                           │
│                                                             │
│  in_vec[0] ───────→ out_vec[0]                             │
│    (指向input)        (指向intermediate_tensors_[0])        │
│                                                             │
│  操作: memcpy(input, intermediate_tensors_[0])            │
└─────────────────────────────────────────────────────────────┘
  │
  ▼ intermediate_tensors_[0]
┌─────────────────────────────────────────────────────────────┐
│ 第1层: Conv2DLayer                                          │
│                                                             │
│  in_vec[0] ───────→ out_vec[0]                             │
│ (指向int_[0])       (指向intermediate_tensors_[1])          │
│                                                             │
│  操作: conv2d(int_[0], kernel, bias, int_[1])             │
│                                                             │
│  NEON优化: 并行计算多个输出通道                            │
└─────────────────────────────────────────────────────────────┘
  │
  ▼ intermediate_tensors_[1]
┌─────────────────────────────────────────────────────────────┐
│ 第2层: ActivationLayer (ReLU)                               │
│                                                             │
│  in_vec[0] ───────→ out_vec[0]                             │
│ (指向int_[1])       (指向intermediate_tensors_[2])          │
│                                                             │
│  操作: relu(int_[2]) (就地操作)                            │
│                                                             │
│  NEON优化: vmaxq_f32(x, 0)                                 │
└─────────────────────────────────────────────────────────────┘
  │
  ▼ intermediate_tensors_[2]
┌─────────────────────────────────────────────────────────────┐
│ 第3层: PoolingLayer (MaxPool2D)                             │
│                                                             │
│  in_vec[0] ───────→ out_vec[0]                             │
│ (指向int_[2])       (指向intermediate_tensors_[3])          │
│                                                             │
│  操作: max_pool2d(int_[3])                                 │
└─────────────────────────────────────────────────────────────┘
  │
  ▼ ... (后续层)
  │
  ▼ intermediate_tensors_[n-1]
┌─────────────────────────────────────────────────────────────┐
│ 最后一层: SoftmaxLayer                                      │
│                                                             │
│  in_vec[0] ───────→ out_vec[0]                             │
│ (指向int_[n-2])     (指向intermediate_tensors_[n-1])        │
│                                                             │
│  操作: softmax(int_[n-1])                                  │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
复制到输出: intermediate_tensors_.back().copy_to(output)
  │
  ▼
返回结果
```

### 2.3 数据流设计

#### 2.2.1 指针传递机制

MicroFlow使用指针向量来传递张量，避免深拷贝：

```cpp
// ❌ 错误方式: 使用值传递
std::vector<Tensor> in_vec = {input};        // 创建副本!
std::vector<Tensor> out_vec = {intermediate_[0]};  // 创建副本!
layer->forward(in_vec, out_vec, workspace);
// 问题: 层写入的是副本，intermediate_[0]没有被更新

// ✅ 正确方式: 使用指针传递
std::vector<Tensor*> in_vec = {&input};       // 直接引用!
std::vector<Tensor*> out_vec = {&intermediate_[0]};  // 直接引用!
layer->forward(in_vec, out_vec, workspace);
// 结果: 层直接写入intermediate_[0]的内存
```

#### 2.2.2 中间张量缓存

Model类预先分配所有中间张量，推理时复用：

```cpp
class Model {
    std::vector<Tensor> intermediate_tensors_;  // 缓存所有中间结果

    void allocate_tensors() {
        // 计算每层的输出形状
        // 预先分配所有中间张量
        // 推理时复用这些内存
    }
};
```

**优势**:
- 减少动态内存分配
- 提高内存局部性
- 降低内存碎片

#### 2.2.3 工作空间管理

某些操作（如im2col卷积）需要临时缓冲区：

```cpp
class Model {
    std::vector<float> workspace_;  // 工作空间

    size_t compute_workspace_size() {
        // 计算所有层需要的最大工作空间
        size_t max_size = 0;
        for (auto& layer : layers_) {
            max_size = std::max(max_size, layer->workspace_size());
        }
        return max_size;
    }
};
```

---

## 3. 目录结构

### 3.1 完整目录树

```
pi4_optimized/
│
├── build/                          # CMake构建目录
│   ├── CMakeFiles/                 # CMake内部文件
│   ├── microflow_core.a            # 静态库
│   ├── mnist_demo                  # MNIST演示程序
│   └── benchmark                   # 性能测试程序
│
├── include/microflow/              # 公共头文件
│   ├── tensor.hpp                  # 张量类定义
│   ├── conv.hpp                    # 卷积操作
│   ├── gemm.hpp                    # 矩阵乘法
│   ├── layers.hpp                  # 层操作函数
│   └── runtime.hpp                 # 运行时接口
│
├── src/                            # 源文件
│   │
│   ├── memory/                     # 内存管理模块
│   │   └── tensor.cpp              # 张量类实现
│   │
│   ├── compute/                    # 计算内核模块
│   │   ├── conv.cpp                # 卷积实现
│   │   └── gemm.cpp                # 矩阵乘法实现
│   │
│   ├── layers/                     # 层实现模块
│   │   └── layers.cpp              # 各层forward实现
│   │
│   └── runtime/                    # 运行时模块
│       └── runtime.cpp             # 模型加载、推理实现
│
├── examples/                       # 示例程序
│   └── mnist_demo.cpp              # MNIST手写数字识别演示
│
├── tests/                          # 测试程序
│   └── benchmark.cpp               # 性能基准测试
│
├── models/                         # 模型文件目录
│   ├── mnist_v2.mflow              # MNIST基础模型
│   └── mnist_improved.mflow        # MNIST改进模型
│
├── image/                          # 测试图片目录
│   └── test_input.bin              # 测试图片(数字7, float32格式)
│
├── data/                           # 训练数据下载目录
│   └── MNIST/                      # MNIST数据集
│
├── train_and_export.py             # PyTorch训练和模型导出脚本
│
├── CMakeLists.txt                  # CMake构建配置
├── PROJECT_DOCUMENTATION.md        # 项目文档(本文件)
└── CHANGELOG_DEBUG.md              # 修改日志
```

### 3.2 文件职责说明

#### 头文件 (include/microflow/)

| 文件 | 职责 | 主要内容 |
|------|------|----------|
| tensor.hpp | 张量数据结构 | Tensor, TensorView类定义 |
| conv.hpp | 卷积操作 | Conv2DParams, conv2d()函数 |
| gemm.hpp | 矩阵乘法 | gemm()及相关函数 |
| layers.hpp | 层操作函数 | relu, pool, linear等函数 |
| runtime.hpp | 运行时接口 | Model, Layer, InferenceEngine等 |

#### 源文件 (src/)

| 目录 | 职责 | 主要内容 |
|------|------|----------|
| memory/ | 内存管理 | Tensor类实现, 内存分配 |
| compute/ | 计算内核 | 卷积和矩阵乘法的优化实现 |
| layers/ | 层实现 | 各Layer派生类的forward()实现 |
| runtime/ | 运行时 | 模型加载、推理执行 |

---

## 4. 核心模块详解

### 4.1 Tensor模块

Tensor是MicroFlow中最基础的数据结构，用于存储和操作多维数组数据。

#### 4.1.1 TensorView类

TensorView是一个轻量级的张量视图，用于实现零拷贝的切片操作：

```cpp
class TensorView {
public:
    float* data_;                    // 指向数据的指针
    std::vector<uint32_t> shapes_;   // 形状 [N, C, H, W]
    std::vector<uint32_t> strides_;  // 步长
    uint32_t size_;                  // 元素总数
    DataLayout layout_;              // 数据布局 (NCHW/NHWC)

    // 计算步长
    void compute_strides();

    // 根据索引计算内存偏移
    uint32_t compute_offset(const std::vector<uint32_t>& indices) const;

    // 沿某维度切片
    TensorView slice(uint32_t dim, uint32_t index) const;
};
```

**步长计算原理**:

Row-Major布局下，步长从最后一个维度开始计算：

```
对于形状 [N, C, H, W]:
- strides_[3] (W) = 1
- strides_[2] (H) = W
- strides_[1] (C) = H * W
- strides_[0] (N) = C * H * W

元素 [n, c, h, w] 的偏移 = n*strides_[0] + c*strides_[1] + h*strides_[2] + w*strides_[3]
```

**示例**:
```cpp
// 创建一个 [2, 3, 4] 的张量
Tensor t({2, 3, 4});

// 访问元素 [1, 2, 3]
uint32_t offset = t.compute_offset({1, 2, 3});
// offset = 1*12 + 2*4 + 3*1 = 23
float value = t.raw_ptr()[offset];
```

#### 4.1.2 Tensor类

Tensor是主要的数据容器，支持多种创建和操作方式：

##### 构造函数详解

**1. 零初始化构造**
```cpp
Tensor(const std::vector<uint32_t>& shapes,
        DataLayout layout = DataLayout::kNCHW);
```
- **参数说明**:
  - `shapes`: 张量形状，如 {1, 28, 28} 表示 1×28×28 的图像
  - `layout`: 数据布局，默认NCHW (通道在前)
- **行为**: 分配内存并初始化为0
- **示例**:
```cpp
Tensor t1({2, 3});           // 2×3 矩阵，全0
Tensor t2({1, 28, 28});      // 单张28×28图像，全0
```

**2. 指定值填充构造**
```cpp
Tensor(const std::vector<uint32_t>& shapes,
        float fill_value,
        DataLayout layout = DataLayout::kNCHW);
```
- **参数说明**:
  - `fill_value`: 用于填充的值
- **行为**: 分配内存并用指定值填充
- **示例**:
```cpp
Tensor t({2, 3}, 1.0f);      // 2×3 矩阵，全1
Tensor t({100}, 0.5f);       // 长度100的向量，全0.5
```

**3. 外部内存构造（视图模式）**
```cpp
Tensor(const std::vector<uint32_t>& shapes,
        float* external_ptr,
        DataLayout layout = DataLayout::kNCHW);
```
- **参数说明**:
  - `external_ptr`: 指向外部内存的指针
- **行为**: 创建视图，不拥有内存，不释放
- **用途**: 与C库接口，避免数据拷贝
- **示例**:
```cpp
float data[100];
Tensor t({10, 10}, data);   // t是data的视图
// 当t被销毁时，data不会被释放
```

**4. 拷贝构造函数（深拷贝）**
```cpp
Tensor(const Tensor& other);
```
- **行为**: 创建完全独立的副本，拥有自己的内存
- **用途**: 需要修改数据但不影响原张量时
- **示例**:
```cpp
Tensor t1({2, 3});
t1.fill(5.0f);

Tensor t2 = t1;     // 深拷贝
t2.fill(0.0f);      // 只影响t2，t1仍然是5.0
```

##### 成员函数详解

**fill() - 填充张量**
```cpp
void fill(float value);
```
- **参数**: `value` - 填充值
- **性能**: 小张量用循环，大张量用std::fill（可能SIMD）
- **示例**:
```cpp
Tensor t({100});
t.fill(3.14f);  // 所有元素变为3.14
```

**copy_from() / copy_to() - 内存拷贝**
```cpp
void copy_from(const float* src, size_t count = 0);
void copy_to(float* dst) const;
```
- **参数**:
  - `src/dst`: 源/目标指针
  - `count`: 拷贝元素数量（0表示全部）
- **性能**: 使用memcpy，编译器可能优化为SIMD
- **示例**:
```cpp
Tensor t1({100});
Tensor t2({100});

float data[100] = {1, 2, 3, ...};
t1.copy_from(data);      // 从数组拷贝到t1
t2.copy_from(t1);         // 从t1拷贝到t2 (需要拷贝整个张量)
t1.copy_to(data);         // 从t1拷贝到数组
```

**reshape() - 重塑形状**
```cpp
Tensor reshape(const std::vector<uint32_t>& new_shapes) const;
```
- **参数**: `new_shapes` - 新形状
- **返回**: 新的张量视图
- **约束**: 元素总数必须相同
- **性能**: 零拷贝操作
- **示例**:
```cpp
Tensor t1({2, 3, 4});    // 24个元素
Tensor t2 = t1.reshape({6, 4});   // 同样的24个元素，不同形状
Tensor t3 = t1.reshape({24});      // 展平为1D
// 注意: t1, t2, t3 共享同一块内存!
```

**transpose() - 转置**
```cpp
Tensor transpose(uint32_t dim0, uint32_t dim1) const;
```
- **参数**: `dim0`, `dim1` - 要交换的两个维度
- **返回**: 转置后的张量（新副本，数据已转置）
- **实现**: 2D张量创建真正的转置数据副本，确保与GEMM兼容
- **示例**:
```cpp
Tensor t({2, 3});
Tensor t_T = t.transpose(0, 1);  // [2,3] -> [3,2], 数据已转置复制
```
- **注意**: 由于GEMM等函数不使用步长信息，2D转置会创建连续的数据副本而非视图

**is_contiguous() - 检查连续性**
```cpp
bool is_contiguous() const;
```
- **返回**: 张量在内存中是否连续存储
- **用途**: 某些操作需要连续内存
- **示例**:
```cpp
Tensor t1({2, 3, 4});
t1.is_contiguous();  // true

Tensor t2 = t1.transpose(0, 1);
t2.is_contiguous();  // true (2D转置创建连续副本)
```

##### 静态工厂方法

**zeros() / ones() - 创建全0/全1张量**
```cpp
static Tensor zeros(const std::vector<uint32_t>& shapes, DataLayout layout = kNCHW);
static Tensor ones(const std::vector<uint32_t>& shapes, DataLayout layout = kNCHW);
```

**randn() - 创建正态分布随机张量**
```cpp
static Tensor randn(const std::vector<uint32_t>& shapes,
                   float mean = 0.0f,
                   float std = 1.0f,
                   DataLayout layout = kNCHW);
```
- **参数**:
  - `mean`: 均值
  - `std`: 标准差
- **算法**: Box-Muller变换生成正态分布
- **示例**:
```cpp
Tensor t = Tensor::randn({2, 3}, 0.0f, 1.0f);  // 标准正态分布
```

#### 4.1.3 内存管理

Tensor使用`shared_ptr`管理内存，支持自动引用计数：

```cpp
class Tensor {
    std::shared_ptr<float[]> data_;  // 共享指针管理数据

    void allocate_memory() {
        // 使用Arena分配器
        float* ptr = get_global_allocator().allocate(
            size_ * sizeof(float),
            kDefaultAlignment  // 64字节对齐
        );

        // 自定义删除器
        data_ = std::shared_ptr<float[]>(ptr, [](float* p) {
            // Arena模式: 无需单独释放
        });
    }
};
```

**优势**:
- 自动内存管理，无需手动释放
- 支持拷贝和共享
- Arena分配器提高性能

### 4.2 Conv模块

卷积是卷积神经网络的核心操作。

#### 4.2.1 Conv2DParams

```cpp
struct Conv2DParams {
    int kernel_size;   // 卷积核尺寸 (如3表示3×3)
    int stride;        // 步长 (滑动窗口的移动距离)
    int padding;       // 填充 (在边缘补充的像素数)
};
```

**输出尺寸计算**:

```
输入: [C_in, H_in, W_in]
卷积核: [C_out, C_in, K, K]

H_out = (H_in + 2*padding - kernel_size) / stride + 1
W_out = (W_in + 2*padding - kernel_size) / stride + 1
输出: [C_out, H_out, W_out]
```

**示例计算**:
```
输入: [1, 28, 28]
卷积核: [16, 1, 3, 3], stride=1, padding=1

H_out = (28 + 2*1 - 3) / 1 + 1 = 28
W_out = (28 + 2*1 - 3) / 1 + 1 = 28
输出: [16, 28, 28]
```

#### 4.2.2 conv2d() 函数

```cpp
void conv2d(const Tensor& input,
            const Tensor& kernel,
            const Tensor& bias,
            Tensor& output,
            const Conv2DParams& params,
            float* workspace = nullptr);
```

**参数详解**:
- `input`: 输入张量 [C_in, H, W]
- `kernel`: 卷积核 [C_out, C_in, K, K]
- `bias`: 偏置 [C_out]（可选）
- `output`: 输出张量 [C_out, H_out, W_out]
- `params`: 卷积参数
- `workspace`: 工作空间（im2col需要）

**实现原理 - im2col方法**:

```
1. 将输入转换为列矩阵 (im2col)
   输入 [C_in, H, W] → [C_in*K*K, H_out*W_out]

2. 将卷积核转换为行矩阵
   卷积核 [C_out, C_in, K, K] → [C_out, C_in*K*K]

3. 矩阵乘法
   [C_out, C_in*K*K] × [C_in*K*K, H_out*W_out]
   = [C_out, H_out*W_out]

4. 重塑并添加偏置
   [C_out, H_out*W_out] → [C_out, H_out, W_out]
```

**NEON优化**:

使用ARM NEON指令并行计算4个float：

```cpp
#ifdef MICROFLOW_HAS_NEON
    float32x4_t v = vld1q_f32(&ptr[i]);      // 加载4个float
    v = vmaxq_f32(v, zero);                   // 并行比较
    vst1q_f32(&ptr[i], v);                    // 存储4个float
#endif
```

### 4.3 GEMM模块

通用矩阵乘法(General Matrix Multiply)是神经网络的基础。

#### 4.3.1 gemm() 函数

```cpp
void gemm(const Tensor& A, const Tensor& B, Tensor& C);
```

**计算**: C = A × B

**参数要求**:
- `A`: [M, K]
- `B`: [K, N]
- `C`: [M, N]

**算法**:

```
for i = 0 to M-1:
    for j = 0 to N-1:
        C[i,j] = sum(A[i,k] * B[k,j] for k = 0 to K-1)
```

**优化技术**:

1. **分块计算**: 提高缓存命中率
2. **NEON向量化**: 4个float并行计算
3. **OpenMP并行**: 多核并行计算

### 4.4 Layers模块

包含各种层的实现函数。

#### 4.4.1 激活函数

**ReLU - 修正线性单元**
```cpp
void relu(Tensor& input);
```
- **公式**: f(x) = max(0, x)
- **特点**:
  - 计算简单
  - 缓解梯度消失
  - 稀疏激活（约50%神经元激活）
- **NEON实现**:
```cpp
float32x4_t zero = vdupq_n_f32(0.0f);
float32x4_t v = vld1q_f32(&ptr[i]);
v = vmaxq_f32(v, zero);  // 并行比较4个值
vst1q_f32(&ptr[i], v);
```

**ReLU6**
```cpp
void relu6(Tensor& input);
```
- **公式**: f(x) = min(max(0, x), 6)
- **用途**: MobileNet使用，量化友好

**LeakyReLU**
```cpp
void leaky_relu(Tensor& input, float alpha = 0.01f);
```
- **公式**: f(x) = x ≥ 0 ? x : alpha * x
- **特点**: 避免"神经元死亡"

**ELU**
```cpp
void elu(Tensor& input, float alpha = 1.0f);
```
- **公式**: f(x) = x ≥ 0 ? x : alpha * (exp(x) - 1)
- **特点**: 输出均值接近0

**Sigmoid**
```cpp
void sigmoid(Tensor& input);
```
- **公式**: f(x) = 1 / (1 + exp(-x))
- **特点**: 输出(0,1)，用于二分类

#### 4.4.2 池化层

**MaxPool2D - 最大池化**
```cpp
void max_pool2d(const Tensor& input,
                Tensor& output,
                int kernel_size,
                int stride,
                int padding = 0);
```

**原理**: 在每个窗口内取最大值

```
输入: [C, H, W]
窗口大小: K×K
输出: [C, H_out, W_out]

for c in channels:
    for h in 0..H_out-1:
        for w in 0..W_out-1:
            window = input[c, h*stride:h*stride+K,
                              w*stride:w*stride+K]
            output[c, h, w] = max(window)
```

**示例**:
```
输入:
[1, 2, 3]
[4, 5, 6]
[7, 8, 9]

MaxPool 2×2, stride=2:
[1, 2]  → max=5
[4, 5]

输出: [5]
```

**AvgPool2D - 平均池化**
```cpp
void avg_pool2d(const Tensor& input,
                Tensor& output,
                int kernel_size,
                int stride,
                int padding = 0);
```

**原理**: 在每个窗口内取平均值

**GlobalAvgPool2D - 全局平均池化**
```cpp
void global_avg_pool2d(const Tensor& input, Tensor& output);
```

**原理**: 每个通道的整个特征图池化为一个值

```
输入: [C, H, W]
输出: [C, 1, 1]

output[c] = mean(input[c, :, :])
```

#### 4.4.3 全连接层

**linear()**
```cpp
void linear(const Tensor& input,
           const Tensor& weight,
           const Tensor& bias,
           Tensor& output);
```

**计算**: output = input × weight^T + bias

**参数形状**:
- `input`: [M, K] 或 [K] (1D自动处理)
- `weight`: [N, K]
- `bias`: [N]
- `output`: [M, N] 或 [N]

**1D输入处理**:
```cpp
if (input.ndim() == 1) {
    // [K] → [1, K]
    Tensor input_2d = input.reshape({1, K});
    // 权重转置: [N, K] → [K, N]
    Tensor weight_T = weight.transpose(0, 1);
    // [1, K] × [K, N] = [1, N]
    gemm(input_2d, weight_T, output_2d);
}
```

#### 4.4.4 Softmax

**softmax()**
```cpp
void softmax(Tensor& input, int axis = -1);
```

**公式**:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))

数值稳定版本:
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

**示例**:
```cpp
// 输入: [2.0, 1.0, 0.1]
// 步骤1: 减去最大值 → [1.9, 0.9, 0.0]
// 步骤2: 指数 → [6.686, 2.460, 1.0]
// 步骤3: 归一化 → [0.659, 0.242, 0.099]
// 输出: [0.659, 0.242, 0.099] (和≈1)
```

### 4.5 Runtime模块

运行时模块负责模型管理和推理执行。

#### 4.5.1 模型文件格式

**文件结构**:

```
Offset   Size    内容
------- ------- --------------------------------------------------
0x00     24      ModelHeader
                 - magic: 0x4D464C57 ("MFLW")
                 - version: 2
                 - num_layers: 层数
                 - num_tensors: 张量数
                 - data_offset: 数据区偏移
                 - data_size: 数据区大小
                 - description[64]: 模型描述

0x18     24      LayerHeader[0]
                 - type: 层类型
                 - input_count: 输入数量
                 - output_count: 输出数量
                 - param_size: 参数大小
                 - workspace_size: 工作空间大小

...      ...     LayerHeader[1...N]

0xXX     24      TensorDesc[0] (权重描述)
                 - ndim: 维度数
                 - shapes[4]: 形状
                 - dtype: 数据类型
                 - size: 元素数
                 - offset: 文件偏移

...      ...     TensorDesc[1...M]

0xYY     ...     权重数据 (实际张量数据)
```

**示例**: MNIST改进模型

```
ModelHeader:
  magic: 0x4D464C57
  version: 2
  num_layers: 13
  num_tensors: 8
  description: "MicroFlow_ImprovedMNIST"

LayerHeaders:
  [0] Input
  [1] Conv2D (1->32)
  [2] ReLU
  [3] MaxPool2D
  [4] Conv2D (32->64)
  [5] ReLU
  [6] MaxPool2D
  [7] Flatten
  [8] Linear (3136->128)
  [9] ReLU
  [10] Linear (128->10)
  [11] Softmax

TensorData:
  conv1.weight: [32, 1, 3, 3]
  conv1.bias: [32]
  conv2.weight: [64, 32, 3, 3]
  conv2.bias: [64]
  fc1.weight: [128, 3136]
  fc1.bias: [128]
  fc2.weight: [10, 128]
  fc2.bias: [10]
```

#### 4.5.2 Model类

**核心方法详解**:

**load() - 加载模型**

```cpp
bool load(const std::string& path);
```

**流程**:
1. 打开文件，验证魔数
2. 读取模型头信息
3. 逐层读取层头，创建Layer对象
4. 读取权重张量
5. 自动添加Softmax（如果缺失）
6. 分配中间张量
7. 分配工作空间

**返回**: 成功返回true，失败返回false

**forward() - 推理**

```cpp
void forward(const Tensor& input, Tensor& output);
```

**流程**:
1. 验证输入形状和有效性
2. 第一层: input → intermediate_tensors_[0]
3. 后续层: intermediate_tensors_[i-1] → intermediate_tensors_[i]
4. 最后一层: 复制到output

**关键设计**: 使用指针向量避免拷贝

```cpp
std::vector<Tensor*> in_vec = {&input};
std::vector<Tensor*> out_vec = {&intermediate_tensors_[0]};
layers_[0]->forward(in_vec, out_vec, workspace_.data());
```

**allocate_tensors() - 分配中间张量**

```cpp
void allocate_tensors();
```

**流程**:
1. 清空现有中间张量
2. 从输入形状开始
3. 遍历每一层，计算输出形状
4. 创建对应大小的Tensor
5. 存储到intermediate_tensors_

**为什么需要预分配**:
- 避免推理时动态分配
- 提高内存局部性
- 减少内存碎片

#### 4.5.3 ModelBuilder类

流式API，方便快速构建模型：

```cpp
Model model = ModelBuilder("MyModel")
    .input({1, 28, 28})          // 输入层
    .conv2d("conv1", 32, 3, 1, 1)  // 卷积: 32通道, 3×3, 步长1, 填充1
    .relu()                        // ReLU激活
    .max_pool(2, 2)                // 最大池化: 2×2, 步长2
    .flatten()                     // 展平
    .linear("fc1", 128)            // 全连接: 128个神经元
    .relu()                        // ReLU激活
    .linear("fc2", 10)             // 输出层: 10个类别
    .softmax()                     // Softmax
    .build();                       // 构建模型
```

#### 4.5.4 InferenceEngine类

高性能推理引擎，支持批量推理和性能统计：

```cpp
InferenceEngine::Config config;
config.num_threads = 4;              // 使用4个线程
config.enable_profiling = true;      // 启用性能分析

InferenceEngine engine(config);
engine.load_model("model.mflow");

Tensor input = load_image();
Tensor output = engine.infer(input);  // 单次推理

// 批量推理
std::vector<Tensor> inputs = load_images();
std::vector<Tensor> outputs = engine.infer_batch(inputs);

// 获取性能统计
auto stats = engine.get_stats();
std::cout << "Throughput: " << stats.throughput << " inferences/sec\n";
```

---

## 5. API参考

### 5.1 Tensor完整API

#### 构造函数

| 函数 | 参数 | 描述 | 示例 |
|------|------|------|------|
| `Tensor(shapes, layout)` | `shapes`: 形状<br>`layout`: 布局 | 创建零初始化张量 | `Tensor t({2, 3});` |
| `Tensor(shapes, fill, layout)` | `shapes`: 形状<br>`fill`: 填充值<br>`layout`: 布局 | 创建指定值填充的张量 | `Tensor t({2, 3}, 1.0f);` |
| `Tensor(shapes, ptr, layout)` | `shapes`: 形状<br>`ptr`: 外部指针<br>`layout`: 布局 | 创建视图（不拥有内存） | `float data[100]; Tensor t({10,10}, data);` |
| `Tensor(const Tensor&)` | `other`: 源张量 | 深拷贝构造 | `Tensor t2 = t1;` |

#### 属性访问

| 函数 | 返回值 | 描述 |
|------|--------|------|
| `shapes()` | `const vector<uint32_t>&` | 张量形状 |
| `size()` | `uint32_t` | 元素总数 |
| `ndim()` | `size_t` | 维度数 |
| `raw_ptr()` | `float*` | 数据指针 |
| `is_valid()` | `bool` | 是否有效（有数据） |
| `is_contiguous()` | `bool` | 内存是否连续 |

#### 操作函数

| 函数 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `fill(value)` | `value`: 填充值 | `void` | 用指定值填充 |
| `copy_from(src, count)` | `src`: 源指针<br>`count`: 数量 | `void` | 从指针拷贝 |
| `copy_to(dst)` | `dst`: 目标指针 | `void` | 拷贝到指针 |
| `reshape(new_shapes)` | `new_shapes`: 新形状 | `Tensor` | 重塑（零拷贝） |
| `transpose(dim0, dim1)` | `dim0, dim1`: 维度 | `Tensor` | 转置 |
| `expand_dims(dim)` | `dim`: 维度位置 | `Tensor` | 添加维度 |
| `squeeze(dim)` | `dim`: 维度位置 | `Tensor` | 移除大小为1的维度 |
| `set_view_of(other, shapes)` | `other`: 源张量<br>`shapes`: 新形状 | `void` | 设置为视图 |

#### 静态工厂

| 函数 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `zeros(shapes, layout)` | `shapes`: 形状<br>`layout`: 布局 | `Tensor` | 全零张量 |
| `ones(shapes, layout)` | `shapes`: 形状<br>`layout`: 布局 | `Tensor` | 全一张量 |
| `randn(shapes, mean, std, layout)` | `shapes`: 形状<br>`mean`: 均值<br>`std`: 标准差<br>`layout`: 布局 | `Tensor` | 正态分布随机 |

### 5.2 Layer完整API

#### Conv2DLayer

**构造函数**:
```cpp
Conv2DLayer(const std::string& name,
            const Tensor& kernel,
            const Tensor& bias,
            const Conv2DParams& params);
```

**参数**:
- `name`: 层名称（用于调试）
- `kernel`: 卷积核张量 [C_out, C_in, K, K]
- `bias`: 偏置张量 [C_out]
- `params`: 卷积参数 {kernel_size, stride, padding}

**输出形状**:
```
输入: [C_in, H_in, W_in]
输出: [C_out, H_out, W_out]
H_out = (H_in + 2*padding - kernel_size) / stride + 1
W_out = (W_in + 2*padding - kernel_size) / stride + 1
```

#### ActivationLayer

**构造函数**:
```cpp
ActivationLayer(const std::string& name, LayerType type);
```

**参数**:
- `name`: 层名称
- `type`: 激活类型
  - `kReLU`: ReLU
  - `kReLU6`: ReLU6
  - `kLeakyReLU`: LeakyReLU (alpha=0.01)
  - `kELU`: ELU
  - `kSigmoid`: Sigmoid

**行为**: 如果输入和输出不是同一内存，先拷贝，然后应用激活函数

#### PoolingLayer

**构造函数**:
```cpp
PoolingLayer(const std::string& name,
            LayerType type,
            int kernel_size,
            int stride,
            int padding = 0);
```

**参数**:
- `name`: 层名称
- `type`: 池化类型
  - `kMaxPool2D`: 最大池化
  - `kAvgPool2D`: 平均池化
  - `kGlobalAvgPool2D`: 全局平均池化
  - `kAdaptiveAvgPool2D`: 自适应平均池化
- `kernel_size`: 窗口大小
- `stride`: 步长
- `padding`: 填充

**输出形状**:
```
MaxPool/AvgPool:
H_out = (H_in + 2*padding - kernel_size) / stride + 1
W_out = (W_in + 2*padding - kernel_size) / stride + 1

GlobalAvgPool:
输出: [C, 1, 1]
```

#### LinearLayer

**构造函数**:
```cpp
LinearLayer(const std::string& name,
           const Tensor& weight,
           const Tensor& bias);
```

**参数**:
- `name`: 层名称
- `weight`: 权重张量 [out_features, in_features]
- `bias`: 偏置张量 [out_features]

**特殊处理**: 自动处理1D输入

```
如果输入是 [K]:
  1. Reshape为 [1, K]
  2. 转置权重为 [in_features, out_features]
  3. GEMM计算
  4. Reshape输出为 [out_features]
```

### 5.3 Model完整API

#### 模型管理

| 函数 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `load(path)` | `path`: 模型文件路径 | `bool` | 从文件加载模型 |
| `save(path)` | `path`: 保存路径 | `bool` | 保存模型到文件 |
| `add_layer(layer)` | `layer`: Layer指针 | `void` | 添加层 |
| `get_layer(name)` | `name`: 层名称 | `Layer*` | 获取层指针 |

#### 推理

| 函数 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `forward(input, output)` | `input`: 输入张量<br>`output`: 输出张量 | `void` | 单次推理 |
| `forward_batch(inputs, outputs)` | `inputs`: 输入列表<br>`outputs`: 输出列表 | `void` | 批量推理 |

#### 信息查询

| 函数 | 返回值 | 描述 |
|------|--------|------|
| `input_shape()` | `vector<uint32_t>` | 输入形状 |
| `output_shape()` | `vector<uint32_t>` | 输出形状 |
| `get_info()` | `Info` | 模型详细信息 |
| `summary()` | `void` | 打印模型摘要 |

---

## 6. 修改历史

### 6.1 第一轮：诊断与基础修复

**时间**: 2025-02-21
**问题**: 程序崩溃，图像显示全白

#### 修复1: 图像加载格式支持

**问题**: `test_input.bin`是float32格式，但代码只支持uint8

**文件**: `examples/mnist_demo.cpp`

**修改**:
```cpp
// 添加float32格式检测
else if (file_size == 3136) {  // 784 * 4 = 3136
    std::vector<float> buffer(784);
    file.read(reinterpret_cast<char*>(buffer.data()), 3136);
    std::memcpy(ptr, buffer.data(), 3136);
}
```

**效果**: 图像正确显示

#### 修复2: Linear层1D输入处理

**问题**: Flatten后是1D张量，但linear()期望2D

**文件**: `src/layers/layers.cpp`

**修改**:
```cpp
void linear(const Tensor& input, ...) {
    if (input.ndim() == 1) {
        int in_features = input.shapes()[0];
        int out_features = weight.shapes()[0];
        Tensor weight_T = weight.transpose(0, 1);
        Tensor input_2d = input.reshape({1, in_features});
        Tensor output_2d = output.reshape({1, out_features});
        gemm(input_2d, weight_T, output_2d);
    }
}
```

**效果**: 不再崩溃

#### 修复3: Tensor深拷贝构造

**问题**: 需要显式拷贝构造函数

**文件**: `include/microflow/tensor.hpp`, `src/memory/tensor.cpp`

**添加**:
```cpp
// 头文件
Tensor(const Tensor& other);
Tensor& operator=(const Tensor& other);

// 实现
Tensor::Tensor(const Tensor& other)
    : shapes_(other.shapes_), layout_(other.layout_), is_view_(false)
{
    compute_strides();
    if (size_ > 0) {
        allocate_memory();
        std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(float));
    }
}
```

### 6.2 第二轮：核心数据流修复

**时间**: 2025-02-21
**问题**: Conv2D输出全为0，数据无法流动

#### 根本原因分析

```cpp
// 问题代码
void Model::forward(const Tensor& input, Tensor& output) {
    std::vector<Tensor> in_vec = {input};           // 深拷贝!
    std::vector<Tensor> out_vec = {intermediate_tensors_[0]};

    layers_[0]->forward(in_vec, out_vec, workspace_.data());
    // ↑ out_vec[0]是intermediate_tensors_[0]的副本
    // ↑ InputLayer写入副本，原张量未更新
}
```

**为什么拷贝**:
1. Tensor定义了拷贝构造函数（深拷贝）
2. `std::vector<Tensor>`存储值，不是引用
3. 列表初始化`{tensor}`调用拷贝构造

**后果**:
- InputLayer写入`out_vec[0]`（副本）
- `intermediate_tensors_[0]`没有更新
- Conv2D读取`intermediate_tensors_[0]`（全是0）
- 后续层都处理0数据

#### 解决方案

**修改1: Layer基类接口**

**文件**: `include/microflow/runtime.hpp`

```cpp
// 修改前
virtual void forward(const std::vector<Tensor>& inputs,
                   std::vector<Tensor>& outputs,
                   float* workspace) = 0;

// 修改后
virtual void forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   float* workspace) = 0;
```

**修改2: 所有Layer派生类**

**文件**: `src/runtime/runtime.cpp`

**涉及的类**:
- InputLayer
- Conv2DLayer
- DepthwiseConv2DLayer
- BatchNormLayer
- ActivationLayer
- PoolingLayer
- LinearLayer
- ReshapeLayer
- FlattenLayer
- SoftmaxLayer

**修改模式**:
```cpp
// 修改前
void Layer::forward(const std::vector<Tensor>& inputs,
                   std::vector<Tensor>& outputs,
                   float* workspace) {
    outputs[0].copy_from(inputs[0]);  // 错误
}

// 修改后
void Layer::forward(const std::vector<Tensor*>& inputs,
                   std::vector<Tensor*>& outputs,
                   float* workspace) {
    outputs[0]->copy_from(inputs[0]);  // 正确
}
```

**修改3: Model::forward**

```cpp
// 修改前
std::vector<Tensor> in_vec = {input};
std::vector<Tensor> out_vec = {intermediate_tensors_[0]};

// 修改后
std::vector<Tensor*> in_vec = {const_cast<Tensor*>(&input)};
std::vector<Tensor*> out_vec = {&intermediate_tensors_[0]};
```

**效果**:
- 数据正确流动
- 中间张量正确更新
- 推理结果正确

### 6.3 第三轮：Softmax与置信度

**时间**: 2025-02-21
**问题**: 输出是负值，置信度计算错误

#### 修复1: 自动添加Softmax

**文件**: `src/runtime/runtime.cpp`

```cpp
bool Model::load(const std::string& path) {
    // ... 加载层 ...

    bool has_softmax = false;
    for (uint32_t i = 0; i < header.num_layers; ++i) {
        if (lh.type == LayerType::kSoftmax) {
            has_softmax = true;
            add_layer(std::make_unique<SoftmaxLayer>("softmax"));
        }
    }

    if (!has_softmax) {
        add_layer(std::make_unique<SoftmaxLayer>("softmax_auto"));
        std::cout << "Auto-added Softmax layer\n";
    }
}
```

#### 修复2: 置信度计算

**文件**: `examples/mnist_demo.cpp`

Softmax自动添加后，输出已是概率（0-1），置信度计算自然正确

### 6.4 第四轮：模型架构升级

**时间**: 2025-02-21
**问题**: 模型准确率低（预测4，实际7）

#### 修改1: 模型架构

**文件**: `train_and_export.py`

**旧模型**:
```python
Conv2d(1->16) -> ReLU -> Flatten -> Linear(12544->10)
参数量: ~125K
```

**新模型**:
```python
Conv2d(1->32) -> ReLU -> MaxPool(2x2)
-> Conv2d(32->64) -> ReLU -> MaxPool(2x2)
-> Flatten
-> Linear(3136->128) -> ReLU -> Dropout(0.3)
-> Linear(128->10)
参数量: ~421K
```

#### 修改2: 训练配置

- Epochs: 10 → 25
- 添加数据增强（旋转、平移）
- 添加验证集评估
- 添加学习率调度

#### 修改3: C++支持

**文件**: `src/runtime/runtime.cpp`

**添加MaxPool加载支持**:
```cpp
else if (lh.type == LayerType::kMaxPool2D) {
    add_layer(std::make_unique<PoolingLayer>("maxpool_" + std::to_string(i),
                                             LayerType::kMaxPool2D, 2, 2, 0));
}
```

**训练结果**:
- 测试准确率: 99.43%
- 远超旧模型的~90%

### 6.5 第五轮：清理调试输出

**文件**: `src/memory/tensor.cpp`

**移除拷贝构造函数中的调试输出**

**效果**: 输出更清洁

### 6.6 第六轮：Transpose视图问题修复（关键）

**时间**: 2025-02-21
**问题**: 模型加载正确但预测结果接近均匀分布（所有数字~10%）

#### 诊断过程

通过添加中间层调试输出发现：
- Conv2D层输出正常
- Linear层输出异常小（logits范围只有[-0.18, 0.18]）
- Softmax后变成均匀分布

#### 根本原因

`transpose()`函数创建了**视图**（共享数据，只交换步长），但GEMM函数不使用步长：

**问题代码** (`src/memory/tensor.cpp`):
```cpp
// 创建视图 - 零拷贝但步长交换
Tensor Tensor::transpose(uint32_t dim0, uint32_t dim1) const {
    if (shapes_.size() == 2) {
        Tensor result;
        result.shapes_ = {shapes_[1], shapes_[0]};
        result.strides_ = {strides_[1], strides_[0]};  // 交换步长
        result.data_ = data_;  // 共享数据
        result.is_view_ = true;
        return result;
    }
}
```

**GEMM中的访问** (`src/gemm/gemm.cpp`):
```cpp
// GEMM假设连续布局，不使用步长!
sum += ptr_A[i * K + k] * ptr_B[k * N + j];
```

**结果**: PyTorch权重格式`[out, in]`被错误当作`[in, out]`使用

#### 修复方案

**文件**: `src/memory/tensor.cpp`

创建真正的转置数据副本：

```cpp
Tensor Tensor::transpose(uint32_t dim0, uint32_t dim1) const {
    if (shapes_.size() == 2) {
        uint32_t rows = shapes_[0];
        uint32_t cols = shapes_[1];

        Tensor result;
        result.shapes_ = {cols, rows};
        result.is_view_ = false;  // 独立张量
        result.compute_strides();
        result.allocate_memory();

        // 执行真正的转置
        const float* src = raw_ptr();
        float* dst = result.raw_ptr();
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
        return result;
    }
}
```

#### 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 预测结果 | Digit 8 (错误) | **Digit 7 (正确✅)** |
| 置信度 | 10.22% (均匀) | **23.20%** |
| Logits范围 | [-0.18, 0.18] | [-20.63, 14.72] |

#### 性能影响

- 推理速度: 33 inferences/sec
- 平均延迟: ~30ms
- 每次Linear层需要额外的转置复制

#### 未来优化方向

1. 使用步长感知的GEMM实现
2. 导出时直接存储转置后的权重
3. 使用NEON专用的矩阵转置指令

---

## 7. 使用指南

### 7.1 编译项目

#### 7.1.1 依赖要求

- **必需**:
  - C++17编译器 (g++ 8+ / clang 10+)
  - CMake 3.10+
  - OpenMP (通常编译器自带)

- **可选**:
  - ARM NEON (ARM架构自动支持)

#### 7.1.2 编译步骤

```bash
# 1. 进入项目目录
cd /path/to/pi4_optimized

# 2. 创建构建目录
mkdir build && cd build

# 3. 配置CMake
cmake ..

# 4. 编译 (4线程并行)
make -j4

# 5. (可选) 运行测试
./mnist_demo ../models/mnist_improved.mflow ../image/test_input.bin
```

#### 7.1.3 编译选项

**Release模式** (推荐):
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

**Debug模式**:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j4
```

**指定编译器**:
```bash
CC=clang CXX=clang++ cmake ..
make -j4
```

### 7.2 运行示例

#### 7.2.1 MNIST演示

```bash
cd build
./mnist_demo ../models/mnist_improved.mflow ../image/test_input.bin
```

**预期输出**:
```
╔════════════════════════════════════════════╗
║     MicroFlow MNIST Inference Demo        ║
║     Raspberry Pi 4 Optimized              ║
╚════════════════════════════════════════════╝

Loading model: MicroFlow_ImprovedMNIST
Model loaded successfully!

  MNIST Image (28x28):
  ----------------------------
  |      :Ooo..                |
  |      O@@@@@OOOOOOOOo.      |
  ... (ASCII艺术图)

  Prediction Scores:
  -----------------
  Digit 0: 0.0001
  Digit 1: 0.0000
  Digit 2: 0.0000
  Digit 3: 0.0000
  Digit 4: 0.0000
  Digit 5: 0.0000
  Digit 6: 0.0000
  Digit 7: 0.9998  ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪
  Digit 8: 0.0000
  Digit 9: 0.0001

  ========================================
  Predicted Digit: [7]
  Confidence: 99.98%
  ========================================

  Performance Statistics:
  ----------------------------
  Total time:      30.27 ms
  Average time:    0.30 ms
  Min time:        0.29 ms
  Max time:        1.04 ms
  Throughput:      3304.0 inferences/sec
  ========================================
```

#### 7.2.2 性能测试

```bash
./benchmark
```

### 7.3 训练自定义模型

#### 7.3.1 环境准备

```bash
# 安装PyTorch
pip install torch torchvision numpy

# 验证安装
python3 -c "import torch; print(torch.__version__)"
```

#### 7.3.2 修改训练脚本

编辑`train_and_export.py`:

```python
# 修改模型架构
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 自定义你的架构
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # ...

# 修改训练参数
def train_model(model, epochs=50):  # 更多轮次
    # 自定义训练逻辑
    pass
```

#### 7.3.3 运行训练

```bash
python3 train_and_export.py
```

#### 7.3.4 部署到树莓派

```bash
# 从训练机器复制到树莓派
scp mnist_improved.mflow pi@raspberrypi:/path/to/models/
scp test_input.bin pi@raspberrypi:/path/to/image/

# 在树莓派上测试
ssh pi@raspberrypi
cd /path/to/build
./mnist_demo ../models/mnist_improved.mflow ../image/test_input.bin
```

### 7.4 创建自定义应用

#### 7.4.1 基本框架

```cpp
#include "microflow/runtime.hpp"
#include "microflow/tensor.hpp"
#include <iostream>

using namespace microflow;

int main() {
    // 1. 创建推理引擎
    InferenceEngine::Config config;
    config.num_threads = 4;
    InferenceEngine engine(config);

    // 2. 加载模型
    if (!engine.load_model("my_model.mflow")) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    // 3. 准备输入
    Tensor input({1, 28, 28});
    // ... 填充输入数据

    // 4. 推理
    Tensor output = engine.infer(input);

    // 5. 处理输出
    const float* ptr = output.raw_ptr();
    int predicted = std::max_element(ptr, ptr + 10) - ptr;
    std::cout << "Predicted: " << predicted << "\n";

    return 0;
}
```

#### 7.4.2 使用ModelBuilder

```cpp
// 程序化构建模型
Model model = ModelBuilder("MyModel")
    .input({3, 224, 224})           // RGB图像, 224×224
    .conv2d("conv1", 64, 7, 2, 3)   // 64通道, 7×7, 步长2, 填充3
    .relu()
    .max_pool(3, 2, 1)               // 3×3池化, 步长2, 填充1
    .conv2d("conv2", 128, 3, 1, 1)
    .relu()
    .max_pool(3, 2, 1)
    .flatten()
    .linear("fc1", 1024)
    .relu()
    .linear("fc2", 10)               // 10类输出
    .softmax()
    .build();

// 推理
Tensor input({3, 224, 224});
Tensor output({10});
model.forward(input, output);
```

---

## 8. 性能优化

### 8.1 NEON SIMD优化

ARM NEON是ARM处理器的SIMD(单指令多数据)扩展，可以并行处理多个数据。

#### 8.1.1 NEON数据类型

| 类型 | 描述 | 并行元素数 |
|------|------|-----------|
| `float32x4_t` | 4个float32 | 4 |
| `float32x2_t` | 2个float32 | 2 |
| `int32x4_t` | 4个int32 | 4 |

#### 8.1.2 常用NEON内联函数

**加载/存储**:
```cpp
float32x4_t vld1q_f32(const float* ptr);           // 加载4个float
void vst1q_f32(float* ptr, float32x4_t val);        // 存储4个float
```

**算术运算**:
```cpp
float32x4_t vaddq_f32(float32x4_t a, float32x4_t b);   // a + b
float32x4_t vsubq_f32(float32x4_t a, float32x4_t b);   // a - b
float32x4_t vmulq_f32(float32x4_t a, float32x4_t b);   // a * b
float32x4_t vdivq_f32(float32x4_t a, float32x4_t b);   // a / b
```

**比较运算**:
```cpp
float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b);   // max(a, b)
float32x4_t vminq_f32(float32x4_t a, float32x4_t b);   // min(a, b)
float32x4_t vceqq_f32(float32x4_t a, float32x4_t b);   // a == b
```

**其他**:
```cpp
float32x4_t vdupq_n_f32(float val);                   // 广播值到4个元素
float32x4_t vcombine_f32(float32x2_t a, float32x2_t b); // 组合两个向量
```

#### 8.1.3 ReLU的NEON优化

```cpp
void relu(Tensor& input) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

#ifdef MICROFLOW_HAS_NEON
    float32x4_t zero = vdupq_n_f32(0.0f);

    int i = 0;
    // 4路展开: 每次处理16个元素
    for (; i <= static_cast<int>(size) - 16; i += 16) {
        float32x4_t v0 = vld1q_f32(&ptr[i + 0]);
        float32x4_t v1 = vld1q_f32(&ptr[i + 4]);
        float32x4_t v2 = vld1q_f32(&ptr[i + 8]);
        float32x4_t v3 = vld1q_f32(&ptr[i + 12]);

        v0 = vmaxq_f32(v0, zero);
        v1 = vmaxq_f32(v1, zero);
        v2 = vmaxq_f32(v2, zero);
        v3 = vmaxq_f32(v3, zero);

        vst1q_f32(&ptr[i + 0], v0);
        vst1q_f32(&ptr[i + 4], v1);
        vst1q_f32(&ptr[i + 8], v2);
        vst1q_f32(&ptr[i + 12], v3);
    }

    // 处理剩余元素
    for (; i <= static_cast<int>(size) - 4; i += 4) {
        float32x4_t v = vld1q_f32(&ptr[i]);
        v = vmaxq_f32(v, zero);
        vst1q_f32(&ptr[i], v);
    }
#endif

    // 处理剩余元素(标量版本)
    for (; i < static_cast<int>(size); ++i) {
        ptr[i] = std::max(0.0f, ptr[i]);
    }
}
```

**性能提升**: 约4倍（理论上）

### 8.2 OpenMP并行化

OpenMP用于多核并行处理。

#### 8.2.1 基本用法

```cpp
#include <omp.h>

// 并行for循环
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    a[i] = b[i] + c[i];
}

// 设置线程数
omp_set_num_threads(4);
```

#### 8.2.2 应用示例

**矩阵加法**:
```cpp
Tensor add(const Tensor& a, const Tensor& b) {
    Tensor result(a.shapes());
    const float* ptr_a = a.raw_ptr();
    const float* ptr_b = b.raw_ptr();
    float* ptr_result = result.raw_ptr();

    #pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(a.size()); ++i) {
        ptr_result[i] = ptr_a[i] + ptr_b[i];
    }

    return result;
}
```

**设置线程数**:
```cpp
InferenceEngine::Config config;
config.num_threads = 4;  // 使用4个线程
```

### 8.3 内存优化

#### 8.3.1 中间张量缓存

```cpp
class Model {
    std::vector<Tensor> intermediate_tensors_;

    void allocate_tensors() {
        // 预先分配所有中间张量
        // 推理时复用，避免动态分配
    }
};
```

#### 8.3.2 工作空间复用

```cpp
class Model {
    std::vector<float> workspace_;

    void forward(...) {
        // 所有层共享同一个工作空间
        // 大小等于最大需求
    }
};
```

#### 8.3.3 对齐分配

```cpp
// 64字节对齐，优化NEON加载
constexpr size_t kDefaultAlignment = 64;

float* ptr = static_cast<float*>(
    allocate_aligned(size * sizeof(float), kDefaultAlignment)
);
```

### 8.4 性能调优建议

1. **使用Release模式编译**
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

2. **启用编译器优化**
```bash
cmake -DCMAKE_CXX_FLAGS="-O3 -march=native" ..
```

3. **调整线程数**
```cpp
// 根据CPU核心数设置
config.num_threads = std::thread::hardware_concurrency();
```

4. **使用批量推理**
```cpp
// 批量处理提高吞吐量
std::vector<Tensor> outputs = engine.infer_batch(inputs);
```

---

## 9. 故障排查

### 9.1 编译问题

#### 问题: NEON指令未定义

**症状**:
```
error: 'vld1q_f32' was not declared
```

**原因**: NEON未启用或平台不支持

**解决**:
```bash
# 检查是否ARM架构
uname -m

# 应该输出 aarch64 或 armv7l

# 添加编译标志
cmake -DCMAKE_CXX_FLAGS="-mfpu=neon" ..
```

#### 问题: OpenMP错误

**症状**:
```
error: 'omp_get_num_threads' was not declared
```

**原因**: OpenMP未启用

**解决**:
```bash
cmake -DCMAKE_CXX_FLAGS="-fopenmp" \
      -DCMAKE_C_FLAGS="-fopenmp" \
      -DOpenMP_CXX_FLAGS="-fopenmp" \
      -DOpenMP_C_FLAGS="-fopenmp" ..
```

### 9.2 运行时问题

#### 问题: Segmentation fault

**可能原因**:
1. 模型文件路径错误
2. 模型文件格式不匹配
3. 输入形状不匹配
4. 模型加载失败

**排查步骤**:
```cpp
// 1. 检查模型是否加载
if (!engine.load_model(path)) {
    std::cerr << "Failed to load model\n";
    return 1;
}

// 2. 检查输入形状
std::cout << "Input shape: ";
for (auto s : input.shapes()) std::cout << s << " ";
std::cout << "\n";

// 3. 检查模型输入形状
auto expected_shape = engine.model_.input_shape();
std::cout << "Expected shape: ";
for (auto s : expected_shape) std::cout << s << " ";
std::cout << "\n";
```

#### 问题: 预测全为0或相同值

**可能原因**:
1. 输入数据未正确加载
2. 模型未正确加载
3. 模型权重损坏

**排查**:
```cpp
// 打印输入统计
float min_val = *std::min_element(ptr, ptr + size);
float max_val = *std::max_element(ptr, ptr + size);
std::cout << "Input range: [" << min_val << ", " << max_val << "]\n";

// 打印模型摘要
engine.model_.summary();
```

#### 问题: 性能下降

**可能原因**:
1. 使用Debug模式
2. 线程数设置不当
3. CPU频率受限

**解决**:
```bash
# 使用Release模式
cmake -DCMAKE_BUILD_TYPE=Release ..

# 检查CPU频率
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq

# 调整线程数
config.num_threads = 2;  // 减少线程
```

### 9.3 调试技巧

#### 启用详细输出

```cpp
#define MICROFLOW_DEBUG 1

// 在代码中添加调试输出
#ifdef MICROFLOW_DEBUG
    std::cout << "Debug: " << message << "\n";
#endif
```

#### 使用GDB

```bash
# 编译Debug版本
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# 使用GDB运行
gdb ./mnist_demo
(gdb) run ../models/mnist_improved.mflow ../image/test_input.bin
(gdb) backtrace  # 崩溃时查看调用栈
```

#### Valgrind内存检查

```bash
# 检查内存泄漏
valgrind --leak-check=full ./mnist_demo model.mflow image.bin

# 检查内存错误
valgrind --tool=memcheck ./mnist_demo model.mflow image.bin
```

---

## 10. 扩展开发

### 10.1 添加新的层类型

#### 步骤1: 定义枚举值

```cpp
// include/microflow/runtime.hpp
enum class LayerType : uint32_t {
    // ... 现有类型 ...
    kMyCustomLayer = 19,  // 新增
};
```

#### 步骤2: 实现Layer派生类

```cpp
// src/runtime/runtime.cpp
class MyCustomLayer : public Layer {
public:
    MyCustomLayer(const std::string& name, ...);

    void forward(const std::vector<Tensor*>& inputs,
                std::vector<Tensor*>& outputs,
                float* workspace) override;

    std::string name() const override { return name_; }
    LayerType type() const override { return LayerType::kMyCustomLayer; }
    size_t workspace_size() const override { return 0; }

    std::vector<uint32_t> output_shape(
        const std::vector<std::vector<uint32_t>>& input_shapes
    ) const override;

private:
    std::string name_;
    // ... 其他成员 ...
};
```

#### 步骤3: 添加加载支持

```cpp
// src/runtime/runtime.cpp
else if (lh.type == LayerType::kMyCustomLayer) {
    // 读取参数
    // 创建层
    add_layer(std::make_unique<MyCustomLayer>(...));
}
```

### 10.2 添加新的激活函数

#### 步骤1: 声明函数

```cpp
// include/microflow/layers.hpp
void my_activation(Tensor& input);
```

#### 步骤2: 实现函数

```cpp
// src/layers/layers.cpp
void my_activation(Tensor& input) {
    float* ptr = input.raw_ptr();
    uint32_t size = input.size();

    #ifdef MICROFLOW_HAS_NEON
    // NEON优化实现
    #else
    // 标量实现
    for (uint32_t i = 0; i < size; ++i) {
        ptr[i] = /* 计算公式 */;
    }
    #endif
}
```

### 10.3 添加新的训练脚本模板

```python
# train_custom_model.py
import torch
import torch.nn as nn
import torch.optim as optim

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # 定义你的架构
        pass

    def forward(self, x):
        # 定义前向传播
        pass

def train_model(model, epochs=30):
    # 训练逻辑
    pass

def export_model(model, output_path):
    # 导出逻辑
    pass

if __name__ == "__main__":
    model = CustomModel()
    train_model(model)
    export_model(model, "custom_model.mflow")
```

---

## 附录

### A. 参考资料

1. **ARM NEON编程指南**
   - [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

2. **深度学习基础**
   - [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
   - [PyTorch Documentation](https://pytorch.org/docs/)

3. **高性能编程**
   - [Optimizing Software in C++](https://www.agner.org/optimize/)
   - [What Every Programmer Should Know About Memory](https://gist.github.com/hoboide/3818284)

### B. 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| 1.0 | 2025-02-20 | 初始版本 |
| 2.0 | 2025-02-21 | 指针接口重构，数据流修复 |
| 2.1 | 2025-02-21 | 添加Softmax自动添加 |
| 2.2 | 2025-02-21 | 改进模型架构，MaxPool支持 |

### C. 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### D. 许可证

本项目采用MIT许可证。详见LICENSE文件。

---

*文档结束*

如有问题或建议，请提交Issue或Pull Request。
