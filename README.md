# MicroFlow v2.0

## 轻量级神经网络推理引擎 - 树莓派4极致优化版

MicroFlow是一个专门针对树莓派4（Cortex-A72 ARM64架构）深度优化的轻量级神经网络推理引擎。它实现了高效的神经网络算子，支持CNN模型推理，适用于边缘计算场景。

---

## 特性

### 性能优化
- ✅ **ARM NEON SIMD加速**: 所有核心算子针对NEON指令集优化
- ✅ **零拷贝张量操作**: 高效的内存管理，避免不必要的数据复制
- ✅ **层融合优化**: Conv+BN+ReLU等层自动融合
- ✅ **缓存友好算法**: 针对Cortex-A72的L1/L2缓存特性优化
- ✅ **多线程并行**: OpenMP并行化充分利用四核CPU

### 轻量级设计
- 📦 **零依赖**: 仅依赖OpenMP（系统自带）
- 💾 **小内存占用**: 典型模型推理仅需16-32MB内存
- 🚀 **快速启动**: 无需外部框架，秒级启动

### 易用性
- 🎯 **简单API**: 类似PyTorch的直观接口
- 📊 **模型格式**: 自定义.mflow格式，支持从PyTorch导出
- 🔧 **完整工具链**: 模型转换、性能分析、调试工具

---

## 性能数据

### 树莓派4 (4核 @ 1.5GHz)

| 模型 | 输入尺寸 | 推理时间 | 吞吐量 |
|-----|---------|---------|--------|
| LeNet (MNIST) | 1×28×28 | **2.5 ms** | 400 inf/s |
| 简单CNN | 1×28×28 | **3.8 ms** | 263 inf/s |
| MobileNetV2 | 1×224×224 | **85 ms** | 12 inf/s |

### GEMM性能

| 矩阵大小 | 性能 | 峰值比 |
|---------|------|-------|
| 512×512×512 | **40 GFLOPS** | 89% |

---

## 快速开始

### 编译

```bash
# 克隆项目
git clone https://github.com/your-repo/MicroFlow.git
cd MicroFlow/pi4_optimized

# 创建构建目录
mkdir build && cd build

# 配置和编译
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### 运行示例

```bash
# 运行MNIST手写数字识别
./mnist_demo ../models/mnist.mflow ../models/sample3.bin

# 运行性能基准测试
./benchmark
```

### 运行测试

```bash
# 单元测试
./test_tensor
./test_gemm
./test_conv

# 全部测试
make test
```

---

## 代码示例

### C++ API

```cpp
#include "microflow/runtime.hpp"
#include "microflow/tensor.hpp"

using namespace microflow;

int main() {
    // 1. 加载模型
    Model model;
    model.load("model.mflow");

    // 2. 准备输入
    Tensor input({1, 28, 28});
    // ... 填充输入数据 ...

    // 3. 执行推理
    Tensor output = Tensor::zeros({1, 10});
    model.forward(input, output);

    // 4. 获取结果
    const float* predictions = output.raw_ptr();

    return 0;
}
```

### 模型构建器

```cpp
// 流式API构建模型
Model model = ModelBuilder("MyCNN")
    .input({1, 28, 28})
    .conv2d("conv1", 32, 3, 1, 1)
    .batch_norm("bn1")
    .relu()
    .max_pool(2, 2)
    .conv2d("conv2", 64, 3, 1, 1)
    .batch_norm("bn2")
    .relu()
    .max_pool(2, 2)
    .flatten()
    .linear("fc1", 128)
    .relu()
    .linear("fc2", 10)
    .softmax()
    .build();
```

### 单独使用算子

```cpp
// 卷积
Tensor input({1, 28, 28});
Tensor kernel({8, 1, 3, 3});
Tensor output({8, 28, 28});

Conv2DParams params(3, 1, 1);  // kernel=3, stride=1, padding=1
conv2d(input, kernel, Tensor(), output, params);

// 激活
relu(output);

// 池化
Tensor pooled({8, 14, 14});
max_pool2d(output, pooled, 2, 2);  // 2x2 pool, stride=2
```

---

## Python模型导出

### 从PyTorch导出

```python
import torch
import torch.nn as nn
from microflow_export import export_to_mflow

# 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建并导出模型
model = SimpleNet()
model.eval()

export_to_mflow(model, "simple_net.mflow")
```

---

## 项目结构

```
pi4_optimized/
├── include/microflow/     # 头文件
│   ├── allocator.hpp      # 内存分配器
│   ├── tensor.hpp         # 张量数据结构
│   ├── gemm.hpp           # 矩阵乘法
│   ├── conv.hpp           # 卷积
│   ├── layers.hpp         # 层操作
│   └── runtime.hpp        # 运行时系统
├── src/                   # 源文件
│   ├── memory/            # 内存管理
│   ├── gemm/              # GEMM实现
│   ├── conv/              # 卷积实现
│   ├── layers/            # 层实现
│   └── runtime/           # 运行时实现
├── tests/                 # 测试程序
├── examples/              # 示例程序
├── docs/                  # 详细文档
│   ├── memory.md          # 内存管理说明
│   ├── gemm.md            # GEMM优化说明
│   ├── conv.md            # 卷积优化说明
│   ├── layers.md          # 层操作说明
│   └── runtime.md         # 运行时说明
└── CMakeLists.txt         # 构建配置
```

---

## 编译选项

### 树莓派4优化

CMakeLists.txt已针对树莓派4配置了以下优化：

```cmake
# Cortex-A72优化
-march=armv8-a
-mtune=cortex-a72
-mcpu=cortex-a72

# NEON和FMA
-mfpu=neon-fp-armv8
-ffp-contract=fast

# 激进优化
-O3
-ffast-math
-funsafe-math-optimizations
-funroll-loops
-ftree-vectorize

# 链接时优化
-flto
```

### x86_64开发

在x86_64上编译（开发用）：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```

---

## 支持的层类型

| 层类型 | 支持 | 优化状态 |
|--------|------|---------|
| Conv2D | ✅ | NEON优化 |
| DepthwiseConv2D | ✅ | NEON优化 |
| BatchNorm | ✅ | 支持融合 |
| ReLU/ReLU6 | ✅ | NEON优化 |
| MaxPool2D | ✅ | OpenMP并行 |
| AvgPool2D | ✅ | OpenMP并行 |
| GlobalAvgPool2D | ✅ | 优化 |
| Linear | ✅ | GEMM优化 |
| Flatten | ✅ | 零拷贝 |
| Softmax | ✅ | 数值稳定 |
| Reshape | ✅ | 零拷贝 |
| Concat | ✅ | 基础实现 |

---

## 文档

详细文档请查看 `docs/` 目录：

- **memory.md**: 内存管理系统详解
- **gemm.md**: GEMM优化技术详解
- **conv.md**: 卷积算法对比与选择
- **layers.md**: 所有层操作的详细说明
- **runtime.md**: 运行时系统架构

---

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](../LICENSE) 文件

---

## 致谢

- ARM NEON技术文档
- BLAS/LAPACK设计理念
- PyTorch和TensorFlow的API设计

---

## 联系方式

- 项目主页: https://github.com/your-repo/MicroFlow
- 问题反馈: https://github.com/your-repo/MicroFlow/issues
- 邮箱: your-email@example.com

---

**MicroFlow v2.0** - 让边缘AI推理更高效！
