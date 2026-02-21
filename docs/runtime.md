# 运行时系统 (Runtime System)

## 概述

运行时系统是MicroFlow的核心组件，负责模型加载、执行和资源管理。它提供了一个灵活、高效的推理执行引擎。

---

## 模型文件格式 (.mflow v2)

### 文件结构

```
+------------------+
|  File Header     |  魔数、版本、元数据
+------------------+
|  Layer Headers   |  每层的类型和参数
+------------------+
|  Tensor Descriptors | 权重张量描述
+------------------+
|  String Table    |  层名称等字符串
+------------------+
|  Weight Data     |  实际权重数据
+------------------+
```

### 文件头

```cpp
struct ModelHeader {
    uint32_t magic;           // 0x4D464C57 ("MFLW")
    uint32_t version;         // 版本号
    uint32_t num_layers;      // 层数
    uint32_t num_tensors;     // 权重张量数
    uint32_t data_offset;     // 数据区偏移
    uint32_t data_size;       // 数据区大小
    char description[64];     // 模型描述
};
```

### 层头

```cpp
struct LayerHeader {
    LayerType type;           // 层类型枚举
    uint32_t name_offset;     // 名称在字符串表中的偏移
    uint32_t input_count;     // 输入数量
    uint32_t output_count;    // 输出数量
    uint32_t param_size;      // 参数区大小
    uint32_t workspace_size;  // 需要的工作空间
};
```

### 设计优势

| 特性 | 优势 |
|-----|------|
| 魔数验证 | 防止加载错误文件 |
| 版本控制 | 向后兼容性 |
| 偏移量索引 | 快速随机访问 |
| 紧凑格式 | 减少文件大小 |

---

## 层系统

### 层基类

所有层都继承自`Layer`基类：

```cpp
class Layer {
public:
    virtual void forward(const std::vector<Tensor>& inputs,
                       std::vector<Tensor>& outputs,
                       float* workspace) = 0;

    virtual std::string name() const = 0;
    virtual LayerType type() const = 0;
    virtual size_t workspace_size() const = 0;
};
```

### 支持的层类型

| 层类型 | 枚举值 | 描述 |
|--------|--------|------|
| Input | 0 | 输入占位层 |
| Conv2D | 1 | 标准卷积 |
| DepthwiseConv2D | 2 | Depthwise卷积 |
| PointwiseConv2D | 3 | 1x1卷积 |
| BatchNorm | 4 | 批归一化 |
| ReLU | 5 | ReLU激活 |
| MaxPool2D | 9 | 最大池化 |
| AvgPool2D | 10 | 平均池化 |
| Linear | 13 | 全连接层 |
| Softmax | 17 | Softmax激活 |

### 层实现示例

```cpp
class Conv2DLayer : public Layer {
public:
    Conv2DLayer(const std::string& name,
               const Tensor& kernel,
               const Tensor& bias,
               const Conv2DParams& params)
        : name_(name)
        , kernel_(kernel)
        , bias_(bias)
        , params_(params)
    {}

    void forward(const std::vector<Tensor>& inputs,
                std::vector<Tensor>& outputs,
                float* workspace) override {
        conv2d(inputs[0], kernel_, bias_, outputs[0],
               params_, workspace);
    }

    // ...
};
```

---

## 推理流程

### 基本推理

```cpp
// 1. 加载模型
Model model;
model.load("model.mflow");

// 2. 准备输入
Tensor input = Tensor::zeros({1, 28, 28});
// 填充输入数据...

// 3. 准备输出
Tensor output = Tensor::zeros({1, 10});

// 4. 执行推理
model.forward(input, output);

// 5. 获取结果
float* result = output.raw_ptr();
```

### 推理引擎

```cpp
// 创建推理引擎
InferenceEngine::Config config;
config.num_threads = 4;
config.enable_profiling = true;

InferenceEngine engine(config);

// 加载模型
engine.load_model("model.mflow");

// 推理
Tensor input = ...;
Tensor output = engine.infer(input);

// 批量推理
std::vector<Tensor> inputs = ...;
auto outputs = engine.infer_batch(inputs);

// 获取性能统计
auto stats = engine.get_stats();
std::cout << "Average: " << stats.avg_time_ms << " ms\n";
std::cout << "Throughput: " << stats.throughput << " inf/s\n";
```

---

## 内存管理

### 工作空间计算

```cpp
size_t Model::compute_workspace_size() {
    size_t max_size = 0;
    for (const auto& layer : layers_) {
        max_size = std::max(max_size, layer->workspace_size());
    }
    return max_size;
}
```

### 中间张量管理

```cpp
void Model::allocate_tensors() {
    intermediate_tensors_.reserve(layers_.size());

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto output_shape = layers_[i]->output_shape(...);
        intermediate_tensors_.emplace_back(
            Tensor::zeros(output_shape)
        );
    }
}
```

### 内存复用策略

| 策略 | 描述 | 节省内存 |
|-----|------|---------|
| 就地操作 | 激活函数直接修改输入 | 50% |
| 张量复用 | 不相连的层共享内存 | 30% |
| 工作空间 | 共享临时缓冲区 | 20% |

---

## 层融合优化

### Conv+BN+ReLU融合

```
原始:
  conv_out = conv(input, weight, bias)
  bn_out = (conv_out - mean) / std * gamma + beta
  relu_out = max(0, bn_out)

融合后:
  weight' = weight * gamma / std
  bias' = (bias - mean) / std * gamma + beta
  output = max(0, conv(input, weight', bias'))
```

### 融合实现

```cpp
void Model::fuse_layers() {
    for (size_t i = 0; i < layers_.size() - 2; ++i) {
        // 检测 Conv + BN + ReLU 模式
        if (layers_[i]->type() == LayerType::kConv2D &&
            layers_[i+1]->type() == LayerType::kBatchNorm &&
            layers_[i+2]->type() == LayerType::kReLU) {

            // 获取层指针
            auto* conv = static_cast<Conv2DLayer*>(layers_[i].get());
            auto* bn = static_cast<BatchNormLayer*>(layers_[i+1].get());

            // 融合权重
            // ... 融合逻辑 ...

            // 标记BN和ReLU为已融合
            layers_[i+1]->set_fused(true);
            layers_[i+2]->set_fused(true);
        }
    }
}
```

---

## 模型构建器API

### 流式API示例

```cpp
Model model = ModelBuilder("MyCNN")
    .input({1, 28, 28})
    .conv2d("conv1", 32, 3, 1, 1)  // out_ch=32, k=3, s=1, p=1
    .batch_norm("bn1")
    .relu()
    .max_pool(2, 2)
    .conv2d("conv2", 64, 3, 1, 1)
    .batch_norm("bn2")
    .relu()
    .max_pool(2, 2)
    .flatten()
    .linear("fc", 128)
    .relu()
    .linear("classifier", 10)
    .softmax()
    .build();
```

### 构建器实现

```cpp
ModelBuilder& ModelBuilder::conv2d(const std::string& name,
                                  int out_channels,
                                  int kernel_size,
                                  int stride,
                                  int padding,
                                  bool bias)
{
    // 计算输出形状
    int C = current_shape_[0];
    int H = current_shape_[1];
    int W = current_shape_[2];
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    // 创建权重
    Tensor kernel({out_channels, C, kernel_size, kernel_size});
    Tensor bias({out_channels});

    // 创建层
    auto layer = std::make_unique<Conv2DLayer>(
        name, kernel, bias,
        Conv2DParams(kernel_size, stride, padding)
    );

    layers_.push_back(std::move(layer));
    current_shape_ = {out_channels, H_out, W_out};

    return *this;
}
```

---

## 性能优化

### 1. 图优化

```cpp
// 常量折叠
if (input.is_const()) {
    output = evaluate_const_layer(layer, input);
    skip_execution = true;
}

// 死代码消除
if (output_unused) {
    remove_layer(layer);
}
```

### 2. 内存规划

```cpp
// 简单的内存规划算法
struct TensorLifetime {
    size_t create_epoch;
    size_t last_use_epoch;
    size_t size;
};

void plan_memory() {
    // 分析每个张量的生命周期
    // 复用不重叠的张量内存
}
```

### 3. 算子选择

```cpp
void forward(...) {
    // 根据输入大小选择最优实现
    if (input.size() < 1024) {
        use_naive_implementation();
    } else if (has_neon()) {
        use_neon_implementation();
    } else {
        use_omp_implementation();
    }
}
```

---

## Python导出工具

### 模型转换脚本

```python
import torch
import struct

def export_pytorch_to_mflow(model, path):
    """将PyTorch模型导出为.mflow格式"""

    with open(path, 'wb') as f:
        # 1. 写入文件头
        header = struct.pack(
            'I', 0x4D464C57  # magic
        )
        f.write(header)

        # 2. 遍历模型层
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Conv2d):
                # 写入卷积层
                write_conv2d(f, name, module)
            elif isinstance(module, torch.nn.Linear):
                # 写入全连接层
                write_linear(f, name, module)

        # 3. 写入权重数据
        write_weights(f, model)

def write_conv2d(f, name, module):
    """写入卷积层定义"""
    # 层类型
    f.write(struct.pack('I', 1))  # Conv2D

    # 参数
    out_ch, in_ch, h, w = module.weight.shape
    f.write(struct.pack('I', out_ch))
    f.write(struct.pack('I', in_ch))
    f.write(struct.pack('I', h))
    # ...
```

### 使用示例

```python
import torch
import torch.nn as nn
from microflow_export import export_pytorch_to_mflow

# 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8*14*14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 导出
model = SimpleNet()
export_pytorch_to_mflow(model, 'simple_net.mflow')
```

---

## 调试工具

### 模型可视化

```cpp
void visualize_model(const Model& model) {
    std::cout << "Model Graph:\n";

    for (const auto& layer : model.layers()) {
        std::cout << "  [" << layer->name() << "]\n";
        std::cout << "    Type: " << layer_type_name(layer->type()) << "\n";
        std::cout << "    Input: " << format_shape(layer->input_shape()) << "\n";
        std::cout << "    Output: " << format_shape(layer->output_shape()) << "\n";
        std::cout << "    Params: " << layer->num_parameters() << "\n";
    }
}
```

### 性能分析

```cpp
struct LayerProfile {
    std::string name;
    double time_ms;
    size_t flops;
    double gflops;
};

std::vector<LayerProfile> profile_model(const Model& model) {
    std::vector<LayerProfile> profiles;

    for (const auto& layer : model.layers()) {
        LayerProfile prof;
        prof.name = layer->name();

        auto start = now();
        // 执行层
        auto end = now();

        prof.time_ms = (end - start).count();
        prof.flops = estimate_flops(layer);
        prof.gflops = prof.flops / prof.time_ms / 1e6;

        profiles.push_back(prof);
    }

    return profiles;
}
```

---

## 最佳实践

### 1. 模型优化

```python
# 导出前优化PyTorch模型
model.eval()

# 融合Conv+BN+ReLU
model = torch.quantization.fuse_modules(model,
    [['conv1', 'bn1', 'relu1']])

# 转换为更优化的格式
model = torch.jit.script(model)
```

### 2. 输入预处理

```cpp
// 预处理输入数据
void preprocess_input(Tensor& input) {
    // 归一化: (x - mean) / std
    float mean = 0.5f;
    float std = 0.5f;

    float* ptr = input.raw_ptr();
    for (uint32_t i = 0; i < input.size(); ++i) {
        ptr[i] = (ptr[i] - mean) / std;
    }
}
```

### 3. 批处理

```cpp
// 批量推理提升吞吐量
std::vector<Tensor> batch_inference(
    const std::vector<Tensor>& inputs)
{
    InferenceEngine engine;
    engine.load_model("model.mflow");

    return engine.infer_batch(inputs);
}
```

---

## 故障排查

### 常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 加载失败 | 文件格式错误 | 检查魔数和版本 |
| 输出NaN | 权重异常 | 验证模型训练 |
| 性能差 | 未使用NEON | 检查编译选项 |
| 内存不足 | 工作空间过大 | 减小batch size |

### 调试模式

```cpp
// 启用详细日志
#define MICROFLOW_DEBUG 1

// 验证每层输出
bool validate_layer_output(Layer* layer) {
    Tensor input = create_test_input();
    Tensor output = layer->forward(input);

    // 检查NaN/Inf
    for (uint32_t i = 0; i < output.size(); ++i) {
        if (std::isnan(output.raw_ptr()[i])) {
            return false;
        }
    }
    return true;
}
```
