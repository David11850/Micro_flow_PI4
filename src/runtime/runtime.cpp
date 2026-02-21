#include "microflow/runtime.hpp"
#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>

namespace microflow {

//==========================================================================
// 具体层实现
//==========================================================================

InputLayer::InputLayer(const std::string& name,
                      const std::vector<uint32_t>& shape)
    : name_(name), shape_(shape)
{
}

void InputLayer::forward(const std::vector<Tensor*>& inputs,
    std::vector<Tensor*>& outputs,
    float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    std::memcpy(outputs[0]->raw_ptr(), inputs[0]->raw_ptr(), inputs[0]->size() * sizeof(float));
}

std::vector<uint32_t> InputLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return shape_;
}

//==========================================================================

Conv2DLayer::Conv2DLayer(const std::string& name,
                        const Tensor& kernel,
                        const Tensor& bias,
                        const Conv2DParams& params)
    : name_(name)
    , kernel_(kernel)
    , bias_(bias)
    , params_(params)
    , workspace_size_(0)
{
    workspace_size_ = compute_conv_workspace_size(
        Tensor({1, 28, 28}), kernel_, params_);
}

void Conv2DLayer::forward(const std::vector<Tensor*>& inputs,
                         std::vector<Tensor*>& outputs,
                         float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    // 标准版本：Conv2D + Bias
    // 注意：即使fuse_relu_=true，我们也在这里统一处理
    // 真正的融合优化需要修改conv2d内部实现来在bias后立即应用ReLU
    conv2d(*inputs[0], kernel_, bias_, *outputs[0], params_, workspace);
}

size_t Conv2DLayer::workspace_size() const {
    return workspace_size_;
}

std::vector<uint32_t> Conv2DLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    if (input_shapes.empty() || input_shapes[0].size() != 3) {
        return {};
    }
    int H = input_shapes[0][1];
    int W = input_shapes[0][2];
    int K = params_.kernel_size;
    int F = kernel_.shapes()[0];
    int H_out = (H + 2 * params_.padding - K) / params_.stride + 1;
    int W_out = (W + 2 * params_.padding - K) / params_.stride + 1;
    return {static_cast<uint32_t>(F), static_cast<uint32_t>(H_out),
            static_cast<uint32_t>(W_out)};
}

//==========================================================================

DepthwiseConv2DLayer::DepthwiseConv2DLayer(const std::string& name,
                                          const Tensor& kernel,
                                          const Tensor& bias,
                                          const Conv2DParams& params)
    : name_(name), kernel_(kernel), bias_(bias), params_(params)
{
}

void DepthwiseConv2DLayer::forward(const std::vector<Tensor*>& inputs,
                                  std::vector<Tensor*>& outputs,
                                  float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    conv2d(*inputs[0], kernel_, bias_, *outputs[0], params_);
}

std::vector<uint32_t> DepthwiseConv2DLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    if (input_shapes.empty() || input_shapes[0].size() != 3) {
        return {};
    }
    int H = input_shapes[0][1];
    int W = input_shapes[0][2];
    int K = params_.kernel_size;
    int C = input_shapes[0][0];
    int H_out = (H + 2 * params_.padding - K) / params_.stride + 1;
    int W_out = (W + 2 * params_.padding - K) / params_.stride + 1;
    return {static_cast<uint32_t>(C), static_cast<uint32_t>(H_out),
            static_cast<uint32_t>(W_out)};
}

//==========================================================================

BatchNormLayer::BatchNormLayer(const std::string& name,
                              const Tensor& mean,
                              const Tensor& var,
                              const Tensor& gamma,
                              const Tensor& beta,
                              float eps)
    : name_(name), mean_(mean), var_(var), gamma_(gamma), beta_(beta), eps_(eps)
{
}

void BatchNormLayer::forward(const std::vector<Tensor*>& inputs,
                            std::vector<Tensor*>& outputs,
                            float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    inputs[0]->copy_to(outputs[0]->raw_ptr());
    batch_norm(*outputs[0], mean_, var_, gamma_, beta_, eps_);
}

std::vector<uint32_t> BatchNormLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return input_shapes.empty() ? std::vector<uint32_t>() : input_shapes[0];
}

//==========================================================================

ActivationLayer::ActivationLayer(const std::string& name, LayerType type)
    : name_(name), activation_type_(type)
{
}

void ActivationLayer::forward(const std::vector<Tensor*>& inputs,
                             std::vector<Tensor*>& outputs,
                             float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    if (inputs[0]->raw_ptr() != outputs[0]->raw_ptr()) {
        inputs[0]->copy_to(outputs[0]->raw_ptr());
    }
    switch (activation_type_) {
        case LayerType::kReLU:
            relu(*outputs[0]);
            break;
        case LayerType::kReLU6:
            relu6(*outputs[0]);
            break;
        case LayerType::kLeakyReLU:
            leaky_relu(*outputs[0], 0.01f);
            break;
        case LayerType::kELU:
            elu(*outputs[0]);
            break;
        case LayerType::kGELU:
            gelu(*outputs[0]);
            break;
        case LayerType::kSigmoid:
            sigmoid(*outputs[0]);
            break;
        default:
            break;
    }
}

std::vector<uint32_t> ActivationLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return input_shapes.empty() ? std::vector<uint32_t>() : input_shapes[0];
}

//==========================================================================

PoolingLayer::PoolingLayer(const std::string& name, LayerType type,
                          int kernel_size, int stride, int padding)
    : name_(name)
    , pool_type_(type)
    , kernel_size_(kernel_size)
    , stride_(stride == 0 ? kernel_size : stride)
    , padding_(padding)
{
}

void PoolingLayer::forward(const std::vector<Tensor*>& inputs,
                          std::vector<Tensor*>& outputs,
                          float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    switch (pool_type_) {
        case LayerType::kMaxPool2D:
            max_pool2d(*inputs[0], *outputs[0], kernel_size_, stride_, padding_);
            break;
        case LayerType::kAvgPool2D:
            avg_pool2d(*inputs[0], *outputs[0], kernel_size_, stride_, padding_);
            break;
        case LayerType::kGlobalAvgPool2D:
            global_avg_pool2d(*inputs[0], *outputs[0]);
            break;
        case LayerType::kAdaptiveAvgPool2D:
            adaptive_avg_pool2d(*inputs[0], *outputs[0],
                               outputs[0]->shapes()[1],
                               outputs[0]->shapes()[2]);
            break;
        default:
            break;
    }
}

std::vector<uint32_t> PoolingLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    if (input_shapes.empty() || input_shapes[0].size() != 3) {
        return {};
    }
    int C = input_shapes[0][0];
    int H = input_shapes[0][1];
    int W = input_shapes[0][2];
    if (pool_type_ == LayerType::kGlobalAvgPool2D) {
        return {static_cast<uint32_t>(C), 1, 1};
    }
    int H_out = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
    int W_out = (W + 2 * padding_ - kernel_size_) / stride_ + 1;
    return {static_cast<uint32_t>(C), static_cast<uint32_t>(H_out),
            static_cast<uint32_t>(W_out)};
}

//==========================================================================

LinearLayer::LinearLayer(const std::string& name,
                        const Tensor& weight,
                        const Tensor& bias)
    : name_(name), weight_(weight), bias_(bias)
{
}

void LinearLayer::forward(const std::vector<Tensor*>& inputs,
                         std::vector<Tensor*>& outputs,
                         float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    linear(*inputs[0], weight_, bias_, *outputs[0]);
}

std::vector<uint32_t> LinearLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    // 自动检测权重格式并返回正确的输出形状
    if (weight_.ndim() == 2 && !input_shapes.empty()) {
        uint32_t in_features = 1;
        for (auto dim : input_shapes[0]) {
            in_features *= dim;
        }
        // 如果权重第一个维度等于输入特征数，是优化格式 [in, out]
        if (weight_.shapes()[0] == in_features) {
            return { weight_.shapes()[1] };
        }
    }
    // 原始格式 [out, in]
    return { weight_.shapes()[0] };
}

//==========================================================================

ReshapeLayer::ReshapeLayer(const std::string& name,
                          const std::vector<uint32_t>& shape)
    : name_(name), shape_(shape)
{
}

void ReshapeLayer::forward(const std::vector<Tensor*>& inputs,
                          std::vector<Tensor*>& outputs,
                          float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    outputs[0]->set_view_of(*inputs[0], shape_);
}

std::vector<uint32_t> ReshapeLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return shape_;
}

//==========================================================================

FlattenLayer::FlattenLayer(const std::string& name)
    : name_(name)
{
}

void FlattenLayer::forward(const std::vector<Tensor*>& inputs,
    std::vector<Tensor*>& outputs,
    float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    std::memcpy(outputs[0]->raw_ptr(), inputs[0]->raw_ptr(), inputs[0]->size() * sizeof(float));
}

std::vector<uint32_t> FlattenLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    if (input_shapes.empty()) {
        return {};
    }
    uint32_t size = 1;
    for (auto dim : input_shapes[0]) {
        size *= dim;
    }
    return {size};
}

//==========================================================================

SoftmaxLayer::SoftmaxLayer(const std::string& name, int axis)
    : name_(name), axis_(axis)
{
}

void SoftmaxLayer::forward(const std::vector<Tensor*>& inputs,
                          std::vector<Tensor*>& outputs,
                          float* workspace)
{
    if (inputs.empty() || outputs.empty()) return;
    if (inputs[0]->raw_ptr() != outputs[0]->raw_ptr()) {
        inputs[0]->copy_to(outputs[0]->raw_ptr());
    }
    softmax(*outputs[0], axis_);
}

std::vector<uint32_t> SoftmaxLayer::output_shape(
    const std::vector<std::vector<uint32_t>>& input_shapes
) const {
    return input_shapes.empty() ? std::vector<uint32_t>() : input_shapes[0];
}

//==========================================================================
// 模型实现
//==========================================================================

static Tensor read_tensor_from_file(std::ifstream& file) {
    TensorDesc desc;
    file.read(reinterpret_cast<char*>(&desc), sizeof(TensorDesc));
    std::vector<uint32_t> shape;
    for (uint32_t i = 0; i < desc.ndim; ++i) {
        shape.push_back(desc.shapes[i]);
    }
    Tensor tensor(shape);
    if (desc.size > 0) {
        file.read(reinterpret_cast<char*>(tensor.raw_ptr()), desc.size * sizeof(float));
    }
    return tensor;
}

Model::Model() : is_loaded_(false) {}

Model::~Model() = default;

bool Model::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    ModelHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(ModelHeader));
    if (header.magic != MFLOW_MAGIC) return false;
    std::cout << "Loading model: " << header.description << "\n";
    std::cout << "  Layers: " << header.num_layers << "\n";
    std::cout << "  Tensors: " << header.num_tensors << "\n";
    std::cout << "  Data offset: " << header.data_offset << "\n";
    input_shape_ = {1, 28, 28};
    bool has_softmax = false;
    for (uint32_t i = 0; i < header.num_layers; ++i) {
        LayerHeader lh;
        file.read(reinterpret_cast<char*>(&lh), sizeof(LayerHeader));
        std::cout << "  Layer " << i << ": type=" << static_cast<uint32_t>(lh.type) << " (";

        // 打印层类型名称
        const char* type_names[] = {"Input", "Conv2D", "Depthwise", "Pointwise", "BN", "ReLU",
                                     "ReLU6", "LeakyReLU", "ELU", "MaxPool", "AvgPool", "GAP", "AdaptAP",
                                     "Linear", "Flatten", "Reshape", "Concat", "Softmax", "Sigmoid"};
        if (static_cast<uint32_t>(lh.type) < 19) {
            std::cout << type_names[static_cast<uint32_t>(lh.type)];
        }
        std::cout << ")\n";

        if (lh.type == LayerType::kInput) {
            add_layer(std::make_unique<InputLayer>("input", input_shape_));
        }
        else if (lh.type == LayerType::kConv2D) {
            Tensor kernel = read_tensor_from_file(file);
            std::cout << "    Kernel shape: [";
            for (size_t i = 0; i < kernel.shapes().size(); ++i) {
                std::cout << kernel.shapes()[i];
                if (i < kernel.shapes().size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";

            Tensor bias = read_tensor_from_file(file);
            std::cout << "    Bias shape: [";
            for (size_t i = 0; i < bias.shapes().size(); ++i) {
                std::cout << bias.shapes()[i];
                if (i < bias.shapes().size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            Conv2DParams params{3, 1, 1};
            add_layer(std::make_unique<Conv2DLayer>("conv_" + std::to_string(i), kernel, bias, params));
        }
        else if (lh.type == LayerType::kReLU) {
            add_layer(std::make_unique<ActivationLayer>("relu_" + std::to_string(i), LayerType::kReLU));
        }
        else if (lh.type == LayerType::kGELU) {
            add_layer(std::make_unique<ActivationLayer>("gelu_" + std::to_string(i), LayerType::kGELU));
        }
        else if (lh.type == LayerType::kMaxPool2D) {
            add_layer(std::make_unique<PoolingLayer>("maxpool_" + std::to_string(i),
                                                     LayerType::kMaxPool2D, 2, 2, 0));
        }
        else if (lh.type == LayerType::kFlatten) {
            add_layer(std::make_unique<FlattenLayer>("flatten_" + std::to_string(i)));
        }
        else if (lh.type == LayerType::kLinear) {
            Tensor weight = read_tensor_from_file(file);
            std::cout << "    Weight shape: [";
            for (size_t i = 0; i < weight.shapes().size(); ++i) {
                std::cout << weight.shapes()[i];
                if (i < weight.shapes().size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            Tensor bias = read_tensor_from_file(file);
            std::cout << "    Bias shape: [";
            for (size_t i = 0; i < bias.shapes().size(); ++i) {
                std::cout << bias.shapes()[i];
                if (i < bias.shapes().size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            add_layer(std::make_unique<LinearLayer>("fc_" + std::to_string(i), weight, bias));
        }
        else if (lh.type == LayerType::kSoftmax) {
            has_softmax = true;
            // 检查是否是连续的重复Softmax层（跳过）
            bool is_duplicate = false;
            if (layers_.size() > 0) {
                if (layers_.back()->type() == LayerType::kSoftmax) {
                    is_duplicate = true;
                    std::cout << "    (Skipping duplicate Softmax layer)\n";
                }
            }
            if (!is_duplicate) {
                add_layer(std::make_unique<SoftmaxLayer>("softmax_" + std::to_string(i)));
            }
        }
    }
    // Auto-append softmax if not present
    if (!has_softmax) {
        add_layer(std::make_unique<SoftmaxLayer>("softmax_auto"));
        std::cout << "Auto-added Softmax layer for probability output\n";
    }

    // 注意：层融合优化已禁用
    // 真正的融合需要在conv2d/linear内核中实现，以避免额外的内存访问

    is_loaded_ = true;

    // 打印分配的张量信息
    std::cout << "\nAllocating intermediate tensors...\n";
    allocate_tensors();
    std::cout << "  Number of intermediate tensors: " << intermediate_tensors_.size() << "\n";
    for (size_t i = 0; i < intermediate_tensors_.size(); ++i) {
        std::cout << "    Tensor " << i << ": [";
        for (size_t j = 0; j < intermediate_tensors_[i].shapes().size(); ++j) {
            std::cout << intermediate_tensors_[i].shapes()[j];
            if (j < intermediate_tensors_[i].shapes().size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    return true;
}

static void write_tensor_to_file(std::ofstream& file, const Tensor& tensor) {
    TensorDesc desc;
    auto shapes = tensor.shapes();
    desc.ndim = static_cast<uint32_t>(shapes.size());
    std::memset(desc.shapes, 0, sizeof(desc.shapes));
    for (uint32_t i = 0; i < desc.ndim && i < 4; ++i) {
        desc.shapes[i] = shapes[i];
    }
    desc.dtype = DataType::kFloat32;
    desc.size = tensor.size();
    desc.offset = 0;
    file.write(reinterpret_cast<const char*>(&desc), sizeof(TensorDesc));
    file.write(reinterpret_cast<const char*>(tensor.raw_ptr()), desc.size * sizeof(float));
}

bool Model::save(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create model file: " << path << "\n";
        return false;
    }
    ModelHeader header;
    std::memset(&header, 0, sizeof(ModelHeader));
    header.magic = MFLOW_MAGIC;
    header.version = 2;
    header.num_layers = static_cast<uint32_t>(layers_.size());
    header.num_tensors = 0;
    std::strncpy(header.description, "MicroFlow_Exported_Model_V2", 63);
    file.write(reinterpret_cast<const char*>(&header), sizeof(ModelHeader));
    for (auto& layer : layers_) {
        LayerHeader lh;
        std::memset(&lh, 0, sizeof(LayerHeader));
        lh.type = layer->type();
        lh.input_count = 1;
        lh.output_count = 1;
        lh.param_size = 0;
        file.write(reinterpret_cast<const char*>(&lh), sizeof(LayerHeader));
        if (lh.type == LayerType::kConv2D) {
            auto* conv = dynamic_cast<Conv2DLayer*>(layer.get());
            if (conv) {
                write_tensor_to_file(file, conv->kernel_);
                write_tensor_to_file(file, conv->bias_);
            }
        }
        else if (lh.type == LayerType::kLinear) {
            auto* fc = dynamic_cast<LinearLayer*>(layer.get());
            if (fc) {
                write_tensor_to_file(file, fc->weight_);
                write_tensor_to_file(file, fc->bias_);
            }
        }
    }
    file.close();
    std::cout << "Model successfully saved to " << path << std::endl;
    return true;
}

void Model::add_layer(std::unique_ptr<Layer> layer) {
    layer_map_[layer->name()] = layer.get();
    layers_.push_back(std::move(layer));
}

Layer* Model::get_layer(const std::string& name) {
    auto it = layer_map_.find(name);
    return it != layer_map_.end() ? it->second : nullptr;
}

// 关键修复：不使用 push_back，直接通过 intermediate_tensors_ 传递数据
void Model::forward(const Tensor& input, Tensor& output) {
    if (!is_loaded_ || layers_.empty()) {
        std::cerr << "ERROR: Model not loaded or has no layers!\n";
        return;
    }
    if (intermediate_tensors_.empty()) allocate_tensors();
    if (workspace_.empty()) {
        workspace_.resize(compute_workspace_size() / sizeof(float) + 1024);
    }
    if (!input.is_valid()) {
        std::cerr << "ERROR: Input tensor is invalid!\n";
        return;
    }
    if (input.shapes() != input_shape_) {
        std::cerr << "ERROR: Input shape mismatch!\n";
        return;
    }

    static bool debug_enabled = false;  // Set to true for debugging

    // 第一层：input -> intermediate_tensors_[0]
    // 使用指针向量避免深拷贝
    std::vector<Tensor*> in_vec = {const_cast<Tensor*>(&input)};
    std::vector<Tensor*> out_vec = {&intermediate_tensors_[0]};
    layers_[0]->forward(in_vec, out_vec, workspace_.data());

    if (debug_enabled) {
        // Print stats after first conv layer
        std::cout << "\n=== Layer 1 (Conv2D) output stats ===\n";
        const float* ptr = intermediate_tensors_[1].raw_ptr();  // After Conv2D
        float min_val = ptr[0], max_val = ptr[0];
        float sum = 0;
        for (size_t i = 0; i < intermediate_tensors_[1].size(); ++i) {
            if (ptr[i] < min_val) min_val = ptr[i];
            if (ptr[i] > max_val) max_val = ptr[i];
            sum += ptr[i];
        }
        std::cout << "  Min: " << min_val << ", Max: " << max_val << ", Mean: " << sum / intermediate_tensors_[1].size() << "\n";
        std::cout << "  First 5 values: [" << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << ", " << ptr[3] << ", " << ptr[4] << "]\n";
    }

    // 后续层：intermediate_tensors_[i-1] -> intermediate_tensors_[i]
    for (size_t i = 1; i < layers_.size(); ++i) {
        in_vec = {&intermediate_tensors_[i-1]};
        out_vec = {&intermediate_tensors_[i]};
        layers_[i]->forward(in_vec, out_vec, workspace_.data());

        if (debug_enabled && i == 8) {
            // Print stats after first Linear layer (Layer 8)
            std::cout << "\n=== Layer " << i << " (Linear 3136->128) output stats ===\n";
            const float* ptr = intermediate_tensors_[i].raw_ptr();
            float min_val = ptr[0], max_val = ptr[0];
            float sum = 0;
            for (size_t j = 0; j < intermediate_tensors_[i].size(); ++j) {
                if (ptr[j] < min_val) min_val = ptr[j];
                if (ptr[j] > max_val) max_val = ptr[j];
                sum += ptr[j];
            }
            std::cout << "  Min: " << min_val << ", Max: " << max_val << ", Mean: " << sum / intermediate_tensors_[i].size() << "\n";
            std::cout << "  First 10 values: [";
            for (size_t j = 0; j < std::min(size_t(10), static_cast<size_t>(intermediate_tensors_[i].size())); ++j) {
                std::cout << ptr[j];
                if (j < 9) std::cout << ", ";
            }
            std::cout << "]\n";
        }

        if (debug_enabled && i == 10) {
            // Print stats after last Linear layer (before softmax)
            std::cout << "\n=== Layer " << i << " (Linear 128->10) output stats ===\n";
            const float* ptr = intermediate_tensors_[i].raw_ptr();
            float min_val = ptr[0], max_val = ptr[0];
            float sum = 0;
            for (size_t j = 0; j < intermediate_tensors_[i].size(); ++j) {
                if (ptr[j] < min_val) min_val = ptr[j];
                if (ptr[j] > max_val) max_val = ptr[j];
                sum += ptr[j];
            }
            std::cout << "  Min: " << min_val << ", Max: " << max_val << ", Mean: " << sum / intermediate_tensors_[i].size() << "\n";
            std::cout << "  Raw logits: [";
            for (size_t j = 0; j < intermediate_tensors_[i].size(); ++j) {
                std::cout << ptr[j];
                if (j < intermediate_tensors_[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            // Calculate expected softmax manually
            std::cout << "  Expected softmax: [";
            float exp_sum = 0;
            std::vector<float> exp_vals(10);
            for (size_t j = 0; j < 10; ++j) {
                exp_vals[j] = std::exp(ptr[j]);
                exp_sum += exp_vals[j];
            }
            for (size_t j = 0; j < 10; ++j) {
                std::cout << (exp_vals[j] / exp_sum);
                if (j < 9) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }

    intermediate_tensors_.back().copy_to(output.raw_ptr());
}

void Model::forward_batch(const std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs)
{
    for (size_t i = 0; i < inputs.size(); ++i) {
        forward(inputs[i], outputs[i]);
    }
}

std::vector<uint32_t> Model::input_shape() const {
    return input_shape_;
}

std::vector<uint32_t> Model::output_shape() const {
    return output_shape_;
}

auto Model::get_info() const -> Info {
    Info info;
    info.name = name_;
    info.description = "";
    info.num_layers = layers_.size();
    info.num_parameters = 0;
    info.model_size = 0;
    return info;
}

void Model::summary() const {
    std::cout << "========================================\n";
    std::cout << "Model Summary: " << name_ << "\n";
    std::cout << "========================================\n";
    std::cout << "Layers: " << layers_.size() << "\n";
    for (const auto& layer : layers_) {
        std::cout << "  - " << layer->name() << "\n";
    }
    std::cout << "========================================\n";
}

void Model::allocate_tensors() {
    intermediate_tensors_.clear();
    if (layers_.empty()) return;
    std::vector<uint32_t> current_shape = input_shape_;
    std::cout << "Allocating intermediate tensors...\n";
    for (size_t i = 0; i < layers_.size(); ++i) {
        std::vector<std::vector<uint32_t>> input_shapes = {current_shape};
        std::vector<uint32_t> out_shape = layers_[i]->output_shape(input_shapes);
        if (!out_shape.empty()) {
            intermediate_tensors_.push_back(Tensor(out_shape));
        }
        current_shape = out_shape;
    }
    output_shape_ = current_shape;
    std::cout << "Memory allocated for " << intermediate_tensors_.size() << " intermediate tensors.\n";
}

size_t Model::compute_workspace_size() {
    size_t max_size = 0;
    for (const auto& layer : layers_) {
        max_size = std::max(max_size, layer->workspace_size());
    }
    return max_size;
}

void Model::fuse_layers() {
}

//==========================================================================
// ModelBuilder 实现
//==========================================================================

ModelBuilder::ModelBuilder(const std::string& name) : name_(name) {}

ModelBuilder& ModelBuilder::input(const std::vector<uint32_t>& shape) {
    current_shape_ = shape;
    layers_.push_back(std::make_unique<InputLayer>("input", shape));
    return *this;
}

ModelBuilder& ModelBuilder::conv2d(const std::string& name, int out_channels, int kernel_size, int stride, int padding, bool bias) {
    Tensor kernel({static_cast<uint32_t>(out_channels), current_shape_[0], static_cast<uint32_t>(kernel_size), static_cast<uint32_t>(kernel_size)});
    Tensor b({static_cast<uint32_t>(out_channels)});
    Conv2DParams p{kernel_size, stride, padding};
    auto layer = std::make_unique<Conv2DLayer>(name, kernel, b, p);
    current_shape_ = layer->output_shape({current_shape_});
    layers_.push_back(std::move(layer));
    return *this;
}

ModelBuilder& ModelBuilder::depthwise_conv2d(const std::string& name, int kernel_size, int stride, int padding, bool bias) {
    uint32_t channels = current_shape_[0];
    Tensor kernel({channels, 1, static_cast<uint32_t>(kernel_size), static_cast<uint32_t>(kernel_size)});
    Tensor b({channels});
    Conv2DParams p{kernel_size, stride, padding};
    auto layer = std::make_unique<DepthwiseConv2DLayer>(name, kernel, b, p);
    current_shape_ = layer->output_shape({current_shape_});
    layers_.push_back(std::move(layer));
    return *this;
}

ModelBuilder& ModelBuilder::batch_norm(const std::string& name) {
    uint32_t channels = current_shape_[0];
    Tensor m({channels}), v({channels}), g({channels}), b({channels});
    layers_.push_back(std::make_unique<BatchNormLayer>(name, m, v, g, b));
    return *this;
}

ModelBuilder& ModelBuilder::relu() {
    layers_.push_back(std::make_unique<ActivationLayer>("relu_" + std::to_string(layers_.size()), LayerType::kReLU));
    return *this;
}

ModelBuilder& ModelBuilder::relu6() {
    layers_.push_back(std::make_unique<ActivationLayer>("relu6_" + std::to_string(layers_.size()), LayerType::kReLU6));
    return *this;
}

ModelBuilder& ModelBuilder::gelu() {
    layers_.push_back(std::make_unique<ActivationLayer>("gelu_" + std::to_string(layers_.size()), LayerType::kGELU));
    return *this;
}

ModelBuilder& ModelBuilder::leaky_relu(float alpha) {
    layers_.push_back(std::make_unique<ActivationLayer>("leaky_" + std::to_string(layers_.size()), LayerType::kLeakyReLU));
    return *this;
}

ModelBuilder& ModelBuilder::max_pool(int kernel_size, int stride, int padding) {
    auto layer = std::make_unique<PoolingLayer>("maxpool_" + std::to_string(layers_.size()), LayerType::kMaxPool2D, kernel_size, stride, padding);
    current_shape_ = layer->output_shape({current_shape_});
    layers_.push_back(std::move(layer));
    return *this;
}

ModelBuilder& ModelBuilder::avg_pool(int kernel_size, int stride, int padding) {
    auto layer = std::make_unique<PoolingLayer>("avgpool_" + std::to_string(layers_.size()), LayerType::kAvgPool2D, kernel_size, stride, padding);
    current_shape_ = layer->output_shape({current_shape_});
    layers_.push_back(std::move(layer));
    return *this;
}

ModelBuilder& ModelBuilder::global_avg_pool() {
    auto layer = std::make_unique<PoolingLayer>("gap", LayerType::kGlobalAvgPool2D, 0, 0, 0);
    current_shape_ = layer->output_shape({current_shape_});
    layers_.push_back(std::move(layer));
    return *this;
}

ModelBuilder& ModelBuilder::linear(const std::string& name, int out_features, bool bias) {
    uint32_t in_features = 1;
    for (auto d : current_shape_) in_features *= d;
    Tensor w({(uint32_t)out_features, in_features});
    Tensor b({(uint32_t)out_features});
    auto layer = std::make_unique<LinearLayer>(name, w, b);
    current_shape_ = layer->output_shape({current_shape_});
    layers_.push_back(std::move(layer));
    return *this;
}

ModelBuilder& ModelBuilder::flatten() {
    auto layer = std::make_unique<FlattenLayer>("flatten");
    current_shape_ = layer->output_shape({current_shape_});
    layers_.push_back(std::move(layer));
    return *this;
}

ModelBuilder& ModelBuilder::reshape(const std::vector<uint32_t>& shape) {
    current_shape_ = shape;
    layers_.push_back(std::make_unique<ReshapeLayer>("reshape", shape));
    return *this;
}

ModelBuilder& ModelBuilder::softmax(int axis) {
    layers_.push_back(std::make_unique<SoftmaxLayer>("softmax", axis));
    return *this;
}

Model ModelBuilder::build() {
    Model model;
    model.name_ = name_;
    if (!layers_.empty()) {
        model.input_shape_ = static_cast<InputLayer*>(layers_[0].get())->shape_;
    }
    for (auto& layer : layers_) {
        model.add_layer(std::move(layer));
    }
    model.is_loaded_ = true;
    model.allocate_tensors();
    return model;
}

//==========================================================================
// 推理引擎实现
//==========================================================================

InferenceEngine::InferenceEngine(const Config& config)
    : config_(config)
{
    std::memset(&stats_, 0, sizeof(stats_));
}

bool InferenceEngine::load_model(const std::string& path) {
    return model_.load(path);
}

Tensor InferenceEngine::infer(const Tensor& input) {
    auto start = std::chrono::high_resolution_clock::now();
    Tensor output = Tensor::zeros(model_.output_shape());
    model_.forward(input, output);
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    inference_times_.push_back(time_ms);
    stats_.num_inferences++;
    return output;
}

std::vector<Tensor> InferenceEngine::infer_batch(
    const std::vector<Tensor>& inputs)
{
    std::vector<Tensor> outputs(inputs.size());
    #pragma omp parallel for
    for (size_t i = 0; i < inputs.size(); ++i) {
        outputs[i] = infer(inputs[i]);
    }
    return outputs;
}

auto InferenceEngine::get_stats() const -> Stats {
    Stats stats = stats_;
    if (!inference_times_.empty()) {
        stats.total_time_ms = 0;
        stats.min_time_ms = inference_times_[0];
        stats.max_time_ms = inference_times_[0];
        for (double t : inference_times_) {
            stats.total_time_ms += t;
            stats.min_time_ms = std::min(stats.min_time_ms, t);
            stats.max_time_ms = std::max(stats.max_time_ms, t);
        }
        stats.avg_time_ms = stats.total_time_ms / inference_times_.size();
        stats.throughput = 1000.0 / stats.avg_time_ms;
    }
    return stats;
}

void InferenceEngine::reset_stats() {
    std::memset(&stats_, 0, sizeof(stats_));
    inference_times_.clear();
}

std::vector<Tensor> InferenceEngine::get_intermediate_outputs() const {
    return model_.intermediate_tensors_;
}

//==========================================================================
// 辅助函数
//==========================================================================

Model create_mnist_model() {
    Model model;
    model.name_ = "MNIST_Classifier";
    return model;
}

Model create_mobilenet_v2_model(int alpha) {
    Model model;
    model.name_ = "MobileNetV2";
    return model;
}

Model create_simple_cnn_model(const std::vector<uint32_t>& input_shape,
                             int num_classes)
{
    Model model;
    model.name_ = "SimpleCNN";
    return model;
}

} // namespace microflow
