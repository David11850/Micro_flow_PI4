#ifndef MICROFLOW_LAYERS_HPP
#define MICROFLOW_LAYERS_HPP

#include "microflow/tensor.hpp"
#include "microflow/gemm.hpp"
#include <cstdint>
#include <cmath>

namespace microflow {

//==========================================================================
// 激活函数
//==========================================================================

/**
 * @brief ReLU激活函数
 *
 * @公式: f(x) = max(0, x)
 *
 * @优势:
 * - 计算简单
 * - 缓解梯度消失
 * - 稀疏激活
 *
 * @优化点:
 * - NEON向量化比较
 * - 无分支实现
 * - 就地操作避免内存分配
 */
void relu(Tensor& input);

/**
 * @brief ReLU6激活函数 (MobileNet使用)
 *
 * @公式: f(x) = min(max(0, x), 6)
 *
 * @用途:
 * - 量化友好 (输出范围固定)
 * - MobileNetV2/V3使用
 *
 * @优化:
 * - 单次NEON指令链: clamp(val, 0, 6)
 */
void relu6(Tensor& input);

/**
 * @brief Leaky ReLU
 *
 * @公式: f(x) = x >= 0 ? x : alpha * x
 *
 * @param input 输入张量
 * @param alpha 负斜率 (通常0.01)
 */
void leaky_relu(Tensor& input, float alpha = 0.01f);

/**
 * @brief ELU (Exponential Linear Unit)
 *
 * @公式: f(x) = x >= 0 ? x : alpha * (exp(x) - 1)
 *
 * @优势:
 * - 输出均值接近0
 * - 缓解梯度消失
 */
void elu(Tensor& input, float alpha = 1.0f);

/**
 * @brief GELU (Gaussian Error Linear Unit)
 *
 * @公式: f(x) = x * Phi(x) 其中Phi是标准正态分布的CDF
 *
 * @近似: f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * @用途:
 * - Transformer模型使用
 * - BERT, GPT等
 */
void gelu(Tensor& input);

/**
 * @brief Sigmoid激活函数
 *
 * @公式: f(x) = 1 / (1 + exp(-x))
 *
 * @优化:
 * - 查表法 (LUT)
 * - 多项式近似
 */
void sigmoid(Tensor& input);

/**
 * @brief Tanh激活函数
 *
 * @公式: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 */
void tanh(Tensor& input);

/**
 * @brief Softmax激活函数
 *
 * @公式: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 *
 * @param input 输入张量 [batch, features]
 * @param axis 计算softmax的维度
 *
 * @数值稳定性:
 * - 减去最大值: exp(x - max(x))
 * - 避免溢出
 */
void softmax(Tensor& input, int axis = -1);

/**
 * @brief LogSoftmax
 *
 * @公式: log_softmax(x) = x - log(sum(exp(x)))
 *
 * @优势:
 * - 直接用于交叉熵损失
 * - 数值更稳定
 */
void log_softmax(Tensor& input, int axis = -1);

//==========================================================================
// 池化层
//==========================================================================

/**
 * @brief 最大池化
 *
 * @param input 输入张量 [C, H, W]
 * @param output 输出张量 [C, H_out, W_out]
 * @param kernel_size 池化窗口大小
 * @param stride 步长 (通常等于kernel_size)
 * @param padding 填充
 *
 * @公式: output[c, h, w] = max(input[c, h*stride:h*stride+k, w*stride:w*stride+k])
 *
 * @优化:
 * - NEON向量化的最大值查找
 * - 软件预取
 * - 特殊化2x2和3x3池化
 */
void max_pool2d(const Tensor& input, Tensor& output,
               int kernel_size, int stride, int padding = 0);

/**
 * @brief 平均池化
 *
 * @公式: output[c, h, w] = mean(input[c, h*stride:h*stride+k, w*stride:w*stride+k])
 *
 * @优化:
 * - 累加使用NEON
 * - 延迟除法 (最后除以kernel_size^2)
 */
void avg_pool2d(const Tensor& input, Tensor& output,
               int kernel_size, int stride, int padding = 0);

/**
 * @brief 全局平均池化
 *
 * @detail:
 * 将每个通道的整个特征图池化为单个值
 * 输入: [C, H, W] -> 输出: [C, 1, 1]
 *
 * @用途:
 * - 分类网络的最终池化
 * - 注意力机制
 */
void global_avg_pool2d(const Tensor& input, Tensor& output);

/**
 * @brief 自适应平均池化
 *
 * @detail:
 * 输出指定尺寸的特征图
 *
 * @param output_size 目标尺寸 [H_out, W_out]
 */
void adaptive_avg_pool2d(const Tensor& input, Tensor& output,
                        int output_height, int output_width);

//==========================================================================
// 归一化层
//==========================================================================

/**
 * @brief BatchNorm推理模式
 *
 * @detail:
 * 推理时使用固定的统计量 (均值和方差)
 *
 * @公式: y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * @优化:
 * - 与前一层融合 (通常Conv)
 * - 预计算缩放因子
 *
 * @param input 输入张量
 * @param mean 训练时的均值 [C]
 * @param var 训练时的方差 [C]
 * @param gamma 缩放参数 [C]
 * @param beta 偏移参数 [C]
 * @param eps 数值稳定性常数
 */
void batch_norm(Tensor& input,
               const Tensor& mean,
               const Tensor& var,
               const Tensor& gamma,
               const Tensor& beta,
               float eps = 1e-5f);

/**
 * @brief LayerNorm
 *
 * @detail:
 * 对每个样本的所有通道进行归一化
 *
 * @公式: y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * @用途:
 * - Transformer
 * - RNN
 *
 * @param axis 归一化的维度
 */
void layer_norm(Tensor& input,
               const Tensor& gamma,
               const Tensor& beta,
               float eps = 1e-5f,
               int axis = -1);

/**
 * @brief GroupNorm
 *
 * @detail:
 * 将通道分组, 对每组进行归一化
 *
 * @param num_groups 分组数
 */
void group_norm(Tensor& input,
               const Tensor& gamma,
               const Tensor& beta,
               int num_groups,
               float eps = 1e-5f);

//==========================================================================
// 全连接层
//==========================================================================

/**
 * @brief 全连接层 (线性层)
 *
 * @detail:
 * Y = X * W^T + bias
 *
 * @param input 输入 [M, K]
 * @param weight 权重 [N, K]
 * @param bias 偏置 [N] (可选)
 * @param output 输出 [M, N]
 *
 * @优化:
 * - 直接调用GEMM
 * - bias加法向量化
 */
void linear(const Tensor& input,
           const Tensor& weight,
           const Tensor& bias,
           Tensor& output);

/**
 * @brief 带激活的线性层
 *
 * @融合: Linear + ReLU -> 单次内存遍历
 */
void linear_relu(const Tensor& input,
                const Tensor& weight,
                const Tensor& bias,
                Tensor& output);

//==========================================================================
// Dropout (推理时为空操作)
//==========================================================================

/**
 * @brief Dropout (仅训练时有效)
 *
 * @detail:
 * 推理时什么都不做
 * 保留接口是为了模型兼容性
 */
inline void dropout(Tensor& input, float dropout_ratio = 0.5f) {
    // 推理时: 空操作
    // 训练时: 随机置零并缩放
}

//==========================================================================
// 填充和裁剪
//==========================================================================

/**
 * @brief 2D填充
 *
 * @param input 输入张量 [C, H, W]
 * @param output 输出张量 [C, H+2*pad_h, W+2*pad_w]
 * @param pad_h 高度填充
 * @param pad_w 宽度填充
 * @param value 填充值 (通常为0)
 */
void pad2d(const Tensor& input, Tensor& output,
          int pad_h, int pad_w, float value = 0.0f);

/**
 * @brief 2D裁剪
 *
 * @detail:
 * 移除指定数量的边界像素
 */
void crop2d(const Tensor& input, Tensor& output,
           int crop_h, int crop_w);

//==========================================================================
// 形状变换
//==========================================================================

/**
 * @brief 展平张量
 *
 * @detail:
 * 将 [C, H, W] 展平为 [C*H*W]
 */
void flatten(const Tensor& input, Tensor& output);

/**
 * @brief 重塑张量
 *
 * @detail:
 * 改变张量形状而不改变数据
 */
inline Tensor reshape(const Tensor& input, const std::vector<uint32_t>& new_shape) {
    return input.reshape(new_shape);
}

/**
 * @brief 维度置换
 */
inline Tensor transpose(const Tensor& input, int dim0, int dim1) {
    return input.transpose(dim0, dim1);
}

/**
 * @brief 添加维度
 */
inline Tensor expand_dims(const Tensor& input, int dim) {
    return input.expand_dims(dim);
}

/**
 * @brief 挤压维度
 */
inline Tensor squeeze(const Tensor& input, int dim) {
    return input.squeeze(dim);
}

//==========================================================================
// 拼接和分割
//==========================================================================

/**
 * @brief 拼接张量
 *
 * @param tensors 要拼接的张量列表
 * @param output 输出张量
 * @param dim 拼接维度
 */
void concat(const std::vector<Tensor>& tensors, Tensor& output, int dim = 0);

/**
 * @brief 分割张量
 *
 * @param input 输入张量
 * @param outputs 输出张量列表
 * @param dim 分割维度
 * @param split_sizes 每份的大小 (如果为空, 则均匀分割)
 */
void split(const Tensor& input, std::vector<Tensor>& outputs,
          int dim, const std::vector<int>& split_sizes = {});

/**
 * @brief 堆叠张量
 *
 * @detail:
 * 沿新维度堆叠张量
 */
void stack(const std::vector<Tensor>& tensors, Tensor& output, int dim = 0);

//==========================================================================
// 上采样和下采样
//==========================================================================

/**
 * @brief 双线性插值上采样
 *
 * @param input 输入张量 [C, H, W]
 * @param output 输出张量 [C, H*scale, W*scale]
 * @param scale_factor 缩放因子
 *
 * @优化:
 * - 可分离的2D插值 (先H后W)
 * - 边界处理优化
 */
void upsample_bilinear(const Tensor& input, Tensor& output,
                      float scale_factor);

/**
 * @brief 最近邻上采样
 *
 * @detail:
 * 简单的上采样方法
 */
void upsample_nearest(const Tensor& input, Tensor& output,
                     int scale_factor);

/**
 * @brief 下采样 (通过stride)
 *
 * @detail:
 * 通过指定stride下采样
 */
void downsample(const Tensor& input, Tensor& output, int stride);

//==========================================================================
// 工具函数
//==========================================================================

/**
 * @brief 克隆张量
 */
Tensor clone(const Tensor& input);

/**
 * @brief 填充张量
 */
void fill(Tensor& input, float value);

/**
 * @brief 创建单位矩阵
 */
Tensor eye(int n);

/**
 * @brief 创建对角矩阵
 */
Tensor diag(const Tensor& input);

} // namespace microflow

#endif // MICROFLOW_LAYERS_HPP
