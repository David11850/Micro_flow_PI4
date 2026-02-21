#ifndef MICROFLOW_TENSOR_HPP
#define MICROFLOW_TENSOR_HPP

#include "microflow/allocator.hpp"
#include <vector>
#include <memory>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace microflow {

/**
 * @brief 张量数据布局类型
 *
 * @设计考量:
 * - NCHW: Batch-Channel-Height-Width (CNN标准布局)
 * - NHWC: Batch-Height-Width-Channel (ARM优化布局，更适合NEON)
 * - CHW: Channel-Height-Width (单张图片)
 */
enum class DataLayout : uint8_t {
    kNCHW,  // [N, C, H, W]
    kNHWC,  // [N, H, W, C] - ARM友好
    kCHW,   // [C, H, W]
    kHWC,   // [H, W, C]
    kUnknown
};

/**
 * @brief 张量视图 - 零拷贝数据访问
 *
 * @设计考量:
 * - 不拥有数据，仅引用外部内存
 * - 支持切片、转置等操作
 * - 用于中间结果传递，避免内存复制
 *
 * @优化点:
 * - 零拷贝: 完全避免内存分配和复制
 * - 引用计数: 使用shared_ptr管理生命周期
 * - 视图链: 支持多层视图操作
 */
class TensorView {
public:
    TensorView() = default;

    /**
     * @brief 从原始指针创建视图
     *
     * @param data 数据指针
     * @param shapes 张量形状
     * @param layout 数据布局
     */
    TensorView(float* data, const std::vector<uint32_t>& shapes,
               DataLayout layout = DataLayout::kNCHW)
        : data_(data)
        , shapes_(shapes)
        , layout_(layout)
    {
        compute_strides();
    }

    /**
     * @brief 从Tensor创建视图
     */
    template<typename Tensor>
    explicit TensorView(Tensor& tensor)
        : data_(tensor.raw_ptr())
        , shapes_(tensor.shapes())
        , layout_(tensor.layout())
    {
        compute_strides();
    }

    // Getters
    float* data() { return data_; }
    const float* data() const { return data_; }
    const std::vector<uint32_t>& shapes() const { return shapes_; }
    const std::vector<uint32_t>& strides() const { return strides_; }
    DataLayout layout() const { return layout_; }
    uint32_t size() const { return size_; }

    /**
     * @brief 获取指定索引的元素
     *
     * @优化点:
     * - 内联函数
     * - 编译器优化后与直接数组访问等价
     */
    float& operator[](const std::vector<uint32_t>& indices) {
        return data_[compute_offset(indices)];
    }

    const float& operator[](const std::vector<uint32_t>& indices) const {
        return data_[compute_offset(indices)];
    }

    /**
     * @brief 创建子视图
     *
     * @param dim 切片维度
     * @param index 切片索引
     * @return TensorView 子视图
     *
     * @优化点:
     * - 零拷贝切片
     * - 仅更新形状和步长
     */
    TensorView slice(uint32_t dim, uint32_t index) const;

private:
    void compute_strides();
    uint32_t compute_offset(const std::vector<uint32_t>& indices) const;

    float* data_ = nullptr;
    std::vector<uint32_t> shapes_;
    std::vector<uint32_t> strides_;
    uint32_t size_ = 0;
    DataLayout layout_ = DataLayout::kUnknown;
};

/**
 * @brief Tensor - 核心数据结构
 *
 * @设计考量:
 * 1. 内存管理: 使用ArenaAllocator避免碎片
 * 2. 数据对齐: 64字节对齐以匹配缓存行
 * 3. 布局优化: 支持NHWC以提升ARM性能
 * 4. 视图模式: 支持零拷贝操作
 *
 * 树莓派4优化点:
 * - 缓存行对齐减少false sharing
 * - NEON友好的内存布局
 * - 预取友好的访问模式
 */
class Tensor {
public:
    using Ptr = std::shared_ptr<Tensor>;

    //==========================================================================
    // 构造函数
    //==========================================================================

    /**
     * @brief 默认构造 - 空张量
     */
    Tensor() = default;

    /**
     * @brief 拷贝构造函数 - 深拷贝数据
     *
     * @设计考量:
     * - 创建新的独立数据副本
     * - 不共享底层数据
     * - 确保forward过程中的数据安全
     */
    Tensor(const Tensor& other);

    /**
     * @brief 拷贝赋值运算符 - 深拷贝数据
     */
    Tensor& operator=(const Tensor& other);

    /**
     * @brief 移动构造函数
     */
    Tensor(Tensor&& other) noexcept = default;

    /**
     * @brief 移动赋值运算符
     */
    Tensor& operator=(Tensor&& other) noexcept = default;

    /**
     * @brief 创建指定形状的张量（零初始化）
     *
     * @param shapes 张量形状
     * @param layout 数据布局
     *
     * @优化点:
     * - 使用ArenaAllocator批量分配
     * - 内存对齐到64字节（缓存行）
     * - 零初始化使用memset优化
     */
    explicit Tensor(const std::vector<uint32_t>& shapes,
                   DataLayout layout = DataLayout::kNCHW);

    /**
     * @brief 创建指定形状的张量（指定填充值）
     */
    explicit Tensor(const std::vector<uint32_t>& shapes,
                   float fill_value,
                   DataLayout layout = DataLayout::kNCHW);

    /**
     * @brief 从外部内存创建张量（View模式）
     *
     * @param shapes 张量形状
     * @param external_ptr 外部内存指针
     * @param layout 数据布局
     *
     * @设计考量:
     * - 不拥有内存，使用no-op删除器
     * - 用于工作空间复用
     * - 避免中间结果的内存分配
     */
    Tensor(const std::vector<uint32_t>& shapes,
           float* external_ptr,
           DataLayout layout = DataLayout::kNCHW);

    /**
     * @brief 从数据创建张量（拷贝）
     *
     * @param shapes 张量形状
     * @param data 数据指针
     * @param layout 数据布局
     */
    Tensor(const std::vector<uint32_t>& shapes,
           const float* data,
           DataLayout layout = DataLayout::kNCHW);

    //==========================================================================
    // 数据访问
    //==========================================================================

    /**
     * @brief 获取原始数据指针
     *
     * @优化点:
     * - 返回const和非const版本
     * - 内联函数
     * - 用于后续SIMD操作
     */
    float* raw_ptr() { return data_.get(); }
    const float* raw_ptr() const { return data_.get(); }

    /**
     * @brief 获取指定维度的步长
     */
    uint32_t stride(uint32_t dim) const {
        return dim < strides_.size() ? strides_[dim] : 1;
    }

    /**
     * @brief 获取所有步长
     */
    const std::vector<uint32_t>& strides() const { return strides_; }

    /**
     * @brief 获取形状
     */
    const std::vector<uint32_t>& shapes() const { return shapes_; }

    /**
     * @brief 获取布局
     */
    DataLayout layout() const { return layout_; }

    /**
     * @brief 获取元素总数
     */
    uint32_t size() const { return size_; }

    /**
     * @brief 获取字节大小
     */
    size_t bytes() const { return size_ * sizeof(float); }

    /**
     * @brief 获取维度数
     */
    uint32_t ndim() const { return static_cast<uint32_t>(shapes_.size()); }

    //==========================================================================
    // 张量操作
    //==========================================================================

    /**
     * @brief 填充张量
     *
     * @param value 填充值
     *
     * @优化点:
     * - 使用std::fill（可能使用SIMD）
     * - 小张量使用循环展开
     */
    void fill(float value);

    /**
     * @brief 从外部数据拷贝
     *
     * @param src 源数据指针
     * @param count 拷贝元素数
     *
     * @优化点:
     * - 使用std::copy或memcpy
     * - 编译器自动优化为SIMD
     */
    void copy_from(const float* src, size_t count = 0);

    /**
     * @brief 拷贝到外部缓冲区
     */
    void copy_to(float* dst) const;

    /**
     * @brief 创建张量视图
     */
    TensorView view() {
        return TensorView(raw_ptr(), shapes_, layout_);
    }

    /**
     * @brief 设置为另一个张量的视图（共享数据）
     *
     * @param other 要引用的张量
     * @param new_shapes 新的形状
     *
     * @设计考量:
     * - 用于ReshapeLayer等需要零拷贝改变形状的场景
     * - 直接共享底层数据指针
     * - 不分配新内存
     */
    void set_view_of(const Tensor& other, const std::vector<uint32_t>& new_shapes);

    /**
     * @brief 重塑张量（不改变数据）
     *
     * @param new_shapes 新形状
     * @return Tensor 重塑后的张量（共享数据）
     *
     * @优化点:
     * - 零拷贝操作
     * - 仅更新形状和步长
     * - 总元素数必须相同
     */
    Tensor reshape(const std::vector<uint32_t>& new_shapes) const;

    /**
     * @brief 转置张量
     *
     * @param dim0 交换维度0
     * @param dim1 交换维度1
     * @return Tensor 转置后的张量
     *
     * @优化点:
     * - 2D转置: 零拷贝，仅更新步长
     * - 非连续转置: 创建新张量
     */
    Tensor transpose(uint32_t dim0, uint32_t dim1) const;

    /**
     * @brief 扩展维度
     *
     * @param dim 扩展位置
     * @return Tensor 扩展后的张量
     *
     * @优化点:
     * - 零拷贝
     * - 仅更新形状
     */
    Tensor expand_dims(uint32_t dim) const;

    /**
     * @brief 挤压维度（移除大小为1的维度）
     */
    Tensor squeeze(uint32_t dim) const;

    //==========================================================================
    // 调试和可视化
    //==========================================================================

    /**
     * @brief 打印张量元数据
     */
    void print_meta() const;

    /**
     * @brief 打印张量内容（用于调试）
     *
     * @param max_elements 最多打印的元素数
     */
    void print_content(size_t max_elements = 20) const;

    /**
     * @brief 验证张量是否有效
     */
    bool is_valid() const { return data_ != nullptr && size_ > 0; }

    /**
     * @brief 检查数据是否连续
     */
    bool is_contiguous() const;

    //==========================================================================
    // 静态工厂方法
    //==========================================================================

    /**
     * @brief 创建零张量
     */
    static Tensor zeros(const std::vector<uint32_t>& shapes,
                       DataLayout layout = DataLayout::kNCHW);

    /**
     * @brief 创建单位张量
     */
    static Tensor ones(const std::vector<uint32_t>& shapes,
                      DataLayout layout = DataLayout::kNCHW);

    /**
     * @brief 创建随机张量（测试用）
     */
    static Tensor randn(const std::vector<uint32_t>& shapes,
                       float mean = 0.0f, float std = 1.0f,
                       DataLayout layout = DataLayout::kNCHW);

private:
    /**
     * @brief 计算步长（Row-Major）
     *
     * 例如 [C, H, W] -> strides [H*W, W, 1]
     *
     * @优化点:
     * - 编译期计算静态张量的步长
     * - 运行时缓存计算结果
     */
    void compute_strides();

    /**
     * @brief 分配对齐的内存
     */
    void allocate_memory();

    std::vector<uint32_t> shapes_;      // 张量形状
    std::vector<uint32_t> strides_;     // 步长
    uint32_t size_ = 0;                 // 元素总数
    DataLayout layout_ = DataLayout::kNCHW;

    // 使用自定义删除器的智能指针
    // View模式: 使用空删除器
    // 普通模式: 使用aligned_free
    std::shared_ptr<float[]> data_;

    // 标记是否为视图模式（不拥有内存）
    bool is_view_ = false;
};

//==========================================================================
// 全局张量操作函数
//==========================================================================

/**
 * @brief 张量加法（广播）
 */
Tensor add(const Tensor& a, const Tensor& b);

/**
 * @brief 张量乘法（逐元素）
 */
Tensor mul(const Tensor& a, const Tensor& b);

/**
 * @brief 矩阵乘法
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief 拼接张量
 */
Tensor concat(const std::vector<Tensor>& tensors, uint32_t dim);

/**
 * @brief 分割张量
 */
std::vector<Tensor> split(const Tensor& tensor, uint32_t parts, uint32_t dim);

} // namespace microflow

#endif // MICROFLOW_TENSOR_HPP
