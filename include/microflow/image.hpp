#ifndef MICROFLOW_IMAGE_HPP
#define MICROFLOW_IMAGE_HPP

#include "microflow/tensor.hpp"
#include <string>
#include <vector>

namespace microflow {

/**
 * @brief 图像加载类
 *
 * 支持从文件加载图像并转换为Tensor格式
 */
class Image {
public:
    /**
     * @brief 从文件加载图像
     *
     * @param filename 图像文件路径
     * @param output 输出张量 [channels, height, width]
     * @param grayscale 是否转为灰度图
     * @return bool 成功返回true
     *
     * @支持格式:
     * - .bin - float32二进制格式 (MNIST格式)
     * - .pgm - 简单灰度图格式
     */
    static bool load(const std::string& filename,
                     Tensor& output,
                     bool grayscale = true);

    /**
     * @brief 从内存加载MNIST格式图像
     *
     * @param buffer 输入buffer
     * @param size buffer大小
     * @param output 输出张量 [1, 28, 28]
     * @return bool 成功返回true
     */
    static bool load_mnist_from_memory(const float* buffer,
                                       size_t size,
                                       Tensor& output);

    /**
     * @brief 调整图像大小到指定尺寸
     *
     * @param input 输入图像 [C, H, W]
     * @param output 输出图像 [C, new_h, new_w]
     * @param new_height 新高度
     * @param new_width 新宽度
     */
    static void resize(const Tensor& input,
                      Tensor& output,
                      uint32_t new_height,
                      uint32_t new_width);

    /**
     * @brief 转换为灰度图
     *
     * @param input 输入 RGB图像 [3, H, W]
     * @param output 输出灰度图 [1, H, W]
     */
    static void to_grayscale(const Tensor& input,
                             Tensor& output);

    /**
     * @brief 归一化图像像素到0-1
     *
     * @param image 输入/输出图像 (就地操作)
     * @param max_val 最大值 (255 for uint8)
     */
    static void normalize(Tensor& image, float max_val = 255.0f);

    /**
     * @brief 反转图像颜色 (黑底白字 <-> 白底黑字)
     *
     * @param image 输入/输出图像 (就地操作)
     */
    static void invert(Tensor& image);

    /**
     * @brief 居中裁剪图像
     *
     * @param input 输入图像
     * @param output 输出裁剪后的图像
     * @param crop_height 裁剪高度
     * @param crop_width 裁剪宽度
     */
    static void center_crop(const Tensor& input,
                           Tensor& output,
                           uint32_t crop_height,
                           uint32_t crop_width);

    /**
     * @brief 二值化图像 (Otsu自适应阈值)
     *
     * @param input 输入图像 [1, H, W]
     * @param output 输出二值化图像 [1, H, W]
     * @param threshold 阈值 (0-1), 如果为负数则使用Otsu算法自动计算
     */
    static void binarize(const Tensor& input,
                        Tensor& output,
                        float threshold = -1.0f);

    /**
     * @brief 自动裁剪图像边框 (去除空白区域)
     *
     * @param input 输入图像 [1, H, W]
     * @param output 输出裁剪后的图像
     * @param padding 在边界周围保留的像素数
     * @param threshold 用于判断"空白"的阈值
     */
    static void auto_crop(const Tensor& input,
                         Tensor& output,
                         uint32_t padding = 2,
                         float threshold = 0.1f);

    /**
     * @brief MNIST预处理完整流程
     *
     * 步骤:
     * 1. 灰度化 (如果是彩色图)
     * 2. 反色 (确保黑底白字)
     * 3. 二值化
     * 4. 自动裁剪边框
     * 5. 缩放到28x28
     *
     * @param input 输入图像
     * @param output 输出预处理后的图像 [1, 28, 28]
     */
    static void preprocess_mnist(const Tensor& input, Tensor& output);

private:
    /**
     * @brief 加载PGM格式灰度图
     */
    static bool load_pgm(const std::string& filename,
                        std::vector<uint8_t>& buffer,
                        int& width, int& height);

    /**
     * @brief 双线性插值调整大小
     */
    static void resize_bilinear(const float* input,
                               float* output,
                               int in_h, int in_w,
                               int out_h, int out_w,
                               int channels);
};

} // namespace microflow

#endif // MICROFLOW_IMAGE_HPP
