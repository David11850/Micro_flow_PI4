/**
 * @file image_preprocess.cpp
 * @brief 图像预处理工具 - 帮助准备用于MNIST识别的图片
 */

#include "microflow/image.hpp"
#include <iostream>
#include <fstream>

using namespace microflow;

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <input.png> <output.bin>\n\n";
    std::cout << "This tool helps preprocess images for MNIST recognition.\n\n";
    std::cout << "Steps to prepare your image:\n";
    std::cout << "1. Take a photo of handwritten digit\n";
    std::cout << "2. Crop to just the digit (tight bounding box)\n";
    std::cout << "3. Ensure white background with black digit\n";
    std::cout << "4. Run this tool to create .bin file\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::cout << "Loading: " << input_path << "\n";

    // 加载图像
    Tensor img;
    if (!Image::load(input_path, img)) {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    auto shapes = img.shapes();
    std::cout << "Original: [" << shapes[0] << " channel, "
              << shapes[1] << "x" << shapes[2] << "]\n";

    // 调整大小到28x28
    Tensor resized;
    Image::resize(img, resized, 28, 28);
    std::cout << "Resized to 28x28\n";

    // 统计信息
    const float* ptr = resized.raw_ptr();
    float sum = 0, min = 1, max = 0;
    int nonzero = 0;
    for (int i = 0; i < 784; ++i) {
        float v = ptr[i];
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
        if (v > 0.1f) nonzero++;
    }
    float avg = sum / 784.0f;

    std::cout << "Stats: avg=" << avg << ", min=" << min << ", max=" << max
              << ", nonzero=" << nonzero << "/784\n";

    // 分析颜色
    if (avg > 0.5f) {
        std::cout << "Note: Image appears to be mostly white (light background)\n";
        std::cout << "      This is GOOD for MNIST!\n";
    } else {
        std::cout << "Note: Image appears to be mostly dark (dark background)\n";
        std::cout << "      Inverting colors...\n";
        Image::invert(resized);
    }

    // 保存为.bin格式
    std::ofstream out(output_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(resized.raw_ptr()), 784 * sizeof(float));
    out.close();

    std::cout << "\nSaved to: " << output_path << "\n";
    std::cout << "Test with: ./image_demo ../models/mnist_optimized.mflow " << output_path << "\n";

    return 0;
}
