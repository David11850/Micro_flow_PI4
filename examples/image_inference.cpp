#include "microflow/runtime.hpp"
#include "microflow/image.hpp"
#include <iostream>

using namespace microflow;

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;

    // 加载图像
    Tensor input;
    if (!Image::load(argv[1], input)) return 1;

    // 加载模型
    Model model;
    if (!model.load("../models/mnist_optimized.mflow")) return 1;

    // 推理
    Tensor output;
    model.forward(input, output);

    // 结果
    const float* probs = output.raw_ptr();
    int predicted = 0;
    float max_prob = probs[0];
    for (int i = 1; i < 10; ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            predicted = i;
        }
    }

    std::cout << "Predicted: " << predicted << "\n";
    std::cout << "Confidence: " << (max_prob * 100.0f) << "%\n";

    return 0;
}
