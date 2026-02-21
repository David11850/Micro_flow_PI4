#!/usr/bin/env python3
"""
重新导出模型 - 优化版本
优化：Linear层权重已转置，消除C++运行时转置开销
在有PyTorch的环境中运行此脚本
"""

import torch
import torch.nn as nn
import struct
import numpy as np
import sys

class ImprovedMNISTModel(nn.Module):
    def __init__(self):
        super(ImprovedMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def write_tensor(f, data):
    shape = list(data.shape)
    padded = shape + [0] * (4 - len(shape))
    desc = struct.pack('IIIIIIII', len(shape), *padded, 0, data.size, 0)
    f.write(desc)
    f.write(data.astype(np.float32).tobytes())

print("Loading trained model...")
model = ImprovedMNISTModel()

# 尝试加载checkpoint
try:
    checkpoint = torch.load('model_checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded from model_checkpoint.pth")
except:
    # 如果没有checkpoint文件，使用随机权重（仅用于测试）
    print("Warning: No checkpoint found, using random weights")
    print("Please run train_and_export.py first to train the model")

print("\nExporting model to mnist_improved.mflow...")

with open('mnist_improved.mflow', 'wb') as f:
    num_layers = 11  # 11层，不包含Softmax（C++会自动添加）
    num_tensors = 8
    header_offset = 88 + num_layers * 24

    f.write(struct.pack('IIIIII64s',
                       0x4D464C57,           # Magic
                       3,                    # Version 3 = 优化版本（转置权重）
                       num_layers,            # Num layers (11)
                       num_tensors,           # Num tensors (8)
                       header_offset,         # Data offset
                       0,                     # Data size
                       b"MicroFlow_OptimizedMNIST".ljust(64, b'\x00')))

    # Layer 0: Input
    f.write(struct.pack('IIIIII', 0, 0, 1, 1, 0, 0))  # kInput=0

    # Layer 1: Conv1 (1 -> 32, 3x3)
    f.write(struct.pack('IIIIII', 1, 0, 1, 1, 0, 0))  # kConv2D=1
    write_tensor(f, model.conv1.weight.detach().numpy())
    write_tensor(f, model.conv1.bias.detach().numpy())

    # Layer 2: ReLU
    f.write(struct.pack('IIIIII', 5, 0, 1, 1, 0, 0))  # kReLU=5

    # Layer 3: MaxPool2D
    f.write(struct.pack('IIIIII', 9, 0, 1, 1, 0, 0))  # kMaxPool2D=9

    # Layer 4: Conv2 (32 -> 64, 3x3)
    f.write(struct.pack('IIIIII', 1, 0, 1, 1, 0, 0))  # kConv2D=1
    write_tensor(f, model.conv2.weight.detach().numpy())
    write_tensor(f, model.conv2.bias.detach().numpy())

    # Layer 5: ReLU
    f.write(struct.pack('IIIIII', 5, 0, 1, 1, 0, 0))  # kReLU=5

    # Layer 6: MaxPool2D
    f.write(struct.pack('IIIIII', 9, 0, 1, 1, 0, 0))  # kMaxPool2D=9

    # Layer 7: Flatten
    f.write(struct.pack('IIIIII', 14, 0, 1, 1, 0, 0))  # kFlatten=14

    # Layer 8: FC1 (3136 -> 128) - 优化：权重已转置
    f.write(struct.pack('IIIIII', 13, 0, 1, 1, 0, 0))  # kLinear=13
    # PyTorch格式: [128, 3136] -> 导出为转置格式 [3136, 128]
    # 这样C++运行时就不需要转置，大幅提升性能
    write_tensor(f, model.fc1.weight.detach().numpy().T)
    write_tensor(f, model.fc1.bias.detach().numpy())

    # Layer 9: ReLU
    f.write(struct.pack('IIIIII', 5, 0, 1, 1, 0, 0))  # kReLU=5

    # Layer 10: FC2 (128 -> 10) - 优化：权重已转置
    f.write(struct.pack('IIIIII', 13, 0, 1, 1, 0, 0))  # kLinear=13
    # PyTorch格式: [10, 128] -> 导出为转置格式 [128, 10]
    write_tensor(f, model.fc2.weight.detach().numpy().T)
    write_tensor(f, model.fc2.bias.detach().numpy())

print(f"\n✅ Model exported successfully!")
print(f"   Version: 3 (Optimized - transposed weights)")
print(f"   Layers: {num_layers} (C++ will auto-add Softmax)")
print(f"   Output: mnist_improved.mflow")
print(f"   ⚡ Optimization: Linear weights pre-transposed")
print(f"   ⚡ Expected speedup: ~2-3x faster inference")
print(f"\nCopy to Pi: scp mnist_improved.mflow pi@raspberrypi:~/code/microflow/pi4_optimized/models/")
