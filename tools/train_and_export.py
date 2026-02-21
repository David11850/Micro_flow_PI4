import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import struct
import numpy as np
import os

# ============================================================================
# 改进的 MNIST 模型 - LeNet-5 风格
# ============================================================================

class ImprovedMNISTModel(nn.Module):
    """
    改进的MNIST模型架构:
    - 两层卷积 + 池化
    - 全连接层
    - 参数量: ~240K
    """
    def __init__(self):
        super(ImprovedMNISTModel, self).__init__()

        # 第一层卷积块: 1 -> 32 通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # 第二层卷积块: 32 -> 64 通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 全连接层
        # 输入: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)  # 训练时使用，推理时关闭

        # 输出层
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层: Conv -> ReLU -> MaxPool
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)  # 28x28 -> 14x14

        # 第二层: Conv -> ReLU -> MaxPool
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)  # 14x14 -> 7x7

        # 展平
        x = x.view(x.size(0), -1)  # [batch, 64*7*7] = [batch, 3136]

        # 全连接层
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # 输出层 (返回logits，CrossEntropyLoss会处理softmax)
        x = self.fc2(x)

        return x

    def export_forward(self, x):
        """
        用于导出时验证的前向传播
        这个版本不使用Dropout（推理模式）
        """
        self.eval()
        with torch.no_grad():
            x = self.conv1(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2)

            x = self.conv2(x)
            x = torch.relu(x)
            x = torch.max_pool2d(x, 2)

            x = x.view(x.size(0), -1)

            x = self.fc1(x)
            x = torch.relu(x)

            x = self.fc2(x)

        return x

# ============================================================================
# 训练函数
# ============================================================================

def train_model(model, epochs=25, device='cpu'):
    print(f"🚀 开始训练 {epochs} 轮...")
    print(f"📊 设备: {device}")

    # 数据增强训练集
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),           # 随机旋转 ±10度
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        # 不使用 Normalize，保持 0-1 范围
    ])

    # 验证集不使用增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 加载数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                  transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=2)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        train_loss /= len(train_dataset)
        train_acc = 100. * train_correct / train_total

        # 验证阶段
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()

        test_loss /= len(test_dataset)
        test_acc = 100. * test_correct / test_total

        scheduler.step()

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:5.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            # 这里可以保存最佳模型状态

    print(f"\n✅ 训练完成！最佳测试准确率: {best_accuracy:.2f}%")
    return model

# ============================================================================
# 导出函数
# ============================================================================

def write_tensor(f, data):
    """写入张量到文件"""
    shape = list(data.shape)
    padded = shape + [0] * (4 - len(shape))
    desc = struct.pack('IIIIIIII', len(shape), *padded, 0, data.size, 0)
    f.write(desc)
    f.write(data.astype(np.float32).tobytes())

def export_model(model, model_path, image_path, device='cpu'):
    """
    导出模型到 .mflow 格式

    层类型枚举:
    kInput=0, kConv2D=1, kReLU=5, kMaxPool2D=9, kFlatten=14, kLinear=13, kSoftmax=17
    """
    print(f"\n📦 导出模型到: {model_path}")

    model.eval()

    # 融合BN到Conv（简化推理）
    # 我们可以导出融合后的权重，这样推理时不需要BN层
    # 但为了简单，这里暂时不融合

    with open(model_path, 'wb') as f:
        # Header: magic, version, num_layers, num_tensors, data_offset, data_size, desc[64]
        # 层数: 11层 (Input + Conv + ReLU + Pool + Conv + ReLU + Pool + Flatten + FC + ReLU + FC)
        # 张量数: 8个 (conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        num_layers = 11
        num_tensors = 8
        header_offset = 88 + num_layers * 24  # Header + LayerHeaders
        f.write(struct.pack('IIIIII64s',
                           0x4D464C57,           # Magic
                           2,                    # Version
                           num_layers,           # Num layers
                           num_tensors,          # Num tensors
                           header_offset,        # Data offset (will be updated)
                           0,                    # Data size
                           b"MicroFlow_ImprovedMNIST".ljust(64, b'\0')))

        # Layer 0: Input
        f.write(struct.pack('IIIIII', 0, 0, 1, 1, 0, 0))  # kInput=0

        # Layer 1: Conv1 (1 -> 32, 3x3, padding=1)
        f.write(struct.pack('IIIIII', 1, 0, 1, 1, 0, 0))  # kConv2D=1
        write_tensor(f, model.conv1.weight.detach().numpy())
        write_tensor(f, model.conv1.bias.detach().numpy())

        # Layer 2: ReLU
        f.write(struct.pack('IIIIII', 5, 0, 1, 1, 0, 0))  # kReLU=5

        # Layer 3: MaxPool2D (2x2, stride=2)
        f.write(struct.pack('IIIIII', 9, 0, 1, 1, 0, 0))  # kMaxPool2D=9

        # Layer 4: Conv2 (32 -> 64, 3x3, padding=1)
        f.write(struct.pack('IIIIII', 1, 0, 1, 1, 0, 0))  # kConv2D=1
        write_tensor(f, model.conv2.weight.detach().numpy())
        write_tensor(f, model.conv2.bias.detach().numpy())

        # Layer 5: ReLU
        f.write(struct.pack('IIIIII', 5, 0, 1, 1, 0, 0))  # kReLU=5

        # Layer 6: MaxPool2D (2x2, stride=2)
        f.write(struct.pack('IIIIII', 9, 0, 1, 1, 0, 0))  # kMaxPool2D=9

        # Layer 7: Flatten
        f.write(struct.pack('IIIIII', 14, 0, 1, 1, 0, 0))  # kFlatten=14

        # Layer 8: FC1 (3136 -> 128)
        # 优化：直接导出转置后的权重，避免C++运行时转置
        # PyTorch格式: [out, in] = [128, 3136]
        # 转置后: [in, out] = [3136, 128]
        f.write(struct.pack('IIIIII', 13, 0, 1, 1, 0, 0))  # kLinear=13
        write_tensor(f, model.fc1.weight.detach().numpy().T)  # 转置权重
        write_tensor(f, model.fc1.bias.detach().numpy())

        # Layer 9: ReLU
        f.write(struct.pack('IIIIII', 5, 0, 1, 1, 0, 0))  # kReLU=5

        # Layer 10: FC2 (128 -> 10) - 输出层
        # 优化：同样导出转置后的权重
        # PyTorch格式: [out, in] = [10, 128]
        # 转置后: [in, out] = [128, 10]
        f.write(struct.pack('IIIIII', 13, 0, 1, 1, 0, 0))  # kLinear=13
        write_tensor(f, model.fc2.weight.detach().numpy().T)  # 转置权重
        write_tensor(f, model.fc2.bias.detach().numpy())

        # 注意: 不导出Softmax层，让C++推理引擎自动添加

    # 导出测试图片（第一个测试样本，数字7）
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                   transform=transforms.ToTensor())
    img, label = test_dataset[0]

    with open(image_path, 'wb') as f:
        f.write(img.numpy().astype(np.float32).tobytes())

    print(f"📤 导出测试图片到: {image_path}")
    print(f"🏷️  测试图片标签: {label}")

    # 验证：用PyTorch预测，确保正确
    print(f"\n🔍 验证PyTorch预测...")
    model.eval()
    with torch.no_grad():
        test_input = img.unsqueeze(0).to(device)  # [1, 1, 28, 28]
        logits = model.export_forward(test_input)
        probs = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item() * 100

    print(f"   PyTorch预测: {predicted} (置信度: {confidence:.2f}%)")
    print(f"   实际标签: {label}")

    if predicted == label:
        print(f"   ✅ 预测正确！")
    else:
        print(f"   ⚠️  预测错误，模型可能需要更多训练")

    # 打印所有概率
    print(f"   完整概率分布:")
    for i in range(10):
        print(f"      Digit {i}: {probs[0][i].item()*100:5.2f}%", end="")
        # 打印条形图
        bar_len = int(probs[0][i].item() * 40)
        print(" " + "▪" * bar_len)

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("  MicroFlow MNIST 模型训练器")
    print("  改进版 LeNet-5 架构")
    print("=" * 60)

    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  使用设备: {device}")

    # 创建模型
    model = ImprovedMNISTModel()

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # 训练模型
    model = train_model(model, epochs=25, device=device)

    # 导出模型
    export_model(model, "mnist_improved.mflow", "../image/test_input.bin", device=device)

    print("\n" + "=" * 60)
    print("  ✅ 全部完成！")
    print("=" * 60)
    print(f"\n模型文件: mnist_improved.mflow")
    print(f"测试图片: ../image/test_input.bin")
    print(f"\n运行推理:")
    print(f"  cd build")
    print(f"  ./mnist_demo ../models/mnist_improved.mflow ../image/test_input.bin")
    print()

if __name__ == "__main__":
    main()
