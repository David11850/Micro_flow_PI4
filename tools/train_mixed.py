#!/usr/bin/env python3
"""
混合训练脚本 - MNIST标准数据 + 你的手写数据
使用方法:
1. 把所有CSV文件放在脚本同目录下
2. 运行: python3 train_mixed.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import struct
import numpy as np
import os
import csv

# ============================================================================
# 自定义数据集 - 加载CSV手写数据
# ============================================================================

class HandwrittenDataset(torch.utils.data.Dataset):
    """从CSV文件加载手写数字"""

    def __init__(self, csv_files):
        """
        Args:
            csv_files: 字典 {label: csv_file_path} 例如 {0: "testData0.csv", 1: "testData1.csv", ...}
        """
        self.data = []
        self.labels = []

        print("📂 加载手写数据...")

        for label, csv_file in csv_files.items():
            if not os.path.exists(csv_file):
                print(f"  ⚠️  文件不存在: {csv_file}")
                continue

            print(f"  读取 {csv_file}...")

            with open(csv_file, 'r') as f:
                reader = csv.reader(f)

                # 跳过标题行
                first_row = next(reader, None)
                if first_row and first_row[0].startswith('pixel'):
                    data_rows = list(reader)
                else:
                    if first_row:
                        data_rows = [first_row] + list(reader)
                    else:
                        data_rows = []

                for row in data_rows:
                    if len(row) < 784:
                        continue

                    # 检查是否有标签列
                    if len(row) >= 785:
                        row_label = int(row[0])
                        pixels = [float(x) for x in row[1:785]]
                    else:
                        pixels = [float(x) for x in row[0:784]]

                    # 转换为28x28数组，归一化
                    # MNIST网站数据格式: 0=黑色(墨水), 255=白色(背景)
                    # 与标准MNIST格式一致，无需反转
                    arr = np.array(pixels, dtype=np.float32).reshape(28, 28)
                    arr = arr / 255.0  # 归一化到0-1

                    self.data.append(arr)
                    self.labels.append(label)

        print(f"  ✅ 总共加载 {len(self.data)} 个手写样本")

        # 统计每个数字的数量
        label_counts = {}
        for lbl in self.labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(f"  分布: {label_counts}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # 转换为tensor [1, 28, 28]
        img = torch.from_numpy(img).unsqueeze(0).float()

        return img, label

# ============================================================================
# 模型定义
# ============================================================================

class ImprovedMNISTModel(nn.Module):
    def __init__(self):
        super(ImprovedMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def export_forward(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
        return x

# ============================================================================
# 训练函数
# ============================================================================

def train_model(model, train_loader, test_loader, epochs=25, device='cpu'):
    print(f"🚀 开始混合训练 {epochs} 轮...")
    print(f"📊 设备: {device}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        # 训练
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

        train_loss /= train_total
        train_acc = 100. * train_correct / train_total

        # 验证
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

        test_loss /= test_total
        test_acc = 100. * test_correct / test_total

        scheduler.step()

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:5.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc

    print(f"\n✅ 训练完成！最佳测试准确率: {best_accuracy:.2f}%")
    return model

def write_tensor(f, data):
    shape = list(data.shape)
    padded = shape + [0] * (4 - len(shape))
    desc = struct.pack('IIIIIIII', len(shape), *padded, 0, data.size, 0)
    f.write(desc)
    f.write(data.astype(np.float32).tobytes())

def export_model(model, output_path="mnist_mixed.mflow", device='cpu'):
    print(f"\n📦 导出模型到: {output_path}")

    model.eval()

    with open(output_path, 'wb') as f:
        num_layers = 11
        num_tensors = 8
        header_offset = 88 + num_layers * 24

        f.write(struct.pack('IIIIII64s',
                           0x4D464C57,
                           3,  # Version 3 = 优化版本
                           num_layers,
                           num_tensors,
                           header_offset,
                           0,
                           b"MicroFlow_MixedMNIST".ljust(64, b'\x00')))

        # Layer 0: Input
        f.write(struct.pack('IIIIII', 0, 0, 1, 1, 0, 0))

        # Layer 1: Conv1
        f.write(struct.pack('IIIIII', 1, 0, 1, 1, 0, 0))
        write_tensor(f, model.conv1.weight.detach().numpy())
        write_tensor(f, model.conv1.bias.detach().numpy())

        # Layer 2: ReLU
        f.write(struct.pack('IIIIII', 5, 0, 1, 1, 0, 0))

        # Layer 3: MaxPool2D
        f.write(struct.pack('IIIIII', 9, 0, 1, 1, 0, 0))

        # Layer 4: Conv2
        f.write(struct.pack('IIIIII', 1, 0, 1, 1, 0, 0))
        write_tensor(f, model.conv2.weight.detach().numpy())
        write_tensor(f, model.conv2.bias.detach().numpy())

        # Layer 5: ReLU
        f.write(struct.pack('IIIIII', 5, 0, 1, 1, 0, 0))

        # Layer 6: MaxPool2D
        f.write(struct.pack('IIIIII', 9, 0, 1, 1, 0, 0))

        # Layer 7: Flatten
        f.write(struct.pack('IIIIII', 14, 0, 1, 1, 0, 0))

        # Layer 8: FC1
        f.write(struct.pack('IIIIII', 13, 0, 1, 1, 0, 0))
        write_tensor(f, model.fc1.weight.detach().numpy().T)
        write_tensor(f, model.fc1.bias.detach().numpy())

        # Layer 9: ReLU
        f.write(struct.pack('IIIIII', 5, 0, 1, 1, 0, 0))

        # Layer 10: FC2
        f.write(struct.pack('IIIIII', 13, 0, 1, 1, 0, 0))
        write_tensor(f, model.fc2.weight.detach().numpy().T)
        write_tensor(f, model.fc2.bias.detach().numpy())

    print(f"✅ 模型已导出: {output_path}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("  MicroFlow 混合训练器")
    print("  MNIST标准数据 + 你的手写数据")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  使用设备: {device}")

    # ==================== 数据准备 ====================

    # 1. 标准MNIST数据
    print("\n📂 加载标准MNIST数据...")

    # 训练集使用数据增强
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    # 测试集不使用增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnist_train = datasets.MNIST('./data', train=True, download=True,
                                  transform=train_transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True,
                                 transform=test_transform)

    print(f"  MNIST训练集: {len(mnist_train)} 张")
    print(f"  MNIST测试集: {len(mnist_test)} 张")

    # 2. 你的手写数据 - 配置CSV文件
    handwritten_files = {
        0: "testData0.csv",
        1: "testData1.csv",
        2: "testData2.csv",
        3: "testData3.csv",
        4: "testData4.csv",
        5: "testData5.csv",
        6: "testData6.csv",
        7: "testData7.csv",
        8: "testData8.csv",
        9: "testData9.csv",
    }

    # 只加载存在的文件
    existing_files = {k: v for k, v in handwritten_files.items() if os.path.exists(v)}

    if not existing_files:
        print("\n❌ 错误: 没有找到手写数据CSV文件！")
        print("\n请确保以下文件存在于当前目录:")
        for label, filename in handwritten_files.items():
            print(f"  - {filename}")
        return

    # 手写数据不使用transform（已经在__getitem__里处理了）
    handwritten_train = HandwrittenDataset(existing_files)
    handwritten_test = HandwrittenDataset(existing_files)

    # 3. 合并数据集
    print(f"\n🔗 合并数据集...")
    from torch.utils.data import ConcatDataset

    combined_train = ConcatDataset([mnist_train, handwritten_train])
    combined_test = ConcatDataset([mnist_test, handwritten_test])

    print(f"  混合训练集: {len(combined_train)} 张 (MNIST: {len(mnist_train)} + 手写: {len(handwritten_train)})")
    print(f"  混合测试集: {len(combined_test)} 张 (MNIST: {len(mnist_test)} + 手写: {len(handwritten_test)})")

    # 4. 创建DataLoader - 手写数据不需要多进程
    train_loader = torch.utils.data.DataLoader(combined_train, batch_size=128,
                                               shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(combined_test, batch_size=256,
                                              shuffle=False, num_workers=0)

    # ==================== 训练 ====================

    model = ImprovedMNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📈 模型参数量: {total_params:,}")

    model = train_model(model, train_loader, test_loader, epochs=25, device=device)

    # ==================== 导出 ====================

    export_model(model, "mnist_mixed.mflow", device=device)

    print("\n" + "=" * 60)
    print("  ✅ 全部完成！")
    print("=" * 60)
    print(f"\n模型文件: mnist_mixed.mflow")
    print(f"\n部署到树莓派:")
    print(f"  scp mnist_mixed.mflow pi@raspberrypi:~/microflow/pi4_optimized/models/")
    print(f"\n运行推理:")
    print(f"  cd ~/microflow/pi4_optimized/build")
    print(f"  ./image_demo ../models/mnist_mixed.mflow ../image/digit_1_1.bin")
    print()

if __name__ == "__main__":
    main()
