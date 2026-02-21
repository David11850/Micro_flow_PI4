#!/usr/bin/env python3
"""
CSV转BIN工具 - 将MNIST网站导出的CSV转换为MicroFlow格式

支持两种格式:
1. "Download digits as CSV with true labels" - 有标签列
2. "Download digits as CSV without labels" - 无标签列

使用方法:
python3 csv_to_bin.py testData9.csv [label]
例如: python3 csv_to_bin.py testData9.csv 9  # 指定这个文件里全是数字9
"""

import csv
import numpy as np
import os
import sys

def csv_to_bin(csv_file, output_dir="../image", label=None):
    """将CSV文件转换为单独的.bin文件"""

    os.makedirs(output_dir, exist_ok=True)

    digits = []

    # 读取CSV文件
    print(f"Reading {csv_file}...")

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)

        # 检查第一行是否是标题
        first_row = next(reader, None)
        # 检查是否是标题行：包含 'label' 或 'pixel0' 等列名
        is_header = False
        if first_row:
            for col in first_row[:5]:  # 只检查前几列
                if col in ['label', 'pixel0', 'pixel1'] or col.startswith('pixel'):
                    is_header = True
                    break

        if is_header:
            print("  Detected header row, skipping...")
            data_rows = list(reader)
        else:
            # 没有标题行，把第一行加回数据
            if first_row:
                data_rows = [first_row] + list(reader)
            else:
                data_rows = []

        # 处理每一行
        for i, row in enumerate(data_rows):
            if len(row) < 784:
                print(f"  Warning: Row {i} has only {len(row)} columns, skipping...")
                continue

            # 检查是否有标签列
            if len(row) >= 785:
                # 有标签列 (label + 784 pixels)
                row_label = int(row[0])
                pixels = [float(x) for x in row[1:785]]
                digits.append((row_label, pixels))
            else:
                # 无标签列，使用指定的标签
                if label is None:
                    print(f"  Error: No label column found and no label specified!")
                    print(f"  Please specify label: python3 csv_to_bin.py {csv_file} <0-9>")
                    return False
                pixels = [float(x) for x in row[0:784]]
                digits.append((int(label), pixels))

    print(f"Found {len(digits)} digits")

    # 统计每个数字的数量
    label_counts = {}
    for lbl, _ in digits:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print("Label distribution:")
    for i in range(10):
        count = label_counts.get(i, 0)
        if count > 0:
            print(f"  Digit {i}: {count}")

    # 保存每个数字，使用序号避免覆盖
    label_indices = {}
    for lbl, pixels in digits:
        # 获取这个标签的序号
        label_indices[lbl] = label_indices.get(lbl, 0) + 1
        idx = label_indices[lbl]

        # 转换为28x28数组
        arr = np.array(pixels, dtype=np.float32).reshape(1, 28, 28)

        # MNIST网站数据格式: 0=黑色(墨水), 255=白色(背景)
        # 这与标准MNIST格式一致: 黑色背景, 白色数字
        # 只需归一化，不需要反转
        arr = arr / 255.0  # 归一化到0-1

        # 保存
        output_file = os.path.join(output_dir, f"digit_{lbl}_{idx}.bin")
        arr.tofile(output_file)

    print(f"\n✓ Saved {len(digits)} files to {output_dir}/")
    print("\nTest with:")
    print("  cd build")
    for lbl in sorted(label_counts.keys()):
        print(f"  ./image_demo ../models/mnist_optimized.mflow ../image/digit_{lbl}_1.bin")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 csv_to_bin.py <digits.csv> [label]")
        print("\nExamples:")
        print("  # CSV with labels")
        print("  python3 csv_to_bin.py digits_with_labels.csv")
        print("\n  # CSV without labels (specify the digit)")
        print("  python3 csv_to_bin.py digits_without_labels.csv 9")
        sys.exit(1)

    csv_file = sys.argv[1]
    label = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    csv_to_bin(csv_file, "../image", label)
