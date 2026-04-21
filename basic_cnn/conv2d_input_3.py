import torch
import torch.nn as nn

# ====================== 1. 定义3通道输入张量 ======================
# 输入形状说明: [batch_size, in_channels, height, width] -> [1, 3, 5, 5]
input_tensor = torch.tensor(
    [
        # 输入通道1
        [
            [3, 4, 6, 5, 7],
            [2, 4, 6, 8, 2],
            [1, 6, 7, 8, 4],
            [9, 7, 4, 6, 2],
            [3, 7, 5, 4, 1]
        ],
        # 输入通道2
        [
            [3, 4, 6, 5, 7],
            [2, 4, 6, 8, 2],
            [1, 6, 7, 8, 4],
            [9, 7, 4, 6, 2],
            [3, 7, 5, 4, 1]
        ],
        # 输入通道3
        [
            [3, 4, 6, 5, 7],
            [2, 4, 6, 8, 2],
            [1, 6, 7, 8, 4],
            [9, 7, 4, 6, 2],
            [3, 7, 5, 4, 1]
        ]
    ],
    dtype=torch.float32
).unsqueeze(0)  # 增加batch维度，适配Conv2d输入要求

# ====================== 2. 定义卷积层（禁用默认权重初始化） ======================
# 卷积层参数说明：
# - in_channels=3: 匹配输入的3通道
# - out_channels=1: 仅输出1个通道
# - kernel_size=3: 3x3卷积核
# - stride=1: 步幅为1
# - padding=0: 无填充
# - bias=False: 不使用偏置项（与原始计算逻辑一致）
conv_layer = nn.Conv2d(
    in_channels=3,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=False
)

# ====================== 3. 手动赋值卷积核权重 ======================
# 自定义权重形状: [out_channels, in_channels, kernel_h, kernel_w] -> [1, 3, 3, 3]
custom_weight = torch.tensor(
    [
        [
            # 输入通道1对应的3x3卷积核
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            # 输入通道2对应的3x3卷积核
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
            # 输入通道3对应的3x3卷积核
            [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        ]
    ],
    dtype=torch.float32
)

# 将自定义权重赋值给卷积层（requires_grad=False：权重固定，不参与训练）
conv_layer.weight = nn.Parameter(custom_weight, requires_grad=False)

# ====================== 4. 执行卷积计算并输出结果 ======================
# 禁用梯度计算：推理阶段无需计算梯度，提升效率且节省内存
with torch.no_grad():
    conv_output = conv_layer(input_tensor)

# 打印最终卷积输出结果
print("卷积计算输出结果：")
print(conv_output)