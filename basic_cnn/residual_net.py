# 导入必要的PyTorch模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    基础残差块（恒等映射版）：输入输出通道数相同，适用于特征图尺寸不变的场景
    结构：Conv2d(3x3) → ReLU → Conv2d(3x3) → 残差连接（x+y） → ReLU
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels  # 输入/输出通道数（残差块保持通道数不变）
        
        # 第一个卷积层：3x3卷积，padding=1保证特征图尺寸不变
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,  # padding=1 抵消kernel_size=3的尺寸缩减，保持H/W不变
            bias=False  # 残差块中通常配合BN使用（本示例简化未加BN，可保留bias）
        )
        
        # 第二个卷积层：与conv1参数一致，保持通道数和尺寸不变
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        """
        前向传播：残差连接核心逻辑
        :param x: 输入张量，形状为 [batch_size, channels, height, width]
        :return: 输出张量，形状与输入完全一致
        """
        # 主干路径：卷积1 → ReLU激活
        y = F.relu(self.conv1(x))
        # 主干路径：卷积2（无激活，留到残差连接后）
        y = self.conv2(y)
        # 残差连接：输入x + 主干输出y → 再激活（符合残差网络的经典设计）
        y = F.relu(x + y)
        
        return y
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.fc = nn.Linear(512, 10)
    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x