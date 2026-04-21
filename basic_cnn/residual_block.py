# 导入必要的PyTorch模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================= 残差块（Residual Block）完整实现 =========================
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

# ========================= 测试残差块功能 =========================
if __name__ == "__main__":
    # 1. 实例化残差块：设置通道数为64（常用的残差块通道数）
    res_block = ResidualBlock(channels=64)
    res_block.eval()  # 推理模式
    
    # 2. 构造测试输入：模拟批量大小为2、64通道、32x32的特征图
    # 形状：[batch_size, channels, height, width]
    test_input = torch.randn(2, 64, 32, 32)
    
    # 3. 前向传播（禁用梯度计算，提升效率）
    with torch.no_grad():
        test_output = res_block(test_input)
    
    # 4. 验证输出形状（残差块应保证输入输出形状完全一致）
    print(f"输入张量形状：{test_input.shape}")
    print(f"输出张量形状：{test_output.shape}")
    
    # 验证形状是否一致（残差连接的核心要求）
    assert test_input.shape == test_output.shape, "残差块输入输出形状不一致！"
    print("✅ 残差块测试通过：输入输出形状完全一致")