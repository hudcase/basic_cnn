# 导入必要的Python库和PyTorch相关模块
import os                      # 用于文件路径处理、目录创建等操作系统交互
import torch                   # PyTorch核心库，提供张量计算和神经网络构建能力
import torch.nn as nn          # PyTorch神经网络模块，包含卷积层、全连接层等基础组件
import torch.nn.functional as F  # PyTorch函数式接口，提供ReLU、池化等无参数操作
from torchvision import transforms  # 提供图片预处理（如转张量、缩放）的工具
from PIL import Image          # Python Imaging Library，用于读取和保存图片

# =========================
# 1. 定义MNIST手写数字识别网络（适配单通道28x28灰度图）
# =========================
class Net(nn.Module):
    """
    自定义MNIST分类网络结构：
    输入：单通道28x28灰度图 -> 输出：10维数字分类概率（0-9）
    网络结构：Conv1 → MaxPool → ReLU → Conv2 → MaxPool → ReLU → Flatten → FC → 输出
    """
    def __init__(self):
        # 继承父类nn.Module的初始化方法，必须调用
        super(Net, self).__init__()
        
        # 第一层卷积：输入通道1（灰度图），输出通道10，卷积核5x5，步幅默认1，无填充
        # 输入形状(1,1,28,28) → 卷积后(1,10,24,24)（28-5+1=24）
        self.conv1 = nn.Conv2d(
            in_channels=1,    # 输入通道数：MNIST是单通道灰度图
            out_channels=10,  # 输出通道数：10个卷积核，提取10种特征
            kernel_size=5     # 卷积核尺寸：5x5
        )
        
        # 第二层卷积：输入通道10（对应conv1的输出），输出通道20，卷积核5x5
        # 输入形状(1,10,12,12)（conv1池化后）→ 卷积后(1,20,8,8)（12-5+1=8）
        self.conv2 = nn.Conv2d(
            in_channels=10,   # 输入通道数匹配conv1的输出通道
            out_channels=20,  # 输出通道数：20个卷积核，提取更复杂的特征
            kernel_size=5     # 卷积核尺寸：5x5
        )
        
        # 全连接层：输入维度320，输出维度10（对应0-9数字分类）
        # 320的计算逻辑：conv2池化后形状(1,20,4,4) → 20*4*4=320
        self.fc = nn.Linear(
            in_features=320,  # 输入特征数：展平后的特征维度
            out_features=10   # 输出类别数：MNIST共10个数字类别
        )
        
        # 最大池化层：池化核2x2，步幅默认等于核尺寸（即2），无填充
        # 作用：降维、保留关键特征、防止过拟合
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        """
        定义网络前向传播逻辑（核心！PyTorch会自动构建反向传播）
        参数x：输入张量，形状为(batch_size, channels, height, width)
        返回值：网络输出张量，形状为(batch_size, 10)
        """
        # 第一层：卷积 → 池化 → ReLU激活
        # x: (1,1,28,28) → conv1 → (1,10,24,24) → pooling → (1,10,12,12) → ReLU → (1,10,12,12)
        x = F.relu(self.pooling(self.conv1(x)))
        
        # 第二层：卷积 → 池化 → ReLU激活
        # x: (1,10,12,12) → conv2 → (1,20,8,8) → pooling → (1,20,4,4) → ReLU → (1,20,4,4)
        x = F.relu(self.pooling(self.conv2(x)))
        
        # 展平操作：将4维张量转为2维（batch_size, 特征数）
        # x: (1,20,4,4) → view → (1, 320)
        # x.size(0)是batch_size（这里为1），-1表示自动计算剩余维度
        x = x.view(x.size(0), -1)
        
        # 全连接层：展平后的特征 → 10维分类输出
        # x: (1,320) → fc → (1,10)
        x = self.fc(x)
        
        # 返回最终输出（注：此处未加Softmax，输出是logits，需后续处理得到概率）
        return x


# =========================
# 2. 初始化模型 + 设置推理模式
# =========================
# 实例化自定义网络
model = Net()
# 设置模型为评估模式（eval）：禁用Dropout、BatchNorm等训练相关层的行为
# 推理/预测时必须调用，否则结果会受训练模式影响
model.eval()

# =========================
# 3. 加载并预处理MNIST图片
# =========================
# 本地MNIST图片路径（需替换为你实际的图片路径）
img_path = "mnist_sample/mnist_2_label_4.png"   
# 使用PIL.Image打开图片（MNIST是单通道灰度图，无需convert("RGB")）
img = Image.open(img_path)

# 定义图片预处理流水线
transform = transforms.Compose([
    # 注释掉的Resize：MNIST原始尺寸是28x28，无需缩放到256x256
    # transforms.Resize((256, 256)),
    # 将PIL Image转为PyTorch张量：
    # 输入：PIL灰度图（0-255）→ 输出：张量（0.0-1.0），形状为(1,28,28)
    transforms.ToTensor()
])

# 对图片执行预处理，并增加batch维度（模型要求输入必须有batch维度）
# x: (1,28,28) → unsqueeze(0) → (1,1,28,28)
# 注：原注释标注的(1,3,256,256)是错误的，已修正为实际形状
x = transform(img).unsqueeze(0)    
# 打印输入张量形状，方便调试（可选）
print(f"输入张量形状：{x.shape}")  # 输出：torch.Size([1, 1, 28, 28])


# =========================
# 4. 前向推理（无梯度计算）
# =========================
# 禁用梯度计算上下文管理器：推理阶段无需计算梯度，提升速度+节省内存
# （原代码未加，此处补充最佳实践）
with torch.no_grad():
    # 将预处理后的图片输入模型，执行前向传播，得到输出
    y = model(x)

# 打印网络输出：形状为(1,10)的张量，每个值是对应数字（0-9）的logits（未归一化的得分）
print("网络输出logits（0-9数字的原始得分）：")
print(y)

# 可选：将logits转为概率（补充实用功能，帮助理解输出）
prob = F.softmax(y, dim=1)  # 对输出维度做Softmax归一化，得到概率
print(f"\n各数字的预测概率: {prob.squeeze().numpy()}")