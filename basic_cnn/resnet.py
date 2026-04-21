import torch
import torch.nn as nn
from torchvision import models

# ========================= 1. 基础配置 =========================
# 设置设备：优先使用GPU，无GPU则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 分类类别数（示例为10分类）
num_classes = 10

# ========================= 2. 加载并修改 ResNet34 =========================
# 加载预训练的 ResNet34（注意：torchvision 新版本中 pretrained=True 已弃用，改用 weights 参数）
# 兼容新旧版本的写法：
try:
    # 新版本（torchvision >= 0.13）
    from torchvision.models import ResNet34_Weights
    model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
except ImportError:
    # 旧版本
    model = models.resnet34(pretrained=True)

# 修改最后一层全连接层（fc）：适配10分类任务
# model.fc.in_features 是原fc层的输入特征数（ResNet34固定为512）
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 将模型移到指定设备（GPU/CPU）
model = model.to(device)

# 设置模型为评估模式（推理时必须！禁用Dropout/BatchNorm的训练行为）
model.eval()

# ========================= 3. 构造测试输入（模拟RGB图片） =========================
# ResNet34 要求的输入规范：
# - 尺寸：至少224x224（ImageNet预训练模型的标准输入）
# - 通道：3通道（RGB）
# - 形状：[batch_size, channels, height, width]
batch_size = 2  # 模拟批量大小为2的输入
test_input = torch.randn(batch_size, 3, 224, 224).to(device)  # 随机生成测试张量
print(f"测试输入形状: {test_input.shape}")  # 输出: torch.Size([2, 3, 224, 224])

# ========================= 4. 前向推理（核心测试） =========================
# 禁用梯度计算：推理阶段无需计算梯度，提升速度+节省内存
with torch.no_grad():
    # 模型前向传播
    output = model(test_input)

# ========================= 5. 解析输出结果 =========================
print("\n===== 推理结果解析 =====")
# 输出形状：[batch_size, num_classes] → 此处为 [2, 10]
print(f"模型输出形状: {output.shape}")

# 输出原始logits（未归一化的得分）
print("\n原始输出（logits）:")
print(output)

# 将logits转为概率（Softmax归一化）
prob = nn.functional.softmax(output, dim=1)
print("\n归一化后的概率:")
print(prob)

# 预测类别（取概率最大的索引）
pred_labels = torch.argmax(prob, dim=1)
print("\n预测类别索引（0-9）:")
print(pred_labels)

# 验证输出合理性
assert output.shape == (batch_size, num_classes), "模型输出形状错误！"
print("\n✅ 模型测试通过：前向推理正常，输出形状符合预期！")
