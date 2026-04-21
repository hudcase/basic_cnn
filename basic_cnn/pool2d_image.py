
import torch
import torchvision.transforms as transforms
from PIL import Image

# 定义图像预处理步骤，这里将图像转换为张量，可根据需求添加图像大小调整等操作
transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # 如果需要可以调整图像大小
    transforms.ToTensor()  # 转换为张量
])



# 指定输入图像路径，这里需要替换为实际的图像路径
image_path = "./input_image.jpg"
# 加载图像并转换为RGB模式
image = Image.open(image_path).convert("RGB")

# 将图像转换为张量并添加批次维度，以便后续输入到神经网络层
input_tensor = transform(image).unsqueeze(0)

# 定义最大池化层参数
kernel_size = 5
# 创建最大池化层
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=kernel_size)
# 使用最大池化层处理图像
output = maxpooling_layer(input_tensor)

# 将处理后的张量转换回PIL图像格式
output_image = transforms.ToPILImage()(output.squeeze(0))

# 保存处理后的图像
output_image.save("./output_image_maxpool.jpg")

