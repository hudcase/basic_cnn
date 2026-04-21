
import torch
import torchvision.transforms as transforms
from PIL import Image


# 定义图像预处理步骤，这里将图像转换为张量，可根据需求添加图像大小调整等操作
transform = transforms.Compose([
    # transforms.Resize((256, 256)),  # 如果需要可以调整图像大小
    transforms.ToTensor()  # 转换为张量
])


if __name__ == '__main__':
    image_path = "./input_image.jpg"
    # 加载图像并转换为RGB模式
    image = Image.open(image_path).convert("RGB")

    # 将图像转换为张量并添加批次维度，以便后续输入到神经网络层
    input_tensor = transform(image).unsqueeze(0)

    # 定义卷积层参数
    in_channels, out_channels = 3, 3
    kernel_size = 10
    # 创建卷积层
    conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
    # 使用卷积层处理图像
    output = conv_layer(input_tensor)

    # 将处理后的张量转换回PIL图像格式
    output_image = transforms.ToPILImage()(output.squeeze(0))

    # 保存处理后的图像
    output_image.save("./output_image_conv2d.jpg")


