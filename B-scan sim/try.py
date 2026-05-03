import torch
import torchvision.models as models

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())       # True
print(torch.cuda.device_count())       # GPU 数量
print(torch.cuda.get_device_name(0))   # GPU 名称

# 创建 MobileNetV3 small 模型
model = models.mobilenet_v3_small(num_classes=2)  # 假设是 2 分类
model.to(device)  # 把模型移动到 GPU

# 打印模型参数所在设备
print(next(model.parameters()).device)