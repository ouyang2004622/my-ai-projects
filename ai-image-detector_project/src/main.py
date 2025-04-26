import os
import csv
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

# 模型加载
model = resnet50(weights='DEFAULT')  # 使用torchvision中的resnet50，默认权重
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 最后一层输出为2个类别
model.load_state_dict(torch.load('./model.pth', map_location='cpu', weights_only=True))  # 加载我们模型的权重
model.eval()

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据我们训练的模型输入尺寸进行调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定义支持的图片扩展名
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 读取测试数据集并进行推理
test_data_dir = '../testdata'
image_files = sorted(os.listdir(test_data_dir))  # 按字典序排序
results = []
with torch.no_grad():
    for img_file in image_files:
        # 忽略非图片文件
        if img_file.lower().endswith(valid_extensions):
            img_path = os.path.join(test_data_dir, img_file)
            try:
                img = transform(Image.open(img_path).convert('RGB'))
                img = img.unsqueeze(0)  # 增加批次维度
                outputs = model(img)  # 推理
                result = 1-torch.argmax(outputs, dim=1).item()  # 判断为 AI 生成图像还是原始图像
                img_name = os.path.splitext(img_file)[0]  # 去除文件扩展名
                results.append((img_name, result))
                print(result)
            except Exception as e:
                print(f"Error processing file {img_file}: {e}")
# 将结果写入CSV文件
with open('../cla_pre.csv', mode='w', newline='') as csv_file:#这里输出./cla_pre.csv 文件
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(results)
