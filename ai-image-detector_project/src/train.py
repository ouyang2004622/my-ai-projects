import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.models import resnet50
from tqdm import tqdm

# 训练参数
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
DATASET_PATH = '训练数据集路径'  # 训练数据集路径
CHECKPOINT_PATH = '模型权重保存路径'  # 模型权重保存路径

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 定义类别映射
custom_class_to_idx = {'real': 0, 'fake': 1}

# 加载数据集
train_dataset = ImageFolder(root=DATASET_PATH, transform=transform)
train_dataset.class_to_idx = custom_class_to_idx  # 手动设置类别映射
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型
model = resnet50(pretrained=True)  # 使用预训练的ResNet50
model.fc = nn.Linear(model.fc.in_features, 2)  # 修改最后一层输出为2个类别
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU或CPU

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
for epoch in range(EPOCHS):
    model.train()  # 设置为训练模式
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 累加损失

    # 输出每个epoch的损失
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}')

# 保存模型权重
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f'Model saved to {CHECKPOINT_PATH}')
