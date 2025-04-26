# 代码说明文档

本说明文档旨在帮助读者理解提供的训练代码和推理代码，以及来构建基于 **ResNet50** 的二分类模型，可以区分AI生成的图像（fake）和真实的图像（real）。本文档详细解释了各个部分的功能、流程和所使用的技术。

## 1. 环境准备

在运行main.py文件之前，确保已安装必要的Python包。您可以在环境下使用以下命令安装所需的依赖如果有则忽略即可：

```bash
pip install torch torchvision pillow tqdm
```

如果安装的很慢的话，可以切换到国内的镜像源进行安装：

```bash
pip install torch torchvision pillow tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 模型推理部分

此部分旨在使用训练好的模型对新的测试图像进行分类，并输出每张图像属于 real 还是 fake。推理过程包括模型加载、数据预处理、推理操作以及结果的保存。

### 2.1 模型加载

首先，我们需要加载训练好的模型权重，并将模型切换到评估模式（model.eval()），确保推理过程中不会进行梯度计算和权重更新。以下是加载模型权重的代码：

```python
model.load_state_dict(torch.load('./model01.pth', map_location='cpu'))  # 加载预训练的权重文件（model01.pth是训练好后的模型）
model.eval()  # 切换模型为评估模式
```

### 2.2 数据预处理

为了保证推理时的输入与训练时一致，我们需要对图像进行相同的预处理。这包括调整图像大小、转换为张量并进行标准化处理。我们使用 torchvision.transforms 进行这些操作，确保图像的格式和数值范围符合模型的输入要求。

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小至 224x224
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 进行标准化处理
])
```

### 2.3 推理过程

推理阶段遍历测试图像的目录，并对每一张图片进行以下操作：

1. 检查文件是否为有效的图片格式（如 .jpg、.png 等）
2. 对图像进行预处理，使其符合模型的输入要求
3. 将处理后的图像输入到模型中，模型输出该图像是 real 还是 fake 的预测结果

```python
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # 支持的图片格式

with torch.no_grad():  # 禁用梯度计算以加速推理
    for img_file in image_files:
        if img_file.lower().endswith(valid_extensions):  # 检查是否为支持的图片格式
            img_path = os.path.join(test_data_dir, img_file)  # 获取图片路径
            img = transform(Image.open(img_path).convert('RGB'))  # 打开图片并进行预处理
            img = img.unsqueeze(0)  # 增加 batch 维度以适应模型输入
            outputs = model(img)  # 模型推理
            result = 1 - torch.argmax(outputs, dim=1).item()  # 获取预测结果，1 代表 fake，0 代表 real
            results.append((os.path.splitext(img_file)[0], result))  # 将结果保存到列表中
            print(f"Image {img_file}: Predicted result is {'fake' if result == 1 else 'real'}")
```

在这个过程中，torch.no_grad() 被用于关闭梯度计算，以减少内存消耗并加速推理。同时，torch.argmax(outputs, dim=1) 被用于获取模型的输出分类，其中"1"表示图像为 fake，"0"表示图像为 real。

### 2.4 结果保存

推理结束后，将结果（图像名称和预测结果）保存到一个 CSV 文件中，供后续分析和处理。

```python
with open('./cla_pre.csv', mode='w', newline='') as csv_file:  # 打开 CSV 文件用于写入
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(results)  # 写入预测结果
```

## 3. 模型训练部分

我们训练部分代码旨在训练一个基于 **ResNet50** 预训练模型的分类器，通过数据增强、优化和损失计算，让我们的模型会学习如何区分 real 和 fake 图像。

### 3.1 数据集准备与处理

我们训练集路径定义为 DATASET_PATH，其结构为：

```
├── real/  # 放置真实图像
└── fake/  # 放置AI生成的图像
```

我们使用 ImageFolder 来加载数据，并通过 transforms 进行预处理和数据增强。

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整成224×224与ResNet50输入层的要求一致
    transforms.RandomHorizontalFlip(),  # 随机水平翻转，用于增加数据多样性
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

手动设置类别映射，将 real 类别映射为 0，fake 类别映射为 1，保证标签与模型输出一致性。

```python
custom_class_to_idx = {'real': 0, 'fake': 1}
train_dataset.class_to_idx = custom_class_to_idx
```

### 3.2 模型构建

使用 torchvision.models.resnet50 加载预训练模型，并修改最后一层的全连接层，使其输出2个类别，以及默认使用 GPU（如果检查可用），否则使用 CPU。

```python
# 初始化模型
model = resnet50(pretrained=True)  # 使用预训练的ResNet50
model.fc = nn.Linear(model.fc.in_features, 2)  # 修改最后一层输出为2个类别
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU或CPU
```

### 3.3 训练过程

定义损失函数为交叉熵损失（CrossEntropyLoss），优化器为 Adam，模型训练通过 40 个 epoch 完成（这里是我们队调试后最好的结果，如果数据集较大也可以适当增加），每个 epoch 期间，模型会计算损失并更新权重。

```python
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

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
```

最后，保存训练好的模型权重：

```python
torch.save(model.state_dict(), CHECKPOINT_PATH)
```