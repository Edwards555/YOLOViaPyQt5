import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.yolo4 import YoloBody
from utils.dataloader import YoloDataset
from utils.utils import get_classes, get_anchors
from utils.loss import YOLOLoss

# 配置参数
classes_path = 'model_data/leg_acupoints_classes.txt'  # 类别文件路径
anchors_path = 'model_data/yolo_anchors.txt'          # 先验框文件路径
model_path = 'model_data/yolov4_leg_acupoints.pth'    # 保存模型路径
input_shape = (608, 608)                              # 输入尺寸
batch_size = 8                                        # 批量大小
learning_rate = 1e-3                                  # 学习率
num_epochs = 50                                       # 训练轮数
cuda = torch.cuda.is_available()                     # 是否使用GPU

# 加载类别和先验框
class_names, num_classes = get_classes(classes_path)
anchors = get_anchors(anchors_path)

# 初始化模型
model = YoloBody(len(anchors[0]), num_classes, backbone="mobilenetv3")
if cuda:
    model = model.cuda()

# 定义损失函数和优化器
yolo_loss = YOLOLoss(anchors, num_classes, input_shape, cuda)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 加载数据集
train_dataset = YoloDataset('train.txt', input_shape, num_classes, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

# 训练模型
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        if cuda:
            images = images.cuda()
            targets = [target.cuda() for target in targets]

        optimizer.zero_grad()
        outputs = model(images)
        loss = yolo_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
