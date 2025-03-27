import torch
from nets.yolo4 import YoloBody
from utils.utils import get_classes, get_anchors

# 配置参数
classes_path = 'model_data/leg_acupoints_classes.txt'  # 类别文件路径
anchors_path = 'model_data/yolo_anchors.txt'          # 先验框文件路径
model_path = 'model_data/yolov4_leg_acupoints.pth'    # 训练好的模型路径
frozen_model_path = 'model_data/yolov4_leg_acupoints_frozen.pth'  # 冻结后的模型路径

# 加载类别和先验框
class_names, num_classes = get_classes(classes_path)
anchors = get_anchors(anchors_path)

# 初始化模型
model = YoloBody(len(anchors[0]), num_classes, backbone="mobilenetv3")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# 冻结模型
for param in model.parameters():
    param.requires_grad = False

# 保存冻结模型
torch.save(model.state_dict(), frozen_model_path)
print(f"Frozen model saved to {frozen_model_path}")
