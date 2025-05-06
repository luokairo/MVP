import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils.data import build_imagenet_captions

# 设置随机种子以确保可重现性
seed = 4
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 数据路径
data_path = "/fs/scratch/PAS2473/MM2025/neurpis2025/dataset/ILSVRC/Data/CLS-LOC"
caption_file = "/fs/scratch/PAS2473/MM2025/neurpis2025/VAR-CLIP/imagenet"
synset_file = "/fs/scratch/PAS2473/MM2025/neurpis2025/dataset/LOC_synset_mapping.txt"
final_reso = 256

train_dir = os.path.join(data_path, 'train')
val_dir = os.path.join(data_path, 'val')
train_caption = os.path.join(caption_file, 'train', 'image_captions.txt')
val_caption = os.path.join(caption_file, 'val', 'image_captions.txt')

print(f"训练目录存在: {os.path.exists(train_dir)}")
print(f"验证目录存在: {os.path.exists(val_dir)}")
print(f"训练描述文件存在: {os.path.exists(train_caption)}")
print(f"验证描述文件存在: {os.path.exists(val_caption)}")

# 构建数据集
print("正在构建数据集...")
num_classes, train_set, val_set = build_imagenet_captions(
    data_path=data_path,
    caption_file=caption_file,
    synset_file=synset_file,
    final_reso=final_reso,
    hflip=True,
    mid_reso=1.125,
)

# 创建DataLoader
train_loader = DataLoader(
    train_set, 
    batch_size=1,  # 每次只取一个样本
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_set, 
    batch_size=1,  # 每次只取一个样本
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 创建保存图像的目录
os.makedirs("sample_images", exist_ok=True)

# 获取一个训练样本
print("\n获取训练样本...")
train_iter = iter(train_loader)
train_image, train_target, train_label_name, train_caption = next(train_iter)

# 打印训练样本信息
print(f"训练样本:")
print(f"target: {train_target.item()}")
print(f"label_name: {train_label_name[0]}")
print(f"caption: {train_caption[0]}")

# 保存训练图像
train_image_path = "sample_images/train_sample.png"
save_image((train_image[0] + 1) / 2, train_image_path)  # 将[-1,1]范围转换回[0,1]
print(f"训练图像已保存到: {train_image_path}")

# 获取一个验证样本
print("\n获取验证样本...")
val_iter = iter(val_loader)
val_image, val_target, val_label_name, val_caption = next(val_iter)

# 打印验证样本信息
print(f"验证样本:")
print(f"target: {val_target.item()}")
print(f"label_name: {val_label_name[0]}")
print(f"caption: {val_caption[0]}")

# 保存验证图像
val_image_path = "sample_images/val_sample.png"
save_image((val_image[0] + 1) / 2, val_image_path)  # 将[-1,1]范围转换回[0,1]
print(f"验证图像已保存到: {val_image_path}")

# 额外打印一些数据集统计信息
print(f"\n数据集统计:")
print(f"训练集大小: {len(train_set)}")
print(f"验证集大小: {len(val_set)}")
print(f"类别数量: {num_classes}")

