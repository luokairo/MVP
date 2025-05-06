import os
import os.path as osp
import random

import PIL.Image as PImage
from pathlib import Path
import torch
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset
from tokenizer import tokenize
from typing import Dict, List, Tuple, Callable, Optional


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def image_transform(final_reso: int, model='train',
    hflip=False, mid_reso=1.125,):
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ] 
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    if model == 'train':
       return train_aug
    else:
        return val_aug

def build_dataset(
    data_path: str, final_reso: int,
    hflip=False, mid_reso=1.125,
):
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    train_aug, val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ], [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # build dataset
    train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=train_aug)
    val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
    num_classes = 1000
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return num_classes, train_set, val_set


def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')


# For CLIP
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]

class ImageNetCaptions(Dataset):
    def __init__(
        self,
        root: str,
        final_reso: int,
        model: str,
        caption_file: str,
        synset_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        hflip=False, 
        mid_reso=1.125,
    ):
        super(ImageNetCaptions, self).__init__()
        assert model in ["train", "val"]
        self.root = root
        self.transform = image_transform(final_reso, model, hflip, mid_reso)

        self.samples = [] # 储存(path, class_idx, class_id)
        self.targets = [] # 储存对应的类别索引

        # 加载同义词映射
        self.synsets = {}  # 映射 class_id -> 同义词列表
        with open(synset_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    class_id = parts[0].strip()
                    synonyms = parts[1].strip().split(', ')
                    self.synsets[class_id] = synonyms
        
        # 创建class_id到class_idx的映射
        self.class_id_to_idx = {}
        for i, class_id in enumerate(sorted(self.synsets.keys())):
            self.class_id_to_idx[class_id] = i
        
        for class_id in sorted(os.listdir(root)):
            class_dir = osp.join(root, class_id)
            if not osp.isdir(class_dir) or class_id not in self.synsets:
                continue
                
            class_idx = self.class_id_to_idx[class_id]
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.endswith(('.JPEG', '.jpg', '.png')):
                    path = osp.join(class_id, img_name)
                    self.samples.append((path, class_idx, class_id))
                    self.targets.append(class_idx)

        self.captions = {} # path -> caption
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    img_path = parts[0].strip()
                    # 移除路径中的./train/前缀
                    if img_path.startswith('./train/'):
                        img_path = img_path[8:]
                    if img_path.startswith('./val/'):
                        img_path = img_path[6:]
                    caption = parts[1].strip()
                    self.captions[img_path] = caption

    def __getitem__(self, index: int):
        path, target, class_id = self.samples[index]
        full_path = osp.join(self.root, path)

        with open(full_path, 'rb') as f:
            img = PImage.open(f).convert('RGB')
        img = self.transform(img)

        synonyms = self.synsets.get(class_id, ["unknown"])
        label_name = "a photo of " + random.choice(synonyms)

        caption = self.captions.get(path, "")
        label_name = tokenize(label_name)
        caption = tokenize(caption)

        return img, target, label_name, caption

    def __len__(self):
        return len(self.samples)

def build_imagenet_captions(
    data_path: str,
    caption_file: str,
    synset_file: str,
    final_reso: int,
    hflip=False,
    mid_reso=1.125,
):
    train_caption_file = os.path.join(caption_file, 'train', 'image_captions.txt')
    val_caption_file = os.path.join(caption_file, 'val', 'image_captions.txt')
    
    train_set = ImageNetCaptions(
        root=osp.join(data_path, 'train'),
        final_reso=final_reso,
        model='train',
        caption_file=train_caption_file,
        synset_file=synset_file,
        hflip=hflip,
        mid_reso=mid_reso,
    )
    val_set = ImageNetCaptions(
        root=osp.join(data_path, 'val'),
        final_reso=final_reso,
        model='val',
        caption_file=val_caption_file,
        synset_file=synset_file,
        hflip=hflip,
        mid_reso=mid_reso,
    )
    num_classes = len(train_set.synsets)
    print(f'[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print_aug(train_set.transform, '[train]')
    print_aug(val_set.transform, '[val]')

    return num_classes, train_set, val_set

    