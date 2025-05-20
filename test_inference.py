import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
from torch.utils.data import DataLoader
from models import VQVAE, build_vae_var
from utils.data import build_imagenet_captions
from models.clip import clip_vit_l14
from clip_util import CLIPWrapper
import torch.nn.functional as F

# 禁用默认参数初始化以加快速度
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

# 模型设置
MODEL_DEPTH = 30
patch_nums = (1,2,3,4,5,6,8,10,13,16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
cfg = 4  # classifier-free guidance

# 加载数据集
print('正在加载数据集...')
num_classes, dataset_train, dataset_val = build_imagenet_captions(
    data_path='/fs/scratch/PAS2473/MM2025/neurpis2025/dataset/ILSVRC/Data/CLS-LOC',
    caption_file='/fs/scratch/PAS2473/MM2025/neurpis2025/VAR-CLIP/imagenet',
    synset_file='/fs/scratch/PAS2473/MM2025/neurpis2025/dataset/LOC_synset_mapping.txt',
    final_reso=256,
    hflip=False,
    mid_reso=1.125,
)

# 创建数据加载器，batch_size为16
batch_size = 16
ld_Dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

# 加载CLIP模型（用于评估）
print('正在加载CLIP模型...')
normalize_clip = True
clip = clip_vit_l14(pretrained=True).to(device)
for param in clip.parameters():
    param.requires_grad = False
clip.eval()
clip = CLIPWrapper(clip, normalize=normalize_clip)

# 加载模型检查点
print('正在加载VAR模型检查点...')
vae_ckpt = '/fs/scratch/PAS2473/MM2025/neurpis2025/ckpt/var/vae_ch160v4096z32.pth'
var_ckpt = f'/fs/scratch/PAS2473/MM2025/neurpis2025/VAR/local_output_c2i_d{MODEL_DEPTH}/ar-ckpt-last.pth'
# 构建vae和var模型
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,    # VQVAE超参数
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    control_strength=0.5, outer_nums=20
)

# 加载检查点
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
# var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
ckpt = torch.load(var_ckpt, map_location='cpu')
var_wo_ddp_state = ckpt['trainer']['var_wo_ddp']
var.load_state_dict(var_wo_ddp_state, strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print('模型加载完成')

# 设置随机种子
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 启用TF32加速
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# 开始推理和评估
print('开始对整个验证集进行推理和评估...')
total_samples = 0
total_clip_score = 0.0
total_batches = len(ld_Dataloader)

for i, batch in enumerate(ld_Dataloader):
    imgs, targets, label_names, captions = batch
    label_names = label_names.to(device)
    
    # 打印当前批次信息
    print(f'处理批次 {i+1}/{total_batches}, 样本数: {len(targets)}')
    
    # 推理
    B = len(targets)
    label_B = targets.to(device)
    
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            # 使用类别进行推理
            recon_B3HW = var.autoregressive_infer_cfg(
                B=B, 
                label_B=label_B, 
                cfg=cfg, 
                top_k=900, 
                top_p=0.95, 
                g_seed=seed, 
                more_smooth=False
            )
            
            # 计算CLIP分数
            # 1. 编码生成的图像
            img_resized = F.interpolate(recon_B3HW, size=(224, 224), mode="bilinear", align_corners=False)
            gen_image_features = clip.encode_image(img_resized.to(device))
            
            # 2. 编码标签文本
            text_features = clip.encode_text(label_names)
            
            # 3. 计算相似度
            similarity = torch.nn.functional.cosine_similarity(gen_image_features, text_features)
            batch_clip_score = similarity.mean().item()
            
            # 更新统计信息
            total_samples += B
            total_clip_score += batch_clip_score * B
            
            # 输出当前批次的平均分数
            print(f'批次 {i+1} 的平均CLIP分数: {batch_clip_score:.4f}')
            
            # 每10个批次输出累计平均分数
            if (i + 1) % 10 == 0 or i == total_batches - 1:
                current_avg_score = total_clip_score / total_samples
                print(f'当前累计平均CLIP分数 ({total_samples}个样本): {current_avg_score:.4f}')

# 输出最终结果
final_avg_score = total_clip_score / total_samples
print(f'\n评估完成!')
print(f'总样本数: {total_samples}')
print(f'最终平均CLIP分数: {final_avg_score:.4f}') 