################## 1. 下载检查点并构建模型
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # 禁用默认参数初始化以提高速度
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # 禁用默认参数初始化以提高速度
from models import VQVAE, build_vae_var
from models.clip import clip_vit_l14
from clip_util import CLIPWrapper
from tokenizer import tokenize
from tqdm import tqdm
normalize_clip = True

clip = clip_vit_l14(pretrained=True).cuda().eval()
clip = CLIPWrapper(clip, normalize=normalize_clip)

MODEL_DEPTH = 30    # TODO: =====> 请指定 MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30, 36}

# 下载检查点
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = '/fs/scratch/PAS2473/MM2025/neurpis2025/ckpt/var/vae_ch160v4096z32.pth', f'/fs/scratch/PAS2473/MM2025/neurpis2025/VAR/local_output_t2i_d{MODEL_DEPTH}/ar-ckpt-best.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# 构建 vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,    # 硬编码的VQVAE超参数
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, 
    shared_aln=False, outer_nums=28,
)

# 加载检查点
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
ckpt = torch.load(var_ckpt, map_location='cpu')
var_wo_ddp_state = ckpt['trainer']['var_wo_ddp']
var.load_state_dict(var_wo_ddp_state, strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'模型准备完成。')

# 设置参数
num_sampling_steps = 250
cfg = 4
more_smooth = True  # 设置为True以获得更平滑的输出
base_seed = 0      # 基础种子，每个提示将使用不同的种子
batch_size = 16      # 每个提示生成的图像数量

# 优化运行速度
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# 创建输出目录
output_base_dir = f'/fs/scratch/PAS2473/MM2025/neurpis2025/VAR/multiple_prompts_output_{base_seed}'
os.makedirs(output_base_dir, exist_ok=True)

# 文本提示列表
text_prompts = [
    "a small dog lying on a sofa",
    "a cup of coffee on a table",
    "a mountain under a blue sky",
    "a single tree in a field",
    "a train passing through a tunnel",
    "a yellow flower in the grass",
    "a person walking on the beach",
    "a bird flying in the sky",
    "a cat sitting on a windowsill",
    "a bicycle parked near a wall",
    "a snowy road in a forest",
    "a lantern hanging on a porch",
    "a city skyline at night",
    "a bench under a tree",
    "a tent in the desert",
    "a child playing with a ball",
    "a boat on a calm lake",
    "a cow standing in a field",
    "a sandwich on a plate",
    "a bookshelf full of books",
]

def generate_images(prompt, prompt_index, seed):
    """为给定的文本提示生成图像"""
    # 设置随机种子
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"正在处理提示 {prompt_index+1}/{len(text_prompts)}: {prompt}")
    
    # 创建此提示的输出目录
    prompt_dir = os.path.join(output_base_dir, f"prompt_{prompt_index+1}")
    os.makedirs(prompt_dir, exist_ok=True)
    
    # 准备文本嵌入
    text_prompt_batch = [prompt] * batch_size
    bs = len(text_prompt_batch)
    text_embedding = tokenize(text_prompt_batch).cuda()
    text_embeddings = clip.encode_text(text_embedding)
    
    # 生成图像
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            recon_B3HW = var.autoregressive_infer_cfg(
                B=bs, 
                label_B=None, 
                text_embedding=text_embeddings, 
                cfg=cfg, 
                top_k=900, 
                top_p=0.95, 
                g_seed=seed, 
                more_smooth=more_smooth
            )
    
    # 保存图像
    for i in range(recon_B3HW.shape[0]):
        img = recon_B3HW[i].permute(1, 2, 0).mul(255).cpu().numpy()
        img = PImage.fromarray(img.astype(np.uint8))
        img.save(os.path.join(prompt_dir, f'image_{i}.png'))
    
    # 创建一个带有提示描述的图像
    img_with_text = img.copy()
    draw = PImageDraw.Draw(img_with_text)
    draw.text((10, 10), prompt, fill=(255, 255, 255))
    img_with_text.save(os.path.join(prompt_dir, 'image_with_text.png'))
    
    return recon_B3HW

# 处理所有提示并收集结果
all_generated_images = []

for i, prompt in enumerate(tqdm(text_prompts)):
    # 为每个提示使用不同的种子
    current_seed = base_seed + i
    generated_images = generate_images(prompt, i, current_seed)
    all_generated_images.append(generated_images[0].unsqueeze(0))  # 只保存每个提示的第一张图片用于网格显示

# 创建所有生成图像的网格并保存
if all_generated_images:
    all_images = torch.cat(all_generated_images, dim=0)
    grid = torchvision.utils.make_grid(all_images, nrow=5, padding=2)
    grid_img = grid.permute(1, 2, 0).mul(255).cpu().numpy()
    grid_img = PImage.fromarray(grid_img.astype(np.uint8))
    grid_img.save(os.path.join(output_base_dir, 'all_prompts_grid.png'))

print(f"所有图像已生成并保存到 {output_base_dir}") 