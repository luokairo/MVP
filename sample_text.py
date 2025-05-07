################## 1. Download checkpoints and build models
import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
from models.clip import clip_vit_l14
from clip_util import CLIPWrapper
from tokenizer import tokenize
normalize_clip = True

clip = clip_vit_l14(pretrained=True).cuda().eval()
clip = CLIPWrapper(clip, normalize=normalize_clip)

MODEL_DEPTH = 20    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 36}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = '/fs/scratch/PAS2473/MM2025/neurpis2025/ckpt/var/vae_ch160v4096z32.pth', f'/fs/scratch/PAS2473/MM2025/neurpis2025/VAR/local_output_t2i/ar-ckpt-last.pth'
# var_ckpt = f'/fs/scratch/PAS2473/MM2025/neurpis2025/ckpt/var/var_d{MODEL_DEPTH}.pth'
if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
patch_nums = (1,2,3,4,5,6,8,10,13,16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
# var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
ckpt = torch.load(var_ckpt, map_location='cpu')
var_wo_ddp_state = ckpt['trainer']['var_wo_ddp']
var.load_state_dict(var_wo_ddp_state, strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')


############################# 2. Sample with classifier-free guidance

# set args
seed = 42 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = (934,)   #@param {type:"raw"}
more_smooth = False # True for more smooth output

text_prompt = "a tabby cat laying on the grass"
text_embedding = tokenize(text_prompt).unsqueeze(0).cuda()
text_embeddings = clip.encode_text(text_embedding)

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

# sample
B = len(class_labels)
label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
with torch.inference_mode():
    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
        recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, text_embedding=text_embeddings, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)

os.makedirs('/fs/scratch/PAS2473/MM2025/neurpis2025/VAR/output_samples', exist_ok=True)
output_path = f'/fs/scratch/PAS2473/MM2025/neurpis2025/VAR/output_samples/sample_seed{seed}.png'
chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
chw = (chw.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
chw_img = PImage.fromarray(chw)
chw_img.save(output_path)
print(f'Saved generated samples to {output_path}')