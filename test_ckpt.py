import torch

# 路径到你的 checkpoint
var_ckpt = '/fs/scratch/PAS2473/MM2025/neurpis2025/VAR/local_output_test2/ar-ckpt-last.pth'  # 改成你自己的

# 加载 checkpoint（只是state_dict，不需要模型结构）
ckpt = torch.load(var_ckpt, map_location='cpu')

print(f"Checkpoint type: {type(ckpt)}")
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    print("Found 'state_dict' inside checkpoint.")
    state_dict = ckpt['state_dict']
else:
    print("Checkpoint is a raw state_dict or unusual format.")
    state_dict = ckpt['trainer']['var_wo_ddp']

# 打印所有 key
print("========= Keys in the checkpoint =========")
for k in state_dict.keys():
    print(k)

# 统计总共有多少个key
print(f"\nTotal {len(state_dict)} keys.")