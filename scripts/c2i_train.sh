#!/bin/bash

# 设置训练环境变量
export CUDA_VISIBLE_DEVICES=0  # 设置使用的GPU

# 单节点训练
torchrun --nproc_per_node=1 /fs/scratch/PAS2473/MM2025/neurpis2025/VAR/c2i_train.py \
  --depth=20 \
  --bs=104 \
  --ep=1 \
  --fp16=1 \
  --tlr=1e-4 \
  --alng=1e-3 \
  --wpe=0.1 \
  --control_strength=1.0 \
  --n_cond_embed=768 \
  --data_load_reso=256

# 多GPU训练示例
# torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 train2.py \
#   --depth=20 \
#   --bs=32 \
#   --ep=50 \
#   --fp16=1 \
#   --tlr=1e-4 \
#   --alng=1e-3 \
#   --wpe=0.1 \
#   --control_strength=1.0 \
#   --n_cond_embed=768 \
#   --data_path=/path/to/your/data \
#   --caption_file=/path/to/caption/file.json \
#   --synset_file=/path/to/synset/file.txt \
#   --data_load_reso=256

# 多节点训练示例
# torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=MASTER_IP --master_port=29500 train2.py \
#   --depth=20 \
#   --bs=32 \
#   --ep=50 \
#   --fp16=1 \
#   --tlr=1e-4 \
#   --alng=1e-3 \
#   --wpe=0.1 \
#   --control_strength=1.0 \
#   --n_cond_embed=768 \
#   --data_path=/path/to/your/data \
#   --caption_file=/path/to/caption/file.json \
#   --synset_file=/path/to/synset/file.txt \
#   --data_load_reso=256