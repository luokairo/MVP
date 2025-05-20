#!/bin/bash

# 设置训练环境变量
export CUDA_VISIBLE_DEVICES=0  # 设置使用的GPU

# 单节点训练
torchrun --nproc_per_node=1 --master_port=29501 /fs/scratch/PAS2473/MM2025/neurpis2025/VAR/t2i_train_512.py \
  --depth=36 \
  --bs=9 \
  --ep=1 \
  --fp16=1 \
  --tlr=8e-5 \
  --tblr=8e-5 \
  --alng=5e-6 \
  --wpe=0.01 \
  --twde=0.08 \
  --control_strength=1.0 \
  --n_cond_embed=768 \
  --data_load_reso=512 \
  --pn=512 \
  --outer_nums=28 \
  --saln=True 