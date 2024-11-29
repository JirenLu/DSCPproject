#!/bin/bash

# 获取任务编号
RANK=$1
DATA_PATH="dataset/split_${RANK}"

# 运行训练脚本
python train.py --data_path $DATA_PATH --epochs 10 --batch_size 64 --latent_dim 100 --img_size 28 --rank $RANK

# 检查是否生成了模型文件
if [ -f checkpoints/generator_rank_${RANK}.pt ] && [ -f checkpoints/discriminator_rank_${RANK}.pt ]; then
  echo "Training results saved for rank ${RANK}."
else
  echo "Error: Training results not found for rank ${RANK}."
fi
