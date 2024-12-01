#!/bin/bash

# 接收参数
RANK=$1  # 分片编号
DATA_BASE_PATH="dataset"
EPOCHS=10
BATCH_SIZE=64
IMG_SIZE=200
LR=0.001
NUM_CLASSES=2

# 动态设置数据路径
DATA_PATH="split_${RANK}"
mkdir -p "checkpoints"
touch checkpoints/dummy.txt

# 检查数据路径是否存在
if [ ! -d "$DATA_PATH" ]; then
  echo "Data path for rank ${RANK} does not exist: ${DATA_PATH}"
  exit 1
fi

# 打印任务信息
echo "Starting task ${RANK} with data path: ${DATA_PATH}..."

# 执行训练任务
# singularity exec osdf:///ospool/uc-shared/public/OSG-Staff/xalim_stat/R_tidyverse_FITSio.sif python cnn_train.py \
python cnn_train.py \
  --data_path "$DATA_PATH" \
  --img_size "$IMG_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --num_classes "$NUM_CLASSES"

# 打印完成信息
echo "Task ${RANK} finished."
