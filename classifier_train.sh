#!/bin/bash

# 環境変数を設定
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# トレーニング用のパラメータ設定
DATA_DIR='datasets/train'
VAL_DATA_DIR='datasets/val'
LOG_DIR='logs'
BATCH_SIZE=4

# shellcheck disable=SC2054
CLASSIFIER_FLAGS=(
  --image_size 256
  --classifier_attention_resolutions 32,16,8
  --classifier_depth 2
  --classifier_width 128
  --classifier_pool attention
  --classifier_resblock_updown True
  --classifier_use_scale_shift_norm True
  --classifier_use_fp16 True
  --num_classes 7
  --classifier_out_channel 7
)

# コマンドを作成して実行
python scripts/classifier_train.py \
  --data_dir $DATA_DIR \
  --val_data_dir $VAL_DATA_DIR \
  --log_dir $LOG_DIR \
  --batch_size $BATCH_SIZE \
  --iterations 150000 \
  --num_classes 7 \
  --classifier_out_channel 7 \
  --dataset_type cifar10 \
  "${CLASSIFIER_FLAGS[@]}"
