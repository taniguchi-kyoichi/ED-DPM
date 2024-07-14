#!/bin/bash

# CUDAデバイスの設定
export CUDA_VISIBLE_DEVICES=0

# 分散トレーニングの設定
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# データとログのディレクトリ設定
DATA_DIR='datasets/train'
VAL_DATA_DIR='datasets/val'
LOG_DIR='logs/image'

# トレーニング用のパラメータ設定
IMAGE_SIZE=256
NUM_CHANNELS=128
NUM_RES_BLOCKS=1
NUM_HEADS=4
ATTENTION_RESOLUTIONS="32,16,8"
DROPOUT=0.1
CLASS_COND=True
USE_FP16=True
DIFFUSION_STEPS=1000
NOISE_SCHEDULE="linear"
TIMESTEP_RESPACING="ddim25"
BATCH_SIZE=2  # メモリ不足を避けるために減らす
#ITERATIONS=150000
#DATASET_TYPE="custom"

# トレーニングスクリプトの実行
python scripts/image_train.py \
  --data_dir $DATA_DIR \
  --log_dir $LOG_DIR \
  --image_size $IMAGE_SIZE \
  --num_channels $NUM_CHANNELS \
  --num_res_blocks $NUM_RES_BLOCKS \
  --num_heads $NUM_HEADS \
  --attention_resolutions $ATTENTION_RESOLUTIONS \
  --dropout $DROPOUT \
  --class_cond $CLASS_COND \
  --use_fp16 $USE_FP16 \
  --diffusion_steps $DIFFUSION_STEPS \
  --noise_schedule $NOISE_SCHEDULE \
  --timestep_respacing $TIMESTEP_RESPACING \
  --batch_size $BATCH_SIZE \
#  --iterations $ITERATIONS \
#  --dataset_type $DATASET_TYPE