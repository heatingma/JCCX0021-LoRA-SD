export OUTPUT_DIR="ddpm/celeba-hq/train"
export DATASET_NAME="ddpm/celeba-hq"

CUDA_VISIBLE_DEVICES=1 python ddpm/train.py \
  --dataset_name=$DATASET_NAME \
  --output_dir=$OUTPUT_DIR \
