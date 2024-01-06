export MODEL_NAME="text2img/stable-diffusion-v1-5"
export OUTPUT_DIR="text2img/lora-finetune/lyf30"
export DATASET_NAME="datasets/ddpm-lyf30-512_512"

CUDA_VISIBLE_DEVICES=3 python text2img/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers 0 \
  --resolution 512 \
  --train_batch_size 1 \
  --caption_column "additional_feature"  \
  --gradient_accumulation_steps 1 \
  --max_train_steps 600 \
  --learning_rate 1e-05 \
  --max_grad_norm 1 \
  --lr_scheduler "cosine" \
  --lr_warmup_steps 0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps 100 \
  --validation_prompt "liuyifei, very beautiful, (((beauty))), (((smooth))), white, ((fairy)), Highest quality, ultra-high definition, masterpieces, 8k,realistic,1girl, skinny, light smile, delicate face, whole face, harmonious" \
  --report_to=wandb \
  --seed=1337

