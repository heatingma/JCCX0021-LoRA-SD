CUDA_VISIBLE_DEVICES=3 python text2img/generate.py \
  --prompt "liuyifei, nice, pretty, beauty, smooth, Highest quality, ultra-high definition, masterpieces" \
  --filename "liuyifei" \
  --samples_num 3 \
  --lora_dir "text2img/lora-finetune/lyf"