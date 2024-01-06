import wget

url = "https://huggingface.co/heatingma/JCCX0021-LoRA-SD/resolve/main/Scarlett.safetensors"
filename = "text2img/lora-finetune/Scarlett/pytorch_lora_weights.safetensors"

wget.download(url, filename)