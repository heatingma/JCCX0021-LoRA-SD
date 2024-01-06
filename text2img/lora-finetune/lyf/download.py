import wget

url = "https://huggingface.co/heatingma/JCCX0021-LoRA-SD/resolve/main/lyf.safetensors"
filename = "text2img/lora-finetune/lyf/pytorch_lora_weights.safetensors"

wget.download(url, filename)