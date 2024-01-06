import wget

url = "https://huggingface.co/google/ddpm-celebahq-256/resolve/main/diffusion_pytorch_model.bin"
filename = "ddpm/ddpm-celebahq-256/diffusion_pytorch_model.bin"

wget.download(url, filename)