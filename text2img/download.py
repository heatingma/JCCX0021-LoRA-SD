"""
download stable-diffusion-v1-5 from hugging face
"""

import wget

url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/{}/{}"
filename = "text2img/stable-diffusion-v1-5/{}/{}"

download_pairs = (
    ["safety_checker", "model.safetensors"],
    ["safety_checker", "pytorch_model.bin"],
    ["text_encoder", "model.safetensors"],
    ["text_encoder", "pytorch_model.bin"],
    ["unet", "diffusion_pytorch_model.bin"],
    ["vae", "diffusion_pytorch_model.bin"],
    ["vae", "diffusion_pytorch_model.safetensors"]
)

for download_pair in download_pairs:
    _url = url.format(download_pair[0], download_pair[1])
    _filename = filename.format(download_pair[0], download_pair[1])
    wget.download(_url, _filename)