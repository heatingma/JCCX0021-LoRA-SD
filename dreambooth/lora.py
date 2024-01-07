import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler
from huggingface_hub import model_info

# LoRA weights ~3 MB
model_path = "lora/lyf_1"

# info = model_info(model_path)
# model_base = info.cardData["base_model"]
# print(model_base)   # CompVis/stable-diffusion-v1-4


# pipe = StableDiffusionPipeline.from_pretrained("dreambooth/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained("lora/lyf_1", torch_dtype=torch.float16)
# pipe = StableDiffusionPipeline.from_pretrained("lora/dili_1", torch_dtype=torch.float16)
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(8)

image = pipe("a nice photo of liuyifei", generator=generator, num_inference_steps=25).images[0]
image.save("lyf.png")
