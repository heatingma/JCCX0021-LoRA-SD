from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os

pipe = StableDiffusionPipeline.from_pretrained("dreambooth/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.to("cuda")
# pipe.load_lora_weights("path-to-save-model_lyf/checkpoint-50", weight_name="pytorch_lora_weights.safetensors")
# pipe.unet.load_attn_procs("path-to-save-model_lyf/checkpoint-1000", weight_name="pytorch_lora_weights.safetensors")
os.makedirs('./lyf', exist_ok=True)
generator = torch.Generator(device='cuda').manual_seed(0)

pipeline_args = {"prompt": 'a nice photo of liuyifei'}
images = []
for _ in range(1):
    image = pipe('a nice photo of liuyifei', generator=generator, num_inference_steps=25).images[0]
    images.append(image)
for i in range(len(images)):
    images[i].save(f'lyf/lyf_{i}.png')