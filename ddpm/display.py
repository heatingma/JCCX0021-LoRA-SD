import tqdm
import torch
import PIL.Image
import numpy as np
from diffusers import UNet2DModel
from diffusers import DDPMScheduler

torch.manual_seed(1234)

def save_sample(sample, filename):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)
    image_pil = PIL.Image.fromarray(image_processed[0])
    image_pil.save(filename)


scheduler = DDPMScheduler.from_config("ddpm/ddpm-celebahq-256")
model = UNet2DModel.from_pretrained("ddpm/ddpm-celebahq-256")
model.to("cuda")
noisy_sample = torch.randn(
    1, model.config.in_channels, model.config.sample_size, model.config.sample_size
)
noisy_sample = noisy_sample.to("cuda")
with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample  
less_noisy_sample = scheduler.step(
    model_output=noisy_residual, timestep=2, sample=noisy_sample
).prev_sample   
sample = noisy_sample


for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
  with torch.no_grad():
      residual = model(sample, t).sample
  sample = scheduler.step(residual, t, sample).prev_sample
  if (i + 1) % 10 == 0 and i > 500:
      filename = "outputs/ddpm/celebahq-display/ddpm_sample_{}.png".format(i+1)
      save_sample(sample, filename)