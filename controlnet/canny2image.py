from share import *
import config
import json
import cv2
import einops
import numpy as np
import torch
import random
import os

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from train_lora import inject_lora

apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
inject_lora(model.control_model)
print(model.load_state_dict(load_state_dict('lightning_logs/version_0/checkpoints/epoch=99-step=1499.ckpt', location='cuda')))
model = model.cuda()
model.eval()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

num_samples = 1
image_resolution = 512
guess_mode = False
strength = 1.0
low_threshold = 150
high_threshold = 300
ddim_steps = 20
scale = 9
seed = 777
eta = 0.
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
dir = 'caricature'
save_dir = 'lyf'
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(dir, 'metadata.jsonl'), 'r') as f:
    for line in f:
        line = json.loads(line.strip())
        prompt = line['additional_feature']
        img_id = line['file_name']
        img = cv2.imread(os.path.join(dir, img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = process(img, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
        for i, item in enumerate(results):
            item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, img_id[:-4] + '_' + str(i) + '.png'), item)
