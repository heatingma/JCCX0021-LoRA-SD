import os
import argparse
from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Training Script.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=False, default=None)
    parser.add_argument("--samples_num", type=int, required=False, default=1)
    args = parser.parse_args()
    return args


def main(args):
    pipe = StableDiffusionPipeline.from_pretrained("text2img/stable-diffusion-v1-5")
    pipe = pipe.to('cuda')
    if args.lora_dir is not None:
        pipe: StableDiffusionPipeline
        pipe.load_lora_weights(args.lora_dir, weight_name="pytorch_lora_weights.safetensors")
    num = 0
    while num < args.samples_num:
        image, has_nsfw_concept = pipe(args.prompt, return_dict=False, num_inference_steps=999)
        if has_nsfw_concept[0] == False:
            save_dir = f"outputs/text2img/{args.filename}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image[0].save(save_dir + f"/{args.filename}_{num}.png")
            num = num + 1


if __name__ == "__main__":
    args = parse_args()
    main(args)