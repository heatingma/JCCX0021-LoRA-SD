from diffusers import DDPMPipeline

ddpm = DDPMPipeline.from_pretrained("ddpm/ddpm-celebahq-256")
num_inference_steps = 1000
image = ddpm(num_inference_steps=num_inference_steps)[0][0]
image.save("outputs/ddpm/celebahq.png")