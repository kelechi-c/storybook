from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableCascadeUNet,
)
# model ids
cascade_model_id = "stabilityai/stable-cascade"
cascade_prior_id = "stabilityai/stable-cascade-prior"
# testing text prompt
prompt = "dim-lit painting of two canary birds renaissance style"
negative_prompt = "low quality"

# unet models for stage C and B
prior_unet = StableCascadeUNet.from_pretrained(
    "stabilityai/stable-cascade-prior", subfolder="prior_lite"
)

decoder_unet = StableCascadeUNet.from_pretrained(
    "stabilityai/stable-cascade", subfolder="decoder_lite"
)

# stage C model for highly compressed latent generation
prior = StableCascadePriorPipeline.from_pretrained(cascade_prior_id, prior=prior_unet) # load prior for stage C

prior.enable_model_cpu_offload() # to save memory during inference
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=1,
    num_inference_steps=20,
)


# load upscaler/decoder for stage B/A
decoder = StableCascadeDecoderPipeline.from_pretrained(
    cascade_model_id, decoder=decoder_unet
) 

decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10,
).images[0]

decoder_output.save("image_cascade.png")