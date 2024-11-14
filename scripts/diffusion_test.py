import torch
# from diffusers import StableDiffusionPipeline, AutoencoderKL, FluxPipeline # Flux will need a new version of diffusers
import imageio
import cv2
from diffusers import StableDiffusionImageVariationPipeline
import argparse

def VAE_encode_decode(input_image_path):

    input_image = imageio.imread(input_image_path)
    input_image = (input_image / 255.0)
    # resize to 256x256
    input_image = cv2.resize(input_image, (256, 256))
    # alpha blending
    input_image = input_image[:, :, :3] * input_image[:, :, 3:4] + (1 - input_image[:, :, 3:4])
    # save original image before VAE encoding and decoding
    imageio.imsave('input.png', (input_image * 255).astype('uint8'))

    input_image = input_image * 2 - 1.0

    # sd 2.1
    # pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers")
    # vae = pipe.vae.to("cuda")

    # # # FLUX
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    # vae = pipe.vae.to("cuda")

    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0",
    )
    vae = sd_pipe.vae.to("cuda")

    input_image = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).to("cuda").float()
    latent_output = vae.encode(input_image).latent_dist.mode() * vae.config.scaling_factor
    generator = torch.Generator()
    generator.manual_seed(0)
    predicted_imgs = vae.decode(latent_output * 1 / vae.config.scaling_factor, return_dict=False)[0] # [b*v, 3, h, w]
    predicted_imgs = (predicted_imgs.permute(0, 2, 3, 1).cpu().detach().numpy() + 1) * 0.5
    predicted_imgs = (predicted_imgs.clip(min=0, max=1) * 255).astype('uint8')
    # save
    imageio.imsave('predicted_img.png', predicted_imgs[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a Neural Gaffer training script.")
    parser.add_argument(
        "--input_image_path",
        type=str,
        default="./demo/vae_test/lego.png",
    )
    args = parser.parse_args()

    VAE_encode_decode(args.input_image_path)



    


