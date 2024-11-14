import torchvision.transforms.functional as TF
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

import sys
sys.path.append('./')

from diffusion_relighting import RelightingStableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection
import kornia
import os
from glob import glob
class RelightingDiffusion(nn.Module):
    def __init__(self, device, fp16=True, t_range=[0.02, 0.98], model_key="kxic/zero123-xl", 
                 checkpoint_path='../logs/neural_gaffer_res256/checkpoint-80000'):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        assert self.fp16, 'Only zero123 fp16 is supported for now.'


        accelerator = Accelerator(
            mixed_precision="fp16" if self.fp16 else None,
            log_with='wandb',
            project_config=None,
        )
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_key, subfolder="image_encoder")
        vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")
        
        scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)
        conv_in_16 = torch.nn.Conv2d(16, unet.conv_in.out_channels, kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding)
        conv_in_16 = conv_in_16.to(weight_dtype)
        unet.conv_in = conv_in_16
        unet.requires_grad_(False)
        # Prepare everything with our `accelerator`.
        unet = accelerator.prepare(unet)

        unet.to(accelerator.device, dtype=weight_dtype)
        # Move vae, image_encoder to device and cast to weight_dtype
        vae.to(accelerator.device, dtype=weight_dtype)
        image_encoder.to(accelerator.device, dtype=weight_dtype)
        accelerator.load_state(checkpoint_path)
         
        self.pipe = RelightingStableDiffusionPipeline.from_pretrained(
            model_key,
            vae=accelerator.unwrap_model(vae).eval(),
            image_encoder=accelerator.unwrap_model(image_encoder).eval(),
            feature_extractor=None,
            unet=accelerator.unwrap_model(unet).eval(),
            scheduler=scheduler,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(self.device)


        self.pipe.image_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.unet.eval()


        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.embeddings = None

    def CLIP_preprocess(self, x):
        dtype = x.dtype
        # following openai's implementation
        # TODO HF OpenAI CLIP preprocessing issue https://github.com/huggingface/transformers/issues/22505#issuecomment-1650170741
        # follow openai preprocessing to keep exact same, input tensor [-1, 1], otherwise the preprocessing will be different, https://github.com/huggingface/transformers/pull/22608
        if isinstance(x, torch.Tensor):
            if x.min() < -1.0 or x.max() > 1.0:
                raise ValueError("Expected input tensor to have values in the range [-1, 1]")
        x = kornia.geometry.resize(x.to(torch.float32), (224, 224), interpolation='bicubic', align_corners=True, antialias=False).to(dtype=dtype)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                                     torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
        return x
    @torch.no_grad()
    def get_img_embeds(self, image_list):
        assert type(image_list) == list, 'input should be a list of images'
        embedding_list = []
        for input_image, hdr_map, ldr_map in image_list:
            # input_image: image tensor in [0, 1]
            input_image = F.interpolate(input_image, (256, 256), mode='bilinear', align_corners=False)
            hdr_map = F.interpolate(hdr_map, (256, 256), mode='bilinear', align_corners=False)
            ldr_map = F.interpolate(ldr_map, (256, 256), mode='bilinear', align_corners=False)
            # import ipdb; ipdb.set_trace()
            # input_image_pil = TF.to_pil_image(input_image.squeeze(0))
            x_clip = self.CLIP_preprocess(input_image)
            c = self.pipe.image_encoder(x_clip.to(self.dtype)).image_embeds
            input_image_vae = self.encode_imgs(input_image.to(self.dtype)) / self.vae.config.scaling_factor
            hdr_latent = self.encode_imgs(hdr_map.to(self.dtype)) / self.vae.config.scaling_factor
            ldr_latent = self.encode_imgs(ldr_map.to(self.dtype)) / self.vae.config.scaling_factor
            cur_embeddings = [c, input_image_vae, hdr_latent, ldr_latent]
            embedding_list.append(cur_embeddings)
        self.embeddings = embedding_list
    
    


    @torch.no_grad()
    def refine(self, pred_rgb, image_idx,
               guidance_scale=3, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        
        self.scheduler.set_timesteps(steps)

        if strength == 0:
            init_step = 0
            latents = torch.randn((batch_size, 4, 32, 32), device=self.device, dtype=self.dtype)
        else:
            init_step = int(steps * strength)
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype)) # has been rescaled
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        cc_emb_list = []
        vae_emb_list = []
        hdr_latent_list = []
        ldr_latent_list = []
        for i in range(len(image_idx)):
            cur_idx = image_idx[i]
            cc_emb = self.embeddings[cur_idx][0].unsqueeze(0)
            vae_emb = self.embeddings[cur_idx][1]
            
            hdr_latent = self.embeddings[cur_idx][2]
            ldr_latent = self.embeddings[cur_idx][3]
            cc_emb_list.append(cc_emb)
            vae_emb_list.append(vae_emb)
            hdr_latent_list.append(hdr_latent)
            ldr_latent_list.append(ldr_latent)
        # import ipdb; ipdb.set_trace()
        cc_emb = torch.cat(cc_emb_list, dim=0)
        vae_emb = torch.cat(vae_emb_list, dim=0)
        hdr_latent = torch.cat(hdr_latent_list, dim=0)
        ldr_latent = torch.cat(ldr_latent_list, dim=0)
        cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)
        vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)
        hdr_latent = torch.cat([hdr_latent, torch.zeros_like(hdr_latent)], dim=0)
        ldr_latent = torch.cat([ldr_latent, torch.zeros_like(ldr_latent)], dim=0)

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            
            x_in = torch.cat([latents] * 2)
            t_in = t.view(1).to(self.device)
            noise_pred = self.unet(
                torch.cat([x_in, vae_emb, hdr_latent, ldr_latent], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample.to(self.unet.dtype) # TODO: WHY should I transform the dtype mannually?
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 256, 256]
        return imgs
    
    def train_step(self, pred_rgb, image_idx:list, step_ratio=None, guidance_scale=5, as_latent=False):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        batch_size = pred_rgb.shape[0]

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

        # hdr_latent = self.encode_imgs(hdr_map.to(self.dtype)) / self.vae.config.scaling_factor
        # ldr_latent = self.encode_imgs(ldr_map.to(self.dtype)) / self.vae.config.scaling_factor

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)
            
            cc_emb = []
            vae_emb = []
            hdr_latent = []
            ldr_latent = []
            for cur_idx in image_idx:
                cc_emb.append(self.embeddings[cur_idx][0].unsqueeze(0))
                vae_emb.append(self.embeddings[cur_idx][1])
                hdr_latent.append(self.embeddings[cur_idx][2])
                ldr_latent.append(self.embeddings[cur_idx][3])
            cc_emb = torch.cat(cc_emb, dim=0)
            vae_emb = torch.cat(vae_emb, dim=0)
            hdr_latent = torch.cat(hdr_latent, dim=0)
            ldr_latent = torch.cat(ldr_latent, dim=0)
            # import ipdb; ipdb.set_trace()
            cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)
            vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)
       

            # hdr_latent = self.embeddings[image_idx][2].repeat(batch_size, 1)
            # ldr_latent = self.embeddings[image_idx][3].repeat(batch_size, 1)
            hdr_latent = torch.cat([hdr_latent, torch.zeros_like(hdr_latent)], dim=0)
            ldr_latent = torch.cat([ldr_latent, torch.zeros_like(ldr_latent)], dim=0)
            noise_pred = self.unet(
                torch.cat([x_in, vae_emb, hdr_latent, ldr_latent], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample.to(self.unet.dtype) #TODO

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum')

        return loss
    

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        # import ipdb; ipdb.set_trace()
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs, mode=False):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample() 
        latents = latents * self.vae.config.scaling_factor

        return latents
    
    
if __name__ == '__main__':
    import cv2
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    import kiui

    parser = argparse.ArgumentParser()

    # parser.add_argument('input', type=str)

    opt = parser.parse_args()

    device = torch.device('cuda')

    # print(f'[INFO] loading image from {opt.input} ...')
    # image = kiui.read_image(opt.input, mode='tensor')
    # image = image.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    # image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)

    print(f'[INFO] loading model ...')
    
    zero123 = RelightingDiffusion(device, model_key="kxic/zero123-xl")
    Data_Base_Path = '/home/hj453/code/dreamgaussian/test_results'
    input_image_dir = os.path.join(Data_Base_Path, 'input_image')
    pred_image_dir = os.path.join(Data_Base_Path, 'pred_image')
    hdr_map_dir = os.path.join(Data_Base_Path, 'target_envmap_hdr')
    ldr_map_dir = os.path.join(Data_Base_Path, 'target_envmap_ldr')
    target_RT_dir = os.path.join(Data_Base_Path, 'target_RT')
    image_paths = glob(os.path.join(input_image_dir, '*_004_*.png'))
    image_paths.sort()
    image_names = [os.path.basename(image_path) for image_path in image_paths]
    image_list = []
    RT_list = []
    pred_image_list = []
    for idx, image_name in enumerate(image_names):
        input_image = kiui.read_image(os.path.join(input_image_dir, image_name), mode='tensor')
        pred_image = kiui.read_image(os.path.join(pred_image_dir, image_name), mode='tensor')
        hdr_map = kiui.read_image(os.path.join(hdr_map_dir, image_name), mode='tensor')
        ldr_map = kiui.read_image(os.path.join(ldr_map_dir, image_name), mode='tensor')
        input_image = input_image.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        hdr_map = hdr_map.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        ldr_map = ldr_map.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        cur_image = [input_image, hdr_map, ldr_map]
        image_list.append(cur_image)
        RT_path = os.path.join(target_RT_dir, image_name.replace('.png', '.npy'))
        RT = np.load(RT_path)
        RT_list.append(RT)
        pred_image = pred_image.permute(2, 0, 1).unsqueeze(dim=0).contiguous().to(device)
        pred_image_list.append(pred_image)

    zero123.get_img_embeds(image_list)
    input_image_list = []
    pred_image = torch.cat(pred_image_list, dim=0)
    # image_idx_list = []
    # for idx, image in enumerate(image_list):
    #     input_image_list.append(image[0])
    #     image_idx_list.append(idx)
    # input_image_list = torch.cat(input_image_list, dim=0)
    # zero123.train_step(input_image_list,image_idx_list)
    # import ipdb; ipdb.set_trace()
    
    temp_image = pred_image[:3]
    temp_index = [0, 1, 2]
    
    outputs = zero123.refine(temp_image, temp_index, strength=0.8)

    # import ipdb; ipdb.set_trace()
    # save
    outputs = outputs.float().cpu().numpy().transpose(0, 2, 3, 1)[1]
    outputs = (outputs * 255).astype(np.uint8)
    outputs = cv2.cvtColor(outputs, cv2.COLOR_RGB2BGR)
    cv2.imwrite('outputs.png', outputs)
    # plt.imshow(outputs.float().cpu().numpy().transpose(0, 2, 3, 1)[0])
    # plt.show()
    # print(f'[INFO] running model ...')
    # # zero123.get_img_embeds(image)

    # azimuth = opt.azimuth
    # image_idx = 0
    # hdr_map, ldr_map = None, None
    # while True:
    #     outputs = zero123.refine(image, image_idx, hdr_map, ldr_map, strength=0)
    #     plt.imshow(outputs.float().cpu().numpy().transpose(0, 2, 3, 1)[0])
    #     plt.show()
    #     azimuth = (azimuth + 10) % 360