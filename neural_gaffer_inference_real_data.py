import os
from pathlib import Path
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm
from dataset.dataset_relighting_eval_real import Relighting_Data

from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline

import torchvision

logger = get_logger(__name__)

from parse_args import parse_args



def log_validation(validation_dataloader, vae, image_encoder, feature_extractor, unet, args, accelerator, weight_dtype, img_per_object=600, split="val"):
    logger.info("Running {} validation... ".format(split))

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = Neural_Gaffer_StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae).eval(),
        image_encoder=accelerator.unwrap_model(image_encoder).eval(),
        feature_extractor=feature_extractor,
        unet=accelerator.unwrap_model(unet).eval(),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()


    predicted_images = [] # [num_validation_batches, ], each element is a np.array of [batch_size, h, w, 3]
    gt_images = [] # [num_validation_batches, ], each element is a np.array of [batch_size, h, w, 3]
    LDR_target_environment_maps = []
    HDR_target_environment_maps = []
    input_images = []
    input_image_names = []
    target_lighting_names = []
    target_view_idx_list = []

    for valid_step, batch in tqdm(enumerate(validation_dataloader)):
        if args.num_validation_batches is not None and valid_step >= args.num_validation_batches:
            break
        input_image = batch["image_cond"].to(dtype=weight_dtype)

        target_envmap_ldr = batch["envir_map_target_ldr"].to(dtype=weight_dtype)
        target_envmap_hdr = batch["envir_map_target_hdr"].to(dtype=weight_dtype)
        cond_image_names =  batch["cond_img_name"]
        target_view_idx = batch["target_view_idx"]
        target_envir_map_names = batch["target_envir_map_name"]
        for i in range(len(cond_image_names)):
            input_image_names.append(cond_image_names[i])
            target_lighting_names.append(target_envir_map_names[i])
            target_view_idx_list.append(target_view_idx[i])

        cur_predicted_images = []  # Initialize here
        batchsize, _, h, w = input_image.shape
        
        generartor_list = [torch.Generator(device=accelerator.device).manual_seed(args.seed) for _ in range(batchsize)]
        for _ in range(args.num_validation_images): # sampled times
            with torch.autocast("cuda"):
                # todo: change the name of "cond_envir_map" to "target_envmap_hdr"
                pipeline_output_images = pipeline(input_imgs=input_image, prompt_imgs=input_image, 
                                first_target_envir_map=target_envmap_hdr, second_target_envir_map=target_envmap_ldr, poses=None, 
                                height=h, width=w,
                                guidance_scale=args.guidance_scale, num_inference_steps=50, generator=generartor_list).images

            cur_predicted_images.append(pipeline_output_images) # PIL image list [num_validation_images, batch_size]
            
        
        # [-1, 1][batch_size, 3, h, w] -> [0, 1][batch_size, h, w, 3]
        envir_map_target_hdr_npy = 0.5 * (np.array(target_envmap_hdr.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        envir_map_target_ldr_npy = 0.5 * (np.array(target_envmap_ldr.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)


        input_image_npy = 0.5 * (np.array(input_image.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        input_images.append(input_image_npy)


        prediction_image_sample0_list = []
        for i in range(batchsize):
            prediction_image_sample0_list.append(np.array(cur_predicted_images[0][i]))
        prediction_image_sample0 = np.array(prediction_image_sample0_list, dtype=np.float32) / 255.0
        # prediction_image_sample1 = np.array([cur_predicted_images[1][i] for i in range(batchsize)], dtype=np.float32) / 255.0
        predicted_images.append(prediction_image_sample0)
        LDR_target_environment_maps.append(envir_map_target_ldr_npy)        
        HDR_target_environment_maps.append(envir_map_target_hdr_npy)
            
    predicted_images = np.concatenate(predicted_images, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    LDR_target_environment_maps = np.concatenate(LDR_target_environment_maps, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    input_images = np.concatenate(input_images, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    
    
    # save results
    save_dir = os.path.join(args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for image_idx in range(predicted_images.shape[0]):

        cur_predicted_image = predicted_images[image_idx]
        cur_target_envmap_ldr = LDR_target_environment_maps[image_idx]
        cur_input_image = input_images[image_idx]
        cur_input_image_name = input_image_names[image_idx]
        cur_target_lighting_name = target_lighting_names[image_idx]
        cur_target_view_idx = target_view_idx_list[image_idx]
        if not os.path.exists(f'{save_dir}/{cur_input_image_name}'):
            os.makedirs(f'{save_dir}/{cur_input_image_name}')
            os.makedirs(f'{save_dir}/{cur_input_image_name}/input_image')
            os.makedirs(f'{save_dir}/{cur_input_image_name}/target_envmap_ldr')
            os.makedirs(f'{save_dir}/{cur_input_image_name}/pred_image')
        input_image_PIL = Image.fromarray((cur_input_image * 255).astype(np.uint8))
        target_envmap_ldr_PIL = Image.fromarray((cur_target_envmap_ldr * 255).astype(np.uint8))
        pred_image_PIL = Image.fromarray((cur_predicted_image * 255).astype(np.uint8))

        target_envmap_ldr_PIL.save(f'{save_dir}/{cur_input_image_name}/target_envmap_ldr/{cur_target_lighting_name}_{cur_target_view_idx:03d}.png')
        pred_image_PIL.save(f'{save_dir}/{cur_input_image_name}/pred_image/{cur_target_lighting_name}_{cur_target_view_idx:03d}.png')
        input_image_PIL.save(f'{save_dir}/{cur_input_image_name}/input_image/{cur_target_lighting_name}_{cur_target_view_idx:03d}.png')


    return True


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)



    # Load scheduler and models
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
    feature_extractor = None
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    # zero init unet conv_in from 8 channels to 16 channels
    conv_in_16 = torch.nn.Conv2d(16, unet.conv_in.out_channels, kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding)
    conv_in_16.requires_grad_(False)
    unet.conv_in.requires_grad_(False)
    torch.nn.init.zeros_(conv_in_16.weight)
    conv_in_16.weight[:,:8,:,:].copy_(unet.conv_in.weight)
    conv_in_16.bias.copy_(unet.conv_in.bias)
    unet.conv_in = conv_in_16
    unet.requires_grad_(False)



    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. 'Please make sure to always have all model weights in full float32 precision when starting training'"
        )


    # Init Dataset
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.resolution, args.resolution), antialias=True),  # 256, 256
            transforms.ToTensor(), # for PIL to Tensor [0, 255] -> [0.0, 1.0] and H×W×C-> C×H×W
            transforms.Normalize([0.5], [0.5]) # x -> (x - 0.5) / 0.5 == 2 * x - 1.0; [0.0, 1.0] -> [-1.0, 1.0]
        ]
    )
 
    

    lighting_per_view = args.lighting_per_view
    total_view = args.total_view
    img_per_object = lighting_per_view * total_view
    validation_dataset = Relighting_Data(
        lighting_dir = args.lighting_dir,
        img_dir = args.validation_data_dir,   
        lighting_per_view=lighting_per_view,
        total_view=total_view,
        image_transforms=image_transforms,
        specific_object=args.specific_object 

    ) 
    # import ipdb; ipdb.set_trace()
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
    )

    # Prepare everything with our `accelerator`.
    unet = accelerator.prepare(unet)
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, image_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist !!!"
            )
            os._exit(1)
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))

    else:
        print("No checkpoint found. Validation Failed")
        
    print("Loading checkpoint finished!!!!")

    
    if validation_dataloader is not None:

        _ = log_validation(
            validation_dataloader,
            vae,
            image_encoder,
            feature_extractor,
            unet,
            args,
            accelerator,
            weight_dtype,
            split='real_img',
            img_per_object=img_per_object
        )
  
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
