import logging
import math
import os
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from dataset.dataset_relighting_training import NeuralGafferTrainingDataLoader, NeuralGafferTrainingData
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPFeatureExtractor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from torchmetrics.image import StructuralSimilarityIndexMeasure
from parse_args import parse_args
import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline

from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import EMAModel
import torchvision
import kornia

import datetime

# torch.distributed.init_process_group('nccl', init_method=None, timeout=datetime.timedelta(seconds=1800), world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None)
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=10000))
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings("ignore", message="cc_projection/diffusion_pytorch_model.safetensors not found")
warnings.filterwarnings("ignore", message="The config attributes {'cc_projection': ['pipeline_zero1to3', 'CCProjection']} were passed to Neural_Gaffer_StableDiffusionPipeline, but are not expected and will be ignored. Please verify your model_index.json configuration file.")
warnings.simplefilter(action='ignore', category=FutureWarning)
# diffusers.logging.set_verbosity_error()

if is_wandb_available():
    import wandb
os.environ['WANDB_CONFIG_DIR'] = "/tmp/.config-" + os.environ['USER']
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.19.0.dev0")

logger = get_logger(__name__)
# from parse_args import parse_args


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def log_validation(validation_dataloader, vae, image_encoder, feature_extractor, unet, args, accelerator, weight_dtype, split="val", cur_step=0):
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

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []

    predicted_images = [] # [num_validation_batches, ], each element is a np.array of [batch_size, h, w, 3]
    gt_images = [] # [num_validation_batches, ], each element is a np.array of [batch_size, h, w, 3]
    
    
    for valid_step, batch in tqdm(enumerate(validation_dataloader)):
        if args.num_validation_batches is not None and valid_step >= args.num_validation_batches:
            break
        gt_image = batch["image_target"].to(dtype=weight_dtype)
        input_image = batch["image_cond"].to(dtype=weight_dtype)
        target_envmap_ldr = batch["envir_map_target_ldr"].to(dtype=weight_dtype)
        target_envmap_hdr = batch["envir_map_target_hdr"].to(dtype=weight_dtype)
        pose = batch["T"].to(dtype=weight_dtype)
        # target_orientation = batch["target_orientation"].to(dtype=weight_dtype)
        # pose = torch.cat([pose, target_orientation], dim=-1)
        cur_predicted_images = []
        batchsize, _, h, w = input_image.shape
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                # todo: change the name of "first_target_envir_map" to "target_envmap_hdr"
                pipeline_output_images = pipeline(input_imgs=input_image, prompt_imgs=input_image, poses=pose, 
                                        first_target_envir_map=target_envmap_hdr , second_target_envir_map=target_envmap_ldr, 
                                        height=h, width=w,
                                        guidance_scale=args.guidance_scale, num_inference_steps=50, generator=generator).images
                

            cur_predicted_images.append(pipeline_output_images) # PIL image list [num_validation_images, batch_size]
        
        # [-1, 1][batch_size, 3, h, w] -> [0, 1][batch_size, h, w, 3]
        envir_map_target_hdr_npy = 0.5 * (np.array(target_envmap_hdr.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        envir_map_target_ldr_npy = 0.5 * (np.array(target_envmap_ldr.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        gt_image_npy = 0.5 * (np.array(gt_image.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        input_image_npy = 0.5 * (np.array(input_image.permute([0, 2, 3, 1]).cpu(), dtype=np.float32) + 1.0)
        
        prediction_image_sample0_list = []
        prediction_image_sample1_list = []
        for i in range(batchsize):
            prediction_image_sample0_list.append(np.array(cur_predicted_images[0][i]))
            prediction_image_sample1_list.append(np.array(cur_predicted_images[1][i]))
        prediction_image_sample0 = np.array(prediction_image_sample0_list, dtype=np.float32) / 255.0
        prediction_image_sample1 = np.array(prediction_image_sample1_list, dtype=np.float32) / 255.0

        predicted_images.append(prediction_image_sample0)
        gt_images.append(gt_image_npy)
        
        # concatenate the images to a single image
        ## [batch_size, h, w, 3] -> [batch_size * h, w, 3]

        input_image_npy_new = input_image_npy.reshape((1,-1, w, 3)).squeeze()
        gt_image_npy_new = gt_image_npy.reshape((1,-1, w, 3)).squeeze().squeeze()
        prediction_image_sample0_new = prediction_image_sample0.reshape((1,-1, w, 3)).squeeze()
        prediction_image_sample1_new = prediction_image_sample1.reshape((1,-1, w, 3)).squeeze()
        envir_map_target_hdr_npy_new = envir_map_target_hdr_npy.reshape((1,-1, w, 3)).squeeze()
        envir_map_target_ldr_npy_new = envir_map_target_ldr_npy.reshape((1,-1, w, 3)).squeeze()
        concatenated_image = np.concatenate([input_image_npy_new, gt_image_npy_new, prediction_image_sample0_new, prediction_image_sample1_new, envir_map_target_ldr_npy_new, envir_map_target_hdr_npy_new], axis=1)

        # save the concatenated image
        concatenated_image = Image.fromarray((concatenated_image * 255).astype(np.uint8))
        # image_logs.append({"gt_image": gt_image, "envir_map_target_hdr":target_envmap_hdr, "envir_map_target_ldr":target_envmap_ldr,  "target_envmap_name":target_envmap_name, "pred_images": images, "pose": pose, "input_image": input_image})
        image_logs.append({"result": concatenated_image})
        
        
    val_metrics = {}
    gt_images = np.concatenate(gt_images, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    predicted_images = np.concatenate(predicted_images, axis=0) # [num_validation_batches * batch_size, h, w, 3]
    # compute metrics
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    ## PSNR
    mse = np.mean((gt_images - predicted_images) ** 2, axis=(1, 2, 3)) 
    PSNR = 10 * np.log10(1.0 / mse)
    # print("PSNR: ", PSNR)
    mean_psnr = np.mean(PSNR)
    
    predicted_images_tensor = torch.tensor(predicted_images).permute([0, 3, 1, 2]) # [num, 3, h, w]
    gt_images_tensor = torch.tensor(gt_images).permute([0, 3, 1, 2]) # [num, 3, h, w]
    ## LPIPS
    mean_lpips_loss = lpips(predicted_images_tensor * 2 - 1, gt_images_tensor * 2 - 1).mean().item()
    # print("LPIPS: ", mean_lpips_loss)
    
    ## SSIM
    mean_ssim_loss = ssim(predicted_images_tensor, gt_images_tensor).mean().item()
    # print("SSIM: ", mean_ssim_loss)
    
    
    logs = {"lpips_loss/{}".format(split): mean_lpips_loss, "ssim_loss/{}".format(split): mean_ssim_loss, "PSNR/{}".format(split): mean_psnr}
    val_metrics.update(logs)
    # after validation, set the pipeline back to training mode
    unet.train()
    vae.train()
    image_encoder.train()


    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []

            for log_id, log in enumerate(image_logs):
                formatted_images.append(wandb.Image(log["result"], caption="{}_result".format(log_id)))

            tracker.log({split: formatted_images}, step=cur_step)
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")


    return image_logs, val_metrics


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_input.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- diffusers
inference: true
---
    """
    model_card = f"""
# zero123-{repo_id}

These are zero123 weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)




def CLIP_preprocess(x):
    dtype = x.dtype
    if isinstance(x, torch.Tensor):
        if x.min() < -1.0 or x.max() > 1.0:
            raise ValueError("Expected input tensor to have values in the range [-1, 1]")
    x = kornia.geometry.resize(x.to(torch.float32), (224, 224), interpolation='bicubic', align_corners=True, antialias=False).to(dtype=dtype)   # not bf16
    x = (x + 1.) / 2.
    # renormalize according to clip
    x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
                                 torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
    return x

def _encode_image(image_encoder, image, device, dtype, do_classifier_free_guidance):

    image = image.to(device=device, dtype=dtype)
    image = CLIP_preprocess(image)
    # if not isinstance(image, torch.Tensor):
    #     # 0-255
    #     print("Warning: image is processed by hf's preprocess, which is different from openai original's.")
    #     image = self.feature_extractor(images=image, return_tensors="pt").pixel_values
    image_embeddings = image_encoder(image).image_embeds.to(dtype=dtype)
    image_embeddings = image_embeddings.unsqueeze(1)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    return image_embeddings.detach()


def _encode_image_without_pose(image_encoder, image, device, dtype, do_classifier_free_guidance):
    img_prompt_embeds = _encode_image(image_encoder, image, device, dtype, False)
    prompt_embeds = img_prompt_embeds
    # follow 0123, add negative prompt, after projection
    if do_classifier_free_guidance:
        negative_prompt = torch.zeros_like(prompt_embeds)
        prompt_embeds = torch.cat([negative_prompt, prompt_embeds])
    return prompt_embeds

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    loading_kwargs = {
        "low_cpu_mem_usage": True,
        "revision": args.revision
    }
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", **loading_kwargs)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", **loading_kwargs)
    feature_extractor = None #CLIPFeatureExtractor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", **loading_kwargs)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", **loading_kwargs)
    
    
    vae.train()
    image_encoder.train()
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
    unet.requires_grad_(True)
    unet.train()



    
    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_tiling()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"UNet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.training_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        [{"params": unet.parameters(), "lr": args.learning_rate}],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )


    # print model info, learnable parameters, non-learnable parameters, total parameters, model size, all in billion
    def print_model_info(model):
        # only rank 0 print
        if accelerator.is_main_process:
            print("="*20)
            # print model class name
            print("model name: ", type(model).__name__)
            # print("model: ", model)
            print("learnable parameters(M): ", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
            print("non-learnable parameters(M): ", sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6)
            print("total parameters(M): ", sum(p.numel() for p in model.parameters()) / 1e6)
            print("model size(MB): ", sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024)

    print_model_info(unet)
    print_model_info(vae)
    print_model_info(image_encoder)
    
    # Init Dataset
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.resolution, args.resolution)),  # 256, 256
            transforms.ToTensor(), # for PIL to Tensor [0, 255] -> [0.0, 1.0] and H×W×C-> C×H×W
            transforms.Normalize([0.5], [0.5]) # x -> (x - 0.5) / 0.5 == 2 * x - 1.0; [0.0, 1.0] -> [-1.0, 1.0]
        ]
    )
 
    
    train_dataset = NeuralGafferTrainingData(
        img_dir = args.train_img_dir,
        lighting_dir = args.train_lighting_dir,
        image_transforms=image_transforms, 
        lighting_per_view=16,
        total_view=12,
        validation=False,
        relighting_only=True,
        image_preprocessed = True,
        dataset_type='training_object_with_seen_envir'
        )
    
    # validate seen training object with unseen lighting, and the input images of are rendered with unseen lighting under unseen camera poses
    training_dataset_unseen_lighting = NeuralGafferTrainingData(
        # lighting_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_preprocessed_environment_resized/unseen_lighting', 
        # img_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_rendered_images_resized/validation/unseen_lighting/',
        lighting_dir = os.path.join(args.val_lighting_dir, 'unseen_lighting'),
        img_dir = os.path.join(args.val_img_dir, 'unseen_lighting'),
        lighting_per_view=8,
        total_view=4,
        image_transforms=image_transforms, 
        validation=True,
        relighting_only=True,
        image_preprocessed = False,
        dataset_type='training_object_with_unseen_envir'
        )   
    
    # validate unseen object with unseen lighting, and the input images of the unseen object are rendered with random area lighting 
    validation_dataset_unseen_lighting_random_light_condition = NeuralGafferTrainingData(
        # lighting_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_preprocessed_environment_resized/unseen_lighting', 
        # img_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_rendered_images_resized/validation/unseen_lighting/', 
        lighting_dir = os.path.join(args.val_lighting_dir, 'unseen_lighting'),
        img_dir = os.path.join(args.val_img_dir, 'unseen_lighting'),
        lighting_per_view=8,
        total_view=4,
        image_transforms=image_transforms, 
        validation=True,
        image_preprocessed = True,  
        relighting_only=True,
        dataset_type='unseen_object_with_random_area_light_condition'
        )       


    validation_dataset_seen_lighting = NeuralGafferTrainingData(
        # lighting_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_preprocessed_environment_resized/seen_lighting', 
        # img_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_rendered_images_resized/validation/seen_lighting/', 
        lighting_dir = os.path.join(args.val_lighting_dir, 'seen_lighting'),
        img_dir = os.path.join(args.val_img_dir, 'seen_lighting'),   
        lighting_per_view=8,
        total_view=4,
        image_transforms=image_transforms, 
        validation=True,
        image_preprocessed = True,  
        relighting_only=True,
        dataset_type='unseen_object_with_seen_envir'
        ) 
    
    validation_dataset_unseen_lighting = NeuralGafferTrainingData(
        # lighting_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_preprocessed_environment_resized/unseen_lighting', 
        # img_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/val_rendered_images_resized/validation/unseen_lighting/', 
        lighting_dir = os.path.join(args.val_lighting_dir, 'unseen_lighting'),
        img_dir = os.path.join(args.val_img_dir, 'unseen_lighting'),
        lighting_per_view=8,
        total_view=4,
        image_transforms=image_transforms, 
        validation=True,
        image_preprocessed = True,  
        relighting_only=True,
        dataset_type='unseen_object_with_unseen_envir'
        )   
    
       
    # for training
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.training_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        # prefetch_factor=3,
    )

    validation_dataloader_random_light_condition = torch.utils.data.DataLoader(
        validation_dataset_unseen_lighting_random_light_condition,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
    )

    # for validation set logs
    training_dataloader_unseen_lighting = torch.utils.data.DataLoader(
        training_dataset_unseen_lighting,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
    )

    validation_dataloader_seen_lighting = torch.utils.data.DataLoader(
        validation_dataset_seen_lighting,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
    )

    # for unseen objects validation set logs    
    validation_dataloader_unseen_lighting = torch.utils.data.DataLoader(
        validation_dataset_unseen_lighting,
        shuffle=False,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
    )

    # for training set logs
    train_log_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=4,
        num_workers=1,
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, train_log_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, train_log_dataloader, lr_scheduler
    )
    

    
    if args.use_ema:
        ema_unet.to(accelerator.device)

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

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
         # specified the run name
        output_basename = os.path.basename(args.output_dir)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs={"wandb": {"name": f'{output_basename}'}})

    # Train!
    total_batch_size = args.training_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    do_classifier_free_guidance = args.guidance_scale > 1.0
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.training_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f" do_classifier_free_guidance = {do_classifier_free_guidance}")
    logger.info(f" conditioning_dropout_prob = {args.conditioning_dropout_prob}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(unet):
                # Convert images to latent space
                input_image = batch["image_cond"].to(dtype=weight_dtype)
                relighting_image_group1 = batch["image_target"].to(dtype=weight_dtype)
                relighting_image_group2 = batch["image_another_target"].to(dtype=weight_dtype)
                pose = batch["T"].to(dtype=weight_dtype)
                pose = torch.cat([pose, pose], dim=0)
                input_image = torch.cat((input_image, input_image), dim=0)
                gt_image = torch.cat((relighting_image_group1, relighting_image_group2), dim=0)
                
                # environment map target
                target_envir_map_ldr_group1 = batch["envir_map_target_ldr"].to(dtype=weight_dtype)
                target_envir_map_hdr_group1 = batch["envir_map_target_hdr"].to(dtype=weight_dtype)
                target_envir_map_ldr_group2 = batch["envir_map_another_target_ldr"].to(dtype=weight_dtype)
                target_envir_map_hdr_group2 = batch["envir_map_another_target_hdr"].to(dtype=weight_dtype)
                target_envir_map_ldr = torch.cat((target_envir_map_ldr_group1, target_envir_map_ldr_group2), dim=0)
                target_envir_map_hdr = torch.cat((target_envir_map_hdr_group1, target_envir_map_hdr_group2), dim=0)
                

                # pose = torch.cat([pose, target_orientation], dim=-1)

                gt_latents = vae.encode(gt_image).latent_dist.sample().detach()
                gt_latents = gt_latents * vae.config.scaling_factor # follow zero123, only target image latent is scaled

                img_latents = vae.encode(input_image).latent_dist.mode().detach()   
                target_envir_map_ldr_latents = vae.encode(target_envir_map_ldr).latent_dist.sample().detach()
                target_envir_map_hdr_latents = vae.encode(target_envir_map_hdr).latent_dist.mode().detach()
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(gt_latents)
                bsz = gt_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gt_latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(gt_latents.to(dtype=torch.float32), noise.to(dtype=torch.float32), timesteps).to(dtype=img_latents.dtype)
                if do_classifier_free_guidance:  # support classifier-free guidance, randomly drop out 5%
                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    random_p = torch.rand(bsz, device=gt_latents.device)
                    
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)

                    img_prompt_embeds = _encode_image(image_encoder, input_image, gt_latents.device, gt_latents.dtype, False)

                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(img_prompt_embeds).detach()
                    img_prompt_embeds = torch.where(prompt_mask, null_conditioning, img_prompt_embeds)

                    prompt_embeds = img_prompt_embeds

                    # Sample masks for the input images.
                    image_mask_dtype = img_latents.dtype
                    image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    img_latents = image_mask * img_latents
                    target_envir_map_ldr_latents = image_mask * target_envir_map_ldr_latents
                    target_envir_map_hdr_latents = image_mask * target_envir_map_hdr_latents
                else:
                    # Get the image_with_pose embedding for conditioning
                    prompt_embeds = _encode_image_without_pose(image_encoder, input_image, gt_latents.device, weight_dtype, False)


                # latent_model_input = torch.cat([noisy_latents, img_latents], dim=1)
                latent_model_input = torch.cat([noisy_latents, img_latents, target_envir_map_hdr_latents, target_envir_map_ldr_latents], dim=1)

                # Predict the noise residual
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                ).sample
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(gt_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = (loss.mean([1, 2, 3])).mean()

                accelerator.backward(loss)

                optimizer.step()

                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    step_log = {}
                    
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")


                    if validation_dataloader_random_light_condition is not None and (global_step % args.validation_steps == 0 or global_step == (100 +initial_global_step)):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs, temp_log = log_validation(
                            validation_dataloader_random_light_condition,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            split='unseen_object_with_random_area_light_condition',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)
                        

                    if validation_dataloader_unseen_lighting is not None and (global_step % args.validation_steps == 0 or global_step == (100 + initial_global_step)):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs, temp_log = log_validation(
                            validation_dataloader_unseen_lighting,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            split='unseen_object_with_unseen_envir',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)
                        
                    if validation_dataloader_seen_lighting is not None and (global_step % args.validation_steps == 0 or global_step == (100 + initial_global_step)):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs, temp_log = log_validation(
                            validation_dataloader_seen_lighting,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            split='unseen_object_with_seen_envir',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)
                    if training_dataloader_unseen_lighting is not None and (global_step % args.validation_steps == 0 or global_step == (100 + initial_global_step)):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        image_logs, temp_log = log_validation(
                            training_dataloader_unseen_lighting,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            split='training_object_with_unseen_envir',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)   
                          
                            
                    if train_log_dataloader is not None and (global_step % args.validation_steps == 0 or global_step == (100 + initial_global_step)):
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        train_image_logs, temp_log = log_validation(
                            train_log_dataloader,
                            vae,
                            image_encoder,
                            feature_extractor,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            'train',
                            cur_step=global_step
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())
                        step_log.update(temp_log)
                    accelerator.log(step_log, step=global_step)
            loss_epoch += loss.detach().item()
            num_train_elems += 1
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "loss_epoch": loss_epoch/num_train_elems,
                    "epoch": epoch}
            
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        # unet.save_pretrained(args.output_dir)

        

        pipeline = Neural_Gaffer_StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            image_encoder=accelerator.unwrap_model(image_encoder),
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=noise_scheduler,
            safety_checker=None,
            torch_dtype=torch.float32,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()




if __name__ == "__main__":
    args = parse_args()
    main(args)
