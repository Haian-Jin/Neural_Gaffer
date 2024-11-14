
import argparse
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Neural Gaffer training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="kxic/zero123-xl",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--lighting_dir",
        type=str,
        default="./preprocessed_lighting_data",
    )
    parser.add_argument(
        "--specific_object",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--cond_lighting_index",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--lighting_per_view",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--total_view",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="zero123-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )


    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale, if guidance_scale>1.0, do_classifier_free_guidance"
    )


    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )



    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",    # log_image currently only for wandb
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers."
    )

    
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='/home/hj453/code/zero123-hf/original_real_input/',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--validation_data_dir",
        type=str,
        default='./Neural_Gaffer/preprocessed_data',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    
    parser.add_argument(
        "--num_validation_batches",
        type=int,
        default=120*24*80,
        # default=24,
        help=(
            "Number of batches to use for validation. If `None`, use all batches."
        ),
    )

    parser.add_argument(
        "--compute_metrics",
        action="store_true",
        help=(
            "A parameter that controls if the metrics should be computed during validation. If `False`, the metrics will not be computed."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
