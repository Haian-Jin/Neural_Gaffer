'''
├── ckpts # (put or link sam checkpoint here)
├── init # (input directory)
│   └──*.png or *.jpg
├── output # (output directory as default, you can alter the path by --out_dir argument)
│   ├──img
│   │   └──*.png # (256x256 image, RGB)
│   └── mask
│        └──*.png # (256x256 image, single channel, >0 means object)
├── README.md
└── run.py

# run example:
    python scripts/segment_foreground.py --img_dir /home/hj453/code/zero123-hf/real_input/real_candidate_selected --sam_ckpt ./models/checkpoints/sam_vit_h_4b8939.pth --out_dir /home/hj453/code/zero123-hf/real_input/real_candidate_selected_mask --gpu_idx 0


'''

# import onnxruntime as ort
import os
import numpy as np
import torch
import cv2
from PIL import Image
from rembg import remove
import time
import argparse
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor

def sam_init(path="ckpts/sam_vit_h_4b8939.pth", device_id=0):
    sam_checkpoint = path
    model_type = "vit_h"

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def sam_out_nosave(predictor, input_image, *bbox_sliders):
    bbox = np.array(bbox_sliders)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    # print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = (masks_bbox[-1] * 255.0).astype(np.uint8) # np.argmax(scores_bbox)
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA') 

# predict bbox of the foreground
def pred_bbox(image):
    image_nobg = remove(image.convert('RGBA'), alpha_matting=True)
    alpha = np.asarray(image_nobg)[:,:,-1]
    x_nonzero = np.nonzero(alpha.sum(axis=0))
    y_nonzero = np.nonzero(alpha.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    return x_min, y_min, x_max, y_max

# contrast correction, rescale and recenter
def image_preprocess_nosave(input_image, lower_contrast=True, rescale=True):

    image_arr = np.array(input_image)
    in_w, in_h = image_arr.shape[:2]

    if lower_contrast:
        alpha = 0.8  # Contrast control (1.0-3.0)
        beta =  0   # Brightness control (0-100)
        # Apply the contrast adjustment
        image_arr = cv2.convertScaleAbs(image_arr, alpha=alpha, beta=beta)
        image_arr[image_arr[...,-1]>200, -1] = 255

    ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    ratio = 0.75
    if rescale:
        side_len = int(max_size / ratio)
    else:
        side_len = in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len//2
    padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
    rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)

    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
    mask = rgba_arr[..., -1]
    return Image.fromarray((rgb * 255).astype(np.uint8)), Image.fromarray((mask * 255).astype(np.uint8), mode="L")


def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256, mask_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256, mask_256

def run_dir(args):
    # initialize the Segment Anything model
    predictor = sam_init(args.sam_ckpt, args.gpu_idx)
    image_names = []
    image_names_temp = os.listdir(args.img_dir)
    for name in image_names_temp:
        if name.endswith('.png') or name.endswith('.jpg'):
            image_names.append(name)
            
    bar = tqdm(image_names, desc="segmenting foreground objects")
    for name in bar:
        input_raw = Image.open(os.path.join(args.img_dir, name))
        # preprocess the input image
        input_256, mask_256 = preprocess(predictor, input_raw)

        clear_name = os.path.splitext(name)[0]
        input_256.save(os.path.join(args.out_dir, "img", clear_name+".png"))
        mask_256.save(os.path.join(args.out_dir, "mask", clear_name+".png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_dir', type=str, default="./init", help='Path to the input directory')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--sam_ckpt', type=str, default="./models/checkpoints/sam_vit_h_4b8939.pth", help='Path to SAM checkpoint')
    parser.add_argument('--out_dir', type=str, default="./output", help='Path to the output directory')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use')
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    # # Set the number of threads for ONNX Runtime
    # sess_options = ort.SessionOptions()
    # sess_options.intra_op_num_threads = args.num_threads
    # sess_options.inter_op_num_threads = args.num_threads

    # # Your existing code to create and run the session
    # session = ort.InferenceSession(args.checkpoint, sess_options)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "mask"), exist_ok=True)

    run_dir(args)