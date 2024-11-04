import os
import imageio
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--mask_dir', type=str, default="./mask", help='Foregroud mask directory')
    parser.add_argument('--lighting_dir', type=str, default="./output", help='Preprocessed lighting data directory')
    parser.add_argument('--relighting_dir', type=str, default="./output", help='Foregroud relighting result directory')
    parser.add_argument('--save_dir', type=str, default="./output", help='Path to the output directory')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    object_list = []
    for object in os.listdir(args.relighting_dir):
        cur_object_dir = os.path.join(args.relighting_dir, object)
        if os.path.isdir(cur_object_dir):
            object_list.append(os.path.basename(cur_object_dir))

    lighting_list = []
    for lighting in os.listdir(args.lighting_dir):
        cur_lighting_dir = os.path.join(args.lighting_dir, lighting)
        if os.path.isdir(cur_lighting_dir):
            lighting_list.append(os.path.basename(cur_lighting_dir))

    for object in tqdm(object_list):
        mask_path = os.path.join(args.mask_dir, f'{object}.png')
        mask = imageio.imread(mask_path) / 255.0 #[256, 256]

        blurred_mask = cv2.GaussianBlur(mask, (3, 3), 0)
        # Apply erosion
        kernel = np.ones((3,3),np.uint8)
        eroded_mask = cv2.erode(blurred_mask, kernel, iterations = 1)
        mask = eroded_mask [..., np.newaxis]
        object_dir = os.path.join(args.relighting_dir, object)

        for lighting in lighting_list:
            
            cur_lighting_dir = os.path.join(args.lighting_dir, lighting)
            if not os.path.exists(os.path.join(args.save_dir, lighting)):
                os.mkdir(os.path.join(args.save_dir, lighting))
            action_dir = os.path.join(object_dir, 'pred_image')
            if os.path.isdir(action_dir):
                images_names = glob(os.path.join(action_dir, f'{lighting}_*.png'))
                images = []

                if len(images_names) == 0:
                    # import ipdb; ipdb.set_trace()
                    print(f'No images found for {object} {lighting}')
                    continue
                images_names.sort()
                images_path = images_names
                if not os.path.exists(os.path.join(args.save_dir, lighting, object)):
                    os.mkdir(os.path.join(args.save_dir, lighting, object))
                for idx, image in enumerate(images_path):
                    image_path = image
                    
                    cur_image = imageio.imread(image_path).astype(np.float32) / 255.0
                    bg_image_path = os.path.join(cur_lighting_dir, 'background', f'{idx}.png')
                    cur_background = imageio.imread(bg_image_path).astype(np.float32) / 255.0
                    # alpha blending
                    cur_image = cur_image * mask + cur_background * (1 - mask)
                    imageio.imsave(os.path.join(args.save_dir, lighting, object, f'{idx:03d}.png'), (cur_image * 255).astype(np.uint8))
                    
                    images.append((cur_image * 255).astype(np.uint8))
                cur_video_save_dir = os.path.join(args.save_dir, lighting, f'{object}.mp4')
                if not os.path.exists(os.path.dirname(cur_video_save_dir)):
                    os.makedirs(os.path.dirname(cur_video_save_dir))
                imageio.mimsave(cur_video_save_dir, images, fps=24, quality=8)