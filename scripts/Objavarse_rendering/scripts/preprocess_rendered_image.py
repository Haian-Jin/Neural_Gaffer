# fix the previous performance drop

import multiprocessing
from multiprocessing import Manager, Lock, Queue
import time
from PIL import Image
import os
import argparse
from glob import glob
# from envmap import EnvironmentMap
from tqdm import tqdm
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import random

# Create a manager
manager = Manager()

# Create a queue for progress bar updates
progress_queue = Queue()

def safe_l2_normalize_numpy(x, dim=-1, eps=1e-6):
    return x / np.linalg.norm(x, axis=dim, keepdims=True).clip(eps, None)

def get_cond_normals_map(normals_path, RT):
    normals_map = plt.imread(normals_path,)
    normals = 2 * normals_map[..., :3] - 1
    alpha_mask = np.array(normals_map)[..., 3]
    normalized_normals = safe_l2_normalize_numpy(normals, dim=-1)
    normals = np.dot(RT[:3, :3], normalized_normals.reshape(-1, 3).T).T
    normals = normals.reshape((*normalized_normals.shape[:-1], 3))
    # alpha blending with white background
    normals_map_new = ((normals + 1) / 2)
    normals_map_new = normals_map_new * alpha_mask[..., None] + (1 - alpha_mask[..., None])
    img = Image.fromarray(np.uint8(normals_map_new * 255.))
    return img

# Define multiprocessing class
class DataPreprocessingProcess(multiprocessing.Process):
    def __init__(
                 self, start_index, end_index, output,
                 input_dir,
                 input_file_paths,
                 output_dir,
                 starting_view = 0,
                 starting_lighting = 0,
                 total_view = 16,
                 lighting_per_view = 8,
                 output_hw = (256, 256),
                 ):
        super(DataPreprocessingProcess, self).__init__()
        self.start_index = start_index
        self.end_index = end_index
        self.output = output
        self.input_dir = input_dir
        self.input_file_paths = input_file_paths
        self.output_dir = output_dir
        self.total_view = total_view
        self.lighting_per_view = lighting_per_view
        self.output_hw = output_hw
        self.starting_view = starting_view
        self.starting_lighting = starting_lighting

    def run(self):
        self.preprocess_data(self.start_index, self.end_index)

    # Define data preprocessing function
    def preprocess_data(self, start_index, end_index):
        for cur_object_idx in range(start_index, end_index):

            filename = os.path.join(self.input_dir, self.input_file_paths[cur_object_idx])
            saved_folder = os.path.join(self.output_dir, self.input_file_paths[cur_object_idx])
            # if not os.path.exists(saved_folder) or len(os.listdir(saved_folder)) != 108:
            # if not os.path.exists(saved_folder):
            os.makedirs(saved_folder, exist_ok=True)
            if len(os.listdir(saved_folder)) != (192+4*9):

                for cur_view_idx in range(self.starting_view, self.total_view):

                    # normals_image_path = os.path.join(filename, 'normal_%03d_0001.png' % cur_view_idx)
                    # target_RT_path = os.path.join(filename, '%03d_RT.npy' % (cur_view_idx))
                    # target_RT = np.load(target_RT_path) # w2c
                    # camera_normals_map = get_cond_normals_map(normals_image_path, target_RT)
                    # camera_normals_map = camera_normals_map.resize(self.output_hw, Image.Resampling.BILINEAR)
                    # saved_path = os.path.join(saved_folder, '%03d_normals.png' % (cur_view_idx))
                    # camera_normals_map.save(saved_path)

                    alpha_image_path = os.path.join(filename, '%03d_alpha.png' % cur_view_idx)
                    random_lighting_image = plt.imread(alpha_image_path)

                    target_mask = random_lighting_image[..., -1] # [H, W]


                    for cur_lighting_idx in range(self.starting_lighting,  self.lighting_per_view):
                        target_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (cur_view_idx, cur_lighting_idx)))[0]

                        if os.path.exists(target_image_path):

                            img = plt.imread(target_image_path)
                            img = img[:, :, :3] * target_mask[:, :, None] + np.array([1., 1., 1.]) * (1 - target_mask[:, :, None])
                            img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))        
                            
                            img = img.resize(self.output_hw, Image.Resampling.BILINEAR)
                            saved_path = os.path.join(saved_folder, os.path.basename(target_image_path))
                            img.save(saved_path)

                        else:
                            print("Processing %s" % filename)
                            print('Target_RT_path or target_envir_map_path does not exist !!!')
                            continue
            

            with lock:
                counter.value += 1
                progress_queue.put(1)
             

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing for HDR environment map")
    # parser.add_argument("--img_dir", type=str, default="/scratch/datasets/hj453/objaverse-rendering/hf-objaverse-v1-50000/views_whole_sphere", help="path to the folder containing environment maps")
    # parser.add_argument("--output_dir", type=str, default="/scratch/datasets/hj453/objaverse-rendering/hf-objaverse-v1-50000/preprocessed_environment_resized", help="path to the folder containing environment maps")
    # parser.add_argument("--lighting_dir", type=str, default="/share/phoenix/nfs05/S8/hj453/Env_map_exr_256", help="path to the folder containing environment maps")
    
    # parser.add_argument("--img_dir", type=str, default="/scratch/datasets/hj453/objaverse-rendering/hf-objaverse-v1-last-700-val/views_whole_sphere", help="path to the folder containing environment maps")
    # parser.add_argument("--output_dir", type=str, default="/scratch/datasets/hj453/objaverse-rendering/hf-objaverse-v1-last-700-val/preprocessed_environment_resized", help="path to the folder containing environment maps")
    # parser.add_argument("--lighting_dir", type=str, default="/home/hj453/blender_download/light_probes_selected_exr", help="path to the folder containing environment maps")



    parser.add_argument("--img_dir", type=str, default="/share/phoenix/nfs02/S6/localdisk/hj453/zero123/objaverse-rendering/filtered_V2_rendering/views_whole_sphere/", help="path to the folder containing environment maps")
    parser.add_argument("--output_dir", type=str, default="/home/hj453/code/zero123-hf/temp_val_result/preprocessed_environment_resized/preprocessed_envir_map_resized", help="path to the folder containing environment maps")
    parser.add_argument("--input_json", type=str, help="input json file specified the object id")

    parser.add_argument("--num_workers", type=int, default=12, help="number of workers for multiprocessing")
    parser.add_argument("--total_view", type=int, default=12)
    parser.add_argument("--lighting_per_view", type=int, default=16)
    parser.add_argument("--starting_view", type=int, default=0)
    parser.add_argument("--starting_lighting", type=int, default=0)
    
    img_paths = []
    args = parser.parse_args()
    
    to_preocess_obj_ids = json.load(open(args.input_json, 'r'))
    to_preocess_obj_ids = to_preocess_obj_ids.keys()
    for obj_id in to_preocess_obj_ids:
        img_paths.append(obj_id)
    
    random.shuffle(img_paths)

    
    # Define the number of processes
    num_processes = args.num_workers

    # Calculate the data range for each process
    data_size = len(img_paths)
    chunk_size = data_size // num_processes
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes - 1)]
    ranges.append(((num_processes - 1) * chunk_size, data_size))
    
    
    # Create a shared counter
    counter = manager.Value('i', 0)

    # Create a lock
    lock = Lock()

    # Create a progress bar
    pbar = tqdm(total=data_size)

    # temp = DataPreprocessingProcess(  
    #                                         0, 
    #                                         data_size, 
    #                                         None, 
    #                                         input_dir=args.img_dir, 
    #                                         input_file_paths=img_paths, 
    #                                         output_dir=args.output_dir, 
    #                                         lighting_per_view=args.lighting_per_view,
    #                                         starting_view=args.starting_view,
    #                                         starting_lighting=args.starting_lighting,
    #                                         total_view=args.total_view                        
    #                                     ) 
    # temp.run()


    # Create a process pool and an output queue
    output_queue = multiprocessing.Queue()
    processes = [   DataPreprocessingProcess(  
                                            start, 
                                            end, 
                                            None, 
                                            input_dir=args.img_dir, 
                                            input_file_paths=img_paths, 
                                            output_dir=args.output_dir, 
                                            lighting_per_view=args.lighting_per_view,
                                            starting_view=args.starting_view,
                                            starting_lighting=args.starting_lighting,
                                            total_view=args.total_view                        
                                        ) 
                    for (start, end) in ranges
                ]

    # Start processes
    start_time = time.time()
    for process in processes:
        process.start()

    # Update the progress bar based on the updates in the queue
    while True:
        pbar.update(progress_queue.get())
        with lock:
            if counter.value == data_size:
                break

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Close the progress bar
    pbar.close()

    # Collect processed data
    processed_data = []
    while not output_queue.empty():
        processed_data.extend(output_queue.get())

    end_time = time.time()
    print(f"Multiprocessing data preprocessing completed, time taken: {end_time - start_time} seconds")