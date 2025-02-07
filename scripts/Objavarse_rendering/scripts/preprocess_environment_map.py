import multiprocessing
import pyexr
from multiprocessing import Manager, Lock, Queue
import numpy as np
import time
import torch
import cv2
import torch.nn.functional as F
from PIL import Image
import os
import argparse
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# Create a manager
manager = Manager()

# Create a queue for progress bar updates
progress_queue = Queue()

def generate_envir_map_dir(envmap_h, envmap_w):
    lat_step_size = np.pi / envmap_h
    lng_step_size = 2 * np.pi / envmap_w
    theta, phi = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h), 
                                torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w)], indexing='ij')

    sin_theta = torch.sin(torch.pi / 2 - theta)  # [envH, envW]
    light_area_weight = 4 * torch.pi * sin_theta / torch.sum(sin_theta)  # [envH, envW]
    assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
    light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]


    view_dirs = torch.stack([   torch.cos(phi) * torch.cos(theta), 
                                torch.sin(phi) * torch.cos(theta), 
                                torch.sin(theta)], dim=-1).view(-1, 3)    # [envH * envW, 3]
    light_area_weight = light_area_weight.reshape(envmap_h, envmap_w)
    
    return light_area_weight, view_dirs

def read_hdr(path):
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    try:
        with open(path, 'rb') as h:
            buffer_ = np.frombuffer(h.read(), np.uint8)
        bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error reading HDR file {path}: {e}")
        rgb = None
        return rgb
    return rgb

def get_light(hdr_rgb, incident_dir, hdr_weight=None, if_weighted=False):
    try:
        envir_map = hdr_rgb

        envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        if hdr_weight is not None:
            hdr_weight = hdr_weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        incident_dir = incident_dir.clamp(-1, 1)
        theta = torch.arccos(incident_dir[:, 2]).reshape(-1) # top to bottom: 0 to pi
        phi = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1) # left to right: pi to -pi
        #  x = -1, y = -1 is the left-top pixel of F.grid_sample's input
        query_y = (theta / np.pi) * 2 - 1 # top to bottom: -1-> 1
        query_y = query_y.clamp(-1+10e-8, 1-10e-8)
        query_x = -phi / np.pi # left to right: -1 -> 1
        query_x = query_x.clamp(-1+10e-8, 1-10e-8)


        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0).float() # [1, 1, N, 2]
        
        if torch.abs(grid).max() > 1:
            print('grid out of range')
            os.system('echo "grid out of range"')
        if if_weighted is False or hdr_weight is None:
            light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
        else:
            weighted_envir_map = envir_map * hdr_weight
            light_rgbs = F.grid_sample(weighted_envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

            light_rgbs = light_rgbs / hdr_weight.reshape(-1, 1)
        if torch.isnan(light_rgbs).any():
            print('light_rgbs has nan')
            os.system('echo "light_rgbs has nan"')
            
    except Exception as e:
        print(f"Error in get_light: {e}")
        os.system(f'echo "Error in get_light: {e}"')
    return light_rgbs

def rotate_and_preprcess_envir_map(envir_map, aligned_RT, light_area_weight=None, view_dirs=None):
    # envir_map: [H, W, 3]
    # aligned_RT: numpy.narray [3, 4] w2c
    # the coordinate system follows Blender's convention
    try:
        # c_x_axis, c_y_axis, c_z_axis = aligned_RT[0, :3], aligned_RT[1, :3], aligned_RT[2, :3]
        env_h, env_w = envir_map.shape[0], envir_map.shape[1]
        if light_area_weight is None or view_dirs is None:
            light_area_weight, view_dirs = generate_envir_map_dir(env_h, env_w)
        axis_aligned_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) # Blender's convention
        axis_aligned_R = axis_aligned_transform @ aligned_RT[:3, :3] # [3, 3]
        view_dirs_world = view_dirs @ axis_aligned_R # [envH * envW, 3]
        # rotated_hdr_rgb = get_light(envir_map, view_dirs_world, hdr_weight=light_area_weight, if_weighted=True)
        rotated_hdr_rgb = get_light(envir_map, view_dirs_world.clamp(-1, 1))
        rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3).cpu().numpy()

        # hdr_raw
        envir_map_hdr_raw = rotated_hdr_rgb

        # ldr
        envir_map_ldr = rotated_hdr_rgb.clip(0, 1)
        envir_map_ldr = envir_map_ldr ** (1/2.2)
        # hdr
        envir_map_hdr = np.log1p(10 * rotated_hdr_rgb)
        # rescale to [0, 1]
        envir_map_hdr_rescaled = (envir_map_hdr / np.max(envir_map_hdr)).clip(0, 1)
        
        envir_map_ldr = np.uint8(envir_map_ldr * 255)
        if np.any(np.isnan(envir_map_ldr)):
            print('envir_map_ldr has nan')
            os.system('echo "envir_map_ldr has nan"')
        envir_map_ldr = Image.fromarray(envir_map_ldr)
        
        if np.any(np.isnan(envir_map_hdr)):
            print('envir_map_hdr has nan')
            os.system('echo "envir_map_hdr has nan"')

        envir_map_hdr = np.uint8(envir_map_hdr_rescaled * 255)
        envir_map_hdr = Image.fromarray(envir_map_hdr)
        # print('envir_map_hdr', envir_map_hdr.size)
    except Exception as e:
        print(f"Error in rotating and preprocessing envir_map: {e}")
        os.system(f'echo "Error in rotating and preprocessing envir_map: {e}"')

    return envir_map_ldr, envir_map_hdr, envir_map_hdr_raw




# Define multiprocessing class
class DataPreprocessingProcess(multiprocessing.Process):
    def __init__(
                 self, start_index, end_index, output,
                 input_dir,
                 input_file_paths,
                 output_dir,
                 lighting_dir = None,
                 total_view = 12,
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
        self.lighting_dir = lighting_dir
        self.total_view = total_view
        self.lighting_per_view = lighting_per_view
        self.output_hw = output_hw
        
        self.envir_map_paths = dict()
        self.light_area_weight, self.view_dirs = generate_envir_map_dir(256, 512)
        

    def run(self):
        self.preprocess_data(self.start_index, self.end_index)

    # Define data preprocessing function
    def preprocess_data(self, start_index, end_index):
        try:
            for cur_object_idx in range(start_index, end_index):
                
                filename = os.path.join(self.input_dir, self.input_file_paths[cur_object_idx])
                saved_folder_ldr = os.path.join(self.output_dir, 'LDR', self.input_file_paths[cur_object_idx])
                saved_folder_hdr = os.path.join(self.output_dir, 'HDR_rescaled',self.input_file_paths[cur_object_idx])
                saved_folder_hdr_raw = os.path.join(self.output_dir, 'HDR_raw',self.input_file_paths[cur_object_idx])
                if not os.path.exists(saved_folder_ldr):
                    os.makedirs(saved_folder_ldr)
                    if not os.path.exists(saved_folder_hdr):
                        os.makedirs(saved_folder_hdr)
                    if not os.path.exists(saved_folder_hdr_raw):
                        os.makedirs(saved_folder_hdr_raw)
                    for cur_view_idx in range(self.total_view):
                        # target_RT_path_temp = glob(os.path.join(filename, '%03d_%03d_*.png' % (cur_view_idx, 0)))[0]
                        target_RT_path = os.path.join(filename,  '%03d_RT.npy' % (cur_view_idx))
                        target_RT = np.load(target_RT_path)

                        target_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (cur_view_idx, 0)))[0]
                            # target image has a name like 000_000_cannon_2k_225.png, the envir map name is cannon_2k_225
                        target_envir_map_name = os.path.basename(target_image_path)[8:-4]
                
                        for cur_lighting_idx in range(self.lighting_per_view):
                            # print('Processing view %d, lighting %d' % (cur_view_idx, cur_lighting_idx))
                            target_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (cur_view_idx, cur_lighting_idx)))[0]
                                # target image has a name like 000_000_cannon_2k_225.png, the envir map name is cannon_2k_225
                            target_envir_map_name = os.path.basename(target_image_path)[8:-4]
                            
                            if os.path.exists(target_RT_path) and os.path.exists(target_image_path):
                                
                                target_envir_map = envir_map_hdr_values[target_envir_map_name]
                                target_envir_map_ldr, target_envir_map_hdr, envir_map_hdr_raw = rotate_and_preprcess_envir_map(target_envir_map, target_RT, light_area_weight=self.light_area_weight, view_dirs=self.view_dirs)
                                target_envir_map_ldr = target_envir_map_ldr.resize(self.output_hw, Image.BILINEAR)
                                target_envir_map_hdr = target_envir_map_hdr.resize(self.output_hw, Image.BILINEAR)
                                saved_path_ldr = os.path.join(saved_folder_ldr, os.path.basename(target_image_path)[:-4] + '.png')
                                saved_path_hdr = os.path.join(saved_folder_hdr, os.path.basename(target_image_path)[:-4] + '.png')
                                target_envir_map_ldr.save(saved_path_ldr)
                                target_envir_map_hdr.save(saved_path_hdr)
                                # save hdr raw as exr file
                                # saved_path_hdr_raw = os.path.join(saved_folder_hdr_raw, os.path.basename(target_image_path)[:-4] + '.exr')
                                # pyexr.write(saved_path_hdr_raw, envir_map_hdr_raw, precision=pyexr.FLOAT)
                            else:
                                print("Processing %s" % filename)
                                print('Target_RT_path or target_envir_map_path does not exist !!!')
                                continue
                    
                            
                    
                # Increment the counter and update the progress bar
                with lock:
                    counter.value += 1
                    progress_queue.put(1)
            
        except Exception as e:
            # print(f"Error in data preprocessing: {e}")
            os.system(f'echo "Error in data preprocessing: {e}"')
            return
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing for HDR environment map")
    # parser.add_argument("--img_dir", type=str, default="/scratch/datasets/hj453/objaverse-rendering/hf-objaverse-v1-last-700-val/views_whole_sphere", help="path to the folder containing environment maps")
    # parser.add_argument("--output_dir", type=str, default="/scratch/datasets/hj453/objaverse-rendering/hf-objaverse-v1-last-700-val/preprocessed_environment_resized", help="path to the folder containing environment maps")
    # parser.add_argument("--lighting_dir", type=str, default="/home/hj453/blender_download/light_probes_selected_exr", help="path to the folder containing environment maps")


    parser.add_argument("--img_dir", type=str, default="/scratch/datasets/hj453/objaverse-rendering/filtered_V2/rendered_images_resized", help="path to the folder containing environment maps")
    parser.add_argument("--output_dir", type=str, default="/scratch/datasets/hj453/objaverse-rendering/filtered_V2/preprocessed_environment_resized_new2", help="path to the folder containing environment maps")
    parser.add_argument("--lighting_dir", type=str, default="/scratch/datasets/hj453/objaverse-rendering/EXR_Env_Map_all_rescaled_rotated_flipped_256/", help="path to the folder containing environment maps")


    parser.add_argument("--num_workers", type=int, default=32, help="number of workers for multiprocessing")
    parser.add_argument("--total_view", type=int, default=12)
    parser.add_argument("--lighting_per_view", type=int, default=16)
    
    img_paths = []
    args = parser.parse_args()
    # include all folders
    for folder in os.listdir(args.img_dir):
        if os.path.isdir(os.path.join(args.img_dir, folder)):
            img_paths.append(folder)
    
    
    # Define the number of processes
    num_processes = args.num_workers

    # Calculate the data range for each process
    data_size = len(img_paths)
    chunk_size = data_size // num_processes
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes - 1)]
    ranges.append(((num_processes - 1) * chunk_size, data_size))
    
    envir_map_paths = dict()

    cur_envir_map_paths = glob(os.path.join(args.lighting_dir, '*/*.exr'))
    for envir_map_path in cur_envir_map_paths:
        envir_map_name = os.path.basename(envir_map_path)[:-4]
        if envir_map_name not in envir_map_paths:
            envir_map_paths[envir_map_name] = envir_map_path
    envir_map_hdr_values = dict()
    for envir_map_name, envir_map_path in tqdm(envir_map_paths.items()):

        target_envir_map = read_hdr(envir_map_path)
        target_envir_map = torch.from_numpy(target_envir_map)
        envir_map_hdr_values[envir_map_name] = target_envir_map
    # import ipdb; ipdb.set_trace()
    # Create a shared counter
    counter = manager.Value('i', 0)

    # Create a lock
    lock = Lock()

    # Create a progress bar
    pbar = tqdm(total=data_size)

    
    # Create a process pool and an output queue
    output_queue = multiprocessing.Queue()
    processes = [   
                    DataPreprocessingProcess(  
                                            start, 
                                            end, 
                                            None, 
                                            input_dir=args.img_dir, 
                                            input_file_paths=img_paths, 
                                            output_dir=args.output_dir, 
                                            lighting_dir=args.lighting_dir, 
                                            lighting_per_view=args.lighting_per_view,
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