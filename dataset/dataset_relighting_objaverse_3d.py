import os
import math
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import numpy as np
import cv2
import webdataset as wds
import matplotlib.pyplot as plt
import sys
from glob import glob 
import torch.nn.functional as F
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import imageio

from tqdm import tqdm
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

class RelightingObjaverseDataLoader():
    def __init__(self, 
                 lighting_dir_train, img_dir_train,  
                 lighting_dir_val, img_dir_val, 
                 batch_size, total_view=12, lighting_per_view=8, num_workers=4):
        # super().__init__(self, img_dir, batch_size, total_view, num_workers)
        self.lighting_dir_train = lighting_dir_train
        self.img_dir_train = img_dir_train
        
        self.lighting_dir_val = lighting_dir_val
        self.img_dir_val = img_dir_val
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view
        self.lighting_per_view = lighting_per_view
        
        image_transforms = [transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]
        self.image_transforms = transforms.Compose(image_transforms)
     
        
    def train_dataloader(self):
        dataset = RelightingObjaverseData(lighting_dir=self.lighting_dir_train, 
                                        img_dir=self.img_dir_train,
                                        total_view=self.total_view, lighting_per_view=8,
                                        validation=False,
                                        image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
                             # sampler=sampler)

    def val_dataloader(self):
        dataset = RelightingObjaverseData(lighting_dir=self.lighting_dir_val, 
                                img_dir=self.img_dir_val,
                                total_view=self.total_view, lighting_per_view=4,
                                validation=True,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class RelightingObjaverseData(Dataset):
    def __init__(self,
                 lighting_dir, 
                 img_dir,
                 image_transforms=None,
                 total_view=120,
                 lighting_per_view=4,
                 cond_lighting_index=0,
                 validation=False,
                 relighting_only=False,
                 json_file=None,
                 specific_object=None,
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.img_dir = img_dir
        self.lighting_dir = lighting_dir
        self.total_view = total_view
        self.lighting_per_view = lighting_per_view
        self.object_views = self.total_view * self.lighting_per_view
        self.tform = image_transforms
        self.validation = validation
        self.relighting_only = relighting_only
        self.json_file = json_file
        self.preprocessed_lighting_dir = lighting_dir
        self.cond_lighting_index = cond_lighting_index

        if self.json_file is not None and os.path.exists(self.json_file):
            with open(self.json_file) as f:
                object_list = json.load(f)
                self.paths = list(object_list.keys())
        else:
            self.paths = []
            print("Didn't find valid_paths.json, will scan the directory")
            # include all folders
            for folder in os.listdir(self.img_dir):
                if os.path.isdir(os.path.join(self.img_dir, folder)):
                    self.paths.append(folder)
        # reverse the order of the paths
        self.paths.sort()

        if specific_object >=0 and specific_object < len(self.paths):
            self.paths = [self.paths[specific_object]]
        self.light_area_weight, self.view_dirs = None, None

        self.tform = image_transforms        
  
        self.envir_map_paths = dict()

        self.envir_map_values = dict()
        cur_envir_map_paths = glob(os.path.join(self.lighting_dir, '*.exr'))
        for envir_map_path in tqdm(cur_envir_map_paths):
            envir_map_name = os.path.basename(envir_map_path)[:-4]
            if envir_map_name not in self.envir_map_paths:
                self.envir_map_paths[envir_map_name] = envir_map_path

    def __len__(self):
        return len(self.paths) * self.total_view * self.lighting_per_view

    def cartesian_to_spherical(self, xyz):
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down

        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        
        cond_orientation = torch.tensor([theta_cond.item(), math.sin(azimuth_cond.item()), math.cos(azimuth_cond.item())])
        target_orientation = torch.tensor([theta_target.item(), math.sin(azimuth_target.item()), math.cos(azimuth_target.item())])
        return d_T, cond_orientation, target_orientation

    
    def load_im_with_mask(self, img_path, mask, color):
        '''
        replace background pixel with white color;
        the main difference with load_im_masked is that the mask is passed as a numpy array
        '''
        try:
            img = plt.imread(img_path)
        except:
            print(img_path)
            sys.exit()
            
        img = img[:, :, :3] * mask[:, :, None] + np.array(color) * (1 - mask[:, :, None])
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img
    def safe_l2_normalize_numpy(self, x, dim=-1, eps=1e-6):
        return x / np.linalg.norm(x, axis=dim, keepdims=True).clip(eps, None)
    
    def get_cond_normals_map(self, normals_path, RT):
        normals_map = plt.imread(normals_path,)
        normals = 2 * normals_map[..., :3] - 1
        alpha_mask = np.array(normals_map)[..., 3]
        normalized_normals = self.safe_l2_normalize_numpy(normals, dim=-1)
        normals = np.dot(RT[:3, :3], normalized_normals.reshape(-1, 3).T).T
        normals = normals.reshape((*normalized_normals.shape[:-1], 3))
        # alpha blending with white background
        normals_map_new = ((normals  + 1) / 2)
        normals_map_new = normals_map_new * alpha_mask[..., None] + (1 - alpha_mask[..., None])
        img = Image.fromarray(np.uint8(normals_map_new * 255.))
        return img

    def load_envir_map(self, path):
        # envir_map = read_hdr(path) # [H, W, 3]
        envir_map = imageio.imread(path) # [H, W, 3]
        envir_map = torch.from_numpy(envir_map).float()

        return envir_map

    def generate_envir_map_dir(self, envmap_h, envmap_w):
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

    def get_light(self, hdr_rgb, incident_dir, hdr_weight=None, if_weighted=False):

        envir_map = hdr_rgb
        envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        if torch.isnan(envir_map).any():
            os.system('echo "nan in envir_map"')
        if hdr_weight is not None:
            hdr_weight = hdr_weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        incident_dir = incident_dir.clip(-1, 1)
        theta = torch.arccos(incident_dir[:, 2]).reshape(-1) - 1e-6 # top to bottom: 0 to pi
        phi = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1) # left to right: pi to -pi

        #  x = -1, y = -1 is the left-top pixel of F.grid_sample's input
        query_y = (theta / np.pi) * 2 - 1 # top to bottom: -1-> 1
        query_y = query_y.clip(-1, 1)
        query_x = - phi / np.pi # left to right: -1 -> 1
        query_x = query_x.clip(-1, 1)
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0).float() # [1, 1, 2, N]

        if if_weighted is False or hdr_weight is None:
            light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
        else:
            weighted_envir_map = envir_map * hdr_weight       
            light_rgbs = F.grid_sample(weighted_envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)

            light_rgbs = light_rgbs / hdr_weight.reshape(-1, 1)

        return light_rgbs    


    def rotate_and_preprcess_envir_map(self, envir_map, aligned_RT, rotation_idx=0):
        # envir_map: [H, W, 3]
        # aligned_RT: numpy.narray [3, 4] w2c
        # the coordinate system follows Blender's convention
        
        # c_x_axis, c_y_axis, c_z_axis = aligned_RT[0, :3], aligned_RT[1, :3], aligned_RT[2, :3]
        env_h, env_w = envir_map.shape[0], envir_map.shape[1]
        
        if self.light_area_weight is None or self.view_dirs is None:
            self.light_area_weight, self.view_dirs = self.generate_envir_map_dir(env_h, env_w)
        
        axis_aligned_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) # Blender's convention
        axis_aligned_R = axis_aligned_transform @ aligned_RT[:3, :3] # [3, 3]
        view_dirs_world = self.view_dirs @ axis_aligned_R # [envH * envW, 3]
        
        if rotation_idx != 0:
            # rotate the envir map along the z-axis
            rotated_z_radius = (-2 * np.pi * rotation_idx / self.total_view) 
            # [3, 3], left multiplied by the view_dirs_world
            rotation_maxtrix = np.array([[np.cos(rotated_z_radius), -np.sin(rotated_z_radius), 0],
                                        [np.sin(rotated_z_radius), np.cos(rotated_z_radius), 0],
                                        [0, 0, 1]])
            view_dirs_world = view_dirs_world @ rotation_maxtrix        
        
        rotated_hdr_rgb = self.get_light(envir_map, view_dirs_world)
        rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3)
        
        rotated_hdr_rgb = np.array(rotated_hdr_rgb, dtype=np.float32)
    
        # ldr
        envir_map_ldr = rotated_hdr_rgb.clip(0, 1)
        envir_map_ldr = envir_map_ldr ** (1/2.2)
        # hdr
        envir_map_hdr = np.log1p(10 * rotated_hdr_rgb)
        # rescale to [0, 1]
        envir_map_hdr = envir_map_hdr / np.max(envir_map_hdr)
        
        envir_map_ldr = Image.fromarray(np.uint8(envir_map_ldr * 255.))
        envir_map_ldr = self.process_im(envir_map_ldr)
        envir_map_hdr = Image.fromarray(np.uint8(envir_map_hdr * 255.))
        envir_map_hdr = self.process_im(envir_map_hdr)
        # print('envir_map_hdr', envir_map_hdr.size)
        
        return envir_map_ldr, envir_map_hdr
        

    def __getitem__(self, index):
        object_idx = index // (self.total_view * self.lighting_per_view)
        object_inside_idx = index % (self.total_view * self.lighting_per_view)
        data = {}
        total_view = self.total_view
        
        target_view_idx = object_inside_idx % total_view
        target_lighting_idx = object_inside_idx // total_view
        
        cond_view_idx = target_view_idx
        cond_lighting_idx = self.cond_lighting_index

        object_id = self.paths[object_idx]
        
        filename = os.path.join(self.img_dir, object_id)

        color = [1., 1., 1.]

        # load condition image
        if cond_lighting_idx < 0:
            cond_image_path = os.path.join(filename, f'random_lighting_{cond_view_idx:03d}.png')
        else:
            cond_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (cond_view_idx, cond_lighting_idx)))[0]

        cond_mask_path = os.path.join(filename, f'random_lighting_{cond_view_idx:03d}.png')
        cond_mask = plt.imread(cond_mask_path)[..., -1] # [H, W]
        cond_im_with_bg = plt.imread(cond_image_path)
        cond_im = cond_im_with_bg[:, :, :3] * cond_mask[:, :, None] + np.array(color) * (1 - cond_mask[:, :, None])
    
        cond_im = Image.fromarray(np.uint8(cond_im[:, :, :3] * 255.))
        cond_im = self.process_im(cond_im)                        
        
        # load target image
        target_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (target_view_idx, target_lighting_idx)))[0]
        target_file_name = os.path.basename(target_image_path)
        target_file_name = os.path.join(object_id, target_file_name)
        target_mask = cond_mask
     
        target_im = self.process_im(self.load_im_with_mask(target_image_path, target_mask, color))
        
        # target image has a name like 000_000_cannon_2k_225.png, the envir map name is cannon_2k_225
        target_envir_map_name = os.path.basename(target_image_path)[8:-4]
        if self.envir_map_values.get(target_envir_map_name) is None:
            target_envir_map_path = self.envir_map_paths[target_envir_map_name]
            target_envir_map = self.load_envir_map(target_envir_map_path)
            self.envir_map_values[target_envir_map_name] = target_envir_map
        else:
            target_envir_map = self.envir_map_values[target_envir_map_name]

        target_RT_path = os.path.join(filename, f'{target_view_idx:03d}_RT.npy')
        cond_RT_path = target_RT_path 

        cond_RT = np.load(cond_RT_path)
        target_RT = np.load(target_RT_path) # w2c
   
        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data['target_file_name'] = target_file_name
        data["envir_map_target_ldr"], data["envir_map_target_hdr"] = self.rotate_and_preprcess_envir_map(target_envir_map, target_RT, rotation_idx=0)
            
        data["T"], data["cond_orientation"], data["target_orientation"] = self.get_T(target_RT, cond_RT)
        data["target_envir_map_name"] = target_envir_map_name
        data["target_RT"] = target_RT
        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)



# main
if __name__ == "__main__":
    
    
    image_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),  # 256, 256
            transforms.ToTensor(), # for PIL to Tensor [0, 255] -> [0.0, 1.0] and H×W×C-> C×H×W
            transforms.Normalize([0.5], [0.5]) # x -> (x - 0.5) / 0.5 == 2 * x - 1.0; [0.0, 1.0] -> [-1.0, 1.0]
        ]
    )
    
    # test dataloader
    dataloader = RelightingObjaverseData(
        lighting_dir = '/share/phoenix/nfs06/S9/hj453/env_map/HDR_Haven_256_new', 
        img_dir = '/home/hj453/code/zero123/objaverse-rendering/rotating_camera_rendering/nerf-synthetic3',
        # img_dir = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/views_whole_sphere', 
        # img_dir = '/share/phoenix/nfs05/S8/hj453/Objaverse_rendered_resized/',    
        lighting_per_view=3,
        total_view=120,
        json_file='/holme/hj453/code/zero123/objaverse-rendering/to_validate_grouped10/objaverse_filtered_V2_all_unseen_object_selected_new2.json',
        image_transforms=image_transforms, 
        validation=True,
        relighting_only=True,
        )

    train_loader = dataloader
    
    for i, data in enumerate(train_loader):
        import ipdb; ipdb.set_trace()
        pass
    
