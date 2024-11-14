import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T


from .ray_utils import *

from glob import glob
class Gaffer3D_Relighting_Dataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, to_relight_idx=4):

        self.image_dir = os.path.join(datadir, 'input_image')
        self.image_relighting_dir = os.path.join(datadir, 'pred_image')
        self.raw_data_dir = os.path.join(datadir, 'target_RT')
        self.to_relight_idx = to_relight_idx
        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (256, 256)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.define_transforms()
        self.read_meta()

        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [0.2, 5]  

        self.scene_bbox = torch.tensor([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]) * self.downsample
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample
        self.num_images = 100

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    # elevation, azimuth, distance
    def compute_spherical(self, elevation, azimuth, distance):
        
        elevation = elevation / 180 * np.pi
        azimuth = azimuth / 180 * np.pi
        
        # the elevation angle is defined from XY-plane up
        z = distance * np.sin(elevation)
        x, y = distance * np.cos(elevation) * np.cos(azimuth), distance * np.cos(elevation) * np.sin(azimuth)
        
        vec = x, y, z
        return vec
    
    def read_meta(self):


        meta = dict()
        meta['imw'], meta['imh'] = 256, 256
        img_wh = (256, 256)

    
        sensor_width = 32
        sensor_height = 32

        # Get camera focal length
        focal_length = 35

        # Get image resolution
        resolution_x = 256
        resolution_y = 256

        # Compute focal length in pixels
        focal_length_px_x = focal_length * resolution_x / sensor_width
        self.focal = focal_length_px_x


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(256, 256, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,256/2],[0,self.focal,256/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_rgbs_relighting = []
        self.all_masks = []
        self.all_depth = []
        self.downsample=1.0

        idxs = list( i for i in range(0, self.__len__()))
        directions = get_ray_directions(img_wh[1], img_wh[0], [self.focal, self.focal])  # [H, W, 3]
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            if self.split == 'train':
                cam_trans_path = glob(os.path.join(self.raw_data_dir, f'{i:03d}_{0:03d}_*.npy'))[0]
                
                w2c = np.load(cam_trans_path)
                w2c = np.vstack([w2c, [0, 0, 0, 1]])  # [4, 4]
                c2w = np.linalg.inv(w2c)
                pose = c2w @ self.blender2opencv
                c2w = torch.FloatTensor(pose)  # [4, 4]
                w2c = torch.linalg.inv(c2w)  # [4, 4]
                # Read ray data
                rays_o, rays_d = get_rays(directions, c2w)
                self.poses += [c2w]

                image_path = glob(os.path.join(self.image_dir, f'{i:03d}_{self.to_relight_idx:03d}_*.png'))[0]
                self.image_paths += [image_path]
                img = Image.open(image_path)
                if self.downsample!=1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3]
                self.all_rgbs += [img]

                image_relighting_path = glob(os.path.join(self.image_relighting_dir, f'{i:03d}_{self.to_relight_idx:03d}_*.png'))[0]
                img = Image.open(image_relighting_path)
                if self.downsample!=1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3]
                self.all_rgbs_relighting += [img]


                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

            elif self.split == 'relighting':
                cam_trans_path = glob(os.path.join(self.raw_data_dir, f'{i+100:03d}_{self.to_relight_idx:03d}_*.npy'))[0]
                
                w2c = np.load(cam_trans_path)
                w2c = np.vstack([w2c, [0, 0, 0, 1]])  # [4, 4]
                c2w = np.linalg.inv(w2c)
                pose = c2w @ self.blender2opencv
                c2w = torch.FloatTensor(pose)  # [4, 4]
                w2c = torch.linalg.inv(c2w)  # [4, 4]
                # Read ray data
                rays_o, rays_d = get_rays(directions, c2w)
                self.poses += [c2w]

                image_path = glob(os.path.join(self.image_dir, f'{i+100:03d}_{self.to_relight_idx:03d}_*.png'))[0]
                self.image_paths += [image_path]
                img = Image.open(image_path)
                if self.downsample!=1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3]
                self.all_rgbs += [img]

                image_relighting_path = glob(os.path.join(self.image_relighting_dir, f'{i+100:03d}_{self.to_relight_idx:03d}_*.png'))[0]
                img = Image.open(image_relighting_path)
                if self.downsample!=1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3]
                self.all_rgbs_relighting += [img]


                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            else:

                elevation = 30
                azimuth = i / len(idxs) * 360.0
                distance = 1.55
                camera_loc = self.compute_spherical(elevation, azimuth, distance)

                z_axis = np.array(camera_loc) / np.linalg.norm(camera_loc)
                # import ipdb;ipdb.set_trace()
                x_axis = np.cross(np.array([0, 0, 1]), z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)

                rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
                c2w = np.eye(4)
                c2w[:3, :3] = rotation
                c2w[:3, 3] = camera_loc
                w2c = np.linalg.inv(c2w)
                pose = c2w @ self.blender2opencv
                c2w = torch.FloatTensor(pose)  # [4, 4]
                w2c = torch.linalg.inv(c2w)  # [4, 4]

                img = np.ones((256, 256, 3), dtype=np.uint8) * 255
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
                img = img[:, :3]
                self.all_rgbs += [img]
                self.all_rgbs_relighting += [img]

                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                self.poses += [c2w]

                self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)


        self.poses = torch.stack(self.poses)


        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs_relighting = torch.cat(self.all_rgbs_relighting, 0)

        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            
            self.all_rgbs_relighting = torch.stack(self.all_rgbs_relighting, 0).reshape(-1, *self.img_wh[::-1], 3)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        if self.split == 'train':
            return 100
        elif self.split == 'relighting':
            return 20
        else:
            return 100

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample
