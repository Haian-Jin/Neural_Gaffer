
import os
import math
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import sys
from glob import glob 
import torch.nn.functional as F

from tqdm import tqdm
import os
class NeuralGafferTrainingDataLoader():
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
        
        image_transforms = [torchvision.transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])]
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
     
        
    def train_dataloader(self):
        dataset = NeuralGafferTrainingData(lighting_dir=self.lighting_dir_train, 
                                        img_dir=self.img_dir_train,
                                        total_view=self.total_view, lighting_per_view=self.lighting_per_view,
                                        validation=False,
                                        relighting_only=True,
                                        image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
                             # sampler=sampler)

    def val_dataloader(self):
        dataset = NeuralGafferTrainingData(lighting_dir=self.lighting_dir_val, 
                                img_dir=self.img_dir_val,
                                total_view=self.total_view, lighting_per_view=4,
                                validation=True,
                                relighting_only=True,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class NeuralGafferTrainingData(Dataset):
    def __init__(self,
                 lighting_dir, 
                 img_dir,
                 image_transforms=None,
                 total_view=12,
                 lighting_per_view=4,
                 validation=False,
                 relighting_only=False,
                 image_preprocessed = False,
                 dataset_type=None
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.img_dir = img_dir
        
        self.total_view = total_view
        self.lighting_per_view = lighting_per_view
        self.tform = image_transforms
        self.validation = validation
        self.relighting_only = relighting_only
        self.image_preprocessed = image_preprocessed

        self.preprocessed_lighting_dir = lighting_dir
        self.dataset_type = dataset_type

        # if rank == 0:
        # total_objects = len(self.paths)
        if self.validation:
            if self.dataset_type == 'training_object_with_unseen_envir':
                # self.paths = self.paths[math.floor(total_objects / 100. * 99.):math.floor(total_objects / 100. * 99.5)]
                self.path_file_name = 'val_seen_object_list.json'
            else:
                # unseen_object_with_unseen_envir or unseen_object_with_seen_envir or unseen_object_with_random_area_light_condition
                # self.paths = self.paths[math.floor(total_objects / 100. * 99.5):]  # used last 0.5% as validation
                self.path_file_name = 'val_unseen_object_list.json'
            self.get_object_id() # assign value to self.paths
            if torch.distributed.get_rank() == 0:
                print(f'========== view of validation dataset ({self.dataset_type}): {len(self.paths)} ==========' )
        else:
            # training_object_with_seen_envir
            # self.paths = self.paths[:math.floor(total_objects / 100. * 99.5)]  # used first 99.5% as training
            self.path_file_name = 'training_object_list.json'
            self.get_object_id() # assign value to self.paths

            if torch.distributed.get_rank() == 0:
                print(f'========== view of training dataset ({self.dataset_type}): {len(self.paths)} ==========')
        
        self.light_area_weight, self.view_dirs = None, None

        self.tform = image_transforms        
  


        

    def __len__(self):
        return len(self.paths)

    def get_object_id(self):
        self.paths = []
        if os.path.exists(os.path.join(self.img_dir, self.path_file_name)):
            with open(os.path.join(self.img_dir, self.path_file_name)) as f:
                self.paths = json.load(f)
        else:
            for object_id in os.listdir(self.img_dir):
                self.paths.append(object_id)



    def cartesian_to_spherical(self, xyz):
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
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


    def load_im(self, path, color=None):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        if color is not None:
            img[img[:, :, -1] == 0.] = color
        
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def load_im_masked(self, img_path, mask_path, color):
        '''
        replace background pixel with white color
        '''
        try:
            img = plt.imread(img_path)
        except:
            print(img_path)
            sys.exit()
            
        mask = plt.imread(mask_path)[..., -1] # [H, W]
        img = img[:, :, :3] * mask[:, :, None] + np.array(color) * (1 - mask[:, :, None])
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img
    
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
    


    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        total_lighting = self.lighting_per_view
        # index_target, index_cond = random.sample(range(total_view), 2)  # without replacement
        
        if self.validation:
            if self.relighting_only:
                index_target, index_cond = 0, 0
            else:
                index_target, index_cond = 0, 1
            lighting_idx_target, lighting_idx_cond = 0, 1
        else:
            if self.relighting_only:
                # randonly sample a view 
                sampled_view = random.sample(range(total_view), 1)[0]
                index_target, index_cond = sampled_view, sampled_view
            else:
                index_target, index_cond = random.choices(range(total_view), k=2) # it is possible to have the same index value
            lighting_idx_cond, lighting_idx_target, lighting_idx_another_target = random.sample(range(total_lighting), 3)  # without replacement, return unique values

            # 10% chance to make lighting_idx_cond (lighting condition if the input condition image) to be -1
            if random.random() < 0.1:
                lighting_idx_cond = -1

        if self.dataset_type == 'unseen_object_with_random_area_light_condition':
            lighting_idx_cond = -1
            
        object_id = self.paths[index]
        filename = os.path.join(self.img_dir, object_id)

        color = [1., 1., 1.]

        
        # try:
        # load condition image
        if lighting_idx_cond == -1:
            cond_image_path = os.path.join(filename, 'random_lighting_%03d.png' % (index_cond))
        else:
            cond_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (index_cond, lighting_idx_cond)))[0]
        cond_im = self.process_im(self.load_im(cond_image_path))                        
        # load target image
        target_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (index_target, lighting_idx_target)))[0]
        target_im = self.process_im(self.load_im(target_image_path))

        
        # target image has a name like 000_000_cannon_2k_225.png, the envir map name is cannon_2k_225
        target_envir_map_name = os.path.basename(target_image_path)[8:-4]
        target_img_file_name = os.path.basename(target_image_path)
        target_envir_map_ldr_path = os.path.join(self.preprocessed_lighting_dir, 'LDR', object_id, target_img_file_name)
        target_envir_map_hdr_normalized_path = os.path.join(self.preprocessed_lighting_dir, 'HDR_rescaled', object_id, target_img_file_name)

        # load another target image if needed
        if not self.validation:
            another_target_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (index_target, lighting_idx_another_target)))[0]
            another_target_im = self.process_im(self.load_im(another_target_image_path))
            another_target_img_file_name = os.path.basename(another_target_image_path)
            another_target_envir_map_ldr_path = os.path.join(self.preprocessed_lighting_dir, 'LDR', object_id, another_target_img_file_name)
            another_target_envir_map_hdr_normalized_path = os.path.join(self.preprocessed_lighting_dir, 'HDR_rescaled', object_id, another_target_img_file_name)

        cond_RT_path = os.path.join(filename, '%03d_RT.npy' % (index_cond))
        target_RT_path = os.path.join(filename, '%03d_RT.npy' % (index_target))

        cond_RT = np.load(cond_RT_path)
        target_RT = np.load(target_RT_path) # w2c
        
        envir_map_target_ldr = self.process_im(self.load_im(target_envir_map_ldr_path))
        envir_map_target_hdr = self.process_im(self.load_im(target_envir_map_hdr_normalized_path))
        if not self.validation:
            another_envir_map_target_ldr = self.process_im(self.load_im(another_target_envir_map_ldr_path))
            another_envir_map_target_hdr = self.process_im(self.load_im(another_target_envir_map_hdr_normalized_path))
        
        # except Exception as e:
        #     # print error
        #     print(e)
            
        #     # # very hacky solution, sorry about this
        #     print('Encounter invalid data, use a valid one instead!!!')
        #     print(filename)
            
        #     filename = os.path.join(self.img_dir, '6d905d96bb5148eaaf8c9db2397208c4')  # this one we know is valid
        #     object_id = '6d905d96bb5148eaaf8c9db2397208c4'
        #     lighting_idx_cond = 1

        #     # load condition image
        #     cond_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (index_cond, lighting_idx_cond)))[0]
        #     cond_im = self.process_im(self.load_im(cond_image_path))                        
        #     # load target image
        #     target_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (index_target, lighting_idx_target)))[0]
        #     target_im = self.process_im(self.load_im(target_image_path))


        #     target_envir_map_name = os.path.basename(target_image_path)[8:-4]
        #     target_img_file_name = os.path.basename(target_image_path)
        #     target_envir_map_ldr_path = os.path.join(self.preprocessed_lighting_dir, 'LDR', object_id, target_img_file_name)
        #     target_envir_map_hdr_normalized_path = os.path.join(self.preprocessed_lighting_dir, 'HDR_rescaled', object_id, target_img_file_name)

        #     # load another target image if needed
        #     if not self.validation:
        #         another_target_image_path = glob(os.path.join(filename, '%03d_%03d_*.png' % (index_target, lighting_idx_another_target)))[0]
        #         another_target_im = self.process_im(self.load_im(another_target_image_path))
                
        #         another_target_img_file_name = os.path.basename(another_target_image_path)
        #         another_target_envir_map_ldr_path = os.path.join(self.preprocessed_lighting_dir, 'LDR', object_id, another_target_img_file_name)
        #         another_target_envir_map_hdr_normalized_path = os.path.join(self.preprocessed_lighting_dir, 'HDR_rescaled', object_id, another_target_img_file_name)

        #     cond_RT_path = os.path.join(filename, '%03d_RT.npy' % (index_cond))
        #     target_RT_path = os.path.join(filename, '%03d_RT.npy' % (index_target))

        #     cond_RT = np.load(cond_RT_path)
        #     target_RT = np.load(target_RT_path) # w2c
            
        #     envir_map_target_ldr = self.process_im(self.load_im(target_envir_map_ldr_path))
        #     envir_map_target_hdr = self.process_im(self.load_im(target_envir_map_hdr_normalized_path))
        #     if not self.validation:
        #         another_envir_map_target_ldr = self.process_im(self.load_im(another_target_envir_map_ldr_path))
        #         another_envir_map_target_hdr = self.process_im(self.load_im(another_target_envir_map_hdr_normalized_path))
            
            
            
        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["envir_map_target_ldr"] = envir_map_target_ldr
        data["envir_map_target_hdr"] = envir_map_target_hdr
        
        if not self.validation:
            data["image_another_target"] = another_target_im
            data["envir_map_another_target_ldr"] = another_envir_map_target_ldr
            data["envir_map_another_target_hdr"] = another_envir_map_target_hdr
            
            
        data["T"], data["cond_orientation"], data["target_orientation"] = self.get_T(target_RT, cond_RT)
        data["target_envir_map_name"] = target_envir_map_name
        
        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)



# main
if __name__ == "__main__":
    # test dataloader
    dataloader = NeuralGafferTrainingDataLoader(
        lighting_dir_train=f'/scratch/datasets/hj453/objaverse-rendering/filtered_V2/preprocessed_environment_resized_new/',
        img_dir_train = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/rendered_images_resized', 
        lighting_dir_val = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/preprocessed_environment_resized_new/', 
        img_dir_val = '/scratch/datasets/hj453/objaverse-rendering/filtered_V2/rendered_images_resized', 
        batch_size=2,
        lighting_per_view=16,
        total_view=12, 
        num_workers=0)

    train_loader = dataloader.train_dataloader()
    
    for j in range(100): 
        for i, data in tqdm(enumerate(train_loader)):
            # import ipdb; ipdb.set_trace()
            pass
            
