import os

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

import webdataset as wds
import matplotlib.pyplot as plt
import sys
from glob import glob 
import torch.nn.functional as F
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import imageio
from tqdm import tqdm

class Relighting_DataLoader():
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
        dataset = Relighting_Data(lighting_dir=self.lighting_dir_train, 
                                        img_dir=self.img_dir_train,
                                        total_view=self.total_view, lighting_per_view=8,
                                        image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
                             # sampler=sampler)

    def val_dataloader(self):
        dataset = Relighting_Data(lighting_dir=self.lighting_dir_val, 
                                img_dir=self.img_dir_val,
                                total_view=self.total_view, lighting_per_view=4,
                                image_transforms=self.image_transforms)
        # sampler = DistributedSampler(dataset)
        
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class Relighting_Data(Dataset):
    def __init__(self,
                 lighting_dir, 
                 img_dir,
                 image_transforms=None,
                 total_view=120,
                 lighting_per_view=4,
                 specific_object=-1,
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
        self.preprocessed_lighting_dir = lighting_dir

        images = os.listdir(os.path.join(self.img_dir))
    
        images = [i for i in images if i.endswith('.png') or i.endswith('.jpg')]
        images.sort()

        # relighting specific single object, chosen by index
        if specific_object >=0 and specific_object < len(images):
            images = [images[specific_object]]

        print(images)
        self.paths = [os.path.join(self.img_dir, i) for i in images]
        
        
        self.light_area_weight, self.view_dirs = None, None

        self.tform = image_transforms        
  
        
        self.lighting_dir_list =[]
        
        for path in os.listdir(self.lighting_dir):
            lighting_dir = os.path.join(self.lighting_dir, path)
            
            if os.path.isdir(lighting_dir):
                self.lighting_dir_list.append(lighting_dir)
        self.lighting_dir_list.sort()
        assert len(self.lighting_dir_list) >= self.lighting_per_view, "lighting_per_view should not be larger than the number of lighting directories"
        


    def __len__(self):
        return len(self.paths) * self.total_view * self.lighting_per_view


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



    def __getitem__(self, index):
        object_idx = index // (self.total_view * self.lighting_per_view)
        object_inside_idx = index % (self.total_view * self.lighting_per_view)
        data = {}
        total_lighting_rotation_view = self.total_view
        
        target_view_idx = object_inside_idx % total_lighting_rotation_view
        target_lighting_idx = object_inside_idx // total_lighting_rotation_view
        
        filename = self.paths[object_idx]

        cond_image_path = filename
        cond_im = self.process_im(self.load_im(cond_image_path))

        target_envir_map_lighting_dir = self.lighting_dir_list[target_lighting_idx]
        lighting_name = target_envir_map_lighting_dir.split('/')[-1]

        data["envir_map_target_ldr"] = self.process_im(self.load_im(os.path.join(target_envir_map_lighting_dir, 'LDR', f'{target_view_idx}.png')))
        data["envir_map_target_hdr"] = self.process_im(self.load_im(os.path.join(target_envir_map_lighting_dir, 'HDR_normalized', f'{target_view_idx}.png')))
   
        data["image_cond"] = cond_im

        data["target_envir_map_name"] = lighting_name
        data["cond_img_name"] = os.path.basename(cond_image_path).split('.')[0]

        data['target_view_idx'] = target_view_idx
        return data 

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)



# main
if __name__ == "__main__":
    
    
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256), antialias=True),  # 256, 256
            transforms.ToTensor(), # for PIL to Tensor [0, 255] -> [0.0, 1.0] and H×W×C-> C×H×W
            transforms.Normalize([0.5], [0.5]) # x -> (x - 0.5) / 0.5 == 2 * x - 1.0; [0.0, 1.0] -> [-1.0, 1.0]
        ]
    )
    # test dataloader
    dataloader = Relighting_Data(
        lighting_dir = '/home/hj453/code/zero123-hf/preprocessed_results/seen_lighting_new2',
        img_dir = '/home/hj453/code/zero123-hf/original_real_input/real_candidate_selected_mask',
        lighting_per_view=1,
        total_view=120,
        image_transforms=image_transforms, 
        )

    train_loader = dataloader
    
    for i, data in enumerate(train_loader):
        import ipdb; ipdb.set_trace()
        pass
    
