from .Gaffer_3D_Relighting import Gaffer3D_Relighting_Dataset
from .Gaffer_3D_TensoRF import Gaffer3D_TensoRF_Dataset

dataset_dict = {'gaffer3d_relighting': Gaffer3D_Relighting_Dataset,
                'gaffer3d_tensorf':Gaffer3D_TensoRF_Dataset,
                }