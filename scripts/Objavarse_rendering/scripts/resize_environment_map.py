import os
from envmap import EnvironmentMap
from glob import glob
from tqdm import tqdm
# import imageio
import numpy as np
import pyexr



input_dir =  "/home/hj453/blender_download/light_probes_selected/val/environment_map_new_added_for_eval_rendering_2k"
output_dir = "/home/hj453/blender_download/light_probes_selected/val_resized_512/hdrmaps"

if os.path.exists(output_dir) is False:
    os.makedirs(output_dir)

# find all .exr file in input_dir

exr_dir = glob(os.path.join(input_dir, '*.exr'))
for envir_path in tqdm(exr_dir):
    saved_path = envir_path.replace(input_dir, output_dir)
    # saved_path = saved_path.replace('_4k.exr', '_2k.exr')
    # remove '[' and ']'and replace ' ' with '_'
    saved_path = saved_path.replace('[', '')
    saved_path = saved_path.replace(']', '')
    saved_path = saved_path.replace(' ', '_')

    if os.path.exists(saved_path):
        continue
    cur_envir = EnvironmentMap(envir_path, 'latlong')
    resized_envir = cur_envir.resize(256)
    # if has nan
    if np.isnan(resized_envir.data).any():
        print('NAN', saved_path)
    if np.isinf(resized_envir.data).any():
        print('INF', saved_path)
    sub_dir = os.path.dirname(saved_path)
    if os.path.exists(sub_dir) is False:
        os.makedirs(sub_dir)
    pyexr.write(saved_path, resized_envir.data, precision=pyexr.FLOAT)
    # imageio.imwrite(saved_path, resized_envir.data.astype(np.float32))

    

