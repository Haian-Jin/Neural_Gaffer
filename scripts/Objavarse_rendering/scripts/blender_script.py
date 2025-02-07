"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

"""


import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import bpy
from mathutils import Vector

from glob import glob
import traceback

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
# parsser.add_argument("--output_dir", type=str, default="{args.output_dir}/views_whole_sphere")
parser.add_argument("--output_dir", type=str, default="/home/hj453/code/zero123/objaverse-rendering/hf-objaverse-v1-last-700/views_whole_sphere")


parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--lighting_per_view", type=int, default=16)
parser.add_argument("--lighting_starting_idx", type=int, default=0)
parser.add_argument("--view_starting_idx", type=int, default=0)


parser.add_argument(
    "--test_light_dir",
    type=str,
    default="/home/hj453/Dataset",
    help="directory containing the test (novel) light probes",
)

    
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"



render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"

#  disable the animation rendering ability


# elevation, azimuth, distance
def compute_spherical(elevation, azimuth, distance):
    
    elevation = elevation / 180 * np.pi
    azimuth = azimuth / 180 * np.pi
    
    # the elevation angle is defined from XY-plane up
    z = distance * np.sin(elevation)
    x, y = distance * np.cos(elevation) * np.cos(azimuth), distance * np.cos(elevation) * np.sin(azimuth)
    
    vec = x, y, z
    return vec



def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    
    x, y, z = compute_spherical(elevation, azimuth, distance)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def set_camera_location_with_xyz(xyz):
    # from https://blender.stackexchange.com/questions/18530/
    
    x, y, z = xyz
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera



# add environment map as the lighting condition
def add_light_env(env=(1, 1, 1, 1), strength=1, rot_vec_rad=(0, 0, 0), scale=(1, 1, 1)):
    r"""Adds environment lighting.
    Args:
        env (tuple(float) or str, optional): Environment map. If tuple,
            it's RGB or RGBA, each element of which :math:`\in [0,1]`.
            Otherwise, it's the path to an image.
        strength (float, optional): Light intensity.
        rot_vec_rad (tuple(float), optional): Rotations in radians around x, y and z.
        scale (tuple(float), optional): If all changed simultaneously, then no effects.
    """

    engine = bpy.context.scene.render.engine
    assert engine == "CYCLES", "Rendering engine is not Cycles"

    if isinstance(env, str):
        bpy.data.images.load(env, check_existing=True)
        env = bpy.data.images[os.path.basename(env)]
    else:
        if len(env) == 3:
            env += (1,)
        assert len(env) == 4, "If tuple, env must be of length 3 or 4"

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    bg_node = nodes.new("ShaderNodeBackground")
    links.new(bg_node.outputs["Background"], nodes["World Output"].inputs["Surface"])

    if isinstance(env, tuple):
        # Color
        bg_node.inputs["Color"].default_value = env
        print(("Environment is pure color, " "so rotation and scale have no effect"))
    else:
        # Environment map
        texcoord_node = nodes.new("ShaderNodeTexCoord")
        env_node = nodes.new("ShaderNodeTexEnvironment")
        env_node.image = env
        mapping_node = nodes.new("ShaderNodeMapping")
        mapping_node.inputs["Rotation"].default_value = rot_vec_rad
        mapping_node.inputs["Scale"].default_value = scale
        links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        links.new(mapping_node.outputs["Vector"], env_node.inputs["Vector"])
        links.new(env_node.outputs["Color"], bg_node.inputs["Color"])

    bg_node.inputs["Strength"].default_value = strength


def remove_unwanted_objects():
    """
    Remove unwanted objects from the scene, such as lights and background plane objects.
    """
    # Remove undesired objects and existing lights
    objs = []
    for o in bpy.data.objects:
        if o.name == 'BackgroundPlane':
            objs.append(o)
        elif o.type == 'LIGHT':
            objs.append(o)
        elif o.active_material is not None:
            for node in o.active_material.node_tree.nodes:
                if node.type == 'EMISSION':
                    objs.append(o)
               
    bpy.ops.object.delete({'selected_objects': objs})



def reset_scene():
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


    scene.use_nodes = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new("CompositorNodeRLayers")

    # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = "MULTIPLY"
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = "ADD"
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])


    
    alpha_normal = nodes.new(type="CompositorNodeSetAlpha")
    links.new(bias_node.outputs[0], alpha_normal.inputs["Image"])
    links.new(render_layers.outputs["Alpha"], alpha_normal.inputs["Alpha"])
    
    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = "Normal Output"
    normal_file_output.base_path = ""
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = "PNG"    
    links.new(alpha_normal.outputs["Image"], normal_file_output.inputs[0])

    # Create albedo output nodes
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = "Albedo Output"
    albedo_file_output.base_path = ""
    # albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = "PNG"
    albedo_file_output.format.color_mode = "RGB"
    albedo_file_output.format.color_depth = "8"
    links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])
    
    scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
    # scene.view_settings.view_transform = 'Raw'
    
  
    return normal_file_output, albedo_file_output

# load the glb model
def load_object(object_path: str) -> None:
    try:
        """Loads a glb model into the scene."""
        if object_path.endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
        elif object_path.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=object_path)
        else:
            raise ValueError(f"Unsupported file type: {object_path}")
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    except:
        os.system(f'echo "{object_path}" >> {args.output_dir}/bug.txt')
    return mesh_objects

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1 * R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    
    
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    # return True

def randomize_lighting() -> None:
    light2 = bpy.data.lights["Area-random"]
    
    light2.energy = random.uniform(300, 500)
    bpy.data.objects["Area-random"].location[0] = random.uniform(-1., 1.)
    bpy.data.objects["Area-random"].location[1] = random.uniform(-1., 1.)
    bpy.data.objects["Area-random"].location[2] = random.uniform(0.5, 1.5)
    
def initialize_random_lighting() -> None:
    light_name = "Area-random"
    if light_name not in bpy.data.lights:
        # Create a new 'AREA' type light
        new_light_data = bpy.data.lights.new(name=light_name, type='AREA')
        new_light_object = bpy.data.objects.new(name=light_name, object_data=new_light_data)
        bpy.context.collection.objects.link(new_light_object)
        bpy.context.view_layer.objects.active = new_light_object
    else:
        new_light_object = bpy.data.objects[light_name]

    new_light_object.data.energy = 3000
    new_light_object.location[2] = 0.5    
    new_light_object.scale[0] = 100
    new_light_object.scale[1] = 100
    new_light_object.scale[2] = 100

def reset_area_lighting() -> None:
    light2 = bpy.data.lights["Area-random"]
    light2.energy = 0
    
    
def transform_from_xyz_to_degree(pos):
    '''
    input: pos, [3,] camera location, point to the origin
    output: elevation, azimuth, distance
    '''
    distance = np.linalg.norm(pos)
    azimuth = np.arctan2(pos[1], pos[0])
    elevation = np.arcsin(pos[2] / distance)
    
    return elevation, azimuth, distance

def save_images(object_file: str) -> None:
    try:
        """Saves rendered images of the object in the scene."""
        object_uid = os.path.basename(object_file).split(".")[0]
        # os.system(f'echo "{object_uid}: loading" >> {args.output_dir}/load.txt')

        os.makedirs(args.output_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(args.output_dir, object_uid)):
            return


        normal_file_output, albedo_file_output = reset_scene()

        # load the object
        objs = load_object(object_file)
        initialize_random_lighting()

        normalize_scene()  

        # create an empty object to track
        empty = bpy.data.objects.new("Empty", None)
        scene.collection.objects.link(empty)
        cam_constraint.target = empty

        os.makedirs(os.path.join(args.output_dir, object_uid), exist_ok=True)
        
        envir_map_list = glob(os.path.join(args.test_light_dir, "*/*.exr"), recursive=True)
        envir_map_list.sort()

        normal_file_output.mute = False
        albedo_file_output.mute = False
        bpy.context.scene.render.film_transparent = True
        # remove_unwanted_objects()
        camera_list = []
        for i in range(args.view_starting_idx, args.num_images):
            elevation = random.uniform(-60., 90.)
            azimuth = random.uniform(0., 360)
            distance = random.uniform(1.5, 2.2)
            para = [elevation, azimuth, distance]
            camera_list.append(para)
        os.system(f'echo "{object_uid}: {camera_list[0][0]}, {camera_list[0][1]}, {camera_list[0][2]}" >> {args.output_dir}/camera.txt')

        for i in range(args.view_starting_idx, args.num_images):
            normal_file_output.mute = False
            albedo_file_output.mute = True

            # first random a image with random white light
            randomize_lighting()
            add_light_env(env=(0.35, 0.35, 0.35), strength = 1.0)
            camera_location = compute_spherical(camera_list[i][0], camera_list[i][1], camera_list[i][2])
            camera = set_camera_location_with_xyz(camera_location)

            render_path = os.path.join(args.output_dir, object_uid, f"random_lighting_{i:03d}.png")
            
            scene.render.filepath = render_path

         
         
            normal_render_path = os.path.join(args.output_dir, object_uid, f"normal_{i:03d}_")
            normal_file_output.file_slots[0].path = normal_render_path
            albedo_render_path = os.path.join(args.output_dir, object_uid, f"albedo_{i:03d}_")
            albedo_file_output.file_slots[0].path = albedo_render_path
            old_view_transform = scene.view_settings.view_transform
            scene.view_settings.view_transform = 'Raw'
            bpy.ops.render.render(write_still=True, animation=False)
            scene.view_settings.view_transform = old_view_transform
            normal_file_output.mute = True
            albedo_file_output.mute = False
            bpy.ops.render.render(write_still=True, animation=False)
            RT = get_3x4_RT_matrix_from_blender(camera)
            RT_path = os.path.join(
                args.output_dir, object_uid, f"{i:03d}_RT.npy" )
            if not os.path.exists(RT_path):
                np.save(RT_path, RT)

        reset_area_lighting()
        remove_unwanted_objects()
        bpy.context.scene.render.film_transparent = False
        for i in range(args.view_starting_idx, args.num_images):
            
            camera = set_camera_location(camera_list[i][0], camera_list[i][1], camera_list[i][2])
            
            for lighting_idx in range(args.lighting_per_view):

                # random select a light probe
                selected_lighting_idx = random.randint(0, len(envir_map_list) - 1)
                envmap_path = envir_map_list[selected_lighting_idx]
                envir_map_name = os.path.basename(envmap_path)

                # render the image
                render_path = os.path.join(
                    args.output_dir, object_uid, f"{i:03d}_{(lighting_idx + args.lighting_starting_idx):03d}_{os.path.basename(envir_map_name).split('.')[0]}.png"
                )


                add_light_env(env=envmap_path, strength = 1.0)

                scene.render.filepath = render_path
                
                bpy.ops.render.render(write_still=True)

            
    except Exception as e:
        # echo error message to a file and which line of code caused the error'
        error_tracing = traceback.format_exc()
        os.system(f'echo "{object_uid}: {error_tracing}" >> {args.output_dir}/error.txt')


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        object_uid = os.path.basename(local_path).split(".")[0]
        # os.system(f'echo "{object_uid}: loading2" >> {args.output_dir}/load.txt')

        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
