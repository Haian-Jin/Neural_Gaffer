## Install required packages:

	``` bash
	cd scripts/Objavarse_rendering
	bash setup.sh
	```

install https://github.com/soravux/skylibs, for HDR environment map processing.

## Download the data:

* [Objaverse Data](https://objaverse.allenai.org/)
* Lighting Data:
	We collected environment maps from various sources. Due to IP constraints, we cannot provide the data directly. Our HDRIs sources include but is not limited to:
	* [Poly Haven](https://polyhaven.com/hdris): It has about 700+ environment maps for **both outdoor indoor scenes. (Outdoors mainly).**
	* [iHDRI-outdoor](https://www.ihdri.com/hdri-skies-outdoor/): It has about 60+ environment maps for **outdoor** scenes.
	* [iHDRI-indoor](https://www.ihdri.com/hdri-skies-indoor/): It has about 34 environment maps for **indoo**r scenes.
	* [iHDRI-roofed](https://www.ihdri.com/hdri-skies-roofed/#categories); it has about 30 environment maps for roofed scenes.
	* [HDRI-Skyies](https://hdri-skies.com/free-hdris/): It has about 54 environment maps for **outdoor** scenes.
	* [HDRMaps](https://hdrmaps.com/freebies/free-hdris/): It has 154 environment maps for **both outdoor indoor scenes. (Outdoors mainly).**

	If the resolution of HDR environment maps are too high, you can resize them to 512x256 using `scripts/Objavarse_rendering/scripts/resize_environment_map.py`
	(**Note: skylibs only supports .exr format. Please convert the environment maps to .exr format if they are not in .exr format.**)




## Rendering command:
	``` bash
	cd scripts/Objavarse_rendering

	python scripts/distribute-general-rendering.py \
		--num_gpus 6 \
		--workers_per_gpu 4 \
		--input_models_path ../../filtered_object_list/all_objaverse_filtered_data.json \
		--output_dir ${YOUR_OUTPUT_DIR} \
		--lighting_dir ${YOUR_LIGHTING_DIR}

	```
After rendering, please preprocess the rendered images and environment maps using `scripts/Objavarse_rendering/scripts/preprocess_rendered_image.py` and `scripts/Objavarse_rendering/scripts/preprocess_environment_map.py`.

## Acknowledgements:

This code is based on the [Zero123](https://github.com/cvlab-columbia/zero123/tree/main/objaverse-rendering)'s rendering code. Thanks for the sharing!
