# Neural Gaffer: Relighting Any Object via Diffusion (NeurIPS 2024)

## [Project Page](https://neural-gaffer.github.io/) |  [Paper](https://arxiv.org/abs/2406.07520)
Neural Gaffer is an end-to-end 2D relighting diffusion model that accurately **relights any object in a single image under various lighting conditions**.
Moreover, by combining with other generative methods, our model enables many downstream 2D tasks, such as text-based relighting and object insertion. Our model can also operate as a strong relighting prior for 3D tasks, such as relighting a radiance field.

https://github.com/Haian-Jin/Neural_Gaffer/assets/79512936/bc35ad6f-134e-4b83-8b4a-4daaae85a674

## 0. TODO List
I’ll be updating the following list this week. If you have any urgent requirements, such as needing to compare this method for an upcoming submission, please contact me via my email.
- [x] Release the checkpoint and the inference script for in-the-wild single image input
- [ ] Release the inference script for Ojaverse-like instances
- [ ] Release the training dataset for the diffusion model
- [ ] Release the training code for the diffusion model
- [x] Release the 3D relighting code




## 1. Preparation 
### 1.1 Installation
```bash
conda create -n neural-gaffer python=3.9
conda activate neural-gaffer 
pip install -r requirements.txt

pip3 install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu118
```
### 1.2 Downloading the checkpoint
The checkpoint file will be saved in the `./log/neural_gaffer_res256` folder.
```bash
cd logs
wget https://huggingface.co/coast01/Neural_Gaffer/resolve/main/neural_gaffer_res256_ckpt.zip

unzip neural_gaffer_res256_ckpt.zip
cd ..
```


### 1.3 Downloading the training dataset
coming soon

## 2. Training
coming soon

## 3. Inference commands for 2D relighting 
### 3.1 Relighting in-the-wild single image input
#### 3.1.1 Image preprocessing: segment, rescale, and recenter
Put the input images under the `--img_dir` folder and run the following command to segment the foreground. The preprocessed data will be saved in `--out_dir`.
Here, we borrow code from One-2-3-45.

*Note: If you input images have masks and you don't want to do rescale and recenter, you can skip this step by manually saving the three-channel foreground and mask of each input image in the `{$out_dir}/img` and `{$out_dir}/mask` folders, respectively.*

```bash
# download the pre-trained SAM to segment the foreground
# only need to run once
cd models/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..

#################################################################

# Segment the foreground
python scripts/segment_foreground.py --img_dir ./demo --sam_ckpt ./models/checkpoints/sam_vit_h_4b8939.pth --out_dir ./preprocessed_data  --gpu_idx 0

# The preprocessed data will be saved in ./'preprocessed_data'
```
#### 3.1.2 Preprocessing the target environment maps
Place the target environment maps in the `--lighting_dir` folder, then run the following command to preprocess them. The preprocessed data will be saved in the `--output_dir `folder. Use `--frame_num` to specify the number of frames for rotating the environment maps 360 degrees along the azimuthal direction. 
```bash
python scripts/generate_bg_and_rotate_envir_map.py --lighting_dir 'demo/environment_map_sample' --output_dir './preprocessed_lighting_data' --frame_num 120
```
#### 3.1.3 Relighting
The following command relights the input images stored in the `--validation_data_dir` folder using preprocessed target lighting data from the `--lighting_dir`. The relighted images will be saved in the `--save_dir` folder. The checkpoint file for the diffusion model is located in the `--output_dir` folder.

In total, this command will generate 2,400 relighted images ($5 \text{ input images} \times 4 \text{ different lighting conditions} \times 120 \text{ rotations per lighting}$). Using a single A6000 GPU, this process takes approximately 20 minutes.
```bash
accelerate launch --main_process_port 25539 --config_file configs/1_16fp.yaml neural_gaffer_inference_real_data.py --output_dir logs/neural_gaffer_res256 --mixed_precision fp16 --resume_from_checkpoint latest --total_view 120 --lighting_per_view 4 --validation_data_dir './preprocessed_data/img' --lighting_dir "./preprocessed_lighting_data" --save_dir ./real_data_relighting 
```

#### 3.1.4 Compositing the background (optional)
```bash
python scripts/composting_background.py --mask_dir ./preprocessed_data/mask --lighting_dir ./preprocessed_lighting_data --relighting_dir ./real_data_relighting/real_img --save_dir ./real_data_relighting/video
```

### 3.2 Relighting the Ojaverse-like instances
Coming soon

## 4. Relighting 3D objects without inverse rendering
Given a radiance field of a 3D object as input, our diffusion model serves as a robust prior to directly relight the radiance field, eliminating the need for physically-based inverse rendering. By default, the resolution is set to 256×256.

#### Relighting Pipeline
- **Section 4.1**: Preparing the data required for Stage 1 of the pipeline.
- **Section 4.2**: Optimizing the radiance field using multiview inputs (we use TensoRF for this purpose).
- **Section 4.3**: Relighting the radiance field directly, leveraging the diffusion model as a data-driven prior. This process bypasses inverse rendering and incorporates both Stage 1 and Stage 2.

#### Note
* Our 3D relighting pipeline assumes the 3D radiance field as input, making the order of **4.1** and **4.2** interchangeable. Additionally, the input camera poses used in **4.1** can be replaced with arbitrary predefined camera poses. However, changing their order may require code modifications to ensure compatibility with the dataloader format.

* We have 6 objects rendered under 4 unseen lighting conditions as the testing dataset. The data are stored in the similar format as many Objaverse rendering data (such as Zero123).For each object, 100 camera poses are sampled to generate
training images and 20 camera poses are sampled for testing images. We use the images rendered under the first lighting condition for training and the images rendered under the other three lighting conditions for relighting evaluation. Here are the commands to download the data:
```bash
wget https://huggingface.co/coast01/Neural_Gaffer/resolve/main/3d_relighting_data.zip
unzip 3d_relighting_data.zip
rm 3d_relighting_data.zip
```

### 4.1 Prepare the data used in stage 1

The following command relights the input images stored in the `--validation_data_dir` folder using HDR environment maps from the `--lighting_dir`. The relighted images will be saved in the `--save_dir` folder. The checkpoint file for the diffusion model is located in the `--output_dir` folder. `--cond_lighting_index` specifies the index of the lighting used as the lighitng of the input images.
In total, this command will generate 2,888 relighted images ($6 \text{ objects} \times 4 \text{ different lighting conditions} \times 120 \text{ rotations per lighting}$). Using a single A6000 GPU, this process takes approximately 24 minutes.
```bash
accelerate launch --main_process_port 25539 --config_file configs/1_16fp.yaml neural_gaffer_inference_objaverse_3d.py --output_dir logs/neural_gaffer_res256 --mixed_precision fp16 --resume_from_checkpoint latest --total_view 120 --lighting_per_view 4 --cond_lighting_index 0 --validation_data_dir './3d_relighting_data' --lighting_dir './demo/hdrmaps_for_3d' --save_dir './prepocessed_3d_relighting_data'
```

### 4.2 Optimize a radiance field
`${to_train}` speficies the object to be optimized. `--basedir` specifies the directory to save the logs and TensoRF checkpoints. `--datadir` specifies the directory of the preprocessed data (from Sec 4.1).

Here, we use the TensoRF to optimize the radiance field. The code is built on [TensoRF](https://github.com/apchenstu/TensoRF).

```bash
cd neural_gaffer_3d_relighting && export PYTHONPATH=. 

to_train=helmet && python train.py --config configs/gaffer3d_tensorf.txt --basedir ./3d_logs/tensorf/log_${to_train} --datadir ../prepocessed_3d_relighting_data/val_unseen_relighting_only/${to_train}
```

### 4.3 Relighting the radiance field without the inverse rendering and with diffusion data prior (Stage 1 & 2)
Assuming we have 4 unseen lighting conditions, the input radiance field was rendered under lighting 0, as by default. Then the `relight_idx` can be set to be 1, 2, and 3, which correspond to the other three unseen lighting conditions. The following command relights the radiance field using the diffusion model as a prior. The relighted radiance field will be saved in the `--save_dir` folder. The checkpoint file for the input radiance field is located in the `--ckpt` folder.
```bash
to_train=helmet && relight_idx=1  && python train_relighitng_3d.py --config configs/gaffer3d_relighting.txt --ckpt ./3d_logs/tensorf/log_${to_train}/tensorf_VM/tensorf_VM.th  --basedir ./3d_logs/neural_gaffer_3d_relighting/log_${to_train} --datadir ../prepocessed_3d_relighting_data/val_unseen_relighting_only/${to_train} --to_relight_idx ${relight_idx}
```
## 5. Limitations
Given the high resource demands of data preprocessing (specifically, rotating the HDR environment
map) and model training, and considering our limited university resources, we trained the model at a
lower image resolution of 256 × 256. 

Low resolution has been our key limitation. We used VAE in our base diffusion model to encode input images into latent maps, and then directly decode them. We found that VAE struggles
to preserve identity for objects with fine details even from latent maps encoded from the input images at this resolution(256 × 256), which in turn results in many relighting failure cases at this resolution. Finetuning our model at a higher resolution will greatly help solve this issue. Changing the base diffusion model to a more powerful one, such as stable diffusion 3 or Flux, will also help.

If you found the relighting results failed to preserve the identity of the object, you can test if this issue is caused by VAE by using the following command to encode then decode the input images. If the decoded images have obvious artifacts, it indicates that the pretrained VAE we used is the main cause of the failure. 
```bash
# --input_image_path specifies the path of the input image you want to test
python scripts/diffusion_test.py --input_image_path "./demo/vae_test/lego.png"  
```
## 6. Acknowledgment

* This work was done while Haian Jin was a full-time student at Cornell.


* The selection of data and the generation of all figures and results was led by Cornell University.

* The codebase is built on top of the [Zero123-HF](https://github.com/kxhit/zero123-hf), a diffuser implementation of Zero123. Thanks for the great work!
## 7. Citation

If you find our code helpful, please cite our paper:

```
@inproceedings{jin2024neural_gaffer,
  title     = {Neural Gaffer: Relighting Any Object via Diffusion},
  author    = {Haian Jin and Yuan Li and Fujun Luan and Yuanbo Xiangli and Sai Bi and Kai Zhang and Zexiang Xu and Jin Sun and Noah Snavely},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024},
}
```
