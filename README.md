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
- [ ] Release the 3D relighting code




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

## 3. Inference commands 
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
The following command relights the input images stored in the `--validataion_data_dir` folder using preprocessed target lighting data from the `--lighting_dir`. The relighted images will be saved in the `--save_dir` folder. The checkpoint file for the diffusion model is located in the `--output_dir` folder.

In total, this command will generate 2,400 relighted images ($5 \text{ input images} \times 4 \text{ different lighting conditions} \times 120 \text{ rotations per lighting}$). Using a single A6000 GPU, this process takes approximately 20 minutes.
```bash
accelerate launch --main_process_port 25539 --config_file configs/1_16fp.yaml neural_gaffer_inference_real_data.py --output_dir logs/neural_gaffer_res256 --mixed_precision fp16 --resume_from_checkpoint latest --save_dir ./real_data_relighting --total_view 120 --lighting_per_view 4 --validataion_data_dir './preprocessed_data/img' --lighting_dir "./preprocessed_lighting_data"
```

#### 3.1.4 Compositing the background (optional)
```bash
python scripts/composting_background.py --mask_dir ./preprocessed_data/mask --lighting_dir ./preprocessed_lighting_data --relighting_dir ./real_data_relighting/real_img --save_dir ./real_data_relighting/video
```

### 3.2 Relighting the Ojaverse instances
Coming soon.

## 4. Acknowledgment

* This work was done while Haian Jin was a full-time student at Cornell.


* The selection of data and the generation of all figures and results was led by Cornell University.

* The codebase is built on top of the [Zero123-HF](https://github.com/kxhit/zero123-hf), a diffuser implementation of Zero123.
## 5. Citation

If you find our code helpful, please cite our paper:

```
@inproceedings{jin2024neural_gaffer,
  title     = {Neural Gaffer: Relighting Any Object via Diffusion},
  author    = {Haian Jin and Yuan Li and Fujun Luan and Yuanbo Xiangli and Sai Bi and Kai Zhang and Zexiang Xu and Jin Sun and Noah Snavely},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024},
}
```
