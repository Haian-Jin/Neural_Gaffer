import os
from tqdm.auto import tqdm
from opt import config_parser


from glob import glob
import kiui
import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys


from kiui.lpips import LPIPS
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast
class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)

@torch.no_grad()
def load_input(file_dir, to_relight_idx=3, device=torch.device("cuda")):
    # load image
    print(f'[INFO] load image from {file_dir}...')

    input_image_dir = os.path.join(file_dir, 'input_image')
    hdr_map_dir = os.path.join(file_dir, 'target_envmap_hdr')
    ldr_map_dir = os.path.join(file_dir, 'target_envmap_ldr')
    target_RT_dir = os.path.join(file_dir, 'target_RT')
    image_paths = glob(os.path.join(input_image_dir, f'*_{to_relight_idx:03d}_*.png'))
    image_paths.sort()

    image_names = [os.path.basename(image_path) for image_path in image_paths]
    image_list = []
    RT_list = []
    for idx, image_name in enumerate(image_names):
        input_image = kiui.read_image(os.path.join(input_image_dir, image_name), mode='tensor')
        hdr_map = kiui.read_image(os.path.join(hdr_map_dir, image_name), mode='tensor')
        ldr_map = kiui.read_image(os.path.join(ldr_map_dir, image_name), mode='tensor')
        input_image = input_image.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        hdr_map = hdr_map.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        ldr_map = ldr_map.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
        cur_image = [input_image, hdr_map, ldr_map]
        image_list.append(cur_image)
        RT_path = os.path.join(target_RT_dir, image_name.replace('.png', '.npy'))
        RT = np.load(RT_path)

        RT_list.append(RT)
    return image_list, RT_list

@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        print(args.ckpt)
        print('!!!!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

@torch.no_grad()
def relighting_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='relighting', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        print(args.ckpt)
        print('!!!!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)

    
    if args.render_relighting:
        os.makedirs(f'{logfolder}/imgs_test_relighting', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_relighting/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):
    args.expname = args.expname + '_with_view_dependent'
    args.Ortho_reg_weight, args.L1_reg_weight, args.TV_weight_density = 0., 0., 0.
    args.lr_init, args.lr_basis, args.lr_decay_iters = 2e-2, 1e-3, 200

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, to_relight_idx=args.to_relight_idx, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, to_relight_idx=args.to_relight_idx, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)


    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)


    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)

    tensorf.freeze_density()

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    num_images = train_dataset.num_images
    image_w, image_h = train_dataset.img_wh
    
    
    # ic(allrays.shape, allrgbs.shape, num_images, image_w, image_h)

    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)



    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    # color transfer input images
    # allrgbs, color_tf = match_colors_for_image_set(allrgbs, style_img.reshape(-1, 3).cpu())
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs_relighting 
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    
    PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all_before_optimization/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    
    for iteration in tqdm(range(2500)):
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)
   
        loss = torch.mean((rgb_map - rgb_train) ** 2)
        psnr = -10.0 * np.log(loss.item()) / np.log(10.0)

        # loss
        total_loss = loss
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if iteration % 50 == 0:
            print(f'PSNR: {psnr}')
    PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all_before_diffusion_guidance/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs_relighting 

    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    refinement_iteration = 500
    pbar = tqdm(range(refinement_iteration), miniters=args.progress_refresh_rate, file=sys.stdout)
    tensorf.turn_on_viewdep()
    from zero123_relighting_utils import RelightingDiffusion
    guidance_diffusion = RelightingDiffusion(tensorf.device, model_key="kxic/zero123-xl")
    with torch.no_grad():
        image_list, RT_list = load_input(args.datadir, to_relight_idx = args.to_relight_idx)
        guidance_diffusion.get_img_embeds(image_list)

    sample_image_num = 1
    lpips_loss_compute = LPIPS(net='vgg').to(tensorf.device)
    for iteration in pbar:
        cam_idx_list = []

        # sample multiple images without replacement
        cam_idx_list = np.random.choice(num_images, sample_image_num, replace=False)
        rays_train = allrays.view(num_images, image_h, image_w, 6)[cam_idx_list].view(-1, 6)
        optimizer.zero_grad()

        rgb_pred, alpha, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=False)

        rgb_pred = rgb_pred.view(-1, image_h, image_w, 3).permute(0, 3, 1, 2).contiguous() # (N, 3, H, W)

        # optional
        w_variance = torch.mean(torch.pow(rgb_pred[:, :, :, :-1] - rgb_pred[:, :, :, 1:], 2))
        h_variance = torch.mean(torch.pow(rgb_pred[:, :, :-1, :] - rgb_pred[:, :, 1:, :], 2))
        img_tv_loss = 1.0 * (h_variance + w_variance) / 2.0

        # SDS_loss = guidance_diffusion.train_step(rgb_pred, cam_idx_list, step_ratio=min(1, 0.9 + 0.1 * (iteration / refinement_iteration) ))
        
        # exponential
        strength = 0.6 + (np.exp(iteration) - 1) / (np.exp(refinement_iteration) - 1) * 0.35
        refined_images = guidance_diffusion.refine(rgb_pred, cam_idx_list, strength=strength).float()
        if iteration % 50 == 0:
            # save refined images
            refined_image_to_save = refined_images[0].permute(1, 2, 0).detach().cpu().numpy()
            refined_image_to_save = (refined_image_to_save * 255).astype(np.uint8)
            imageio.imwrite(f'{logfolder}/imgs_vis/refined_image_{iteration}.png', refined_image_to_save)

        lpips_loss = lpips_loss_compute(refined_images, rgb_pred).mean()
        # L1 loss
        refinement_l1_loss = torch.nn.functional.l1_loss(refined_images, rgb_pred)
        # L2 loss
        # refinement_l2_loss = torch.nn.functional.mse_loss(refined_images, rgb_pred)
        loss = img_tv_loss + 0.5 * lpips_loss + 0.5 * refinement_l1_loss
        # loss = 0.5 * lpips_loss + 0.5 * refinement_l1_loss
        loss.backward()

        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % 10 == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' refinement loss = {loss.item()}'
            )


    tensorf.save(f'{logfolder}/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device, color_tf=color_tf)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    elif args.render_relighting:
        relighting_test(args)
    else:
        reconstruction(args)

