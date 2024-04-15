from painter_dTypo import Painter, PainterOptimizer
from losses import SDSVideoLoss, PerceptualLoss, ConformalLoss
import utils
import os
import matplotlib.pyplot as plt
import torch
import pydiffvg
from tqdm import tqdm
from pytorch_lightning import seed_everything
import argparse
import wandb
import numpy as np
from torchvision import transforms
from easydict import EasyDict as edict
import torchvision
import copy
import json

def parse_arguments():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--font', type=str, default="KaushanScript-Regular", help="font name")
    parser.add_argument('--word', type=str, default="DRAGON", help="the text to work on")
    parser.add_argument('--optimized_letter', type=str, default="R", help="the letter in the word to optimize")
    parser.add_argument("--caption", type=str, default="", help="Prompt for animation. verify first that this prompt works with the original text2vid model.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--resize_by_last_frame", type=int, default=1)

    # Diffusion related & Losses
    parser.add_argument("--model_name", type=str, default="damo-vilab/text-to-video-ms-1.7b")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--guidance_scale", type=float, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--render_size", type=int, default=320, help="should fit the default settings of the chosen video model (under 'model_name')")
    parser.add_argument("--render_size_h", type=int, default=256, help="should fit the default settings of the chosen video model (under 'model_name')")
    parser.add_argument("--render_size_w", type=int, default=256, help="should fit the default settings of the chosen video model (under 'model_name')")
    parser.add_argument("--num_frames", type=int, default=24, help="should fit the default settings of the chosen video model (under 'model_name')")
    
    # SDS relted
    parser.add_argument("--sds_timestep_low", type=int, default=50) 
    parser.add_argument("--same_noise_for_frames", action="store_true", help="sample noise for one frame and repeat across all frames")
    parser.add_argument("--augment_frames", type=bool, default=True, help="whether to randomely augment the frames to prevent adversarial results")

    parser.add_argument("--level_of_cc", type=int, default=1, help="level of control, 0 for less control, 2 for more control")
    parser.add_argument("--trainable", type=bool, default=False, help="whether to optimize the location of the points or not")
    parser.add_argument("--schedule_mode", type=str, default='cos', help="choose the mode to schedule the loss from ['exp', 'cos', 'none']")
    parser.add_argument("--schedule_base", type=float, default=1.0, help="the base for loss schedule")
    parser.add_argument("--schedule_rate", type=float, default=4.0, help="the rate for loss schedule")
    parser.add_argument("--no_decay", action="store_true", help="whether to decay loss only or not")
    parser.add_argument("--use_perceptual_loss", action="store_true", help="whether to use perceptual loss or not")
    parser.add_argument("--perceptual_weight", type=float, default=1e3, help="weight of the perceptual loss")
    parser.add_argument("--use_conformal_loss", action="store_true", help="whether to use conformal loss or not")
    parser.add_argument("--angles_w", type=float, default=1e3, help="weight of the conformal loss")
    parser.add_argument("--use_transition_loss", action="store_true", help="whether to use transition loss or not")
    parser.add_argument("--transition_weight", type=float, default=2e4, help="weight of the transition loss")
    parser.add_argument("--log_mode", action="store_true", help="whether to log loss only or not")
    parser.add_argument("--difficulty", type=str, default='easy', help="choose the difficulty of the animation from the ['easy', 'hard']")

    # Memory saving related
    parser.add_argument("--use_xformers", action="store_true", help="Enable xformers for unet")
    parser.add_argument("--del_text_encoders", action="store_true", help="delete text encoder and tokenizer after encoding the prompts")

    # Optimization related
    parser.add_argument("--anneal", action="store_true", help="Whether to optimize in coarse-to-fine manner")
    parser.add_argument("--canonical", action="store_true", help="Whether to learn a shared canonical template (base shape) for animation")
    parser.add_argument("--no_freq_encoding", action="store_true", help="Whether to not use frequency encoding")
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--optim_points", type=bool, default=True, help="whether to optimize the points (x,y) of the object or not")
    parser.add_argument("--opt_points_with_mlp", type=bool, default=True, help="whether to optimize the points with an MLP")
    parser.add_argument("--split_global_loss", type=bool, default=True, help="whether to use a different loss for the center prediction")
    parser.add_argument("--guidance_scale_global", type=float, default=40, help="SDS guidance scale for the global path")
    parser.add_argument("--lr_base_global", type=float, default=0.001, help="Base learning rate for the global path")

    # MLP architecture (points)
    parser.add_argument("--predict_global_frame_deltas", type=float, default=1, help="whether to predict a global delta per frame, the value is the weight of the output")
    parser.add_argument("--predict_only_global", action='store_true', help="whether to predict only global deltas")
    parser.add_argument("--inter_dim", type=int, default=128)
    parser.add_argument("--use_shared_backbone_for_global", action='store_true',
                        help="Whether to use the same backbone for the global prediction as for per point prediction")
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--normalize_input", type=int, default=1)
    parser.add_argument("--translation_layer_norm_weight", type=int, default=1)

    parser.add_argument("--rotation_weight", type=float, default=0.01, help="Scale factor for global transform matrix 'rotation' terms")
    parser.add_argument("--scale_weight", type=float, default=0.05, help="Scale factor for global transform matrix 'scale' terms")
    parser.add_argument("--shear_weight", type=float, default=0.1, help="Scale factor for global transform matrix 'shear' terms")
    parser.add_argument("--translation_weight", type=float, default=2, help="Scale factor for global transform matrix 'translation' terms")
    parser.add_argument("--local_translation_weight", type=float, default=1, help="Scale factor for local translation terms")
    # Learning rate related (can be simplified, taken from vectorFusion)
    parser.add_argument("--lr_different_scheduler", type=bool, default=True, help="Whether to use different learning rate scheduler")
    parser.add_argument("--lr_local", type=float, default=0.005)
    parser.add_argument("--lr_init", type=float, default=0.002)
    parser.add_argument("--lr_final", type=float, default=0.0008)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    # parser.add_argument("--lr_delay_steps", type=float, default=100)
    parser.add_argument("--const_lr", type=int, default=0)

    # Display related
    parser.add_argument("--display_iter", type=int, default=50)
    parser.add_argument("--save_vid_iter", type=int, default=100)

    # wandb
    parser.add_argument("--report_to_wandb", action='store_true')
    parser.add_argument("--wandb_user", type=str)
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--folder_as_wandb_run_name", type=bool, default=True)

    args = parser.parse_args()
    seed_everything(args.seed)

    args.lr_delay_steps = int(args.num_iter / 10)
    args.change_point = 250

    args.schedule_decay = not args.no_decay

    args.transition_weight *= 0.5

    if args.difficulty == 'hard':
        args.angles_w *= 4
        args.schedule_base = 0.0
        # args.schedule_rate *= 1.25
        args.schedule_mode = 'linear'
        args.schedule_decay = False
        args.lr_different_scheduler = False
        

    args.letter = f"{args.font}_{args.optimized_letter}_scaled"
    args.target = f"data/init/{args.letter}"    

    if not args.caption:
        args.caption = utils.get_caption(args.target)
        
    print("=" * 50)
    print("target:", args.target)
    print("caption:", args.caption)
    print("=" * 50)

    args.keep_legibility = args.use_perceptual_loss
    args.transition = args.use_transition_loss or args.use_conformal_loss
    args.freq_encoding = not args.no_freq_encoding
    if args.no_freq_encoding:
        args.anneal = False

    args.output_folder = f"{args.word}_{args.optimized_letter}_{args.font}"
    if args.folder_as_wandb_run_name:
        args.wandb_run_name = args.output_folder

    args.wandb_project_name = f"c{args.canonical}_a{args.anneal}_r{args.keep_legibility}_t{args.transition}_f{args.freq_encoding}"
    args.output_folder = "./videos/" + args.output_folder 
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(f"{args.output_folder}/svg_logs", exist_ok=True)
    os.makedirs(f"{args.output_folder}/mp4_logs", exist_ok=True)
    
    
    if args.report_to_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                    config=args, name=args.wandb_run_name, id=wandb.util.generate_id())


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    # Serialize and save the configuration
    config_path = os.path.join(args.output_folder, "config.json")
    with open(config_path, "w") as f:
        # Convert args Namespace to dictionary, excluding non-serializable entries
        args_dict = {k: v for k, v in vars(args).items() if type(v) in [str, int, float, bool, list, dict, tuple]}
        json.dump(args_dict, f, indent=4)

    print(f"Configuration saved to {config_path}")
    return args

def plot_video_seq(x_aug, orig_aug, cfg, step):
    pair_concat = torch.cat([orig_aug.squeeze(0).detach().cpu(), x_aug.squeeze(0).detach().cpu()])
    grid_img = torchvision.utils.make_grid(pair_concat, nrow=cfg.num_frames)
    plt.figure(figsize=(30,10))
    plt.imshow(grid_img.permute(1, 2, 0), vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"frames_iter{step}")
    plt.tight_layout()
    if cfg.report_to_wandb:
        wandb.log({"frames": wandb.Image(plt)}, step=step)

def init_shapes(cfg):
    svg = f'{cfg.target}.svg'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg)
    parameters = edict()
    # path points
    parameters.point = []
    for path in shapes_init:
        path.points.requires_grad = cfg.trainable
        parameters.point.append(path.points.to(cfg.device))

    return shapes_init, shape_groups_init, parameters

def get_augmentations():
    augemntations = []
    augemntations.append(transforms.RandomPerspective(
        fill=1, p=1.0, distortion_scale=0.5))
    augemntations.append(transforms.RandomResizedCrop(
        size=(256,256), scale=(0.4, 1), ratio=(1.0, 1.0)))
    augment_trans = transforms.Compose(augemntations)
    return augment_trans

def get_cos_scheduler(step, num_iter=1000, rate=5.0, base=0.0, decay=True):
    if step > (num_iter/2) and (not decay):
        return rate + base
    return (rate / 2) * (1 + np.cos((2 * np.pi * (step - (num_iter / 2))) / num_iter)) + base

def linear_scheduler(step, num_iter=1000, change_point=300, rate1=4.0, base=0.0):
    rate2 = rate1 * 2
    if step <= change_point:
        # Linearly increase from base to rate1
        return ((rate1 - base) / change_point) * step + base
    else:
        # Linearly increase from rate1 to rate2 to the end
        return ((rate2 - rate1) / (num_iter - change_point)) * (step - change_point) + rate1

def get_exp_scheduler(step, num_iter=1000, change_point=250, rate=5.0, base=0.0, decay=True):
    if step > change_point and (not decay):
        return (rate / (num_iter - change_point)) * (step - change_point) + rate + base
    return rate * np.exp(-(1/5) * ((step - change_point) / 60) ** 2) + base

if __name__ == "__main__":
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

    cfg = parse_arguments()

    # Preprocessing the font and the word
    print("preprocessing")
    utils.preprocess(cfg.font, cfg.word, cfg.optimized_letter, cfg.level_of_cc)

    # initialize shape
    print('initializing shape')
    shapes, shape_groups, parameters = init_shapes(cfg)
    h, w = cfg.render_size, cfg.render_size
    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)

    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
               torch.ones(img_init.shape[0], img_init.shape[1], 3, device=cfg.device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]


    # Everything about rasterization and curves is defined in the Painter class
    painter = Painter(cfg, cfg.target, num_frames=cfg.num_frames, device=cfg.device)
    optimizer = PainterOptimizer(cfg, painter)
    data_augs = get_augmentations()

    # Just to test that the svg and initial frames were loaded as expected
    with torch.no_grad():
        frames_tensor, frames_svg, points_init_frame = painter.render_frames_to_tensor()
    output_vid_path = f"{cfg.output_folder}/init_vid.mp4"
    utils.save_mp4_from_tensor(frames_tensor, output_vid_path)

    if cfg.report_to_wandb:
        video_to_save = frames_tensor.permute(0,3,1,2).detach().cpu().numpy()
        video_to_save = ((video_to_save / video_to_save.max()) * 255).astype(np.uint8)
        wandb.log({"video_init": wandb.Video(video_to_save, fps=8)})
                       
    sds_loss = SDSVideoLoss(cfg, cfg.device)

    # If requested, set up a loss with different params for the global path.
    # Will re-use the same text-2-video diffusion pipeline
    if cfg.predict_global_frame_deltas and cfg.split_global_loss:
        global_cfg = copy.deepcopy(cfg)
        if cfg.guidance_scale_global is not None:
            global_cfg.guidance_scale = cfg.guidance_scale_global
            global_sds_loss = SDSVideoLoss(global_cfg, global_cfg.device, reuse_pipe=True)

    if cfg.log_mode:
        cfg.use_conformal_loss = True
        cfg.use_perceptual_loss = True
        cfg.use_transition_loss = True

    if not cfg.canonical:
        cfg.perceptual_weight *= 1/10
        cfg.angles_w *= 1/10
        # cfg.transition_weight *= 1/10

    if cfg.use_conformal_loss or cfg.use_transition_loss:
        conformal_loss = ConformalLoss(parameters, cfg.device, cfg.optimized_letter, shape_groups)

    if cfg.use_perceptual_loss:
        perceptual_loss = PerceptualLoss(cfg)
        perceptual_loss.set_image_init(img_init)
    
    cfg.num_frames += 1

    orig_frames = frames_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3) # (K, 256, 256, 3) -> (1, K, 3, 256, 256)
    orig_frames = orig_frames.repeat(cfg.batch_size, 1, 1, 1, 1)

    sds_losses_and_opt_kwargs = []

    if cfg.predict_global_frame_deltas:
        sds_losses_and_opt_kwargs.append((sds_loss, {"skip_global": True}))
        sds_losses_and_opt_kwargs.append((global_sds_loss, {"skip_points": True}))
    else:
        sds_losses_and_opt_kwargs.append((sds_loss, {}))

    t_range = tqdm(range(cfg.num_iter + 1))
    for step in t_range:
        for curr_sds_loss, opt_kwargs in sds_losses_and_opt_kwargs:
            loss_kwargs = {}
            logs = {}
            optimizer.zero_grad_()

            # Render the frames (inc. network forward pass)
            vid_tensor, frames_svg, new_points = painter.render_frames_to_tensor(step=step) # (K, 256, 256, 3)
            x = vid_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)  # (K, 256, 256, 3) -> (1, K, 3, 256, 256)
            x = x.repeat(cfg.batch_size, 1, 1, 1, 1)

            # Apply augmentations if needed
            if cfg.augment_frames:
                augmented_pair = data_augs(torch.cat([x.squeeze(0), orig_frames.squeeze(0)]))
                x_aug = augmented_pair[:cfg.num_frames].unsqueeze(0)
                orig_frames_aug = augmented_pair[cfg.num_frames:].unsqueeze(0)
            else:
                x_aug = x
                orig_frames_aug = orig_frames
            
            # Compute SDS loss. Note: The returned loss value is always a placeholder "1".
            # SDS is applied by changing the backprop calculation, see SpecifyGradient in losses.py 
            if cfg.canonical:
                loss_sds = curr_sds_loss(x_aug[:, :-1, :, :, :] , **loss_kwargs)
            else:
                loss_sds = curr_sds_loss(x_aug, **loss_kwargs)
            # loss_sds = curr_sds_loss(x_aug, **loss_kwargs)
            loss = loss_sds

            if cfg.use_perceptual_loss:
                # Extract the last frame from the video tensor
                if cfg.canonical:
                    template = x[:, -1, :, :, :] 
                    p_loss = perceptual_loss(template)
                else:
                    p_loss = 0.0
                    for frame_index in range(cfg.num_frames - 1):
                        p_loss += perceptual_loss(x[:, frame_index, :, :, :])
                    p_loss = p_loss / (cfg.num_frames - 1)
                
                logs.update({f"perceptual_loss": p_loss.detach().item()})
                if not cfg.log_mode:     
                    if cfg.schedule_mode == 'exp':
                        loss += cfg.perceptual_weight * p_loss * get_exp_scheduler(step, cfg.num_iter, cfg.change_point, cfg.schedule_rate, cfg.schedule_base, cfg.schedule_decay)
                    elif cfg.schedule_mode == 'cos':
                        loss += cfg.perceptual_weight * p_loss * get_cos_scheduler(step, cfg.num_iter, cfg.schedule_rate, cfg.schedule_base, cfg.schedule_decay)
                    elif cfg.schedule_mode == 'linear':
                        loss += cfg.perceptual_weight * p_loss * linear_scheduler(step, cfg.num_iter, cfg.change_point, cfg.schedule_rate, cfg.schedule_base)
                    else:
                        loss += cfg.perceptual_weight * p_loss

            if cfg.use_conformal_loss:
                if cfg.canonical:
                    cf_loss = conformal_loss(new_points[-1])
                else:
                    cf_loss = 0.0
                    for frame_index in range(cfg.num_frames - 1):
                        cf_loss += conformal_loss(new_points[frame_index])
                    cf_loss = cf_loss / (cfg.num_frames - 1)
                logs.update({f"conformal_loss": cf_loss.detach().item()})
                if not cfg.log_mode:
                    loss += cfg.angles_w * cf_loss
            
            if cfg.use_transition_loss:
                tr_loss = 0.0
                conformal_loss.update_template(new_points[-1])
                for i in range(0, cfg.num_frames - 1):
                    frame_weight = 1 - 1 * (i / (cfg.num_frames - 1))
                    tr_loss += conformal_loss(new_points[i], new_points[i+1])
                tr_loss = tr_loss / (cfg.num_frames - 1)
                logs.update({f"transition_loss": tr_loss.detach().item()})
                if not cfg.log_mode:
                    loss += cfg.transition_weight * tr_loss

            t_range.set_postfix({'loss': loss.item()})
            loss.backward()
            optimizer.step_(**opt_kwargs)
            
            loss_suffix = "_global" if "skip_points" in opt_kwargs else ""
            logs.update({f"loss{loss_suffix}": loss.detach().item()}) 
            
        if not cfg.const_lr:
            optimizer.update_lr()

        logs.update({"lr_points": optimizer.get_lr("points"), "step": step})
        if cfg.canonical:
            logs.update({"lr_stylize": optimizer.get_lr("stylize"), "step": step})

        if cfg.report_to_wandb:
            wandb.log(logs, step=step)
    

        if step % cfg.save_vid_iter == 0:
            utils.save_mp4_from_tensor(vid_tensor[:-1, :, :, :], f"{cfg.output_folder}/mp4_logs/{step}.mp4")
            utils.save_vid_svg(frames_svg, f"{cfg.output_folder}/svg_logs", step, painter.canvas_width, painter.canvas_height)
            if cfg.resize_by_last_frame:
                last_frame_shapes=utils.compute_last_frame_size(f"{cfg.output_folder}/svg_logs",step,cfg.num_frames-1)
                for i in range(cfg.num_frames):
                    utils.combine_word(cfg.word,cfg.optimized_letter,cfg.font,f"{cfg.output_folder}/svg_logs",step,i,cfg.resize_by_last_frame,last_frame_shapes)
            else:
                for i in range(cfg.num_frames):
                    utils.combine_word(cfg.word,cfg.optimized_letter,cfg.font,f"{cfg.output_folder}/svg_logs",step,i)

            utils.save_hq_video_concate(path_to_outputs=cfg.output_folder,iter_=step)
            if cfg.report_to_wandb:
                video_to_save = vid_tensor.permute(0,3,1,2).detach().cpu().numpy()
                video_to_save = ((video_to_save / video_to_save.max()) * 255).astype(np.uint8)
                wandb.log({"video": wandb.Video(video_to_save, fps=8)}, step=step)
                plot_video_seq(x_aug, orig_frames_aug, cfg, step)
            
            if step > 0:
                painter.log_state(f"{cfg.output_folder}/models/")                
        
    if cfg.report_to_wandb:
        wandb.finish()
    
    # Saves a high quality .gif from the final SVG frames
    utils.save_hq_video(cfg.output_folder, iter_=cfg.num_iter)