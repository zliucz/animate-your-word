import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg

from positional_encoding import *

import utils


class Painter(torch.nn.Module):
    def __init__(self,
                 args,
                 svg_path: str,
                 num_frames: int,
                 device,
                 path_to_trained_mlp=None,
                 inference=False):
        super(Painter, self).__init__()
        self.svg_path = svg_path
        self.num_frames = num_frames
        self.device = device
        self.optim_points = args.optim_points
        self.opt_points_with_mlp = args.opt_points_with_mlp
        self.render = pydiffvg.RenderFunction.apply
        self.normalize_input = args.normalize_input
        self.canonical = args.canonical
        
        self.init_shapes()
        if self.opt_points_with_mlp:
            self.points_mlp_input_ = self.points_mlp_input.unsqueeze(0).to(device)
            frame_indices = torch.arange(1, self.num_frames + 1).unsqueeze(-1).unsqueeze(-1).repeat(1, self.points_per_frame, 1).float() / self.num_frames
            self.frame_id_ = frame_indices.view(1, -1, 1).to(device)
            if self.canonical:
                self.mlp_stylization = PointStylization(input_dim=torch.numel(self.points_mlp_input),
                                            inter_dim=args.inter_dim,
                                            embed_dim=args.embed_dim,
                                            num_points_per_frame=self.points_per_frame,
                                            num_frames=num_frames,
                                            device=device,
                                            freq_encoding=args.freq_encoding,
                                            anneal=args.anneal).to(device)
            
            self.mlp_points = PointModel(input_dim=torch.numel(self.points_mlp_input) * self.num_frames,
                                        inter_dim=args.inter_dim,
                                        embed_dim=args.embed_dim,
                                        num_points_per_frame=self.points_per_frame,
                                        num_frames=num_frames,
                                        device=device,
                                        predict_global_frame_deltas=args.predict_global_frame_deltas,
                                        predict_only_global=args.predict_only_global, 
                                        inference=inference,
                                        rotation_weight=args.rotation_weight, 
                                        scale_weight=args.scale_weight,
                                        shear_weight=args.shear_weight,
                                        translation_weight=args.translation_weight,
                                        local_weight=args.local_translation_weight,
                                        freq_encoding=args.freq_encoding,
                                        anneal=args.anneal).to(device)
            
            if path_to_trained_mlp:
                print(f"Loading MLP from {path_to_trained_mlp}")
                self.mlp_points.load_state_dict(torch.load(path_to_trained_mlp))
                self.mlp_points.eval()

            # Init the weights of LayerNorm for global translation MLP if needed.
            if args.translation_layer_norm_weight:
                self.init_translation_norm(args.translation_layer_norm_weight)

    def init_shapes(self):
        """
        Loads the svg file from svg_path and set grads to the parameters we want to optimize
        In this case, we optimize the delta from the center and the deltas from the original points
        """
        parameters = edict()
        # a list of points (x,y) ordered by shape, len = num_frames * num_shapes_per_frame 
        # each element in the list is a (num_point_in_shape, 2) tensor
        parameters.point_delta = []

        svg_cur_path = f'{self.svg_path}.svg'
        # init the canvas_width, canvas_height
        self.canvas_width, self.canvas_height, shapes_init_, shape_groups_init_ = pydiffvg.svg_to_scene(svg_cur_path)
        print("self.canvas_width, self.canvas_height", self.canvas_width, self.canvas_height)
        self.points_per_frame = 0
        for s_ in shapes_init_:
            self.points_per_frame += s_.points.shape[0]

        print(f"A single frame contains {self.points_per_frame} points")
        # save the original center
        center_, all_points = get_center_of_mass(shapes_init_)
        self.original_center = center_.clone()
        self.original_center.requires_grad = False
        self.original_center = self.original_center.to(self.device)

        # # extending the initial SVG into num_frames (default 24) frames
        canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg_cur_path)
        center_cur, all_points = get_center_of_mass(shapes_init)
        # init the learned (x,y) deltas from center
        deltas_from_center = get_deltas(all_points, center_, self.device)
        for k in range(len(shapes_init)):
            points_p = deltas_from_center[k].to(self.device)
            if self.optim_points and not self.opt_points_with_mlp:
                points_p.requires_grad = True
            parameters.point_delta.append(points_p)

        # we add the shapes to the list after we set the grads
        self.shapes = shapes_init
        self.shapes_group = shape_groups_init

        self.xy_deltas_from_center = deltas_from_center  # note that frames_xy_deltas_from_center points to parameters.point_delta so these values are being updated as well
        self.points_mlp_input = torch.cat(self.xy_deltas_from_center)
        self.parameters_ = parameters
    
    def stylize(self, t):
        prev_points = self.points_mlp_input_.clone().squeeze(0) + self.original_center
        frame_input = self.points_mlp_input_   #frame_input
        if self.normalize_input:
            frame_input = utils.normalize_tensor(frame_input)
            frame_input = (1.0 + frame_input) / 2.0
        stylization_prediction=self.mlp_stylization(frame_input, t)
        self.template = prev_points + stylization_prediction

        # update the center
        self.template_center=torch.mean(self.template.view(-1,2),dim=0).clone().detach()
        self.template_center.requires_grad = False
        self.new_deltas=self.template.view(-1,2)-self.template_center
        self.points_template = self.new_deltas.unsqueeze(0).repeat(1, self.num_frames, 1)

    def render_frames_to_tensor_mlp(self, t):
        # stylize and update
        if self.canonical:
            self.stylize(t)
            prev_points = self.new_deltas.clone().squeeze(0) + self.template_center
            frame_input = self.points_template
        else:
            prev_points = self.points_mlp_input_.clone().squeeze(0) + self.original_center
            frame_input = self.points_mlp_input_.repeat(1, self.num_frames, 1)

        frames_init, frames_svg, all_new_points = [], [], []
        frame_id = self.frame_id_
        # normalize the frame_input to be between 0 and 1
        if self.normalize_input:
            frame_input = utils.normalize_tensor(frame_input)
            frame_input = (1.0 + frame_input) / 2.0
        frame_input = torch.cat([frame_input, frame_id], dim=-1)

        delta_prediction = self.mlp_points(frame_input, t) # [num_frames * points_per_frame, 2]

        for i in range(self.num_frames + 1):
            shapes, shapes_groups = self.shapes, self.shapes_group
            new_shapes, new_shape_groups, frame_new_points = [], [], []  # for SVG frames saving
            if i == self.num_frames:
                points_cur_frame = prev_points
            else:
                if self.canonical:
                    frame_weight = 1 - 1 * (i / (self.num_frames))
                else:
                    frame_weight = 1
                start_frame_slice = i * self.points_per_frame
                # take all deltas for current frame
                point_delta_leanred_cur_frame = delta_prediction[
                                                start_frame_slice: start_frame_slice + self.points_per_frame,
                                                :]  # [64, 2] -> [points_per_frame, 2]
                points_cur_frame = prev_points + point_delta_leanred_cur_frame * frame_weight

            counter = 0
            for j in range(len(shapes)):
                # for differentiability we need to redefine and render all paths
                shape, shapes_group = shapes[j], shapes_groups[j]
                points_vars = shape.points.clone().to(self.device)
                points_vars[:, 0] = points_cur_frame[counter:counter + shape.points.shape[0], 0]
                points_vars[:, 1] = points_cur_frame[counter:counter + shape.points.shape[0], 1]

                counter += shape.points.shape[0]
                frame_new_points.append(points_vars)
                path = pydiffvg.Path(
                    num_control_points=torch.full([int(points_vars.shape[0]/3)], 2), points=points_vars,
                    stroke_width=shape.stroke_width, is_closed=shape.is_closed)
                new_shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(new_shapes) - 1]),
                    fill_color=shapes_group.fill_color,
                    stroke_color=torch.tensor([0, 0, 0, 1]))
                new_shape_groups.append(path_group)

            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes,
                                                                 new_shape_groups)
            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]
            frames_init.append(cur_im)
            frames_svg.append((new_shapes, new_shape_groups))
            all_new_points.append(frame_new_points)
        
        return torch.stack(frames_init), frames_svg, all_new_points

    def render_frames_to_tensor(self, step=0):
        return self.render_frames_to_tensor_mlp(step)
    
    def get_stylized_params(self):
        return self.mlp_stylization.get_params()
    
    def get_points_params(self):
        if self.opt_points_with_mlp:
            return self.mlp_points.get_points_params()
        return self.parameters_["point_delta"]

    def get_global_params(self):
        return self.mlp_points.get_global_params()
    
    def log_state(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        torch.save(self.mlp_points.state_dict(), f"{output_path}/model.pt")
        print(f"Model saved to {output_path}/model.pt")
        return

    def init_translation_norm(self, translation_layer_norm_weight):
        print(f"Initializing translation layerNorm to {translation_layer_norm_weight}")
        for child in self.mlp_points.frames_rigid_translation.children():
            if isinstance(child, nn.LayerNorm):
                with torch.no_grad():
                    child.weight *= translation_layer_norm_weight


class PointModel(nn.Module):
    def __init__(self, input_dim, inter_dim, embed_dim, num_points_per_frame, num_frames,
                 device, predict_global_frame_deltas, predict_only_global, inference=False, 
                 rotation_weight=1e-2, scale_weight=5e-2, shear_weight=5e-2, translation_weight=1, local_weight=1, 
                 multires=6, freq_encoding=True, anneal=False, num_iter=1000):

        super().__init__()
        self.input_dim = input_dim
        self.num_points_per_frame = num_points_per_frame
        self.num_frames = num_frames
        self.inter_dim = inter_dim
        self.embed_dim = embed_dim
        self.predict_global_frame_deltas = predict_global_frame_deltas
        self.predict_only_global = predict_only_global
        self.num_probability =  int(num_points_per_frame / 3)
        self.inference = inference
        self.freq_encoding = freq_encoding
        self.anneal = anneal
        self.num_iter = num_iter
        self.local_weight = local_weight
        self.wo_original_cordinate = False

        if self.freq_encoding:
            embed_fn, input_ch = get_embedder(multires, input_dims=3)
            self.init_dim = input_ch + self.embed_dim
            self.embed_fn = embed_fn
            if not self.wo_original_cordinate:
                self.init_dim += 3
            print("input_ch", input_ch)
            print("self.init_dim", self.init_dim)
            if self.anneal:
                self.annealer = AnnealedEmbedding(in_channels=3, N_freqs=multires, annealed_step=int(1/2 * self.num_iter), annealed_begin_step=int(1/4 * self.num_iter))
        else:
            self.init_dim = 3 + self.embed_dim
        
        self.embedding = nn.Embedding(int(input_dim / 2), self.embed_dim)
        self.inds = torch.tensor(range(int(input_dim / 2))).to(device)
        self.project_points = nn.Sequential(nn.Linear(self.init_dim, inter_dim),
                                             nn.LayerNorm(inter_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(inter_dim, inter_dim))

        if predict_global_frame_deltas:
            self.rotation_weight = rotation_weight
            self.scale_weight = scale_weight
            self.shear_weight = shear_weight
            self.translation_weight = translation_weight
            self.frames_rigid_shared = nn.Sequential(nn.Linear(int(input_dim * self.inter_dim / 2), inter_dim),
                                              nn.LayerNorm(inter_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(inter_dim, inter_dim),
                                              nn.LayerNorm(inter_dim),
                                              nn.LeakyReLU())

            self.frames_rigid_translation = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                          nn.LayerNorm(inter_dim),
                                                          nn.LeakyReLU(),
                                                          nn.Linear(inter_dim, inter_dim),
                                                          nn.LayerNorm(inter_dim),
                                                          nn.LeakyReLU(),
                                                          nn.Linear(inter_dim, inter_dim),
                                                          nn.LayerNorm(inter_dim),
                                                          nn.LeakyReLU(),
                                                          nn.Linear(inter_dim, self.num_frames * 2))

            self.frames_rigid_rotation = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                       nn.LayerNorm(inter_dim),
                                                       nn.LeakyReLU(),
                                                       nn.Linear(inter_dim, self.num_frames * 1))
            
            self.frames_rigid_shear = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                    nn.LayerNorm(inter_dim),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(inter_dim, self.num_frames * 2))

            self.frames_rigid_scale = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                    nn.LayerNorm(inter_dim),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(inter_dim, self.num_frames * 2))

            self.global_layers = nn.ModuleList([self.frames_rigid_shared, 
                                  self.frames_rigid_translation, 
                                  self.frames_rigid_rotation, 
                                  self.frames_rigid_shear, 
                                  self.frames_rigid_scale,
                                  ])
        
        self.local_model = nn.Sequential(
            nn.Linear(int(input_dim * self.inter_dim / 2), inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
            nn.Linear(inter_dim, self.input_dim))


    def get_position_encoding_representation(self, init_points, t):
        # input dim: init_points [num_frames * points_per_frame, 3]
        init_points = init_points.squeeze(0)
        if self.freq_encoding:
            emb_xy = self.embed_fn(init_points) 
            if self.anneal:
                emb_xy = self.annealer(emb_xy, t)
            if not self.wo_original_cordinate:
                emb_xy = torch.cat([init_points, emb_xy], dim=-1)
        else:
            emb_xy = init_points
        embed = self.embedding(self.inds) * math.sqrt(self.embed_dim)
        init_points_pos_enc = torch.cat([emb_xy, embed], dim=-1)
        return self.project_points(init_points_pos_enc)

    def get_frame_deltas(self, init_points, init_points_pos_enc):
        init_points = init_points[:, :, :2]
        frame_deltas = None
        if self.predict_global_frame_deltas:
            shared_params = self.frames_rigid_shared(init_points_pos_enc.flatten().unsqueeze(0))
            # calculate transform matrix parameters
            dx, dy = self.frames_rigid_translation(shared_params).reshape(self.num_frames, 2).chunk(2, axis=-1)
            dx = dx * self.translation_weight
            dy = dy * self.translation_weight

            theta = self.frames_rigid_rotation(shared_params).reshape(self.num_frames, 1) * self.rotation_weight
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            shear_x, shear_y = self.frames_rigid_shear(shared_params).reshape(self.num_frames, 2).chunk(2, axis=-1)

            shear_x = shear_x * self.shear_weight
            shear_y = shear_y * self.shear_weight

            scale_x, scale_y = self.frames_rigid_scale(shared_params).reshape(self.num_frames, 2).chunk(2, axis=-1)
            
            scale_x = torch.ones_like(dx) + scale_x * self.scale_weight
            scale_y = torch.ones_like(dy) + scale_y * self.scale_weight

            # prepare transform matrix
            l1 = torch.concat([scale_x * (cos_theta - sin_theta * shear_x), scale_y * (cos_theta * shear_y - sin_theta), dx], axis=-1)
            l2 = torch.concat([scale_x * (sin_theta + cos_theta * shear_x), scale_y * (sin_theta * shear_y + cos_theta), dy], axis=-1)
            l3 = torch.concat([torch.zeros_like(dx), torch.zeros_like(dx), torch.ones_like(dx)], axis=-1)

            transform_mat = torch.stack([l1, l2, l3], axis=1)

            transform_mat = torch.repeat_interleave(transform_mat, self.num_points_per_frame, dim=0)

            # extend points for calculation
            points_with_z = torch.concat([init_points, torch.ones_like(init_points)[:,:,0:1]], axis=-1)
            points_with_z = points_with_z.reshape(-1, 3, 1)

            # calculate new coordinates and deltas
            transformed_points = torch.matmul(transform_mat, points_with_z)[:, 0:2, :].reshape(1, -1, 2)
            frame_deltas = transformed_points - init_points
            # frame_deltas *= self.predict_global_frame_deltas

        return frame_deltas

    def forward(self, init_points, t):
        init_points_pos_enc = self.get_position_encoding_representation(init_points, t)
        frame_deltas = self.get_frame_deltas(init_points, init_points_pos_enc)
        if self.predict_only_global:
            return frame_deltas.squeeze(0)
        delta_xy = self.local_model(init_points_pos_enc.flatten().unsqueeze(0)).reshape(-1, self.num_frames * self.num_points_per_frame, 2) * self.local_weight
        if self.predict_global_frame_deltas:
            delta_xy = delta_xy + frame_deltas
        
        return delta_xy.squeeze(0)
    
    def get_shared_params(self):
        shared_params = []
        shared_params += list(self.embedding.parameters())
        shared_params += list(self.project_points.parameters())
        return shared_params
    
    def get_points_params(self):
        shared_params = self.get_shared_params()
        model_p = list(self.local_model.parameters())
        return shared_params + model_p
        
    def get_global_params(self):
        shared_params = self.get_shared_params()
        delta_p = list(self.global_layers.parameters())
        return shared_params + delta_p


class PointStylization(nn.Module):
    def __init__(self, input_dim, inter_dim, embed_dim, num_points_per_frame, num_frames, device, multires=6, freq_encoding=True, anneal=False, num_iter=1000):

        super().__init__()
        self.num_points_per_frame = num_points_per_frame
        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.embed_dim = int(embed_dim/2)
        self.freq_encoding = freq_encoding
        self.anneal = False
        self.num_iter = num_iter
        self.wo_original_cordinate = False

        if self.freq_encoding:
            embed_fn, input_ch = get_embedder(multires, input_dims=2)
            self.init_dim = input_ch + self.embed_dim
            self.embed_fn = embed_fn
            # self.init_dim = input_ch
            if not self.wo_original_cordinate:
                self.init_dim += 2
            print("input_ch", input_ch)
            print("self.init_dim", self.init_dim)
            if self.anneal:
                self.annealer = AnnealedEmbedding(in_channels=2, N_freqs=multires, annealed_step=int(1/2 * self.num_iter), annealed_begin_step=int(1/4 * self.num_iter))
        else:
            self.init_dim = 2 + self.embed_dim
        
        self.embedding = nn.Embedding(int(input_dim / 2), self.embed_dim)
        self.inds = torch.tensor(range(int(input_dim / 2))).to(device)
        self.project_points = nn.Sequential(nn.Linear(self.init_dim, inter_dim),
                                             nn.LayerNorm(inter_dim),
                                             nn.LeakyReLU())
        
        self.stylization = nn.Sequential(
            nn.Linear(int(input_dim * self.inter_dim / 2), inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
            nn.Linear(inter_dim, self.input_dim))
        
    def get_position_encoding_representation(self, init_points, t):
        # input dim: init_points [points_per_frame, 2]
        init_points = init_points.squeeze(0)
        if self.freq_encoding:
            emb_xy = self.embed_fn(init_points) 
            if self.anneal:
                emb_xy = self.annealer(emb_xy, t)
            if not self.wo_original_cordinate:
                emb_xy = torch.cat([init_points, emb_xy], dim=-1)
        else:
            emb_xy = init_points
        embed = self.embedding(self.inds) * math.sqrt(self.embed_dim) 
        init_points_pos_enc = torch.cat([emb_xy, embed], dim=-1)
        init_points_pos_enc = self.project_points(init_points_pos_enc)
        return init_points_pos_enc
    
    def forward(self, init_points, t):
        init_points_pos_enc = self.get_position_encoding_representation(init_points, t)
        delta_style_xy=self.stylization(init_points_pos_enc.flatten().unsqueeze(0)).reshape(self.num_points_per_frame, 2)
        return delta_style_xy
    
    def get_params(self):
        embedding_p = list(self.embedding.parameters())
        project_points_p = list(self.project_points.parameters())
        stylization_p = list(self.stylization.parameters())
        return embedding_p + project_points_p + stylization_p

class PainterOptimizer:
    def __init__(self, args, painter):
        self.painter = painter
        self.lr_local = args.lr_local
        self.lr_base_global = args.lr_base_global
        self.lr_init = args.lr_init
        self.lr_final = args.lr_final
        self.lr_final_stylize = (args.lr_final * 1/5)
        self.lr_delay_mult = args.lr_delay_mult
        self.lr_delay_steps = args.lr_delay_steps
        self.max_steps = args.num_iter
        self.lr_lambda = lambda step: self.learning_rate_decay(step) / self.lr_init
        self.different_scheduler = args.lr_different_scheduler
        self.stylize_lr_lambda = lambda step: self.stylize_learning_rate_decay(step) / self.lr_init
        self.optim_points = args.optim_points
        self.optim_global = args.split_global_loss
        self.canonical = args.canonical
        self.init_optimizers()

    def learning_rate_decay(self, step):
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp
       
    def stylize_learning_rate_decay(self, step):
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final_stylize) * t)
        return delay_rate * log_lerp
    
    def init_optimizers(self):
        if self.canonical:
            stylize_frame_params = self.painter.get_stylized_params()
            self.stylize_optimizer = torch.optim.Adam(stylize_frame_params, lr=self.lr_local * 1.0,
                                                            betas=(0.9, 0.9), eps=1e-6)
            if self.different_scheduler:
                self.scheduler_stylize = LambdaLR(self.stylize_optimizer, lr_lambda=self.stylize_lr_lambda, last_epoch=-1)
            else:
                self.scheduler_stylize = LambdaLR(self.stylize_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)
        
        if self.optim_global:
            global_frame_params = self.painter.get_global_params()
            self.global_delta_optimizer = torch.optim.Adam(global_frame_params, lr=self.lr_base_global,
                                                           betas=(0.9, 0.9), eps=1e-6)
            self.scheduler_global = LambdaLR(self.global_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

        if self.optim_points:
            points_delta_params = self.painter.get_points_params()
            self.points_delta_optimizer = torch.optim.Adam(points_delta_params, lr=self.lr_local,
                                                           betas=(0.9, 0.9), eps=1e-6)
            self.scheduler_points = LambdaLR(self.points_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

    def update_lr(self):
        if self.canonical:
            self.scheduler_stylize.step()
        if self.optim_global:
            self.scheduler_global.step()
        if self.optim_points:
            self.scheduler_points.step()

    def zero_grad_(self):
        if self.canonical:
            self.stylize_optimizer.zero_grad()
        if self.optim_global:
            self.global_delta_optimizer.zero_grad()
        if self.optim_points:
            self.points_delta_optimizer.zero_grad()

    def step_(self, skip_global=False, skip_points=False):
        if self.canonical:
            self.stylize_optimizer.step()
        if self.optim_global and not skip_global:
            self.global_delta_optimizer.step()
        if self.optim_points and not skip_points:
            self.points_delta_optimizer.step()

    def get_lr(self, optim="points"):
        if optim == "points" and self.optim_points:
            return self.points_delta_optimizer.param_groups[0]['lr']
        if optim == "stylize":
            return self.stylize_optimizer.param_groups[0]['lr']
        else:
            return None


def get_center_of_mass(shapes):
    all_points = []
    for shape in shapes:
        all_points.append(shape.points)
    points_vars = torch.vstack(all_points)
    center = points_vars.mean(dim=0)
    return center, all_points


def get_deltas(all_points, center, device):
    deltas_from_center = []
    for points in all_points:
        deltas = (points - center).to(device)
        deltas_from_center.append(deltas)
    return deltas_from_center