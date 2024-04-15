from diffusers import DiffusionPipeline
import torch.nn as nn
import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from model_utils import configure_lora
import einops

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import Delaunay
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
import lpips

# =============================================
# ===== Helper function for SDS gradients =====
# =============================================
class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


# ========================================================
# ===== Basic class to extend with SDS loss variants =====
# ========================================================
class SDSLossBase(nn.Module):

    _global_pipe = None

    def __init__(self, cfg, device, reuse_pipe=True):
        super(SDSLossBase, self).__init__()

        self.cfg = cfg
        self.device = device

        # initiate a diffusion pipeline if we don't already have one / don't want to reuse it for both paths
        self.maybe_init_pipe(reuse_pipe) 

        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        if cfg.use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()

        self.text_embeddings = self.embed_text(self.cfg.caption)

        if self.cfg.del_text_encoders:
            del self.pipe.tokenizer
            del self.pipe.text_encoder

    def maybe_init_pipe(self, reuse_pipe):
        if reuse_pipe:
            if SDSLossBase._global_pipe is None:
                SDSLossBase._global_pipe = DiffusionPipeline.from_pretrained(self.cfg.model_name, torch_dtype=torch.float16, variant="fp16")
                SDSLossBase._global_pipe = SDSLossBase._global_pipe.to(self.device)
            self.pipe = SDSLossBase._global_pipe
        else:
            self.pipe = DiffusionPipeline.from_pretrained(self.cfg.model_name, torch_dtype=torch.float16, variant="fp16")
            self.pipe = self.pipe.to(self.device)

    def embed_text(self, caption):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
            
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        text_embeddings = text_embeddings.repeat_interleave(self.cfg.batch_size, 0)

        return text_embeddings

        
    def prepare_latents(self, x_aug):
        x = x_aug * 2. - 1. # encode rendered image, values should be in [-1, 1]
        
        with torch.cuda.amp.autocast():
            batch_size, num_frames, channels, height, width = x.shape # [1, K, 3, 256, 256], for K frames
            x = x.reshape(batch_size * num_frames, channels, height, width) # [K, 3, 256, 256], for the VAE encoder
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample()) # [K, 4, 32, 32]
            frames, channel, h_, w_ = init_latent_z.shape
            init_latent_z = init_latent_z[None, :].reshape(batch_size, num_frames, channel, h_, w_).permute(0, 2, 1, 3, 4) # [1, 4, K, 32, 32] for the video model
            
        latent_z = self.pipe.vae.config.scaling_factor * init_latent_z  # scaling_factor * init_latents

        return latent_z

    def add_noise_to_latents(self, latent_z, timestep, return_noise=True, eps=None):
        
        # sample noise if not given some as an input
        if eps is None:
            if self.cfg.same_noise_for_frames: # This works badly. Do not use.
                eps = torch.randn_like(latent_z[:, :, 0, :, :]) # create noise for single frame
                eps = einops.repeat(eps, 'b c h w -> b c f h w', f=latent_z.shape[2])
            else:
                eps = torch.randn_like(latent_z)

        # zt = alpha_t * latent_z + sigma_t * eps
        noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

        if return_noise:
            return noised_latent_zt, eps

        return noised_latent_zt
    
    # overload this if inheriting for VSD etc.
    def get_sds_eps_to_subract(self, eps_orig, z_in, timestep_in):
        return eps_orig

    def drop_nans(self, grads):
        assert torch.isfinite(grads).all()
        return torch.nan_to_num(grads.detach().float(), 0.0, 0.0, 0.0)

    def get_grad_weights(self, timestep):
        return (1 - self.alphas[timestep])

    def sds_grads(self, latent_z, **sds_kwargs):

        with torch.no_grad():
            # sample timesteps
            timestep = torch.randint(
                low=self.cfg.sds_timestep_low,
                high=min(950, self.cfg.timesteps) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            noised_latent_zt, eps = self.add_noise_to_latents(latent_z, timestep, return_noise=True)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep_in, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)
            
            eps_t = eps_t_uncond + self.cfg.guidance_scale * (eps_t - eps_t_uncond)

            eps_to_subtract = self.get_sds_eps_to_subract(eps, z_in, timestep_in, **sds_kwargs)

            w = self.get_grad_weights(timestep)
            grad_z = w * (eps_t - eps_to_subtract)
            
            grad_z = self.drop_nans(grad_z)

        return grad_z

# =======================================
# =========== Basic SDS loss  ===========
# =======================================
class SDSVideoLoss(SDSLossBase):
    def __init__(self, cfg, device, reuse_pipe=True):
        super(SDSVideoLoss, self).__init__(cfg, device, reuse_pipe=reuse_pipe)

    def forward(self, x_aug, grad_scale=1.0):
        latent_z = self.prepare_latents(x_aug)

        grad_z = grad_scale * self.sds_grads(latent_z)

        sds_loss = SpecifyGradient.apply(latent_z, grad_z)

        return sds_loss    

# =======================================
# =========== Perceptual Loss  ==========
# =======================================

class PerceptualLoss(nn.Module):
    def __init__(self, cfg):
        super(PerceptualLoss, self).__init__()
        # Load a pre-trained VGG19 model and use its features for perceptual loss
        print('before_load')
        self.lpips_loss = lpips.LPIPS(net='vgg').to('cuda')
        print('after_load')
        self.imit = None

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)

    def forward(self, image):
        # print(self.lpips_loss(self.im_init.detach(), image))
        return self.lpips_loss(self.im_init.detach(), image).squeeze()
 
# =======================================
# ===== Structure Preservation Loss =====
# =======================================
           
class ConformalLoss:
    def __init__(self, parameters: EasyDict, device: torch.device, target_letter: str, shape_groups):
        self.parameters = parameters
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.device = device

        self.faces = self.init_faces(self.device)
        print('device_faces',self.faces[0].device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.reset(device)

    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []

        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_

    def get_template_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []

        for i in range(len(self.template_faces)):
            triangles = points[self.template_faces[i]]
            triangles_roll_a = points[self.template_faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_

    def reset(self, device):
        points = torch.cat([point.clone().detach() for point in self.parameters.point])
        self.angles = self.get_angles(points)
        self.angles = [angle.to(device) for angle in self.angles]

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters.point[i].clone().detach().cpu().numpy() for i in range(len(self.parameters.point))]
            shapes_per_letter=len(self.shape_groups) #yihao-based on zichen's pre-setting, the shapes number should equal to shapes_group number
            holes = []
            # if shapes_per_letter > 1:
            #     holes = points_np[1:1]   #yihao: except the first shape(path), all other paths are holes  
            poly = Polygon(points_np[0], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=np.bool)
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))

        return faces_
    
    def update_template(self, template_points) -> torch.Tensor:
        # Triangulate the template in each iteration
        with torch.no_grad():
            self.template_faces = self.triangulate_template(template_points)
            self.template_faces_roll_a = [torch.roll(self.template_faces[i], 1, 1) for i in range(len(self.template_faces))]   

    def triangulate_template(self, template_points) -> torch.Tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [template_points[i].clone().detach().cpu().numpy() for i in range(len(template_points))]
            shapes_per_letter = len(self.shape_groups)
            holes = []
            poly = Polygon(points_np[0], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=np.bool)
            faces_.append(torch.from_numpy(faces[is_intersect]).to(self.device, dtype=torch.int64))
        return faces_
    
    def __call__(self, parameters1, parameters2=None) -> torch.Tensor:
        # Modify __call__ to accept parameters as an argument
        loss_angles = 0
        if parameters2 is not None:
            points1 = torch.cat(parameters1)
            angles1 = self.get_template_angles(points1)
            points2=torch.cat(parameters2)
            angles2 = self.get_template_angles(points2)
            for i in range(len(self.faces)):
                loss_angles += nnf.mse_loss(angles1[i], angles2[i].detach())
        else:
            points1 = torch.cat(parameters1)
            angles1 = self.get_angles(points1)
            for i in range(len(self.faces)):
                loss_angles += nnf.mse_loss(angles1[i], self.angles[i])

        return loss_angles