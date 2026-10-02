
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from tqdm.auto import tqdm
from collections import namedtuple

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

                   
def palette_beta_schedule(timesteps, start=0.0001, end=0.02, tau=1.0):
                           
    steps = timesteps
    t = torch.linspace(0, 1, steps, dtype=torch.float64)
    
              
    betas = start + (end - start) * (1 - torch.cos(t * math.pi / 2)) ** tau
    return betas

class PerceptualLoss(nn.Module):
                          
    def __init__(self):
        super().__init__()
                          
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target, mask=None):
              
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        
              
        grad_loss = F.mse_loss(pred_grad_x, target_grad_x, reduction='none') + \
                   F.mse_loss(pred_grad_y, target_grad_y, reduction='none')
        
        if mask is not None:
            grad_loss = grad_loss * mask
            
        return grad_loss.mean()

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='hybrid',          
        objective='pred_noise',
        beta_schedule='cosine',              
        p2_loss_weight_gamma=0.,
        p2_loss_weight_k=1,
        ddim_sampling_eta=0.,
              
        use_perceptual_loss=True,
        perceptual_weight=0.1,
        use_self_conditioning=False,
        self_condition_prob=0.5,
        classifier_free_guidance=True,
        guidance_scale=7.5,
                     
        use_palette_schedule=False,
        condition_dropout_prob=0.1,
    ):
        super().__init__()
        
        self.model = model
        self.image_size = image_size
        self.objective = objective
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_weight = perceptual_weight
        self.use_self_conditioning = use_self_conditioning
        self.self_condition_prob = self_condition_prob
        self.classifier_free_guidance = classifier_free_guidance
        self.guidance_scale = guidance_scale
        self.condition_dropout_prob = condition_dropout_prob
        
                
        if use_palette_schedule:
            beta_schedule_fn = palette_beta_schedule
        elif beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

                
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

               
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

                
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

              
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

                
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)
        
              
        if use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss()

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / 
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, attention_map, text_features, t, x_self_cond=None, 
                         clip_x_start=False, use_text_attention=True, seg_strength=1.0):
                              
        
                                       
        if self.training and self.classifier_free_guidance:
            if torch.rand(1).item() < self.condition_dropout_prob:
                attention_map = torch.zeros_like(attention_map)
                use_text_attention = False
        
                
        model_output = self.model(
            x, attention_map, t, text_features, x_self_cond,
            use_text_attention=use_text_attention,
            seg_strength=seg_strength
        )
        
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else lambda x: x

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, attention_map, text_features, t, x_self_cond=None, 
                       clip_denoised=True, use_text_attention=True, seg_strength=1.0):
        preds = self.model_predictions(
            x, attention_map, text_features, t, x_self_cond,
            use_text_attention=use_text_attention, seg_strength=seg_strength
        )
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, attention_map, text_features, t, x_self_cond=None, 
                brain_mask=None, use_text_attention=True, seg_strength=1.0):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
                                  
        if self.classifier_free_guidance and not self.training:
                   
            model_mean_uncond, _, model_log_variance_uncond, x_start_uncond = self.p_mean_variance(
                x=x, attention_map=torch.zeros_like(attention_map), 
                text_features=text_features, t=batched_times, 
                x_self_cond=x_self_cond, clip_denoised=True,
                use_text_attention=False, seg_strength=0.0
            )
            
                   
            model_mean_cond, _, model_log_variance_cond, x_start_cond = self.p_mean_variance(
                x=x, attention_map=attention_map, text_features=text_features, 
                t=batched_times, x_self_cond=x_self_cond, clip_denoised=True,
                use_text_attention=use_text_attention, seg_strength=seg_strength
            )
            
                  
            model_mean = model_mean_uncond + self.guidance_scale * (model_mean_cond - model_mean_uncond)
            model_log_variance = model_log_variance_cond
            x_start = x_start_uncond + self.guidance_scale * (x_start_cond - x_start_uncond)
            
        else:
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(
                x=x, attention_map=attention_map, text_features=text_features, 
                t=batched_times, x_self_cond=x_self_cond, clip_denoised=True,
                use_text_attention=use_text_attention, seg_strength=seg_strength
            )
        
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        
               
        if brain_mask is not None:
            pred_img = pred_img * brain_mask
            
        return pred_img, x_start

    @torch.no_grad()
    def sample(self, ncct, attention_map, text_features, brain_mask=None, 
               use_text_attention=True, seg_strength=1.0, return_all_timesteps=False):
                     
        batch_size, device = ncct.shape[0], ncct.device
        shape = ncct.shape
        
        img = torch.randn(shape, device=device)
        
                 
        if brain_mask is not None:
            img = img * brain_mask
        
        imgs = [img] if return_all_timesteps else None
        x_self_cond = None
        
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step'):
                 
            if self.use_self_conditioning:
                with torch.no_grad():
                    x_self_cond = self.model_predictions(
                        img, attention_map, text_features, 
                        torch.full((batch_size,), t, device=device, dtype=torch.long),
                        use_text_attention=use_text_attention, seg_strength=seg_strength
                    ).pred_x_start
                    x_self_cond.detach_()
            
            img, _ = self.p_sample(
                img, attention_map, text_features, t, 
                x_self_cond=x_self_cond, brain_mask=brain_mask,
                use_text_attention=use_text_attention, seg_strength=seg_strength
            )
            
            if return_all_timesteps:
                imgs.append(img)
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss
        elif self.loss_type == 'hybrid':
            return lambda x, y, **kwargs: F.mse_loss(x, y, **kwargs) + 0.1 * F.l1_loss(x, y, **kwargs)
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, ncct, attention_map, text_features, t, brain_mask=None, 
                noise=None, use_text_attention=True, seg_strength=1.0):
   
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

                
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        
               
        if brain_mask is not None:
            x = x * brain_mask
            x_start = x_start * brain_mask

             
        x_self_cond = None
        if self.use_self_conditioning and torch.rand(1).item() < self.self_condition_prob:
            with torch.no_grad():
                x_self_cond = self.model_predictions(
                    x, attention_map, text_features, t,
                    use_text_attention=use_text_attention, seg_strength=seg_strength
                ).pred_x_start
                x_self_cond.detach_()

              
        model_out = self.model(
            x, attention_map, t, text_features, x_self_cond,
            use_text_attention=use_text_attention, seg_strength=seg_strength
        )
        
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

              
        loss = self.loss_fn(model_out, target, reduction='none')
        
              
        if self.use_perceptual_loss and self.objective == 'pred_x0':
            pred_x_start = model_out
            perceptual_loss = self.perceptual_loss(pred_x_start, x_start, brain_mask)
            loss = loss + self.perceptual_weight * perceptual_loss
        
                  
        if brain_mask is not None:
            loss = loss * brain_mask
            
                          
        if attention_map is not None:
                         
            region_weight = 1.0 + attention_map * 2.0            
            loss = loss * region_weight
            
        loss = loss.mean(dim=[1, 2, 3])
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        
        return loss.mean()

                
    def get_current_timestep_range(self, epoch, total_epochs):
                                
        if epoch < total_epochs * 0.3:
                                
            min_t = self.num_timesteps // 2
            max_t = self.num_timesteps
        elif epoch < total_epochs * 0.7:
                       
            min_t = self.num_timesteps // 4
            max_t = self.num_timesteps
        else:
                       
            min_t = 0
            max_t = self.num_timesteps
            
        return min_t, max_t

                  
    def adaptive_loss_weights(self, seg_loss, diffusion_loss, epoch, total_epochs):
                            
                       
        progress = epoch / total_epochs
        seg_weight = 1.0 - 0.5 * progress             
        diff_weight = 0.5 + 0.5 * progress             
        
        return seg_weight * seg_loss + diff_weight * diffusion_loss
