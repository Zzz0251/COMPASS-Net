import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from collections import namedtuple
from .segmentation_model import ResnetBlock, LayerNorm, SinusoidalPosEmb, Upsample, Downsample
from einops import rearrange

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.prenorm(x)
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale
        
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class CrossAttention(nn.Module):
                    
    def __init__(self, dim, text_dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.text_norm = nn.LayerNorm(text_dim)
        
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Linear(text_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
                  
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x, text_features, use_text_attention=True):
        if not use_text_attention or text_features is None:
            return x * 0                
            
        b, c, h, w = x.shape
        
        x_norm = self.norm(x)
        text_norm = self.text_norm(text_features)
        
        q = self.to_q(x_norm)
        q = q.view(b, self.heads, -1, h * w).transpose(-2, -1)
        
        kv = self.to_kv(text_norm)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(b, self.heads, 1, -1)
        v = v.view(b, self.heads, 1, -1)
        
        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = sim.softmax(dim=-2)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(-2, -1).contiguous().view(b, -1, h, w)
        
        out = self.to_out(out)
        return out * self.gate        

class SegmentationAttentionModule(nn.Module):
                             
    def __init__(self, dim, seg_channels=1):
        super().__init__()
        self.seg_channels = seg_channels
        
               
        self.seg_encoder = nn.Sequential(
            nn.Conv2d(seg_channels, dim//4, 3, padding=1),
            nn.GroupNorm(8, dim//4),
            nn.ReLU(),
            nn.Conv2d(dim//4, dim//2, 3, padding=1),
            nn.GroupNorm(8, dim//2),
            nn.ReLU(),
            nn.Conv2d(dim//2, dim, 3, padding=1)
        )
        
               
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GroupNorm(8, dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.Sigmoid()
        )
        
               
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1),
            nn.ReLU(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )
        
                
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1)) 
        
    def forward(self, x, seg_map, seg_strength=1.0):

        b, c, h, w = x.shape
        
               
        if seg_map.shape[-2:] != (h, w):
            seg_map = F.interpolate(seg_map, size=(h, w), mode='bilinear', align_corners=False)
        
        seg_features = self.seg_encoder(seg_map)
        
               
        combined = torch.cat([x, seg_features], dim=1)
        spatial_att = self.spatial_attn(combined)
        channel_att = self.channel_attn(seg_features)
        
               
        enhanced_features = x * (1 + self.alpha * spatial_att * seg_strength)
        enhanced_features = enhanced_features * (1 + self.beta * channel_att * seg_strength)
        
              
        return enhanced_features + seg_features * seg_strength

class FiLM(nn.Module):
                                                       
    def __init__(self, dim, condition_dim):
        super().__init__()
        self.scale_shift = nn.Linear(condition_dim, dim * 2)
        
    def forward(self, x, condition):

        scale, shift = self.scale_shift(condition).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift

class DiffusionUNet(nn.Module):
                    
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        output_channels=1,
        text_dim=256,
        resnet_block_groups=8,
        self_condition=False,
        use_film=True,              
    ):
        super().__init__()
        
        self.channels = channels
        self.self_condition = self_condition
        self.use_film = use_film
        input_channels = channels * (2 if self_condition else 1)
        
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        
              
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
             
        self.init_conv = nn.Conv2d(input_channels, dims[0], 7, padding=3)
        
                 
        self.seg_attention_modules = nn.ModuleList([
            SegmentationAttentionModule(dim_size) for dim_size in dims
        ])
        
               
        self.downs = nn.ModuleList([])
        self.cross_attns = nn.ModuleList([])
        self.film_layers = nn.ModuleList([]) if use_film else None
        
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (len(in_out) - 1)
            
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(Attention(dim_in)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))
            
                     
            self.cross_attns.append(CrossAttention(dim_in, text_dim))
            
                     
            if use_film:
                self.film_layers.append(FiLM(dim_in, text_dim))
        
             
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(Attention(mid_dim))
        self.mid_cross_attn = CrossAttention(mid_dim, text_dim)
        self.mid_seg_attn = SegmentationAttentionModule(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        
        if use_film:
            self.mid_film = FiLM(mid_dim, text_dim)
        
               
        self.ups = nn.ModuleList([])
        
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(Attention(dim_out)),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))
        
             
        self.final_res_block = block_klass(dims[0] * 2, dims[0], time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dims[0], output_channels, 1)
        
                       
        self.final_seg_attn = SegmentationAttentionModule(dims[0])
        
    def forward(self, x, attention_map, time, text_features=None, x_self_cond=None,
                use_text_attention=True, seg_strength=1.0):
 
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        
        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(time)
        h = []
        
             
        for i, ((block1, block2, attn, downsample), cross_attn) in enumerate(zip(self.downs, self.cross_attns)):
            x = block1(x, t)
            h.append(x)
            
                     
            x = self.seg_attention_modules[i](x, attention_map, seg_strength)
            
            x = block2(x, t)
            x = attn(x)
            
                         
            text_influence = cross_attn(x, text_features, use_text_attention)
            x = x + text_influence
            
                    
            if self.use_film and text_features is not None and use_text_attention:
                x = self.film_layers[i](x, text_features)
            
            h.append(x)
            x = downsample(x)
        
             
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        
                  
        x = self.mid_seg_attn(x, attention_map, seg_strength)
        
                  
        text_influence = self.mid_cross_attn(x, text_features, use_text_attention)
        x = x + text_influence
        
        if self.use_film and text_features is not None and use_text_attention:
            x = self.mid_film(x, text_features)
            
        x = self.mid_block2(x, t)
        
             
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            
            x = upsample(x)
        
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        
                   
        x = self.final_seg_attn(x, attention_map, seg_strength)
        
        return self.final_conv(x)