
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + (self.b if self.b is not None else 0)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

def Upsample(dim, dim_out=None):
    dim_out = dim_out or dim
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, dim_out, 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    dim_out = dim_out or dim
    return nn.Sequential(
        nn.Conv2d(dim, dim * 4, 3, stride=2, padding=1),
        nn.Conv2d(dim * 4, dim_out, 1)
    )
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.prenorm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)

class SpatialAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.attention = Attention(dim, heads, dim_head)

    def forward(self, x):
        return self.attention(x)

class SegmentationUNet(nn.Module):

    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,  
        output_channels=1,  
        resnet_block_groups=8,
        use_attention=True,  
        deep_supervision=True,  
    ):
        super().__init__()
        
        self.channels = channels
        self.use_attention = use_attention
        self.deep_supervision = deep_supervision
        
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(channels, dims[0]//2, 3, padding=1),
            nn.GroupNorm(resnet_block_groups, dims[0]//2),
            nn.SiLU(),
            nn.Conv2d(dims[0]//2, dims[0], 7, padding=3)
        )
        

        self.downs = nn.ModuleList([])
        self.down_attentions = nn.ModuleList([]) if use_attention else None
        
        for i, (dim_in, dim_out) in enumerate(in_out[:-1]):
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in),
                block_klass(dim_in, dim_in),
                Downsample(dim_in, dim_out)
            ]))
            
            if use_attention:
                self.down_attentions.append(SpatialAttention(dim_in))
        

        dim_in, dim_out = in_out[-1]
        self.downs.append(nn.ModuleList([
            block_klass(dim_in, dim_in),
            block_klass(dim_in, dim_in),
            nn.Conv2d(dim_in, dim_out, 3, padding=1)
        ]))
        
        if use_attention:
            self.down_attentions.append(SpatialAttention(dim_in))
        

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = SpatialAttention(mid_dim) if use_attention else nn.Identity()
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        

        self.ups = nn.ModuleList([])
        self.up_attentions = nn.ModuleList([]) if use_attention else None
        
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out),
                block_klass(dim_out + dim_in, dim_out),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))
            
            if use_attention:
                self.up_attentions.append(SpatialAttention(dim_out))
        

        self.final_res_block = block_klass(dims[0] * 2, dims[0])
        self.final_conv = nn.Conv2d(dims[0], output_channels, 1)
        

        if deep_supervision:
            self.aux_outputs = nn.ModuleList([
                nn.Conv2d(dim, output_channels, 1) for dim in dims[1:-1]
            ])
        

        self.feature_extractor = nn.ModuleDict({

            f'down_{i}': nn.Conv2d(dim, dim//2, 1) for i, dim in enumerate(dims[:-1])
        })
        

        up_dims = []
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            up_dims.append(dim_out)
        
        for i, dim in enumerate(up_dims):
            self.feature_extractor[f'up_{i}'] = nn.Conv2d(dim, dim//2, 1)
        
        self.feature_extractor['bottleneck'] = nn.Conv2d(dims[-1], dims[-1]//2, 1)
        
    def forward(self, x, return_features=False):

        original_size = x.shape[-2:]
        x = self.init_conv(x)
        r = x.clone()
        
        h = []
        features = {}
        aux_outputs = []
        

        for i, (block1, block2, downsample) in enumerate(self.downs):
            x = block1(x)
            

            if self.use_attention and i < len(self.down_attentions):
                x = x + self.down_attentions[i](x)
            
            h.append(x)
            

            if return_features and f'down_{i}' in self.feature_extractor:
                features[f'down_{i}'] = self.feature_extractor[f'down_{i}'](x)
            
            x = block2(x)
            h.append(x)
            

            if self.deep_supervision and i > 0 and i < len(self.downs) - 1:
                if i-1 < len(self.aux_outputs):
                    aux_out = F.interpolate(
                        self.aux_outputs[i-1](x), 
                        size=original_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                    aux_outputs.append(aux_out)
            
            x = downsample(x)
        

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        

        if return_features:
            features['bottleneck'] = self.feature_extractor['bottleneck'](x)
        

        for i, (block1, block2, upsample) in enumerate(self.ups):

            skip_conn = h.pop()
            if x.shape[-2:] != skip_conn.shape[-2:]:
                x = F.interpolate(x, size=skip_conn.shape[-2:], mode='nearest')
            x = torch.cat((x, skip_conn), dim=1)
            x = block1(x)
            

            if self.use_attention and i < len(self.up_attentions):
                x = x + self.up_attentions[i](x)
            
            skip_conn = h.pop()
            if x.shape[-2:] != skip_conn.shape[-2:]:
                x = F.interpolate(x, size=skip_conn.shape[-2:], mode='nearest')
            x = torch.cat((x, skip_conn), dim=1)
            x = block2(x)
            

            if return_features and f'up_{i}' in self.feature_extractor:
                features[f'up_{i}'] = self.feature_extractor[f'up_{i}'](x)
            
            x = upsample(x)
        
              
        if x.shape[-2:] != r.shape[-2:]:
            x = F.interpolate(x, size=r.shape[-2:], mode='nearest')
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x)
        
        main_output = self.final_conv(x)
        
        if return_features:
                                         
                                         
            pooled_features = F.adaptive_avg_pool2d(x, 1).flatten(1)                    
            return main_output, pooled_features, aux_outputs
        else:
            return main_output

class EnhancedAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_boundary_aware=True):
        super().__init__()
        self.use_boundary_aware = use_boundary_aware
        
                   
        self.prob_to_attention = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.GroupNorm(8, 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, 1),
            nn.Sigmoid()
        )
        
                    
        if use_boundary_aware:
            self.boundary_detector = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, out_channels, 1),
                nn.Sigmoid()
            )
        
    def forward(self, segmentation_pred, brain_mask=None):
       
        if segmentation_pred.shape[1] != 1:
            print(f"WARNING: Expected 1 channel, got {segmentation_pred.shape[1]} channels")
            segmentation_pred = segmentation_pred[:, 0:1]          
        
               
        infarct_prob = torch.sigmoid(segmentation_pred)
        
                  
        attention_map = self.prob_to_attention(infarct_prob)
        
                  
        if self.use_boundary_aware and hasattr(self, 'boundary_detector'):
                       
            if brain_mask is not None:
                boundary_map = self.boundary_detector(brain_mask)
                             
                attention_map = attention_map * (1.0 + 0.3 * boundary_map)
        
                         
        attention_map = attention_map + infarct_prob * 0.5
        
               
        if brain_mask is not None:
            attention_map = attention_map * brain_mask
            
                    
        attention_map = torch.clamp(attention_map, 0, 1)
        
        return attention_map

