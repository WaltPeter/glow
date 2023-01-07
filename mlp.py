import math 
import torch 
import torch.nn as nn
import warnings 

from embeddings import ImagePatchEmbed 
from dense import MlpBlock 


class RevMlpMixer(nn.Module):

    class MixerBlock(nn.Module):

        def __init__(
            self, 
            sequence_length, 
            hidden_size, 
            mlp_dim_D_S, 
            mlp_dim_D_C, 
            checkerbox=True, 
            reversed=True, 
            lite=False, 
        ):
            super().__init__()

            self.norm_1 = nn.LayerNorm(hidden_size)
            self.norm_2 = nn.LayerNorm(hidden_size)

            self.token_mlp_block = MlpBlock(sequence_length, mlp_dim_D_S, lite=lite)
            self.channel_mlp_block = MlpBlock(hidden_size, mlp_dim_D_C, lite=lite) 

            self.checkerbox = checkerbox 
            self.reversed = reversed 

        def _f1(self, x): 
            x = self.norm_1(x) 
            x = x.permute(0, 2, 1) 
            x = self.token_mlp_block(x) 
            x = x.permute(0, 2, 1) 
            return x 

        def _f2(self, x): 
            x = self.norm_2(x) 
            x = self.channel_mlp_block(x) 
            return x 

        def _f(self, x1, x2): 
            y2 = x2 + self._f1(x1) 
            y1 = x1 + self._f2(y2) 
            return y1, y2 

        def _f_r(self, y1, y2): 
            x1 = y1 - self._f2(y2) 
            x2 = y2 - self._f1(x1) 
            return x1, x2 

        def _split(self, x): 
            if self.checkerbox: x = x.view(*x.shape[:-1], x.shape[-1]//2, 2) 
            x1, x2 = x.chunk(2, -1)[::-1] if self.reversed else x.chunk(2, -1) 
            if self.checkerbox: x1, x2 = [i.squeeze(-1) for i in [x1, x2]] 
            return x1, x2 

        def _merge(self, y1, y2): 
            if self.checkerbox: y1, y2 = [i.unsqueeze(-1) for i in [y1, y2]] 
            # y = torch.cat([y2, y1], -1) if self.reversed else torch.cat([y1, y2], -1) 
            y = torch.cat([y1, y2], -1) 
            if self.checkerbox: y = y.view(*y.shape[:-2], y.shape[-2]*2) 
            return y 

        def forward(self, x): 
            x1, x2 = self._split(x)
            y1, y2 = self._f(x1, x2) 
            return self._merge(y1, y2)
        
        def reverse(self, y): 
            y1, y2 = self._split(y)
            x1, x2 = self._f_r(y1, y2) 
            return self._merge(x1, x2) 


    def __init__(
        self,
        n_frames, 
        image_size,
        patch_res,
        mlp_dim_D_S,
        mlp_dim_D_C, 
        n_layers,
        lite=False, 
    ):
        super().__init__()
        assert image_size % patch_res == 0, "`image_size` must be divisible by `patch_res`." 
        if not n_layers % 4 == 0: warnings.warn("`n_layers` is not divisible by 4.") 
        sequence_length = (image_size // patch_res) ** 2 
        hidden_size = patch_res**2 * 3 // 2

        self.patch_embedder = ImagePatchEmbed(
                    img_size=image_size, 
                    patch_size=patch_res, 
                )

        l_checkerbox = [True, True, False, False] * max(n_layers // 4, 1) 
        
        self.spatial_blocks = nn.ModuleList(
            [
                self.MixerBlock(
                    sequence_length=sequence_length,
                    hidden_size=hidden_size,
                    mlp_dim_D_S=mlp_dim_D_S,
                    mlp_dim_D_C=mlp_dim_D_C,
                    checkerbox=l_checkerbox[i], 
                    lite=lite, 
                )
                for i in range(n_layers)
            ]
        )

        self.temporal_blocks = nn.ModuleList(
            [
                self.MixerBlock(
                    sequence_length=n_frames,
                    hidden_size=hidden_size,
                    mlp_dim_D_S=mlp_dim_D_S,
                    mlp_dim_D_C=mlp_dim_D_C,
                    checkerbox=l_checkerbox[i], 
                    lite=lite, 
                )
                for i in range(n_layers // 2)
            ]
        )

        self.bridge_blocks = nn.ModuleList(
            [
                self.MixerBlock(
                    sequence_length=sequence_length-1,
                    hidden_size=hidden_size,
                    mlp_dim_D_S=mlp_dim_D_S,
                    mlp_dim_D_C=mlp_dim_D_C,
                    checkerbox=l_checkerbox[i], 
                    lite=lite, 
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        x_embed = self.patch_embedder(x) 

        b, t, n, d = x_embed.shape
        x = x_embed.view(b*t, n, d) 
        for blk in self.spatial_blocks: x = blk(x) 

        x_0 = x[:,0,:].view(b, t, d) 
        for blk in self.temporal_blocks: x_0 = blk(x_0) 

        x = x[:,1:,:] 
        for blk in self.bridge_blocks: x = blk(x) 
        # return x_embed, (x_0, x) 
        return x_0, x

    def reverse(self, x): 
        x_0, x = x 
        for blk in self.bridge_blocks[::-1]: x = blk.reverse(x) 

        for blk in self.temporal_blocks[::-1]: x_0 = blk.reverse(x_0) 
        b, t, d = x_0.shape 
        x_0 = x_0.view(b*t, 1, d) 

        x = torch.cat([x_0, x], dim=1) 
        for blk in self.spatial_blocks[::-1]: x = blk.reverse(x) 
        x_embed = x.view(b, t, -1, d) 
        x = self.patch_embedder.reverse(x_embed) 
        # return x_embed, x 
        return x 

    
class Model(nn.Module): 
    def __init__(
        self,
        n_frames, 
        image_size,
        patch_res,
        mlp_dim_D_S,
        mlp_dim_D_C, 
        n_layers,
        lite=False, 
    ):
        super().__init__()
        self.visual_nn, self.audio_nn =\
            [RevMlpMixer(
                n_frames=n_frames, 
                image_size=image_size, 
                patch_res=patch_res, 
                mlp_dim_D_S=mlp_dim_D_S, 
                mlp_dim_D_C=mlp_dim_D_C, 
                n_layers=n_layers, 
                lite=lite, 
            ) for _ in range(2)] 
    
    def forward(self, x):
        x = self.visual_nn(x) 
        x = self.audio_nn.reverse(x) 
        return x 
    
    def reverse(self, x): 
        x = self.audio_nn(x) 
        x = self.visual_nn.reverse(x) 
        return x 


if __name__ == "__main__": 
    n_frames = 5 
    image_size = 256 
    patch_res = 16 
    mlp_dim_D_S = 512 
    mlp_dim_D_C = 4096 
    n_layers = 2 #4 #24 

    model = RevMlpMixer(
                n_frames=n_frames, 
                image_size=image_size, 
                patch_res=patch_res, 
                mlp_dim_D_S=mlp_dim_D_S, 
                mlp_dim_D_C=mlp_dim_D_C, 
                n_layers=n_layers
            ) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device) 

    x = torch.randn(1, n_frames, 3, image_size, image_size).to(device)
    y = model(x)
    x_ = model.reverse(y[1]) 
    print(torch.dist(x, x_[1])) 