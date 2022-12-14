import torch
from torch import nn 
from transformer import Transformer 


class RevViT(nn.Module): 

    class Block(nn.Module):

        def __init__(
                    self, 
                    batch_size, 
                    dim, 
                    n_heads, 
                    n_patches, 
                    layer_factor, 
                    device, 
                    block_size=64, 
                    mlp_ratio=4.0, 
                    qkv_bias=True, 
                    k=8, 
                ):
            super().__init__()

            self.layer_factor = layer_factor
            self.n_patches = n_patches 
            self.batch_size = batch_size 

            n_flow = 1*layer_factor ##TODO

            self.flows = nn.ModuleList(
                [
                    Transformer(
                        batch_size=batch_size, 
                        dim=dim // 2, 
                        n_heads=n_heads, 
                        n_patches=n_patches, 
                        block_size=block_size, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        k=k, 
                        device=device, 
                    )
                for _ in range(n_flow)]
            ) 
            print("depth:", n_flow)

        def forward(self, x): 
            n_samples, n_patches, embed_dim = x.shape 
            x = x.view(n_samples, n_patches, embed_dim // 2, 2)

            for i, flow in enumerate(self.flows): 
                x = flow(x) 

            x = x.view(n_samples, n_patches, embed_dim)

            return x 

        def reverse(self, x):
            n_samples, n_patches, embed_dim = x.shape 
            x = x.view(n_samples, n_patches, embed_dim // 2, 2)

            for i, flow in enumerate(self.flows[::-1]): 
                x = flow.reverse(x) 

            x = x.view(n_samples, n_patches, embed_dim) 

            return x 
            

    """Custom implementation of the Vision transformer.
    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).
    patch_size : int
        Both height and the width of the patch (it is a square).
    in_chans : int
        Number of input channels.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    """

    def __init__(
            self,
            batch_size, 
            n_frames, 
            img_size,
            patch_size,
            PatchEmbed, 
            in_chans,
            n_heads,
            layer_factor, 
            device, 
            block_size=64,
            mlp_ratio=4.,
            qkv_bias=True,
            k=8, 
    ):
        n_patches = (img_size // patch_size) ** 2
        dim = (img_size ** 2) * in_chans // n_patches 
        self.n_frames = n_frames

        super().__init__()

        if PatchEmbed.TYPE == "IMAGE": 
            self.patch_embed = PatchEmbed(
                img_size=img_size, 
                patch_size=patch_size, 
                in_chans=in_chans, 
                embed_dim=dim
            )
        else: raise NotImplementedError 

        # Position embedding. 
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames-1, n_patches, dim)) # 1 + n_patches, dim)) #TODO

        self.space_token = nn.Parameter(torch.randn(1, 1, dim)) 

        self.space_transformer = self.Block(
                                        batch_size=batch_size*(n_frames-1), #batch_size, 
                                        dim=dim,
                                        block_size=block_size,
                                        layer_factor=layer_factor, 
                                        n_heads=n_heads,
                                        n_patches=n_patches, # 1 + n_patches, #TODO
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        k=k, 
                                        device=device, 
                                    )

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))

        self.temporal_transformer = self.Block(
                                        batch_size=batch_size, 
                                        dim=dim,
                                        block_size=block_size,
                                        layer_factor=layer_factor, 
                                        n_heads=n_heads,
                                        n_patches=n_frames, # 1 + n_frames, #TODO
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        k=k, 
                                        device=device, 
                                    ) 

    def forward(self, x): 
        x_embed = self.patch_embed(x)

        b, t, n, d = x_embed.shape
        space_tokens = self.space_token.expand(b, t, 1, -1)  # (b, t, 1, d)

        #TODO: Arrange n_patches to be divisible by transformer.block_size? 
        x = torch.cat((space_tokens, x_embed), dim=2)[:, :, :-1, :]  # (b, t, 1 + n - 1, d)
        x += self.pos_embed  # (b, t, 1 + n - 1, d) 
        x = x.view(b*t, n, d) ## x = x.view(b*t, 1+n, d) 
        x = self.space_transformer(x) 

        x_0 = x[:,0,:].view(b, -1, d) 
        temporal_tokens = self.temporal_token.expand(b, 1, -1)  # (b, 1, d)
        x_0 = torch.cat((temporal_tokens, x_0), dim=1)  # (b, 1 + t, d) 
        x_0 = self.temporal_transformer(x_0) 

        return x_embed[:,:,:-1,:], (x_0, x[:,1:,:]) 

    def reverse(self, x): 
        x_0, x = x 

        x_0 = self.temporal_transformer.reverse(x_0) 
        _, x_0 = x_0.split([1, x_0.shape[1]-1], dim=1) 

        b, t, d = x_0.shape 
        x_0 = x_0.view(b*t, 1, d) 

        x = torch.cat((x_0, x), dim=1) 
        x = self.space_transformer.reverse(x) 

        _, n, d = x.shape 
        x = x.view(b, t, n, d) 
        x -= self.pos_embed 
        _, x = x.split([1, x.shape[2]-1], dim=2) 

        #TODO: Arrange n_patches to be divisible by transformer.block_size? 
        x_embed = x 
        x = torch.cat((x, torch.zeros_like(x[:,:,0]).unsqueeze(2)), dim=2) 
        x = self.patch_embed.reverse(x)

        return x_embed, x 


class RevViTRev(nn.Module): 
    def __init__(
            self, batch_size, n_frames, img_size, patch_size, PatchEmbed, in_chans, n_heads, 
            layer_factor, device, block_size=64, mlp_ratio=4, qkv_bias=True, k=8
        ):
        self.self = RevViT(
            batch_size, n_frames, img_size, patch_size, PatchEmbed, in_chans, n_heads, 
            layer_factor, device, block_size, mlp_ratio, qkv_bias, k
        )

    def forward(self, x): 
        return self.self.reverse(x) 
    
    def reverse(self, x):
        return self.self.forward(x) 
