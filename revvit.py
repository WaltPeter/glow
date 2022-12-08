from tqdm import tqdm 
import numpy as np 
from PIL import Image
import math

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils 
from math import log, pi, exp

class GELU(nn.Module): 
    def forward(self, x): 
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class RevViT_GLOW(nn.Module): 

    class Block(nn.Module):

        class Transformer(nn.Module):

            class Att(nn.Module): 
                '''Attention mechanism.
                Parameters
                ----------
                dim : int
                    The input and out dimension of per token features.
                n_heads : int
                    Number of attention heads.
                qkv_bias : bool
                    If True then we include bias to the query, key and value projections.
                attn_p : float
                    Dropout probability applied to the query, key and value tensors.
                proj_p : float
                    Dropout probability applied to the output tensor.
                '''

                def __init__(self, dim, n_heads=12, qkv_bias=True):
                    super().__init__()
                    self.n_heads = n_heads 
                    self.dim = dim 
                    self.head_dim = dim // n_heads 
                    self.scale = self.head_dim ** -0.5 

                    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
                    self.proj = nn.Linear(dim, dim)
                
                def forward(self, x):
                    n_samples, n_tokens, dim = x.shape

                    if dim != self.dim:
                        raise ValueError

                    qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
                    qkv = qkv.reshape(
                            n_samples, n_tokens, 3, self.n_heads, self.head_dim
                    )  # (n_samples, n_patches + 1, 3, n_heads, head_dim)
                    qkv = qkv.permute(
                            2, 0, 3, 1, 4
                    )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

                    q, k, v = qkv[0], qkv[1], qkv[2]
                    k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
                    dp = (
                    q @ k_t
                    ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
                    attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)

                    weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
                    weighted_avg = weighted_avg.transpose(
                            1, 2
                    )  # (n_samples, n_patches + 1, n_heads, head_dim)
                    weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

                    x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)

                    return x
                
            class MLP(nn.Module):
                """Multilayer perceptron.
                Parameters
                ----------
                in_features : int
                    Number of input features.
                hidden_features : int
                    Number of nodes in the hidden layer.
                out_features : int
                    Number of output features.
                p : float
                    Dropout probability.
                """

                def __init__(self, in_features, hidden_features, out_features, p=0.):
                    super().__init__()
                    self.fc1 = nn.Linear(in_features, hidden_features)
                    self.act = GELU()
                    self.fc2 = nn.Linear(hidden_features, out_features)
                    self.drop = nn.Dropout(p)

                def forward(self, x):
                    x = self.fc1(x) 
                    x = self.act(x) 
                    x = self.drop(x) 
                    x = self.fc2(x)
                    x = self.drop(x) 
                    return x 

            """Transformer block.
            Parameters
            ----------
            dim : int
                Embedding dimension.
            n_heads : int
                Number of attention heads.
            mlp_ratio : float
                Determines the hidden dimension size of the `MLP` module with respect
                to `dim`.
            qkv_bias : bool
                If True then we include bias to the query, key and value projections.
            p, attn_p : float
                Dropout probability.
            """

            def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim, eps=1e-6)
                self.attn = self.Att(
                        dim=dim,
                        n_heads=n_heads,
                        qkv_bias=qkv_bias,
                )
                self.norm2 = nn.LayerNorm(dim, eps=1e-6)
                hidden_features = int(dim * mlp_ratio)
                self.mlp = self.MLP(
                        in_features=dim,
                        hidden_features=hidden_features,
                        out_features=dim,
                )

            def forward(self, x): 
                x2, x1 = x.chunk(2, -1)
                x2, x1 = [i.squeeze(-1) for i in [x2, x1]]
                y2 = x2 + self.attn(self.norm1(x1)) 
                y1 = x1 + self.mlp(self.norm2(y2)) 
                y2, y1 = [i.unsqueeze(-1) for i in [y2, y1]] 
                return torch.cat([y1, y2], -1) 

            def reverse(self, y): 
                y1, y2 = y.chunk(2, -1) 
                y1, y2 = [i.squeeze(-1) for i in [y1, y2]] 
                x1 = y1 - self.mlp(self.norm2(y2)) 
                x2 = y2 - self.attn(self.norm1(x1)) 
                x1, x2 = [i.unsqueeze(-1) for i in [x1, x2]]
                return torch.cat([x2, x1], -1) 

        def _sample_z(self, mean, sigma): 
            eps = torch.randn_like(mean)
            return mean + torch.exp(sigma / 2) * eps

        def __init__(self, dim, n_heads, n_flow, n_patches, mlp_ratio=4.0, qkv_bias=True):
            super().__init__()

            self.layer_factor = 4

            self.flows = nn.ModuleList(
                [
                    self.Transformer(
                        dim=dim // 2, 
                        n_heads=n_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                    )
                    for _ in range(n_flow * self.layer_factor)
                ]
            )

            latent_dim = 16*16 #TODO
            self.norm = nn.LayerNorm(dim*n_patches, eps=1e-6) 
            self.prior = nn.Linear(dim*n_patches, latent_dim*2)  

        def forward(self, x):
            n_samples, n_patches, embed_dim = x.shape 
            x = x.view(n_samples, n_patches, embed_dim // 2, 2)

            x_outs = list() 
            for i, flow in enumerate(self.flows): 
                x = flow(x) 
                
                if (i+1) % self.layer_factor == 0: 
                    x, x_out = x.split([x.shape[1]-x.shape[1]//2, x.shape[1]//2], dim=1) 
                    x_outs.append(x_out) 
            
            x_outs.append(x) 
            x = torch.cat(x_outs[::-1], dim=1) 

            x = x.view(n_samples, n_patches, embed_dim)

            flatten = torch.flatten(x, start_dim=1) 
            mean, sigma = torch.mean( self.prior( self.norm(flatten) ), dim=0).chunk(2, -1) 

            return x, mean, sigma 

        def reverse(self, x):
            n_samples, n_patches, embed_dim = x.shape 
            x = x.view(n_samples, n_patches, embed_dim // 2, 2)

            for flow in self.flows[::-1]:
                x = flow.reverse(x) 

            x = x.view(n_samples, n_patches, embed_dim) 

            return x 

    class PatchEmbed(nn.Module):
        # Split image into patches and then embed them.
        def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
            super().__init__()
            self.patch_size = patch_size
            self.n_patches = img_size // patch_size 

        def forward(self, x):
            b, t, c, _, _ = x.shape 
            x = x.view(b, t, c, self.n_patches, self.patch_size, self.n_patches, self.patch_size) 
            x = x.permute(0,1,3,5,4,6,2).contiguous()
            x = x.view(b, t, self.n_patches*self.n_patches, self.patch_size*self.patch_size*c) 
            # print("Embed size:", x.shape)
            return x 

        def reverse(self, x): 
            b, t, n, d = x.shape 
            x = x.view(b, t, self.n_patches, self.n_patches, self.patch_size, self.patch_size, -1) 
            c = x.shape[-1]
            x = x.permute(0,1,6,2,4,3,5).contiguous() 
            x = x.view(b, t, c, self.n_patches*self.patch_size, self.n_patches*self.patch_size) 
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
            n_frames=1, 
            img_size=384,
            patch_size=16,
            in_chans=3,
            n_heads=12,
            n_flow=12, 
            mlp_ratio=4.,
            qkv_bias=True,
    ):
        n_patches = (img_size // patch_size) ** 2
        dim = (img_size ** 2) * in_chans // n_patches 
        self.n_frames = n_frames

        # print("depth: {}, n_patches: {}+{}".format(depth, split_sz, n_patches)) 

        super().__init__()

        self.patch_embed = self.PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=dim
        )

        # Position embedding. 
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, 1 + n_patches, dim)) 

        self.space_token = nn.Parameter(torch.randn(1, 1, dim)) 

        self.space_transformer = self.Block(
                                        dim=dim,
                                        n_flow=n_flow, 
                                        n_heads=n_heads,
                                        n_patches=1 + n_patches, 
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                    )

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))

        self.temporal_transformer = self.Block(
                                        dim=dim,
                                        n_flow=n_flow, 
                                        n_heads=n_heads,
                                        n_patches=1 + n_frames, 
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                    ) 

    def forward(self, x): 
        x = self.patch_embed(x)

        b, t, n, d = x.shape
        space_tokens = self.space_token.expand(b, t, 1, -1)  # (b, t, 1, d)
        x = torch.cat((space_tokens, x), dim=2)  # (b, t, 1 + n, d)
        x += self.pos_embed  # (b, t, 1 + n, d) 

        x = x.view(b*t, 1+n, d) 
        x, mean_1, sigma_1 = self.space_transformer(x) 
        x = x[:,0].view(b, t, -1) 

        temporal_tokens = self.temporal_token.expand(b, 1, -1)  # (b, 1, d)
        x = torch.cat((temporal_tokens, x), dim=1)  # (b, 1 + t, d) 
         
        x, mean_2, sigma_2 = self.temporal_transformer(x) 

        mean_list = torch.cat([mean_1.unsqueeze(0), mean_2.unsqueeze(0)], dim=0) 
        sigma_list = torch.cat([sigma_1.unsqueeze(0), sigma_2.unsqueeze(0)], dim=0) 

        return x, mean_list, sigma_list 

    def reverse(self, x): 
        x = self.temporal_transformer.reverse(x) 

        _, x = x.split([1, x.shape[1]-1], dim=1) 

        b, t, d = x.shape 
        x = x.contiguous().view(b*t, 1, d)
        x = self.space_transformer.reverse(x) 

        _, n, d = x.shape 
        x = x.view(b, t, n, d) 
        
        x -= self.pos_embed 
        _, x = x.split([1, x.shape[2]-1], dim=2) 

        x = self.patch_embed.reverse(x)

        return x


# Dataset 
path = r"D:\Dataset\img_align_celeba"
batch_size = 16 
n_frames = 16
img_size = 96 #384
n_bits = 5 

# Model 
patch_size = 16
in_chans = 3
n_heads = 12
n_flow = 4 # 12 
mlp_ratio = 4.
qkv_bias = True

# Train 
lr = 1e-4
iteration = 200000
kld_weight = 1e-3 


# Dataset 
def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size,image_size)),
            transforms.CenterCrop((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=0
            )
            loader = iter(loader)
            yield next(loader)

dataset = iter(sample_data(path, batch_size, img_size))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") 

model = RevViT_GLOW(
                n_frames=n_frames, 
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                n_heads=n_heads,
                n_flow=n_flow, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            )

# net = nn.DataParallel(model).to(device)
net = model.to(device)

'''
# Optimizer 
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

def calc_loss(recons, image, mean, sigma): 
    kld_loss = torch.mean(-0.5 * torch.sum(1 + sigma - mean ** 2 - sigma.exp(), dim=1), dim=0) 
    recons_loss = F.l1_loss(recons, image) 
    loss = recons_loss + kld_weight * kld_loss 
    return loss, recons_loss.detach(), kld_loss.detach() 


# Train 
n_bins = 2.0 ** n_bits


with tqdm(range(iteration)) as pbar: 
    for i in pbar: 
        image, _ = next(dataset)
        image = image.to(device) 
        image = image * 255 

        if n_bits < 8: 
            image = torch.floor(image/2 ** (8-n_bits)) 
        
        image = image / n_bins - 0.5 

        image = image.unsqueeze(1) # Temp

        x, mean_list, sigma_list = net(image + torch.rand_like(image) / n_bins) 

        recons = net.reverse(x) 

        loss, recons_loss, kld_loss = calc_loss(recons, image, mean_list, sigma_list) 

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

        pbar.set_description(
            "Loss: {:.5f}, L1: {:.5f}, KL: {:.5f}".format(loss.item(), recons_loss.item(), kld_loss.item()) 
        )

        # if i % 100 == 0:
        #         with torch.no_grad():
        #             utils.save_image(
        #                 model.reverse(z_sample).cpu().data,
        #                 "sample/{}.png".format(str(i + 1).zfill(6)),
        #                 normalize=True,
        #                 nrow=10,
        #                 range=(-0.5, 0.5),
        #             ) # TODO: zsample

        if i % 10000 == 0:
            torch.save(
                model.state_dict(), "checkpoint/model_{}.pt".format(str(i + 1).zfill(6))
            )
            torch.save(
                optimizer.state_dict(), "checkpoint/optim_{}.pt".format(str(i + 1).zfill(6))
            )

'''