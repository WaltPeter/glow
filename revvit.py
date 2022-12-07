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

                def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0):
                    super().__init__()
                    self.n_heads = n_heads 
                    self.dim = dim 
                    self.head_dim = dim // n_heads 
                    self.scale = self.head_dim ** -0.5 

                    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
                    self.attn_drop = nn.Dropout(attn_p)
                    self.proj = nn.Linear(dim, dim)
                    self.proj_drop = nn.Dropout(proj_p) 
                
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
                    attn = self.attn_drop(attn)

                    weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
                    weighted_avg = weighted_avg.transpose(
                            1, 2
                    )  # (n_samples, n_patches + 1, n_heads, head_dim)
                    weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

                    x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
                    x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

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

            def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim, eps=1e-6)
                self.attn = self.Att(
                        dim=dim,
                        n_heads=n_heads,
                        qkv_bias=qkv_bias,
                        attn_p=attn_p,
                        proj_p=p
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

        def __init__(self, dim, n_heads, n_flow, n_patches, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0., split_sz=4, split=True):
            super().__init__()
            self.split = split 
            self.split_sz = split_sz 

            self.flows = nn.ModuleList(
                [
                    self.Transformer(
                        dim=dim // 2, 
                        n_heads=n_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        p=p, 
                        attn_p=attn_p 
                    )
                    for _ in range(n_flow)
                ]
            )

            latent_dim = 16*16
            self.norm = nn.LayerNorm(split_sz*dim, eps=1e-6) 
            self.prior = nn.Linear(split_sz*dim, latent_dim*2)  

        def forward(self, x):
            n_samples, n_patches, embed_dim = x.shape 
            x = x.view(n_samples, n_patches, embed_dim // 2, 2)

            for flow in self.flows: 
                x = flow(x) 

            x = x.view(n_samples, n_patches, embed_dim)

            if self.split: 
                z_new, x = x.split([self.split_sz, x.shape[1]-self.split_sz], dim=1) # Split out the CLS token
            else: 
                z_new = x 

            flatten = torch.flatten(z_new, start_dim=1) 
            mean, sigma = self.prior( self.norm(flatten) ).chunk(2, -1) 

            return x, mean, sigma, z_new 

        def reverse(self, x, z=None):
            if self.split: 
                x = torch.cat([z,x], 1) 

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
            self.img_size = img_size
            self.patch_size = patch_size
            self.n_patches = (img_size // patch_size) ** 2
            self.embed_dim = embed_dim
            self.in_chans = in_chans 

        def forward(self, x):
            n_split = int(self.n_patches ** 0.5) 
            x = torch.cat(
                [b.contiguous().view(x.shape[0], self.embed_dim, 1) for a in x.chunk(n_split, dim=2) \
                    for b in a.chunk(n_split, dim=3)], 
                dim=-1) 

            x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)
            return x

        def reverse(self, x): 
            n_split = int(self.n_patches ** 0.5) 
            n_samples = x.shape[0]

            x = x.view(n_samples, n_split, n_split, self.in_chans, self.patch_size, self.patch_size)\
                .permute(0,3,1,4,2,5).contiguous()\
                    .view(n_samples, self.in_chans, self.img_size, self.img_size)
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
            img_size=384,
            patch_size=16,
            in_chans=3,
            n_heads=12,
            n_flow=12, 
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
            split_sz=4, 
    ):
        n_patches = (img_size // patch_size) ** 2
        dim = (img_size ** 2) * in_chans // n_patches 
        depth = int(1 / split_sz * n_patches + 1) 

        print("depth: {}, n_patches: {}+{}".format(depth, split_sz, n_patches)) 

        super().__init__()

        self.patch_embed = self.PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=dim
        )

        # Learnable parameter that will represent the first token in the sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, split_sz, dim)) 

        # Position embedding. 
        self.pos_embed = nn.Parameter(torch.zeros(1, split_sz + n_patches, dim)) 

        n_patches += split_sz
        self.split_sz = split_sz 

        self.blocks = nn.ModuleList()
        for i in range(depth-1):
            self.blocks.append(
                self.Block(
                    dim=dim,
                    n_flow=n_flow, 
                    n_heads=n_heads,
                    n_patches=n_patches, 
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p, 
                    split_sz=split_sz 
                )
            )
            n_patches -= split_sz

        self.blocks.append(
            self.Block(
                dim=dim,
                n_flow=n_flow, 
                n_heads=n_heads,
                n_patches=n_patches, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p, 
                split_sz=split_sz, 
                split=False 
            )
        )

    def forward(self, x): 
        x = self.patch_embed(x)
        mean_list = list() 
        sigma_list = list() 
        z_outs = list() 

        n_samples = x.shape[0]
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, split_sz, embed_dim)

        x = torch.cat((cls_token, x), dim=1)  # (n_samples, split_sz + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, split_sz + n_patches, embed_dim) 

        for block in self.blocks:
            x, mean, sigma, z_new = block(x)
            mean_list.append(mean.unsqueeze(1)) 
            sigma_list.append(sigma.unsqueeze(1)) 
            z_outs.append(z_new.unsqueeze(1)) 

        mean_list = torch.cat(mean_list, dim=1) 
        sigma_list = torch.cat(sigma_list, dim=1) 
        z_outs = torch.cat(z_outs, dim=1) 

        return mean_list, sigma_list, z_outs 

    def reverse(self, z_list): 

        z_list = [i.squeeze(1) for i in z_list.chunk(z_list.shape[1], 1)]
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0: 
                x = block.reverse(z_list[-(i + 1)]) 
            else: 
                x = block.reverse(x, z_list[-(i + 1)]) 
        
        x = x - self.pos_embed 
        _, x = x.split([self.split_sz, x.shape[1]-self.split_sz], dim=1)

        x = self.patch_embed.reverse(x)

        return x


# Dataset 
path = r"D:\Dataset\img_align_celeba"
batch_size = 16 
img_size = 32 #64 
n_bits = 5 
n = 0 

# Model 
img_size = 96 #384
patch_size = 16
in_chans = 3
n_heads = 12
n_flow = 2 # 12 
mlp_ratio = 4.
qkv_bias = True
p = 0.
attn_p = 0.

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
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=n)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=n
            )
            loader = iter(loader)
            yield next(loader)

dataset = iter(sample_data(path, batch_size, img_size))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") 

model = RevViT_GLOW(img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    n_heads=n_heads,
                    n_flow=n_flow, 
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
            )

# net = nn.DataParallel(model).to(device)
net = model.to(device)


# Optimizer 
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

def calc_loss(recons, image, mean, sigma): 
    mean, sigma = [i.view(i.shape[0], i.shape[1] * i.shape[2]) for i in (mean, sigma)] 
    kld_loss = torch.mean(-0.5 * torch.sum(1 + sigma - mean ** 2 - sigma.exp(), dim=1), dim=0) 
    recons_loss = F.l1_loss(recons, image) 
    loss = recons_loss + kld_weight * kld_loss 
    return loss, recons_loss.detach(), kld_loss.detach() 


# Train 
n_bins = 2.0 ** n_bits

'''
with tqdm(range(iteration)) as pbar: 
    for i in pbar: 
        image, _ = next(dataset)
        image = image.to(device) 
        image = image * 255 

        if n_bits < 8: 
            image = torch.floor(image/2 ** (8-n_bits)) 
        
        image = image / n_bins - 0.5 

        mean_list, sigma_list, z_outs = net(image + torch.rand_like(image) / n_bins) 

        with torch.no_grad(): 
            recons = net.reverse(z_outs) 

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