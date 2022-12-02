from tqdm import tqdm 
import numpy as np 
from PIL import Image 
from math import log

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils 
from math import log, pi, exp

# TODO: GELU not avail for 1.1
class GELU(nn.Module): 
    def forward(self, input): 
        return F.relu(input)


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
                x2, x1 = x.chunk(2, 1)
                y2 = x2 + self.attn(self.norm1(x1)) 
                y1 = x1 + self.mlp(self.norm2(y2)) 
                return torch.cat([y1, y2], 1) 

            def reverse(self, y): 
                y1, y2 = y.chunk(2, 1) 
                x1 = y1 - self.mlp(self.norm2(y2)) 
                x2 = y2 - self.attn(self.norm1(x1)) 
                return torch.cat([x2, x1], 1) 

        # class ZeroConv2d(nn.Module):
        #     def __init__(self, in_channel, out_channel, padding=1):
        #         super().__init__()

        #         self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        #         self.conv.weight.data.zero_()
        #         self.conv.bias.data.zero_()
        #         self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

        #     def forward(self, input):
        #         out = F.pad(input, [1, 1, 1, 1], value=1)
        #         out = self.conv(out)
        #         out = out * torch.exp(self.scale * 3)

        #         return out

        def _gaussian_log_p(self, x, mean, log_sd):
            '''
            f(x)=1./[std*(2*pi)^0.5]*exp{-0.5*[(x-mean)/std]^2}
            log f(x)=-log[std*(2*pi)^0.5]-0.5*[(x-mean)/std]^2
                    =-0.5*log(2*pi)-log_std-0.5*(x-mean)^2/[exp(log_std)]^2
                    =-0.5*log(2*pi)-log_std-0.5*(x-mean)^2/[exp(2*log_std)]
            '''
            return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

        def _gaussian_sample(self, eps, mean, log_sd):
            return mean + torch.exp(log_sd) * eps

        def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0., split=True):
            super().__init__()

            dim //= 2

            self.split = split
            self.flow = self.Transformer(
                dim=dim, 
                n_heads=n_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                p=p, 
                attn_p=attn_p 
            )

            # n = 1 if split else 2 
            # self.prior = self.ZeroConv2d(in_channel * 2*n, in_channel * 4*n)

            self.norm = nn.LayerNorm(dim, eps=1e-6) 
            self.prior = nn.Linear(dim, dim * 2)  

            # TODO
            self.head_norm = nn.LayerNorm(dim, eps=1e-6)
            self.head = nn.Linear(dim, dim)

        def forward(self, x):
            n_samples, n_patches, embed_dim = x.shape 
            squeezed = x.view(n_samples, n_patches, embed_dim // 2, 2)
            squeezed = squeezed.permute(0, 1, 3, 2)  # n_samples, n_patches, 2, embed_dim // 2
            x = squeezed.contiguous().view(n_samples, n_patches * 2, embed_dim // 2)

            # x = torch.cat([x, x], -1) # TODO
            x = self.flow(x) 

            if self.split:
                x, z_new = x.chunk(2, 1) 
                # x, z_new = x.split([n_patches-1, 1], dim=1)

                mean, log_sd = self.prior( self.norm(x) ).chunk(2, -1) # TODO
                log_p = self._gaussian_log_p(z_new, mean, log_sd) 
                log_p = log_p.view(n_samples, -1).sum(1) 

                # z_new = self.head_norm(z_new) 
                # cls_token_final = z_new[:, 0]  # just the CLS token
                # z_new = self.head(cls_token_final)
            else: 
                zero = torch.zeros_like(x) 
                mean, log_sd = self.prior( self.norm(zero) ).chunk(2, -1) 
                log_p = self._gaussian_log_p(x, mean, log_sd) 
                log_p = log_p.view(n_samples, -1).sum(1) 

                x = self.head_norm(x) 
                cls_token_final = x[:, 0]  # just the CLS token
                x = self.head(cls_token_final)

                z_new = x 

            return x, log_p, z_new

        def reverse(self, x, eps=None, reconstruct=False):
            if reconstruct:
                if self.split:
                    x = torch.cat([x, eps], 1) 
                else:
                    x = eps

            else:
                if self.split:
                    # TODO: ????
                    # mean, log_sd = self.prior( self.norm(x) ).chunk(2, -1)
                    # z = self._gaussian_sample(eps, mean, log_sd)
                    # x = torch.cat([x, z], 1)

                    x = torch.cat([x, eps], 1)

                else:
                    # TODO: ???? 
                    # zero = torch.zeros_like(input)
                    # mean, log_sd = self.prior( self.norm(zero) ).chunk(2, -1)
                    # z = self._gaussian_sample(eps, mean, log_sd)
                    # x = z

                    x = eps 

            x = self.flow.reverse(x) 

            n_samples, n_patches_, embed_dim_ = x.shape 
            unsqueezed = x.view(n_samples, n_patches_ // 2, 2, embed_dim_)
            unsqueezed = unsqueezed.permute(0, 1, 3, 2)
            unsqueezed = unsqueezed.contiguous().view(n_samples, n_patches_ // 2, embed_dim_ * 2)

            return unsqueezed

    class PatchEmbed(nn.Module):
        # Split image into patches and then embed them.
        def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.n_patches = (img_size // patch_size) ** 2

            self.proj = nn.Conv2d(
                    in_chans,
                    embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
            )

        def forward(self, x):
            x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
            x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
            x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)
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
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()
        self.patch_embed = self.PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        ) # Learnable parameter that will represent the first token in the sequence.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList()
        dim = embed_dim 
        for i in range(depth - 1):
            self.blocks.append(
                self.Block(
                    dim=dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p 
                )
            )
            dim //= 2

        self.blocks.append(
            self.Block(
                dim=dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p,
                split=False
            )
        )

    def forward(self, x): 
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim) #TODO: Maybe change global pos token to multi token per layer
        x = self.pos_drop(x)

        log_p_sum = 0 
        z_outs = list() 

        for block in self.blocks:
            x, log_p, z_new = block(x)
            z_outs.append(z_new) 

            if log_p is not None: 
                log_p_sum += log_p 

        logdet = 0 
        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False): 
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)
            else:
                x = block.reverse(x, z_list[-(i + 1)], reconstruct=reconstruct)

        return x


# Dataset 
path = r"D:\Dataset\archive\maps\maps"
batch_size = 4 #16 
img_size = 32 #64 
n_bits = 5 
n = 0 

# Model 
n_flow = 32 
n_block = 4 #4
use_affine = False 
use_lu = True 

# Train 
lr = 1e-4
iteration = 200000


# Dataset 
def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size,image_size*2)),
            transforms.CenterCrop((image_size,image_size*2)),
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

model = RevViT_GLOW(img_size=96, #384,
                    patch_size=16,
                    in_chans=3,
                    embed_dim=768,
                    depth=4, #12, # n_block
                    n_heads=12,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    p=0.,
                    attn_p=0.
                )

# net = nn.DataParallel(model).to(device)
net = model.to(device)

'''
# Optimizer 
optimizer = optim.Adam(net.parameters(), lr=lr)

def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel # loss considering noise inputted. 
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


# Train 
n_bins = 2.0 ** n_bits

# z_sample = []
# z_shapes = calc_z_shapes(3, img_size, args.n_flow, args.n_block)
# for z in z_shapes:
#     z_new = torch.randn(args.n_sample, *z) * args.temp
#     z_sample.append(z_new.to(device)) # TODO

with tqdm(range(iteration)) as pbar: 
    for i in pbar: 
        image, _ = next(dataset)
        image = image.to(device) 
        image = image * 255 

        if n_bits < 8: 
            image = torch.floor(image/2 ** (8-n_bits)) 
        
        image = image / n_bins - 0.5 

        x = image[:,:,:,:img_size].contiguous().clone().detach().requires_grad_(True) 
        y = image[:,:,:,img_size:].contiguous().clone().detach().requires_grad_(True)

        # if i == 0:
        #     with torch.no_grad():
        #         log_p, logdet, y_pred = net.module(
        #             x + torch.rand_like(x) / n_bins 
        #         )

        #         continue

        # else:
        log_p, logdet, y_pred = net(x + torch.rand_like(x) / n_bins)  

        logdet = logdet.mean()

        loss1, log_p, log_det = calc_loss(log_p, logdet, img_size, n_bins)

        loss = loss1 
        net.zero_grad()
        loss.backward() 

        _lr = lr # * min(1, i * batch_size / (50000 * 10))
        optimizer.param_groups[0]["lr"] = _lr
        optimizer.step() 

        pbar.set_description(
            # "BPDLoss: {:.5f}, L1Loss: {:.5f}".format(loss1.item(), loss2.item()) 
            "BPDLoss: {:.5f}".format(loss1.item()) # Temp 
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