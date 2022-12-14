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

from embeddings import ImagePatchEmbed 
from revvit import RevViT, RevViTRev


# Dataset 
path = r"D:\Dataset\img_align_celeba"
batch_size = 1 
n_frames = 64 #128 #16
img_size = 128 #192 #384
n_bits = 5 

# Model 
patch_size = 16
in_chans = 3
n_heads = 12
layer_factor = 3 #6 #12 
block_size = 8 #16 #64 
mlp_ratio = 2. #4.
qkv_bias = True
k = 8 

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


# Utils 
def save_image(image, name="temp.png"): 
    utils.save_image(
            image,
            name,
            normalize=True,
            nrow=10,
            range=(-0.5, 0.5),
        )

# Model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") 

model = RevViT(
                batch_size=batch_size, 
                n_frames=n_frames, 
                img_size=img_size,
                patch_size=patch_size,
                PatchEmbed=ImagePatchEmbed, 
                in_chans=in_chans,
                n_heads=n_heads,
                layer_factor=layer_factor, 
                block_size=block_size, 
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                k=k, 
                device=device, 
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