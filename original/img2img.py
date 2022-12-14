from tqdm import tqdm 
import numpy as np 
from PIL import Image 
from math import log

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from original.model import Glow 

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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") 

# Model 
class InvGlow(Glow): 
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        Glow.__init__(self, in_channel, n_flow, n_block, affine=affine, conv_lu=conv_lu) 
        _f = self.forward 
        self.forward = self.reverse 
        self.reverse = _f 

class Model(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True): 
        super().__init__()
        self.g = Glow(in_channel, n_flow, n_block, affine=affine, conv_lu=conv_lu) 
        self.h = InvGlow(in_channel, n_flow, n_block, affine=affine, conv_lu=conv_lu) 
    
    def forward(self, x):
        log_p, logdet, z = self.g(x)
        return log_p, logdet, self.h(z)
    
    def reverse(self, y): 
        log_p, logdet, z = self.h.reverse(y)
        return log_p, logdet, self.g.reverse(z)

model = Model(3, n_flow, n_block, affine=use_affine, conv_lu=use_lu)
# net = nn.DataParallel(model).to(device)
net = model.to(device)


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
l1_loss = nn.L1Loss().to(device)

n_bins = 2.0 ** n_bits

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

        loss2 = l1_loss(y_pred, x) 
        loss = loss1 + loss2 
        net.zero_grad()
        loss.backward() 

        _lr = lr # * min(1, i * batch_size / (50000 * 10))
        optimizer.param_groups[0]["lr"] = _lr
        optimizer.step() 

        pbar.set_description(
            "BPDLoss: {:.5f}, L1Loss: {:.5f}".format(loss1.item(), loss2.item()) 
            # "BPDLoss: {:.5f}".format(loss1.item()) # Temp 
        )

        if i % 100 == 0:
            with torch.no_grad():
                utils.save_image(
                    torch.tensor(np.concatenate((x.cpu().data, y_pred.cpu().data), axis=3)).data,
                    # x.cpu().data, # Temp 
                    "sample/{}.png".format(str(i + 1).zfill(6)),
                    normalize=True,
                    nrow=10,
                    range=(-0.5, 0.5),
                )

        if i % 10000 == 0:
            torch.save(
                net.state_dict(), "checkpoint/model_{}.pt".format(str(i + 1).zfill(6))
            )
            torch.save(
                optimizer.state_dict(), "checkpoint/optim_{}.pt".format(str(i + 1).zfill(6))
            )