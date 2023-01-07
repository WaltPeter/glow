from tqdm import tqdm 
import numpy as np 
import math

import torch 
from torch import optim
import torch.nn.functional as F
from torchvision import utils 

from config import * 
from mlp import Model  
from dataloader import S4Dataset 


# Utils 
def save_image(image, name="temp.png"): 
    utils.save_image(
            image,
            name,
            normalize=True,
            nrow=10,
            range=(-0.5, 0.5),
        )


# Dataset 
train_dataset = S4Dataset("train")
train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True, 
                    )

val_dataset = S4Dataset("val")
val_dataloader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True, 
                    )


# Model 
model = Model(
            n_frames=n_frames, 
            image_size=img_size, 
            patch_res=patch_res, 
            mlp_dim_D_S=mlp_dim_D_S, 
            mlp_dim_D_C=mlp_dim_D_C, 
            n_layers=n_layers, 
            lite=lite, 
        )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") 

# net = nn.DataParallel(model).to(device)
net = model.to(device)


# Optimizer 
optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay) 

# Train 
min_loss = math.inf 
with tqdm(range(n_epoches)) as pbar1: 
    for epoch in pbar1: 
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader)) as pbar2: 
            for n_iter, batch_data in pbar2: 
                imgs, wavs = batch_data 
                imgs = imgs.to(device) 
                wavs = wavs.to(device) 
                y = model(imgs) 
                loss = F.mse_loss(y, wavs) 

                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step()

                pbar2.set_description(
                    "Loss: {:.5f}".format(loss.item()) 
                )
        print() 

        # Validation
        model.eval()
        with torch.no_grad():
            with tqdm(enumerate(val_dataloader), total=len(val_dataloader)) as pbar2: 
                l_list = list() 
                for n_iter, batch_data in pbar2: 
                    imgs, wavs = batch_data 
                    imgs = imgs.to(device) 
                    wavs = wavs.to(device) 
                    y = model(imgs) 
                    loss = F.mse_loss(y, wavs) 
                    l_list.append(loss.item()) 

                    pbar2.set_description(
                        "ValLoss: {:.5f}".format(loss.item()) 
                    ) 
            l = np.mean(l_list) 
            if l < min_loss: 
                torch.save(model.state_dict(), "checkpoint/best_{}.pt".format(epoch)) 
                torch.save(optimizer.state_dict(), "checkpoint/optim_{}.pt".format(epoch)) 
                min_loss = l 

        print()  

'''
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