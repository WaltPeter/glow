import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import * 


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split="train"):
        super(S4Dataset, self).__init__()
        self.split = split
        df_all = pd.read_csv(path_meta_csv, sep=",")
        self.df_split = df_all[df_all["split"] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), 
            transforms.ToTensor(), 
        ])

    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path =  os.path.join(path_img, self.split, category, video_name)
        wav_base_path = os.path.join(path_audio_log_mel_img, self.split, category, video_name)
        imgs, wavs = list(), list() 
        for img_id in range(1, 6):
            img = self._load_image_in_PIL_to_Tensor(
                    os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), 
                    transform=self.img_transform
                )
            imgs.append(img)
            wav = self._load_image_in_PIL_to_Tensor(
                    os.path.join(wav_base_path, "%s_%d.png"%(video_name, img_id)), 
                    transform=self.img_transform
                ) 
            wavs.append(wav) 
        imgs_tensor = torch.stack(imgs, dim=0)
        wavs_tensor = torch.stack(wavs, dim=0) 

        return imgs_tensor, wavs_tensor 

        # if self.split == 'train':
        #     return imgs_tensor, audio_log_mel, masks_tensor
        # else:
        #     return imgs_tensor, audio_log_mel, masks_tensor, category, video_name

    def __len__(self):
        return len(self.df_split)

    def _load_image_in_PIL_to_Tensor(self, path, mode="RGB", transform=None):
        img_PIL = Image.open(path).convert(mode)
        if transform:
            img_tensor = transform(img_PIL)
            return img_tensor
        return img_PIL