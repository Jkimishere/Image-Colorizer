import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision.transforms as transforms

from PIL import Image

import os
import os.path
import numpy as np
class Cal256Loader(data.Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.imgfolder = os.listdir(data_dir)
        self.imgs = []
        for fol in self.imgfolder:
            dir = os.path.join('../data', fol)
            for f in os.listdir(dir):  
                self.imgs.append(os.path.join(dir, f))
        import random
        random.shuffle(self.imgs)
        self.transform = transform

        if split == 'train':
            self.imgs = self.imgs[0:int(len(self.imgs) * 0.9)]
        if split == 'test':
            self.imgs = self.imgs[int(len(self.imgs) * 0.9):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        x = self.imgs[index]
        y = x
        x = Image.open(x).convert('RGB')
        y = Image.open(y).convert('L')
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)


        return x, y 


