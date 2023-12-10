from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pdb
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
import collections
import os

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose(
                        [#transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
                         #transforms.RandomHorizontalFlip(),
                         CIFAR10Policy(),
			             transforms.ToTensor(),
                         #Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                         #transforms.Normalize(...)
                         #transforms.Grayscale(num_output_channels=1),
                         ]))

#batch_size=10
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size)

imgs, count = [], 0
file_count = collections.defaultdict(int)
path = os.path.join(os.getcwd(),'autoaug')
if not os.path.exists(path):
    os.makedirs(path)
for img in train_loader:
    item = img[1].item()
    img = np.transpose(img[0][0].numpy() * 255, (1, 2, 0)).astype(np.uint8)
    if file_count[str(item)] < 1200:
        file_count[str(item)] += 1
        plt.axis("off")
        subpath = os.path.join(path,str(item))
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        plt.imshow(img)
        plt.savefig(os.path.join(subpath, str(file_count[str(item)]) + '.png'), bbox_inches='tight', pad_inches=0.0)
        # save image,update number
    else:
        pass

    if all(i >= 1200 for i in list(file_count.values())):
        break
