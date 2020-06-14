import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch import utils
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
# import torchsummary
from torch.optim import lr_scheduler
import numpy as np
import random
from os.path import *
from os import listdir
from tqdm import tqdm
from ResNet import resnet50
# from conv import CustomCNN
import copy
from sklearn.preprocessing import normalize
import csv

torch.manual_seed(123)
import torch.nn as nn

#####################################
train_dataset = ImageFolder(root='corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train', transform=transforms.Compose([
                               transforms.Resize(128),       # 128 on one dimension
                               transforms.CenterCrop(128),  # square
                               transforms.ToTensor(),       # Tensor (= 0~1 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s
                           ]))
test_dataset = ImageFolder(root='corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test' , transform=transforms.Compose([
                               transforms.Resize(128),       # 128 on one dimension
                               transforms.CenterCrop(128),  # square
                               transforms.ToTensor(),       # Tensor (= 0~1 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s
                           ]))

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
for i,data in enumerate(train_loader):
    print("{},{}".format(i,data[1]))

