import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch import utils
import torchvision
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
import torchvision.models as md
from PIL import Image
# from torchsummary import summary
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


np.random.seed(0)
torch.manual_seed(0)

#####################################
train_dataset = ImageFolder(root='corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train', transform=transforms.Compose([
                               transforms.Resize(128),       # 128 on one dimension
                               transforms.CenterCrop(128),  # square
                               transforms.ToTensor(),       # CxHxW FloatTensor (= 0~1 normalize automatically)
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
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
print("==> DATA LOADED")
# for i,data in enumerate(train_loader):
#     print("{},{}".format(i,data[1]))

single_batch = next(iter(train_loader))
batch_grid = utils.make_grid(single_batch[0], nrow=4)
plt.figure(figsize= (10,10))
plt.imshow(batch_grid.permute(1,2,0))
######################################################

# model = resnet50(3, 2) #3 channels & 2 classes
# print("==> MODEL LOADED")
#
# #hyperparams
# criterion = nn.BCELoss()
#
#
# trn_loss_list = []
# tst_loss_list = []
# trn_acc_list = []
# tst_acc_list = []
#
# def
