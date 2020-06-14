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
import torchvision.models as models
from PIL import Image
# from torchsummary import summary
from torch.optim import lr_scheduler
import numpy as np
import random
from os.path import *
from os import listdir
from tqdm import tqdm
from ResNet import resnet50, resnet18, resnet34
# from conv import CustomCNN
import copy
from sklearn.preprocessing import normalize
import csv


np.random.seed(0)
torch.manual_seed(0)

#####################################
train_dataset = ImageFolder(root='corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train', transform=transforms.Compose([
                               transforms.Resize(224),       # 128 on one dimension
                               transforms.CenterCrop(224),  # square
                               transforms.ToTensor(),       # CxHxW FloatTensor (= 0~1 normalize automatically)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s
                           ]))
test_dataset = ImageFolder(root='corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test' , transform=transforms.Compose([
                               transforms.Resize(224),       # 128 on one dimension
                               transforms.CenterCrop(224),  # square
                               transforms.ToTensor(),       # Tensor (= 0~1 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s
                           ]))

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)
print("==> DATA LOADED")

#visualize
# single_batch = next(iter(train_loader))
# single_img = single_batch[0][0]
# # t_img = single_img.view(single_img.shape[1], single_img.shape[2], single_img.shape[0])
# batch_grid = utils.make_grid(single_batch[0], nrow=4)
# plt.figure(figsize= (10,10))
# plt.imshow(batch_grid.permute(1,2,0))
# plt.show()
# print("==> first batch")

######################################################
model = models.resnet18()
# model = resnet18(3, 2) #3 channels & 2 classes
# model = resnet34(3,2)
print("==> MODEL LOADED")

#hyperparams
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
num_epochs = 20
num_batches = len(train_loader)

#use GPU
device = torch.device('cuda:0')
model = model.to(device)
criterion = criterion.to(device)


trn_loss_list = []
tst_loss_list = []
trn_acc_list = []
tst_acc_list = []
for epoch in tqdm(range(num_epochs)):
    #data for each epoch
    trn_loss = 0.0
    trn_correct = 0
    trn_total = 0
    for i,trainset in enumerate(train_loader): #2680 / batch_size = # of iterations
        model.train()

        train_in, train_out = trainset
        train_in, train_out = train_in.to(device), train_out.to(device)

        optimizer.zero_grad()

        train_pred = model(train_in) #logits

        _, label_pred = torch.max(train_pred, 1) #value , label
        trn_total += train_out.size(0) #20
        trn_correct += (label_pred == train_out).sum().item() #works

        t_loss = criterion(train_pred, train_out)
        t_loss.backward()
        optimizer.step()

        trn_loss += t_loss.item()

        test_term = 20
        if(i+1)%test_term == 0: #after 20 updates
            model.eval()
            with torch.no_grad(): #for validation!
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                for j, testset in enumerate(test_loader):
                    test_in, test_out = testset
                    test_in, test_out = test_in.to(device), test_out.to(device)

                    test_pred = model(test_in)

                    _, test_label_pred = torch.max(test_pred.data,1)
                    test_total += test_out.size(0)
                    test_correct += (test_label_pred == test_out).sum().item()

                    v_loss = criterion(test_pred, test_out)
                    test_loss += v_loss.item()


            print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}% | test loss: {:.4f} | test acc: {:.2f}%".format(
                epoch+1, num_epochs,
                i+1, len(train_loader),
                trn_loss / test_term, #accumulated over term -> then reset
                100*(trn_correct / trn_total),
                test_loss / len(test_loader),
                100*(test_correct / test_total)
            ))

            trn_loss_list.append(trn_loss / test_term)
            tst_loss_list.append(test_loss / len(test_loader))
            trn_acc_list.append(100 * (trn_correct / trn_total))
            tst_acc_list.append(100 * (test_correct / test_total))


            #reinitialize
            trn_loss = 0.0
            trn_total = 0
            trn_correct = 0


# Summarize history for accuracy
plt.plot(trn_acc_list)
plt.plot(tst_acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('mini-batch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(trn_loss_list)
plt.plot(tst_loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('mini-batch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()