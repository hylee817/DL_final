import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.optim import lr_scheduler
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
from CNN2 import CustomCNN
import copy
from sklearn.preprocessing import normalize
import csv
from CNN import CNNModel


np.random.seed(123)
torch.manual_seed(123)
img_size = 255
#####################################

train_dataset = ImageFolder(root='corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train', transform=transforms.Compose([
                               transforms.Resize(img_size),       # 128 on one dimension
                               transforms.CenterCrop(img_size),  # square
                               transforms.ToTensor(),       # CxHxW FloatTensor (= 0~1 normalize automatically)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s
                           ]))
valid_dataset = ImageFolder(root='corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test' , transform=transforms.Compose([
                               transforms.Resize(img_size),       # 128 on one dimension
                               transforms.CenterCrop(img_size),  # square
                               transforms.ToTensor(),       # Tensor (= 0~1 normalize)
                               transforms.Normalize((0.5, 0.5, 0.5),  # -1 ~ 1 normalize
                                                    (0.5, 0.5, 0.5)), # (c - m)/s
                           ]))


batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
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
# model = models.resnet18() #predefined model. try not to use.
model = resnet18(3, 1) #3 channels & 2 classes
# model = resnet34(3,2)
# model = CNNModel()
# model = CustomCNN((batch_size, 3, 255, 255) , 1)
print("==> MODEL LOADED")
print(model)

#hyperparams
num_epochs = 20
num_batches = len(train_loader)
criterion = nn.BCEWithLogitsLoss()
learning_rate = 0.001
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = None
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.8,patience=10,verbose=True)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


#use GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)
print("==> MODEL ON GPU: {}".format(device))

#binary
def get_bin_label(y_pred, y_test): # [-7.8995]...
    y_pred_tag = torch.round(torch.sigmoid(y_pred)) #sigmoid -> 0-1 -> round -> 0 or 1
    corrects = (y_pred_tag == y_test).sum()
    # acc = corrects / y_test.shape[0] # num of corrects / # total items
    # acc = torch.round(acc)

    return y_pred_tag, corrects




#print hyper params
print("\n\n#########################################")
print("hyper params: \n BATCH_SIZE>> {} \n LOSS_F>> {} \n OPT>> {} \n SCHEDULER>> {} \n IMG_SIZE>> {} \n DATA(train)>> {}\n DATA(test)>> {}".format(
    batch_size,
    criterion,
    optimizer,
    scheduler,
    img_size,
    len(train_loader),
    len(valid_loader)
))
print("#########################################\n\n")





###########################################################################################
trn_loss_list = []
val_loss_list = []
trn_acc_list = []
val_acc_list = []
for epoch in tqdm(range(num_epochs)):
    print("")
    #data for each epoch
    trn_loss = 0.0
    trn_correct = 0
    trn_total = 0
    for i,trainset in enumerate(train_loader): #2680 / batch_size = # of iterations
        model.train()

        train_in, train_out = trainset
        train_in, train_out = train_in.to(device), train_out.to(device)

        optimizer.zero_grad()

        #for binary
        train_out = train_out.unsqueeze(1)
        train_pred = model(train_in) #logits (binary: > 0 or < 0)
        label_pred, t_correct = get_bin_label(train_pred, train_out)
        trn_correct += t_correct.item()
        #for 2classes
        # _, label_pred = torch.max(train_pred, 1) #value , label
        trn_total += train_out.size(0) #20
        # trn_correct += (label_pred == train_out).sum().item() #works

        #calculate loss
        #for bceloss
        train_out = train_out.type_as(train_pred)
        #for both
        t_loss = criterion(train_pred, train_out)
        t_loss.backward()
        optimizer.step()

        trn_loss += t_loss.item()



        valid_term = 20
        if(i+1)%valid_term == 0: #after 20 updates
            model.eval()
            with torch.no_grad(): #for validation!
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                for j, validset in enumerate(valid_loader):
                    valid_in, valid_out = validset
                    valid_in, valid_out = valid_in.to(device), valid_out.to(device)

                    #for binary
                    valid_out = valid_out.unsqueeze(1)
                    val_pred = model(valid_in)  # logits (binary: > 0 or < 0)
                    val_label_pred, v_correct = get_bin_label(val_pred, valid_out)
                    val_correct += v_correct.item()
                    #for 2classes
                    # _, val_label_pred = torch.max(val_pred.data,1)
                    val_total += valid_out.size(0)
                    # val_correct += (val_label_pred == valid_out).sum().item()

                    #calculate loss
                    #for bce loss
                    valid_out = valid_out.type_as(val_pred)
                    #for both
                    v_loss = criterion(val_pred, valid_out)
                    val_loss += v_loss.item()

                    # scheduler.step(v_loss)
                    # scheduler.step()
                    lr = optimizer.param_groups[0]['lr']


            print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}% | test loss: {:.4f} | test acc: {:.2f}% | lr: {:.6f}".format(
                epoch+1, num_epochs,
                i+1, len(train_loader),
                trn_loss / valid_term, #accumulated over term -> then reset
                100*(trn_correct / trn_total),
                val_loss / len(valid_loader),
                100*(val_correct / val_total),
                lr
            ))

            trn_loss_list.append(trn_loss / valid_term)
            val_loss_list.append(val_loss / len(valid_loader))
            trn_acc_list.append(100 * (trn_correct / trn_total))
            val_acc_list.append(100 * (val_correct / val_total))


            #reinitialize
            trn_loss = 0.0
            trn_total = 0
            trn_correct = 0


######################################################################################

# Summarize history for accuracy
plt.plot(trn_acc_list)
plt.plot(val_acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('mini-batch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(trn_loss_list)
plt.plot(val_loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('mini-batch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()