from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from os.path import *
import csv
from os import listdir, path
import shutil
import os

train_dir = 'corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'
test_dir = 'corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'
info_dir = 'corona_dataset/Chest_xray_Corona_Metadata.csv'

# meta_f = open(info_dir)
# rdr = csv.reader(meta_f)
# for i,line in enumerate(rdr):
#     if (i==0):continue
#     src_path = train_dir
#     if (line[3] == 'TEST'):
#         src_path = test_dir
#     isCorona = 0
#     if (line[2] == 'Pnemonia'):
#         isCorona = 1
#     shutil.move(src_path + line[1], src_path + str(isCorona) + '/' + line[1])
#     print('Saved : {} - {} - {}'.format(line[3], isCorona ,line[1]))
#
# meta_f.close()

# for folder in os.listdir(train_dir):
#     for i, file in enumerate(os.listdir(train_dir + folder)):
#         if(i >= 1340):
#             shutil.move(train_dir + folder +'/'+ file, 'corona_dataset/leftover/train/' +  folder + '/'+file)
#         print('corona_dataset/leftover/' + folder + '/'+file)
# for folder in os.listdir(test_dir):
#     for i, file in enumerate(os.listdir(test_dir + folder)):
#         if(i >= 230):
#             shutil.move(test_dir + folder +'/'+ file, 'corona_dataset/leftover/test/' + folder + '/'+file)