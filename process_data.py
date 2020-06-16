from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from os.path import *
import csv
# from os import listdir, path
import shutil
import os

data_dir = 'corona_dataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/'
info_dir = 'corona_dataset/Chest_xray_Corona_Metadata.csv'


#create directories
paths = [data_dir + "train/1", data_dir + "train/0", data_dir + "test/0", data_dir + "test/1"]
for p in paths:
    os.makedirs(p, exist_ok=True)


meta_f = open(info_dir)
rdr = csv.reader(meta_f)

#move img to matching folders
count_1 = 0
count_0 = 0
for i,line in enumerate(rdr):
    if (i==0):continue #title row
    src_path = data_dir #default = train
    if(line[2] == 'Pnemonia'):
        if(count_1 >= 1250):
            if(count_1 < 1570):
                shutil.move(src_path + line[1], src_path + 'test/1/' + line[1])
        else:
            shutil.move(src_path + line[1], src_path + 'train/1/' + line[1])
        count_1 += 1
    elif(line[2] == 'Normal'):
        if(count_0 >= 1250):
            if(count_0 < 1570):
                shutil.move(src_path + line[1], src_path + 'test/0/' + line[1])
        else:
            shutil.move(src_path + line[1], src_path + 'train/0/' + line[1])
        count_0 += 1
    print('{} Saved : | class {} --> {}'.format(i, line[2] ,line[1]))

