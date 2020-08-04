import os
import glob
import numpy as np
import torch
import csv

from PIL import Image
from torchvision.transforms import *

extension = '.jpg'

def import_basic_image(path):
    return Image.open(path).convert('RGB')

class Video_Folder(torch.utils.data.Dataset):

    def __init__(self,root,csv_path,numOfFrames,transform=None):
        self.csv_label = csv_path
        self.classes=[] 
        self.all_data=[]
        self.dict_gestures = dict()
        with (open,csv_path) as csvfile:
            csv_reader = csv.reader(csvfile,delimiter=';')
            for row in csv_reader:
                all_data.append((row[0],row[2])) #0 - video id ; #3- gesture id
                self.dict_gestures[row[0]]=row[2]
                if (row[2] not in self.classes) self.classes.append(row[2])
        self.root = root
        self.numOfFrames=numOfFrames
        self.transform=transform

    def __getitem__(self,index):
        item=all_data[index]
        folder = os.path.join(self.root,item[0])
        images = []
        all_paths = self.get_all_paths(self.root)
        for path in all_paths:
            img = import_basic_image(path)
            img=self.transform(img)
            imgs.append(torch.unsqueeze(img,0))
        data = torch.cat(img)
        target_idx = dict_gestures[index]
        data=data.permute(1,0,2,3)
        return (data,target_idx)
        #TODO

    def __len__(self):
        return (len(self.all_data))

    def get_all_paths(root):
        allpaths = []
        for i in range(self.numOfFrames):
            allpaths.append(os.path.join(root,str(i),extension))
        return allpaths

