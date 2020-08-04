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
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile,delimiter=',')
            for row in csv_reader:
                self.all_data.append((row[0],row[3])) #0 - video id ; #3- gesture id
                #print(self.all_data[len(self.all_data)-1])
                self.dict_gestures[row[0]]=row[3]
                if (row[2] not in self.classes):
                     self.classes.append(row[3])
        self.root = root
        self.numOfFrames=numOfFrames
        self.transform=transform

    def __getitem__(self,index):
        item=self.all_data[index]
        item = str(item[0])
        folder = os.path.join(self.root,item)
        imgs = []
        all_paths = self.get_all_paths(folder)
        for path in all_paths:
            img = import_basic_image(path)
            img=self.transform(img)
            imgs.append(img.unsqueeze(0))
        data = torch.cat(imgs)
        #print(self.dict_gestures)
        target_idx = self.dict_gestures[item]
        data=data.permute(1,0,2,3)
        return (data,target_idx)
        #TODO

    def __len__(self):
        return (len(self.all_data))

    def get_all_paths(self,root):
        allpaths = []
        for i in range(self.numOfFrames):
            stri = str(i+1)
            for i in range(5-len(stri)):
                stri='0'+stri
            allpaths.append(os.path.join(root,stri+extension))
        return allpaths

if __name__=='__main__':
    transform = Compose([CenterCrop(84),ToTensor()])
    trainVideoFolder = Video_Folder('D:\\gestures\\Train','D:\\gestures\\Train.csv',37,transform)
    print(trainVideoFolder.__getitem__(1))