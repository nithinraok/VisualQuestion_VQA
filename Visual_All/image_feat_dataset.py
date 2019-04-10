import numpy as np 
import torchvision.transforms as transforms
import json
import _pickle as cPickle
from torch.utils.data import Dataset, DataLoader
import os
import utils
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F
from models import *
from tqdm import tqdm
import time



class ImageFeatDataset(Dataset):
    """Pytorch Dataset which returns a tuple of image, question tokens and the answer label
    """
    def __init__(self,root_dir,choice='train',transform_list=None):
        self.img_dir=os.path.join(root_dir,choice+"2014")
        self.choice=choice
        self.entries=os.listdir(self.img_dir)
        self.transform=transform_list
        
    def __getitem__(self,index):
        """returns a single image
        """
        filename=self.entries[index]
        im=Image.open(os.path.join(self.img_dir,filename))
        im=im.convert('RGB')
        if(self.transform is not None):
            image=self.transform(im)
        return(image,os.path.join(self.img_dir,filename))

    def __len__(self):
        return(len(self.entries))

if __name__ == "__main__":
    root_dir="/data/digbose92/VQA/COCO/"
    train_transform = transforms.Compose([ 
        transforms.Resize((224,224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    image_dataset=ImageFeatDataset(root_dir=root_dir,transform_list=train_transform)
    train_loader = DataLoader(image_dataset, 4, shuffle=True, num_workers=1)
    image,filepaths=next(iter(train_loader))

    print(image.size())
    print(type(filepaths))
