import argparse
import os
import time
import h5py
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision import models
from image_feat_dataset import *
import torch.nn as nn

def model_extract(model_str,layer_option="embedding"):

    if(model_str=="vgg16"):
        model=models.vgg16(pretrained=True)
        if(layer_option=="embedding"):
            """extracts the image embeddings
            """
            features = list(model.classifier.children())[:-1]
            model.classifier=nn.Sequential(*features)
            
        elif(layer_option=="pool"):
            """extracts the pooled features
            """
            features=nn.Sequential(*list(model.features.children())[:-1])
            model=features

        for param in model.parameters():
                param.require_grad=False

    elif(model_str=="resnet152"):
        model=models.resnet152(pretrained=True)
        if(layer_option=="embedding"):
            """extracts the image embeddings
            """
            features=nn.Sequential(*list(model.children())[:-1])
            model=features
            print(model)
        
        elif(layer_option=="pool"):
            """extracts the pooled features
            """
            features=nn.Sequential(*list(model.children())[:-2])
            model=features

        for param in model.parameters():
                param.require_grad=False

    return(model)

def main(args):

    #get the appropriate model
    model=model_extract(args.arch,args.layer_option)
    model.eval()

    #move the model to device
    device=3
    torch.cuda.set_device(device)
    model=model.to(device)

    #obtain the dataloader
    train_transform = transforms.Compose([ 
        transforms.Resize((224,224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    image_dataset=ImageFeatDataset(root_dir=args.dir_data,transform_list=train_transform)
    print('Computing training image features:')
    loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    #hdf5_file initialization for computing training features
    train_hdf5_path=os.path.join(args.destination_dir,'train_feats.hdf5')

    #hdf5 file initialization for computing validation features



    print('Computing validation features')
    #image=image.to(device)
    #output=model(image)
    #print(output.size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Feature extraction script')
    parser.add_argument('--dir_data', default='/data/digbose92/VQA/COCO/',
                    help='dir where the COCO images are kept')
    parser.add_argument('--choice', default='train', type=str,
                    help='Data split to consider')
    parser.add_argument('--arch',default='vgg16',type=str)
    parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=32, type=int,
                    help='mini-batch size')
    parser.add_argument('--destination_dir', default="/data/digbose92/VQA/COCO/train_hdf5_COCO", type=str,
                    help='destination location')
    parser.add_argument('--layer_option', default="pool", type=str,
                    help='option between embedding and last pooled features')
    parser.add_argument('--size', default=224, type=int,
                    help='Image size for the network')
    args=parser.parse_args()
    main(args)
