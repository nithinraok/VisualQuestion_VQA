#repo for running the main part of VQA
#assume that vqa_utils is already run 
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset_vqa import Dictionary, VQAFeatureDataset


def main(args):

    #defining torch configurations
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #defining dictionary and VQAFeatureDataset
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dataset = VQAFeatureDataset('train', dictionary)
    eval_dataset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size

    #model definition 

    #Dataloader initialization
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dataset, batch_size, shuffle=True, num_workers=1)

    features, spatials, question, target = next(iter(train_loader))
    #(type(features))
    #print(type(spatials))
    #print(type(question))
    #print(type(target))

    feats=features.numpy()
    spatials=spatials.numpy()
    question=question.numpy()
    target=target.numpy()
    print(feats.shape)
    print(question.shape)
    print(question[0])
    print(target)

    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    main(args)






