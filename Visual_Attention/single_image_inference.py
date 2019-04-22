import sys
sys.path.insert(0,'/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All')
from torch.utils.data import Dataset, DataLoader
from model_combined import *
from vqa_dataset_attention import *
from dataset_vqa import Dictionary, VQAFeatureDataset
import torch
from collections import OrderedDict 
import argparse 
import torch.nn.parallel
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision import models
import torch.nn as nn
import json
from PIL import Image
import pickle 
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

def load_model(args):
    model_checkpoint=torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in model_checkpoint.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

    #new_state_dict["classifier.main.2.bias"]=new_state_dict.pop("classifier.main.3.bias")
    #new_state_dict["classifier.main.2.weight_g"]=new_state_dict.pop("classifier.main.3.weight_g")
    #new_state_dict["classifier.main.2.weight_v"]=new_state_dict.pop("classifier.main.3.weight_v")
    
    print('Model checkpoint loaded')
    dictionary=Dictionary.load_from_file(args.pickle_path)
    train_dataset=Dataset_VQA(img_root_dir=args.image_root_dir,feats_data_path=args.feats_data_path,dictionary=dictionary,choice='train',dataroot=args.data_root,arch_choice=args.arch_choice,layer_option=args.layer_option)

    #train_dataset=Dataset_VQA(img_root_dir=args.image_root_dir,feats_data_path=args.feats_data_path,dictionary=dictionary,choice='train',dataroot=args.data_root,arch_choice=args.arch_choice,layer_option=args.layer_option)
    print('Loading the attention model')

    attention_model = attention_baseline(train_dataset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, drop_L=args.dropout_L, drop_G=args.dropout_G,\
                               drop_W=args.dropout_W, drop_C=args.dropout_C)
    #attention_model=attention_mfh(train_dataset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               #activation=args.activation, drop_L=args.dropout_L, drop_G=args.dropout_G,\
                               #drop_W=args.dropout_W, drop_C=args.dropout_C,mfb_out_dim=args.mfb_out_dim)
    attention_model.load_state_dict(new_state_dict)
    attention_model.train(False)
    torch.cuda.manual_seed_all(args.seed)
    #torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    attention_model.to(args.device)

    return(attention_model)

def preproc_question(str,max_length,dictionary):
    tokens = dictionary.tokenize(str, False)
    tokens = tokens[:max_length]
    if len(tokens) < max_length:
        padding = [dictionary.padding_idx] * (max_length - len(tokens))
        tokens = padding + tokens
    return(tokens)

if __name__ == "__main__":
    #parser.add_argument('--epochs', type=int, default=40)
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root_dir', type=str, default="/data/digbose92/VQA/COCO")
    parser.add_argument('--pickle_path', type=str, default="../Visual_All/data/dictionary.pkl")
    parser.add_argument('--feats_data_path', type=str, default="/data/digbose92/VQA/COCO/train_hdf5_COCO/")
    parser.add_argument('--data_root', type=str, default="/proj/digbose92/VQA/VisualQuestion_VQA/common_resources")
    parser.add_argument('--npy_file', type=str, default="../../VisualQuestion_VQA/Visual_All/data/glove6b_init_300d.npy")
    parser.add_argument('--model_path', type=str, default="results_GRU_uni/results_resnet_152_hid_512_YES_NO_ADAM/model.pth")
    parser.add_argument('--image_model', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_hid', type=int, default=512) # they used 1024
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dropout_L', type=float, default=0.1)
    parser.add_argument('--dropout_G', type=float, default=0.2)
    parser.add_argument('--dropout_W', type=float, default=0.4)
    parser.add_argument('--dropout_C', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='PReLU, ReLU, LeakyReLU, Tanh, Hardtanh, Sigmoid, RReLU, ELU, SELU')
    parser.add_argument('--norm', type=str, default='weight', help='weight, batch, layer, none')
    parser.add_argument('--choice', type=str, default='val', help='choice of the split')
    parser.add_argument('--seed', type=int, default=9731, help='random seed')
    parser.add_argument('--arch_choice', type=str, default='resnet152', help='choice of the network')
    parser.add_argument('--layer_option', type=str, default='pool', help='choice of the layer')
    parser.add_argument('--num_workers', type=int, default=4, help='number of the workers')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--class_metadata_file', type=str, default='/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/Train_Class_Distribution.csv', help='Path of class metadata file')
    parser.add_argument('--rcnn_path',type=str,default="/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/val36_imgid2idx.pkl",help="Path of the rcnn features file")
    parser.add_argument('--bert_option',type=bool,default=False,help="Whether to use bert or not")
    parser.add_argument('--mfb_out_dim', type=int, default=1000, help='mfb output dimension')
    
    args = parser.parse_args()
    
    class_meta_data=pd.read_csv('/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/Train_Class_Distribution.csv')
    #class_meta_data={}
    class_label_map=class_meta_data['Label_names'].tolist()
    #class_label_map=['no','yes']
    valid_rcnn_pickle_file="/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/val36_imgid2idx.pkl"
    pkl_path=pickle.load(open(valid_rcnn_pickle_file,'rb'))
    
    model=load_model(args)
    #model.eval()
    print('Load the validation json file')
    valid_questions=json.load(open('/proj/digbose92/VQA/VisualQuestion_VQA/common_resources/v2_OpenEnded_mscoco_val2014_yes_no_questions.json'))['questions']
    valid_entry=valid_questions[0] 
    print(valid_entry)
    dictionary=Dictionary.load_from_file('../Visual_All/data/dictionary.pkl')
    print(valid_entry['question'])
    tokens=preproc_question(valid_entry['question'],14,dictionary)
    pkl_data=pickle.load(open('/proj/digbose92/VQA/VisualQuestion_VQA/common_resources/val_target_yes_no_ans.pkl','rb'))
    question_ids=[pkl_data[i]['question_id'] for i, question in enumerate(pkl_data)]
    id=question_ids.index(valid_entry['question_id'])
    print(pkl_data[id])
    

    print(id)

    ########################################## RCNN features extraction here ###############################
    #hdfeatures="/data/digbose92/VQA/COCO/train_hdf5_COCO/train_rcnn_36.hdf5"
    #h5_features=h5py.File(hdfeatures)['image_features']

    ############################################ resnet152 feature extraction here ##########################
    print("================== LOADING HDF5 resnet152 features==================")
    hdfeatures="/data/digbose92/VQA/COCO/train_hdf5_COCO/val_feats_resnet152_pool.hdf5"
    h5_features=h5py.File(hdfeatures)['feats']
    file_path="/data/digbose92/VQA/COCO/train_hdf5_COCO/val_filenames_resnet152.txt"
    fl_p=open(file_path)
    file_list=list(fl_p.readlines())
    file_list=[filename.split("\n")[0] for filename in file_list]
    choice='val'
    image_id=pkl_data[id]['image_id']
    image_path='COCO_'+choice+'2014_'+str(image_id).zfill(12)+'.jpg'
    folder="/data/digbose92/VQA/COCO/val2014"
    file_path=os.path.join(folder,image_path)
    idx=file_list.index(file_path)
    #idx=pkl_path[image_id]

    feat=torch.from_numpy(h5_features[idx])
    feat=feat.view(feat.size(0),feat.size(1)*feat.size(2))
    feat=feat.transpose(1,0)
    feat=feat.unsqueeze(0)
    tokens=torch.from_numpy(np.array(tokens))
    tokens=tokens.unsqueeze(0)
    feat=feat.to(0)
    tokens=tokens.to(0)
    print(feat.size())
    print(tokens.size())
    pred=model(feat,tokens)
    logits = torch.max(pred, 1)[1].data
    print(logits)
    print(pkl_data[id]['Class_Label'])
    print('Actual label:',class_label_map[pkl_data[id]['Class_Label']])
    print('Predicted label:',class_label_map[logits.cpu().numpy()[0]])


    
    

