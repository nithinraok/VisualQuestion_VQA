import sys
sys.path.insert(0,'/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All')
import torch 
import numpy as np 
from image_feats_extract import *
from dataset_vqa import Dictionary, VQAFeatureDataset
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm 
import torch.nn.functional as F
from vqa_dataset_attention import *
from model_combined import *
from collections import OrderedDict 
import pandas as pd

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    class_labels=torch.max(labels, 1)[1].data
    
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores, logits, class_labels

def evaluate_attention_model(args):

    class_data=pd.read_csv(args.class_metadata_file)
    class_label_map=class_data['Label_names'].tolist()

    print('Loading model checkpoint')
    attention_model_checkpoint=torch.load(args.model_path)
    
    new_state_dict = OrderedDict()
    for k, v in attention_model_checkpoint.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    print('Model checkpoint loaded')
    
    
    print('Loading Dictionary')
    dictionary=Dictionary.load_from_file(args.pickle_path)

    train_dataset=Dataset_VQA(img_root_dir=args.image_root_dir,feats_data_path=args.feats_data_path,dictionary=dictionary,choice='train',dataroot=args.data_root,arch_choice=args.arch_choice,layer_option=args.layer_option)
    print('Loading the attention model')
    attention_model = attention_baseline(train_dataset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, drop_L=args.dropout_L, drop_G=args.dropout_G,\
                               drop_W=args.dropout_W, drop_C=args.dropout_C)
    attention_model.load_state_dict(new_state_dict)
    attention_model.eval()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    attention_model.to(args.device)
    if(args.image_model is None):
        """use extracted features as a Dataset and Dataloader
        """
        print('Using validation features')
        dataset_temp=Dataset_VQA(img_root_dir=args.image_root_dir,feats_data_path=args.feats_data_path,dictionary=dictionary,choice=args.choice,dataroot=args.data_root,arch_choice=args.arch_choice,layer_option=args.layer_option)
        loader=DataLoader(dataset_temp, batch_size=args.batch_size, shuffle=False, num_workers=10)
        print('Length of validation dataloader:', len(loader))
        upper_bound = 0
        num_data = 0
        V_loss=0 
        score=0
        print('Validation data loading starting')
        actual_class_labels=[]
        predicted_class_labels=[]
        question_set=[]
        for data in tqdm(loader):

            feat,quest,quest_sent,target = data
            
            feat = feat.to(args.device)
            quest = quest.to(args.device)
            target = target.to(args.device)
            pred = attention_model(feat, quest, target)
            question_set=question_set+list(quest_sent)
            loss = instance_bce_with_logits(pred, target)
            V_loss += loss.item() * feat.size(0)
            score_temp, logits, class_labels= compute_score_with_logits(pred, target.data)
            actual_class_labels=actual_class_labels+list(class_labels.cpu().numpy())
            predicted_class_labels=predicted_class_labels+list(logits.cpu().numpy())
            batch_score=score_temp.sum()
            score += batch_score
            upper_bound += (target.max(1)[0]).sum()
            num_data += pred.size(0)
            
        

        class_predicted_name=[class_label_map[id] for id in predicted_class_labels]
        class_actual_name=[class_label_map[id] for id in actual_class_labels]

        predicted_df=pd.DataFrame({'Questions':question_set,'Actual_Answers':class_actual_name,'Predicted_Answers':class_predicted_name})
        predicted_df.to_csv('Validation_Stats.csv')
        score = score / len(loader.dataset)
        V_loss /= len(loader.dataset)
        upper_bound = upper_bound / len(loader.dataset)
        print(score,V_loss)
        #print(pred)
    else:   
        print("Extract features and then come back")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--eval', action='store_true', help='set this to evaluate.')

    #parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--image_root_dir', type=str, default="/data/digbose92/VQA/COCO")
    parser.add_argument('--pickle_path', type=str, default="../Visual_All/data/dictionary.pkl")
    parser.add_argument('--feats_data_path', type=str, default="/data/digbose92/VQA/COCO/train_hdf5_COCO/")
    parser.add_argument('--data_root', type=str, default="/proj/digbose92/VQA/VisualQuestion_VQA/common_resources")
    parser.add_argument('--npy_file', type=str, default="../../VisualQuestion_VQA/Visual_All/data/glove6b_init_300d.npy")
    parser.add_argument('--model_path', type=str, default="results_resnet_152_1000_CLASSES/model_resnet_152.pth")
    parser.add_argument('--image_model', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_hid', type=int, default=1024) # they used 1024
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
    parser.add_argument('--num_workers', type=int, default=10, help='number of the workers')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--class_metadata_file', type=str, default='/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/Train_Class_Distribution.csv', help='Path of class metadata file')
    args = parser.parse_args()
    evaluate_attention_model(args)
    
