import sys
sys.path.insert(0, '/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All')
import numpy as np 
import torch
import torchvision.transforms as transforms
import json
import _pickle as cPickle
from torch.utils.data import Dataset, DataLoader
import os
import utils
from PIL import Image
from dataset_vqa import Dictionary, VQAFeatureDataset
import glob
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F
from models import *
from tqdm import tqdm
import time
import h5py
def _create_entry(question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'question'    : question['question'],
        'answer'      : answer}
    return entry

def _load_dataset(dataroot,name):
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_1000_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, '%s_target_top_1000_ans.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    utils.assert_eq(len(questions), len(answers))
    entries = []
    print(len(questions))
    print(len(answers))
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if(len(answer['scores'])>0):
            entries.append(_create_entry(question, answer))
    return(entries)

class Dataset_VQA(Dataset):
    """Dataset for VQA applied to the attention case with image features in .hdf5 file 
    """
    def __init__(self,img_root_dir,feats_data_path,dictionary,dataroot,filename_len=12,choice='train',arch_choice='resnet152',layer_option='pool',transform_set=None):

        #initializations
        self.data_root=dataroot
        self.feats_data_path=feats_data_path
        self.img_dir=os.path.join(img_root_dir,choice+"2014")
        self.choice=choice
        self.transform=transform_set
        
        self.dictionary=dictionary
        self.arch=arch_choice 
        self.layer_option=layer_option 
        self.filename_len=filename_len
        #operations (reading features and making the entries)
        #train_filenames_resnet152.txt
        #reading the features
        print('Loading the hdf5 features')
        start_time=time.time()
        h5_path = os.path.join(feats_data_path,self.choice+'_feats_'+self.arch+'_'+self.layer_option+'.hdf5')
        file_path=os.path.join(feats_data_path,self.choice+'_filenames_'+self.arch+'.txt')
        fl_p=open(file_path)
        self.file_list=list(fl_p.readlines())
        self.file_list=[filename.split("\n")[0] for filename in self.file_list]
        
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('feats'))
        end_time=time.time()
        elapsed_time=end_time-start_time
        print('Total elapsed time: %f' %(elapsed_time))

        self.entries=_load_dataset(self.data_root,self.choice)
        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        #self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            answer = entry['answer']
            class_labels = np.array(answer['Class_Label'])
            labels=np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                class_labels=torch.from_numpy(class_labels)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
                entry['answer']['Class_Label']=class_labels
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
                entry['answer']['Class_Label']=None


    def __getitem__(self,index):
        entry=self.entries[index]
        #filtering of the labels based on the score
        question=entry['q_token']
        answer_data=entry['answer']
        label=answer_data['Class_Label']
        image_id=entry['image_id']

        filename='COCO_'+self.choice+'2014_'+str(image_id).zfill(self.filename_len)+'.jpg'
        idx=self.file_list.index(os.path.join(self.img_dir,filename))
        
        feat=torch.from_numpy(self.features.numpy()[idx])
        return(feat,question,label)

    def __len__(self):
        return(len(self.entries))


if __name__ == "__main__":
    image_root_dir="/data/digbose92/VQA/COCO"
    dictionary=Dictionary.load_from_file('../Visual_All/data/dictionary.pkl')
    feats_data_path="/data/digbose92/VQA/COCO/train_hdf5_COCO/"
    data_root="/proj/digbose92/VQA/VisualQuestion_VQA/common_resources"
    train_dataset=Dataset_VQA(img_root_dir=image_root_dir,feats_data_path=feats_data_path,dictionary=dictionary,dataroot=data_root,arch_choice="vgg16",layer_option="embedding")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)

    feat,question,label=next(iter(train_loader))

    print(feat.shape)
    print(question)
    print(label)