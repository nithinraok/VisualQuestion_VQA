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

def _create_entry(img, question, answer):
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
    answer_path = os.path.join(dataroot, '%s_target_yes_no.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if(len(answer['scores'])>0):
            entries.append(_create_entry(question, answer))

class Dataset_VQA(Dataset):
    """Dataset for VQA applied to the attention case with image features in .hdf5 file 
    """
    def __init__(self,img_root_dir,feats_data_path,dictionary,dataroot,filename_len=12,file_list_path=None,choice='train',arch_choice='resnet152',layer_option='pool',transform_set=None):

        #initializations
        self.data_root=dataroot
        self.feats_data_path=feats_data_path
        self.img_dir=os.path.join(img_root_dir,choice+"2014")
        self.choice=choice
        self.transform=transform_set
        self.file_list_path=file_list_path
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

    def __getitem__(self,index):
        entry=self.entries[index]
        #filtering of the labels based on the score
        question=entry['q_token']
        answer_data=entry['answer']
        label=entry['Class_label']
        image_id=entry['image_id']

        filename='COCO_'+self.choice+'2014_'+str(image_id).zfill(self.filename_len)+'.jpg'
        idx=self.file_list.index(os.path.join(self.img_dir,filename))

        feat=self.features[idx]

        return(feat,question,label)




