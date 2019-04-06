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
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F
from models import *
from tqdm import tqdm
import time

def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry

def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target_yes_no.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])
    try:
        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if(len(answer['scores'])>0):
                entries.append(_create_entry(img_id2val[img_id], question, answer))
    except:
        #question_id_filter=[]
        question_id_filter=[answer['question_id'] for answer in answers]
        #questions_filter=[]
        #for question_id_curr in question_id_filter:
        #    for entr in questions:
        #        if(entr['question_id']==question_id_curr):
        #            questions_filter.append(entr)
        #            break

        question_id_tot=[entr['question_id'] for entr in questions]
        print('Finding matches')
        question_id_filter.sort()
        set_q_id=set(question_id_filter)
        start_time=time.time()
        id_matches=[id for id,val in enumerate(question_id_tot) if val in set_q_id]
        end_time=time.time()
        print('Matches found')
        time_lpsd=end_time-start_time
        print('Time elapsed:%f' %(time_lpsd))
        questions_filter=[questions[id] for id in id_matches]
        utils.assert_eq(len(questions_filter), len(answers))
        entries = []
        for question, answer in zip(questions_filter, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if(len(answer['scores'])>0):
                entries.append(_create_entry(img_id2val[img_id], question, answer))
        #print(question_id_filter)

    return entries

class VQADataset(Dataset):
    """VQADataset which returns a tuple of image, question tokens and the answer label
    """
    def __init__(self,image_root_dir,dictionary,dataroot,filename_len=12,choice='train',transform_set=None):

        #initializations
        self.img_root=image_root_dir
        self.data_root=dataroot
        self.img_dir=os.path.join(image_root_dir,choice+"2014")
        #print(os.path.exists(self.img_dir))
        self.choice=choice
        self.transform=transform_set
        self.filename_len=filename_len

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        #id_map_path=os.path.join(dataroot,'cache', '%s_target.pkl' % choice)


        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % choice),'rb'))
        self.dictionary=dictionary

        self.entries = _load_dataset(dataroot, choice, self.img_id2idx)
        self.tokenize()
        #self.entry_list=cPickle.load(open(id_map_path, 'rb'))
    
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
        max_id=answer_data['scores'].index(max(answer_data['scores'])) #finding the maximum score index
        label=int(answer_data['labels'][max_id])
        if(label==3):
            label=0
        elif(label==9):
            label=1
        image_id=entry['image_id']
        
        
        filename='COCO_'+self.choice+'2014_'+str(image_id).zfill(self.filename_len)+'.jpg'

        if(len(filename)>0):
            #print(filename[0])
            im=Image.open(os.path.join(self.img_dir,filename))
            im=im.convert('RGB')
            if(self.transform is not None):
                image=self.transform(im)
            question=torch.from_numpy(np.array(question))
            return(image,question,label)
        else:
            print(filename)
            print('Filepath not found')
            
    def __len__(self):
        return(len(self.entries))


if __name__ == "__main__":
    image_root_dir="/data/digbose92/VQA/COCO"
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    dataroot='/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data'

    transform_list=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    train_dataset=VQADataset(image_root_dir=image_root_dir,dictionary=dictionary,dataroot=dataroot,transform_set=transform_list)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    image,question,label=next(iter(train_loader))

    device=2
    torch.cuda.set_device(device)
    weights=np.load("data/glove6b_init_300d.npy")
    encoder_CNN=EncoderCNN(embed_size=1024).to(device)
    question_encoder=EncoderLSTM(hidden_size=512,weights_matrix=weights,fc_size=1024,max_seq_length=14,batch_size=8).to(device)
    fusion_network=FusionModule(fuse_embed_size=1024,input_fc_size=1024).to(device)

    print(encoder_CNN)
    print(question_encoder)
    print(fusion_network)

    #image_feats=encoder_CNN(image)
    #print(image_feats.size())



    """ques_token=question.numpy()
    lab=label.numpy()
    #image = F.to_pil_image(image)
    #print(image.size)
    image_numpy=image.numpy()[0]
   
    quest_list_new=ques_token[0].tolist()
    a = [x for x in quest_list_new if x != 19900]
    
    data=cPickle.load(open('data/dictionary.pkl','rb'))
    answer_lab=cPickle.load(open('data/cache/trainval_label2ans.pkl','rb'))
    index2word_map=data[1]
    
    word_list=[index2word_map[id] for id in a]
    ques=','.join(word_list)
    print(ques)
    #print(word_list)
    print(answer_lab[lab[0]])
    #print(image_numpy.shape)
    tensor2pil = transforms.ToPILImage()

    b=np.transpose(image_numpy,(1,2,0))
    b=b*255
    b=b.astype(int)
    print(np.max(b))
    cv2.imwrite(ques+'.jpg',b)
    #fig = plt.figure()
    #title_obj = plt.title(ques)
    #plt.imshow(b)
    #plt.show()
    #plt.savefig('COCO_sample.png')"""

