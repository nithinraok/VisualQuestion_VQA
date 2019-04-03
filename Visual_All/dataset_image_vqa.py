import torch
import torchvision.transforms as transforms
import json
import _pickle as cPickle
from torch.utils.data import Dataset
import os
import utils
from PIL import Image

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
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if(len(answer['scores'])>0):
            entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries

class VQADataset(Dataset):
    """VQADataset which returns a tuple of image, question tokens and the answer label
    """

    def __init__(self,image_root_dir,dictionary,dataroot,json_root_dir,choice='train',transforms=None):

        #initializations
        self.img_root=image_root_dir
        self.data_root=dataroot
        self.img_dir=os.path.join(image_root_dir,choice+"2014")
        self.choice=choice
        self.transform=transforms

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        #id_map_path=os.path.join(dataroot,'cache', '%s_target.pkl' % choice)


        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % choice),'rb'))
        self.dictionary=dictionary

        self.entries = _load_dataset(dataroot, choice, self.img_id2idx)
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
        label=answer_data['labels'][max_id]
        image_id=entry['image_id']
        image_path='COCO_'+self.choice+'2014_'+'000000'+image_id+'.jpg'
        try:
            image=Image.open(os.path.join(self.img_dir,image_path)).convert('RGB')
            if(self.transform is not None):
                image=self.transform(image)
        except:
            print('Image path not found')

        #tensorize everything and return the tensors
        label=torch.LongTensor(label)
        question=torch.LongTensor(question)

        return(image,question,label)


    def __len__(self):
        return(len(self.entries))


if __name__ == "__main__":
    image_root_dir="/data/digbose92/VQA/COCO"
    


