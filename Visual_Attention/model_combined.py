import sys
sys.path.insert(0, '/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All')
import torch
import torch.nn as nn
from attention_models import Base_Att
from language_models import WordEmbedding, QuestionEmbedding
from classifier_models import SimpleClassifier, ClassifierAdv
from fc import FCNet, GTH
import dataset_vqa
from dataset_vqa import Dictionary, VQAFeatureDataset
from dataset_image_vqa import VQADataset
import torchvision.transforms as transforms
# Dropout p: probability of an element to be zeroed. Default: 0.5
"""
Batch size first is required for all the images
"""
class VQA_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(VQA_Model, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]


        att = self.v_att(v, q_emb) # [batch, 1, v_dim]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits



def attention_baseline(dataset, num_hid, dropout, norm, activation, drop_L , drop_G, drop_W, drop_C):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=drop_W)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=drop_G, rnn_type='GRU')

    v_att = Base_Att(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, norm= norm, act= activation)
    q_net = FCNet([num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
    v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)

    classifier = SimpleClassifier(
        in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=drop_C, norm= norm, act= activation)
    return VQA_Model(w_emb, q_emb, v_att, q_net, v_net, classifier)



if __name__ == "__main__":
    dictionary=Dictionary.load_from_file('../Visual_All/data/dictionary.pkl')
    num_hid=1024
    dropout=0.3
    dropout_L=0.1
    dropout_G=0.2
    dropout_W=0.4
    dropout_C=0.5
    activation='ReLU'
    norm='weight'
    crop_size=224
    train_transform = transforms.Compose([ 
        transforms.Resize((crop_size,crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    img_root_dir="/data/digbose92/VQA/COCO"
    data_root_dir="/proj/digbose92/VQA/VisualQuestion_VQA/common_resources"
    train_dataset = VQADataset(image_root_dir=img_root_dir,dictionary=dictionary,dataroot=data_root_dir,choice='train',transform_set=train_transform)

    model = attention_baseline(train_dataset, num_hid=num_hid, dropout= dropout, norm=norm,\
                               activation=activation, drop_L=dropout_L, drop_G=dropout_G,\
                               drop_W=dropout_W, drop_C=dropout_C)

    print(model)
    
