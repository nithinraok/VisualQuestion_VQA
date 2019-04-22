import sys
sys.path.insert(0, '/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All')
import torch
import torch.nn as nn
from attention_models import Base_Att
from language_models import WordEmbedding, QuestionEmbedding, BertEmbedding
from classifier_models import SimpleClassifier, ClassifierAdv
from fc import FCNet, GTH
import dataset_vqa
from dataset_vqa import Dictionary, VQAFeatureDataset
from dataset_image_vqa import VQADataset
import torchvision.transforms as transforms
from fusion_models import mfh_baseline
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

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]
        #print(q_emb.size())

        att = self.v_att(v, q_emb) # [batch, 1, v_dim]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

class VQA_Model_Bert(nn.Module):
    def __init__(self, bert_emb, v_att, q_net, v_net, classifier):
        super(VQA_Model_Bert, self).__init__()
        self.bert_emb = bert_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """   
        q_emb = self.bert_emb(q)   # run Linear + Bert on document embeddings [batch, q_dim]
        #print(q_emb.size())

        att = self.v_att(v, q_emb) # [batch, 1, v_dim]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

class VQA_Model_MFH(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, mfh_net, classifier):
        super(VQA_Model_MFH, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.mfh_net = mfh_net
        self.classifier = classifier

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """   
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]  # run Linear + Bert on document embeddings [batch, q_dim]
        #print(q_emb.size())

        att = self.v_att(v, q_emb) # [batch, 1, v_dim]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr=self.mfh_net(q_repr,v_repr)
        #joint_repr = q_repr * v_repr

        #invoke MFH for fusion of q_repr and v_repr

        logits = self.classifier(joint_repr)
        return logits


class VQA_Model_MFH_classifier(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, mfh_net):
        super(VQA_Model_MFH_classifier, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.mfh_net = mfh_net
        #self.classifier = classifier

    def forward(self, v, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """   
        w_emb = self.w_emb(q)       # get word embeddings
        q_emb = self.q_emb(w_emb)   # run GRU on word embeddings [batch, q_dim]  # run Linear + Bert on document embeddings [batch, q_dim]
        #print(q_emb.size())

        att = self.v_att(v, q_emb) # [batch, 1, v_dim]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        #joint_repr=self.mfh_net(q_repr,v_repr)
        logits=self.mfh_net(q_repr,v_repr)
        #joint_repr = q_repr * v_repr

        #invoke MFH for fusion of q_repr and v_repr

        #logits = self.classifier(joint_repr)
        return logits

class VQA_Model_MFH_BERT_fusion(nn.Module):
    def __init__(self, bert_emb, v_att, q_net, v_net, mfh_net,classifier):
        super(VQA_Model_MFH_BERT_fusion, self).__init__()
        self.bert_emb = bert_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.mfh_net = mfh_net
        self.classifier = classifier

    def forward(self, v, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """   
        q_emb = self.bert_emb(q)
        #print(q_emb.size())

        att = self.v_att(v, q_emb) # [batch, 1, v_dim]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        #joint_repr=self.mfh_net(q_repr,v_repr)
        joint_repr=self.mfh_net(q_repr,v_repr)
        #joint_repr = q_repr * v_repr

        #invoke MFH for fusion of q_repr and v_repr

        logits = self.classifier(joint_repr)
        return logits

############# ATTENTION BASELINE ############
def attention_baseline(dataset, num_hid, dropout, norm, activation, drop_L , drop_G, drop_W, drop_C, bidirect_val=False):
    print('Here in the attention baseline')
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=drop_W)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=bidirect_val, dropout=drop_G, rnn_type='GRU')
    #bert_emb=BertEmbedding(in_dim=7168,num_hid=num_hid)

    v_att = Base_Att(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, bidirect=bidirect_val,norm= norm, act= activation)
    if(bidirect_val is False):
        q_net = FCNet([num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        #v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    else:
        q_net = FCNet([2*num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        
    v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=drop_C, norm= norm, act= activation)
    return(VQA_Model(w_emb,q_emb,v_att,q_net,v_net,classifier))



########## ATTENTION + BERT ############
def attention_bert_baseline(dataset, num_hid, dropout, norm, activation, drop_L , drop_G, drop_W, drop_C, bidirect_val=False):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=drop_W)
    #q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=bidirect_val, dropout=drop_G, rnn_type='GRU')
    bert_emb=BertEmbedding(in_dim=3072,num_hid=num_hid)

    v_att = Base_Att(v_dim= dataset.v_dim, q_dim= num_hid, num_hid= num_hid, dropout= dropout, bidirect=bidirect_val,norm= norm, act= activation)
    if(bidirect_val is False):
        q_net = FCNet([num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        #v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    else:
        q_net = FCNet([2*num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        
    v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=drop_C, norm= norm, act= activation)
    return(VQA_Model_Bert(bert_emb,v_att,q_net,v_net,classifier))



######## ATTENTION + MFH ###########
def attention_mfh(dataset, num_hid, dropout, norm, activation, drop_L , drop_G, drop_W, drop_C, mfb_out_dim, bidirect_val=False):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=drop_W)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=bidirect_val, dropout=drop_G, rnn_type='GRU')
    v_att = Base_Att(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, bidirect=bidirect_val,norm= norm, act= activation)
    if(bidirect_val is False):
        q_net = FCNet([num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        #v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    else:
        q_net = FCNet([2*num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        
    v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    mfh_net=mfh_baseline(QUEST_EMBED=num_hid,VIS_EMBED=num_hid,MFB_OUT_DIM=mfb_out_dim)
    classifier = SimpleClassifier(in_dim=2*mfb_out_dim, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=drop_C, norm= norm, act= activation)
    return(VQA_Model_MFH(w_emb,q_emb,v_att,q_net,v_net,mfh_net,classifier))
    #return VQA_Model_Bert(bert_emb, v_att, q_net, v_net, classifier)


######### ATTENTION + MFH + MFH CLASSIFIER ##############
def attention_mfh_classifier(dataset, num_hid, dropout, norm, activation, drop_L , drop_G, drop_W, drop_C, mfb_out_dim, bidirect_val=False):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=drop_W)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=bidirect_val, dropout=drop_G, rnn_type='GRU')
    v_att = Base_Att(v_dim= dataset.v_dim, q_dim= q_emb.num_hid, num_hid= num_hid, dropout= dropout, bidirect=bidirect_val,norm= norm, act= activation)
    if(bidirect_val is False):
        q_net = FCNet([num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        #v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    else:
        q_net = FCNet([2*num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        
    v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    mfh_net=mfh_baseline(QUEST_EMBED=num_hid,VIS_EMBED=num_hid,MFB_OUT_DIM=mfb_out_dim)
    #classifier = SimpleClassifier(in_dim=2*mfb_out_dim, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=drop_C, norm= norm, act= activation)
    return(VQA_Model_MFH_classifier(w_emb,q_emb,v_att,q_net,v_net,mfh_net))


###### ATTENTION + BERT + MFH FUSION #############
def attention_bert_mfh_fusion(dataset, num_hid, dropout, norm, activation, drop_L , drop_G, drop_W, drop_C, mfb_out_dim, bidirect_val=False):
    #w_emb = WordEmbedding(dataset.dictionary.ntoken, emb_dim=300, dropout=drop_W)
    #q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=bidirect_val, dropout=drop_G, rnn_type='GRU')

    bert_emb=BertEmbedding(in_dim=3072,num_hid=num_hid)
    v_att = Base_Att(v_dim= dataset.v_dim, q_dim= num_hid, num_hid= num_hid, dropout= dropout, bidirect=bidirect_val,norm= norm, act= activation)
    if(bidirect_val is False):
        q_net = FCNet([num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        #v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    else:
        q_net = FCNet([2*num_hid, num_hid], dropout= drop_L, norm= norm, act= activation)
        
    v_net = FCNet([dataset.v_dim, num_hid], dropout= drop_L, norm= norm, act= activation)
    mfh_net=mfh_baseline(QUEST_EMBED=num_hid,VIS_EMBED=num_hid,MFB_OUT_DIM=mfb_out_dim)
    classifier = SimpleClassifier(in_dim=2*mfb_out_dim, hid_dim=2 * num_hid, out_dim=dataset.num_ans_candidates, dropout=drop_C, norm= norm, act= activation)
    return(VQA_Model_MFH_BERT_fusion(bert_emb,v_att,q_net,v_net,mfh_net,classifier))


def weights_init_xn(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier_normal(m.bias.data)
def weights_init_xu(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)

# a=0.01 for Leaky RelU
def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.01)
def weights_init_ku(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data, a=0.01)


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
    
