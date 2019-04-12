import torch
import torch.nn as nn
from attention_models import Base_Att
from language_models import WordEmbedding, QuestionEmbedding
from classifier_models import SimpleClassifier, ClassifierAdv
from fc import FCNet, GTH
from Visual_All import dataset_vqa
from dataset_vqa import Dictionary, VQAFeatureDataset
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

if __name__ == "__main__":
    dictionary=Dictionary.load_from_file('Visual_All/data/dictionary.pkl')
    
