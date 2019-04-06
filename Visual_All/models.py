import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary
import numpy as np


def create_embedding_layer(weights_matrix,non_trainable=False):
    """loads the weights from glove numpy array and returns the layer, embeddings and the embedding dimensions
    """
    num_embeddings,embedding_dim=weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    print(weights_matrix.shape)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    if non_trainable is False:
        print('Here')
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        for param in self.resnet.parameters():
            param.requires_grad = False
        #self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        #with torch.no_grad():
        #    features = self.resnet(images)
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features=self.linear(features)
        features=torch.tanh(features)
        #features=F.relu(features)
        #features = self.bn(self.linear(features))
        return features

class EncoderLSTM(nn.Module):
    def __init__(self, hidden_size,weights_matrix,train_embed=False,use_gpu=True,fc_size=2048, num_layers=1, max_seq_length=14, batch_size=32,dropout_rate=0.5):
        """Module for stacked LSTM which returns a single hidden state for the input question
        """
        super(EncoderLSTM, self).__init__()
        self.use_gpu=use_gpu
        self.max_seg_length = max_seq_length
        self.batch_size=batch_size
        self.hidden_dim=hidden_size
        
        self.linear_fc_size=fc_size
        self.nlayers=num_layers
        self.timesteps=max_seq_length
        #self.vocab_len=19901
        #self.embed_len=300
        #self.embed=nn.Embedding(self.vocab_len, self.embed_len)
        self.embed , self.vocab_len , self.embed_len = create_embedding_layer(weights_matrix,non_trainable=train_embed) 
        self.lstm = nn.LSTM(input_size=self.embed_len, hidden_size=self.hidden_dim, num_layers=self.nlayers)
        self.linear = nn.Linear(self.hidden_dim, self.linear_fc_size)
        self.dropout = nn.Dropout(dropout_rate)
        # self.hidden = self.init_hidden()
        
    def init_hidden(self,batch_size):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:

            init_val=(torch.randn(self.nlayers,batch_size,self.hidden_dim).cuda(),
                        torch.randn(self.nlayers,batch_size,self.hidden_dim).cuda())
            return(init_val)
            #return 
            
            #(Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
            #         Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            init_val=(torch.randn(self.nlayers,batch_size,self.hidden_dim),
                        torch.randn(self.nlayers,batch_size,self.hidden_dim))
            return(init_val)

    def forward(self, input_sentence):
        """Forward pass of the encoderLSTM network
        """
        
        input = self.embed(input_sentence).view(self.timesteps,-1,self.embed_len)
        batch_size=input.shape[1]
        hidden_val=self.init_hidden(batch_size)
        lstm_out, hidden_fin = self.lstm(input, hidden_val)
        #print(hidden_fin[0][-1].size())
        linear_scores=self.linear(lstm_out[-1])
        #linear_scores=self.dropout(linear_scores)
        #act_vals=F.relu(linear_scores)
        act_vals=torch.tanh(linear_scores)
        return(act_vals)

class FusionModule(nn.Module):
    def __init__(self,qnetwork,img_network,fuse_embed_size=2048,input_fc_size=1024,class_size=3123,dropout_rate=0.2):
        """Module for fusing the mean pooled image features and lstm hidden states
        """
        super(FusionModule, self).__init__()
        self.fuse_embed_size=fuse_embed_size
        self.num_classes=class_size
        self.input_fc_size=input_fc_size
        self.q_net=qnetwork
        self.im_net=img_network
        
        self.embed_layer=nn.Linear(self.input_fc_size,self.fuse_embed_size)
        self.class_layer=nn.Linear(self.fuse_embed_size,self.num_classes)
        #elf.dropout=nn.Dropout(dropout_rate)
        

    def forward(self,sent_batch, image_batch):
        """Forward pass of the Fusion module
        """
        #adding one initial Linear operation
        encoder_hidden_states=self.q_net(sent_batch)
        image_features=self.im_net(image_batch)
        fuse_embed=encoder_hidden_states*image_features
        #fuse_embed=self.dropout(fuse_embed)
        lin_op=self.embed_layer(fuse_embed)
        lin_vals=torch.tanh(lin_op)
        #lin_vals=torch.tanh(lin_op)
        #lin_vals=self.dropout(lin_vals)
        #print('Weights printing')
        #print(self.q_net.linear.weight.data)
        class_embed=self.class_layer(lin_vals)
        class_vals=F.softmax(class_embed,dim=1)

        return(class_vals)


if __name__ == "__main__":

        weights_file="data/glove6b_init_300d.npy"
        embed_weights=np.load(weights_file)

        test_torch_tensor=torch.tensor([[0,1,3],[0,3,1]],dtype=torch.long)
        test_image_feats=torch.tensor([[0.1,1.2,3.1],[0.3,3.2,1.3]],dtype=torch.float)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #testing the EncoderLSTM part
        encode_test=EncoderLSTM(hidden_size=3,weights_matrix=embed_weights,use_gpu=True,fc_size=3,max_seq_length=3,batch_size=2)
        encode_test=encode_test.to(device)
        test_torch_tensor=test_torch_tensor.to(device)
        test_image_tensor=test_image_feats.to(device)
        encoder_ops=encode_test(test_torch_tensor)
        print(encode_test)
        #testing the FusionModule part 
        fuse_test=FusionModule(fuse_embed_size=3,class_size=2)
        #summary(encode_test,(1,3))
        fuse_test=fuse_test.to(device)
        class_op=fuse_test(encoder_ops,test_image_tensor)
        print(class_op.size())
        
