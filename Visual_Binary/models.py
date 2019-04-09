import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
#from torchsummary import summary
import numpy as np
from torch.nn.utils.weight_norm import weight_norm as WN

def savemodel(model,device,name):
    model.to('cpu')
    # model.class_to_idx = classes_labels
    checkpoint = {'input_size': 3*224*224,
                  'output_size': 2,
                  'model': model,
                  'state_dict': model.state_dict()
                  }

    torch.save(checkpoint, 'savedmodels/final_model_'+name+'.pth')
    model.to(device)


def create_embedding_layer(weights_matrix,non_trainable=False):
    """loads the weights from glove numpy array and returns the layer, embeddings and the embedding dimensions
    """
    num_embeddings,embedding_dim=weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    print(weights_matrix.shape)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class Vgg16_4096(nn.Module):
    def __init__(self, original_model):
        super(Vgg16_4096, self).__init__()
        bottle1 = []
        bottle1.append(list(original_model.children())[0])
       # bottle1.extend(list(original_model.children())[1])
        bottle2 = []
    
        bottle2.append(list(original_model.children())[1][:-3])
        self.features1 = nn.Sequential(*bottle1)
        self.features2 = nn.Sequential(*bottle2)
        for param in original_model.parameters():
            param.requires_grad=False

    def forward(self, x):
        x = self.features1(x)
        x = x.view(x.shape[0],-1)
        x = self.features2(x)
        return x

class LinearImageModel(nn.Module):
    def __init__(self,n_input=4096,n_output=1024):
        super(LinearImageModel,self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.pre_vgg = Vgg16_4096(vgg16)
        self.model = nn.Sequential(WN(nn.Linear(n_input,n_output)),
                        nn.Tanh()
                        )
    def forward(self,x):
        pre_trained=self.pre_vgg(x)
        pre_trained=F.normalize(pre_trained,p=2,dim=1)
        out=self.model(pre_trained)
        return out

class EncoderLSTM(nn.Module):
    def __init__(self, hidden_size,weights_matrix,train_embed=False,use_gpu=True,fc_size=1024, num_layers=2, max_seq_length=14, batch_size=32):
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

        self.embed , self.vocab_len , self.embed_len = create_embedding_layer(weights_matrix,non_trainable=train_embed) 
        self.lstm = nn.LSTM(input_size=self.embed_len, hidden_size=self.hidden_dim, num_layers=self.nlayers)
        self.linear = WN(nn.Linear(self.hidden_dim, self.linear_fc_size))

        
    def init_hidden(self,batch_size):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (torch.randn(self.nlayers,batch_size,self.hidden_dim).cuda(),
                        torch.randn(self.nlayers,batch_size,self.hidden_dim).cuda())
        else:
            return (torch.randn(self.nlayers,batch_size,self.hidden_dim),
                        torch.randn(self.nlayers,batch_size,self.hidden_dim))

    def forward(self, input_sentence):
        """Forward pass of the encoderLSTM network
        """
        
        input = self.embed(input_sentence).view(self.timesteps,-1,self.embed_len)
        batch_size=input.shape[1]
        hidden_val = self.init_hidden(batch_size)
        lstm_out, hidden_fin = self.lstm(input, hidden_val)
        linear_scores=self.linear(hidden_fin[0][-1])
#        act_vals=torch.tanh(linear_scores)
        act_vals= nn.Tanh()(linear_scores)
        return(act_vals)

class FusionModule(nn.Module):
    def __init__(self,qnetwork,img_network,fuse_embed_size=1024,fc_size=512,class_size=2):
        """Module for fusing the mean pooled image features and lstm hidden states
        """
        super(FusionModule, self).__init__()
        self.fuse_size=fuse_embed_size
        self.num_classes=class_size
        self.fc_size=fc_size
        self.q_net = qnetwork
        self.im_net=img_network
 
        self.embed_layer=WN(nn.Linear(self.fuse_size,self.fc_size))
        self.class_layer=WN(nn.Linear(self.fc_size,self.num_classes))


    def forward(self,question_batch, image_batch):
        """Forward pass of the Fusion module
        """
        encoder_hidden_states=self.q_net(question_batch)
        image_features=self.im_net(image_batch)
        fuse_embed=torch.mul(encoder_hidden_states,image_features)
        lin_op=self.embed_layer(fuse_embed)
        #print("Printing Weights")
        #print(self.im_net.model[0].weight.data)
        #input()
        print("Printing fuse_embed layer")
        print(lin_op)
        input()
#        lin_vals=torch.tanh(lin_op)
        lin_vals=nn.Tanh()(lin_op)  
        class_embed=self.class_layer(lin_vals)
        class_vals=F.log_softmax(class_embed,dim=1)
        print("Printing class_embed layer")
        print(self.embed_layer.weight.data)
        input()
        print("Printing class_vals layer")
        print(self.class_layer.weight.data)
        input()
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


