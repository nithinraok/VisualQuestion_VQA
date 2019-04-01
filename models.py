import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class EncoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, use_gpu=True, word_embed_size=300,fc_size=1024, num_layers=2, max_seq_length=14, batch_size=32):
        """Module for stacked LSTM which returns a single hidden state for the input question
        """
        super(EncoderLSTM, self).__init__()
        self.use_gpu=use_gpu
        self.max_seg_length = max_seq_length
        self.batch_size=batch_size
        self.hidden_dim=hidden_size
        self.vocab_len=vocab_size
        self.embed_len=word_embed_size
        self.linear_fc_size=fc_size
        self.nlayers=num_layers
        self.timesteps=max_seq_length

        self.embed = nn.Embedding(self.vocab_len, self.embed_len)
        self.lstm = nn.LSTM(self.embed_len, self.hidden_dim, self.nlayers)
        self.linear = nn.Linear(self.hidden_dim, self.linear_fc_size)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                     Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, input_sentence):
        """Forward pass of the encoderLSTM network
        """
        #print(self.embed(input_sentence).size())
        input = self.embed(input_sentence).view(self.timesteps,self.batch_size,-1)
        print(input.size())
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        #print((self.hidden[0][-1].size()))
        linear_scores=self.linear(self.hidden[0][-1])
        return(linear_scores )

if __name__ == "__main__":
        test_torch_tensor=torch.tensor([[0,1,3],[0,3,1]],dtype=torch.long)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encode_test=EncoderLSTM(hidden_size=3,vocab_size=5,use_gpu=True,word_embed_size=4,max_seq_length=3,batch_size=2)
        encode_test=encode_test.to(device)
        test_torch_tensor=test_torch_tensor.to(device)
        linear_scores=encode_test.forward(test_torch_tensor)
        
        #print(lstm_out.size())