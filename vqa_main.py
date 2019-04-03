#repo for running the main part of VQA
#assume that vqa_utils is already run 
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from dataset_vqa import Dictionary, VQAFeatureDataset
from models import EncoderLSTM, FusionModule


def question_parse(token_list):
    data=pickle.load(open('data/dictionary.pkl','rb'))
    index2word_map=data[1]
    word_list=[]

    for idval in token_list.tolist():
        if(idval==19901):
            word_list.append(index2word_map[idval-1])
        else:
            word_list.append(index2word_map[idval])
    #word_list=[index2word_map[id] for id in token_list.tolist()]
    print(word_list)

def preproc_question_tokens(question_array):

    num_questions,seq_length=question_array.shape
    for i in np.arange(num_questions):
        index=np.where(question_array==19901)
        question_array[index]=19900
    return(question_array)

def convert_one_hot2int(one_hot):
    one_hot=one_hot.astype(int)
    class_ind=np.argmax(one_hot,axis=1)
    return(class_ind)

def main(args):

    #defining torch configurations
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True


    #extract weights from the weight matrices
    weights=np.load(args.file_name)

    # CUDA for PyTorch
    #if cuda:
    device=2
    torch.cuda.set_device(device)
    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #defining dictionary and VQAFeatureDataset
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dataset = VQAFeatureDataset('train', dictionary)
    eval_dataset = VQAFeatureDataset('val', dictionary)
    

    #model definition 
    question_encoder=EncoderLSTM(hidden_size=args.num_hid,weights_matrix=weights,fc_size=args.q_embed,max_seq_length=args.max_sequence_length,batch_size=args.batch_size).to(device)
    fusion_network=FusionModule(fuse_embed_size=args.q_embed,fc_size=args.fuse_embed,class_size=args.num_class).to(device)

    print(question_encoder)
    print(fusion_network)
    input()
    

    #Dataloader initialization
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dataset, args.batch_size, shuffle=True, num_workers=1)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(question_encoder.parameters()) + list(fusion_network.parameters()) 
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(train_loader)
    step=0
    #Training starts
    for epoch in range(args.epochs):
        for i, (image_features,spatials,question_tokens,labels) in enumerate(train_loader):
            class_indices=convert_one_hot2int(labels.numpy())
            image_feats=torch.mean(image_features,dim=1)
            image_feats=image_feats.to(device)
            class_indices=torch.from_numpy(class_indices).long().to(device)
            #labels=labels.to(device)

            #preproc the tokens after converting from tensor to numpy. Then numpy to tensor before passing to loss fn
            question_array=preproc_question_tokens(question_tokens.cpu().numpy())
            question_tokens=torch.from_numpy(question_array).to(device)
            
            #fusion_network.zero_grad()
            optimizer.zero_grad()
            #Forward, Backward and Optimize
            question_features=question_encoder(question_tokens)
            class_outputs=fusion_network(question_features,image_feats)

            loss = criterion(class_outputs, class_indices)
            #question_encoder.zero_grad()
            loss.backward()
            optimizer.step()
            if(step%1000==0):
            #optimizer.zero_grad()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.epochs, i, total_step, loss.item())) 
            step=step+1
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=512)
    #parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--file_name', type=str, default="data/glove6b_init_300d.npy")
    parser.add_argument('--output', type=str, default='saved_models')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_sequence_length', type=int, default=14)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--q_embed',type=int, default=2048, help='embedding output of the encoder RNN')
    parser.add_argument('--fuse_embed',type=int, default=1024, help='Overall embedding size of the fused network')
    parser.add_argument('--num_class',type=int, default=3129, help='Number of output classes')
    parser.add_argument('--learning_rate',type=float,default=0.01,help='Learning rate')
    args = parser.parse_args()
    main(args)






