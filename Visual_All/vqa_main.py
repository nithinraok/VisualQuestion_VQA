#repo for running the main part of VQA
#assume that vqa_utils is already run 
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from dataset_vqa import Dictionary, VQAFeatureDataset
from models import EncoderLSTM, FusionModule, EncoderCNN
from dataset_image_vqa import VQADataset
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.autograd import Variable
import pdb

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
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #torch.backends.cudnn.benchmark = True


    #extract weights from the weight matrices
    weights=np.load(args.file_name)

    # CUDA for PyTorch
    #if cuda:
    device=2
    torch.cuda.set_device(device)

    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #defining dictionary and VQAFeatureDataset
    #transforms for pretrained network(transform for resnet now)
    train_transform = transforms.Compose([ 
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    validate_transform=transforms.Compose([ 
        transforms.Resize((args.crop_size,args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dataset = VQADataset(image_root_dir=args.img_root_dir,dictionary=dictionary,dataroot=args.data_root_dir,choice='train',transform_set=train_transform)
    # eval_dataset = VQADataset(image_root_dir=args.img_root_dir,dictionary=dictionary,dataroot=args.data_root_dir,choice='val',transform_set=validate_transform)
    

    #model definition 
    print('Loading the models')
    image_encoder=EncoderCNN(embed_size=args.img_feats).to(device)
    question_encoder=EncoderLSTM(hidden_size=args.num_hid,weights_matrix=weights,fc_size=args.q_embed,max_seq_length=args.max_sequence_length,batch_size=args.batch_size).to(device)
    fusion_network=FusionModule(qnetwork=question_encoder,img_network=image_encoder,fuse_embed_size=args.fuse_embed,input_fc_size=args.img_feats,class_size=args.num_class).to(device)
    #print(list(fusion_network.parameters()))
    print(fusion_network)
    #input()
    

    #Dataloader initialization
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=16)
    # eval_loader =  DataLoader(eval_dataset, args.batch_size, shuffle=True, num_workers=1)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #params=lis
    #params = list(image_encoder.linear.parameters())+list(image_encoder.bn.parameters())+list(question_encoder.parameters()) + list(fusion_network.parameters()) 
    optimizer = torch.optim.RMSprop(fusion_network.parameters(), lr=args.learning_rate)

    # Train the models
    total_step = len(train_loader)
    step=0
    #Training starts
    #print('Training Starting ......................')

    def evaluate_val(model,train_loader,criterion,device):
        loss=0
        accuracy=0
        with torch.no_grad():
            for image_sample,question_token,labels in iter(train_loader):
                image_sample,question_token,labels = image_sample.to(device),question_token.to(device),labels.to(device)
                output=model.forward(question_token,image_sample)
                loss+= criterion(output,labels).item()
                ps = torch.exp(output)
                equality= (labels.data == ps.max(dim=1)[1])
                accuracy+=equality.type(torch.FloatTensor).mean()
        return loss,accuracy

    file_train=open('train_loss_log.txt','a+')
    loss_save=[]

    for epoch in range(args.epochs):

        running_loss = 0.0
        running_corrects = 0
        step=0
        for data in tqdm(train_loader):
            image_samp,question_toks,labels=data
            image_samp=image_samp.to(device)
            question_toks=question_toks.to(device)
            labels=labels.to(device)
            
            class_outputs=fusion_network(question_toks,image_samp)
            _, preds = torch.max(class_outputs, 1)
            loss = criterion(class_outputs, labels)
            #question_encoder.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('Enter some key')
            #input()
            # statistics
            running_loss += loss.item() * image_samp.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if(step%300==0):
            #optimizer.zero_grad()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch, args.epochs, step, total_step, loss.item()))
            step=step+1
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(epoch_loss)
        #loss_save.append(val_loss)
        
        val_loss,accuracy = evaluate_val(fusion_network,train_loader,criterion,device)
        string='Epoch {}:{} loss: {} \t'.format(epoch,args.epochs,running_loss)
        string+='Accuracy : '.format(accuracy)
        file_train.write(string)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
    file_train.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_hid', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--img_root_dir', type=str, default="/data/digbose92/VQA/COCO", help='location of the COCO images')
    parser.add_argument('--data_root_dir', type=str, default="/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data", help='location of the associated data')
    #parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--file_name', type=str, default="data/glove6b_init_300d.npy")
    parser.add_argument('--output', type=str, default='saved_models')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_sequence_length', type=int, default=14)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--q_embed',type=int, default=1024, help='embedding output of the encoder RNN')
    parser.add_argument('--img_feats',type=int, default=1024, help='input feature size of the image space')
    parser.add_argument('--fuse_embed',type=int, default=1000, help='Overall embedding size of the fused network')
    parser.add_argument('--num_class',type=int, default=2, help='Number of output classes')
    parser.add_argument('--learning_rate',type=float,default=0.01,help='Learning rate')
    args = parser.parse_args()
    main(args)






