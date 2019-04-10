#repo for running the main part of VQA
#assume that vqa_utils is already run 
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models,transforms
import numpy as np
from tqdm import tqdm
from dataset_vqa_binary import Dictionary, VQAFeatureDataset
from models import EncoderLSTM, FusionModule,LinearImageModel,Vgg16_4096,savemodel

def main(args):

    #defining torch configurations
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    #transforms
    tfms = transforms.Compose([transforms.ToPILImage(),
                           transforms.Resize((224,224)),
                            transforms.ToTensor(),
                          transforms.Normalize((0.485, 0.456, 0.406),
                                               (0.229, 0.224, 0.225))])

    #extract weights from the weight matrices
    weights=np.load(args.file_name)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #defining dictionary and VQAFeatureDataset
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dataset = VQAFeatureDataset('train', dictionary,tfms=tfms)
    eval_dataset = VQAFeatureDataset('val', dictionary,tfms=tfms)
    

    #model definition 
    image_model = LinearImageModel(n_input=4096,n_output=1024)
    question_encoder=EncoderLSTM(hidden_size=args.num_hid,weights_matrix=weights,train_embed=True,use_gpu=True,
                                fc_size=args.q_embed,max_seq_length=args.max_sequence_length,
                                batch_size=args.batch_size).to(device)
    fusion_network=FusionModule(qnetwork=question_encoder,img_network=image_model,
                    fuse_embed_size=1024,fc_size=512).to(device)

    #Dataloader initialization
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2)
    eval_loader =  DataLoader(eval_dataset, args.batch_size, shuffle=True, num_workers=2)

    # Loss and optimizer
    criterion = nn.NLLLoss()
    # params = list(image_model.parameters())+list(question_encoder.parameters()) + list(fusion_network.parameters()) 
    optimizer = torch.optim.Adamax(fusion_network.parameters(), lr=args.learning_rate)
    
    # Train the models
    total_step = len(train_loader)

    def evaluate_val(model,train_loader,criterion,device):
        print("Evaluating Validation Loader")
        loss=0
        accuracy=0
        with torch.no_grad():
            for image_sample,question_token,labels in iter(train_loader):
                image_sample,question_token,labels = image_sample.to(device),question_token.to(device),labels.to(device)
                output=model.forward(question_token,image_sample)
                ps = torch.exp(output)
                if(loss==0):
                    print("Output : \n ",ps)
                loss+= criterion(output,labels).item()

                equality= (labels.data == ps.max(dim=1)[1])
                accuracy+=equality.type(torch.FloatTensor).mean()
        return loss,accuracy
    
    logger=open('train_loss_log.txt','w')
    loss_save=[]
#    img_sample, ques_token, target=next(iter(train_loader))

    #Training starts
    fusion_network.to(device) 
    #val_loss,accuracy = evaluate_val(fusion_network,eval_loader,criterion,device)
    for epoch in range(args.epochs):
        running_loss=0
        step=0
            
        #print("Val Loss: {} Accuracy :{} ".format(val_loss,accuracy))
        for img_sample, ques_token, target in tqdm(train_loader):
            # print("Image file  size  : ",img_sample.shape)
            # print("Question token: ",ques_token.shape)
            # print("target :",target)


            image_feats=img_sample.to(device)
            question_tokens=ques_token.to(device)
            target=target.to(device)
            # #Forward, Backward and Optimize
            optimizer.zero_grad()
            class_outputs=fusion_network(question_tokens,image_feats)
            loss = criterion(class_outputs, target)
            loss.backward()
            nn.utils.clip_grad_norm_(fusion_network.parameters(), 0.25)
            optimizer.step()

            running_loss+=loss.item()*image_feats.size(0)
            if(step%20==0):
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.epochs, step, total_step, loss.item())) 
            step+=1
        
        epoch_loss=running_loss/len(train_dataset)
        print("Train Epoch Loss: ",epoch_loss)
        
        val_loss,accuracy = evaluate_val(fusion_network,eval_loader,criterion,device)
        string='Epoch {}:{} loss: {} \t'.format(epoch,args.epochs,val_loss)
        string+='Accuracy : {}\n'.format(accuracy)
        print(string)
        logger.write(string)
#        savemodel(image_model,device,"image_model")
#        savemodel(question_encoder,device,"question_encoder")
        savemodel(fusion_network,device,"fusion_network")
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=512)
    #parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--file_name', type=str, default="data/glove6b_init_300d.npy")
    parser.add_argument('--output', type=str, default='saved_models')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_sequence_length', type=int, default=14)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--q_embed',type=int, default=1024, help='embedding output of the encoder RNN')
    parser.add_argument('--fuse_embed',type=int, default=512, help='Overall embedding size of the fused network')
    parser.add_argument('--num_class',type=int, default=2, help='Number of output classes')
    parser.add_argument('--learning_rate',type=float,default=2*0.001,help='Learning rate')
    args = parser.parse_args()
    main(args)






