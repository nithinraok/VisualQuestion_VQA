import sys
sys.path.insert(0,'/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All')
from torch.utils.data import Dataset, DataLoader
from model_combined import *
from vqa_dataset_attention import *
from dataset_vqa import Dictionary, VQAFeatureDataset
import torch
from collections import OrderedDict 
import argparse 
import torch.nn.parallel
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torchvision import models
import torch.nn as nn
import json
from PIL import Image
import pickle 
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""class resnet_intermediate(nn.Module):
     def __init__(self, vqamodel):
        super(resnet_intermediate, self).__init__()

        self.model=vqamodel """

def model_extract(model_str,layer_option="embedding"):

    if(model_str=="vgg16"):
        model=models.vgg16(pretrained=True)
        if(layer_option=="embedding"):
            """extracts the image embeddings
            """
            features = list(model.classifier.children())[:-1]
            model.classifier=nn.Sequential(*features)
            
        elif(layer_option=="pool"):
            """extracts the pooled features
            """
            features=nn.Sequential(*list(model.features.children())[:-1])
            model=features

        for param in model.parameters():
                param.require_grad=False

    elif(model_str=="resnet152"):
        model=models.resnet152(pretrained=True)
        if(layer_option=="embedding"):
            """extracts the image embeddings
            """
            features=nn.Sequential(*list(model.children())[:-1])
            model=features
            print(model)
        
        elif(layer_option=="pool"):
            """extracts the pooled features
            """
            features=nn.Sequential(*list(model.children())[:-2])
            model=features

        for param in model.parameters():
                param.require_grad=False

    return(model)


def model_regen(args):
    print('Loading model checkpoint')
    attention_model_checkpoint=torch.load(args.model_path)
    new_state_dict = OrderedDict()
    for k, v in attention_model_checkpoint.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    print('Model checkpoint loaded')
    
    print(new_state_dict.keys())
    dictionary=Dictionary.load_from_file(args.pickle_path)

    train_dataset=Dataset_VQA(img_root_dir=args.image_root_dir,feats_data_path=args.feats_data_path,dictionary=dictionary,choice='train',dataroot=args.data_root,arch_choice=args.arch_choice,layer_option=args.layer_option)
    print('Loading the attention model')
    attention_model = attention_baseline(train_dataset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, drop_L=args.dropout_L, drop_G=args.dropout_G,\
                               drop_W=args.dropout_W, drop_C=args.dropout_C)
    attention_model.load_state_dict(new_state_dict)
    attention_model.eval()
    
    #print('Saving the entire model')
    #torch.save(attention_model,'gradcam_models/resnet_152_attention_baseline_model.pth')
    return(attention_model)

class VQA_Model_combined(nn.Module):
    def __init__(self,args):
        torch.cuda.manual_seed_all(args.seed)
        super(VQA_Model_combined, self).__init__()
        model=models.resnet152(pretrained=True)
        self.img_model = nn.Sequential(*list(model.children())[:-2])
        self.img_model.train(False)
        attention_model_checkpoint=torch.load(args.model_path)
        new_state_dict = OrderedDict()
        for k, v in attention_model_checkpoint.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        print('Model checkpoint loaded')
    
        print(new_state_dict.keys())
        dictionary=Dictionary.load_from_file(args.pickle_path)

        train_dataset=Dataset_VQA(img_root_dir=args.image_root_dir,feats_data_path=args.feats_data_path,dictionary=dictionary,choice='train',dataroot=args.data_root,arch_choice=args.arch_choice,layer_option=args.layer_option)
        print('Loading the attention model')
        attention_model = attention_baseline(train_dataset, num_hid=args.num_hid, dropout= args.dropout, norm=args.norm,\
                               activation=args.activation, drop_L=args.dropout_L, drop_G=args.dropout_G,\
                               drop_W=args.dropout_W, drop_C=args.dropout_C)
        attention_model.load_state_dict(new_state_dict)
        self.vqa_model=attention_model
        self.vqa_model.train(False)
        #self.attention_model.train(False)
        
        #
        # 
        # 
        # 
        # self.vqa_model = model_regen(args)
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    
    def forward(self, img, q_tok):
        """Forward
        return: logits, not probs
        """
        img_feats=self.img_model(img)
        h = img_feats.register_hook(self.activations_hook)
        print(img_feats.size())
        img_feats=img_feats.view(img_feats.size(0),img_feats.size(1),img_feats.size(2)*img_feats.size(3))
        img_feats=img_feats.transpose(2,1)
        #resize the image features 
        logits = self.vqa_model(img_feats,q_tok)

        return logits


    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.img_model(x)

def preproc_image(filepath):
    im=Image.open(filepath)
    im=im.convert('RGB')
    im_array=np.array(im)
    transform = transforms.Compose([ 
        transforms.Resize((224,224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    image=transform(im)
    return(image)

def preproc_question(str,max_length,dictionary):
    tokens = dictionary.tokenize(str, False)
    tokens = tokens[:max_length]
    if len(tokens) < max_length:
        padding = [dictionary.padding_idx] * (max_length - len(tokens))
        tokens = padding + tokens
    return(tokens)



if __name__ == "__main__":
    #parser.add_argument('--epochs', type=int, default=40)
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root_dir', type=str, default="/data/digbose92/VQA/COCO")
    parser.add_argument('--pickle_path', type=str, default="../Visual_All/data/dictionary.pkl")
    parser.add_argument('--feats_data_path', type=str, default="/data/digbose92/VQA/COCO/train_hdf5_COCO/")
    parser.add_argument('--data_root', type=str, default="/proj/digbose92/VQA/VisualQuestion_VQA/common_resources")
    parser.add_argument('--npy_file', type=str, default="../../VisualQuestion_VQA/Visual_All/data/glove6b_init_300d.npy")
    parser.add_argument('--model_path', type=str, default="results_GRU_uni/results_resnet_152_hid_512_YES_NO_ADAM/model.pth")
    parser.add_argument('--image_model', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_hid', type=int, default=512) # they used 1024
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dropout_L', type=float, default=0.1)
    parser.add_argument('--dropout_G', type=float, default=0.2)
    parser.add_argument('--dropout_W', type=float, default=0.4)
    parser.add_argument('--dropout_C', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='PReLU, ReLU, LeakyReLU, Tanh, Hardtanh, Sigmoid, RReLU, ELU, SELU')
    parser.add_argument('--norm', type=str, default='weight', help='weight, batch, layer, none')
    parser.add_argument('--choice', type=str, default='val', help='choice of the split')
    parser.add_argument('--seed', type=int, default=9731, help='random seed')
    parser.add_argument('--arch_choice', type=str, default='resnet152', help='choice of the network')
    parser.add_argument('--layer_option', type=str, default='pool', help='choice of the layer')
    parser.add_argument('--num_workers', type=int, default=4, help='number of the workers')
    parser.add_argument('--device', type=int, default=0, help='GPU device id')
    parser.add_argument('--class_metadata_file', type=str, default='/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/Train_Class_Distribution.csv', help='Path of class metadata file')
    parser.add_argument('--rcnn_path',type=str,default="/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/val36_imgid2idx.pkl",help="Path of the rcnn features file")
    parser.add_argument('--bert_option',type=bool,default=False,help="Whether to use bert or not")
    parser.add_argument('--mfb_out_dim', type=int, default=1000, help='mfb output dimension')
    args = parser.parse_args()


    

    VQA_model_exp=VQA_Model_combined(args)
    VQA_model_exp.to(args.device)
    #VQA_model_exp.img_model.eval()
    #VQA_model_exp.vqa_model.eval()
    #print(type(VQA_model_exp))
    #VQA_model_exp.eval()
    #VQA_model_exp.to(0)
    class_meta_data=pd.read_csv('/proj/digbose92/VQA/VisualQuestion_VQA/Visual_All/data/Train_Class_Distribution.csv')
    class_label_map=class_meta_data['Label_names'].tolist()

    print('Load the validation json file')
    valid_questions=json.load(open('/proj/digbose92/VQA/VisualQuestion_VQA/common_resources/v2_OpenEnded_mscoco_val2014_yes_no_questions.json'))['questions']

    valid_entry=valid_questions[77]
    dictionary=Dictionary.load_from_file('../Visual_All/data/dictionary.pkl')
    print(valid_entry['question'])
    tokens=preproc_question(valid_entry['question'],14,dictionary)
    print(tokens)
    #print(valid_entry)
    
    pkl_data=pickle.load(open('/proj/digbose92/VQA/VisualQuestion_VQA/common_resources/val_target_yes_no_ans.pkl','rb'))
    
    question_ids=[pkl_data[i]['question_id'] for i, question in enumerate(pkl_data)]
    #print(question_ids)
    
    id=question_ids.index(valid_entry['question_id'])
    #print(pkl_data[id])
    image_id=pkl_data[id]['image_id']
    
    choice='val'
    image_path='COCO_'+choice+'2014_'+str(image_id).zfill(12)+'.jpg'
    folder="/data/digbose92/VQA/COCO/val2014"

    print(os.path.join(folder,image_path))

    image_tensor=preproc_image(os.path.join(folder,image_path))
    img=cv2.imread(os.path.join(folder,image_path))

    image_tensor=image_tensor.unsqueeze(0)
    tokens=torch.from_numpy(np.array(tokens))

    tokens=tokens.unsqueeze(0)
    print(tokens.size())
    print(image_tensor.size())
    image_tensor=image_tensor.to(args.device)
    tokens=tokens.to(args.device)
    #image_tensor=image_tensor.to(0)
    #tokens=tokens.to(0)
    logit=VQA_model_exp(image_tensor,tokens)
    #print(logit)
    logit_max_location = torch.max(logit, 1)[1]
    print(logit_max_location)
    print('actual label:',class_label_map[pkl_data[id]['Class_Label']])
    print('predicted label:',class_label_map[logit_max_location])
    #print(pkl_data[id])
    #print(logits)
    #print(logit.size())
    """logit[:,logit_max_location].backward()
    #one_hot_output = torch.FloatTensor(1, logits.size()[-1]).zero_()
    #one_hot_output[0][1] = 1

    #VQA_model_exp.img_model.eval()
    #VQA_model_exp.vqa_model.eval()

    grad_val=VQA_model_exp.get_activations_gradient()
    print(grad_val.size())

    pooled_gradients = torch.mean(grad_val, dim=[0, 2, 3])

    activations=VQA_model_exp.get_activations(image_tensor).detach()

    for i in range(2048):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    heatmap=heatmap.numpy()

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap + img
    cv2.imwrite('heatmap.jpg', superimposed_img)

    #plt.matshow(heatmap.squeeze())
    #plt.show()
    #print(pooled_gradients.size())
    #print(activations.size())
    #print(image_tensor.size())"""
