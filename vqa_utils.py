import json
import os 
import numpy as np 
from dataset_vqa import Dictionary 
from compute_softscore import *
from image_feature_extractor import * 

def create_dictionary(dataroot):
    """Creates a dictionary object for future usage
    """
    dictionary = Dictionary()
    #questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary

def create_glove_embedding_init(idx2word, glove_file):
    """creates the glove embedding matrix for all the words in the questions
    """
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    
    for idx, word in enumerate(idx2word):
        #print(idx,word)
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

def main_run(dataroot,pkl_filename,glove_filename,filenames_dict,image_filenames_dict,emb_dim=300):

    dictionary=create_dictionary(dataroot)
    dictionary.dump_to_file(os.path.join('data',pkl_filename))
    d = Dictionary.load_from_file((os.path.join('data',pkl_filename)))
    print(d.idx2word)
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_filename)
    np.save('data/glove6b_init_%dd.npy' % emb_dim, weights)

    #extract the raw data from json
    train_questions = json.load(open(filenames_dict['train_question_file']))['questions']
    train_answers = json.load(open(filenames_dict['train_answer_file']))['annotations']
    validation_questions = json.load(open(filenames_dict['validation_question_file']))['questions']
    validation_answers = json.load(open(filenames_dict['validation_answer_file']))['annotations']

    #generate the question labels and the id maps 
    answers = train_answers + validation_answers
    occurence = filter_answers(answers, 9)
    ans2label = create_ans2label(occurence, 'trainval')
    train_target=compute_target(train_answers, ans2label, 'train')
    validation_target=compute_target(validation_answers, ans2label, 'val')


    #image feature extraction here based on functions in image_feature_extractor
    image_feats_converter(image_filenames_dict)




if __name__ == "__main__":
    user="digbose92"
    if user=="nithin_rao":
        root_folder = "/Users/nithin_rao/MyFiles/MS/USC/Spring_2019/CSCI_599_DL/Project/VisualQuestion_VQA/"
    else:
        root_folder = "/data/digbose92"
        
    dataroot=root_folder+"/VQA/questions_answers"
    pkl_file='dictionary.pkl'
    glove_filename=root_folder+"/VQA/glove_dataset/data/glove/glove.6B.300d.txt"

    train_questions_filenames=os.path.join(dataroot,"v2_OpenEnded_mscoco_train2014_questions.json")
    train_answer_filenames=os.path.join(dataroot,"v2_mscoco_train2014_annotations.json")
    val_questions_filenames=os.path.join(dataroot,"v2_OpenEnded_mscoco_val2014_questions.json")
    val_answer_filenames=os.path.join(dataroot,"v2_mscoco_val2014_annotations.json")
    filenames_dict={'train_question_file':train_questions_filenames,'train_answer_file':train_answer_filenames,'validation_question_file':val_questions_filenames,'validation_answer_file':val_answer_filenames}
    
    image_filenames_dict={'train_data_file':'data/train36.hdf5','val_data_file':'data/val36.hdf5','train_ids_file':'data/train_ids.pkl','val_ids_file':'data/val_ids.pkl','infile':'/data/digbose92/VQA/image_features/data/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv','train_indices_file':'data/train36_imgid2idx.pkl','val_indices_file':'data/val36_imgid2idx.pkl'}
    main_run(dataroot,pkl_file,glove_filename,filenames_dict,image_filenames_dict)

    

