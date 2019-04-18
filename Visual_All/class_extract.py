import json
import pandas as pd 
import numpy as np 
import pickle 
from collections import Counter
import operator
from collections import OrderedDict
from tqdm import tqdm
import os

def modify_entry(label_list,entry):
    label_set=entry['labels']
    label_name=[label_list[val] for id,val in enumerate(label_set)]
    entry['Label_name']=label_name
    return(entry)

pickle_file="../common_resources/train_target.pkl"
val_pickle_file="../common_resources/val_target.pkl"
label_map_file="../common_resources/trainval_label2ans.pkl"
num_classes_select=2
dataroot='data'

data=pickle.load(open(pickle_file,'rb'))
answer_data=pickle.load(open(label_map_file,'rb'))
validation_data=pickle.load(open(val_pickle_file,'rb'))

#print(len(data))
label_list=[]
print(answer_data)


for data_sample in data:
    #print(data_sample.keys())
    score=data_sample['scores']
    labels=data_sample['labels']
    if(len(score)>0):
        max_id=score.index(max(score))
        label_list.append(labels[max_id])
    else:
        pass

print(len(label_list))
count_set=Counter(label_list)
y = OrderedDict(count_set.most_common())
#scount_set=OrderedDict(label_list)
keys_set=list(y.keys())
keys_name=[answer_data[id] for id in keys_set]

dict_set=dict(zip(keys_name,list(y.values())))
#sorted_x = sorted(dict_set.items(), key=operator.itemgetter(1))

df=pd.DataFrame({'Label_names':keys_name,'Occurences':list(dict_set.items()),'Label_indices':keys_set})
df.to_csv('data/Train_Class_Distribution.csv',columns=['Label_names','Label_indices','Occurences'],index=False)


entry_1000_classes=[]
class_list_set=set(keys_set[0:num_classes_select])

print('Sampling training data')
label_sampled=[]
for data_sample in tqdm(data):
    #print(data_sample.keys())
    score=data_sample['scores']
    labels=data_sample['labels']
    if(len(score)>0):
        max_id=score.index(max(score))
        label_question=labels[max_id]
        if(label_question in class_list_set):
            label_sampled.append(label_question)
            data_sample=modify_entry(answer_data,data_sample)
            entry_1000_classes.append(data_sample)

print('Sampling validation data')
label_validation_sampled=[]
entry_validation_1000_classes=[]
for data_sample in tqdm(validation_data):
    #print(data_sample.keys())
    score=data_sample['scores']
    labels=data_sample['labels']
    if(len(score)>0):
        max_id=score.index(max(score))
        label_question=labels[max_id]
        if(label_question in class_list_set):
            label_validation_sampled.append(label_question)
            data_sample=modify_entry(answer_data,data_sample)
            entry_validation_1000_classes.append(data_sample)


label_sample_set=list(set(label_sampled))
print(len(label_sample_set))
print(len(list(class_list_set)))

intersect_list=set(label_sampled).intersection(list(class_list_set))


with open('../common_resources/train_target_yes_no_bin.pkl','wb') as f:
    pickle.dump(entry_1000_classes,f)
with open('../common_resources/validation_target_yes_no_bin.pkl','wb') as f:
    pickle.dump(entry_validation_1000_classes,f)



print('Resampling the training json data')
train_question_path = os.path.join(
        dataroot, 'OpenEnded_mscoco_train2014_questions.json')
train_questions = sorted(json.load(open(train_question_path))['questions'],
                       key=lambda x: x['question_id'])


print('Finding the matches in training data')
question_train_tot_id=[entr['question_id'] for entr in train_questions]
question_train_sample_id=[entr['question_id'] for entr in entry_1000_classes]
set_sample_train=set(question_train_sample_id)
id_matches_train=[id for id,val in enumerate(question_train_tot_id) if val in set_sample_train]
question_train_set=[train_questions[id_new] for id_new in id_matches_train]
print(len(question_train_set))
print(len(entry_1000_classes))
train_questions_dict={}
train_questions_dict['questions']=question_train_set
with open('data/OpenEnded_mscoco_train2014_2_questions.json', 'w') as fp:
    json.dump(train_questions_dict, fp)



print('Resampling the validation json data')
valid_question_path = os.path.join(
        dataroot, 'OpenEnded_mscoco_val2014_questions.json')
valid_questions = sorted(json.load(open(valid_question_path))['questions'],
                       key=lambda x: x['question_id'])


print('Finding the matches in validation data')
question_valid_tot_id=[entr['question_id'] for entr in valid_questions]
question_valid_sample_id=[entr['question_id'] for entr in entry_validation_1000_classes]
set_sample_valid=set(question_valid_sample_id)
id_matches_valid=[id for id,val in enumerate(question_valid_tot_id) if val in set_sample_valid]
question_valid_set=[valid_questions[id_new] for id_new in id_matches_valid]
print(len(question_valid_set))
print(len(entry_validation_1000_classes))
valid_questions_dict={}
valid_questions_dict['questions']=question_valid_set
with open('data/OpenEnded_mscoco_val2014_2_questions.json', 'w') as fp:
    json.dump(valid_questions_dict, fp)




#resampling to 1000 classes

#iterator = iter(dict_set.items())
#for i in range(5):
#    print(next(iterator))

#df = pd.DataFrame.from_dict(d, orient="index")

#print(list(count_set.values())[0:100])















