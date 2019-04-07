import pickle 
import pandas as pd 
from tqdm import tqdm
import json
import time

pkl_file='data/cache/val_target.pkl'
question_path='data/v2_OpenEnded_mscoco_val2014_questions.json'


id_data=pickle.load(open(pkl_file,'rb'))

yes_lab=3
no_lab=9

entry_list=[]
id_list=[]
for index,data in enumerate(id_data):
    lab=data['labels']
    try:
        if(lab.index(yes_lab) is not None or lab.index(no_lab) is not None):
            print(lab) 
            id_list.append(index)
            entry_list.append(data)
    except:
        print('Not here')

print(len(entry_list))

entry_filter_list=[]

for entry_val in entry_list:
    score=entry_val['scores']
    max_id=score.index(max(score))
    if(entry_val['labels'][max_id]==3 or entry_val['labels'][max_id]==9):
        entry_filter_list.append(entry_val)


questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
question_id_tot=[question['question_id'] for question in questions]
question_id_filter=[entr['question_id'] for entr in entry_filter_list]
question_id_filter.sort()
question_id_set=set(question_id_filter)
print('Finding matches')
start=time.time()
id_set=[id for id,val in enumerate(question_id_tot) if val in question_id_set]
end=time.time()
time=end-start
print('Time elapsed:%f' %(time))
#print(len(entry_filter_list))
with open('data/cache/val_target_yes_no.pkl','wb') as f:
    pickle.dump(entry_filter_list,f)




