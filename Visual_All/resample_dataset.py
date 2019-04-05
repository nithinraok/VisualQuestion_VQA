import pickle 
import pandas as pd 
from tqdm import tqdm

pkl_file='data/cache/train_target.pkl'


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


#print(len(entry_filter_list))
with open('data/cache/train_target_yes_no.pkl','wb') as f:
    pickle.dump(entry_filter_list,f)


