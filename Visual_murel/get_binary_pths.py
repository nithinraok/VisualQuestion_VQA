import sys
import torch as t
file=sys.argv[1]
a=t.load(file)
a_bin=a.copy()
a_ann0=[]
a_ques=[]

for i,ann in enumerate(a['annotations']):
    if(ann['answer_id']<=1):
        a_ann0.append(ann)
        a_ques.append(a['questions'][i])


a_bin['questions']=a_ques
a_bin['annotations']=a_ann0
savefile=file.split('.')[0]+'_bin.pth'
t.save(a_bin,savefile)
