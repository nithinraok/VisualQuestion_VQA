import sys
import pickle
file=sys.argv[1]
a=pickle.load(open(file,'rb'))

b=a.copy()

for i,id in enumerate(a):
    for il,k in enumerate(id['labels']):
        if(k==3):
            b[i]['labels'][il]=0
        elif(k==9):
             b[i]['labels'][il]=1

w_file=file.split('.')[0]+'_bin.pkl'
pickle.dump(b,open(w_file,'wb'))
