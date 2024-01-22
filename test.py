import pickle,os
from multiprocessing import Pool
from tqdm import tqdm
import gc
corpus_path='/home/sysevr/Implementation/corpus'
vector_path='/home/sysevr/Implementation/vector'
def insert_focus(params):
    vector_path,corpus_path,cve,filename=params
    try:
        with open(vector_path+'/'+cve+'/'+filename,'rb') as f:
                data=pickle.load(f)
        if len(data)<5:
            if os.path.exists(corpus_path+'/'+cve+'/'+filename):
                with open(corpus_path+'/'+cve+'/'+filename,'rb') as f:
                    data2=pickle.load(f)
                data.insert(2,data2[2])
                with open(vector_path+'/'+cve+'/'+filename,'wb')as f:
                    pickle.dump(data,f)
                # del data2,data
                # gc.collect()
            else:
                print(corpus_path+'/'+cve+'/'+filename)
    except:
        print(cve+'/'+filename)
        
params=[]
for cve in os.listdir(vector_path):
    for filename in os.listdir(vector_path+'/'+cve):
        params.append((vector_path,corpus_path,cve,filename))
pbar=tqdm(total=len(params))
with Pool(32) as pool:
    rets=pool.imap_unordered(insert_focus,params)
    for ret in rets:
        pbar.update(1)