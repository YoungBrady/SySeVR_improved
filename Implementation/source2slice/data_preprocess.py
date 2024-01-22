## coding:utf-8

import pickle
import os
import json
from tqdm  import tqdm
import random
from multiprocessing import Pool
def handle_cve(params):
    software_id,label_dict=params
    slice_path = '../../slice_all/'
    folder_path = '../label_source/'
    hash2tup={}
    for type in os.listdir(slice_path):
        for batch in os.listdir(os.path.join(slice_path,type)):
            for software in os.listdir(os.path.join(slice_path,type,batch)):
                if software !=software_id:
                    continue
                for cve in os.listdir(os.path.join(slice_path,type,batch,software)):
                    if not os.path.exists(os.path.join(slice_path,type,batch,software,cve,'slice_source')):
                        continue
                    os.makedirs(os.path.join(folder_path,cve),exist_ok=True)
                    
                    for filename in ['api_slices.txt','arraysuse_slices.txt','integeroverflow_slices.txt','pointersuse_slices.txt']:
                        # try:
                            filepath = os.path.join(slice_path,type,batch,software,cve,'slice_source',filename)
                            if not os.path.exists(filepath):
                                continue
                            try:
                                f = open(filepath,'r')
                                slicelists = f.read().split('------------------------------\n')
                                f.close()
                            except:
                                continue
                            
                            if slicelists[0] == '':
                                del slicelists[0]
                            if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
                                del slicelists[-1]


                            for slicelist in slicelists:
                                # try:
                                    sentences = slicelist.split('\n')
                                    if sentences[0] == '\r' or sentences[0] == '':
                                        del sentences[0]
                                    if sentences == []:
                                        continue
                                    if sentences[-1] == '':
                                        del sentences[-1]
                                    if sentences[-1] == '\r':
                                        del sentences[-1]
                                    key = sentences[0]
                                    if key not in label_dict:
                                        continue
                                    if len(sentences)<=6:
                                        continue
                                    hash_value=hash(" ".join(sentences[1:]))
                                    label = label_dict[key]
                                    if hash_value in hash2tup:
                                        if label == 0:
                                            continue
                                    hash2tup[hash_value]={
                                            "slice":sentences,
                                            "label":label,
                                            "filepath":os.path.join(folder_path,cve,filename)
                                        }
    samples0=[]
    samples1=[]
    for hash_value in hash2tup:
        if hash2tup[hash_value]["label"]==0:
            samples0.append(hash2tup[hash_value])
        else:
            samples1.append(hash2tup[hash_value])
    random.shuffle(samples0)
    random.shuffle(samples1)
    min_len=min(len(samples0),len(samples1))
    print(software_id,len(samples0),len(samples1))
    samples0=samples0[:min_len]
    samples1=samples1[:min_len]
    final=samples0+samples1
    for sample in final:
        with open(sample["filepath"],'a+') as f:
            for sentence in sample['slice']:
                f.write(str(sentence)+'\n')
            f.write(str(sample['label'])+'\n')
            f.write('------------------------------'+'\n')
    return software_id,len(final)
if __name__ == '__main__':
    os.chdir("/home/sysevr/Implementation/source2slice")
    slice_path = '../../slice_all/'
    label_path = '../labels/labels.json'
    folder_path = '../label_source/'
    os.makedirs(folder_path, exist_ok=True)
    with open(label_path, 'r') as f:
        labellists = json.load(f)
    
    total_len=0
    params=[]
    software_set=set()
    for type in os.listdir(slice_path):
        for batch in os.listdir(os.path.join(slice_path,type)):
            for software in os.listdir(os.path.join(slice_path,type,batch)):
                software_set.add(software)
    labels={} 
    for key in tqdm(labellists.keys()):
        try:
            software=key.split('/')[7]
            if software not in labels:
                labels[software]={}
            labels[software][key]=labellists[key]
        except:
            continue
    for software in tqdm(software_set):
        if software not in labels:
            continue
        params.append((software,labels[software]))
    pbar=tqdm(total=len(params))
    with Pool(16)as p:
        ret=p.imap_unordered(handle_cve,params)
        for r in ret:
            pbar.set_description(r[0]+" "+str(r[1]))
            pbar.update(1)
    

print('\success!')
                            
            
    
    
