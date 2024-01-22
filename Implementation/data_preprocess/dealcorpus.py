import pickle
from tqdm import tqdm
from multiprocessing import Pool
import os
def deal_single_file(file_path):
    maxlen=100
    not_change_num=0
    cut_num=0
    del_num=0
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)
        except EOFError:
            os.remove(file_path)
            print("rm file:",file_path)
            return None
    if len(data)<5:
        os.remove(file_path)
        print("rm file:",file_path)
        
        return None
    # elif  len(data)==5 and len(data[0])==5:
    #     is_wrong=False
    #     for label in data[1]:
    #         if type(label)!=int:
    #             is_wrong=True
    #     if is_wrong:
    #         new_slicefile_corpus,new_slicefile_labels,new_slicefile_focus,new_slicefile_func,new_slicefile_filenames=[],[],[],[],[]
    #         for corpus,label,focus,func,filename in data:
    #             new_slicefile_corpus.append(corpus)
    #             new_slicefile_labels.append(label)
    #             new_slicefile_focus.append(focus)
    #             new_slicefile_func.append(func)
    #             new_slicefile_filenames.append(filename)
    #         with open(file_path, 'wb') as file:
    #             pickle.dump([new_slicefile_corpus,new_slicefile_labels,new_slicefile_focus,new_slicefile_func,new_slicefile_filenames], file)
    #         return None
            
    else:
        cut_num=0
        del_num=0
        not_change_num=0
        new_slicefile_corpus,new_slicefile_labels,new_slicefile_focus,new_slicefile_func,new_slicefile_filenames=[],[],[],[],[]
        hashset=set()

        slicefile_corpus,slicefile_labels,slicefile_focus,slicefile_func,slicefile_filenames=data

        # pbar=tqdm(range(len(slicefile_corpus)))
        for corpus,label,focus,func,filename in zip(slicefile_corpus,slicefile_labels,slicefile_focus,slicefile_func,slicefile_filenames):
            if type(corpus)==int:
                print(file_path)
            if len(corpus)<maxlen:
                not_change_num+=1
            elif len(corpus)==maxlen:
                not_change_num+=1
                hash_value=hash("".join(corpus))
                hashset.add(hash_value)
            elif len(corpus)>maxlen:
                startpoint = int(focus - round(maxlen / 2.0))
                endpoint =  int(startpoint + maxlen)
                if startpoint < 0:
                    startpoint = 0
                    endpoint = maxlen
                if endpoint >= len(corpus):
                    startpoint = -maxlen
                    endpoint = None
                new_corpus=corpus[startpoint:endpoint]
                hash_value=hash("".join(new_corpus))
                if hash_value not in hashset:
                    cut_num+=1
                    corpus=new_corpus
                    hashset.add(hash_value)
                else:
                    del_num+=1
                    continue
            new_slicefile_corpus.append(corpus)
            new_slicefile_labels.append(label)
            new_slicefile_focus.append(focus)
            new_slicefile_func.append(func)
            new_slicefile_filenames.append(filename)
            # pbar.update(1)
        # with open(file_path,'wb')as f:
        #     pickle.dump([new_slicefile_corpus,new_slicefile_labels,new_slicefile_focus,new_slicefile_func,new_slicefile_filenames],f)
    # print("not change num:",not_change_num)
    # print("delete num:",del_num)
    # print("cut num:",cut_num)
    return not_change_num,del_num,cut_num


if __name__ == '__main__':
    os.chdir("/home/sysevr/Implementation/data_preprocess")
    corpus_path = "../corpus"
    params=[]
    for cve in os.listdir(corpus_path):
        for filename in os.listdir(os.path.join(corpus_path,cve)):
            params.append(os.path.join(corpus_path,cve,filename))
    pbar=tqdm(total=len(params))
    with Pool(32)as p:
        rets=p.imap_unordered(deal_single_file,params)
        for ret in rets:
            if ret is not None:
                pbar.set_description("not change num:%d,delete num:%d,cut num:%d"%ret)
            pbar.update(1)