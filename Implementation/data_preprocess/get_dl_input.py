## coding: utf-8
'''
This python file is used to split database into 80% train set and 20% test set, tranfer the original code into vector, creating input file of deap learning model.
'''

from gensim.models.word2vec import Word2Vec
import pickle
import os
import numpy as np
import random
import gc
import shutil
from tqdm import tqdm
from multiprocessing import Pool
import joblib
'''
generate_corpus function
-----------------------------
This function is used to create input of deep learning model

# Arguments
    w2vModelPath: the path saves word2vec model
    samples: the list of sample
    
'''
def generate_corpus(model, samples): 
    
    # print("begin generate input...")
    dl_corpus = [[model.wv[word] for word in sample] for sample in samples]
    # print("generate input success...")

    return dl_corpus

'''
get_dldata function
-----------------------------
This function is used to create input of deep learning model

# Arguments
    filepath: the path saves data
    dlTrainCorpusPath: the path saves train dataset
    dlTestCorpusPath: the path saves test dataset
    split: split ratio
    seed: random number seed 
    
'''
def read_data(file):
    try:
        with open(file, 'rb') as f:
            data=pickle.load(f)
    except:
        return None
    ids=[]
    if len(data)<5:
        print(file)
        return None
    else:
        id_length = len(data[1])
        for j in range(id_length):
            ids.append(file.split('/')[-2])
        data.append(ids)
        return  data
def get_dldata_new(filepath, dlTrainCorpusPath, dlTestCorpusPath, split=0.8, seed=113):
    folders = os.listdir(filepath)
    np.random.seed(seed)
    np.random.shuffle(folders)

    folders_train = folders[:int(len(folders)*split)]
    folders_test = folders[int(len(folders)*split):]
    for dirs,split_num,save_path in [(folders_train,8,dlTrainCorpusPath), (folders_test,2,dlTestCorpusPath)]:
        params=[]
        for cve in dirs:
            for filename in os.listdir(os.path.join(filepath,cve)):
                params.append(os.path.join(filepath,cve,filename))
        # file_size=[]
        # for file in params:
        #     file_size.append((file,os.path.getsize(file)))
        # file_size.sort(key=lambda x:x[1])
        # params=[x[0] for x in file_size]
        with Pool(30)as pool:
            for i in range(split_num):
                train_set= [[], [], [], [], [], []]
                pbar=tqdm(total=int(len(params)/split_num))
                ret=pool.imap_unordered(read_data,params[i*int(len(params)/split_num):(i+1)*int(len(params)/split_num)])
                for r in ret :
                    if r is not None:
                        train_set=[train_set[i]+r[i] for i in range(6)]
                    pbar.update(1)
                f_train = open(save_path + str(i)+ "_.pkl", 'wb')
                pickle.dump(train_set, f_train, protocol=pickle.HIGHEST_PROTOCOL)
                f_train.close()
                # f_train = open(save_path + str(i)+ "_.bin", 'wb')
                # joblib.dump(train_set, f_train)
                # f_train.close()
        

def get_dldata(filepath, dlTrainCorpusPath, dlTestCorpusPath, split=0.8, seed=113):
    folders = os.listdir(filepath)
    np.random.seed(seed)
    np.random.shuffle(folders)

    folders_train = folders[:int(len(folders)*split)]
    folders_test = folders[int(len(folders)*split):]
      
    for mode in ["api", "arraysuse", "pointersuse", "integeroverflow"]:
        if mode == "api":
            N = 4
            num = [0,1,2,3]
        if mode == "arraysuse":
            N = 2
            num = [0,1]
        if mode == "integeroverflow":
            N = 2
            num = [0,1]
        if mode == "pointersuse":
            N =6
            num = [0,1,2,3,4,5]
        for i in num:
            train_set = [[], [], [], [], [], []]
            ids = []
            for folder_train in tqdm(folders_train[int(i*len(folders_train)/N) : int((i+1)*len(folders_train)/N)]):
                for filename in os.listdir(filepath + folder_train + '/'):
                    if mode in filename:
                        if folder_train not in os.listdir(dlTrainCorpusPath):   
                            folder_path = os.path.join(dlTrainCorpusPath, folder_train)
                            os.mkdir(folder_path)
                        shutil.copyfile(filepath + folder_train + '/'+filename , dlTrainCorpusPath + folder_train + '/'+filename)
                        f = open(filepath + folder_train + '/' + filename, 'rb')
                        data = pickle.load(f)
                        id_length = len(data[1])
                        for j in range(id_length):
                            ids.append(folder_train)
                        for n in range(5):
                            train_set[n] = train_set[n] + data[n]
                        train_set[-1] = ids
            if train_set[0] == []:
                continue
            f_train = open(dlTrainCorpusPath + mode + "_" + str(i)+ "_.pkl", 'wb')
            pickle.dump(train_set, f_train, protocol=pickle.HIGHEST_PROTOCOL)
            f_train.close()
            del train_set
            gc.collect()     
                    
    for mode in ["api", "arraysuse", "pointersuse", "integeroverflow"]:
        N = 4
        num = [0,1,2,3]
        if mode == "pointersuse":
            N = 8
            num = [4,5]
        for i in num:
            test_set = [[], [], [], [], [], []]
            ids = []
            for folder_test in folders_test[int(i*len(folders_test)/N) : int((i+1)*len(folders_test)/N)]:
                for filename in os.listdir(filepath + folder_test + '/'):
                    if mode in filename:
                        if folder_test not in os.listdir(dlTestCorpusPath):
                            folder_path = os.path.join(dlTestCorpusPath, folder_test)
                            os.mkdir(folder_path)
                        shutil.copyfile(filepath + folder_test + '/'+filename , dlTestCorpusPath + folder_test + '/'+filename) 
                        f = open(filepath + folder_test + '/' + filename, 'rb')
                        data = pickle.load(f)
                        id_length = len(data[1])
                        for j in range(id_length):
                            ids.append(folder_test)
                        for n in range(5):
                            test_set[n] = test_set[n] + data[n]
                        test_set[-1] = ids
            if test_set[0] == []:
                continue
            f_test = open(dlTestCorpusPath + mode + "_" + str(i)+ ".pkl", 'wb')
            pickle.dump(test_set, f_test, protocol=pickle.HIGHEST_PROTOCOL)
            f_test.close()
            del test_set
            gc.collect()

def generate_vectors(params):
    corpusfiles,CORPUSPATH,VECTORPATH,w2v_model=params
    for corpusfile in os.listdir(CORPUSPATH + corpusfiles):
        corpus_path = os.path.join(CORPUSPATH, corpusfiles, corpusfile)
        f_corpus = open(corpus_path, 'rb')
        data = pickle.load(f_corpus)
        f_corpus.close()
        data[0] = generate_corpus(w2v_model, data[0])
        vector_path = os.path.join(VECTORPATH, corpusfiles, corpusfile)
        f_vector = open(vector_path, 'wb')
        pickle.dump(data, f_vector, protocol=pickle.HIGHEST_PROTOCOL)
        f_vector.close()
    # print("w2v over...")
if __name__ == "__main__":
    # os.chdir('/home/sysevr/Implementation/data_preprocess')
    
    CORPUSPATH = "../corpus/"
    VECTORPATH = "../vector/"
    W2VPATH = "../w2v_model/wordmodel-2.6.0"
    os.makedirs(VECTORPATH, exist_ok=True)
    # print("turn the corpus into vectors...")
    # model = Word2Vec.load(W2VPATH)
    # params=[]
    # for corpusfiles in tqdm(os.listdir(CORPUSPATH)):
    #     if corpusfiles not in os.listdir(VECTORPATH): 
    #         folder_path = os.path.join(VECTORPATH, corpusfiles)
    #         os.mkdir(folder_path)
    #     params.append((corpusfiles,CORPUSPATH,VECTORPATH,model))
    # pbar=tqdm(total=len(params))
    # with Pool(32)as p:
    #     rets=p.imap_unordered(generate_vectors,params)
    #     for ret in rets:
    #         pbar.update(1)
    # pbar.close()
    # print("w2v over...")

    print("spliting the train set and test set...")
    dlTrainCorpusPath = "../data/train/"
    dlTestCorpusPath = "../data/test/"
    os.makedirs(dlTrainCorpusPath,exist_ok=True)
    os.makedirs(dlTestCorpusPath,exist_ok=True)
    get_dldata_new(VECTORPATH, dlTrainCorpusPath, dlTestCorpusPath)
    
    print("success!")
