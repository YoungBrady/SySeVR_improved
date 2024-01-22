## coding: utf-8
'''
This python file is used to precess the vulnerability slices, including read the pkl file and split codes into corpus.
Run main function and you can get a corpus pkl file which map the same name slice file.
'''

import os
import pickle
from mapping import *
import json
from tqdm import tqdm
from multiprocessing import Pool,Process
'''
get_sentences function
-----------------------------
This function is used to split the slice file and split codes into words

# Arguments
    _path: String type, the src of slice files
    labelpath: String type, the src of label files
    deletepath: delete list, delete repeat slices
    corpuspath: String type, the src to save corpus
    maptype: bool type, choose do map or not

# Return
    [slices[], labels[], focus[]]
'''
def get_sentences(_path,labelpath,corpuspath,maptype=True):
    FLAGMODE = False
    if "SARD" in _path:
        FLAGMODE = True
    
    for filename in os.listdir(_path):
        if(filename.endswith(".txt") is False):
            continue
        print(filename)

        filepath = os.path.join(_path, filename)
        f1 = open(filepath, 'r')
        slicelists = f1.read().split("------------------------------")
        f1.close()

        if slicelists[0] == '':
            del slicelists[0]
        if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
            del slicelists[-1]

        filepath = os.path.join(labelpath, "labels.json")
        with open(filepath, 'r') as f1:
            
            labellists = json.load(f1)

	
	
        lastprogram_id = 0
        program_id = 0
        index = -1
        slicefile_corpus = []
        slicefile_labels = []
        slicefile_focus = []
        slicefile_filenames = []
        slicefile_func = []
        focuspointer = None 
        for slicelist in tqdm(slicelists):
            slice_corpus = []
            focus_index = 0
            flag_focus = 0 

            index = index + 1

            sentences = slicelist.split('\n')

            if sentences[0] == '\r' or sentences[0] == '':
                del sentences[0]
            if sentences == []:
                continue
            if sentences[-1] == '':
                del sentences[-1]
            if sentences[-1] == '\r':
                del sentences[-1]
            focuspointer = sentences[0].split(" ")[-2:]
            sliceid = index

            file_name = sentences[0]
 
            if FLAGMODE:    
                program_id = sentences[0].split(" ")[1].split("/")[-4] + sentences[0].split(" ")[1].split("/")[-3] + sentences[0].split(" ")[1].split("/")[-2]
            else:
                program_id = sentences[0].split(" ")[1].split("/")[-1]
            if lastprogram_id == 0:
                lastprogram_id = program_id

            if not(lastprogram_id == program_id):
                folder_path = os.path.join(corpuspath, str(lastprogram_id))
                savefilename = folder_path + '/' + filename + '.pkl'
                if lastprogram_id not in os.listdir(corpuspath):    
                    os.mkdir(folder_path)
                if savefilename not in os.listdir(folder_path):    
                    f1 = open(savefilename, 'wb')               
                    pickle.dump([slicefile_corpus,slicefile_labels,slicefile_focus,slicefile_func,slicefile_filenames], f1)
                else:
                    f1 = open(savefilename, 'rb')        
                    data = cPickle.load(f1)
                    f1.close()
                    f1 = open(savefilename, 'wb')              
                    pickle.dump([slicefile_corpus+data[0],slicefile_labels+data[1],slicefile_focus+data[2],slicefile_func+data[3],slicefile_filenames+data[4]], f1)
                f1.close()
                slicefile_corpus = []
                slicefile_focus = []
                slicefile_labels = []
                slicefile_filenames = []
                slicefile_func = []
                lastprogram_id = program_id
            sentences = sentences[1:]
            for sentence in sentences:
                if sentence.split(" ")[-1] == focuspointer[1] and flag_focus == 0:
                    flag_focus = 1  
                sentence = ' '.join(sentence.split(" ")[:-1])
                start = str.find(sentence,r'printf("')
                if start != -1:
                    start = str.find(sentence,r'");')
                    sentence = sentence[:start+2]
                
                fm = str.find(sentence,'/*')
                if fm != -1:
                    sentence = sentence[:fm]
                else:
                    fm = str.find(sentence,'//')
                    if fm != -1:
                        sentence = sentence[:fm]
                    
                sentence = sentence.strip()
                list_tokens = create_tokens(sentence)

                if flag_focus == 1:#当前行是切片出发行
                    if "expr" in filename:
                        focus_index = focus_index + int(len(list_tokens)/2)
                        flag_focus = 2  
                        slicefile_focus.append(focus_index)
                    else:               
                        if focuspointer[0] in list_tokens:
                            focus_index = focus_index + list_tokens.index(focuspointer[0])
                            flag_focus = 2  
                            slicefile_focus.append(focus_index)
                        else:  
                            if '*' in focuspointer[0]:
                                if focuspointer[0] in list_tokens:
                                    focus_index = focus_index + list_tokens.index(focuspointer[0].replace('*',''))
                                    flag_focus = 2
                                    slicefile_focus.append(focus_index)
                                else:
                                    flag_focus = 0
                            else:
                                flag_focus = 0
                if flag_focus == 0:
                    focus_index = focus_index + len(list_tokens)
      
                if maptype:
                    slice_corpus.append(list_tokens)
                else:
                    slice_corpus = slice_corpus + list_tokens

            if flag_focus == 0:
                continue
            slicefile_labels.append(labellists[file_name])
            slicefile_filenames.append(file_name)

            if maptype:
                slice_corpus, slice_func = mapping(slice_corpus)
                slice_func = list(set(slice_func))
                if slice_func == []:
                    slice_func = ['main']
                sample_corpus = []
                for sentence in slice_corpus:
                    list_tokens = create_tokens(sentence)
                    sample_corpus = sample_corpus + list_tokens
                slicefile_corpus.append(sample_corpus)
                slicefile_func.append(slice_func)
            else:
                slicefile_corpus.append(slice_corpus)

        folder_path = os.path.join(corpuspath, str(lastprogram_id))
        savefilename = folder_path + '/' + filename + '.pkl'
        if lastprogram_id not in os.listdir(corpuspath):   
            os.mkdir(folder_path)
        if savefilename not in os.listdir(folder_path):    
            f1 = open(savefilename, 'wb')                 
            pickle.dump([slicefile_corpus,slicefile_labels,slicefile_focus,slicefile_func,slicefile_filenames], f1)
        else:
            f1 = open(savefilename, 'rb')              
            data = pickle.load(f1)
            f1.close()
            f1 = open(savefilename, 'wb')                 
            pickle.dump([slicefile_corpus+data[0],slicefile_labels+data[1],slicefile_focus+data[2],slicefile_func+data[3],slicefile_filenames+data[4]], f1)
        f1.close()
def get_sentence_corpus(params):
    slice,labels_dict=params
    maxlen=500
    sentences=slice.split("\n")
    if sentences[0] == '\r' or sentences[0] == '':
        del sentences[0]
    if sentences == []:
        return None
    if sentences[-1] == '':
        del sentences[-1]
    if sentences[-1] == '\r':
        del sentences[-1]
    slice_filename=sentences[0]
    focuspointer = sentences[0].split(" ")[-2:]
    if slice_filename not in labels_dict:
        return None
    label=labels_dict[slice_filename]
    flag_focus = 0
    sentences=sentences[1:]
    slice_corpus=[]
    focus_index=0
    corpus_len=0
    for sentence in sentences:
        if sentence.split(" ")[-1] == focuspointer[1] and flag_focus == 0:
            flag_focus = 1  
        sentence = ' '.join(sentence.split(" ")[:-1])
        sentence = sentence.strip()
        list_tokens = create_tokens(sentence)
        slice_corpus.append(list_tokens)
        if flag_focus == 1:#当前行是切片出发行
            focus_index = corpus_len
            flag_focus = 2
        corpus_len+=len(list_tokens)

    slice_corpus, slice_func = mapping(slice_corpus)
    slice_func = list(set(slice_func))
    if slice_func == []:
        slice_func = ['main']
    sample_corpus = []
    for sentence in slice_corpus:
        list_tokens = create_tokens(sentence)
        sample_corpus = sample_corpus + list_tokens
    return [sample_corpus,label,focus_index, slice_func,slice_filename]
    # return focus_index
def get_slice_corpus(params):
    _path,labels_dict,CORPUSPATH=params
    

    # with Pool(30)as pool:
        
    for filename in os.listdir(_path):
        params = []
        slicefile_labels = []
        slicefile_filenames = []
        slicefile_func = []
        slicefile_corpus = []
        slicefile_focus=[]
        filepath = os.path.join(_path, filename)
        f1 = open(filepath, 'r')
        slicelists = f1.read().split("------------------------------\n")
        f1.close()
        if slicelists[0] == '':
            del slicelists[0]
        if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
            del slicelists[-1]
        for slice in slicelists:
            params.append([slice,labels_dict])
        # results = pool.imap_unordered(get_sentence_corpus, params)
        # pbar=tqdm(total=len(params))
        # for ret in results:
        #     pbar.update(1)
        for param  in params:
            ret=get_sentence_corpus(param)
            if ret == None:
                continue
            # focus_index=ret
            sample_corpus,label,focus_index, slice_func,slice_filename=ret
            slicefile_corpus.append(sample_corpus)
            slicefile_func.append(slice_func)
            slicefile_labels.append(label)
            slicefile_filenames.append(slice_filename)
            slicefile_focus.append(focus_index)
        save_dir=os.path.join(CORPUSPATH,_path.split('/')[-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename='.'.join(filename.split('.')[:-1])+'.pkl'
        save_path=os.path.join(save_dir, save_filename)
        if not os.path.exists(save_path):    
            with open(save_path,'wb')as f:
                pickle.dump([slicefile_corpus,slicefile_labels,slicefile_focus,slicefile_func,slicefile_filenames], f)
            # continue
        # else:
        #     with open(save_path,'rb')as f:
        #         data=pickle.load(f)
        #         slicefile_corpus,slicefile_labels,slicefile_func,slicefile_filenames=data
        #     with open(save_path,'wb')as f:
        #         pickle.dump([slicefile_corpus,slicefile_labels,slicefile_focus,slicefile_func,slicefile_filenames], f)


def multiprocessing_slice_corpus(SLICEPATH, LABELPATH, CORPUSPATH, MAPTYPE):
    params = []
    with open(os.path.join(LABELPATH, 'labels.json'), 'r') as f:
        labels_dict = json.load(f)
    cve2labels = {}
    for key in labels_dict:
        _list=key.split('/')
        if len(_list)<=8:
            continue
        else:
            cve=_list[8]
        if cve not in cve2labels:
            cve2labels[cve]={}
        
        cve2labels[cve][key]=labels_dict[key]
    for cve in os.listdir(SLICEPATH):
        # if cve !="CVE-2017-6886":
        #     continue
        if cve not in cve2labels:
            continue
        # if os.path.exists(os.path.join(CORPUSPATH, cve)):
        #     continue
        if os.listdir(os.path.join(SLICEPATH, cve))==[]:
            continue
        params.append((os.path.join(SLICEPATH, cve), cve2labels[cve],CORPUSPATH))
    pbar=tqdm(total=len(params))
    with Pool(30) as p:
        rets=p.imap_unordered(get_slice_corpus, params)
        for ret in rets:
            pbar.update(1)
    # for param in tqdm(params):
    #     get_slice_corpus(param)
        
                
    
    
if __name__ == '__main__':
    os.chdir('/home/sysevr/Implementation/data_preprocess')
    SLICEPATH = '../label_source/'
    LABELPATH = '../labels/'
    CORPUSPATH = '../corpus/'
    os.makedirs(CORPUSPATH, exist_ok=True)
    MAPTYPE = True

    # sentenceDict = get_sentences(SLICEPATH, LABELPATH, CORPUSPATH, MAPTYPE)
    multiprocessing_slice_corpus(SLICEPATH, LABELPATH, CORPUSPATH, MAPTYPE)

    print('success!')
