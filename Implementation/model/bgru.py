## -*- coding: utf-8 -*-
'''
This python file is used to train four class focus data in blstm model

'''

from keras.preprocessing import sequence
# from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adadelta
from keras.models import Sequential, load_model
from keras.layers.core import Masking, Dense, Dropout, Activation
from keras.layers.recurrent import LSTM,GRU
from preprocess_dl_Input_version5 import *
from keras.layers.wrappers import Bidirectional
from collections import Counter
import numpy as np
import pickle
import random
import time
import math
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

RANDOMSEED = 2018  # for reproducibility
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def build_model(maxlen, vector_dim, layers, dropout):
    print('Build model...')
    model = Sequential()
    
    model.add(Masking(mask_value=0.0, input_shape=(maxlen, vector_dim)))
    
    for i in range(1, layers):
        model.add(Bidirectional(GRU(units=256, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True)))
        model.add(Dropout(dropout))
        
    model.add(Bidirectional(GRU(units=256, activation='tanh', recurrent_activation='hard_sigmoid')))
    model.add(Dropout(dropout))
    
    model.add(Dense(1, activation='sigmoid'))
          
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    model.summary()
 
    return model
def get_data(dirs):
    dataset = []
    labels = []
    for filename in dirs:
        if(not filename.endswith(".pkl")):
            continue
        print(filename)
        f = open(filename,"rb")
        dataset_file,labels_file,funcs_file,filenames_file,testcases_file = pickle.load(f)
        f.close()
        dataset += dataset_file
        labels += labels_file
    print(len(dataset), len(labels))
    bin_labels = []
    for label in labels:
        bin_labels.append(multi_labels_to_two(label))
    labels = bin_labels
    return dataset,labels

def main(traindataSet_path, testdataSet_path, realtestpath, weightpath, resultpath, batch_size, maxlen, vector_dim, layers, dropout):
    print("Loading data...")
    
    model = build_model(maxlen, vector_dim, layers, dropout)
    
    #model.load_weights(weightpath)  #load weights of trained model
    
    print("Train...")

    dirs=os.listdir(traindataSet_path)
    train_dir=[os.path.join(traindataSet_path,file) for file in dirs[:-1]]
    valid_dir=[os.path.join(traindataSet_path,dirs[-1]) ]
    train_dataset,train_labels = get_data(train_dir)
    valid_dataset,valid_labels = get_data(valid_dir)


     
    np.random.seed(RANDOMSEED)
    np.random.shuffle(train_dataset)
    np.random.seed(RANDOMSEED)
    np.random.shuffle(train_labels)
    
    # train_generator = generator_of_data(dataset, labels, batch_size, maxlen, vector_dim)
    train_generator = tf.data.Dataset.from_generator(
    generator_of_data,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, maxlen, vector_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.int32)
    ),
    args=(train_dataset, train_labels, batch_size, maxlen, vector_dim)
)
    valid_generator = tf.data.Dataset.from_generator(
    generator_of_data,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, maxlen, vector_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.int32)
    ),
    args=(valid_dataset,valid_labels, batch_size, maxlen, vector_dim)
)
    all_train_samples = len(train_dataset)
    steps_epoch = int(all_train_samples / batch_size)
    print("start")
    t1 = time.time()
    # model.fit_generator(train_generator, steps_per_epoch=steps_epoch, epochs=10)
    # model.fit(train_generator, steps_per_epoch=steps_epoch, epochs=10)
    # 定义一个 ModelCheckpoint 回调函数
    checkpoint = ModelCheckpoint(filepath=weightpath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)

    # 使用回调函数训练模型
    model.fit(train_generator, steps_per_epoch=steps_epoch, epochs=10, validation_data=valid_generator, callbacks=[checkpoint])

    t2 = time.time()
    train_time = t2 - t1

    # model.save_weights(weightpath)
    
    #model.load_weights(weightpath)
    print("Test1...")
    dataset = []
    labels = []
    testcases = []
    filenames = []
    funcs = []
    for filename in os.listdir(testdataSet_path):
        if(filename.endswith(".pkl") is False):
           continue
        print(filename)
        f = open(os.path.join(testdataSet_path, filename),"rb")
        datasetfile,labelsfile,funcsfiles,filenamesfile,testcasesfile = pickle.load(f)
        f.close()
        dataset += datasetfile
        labels += labelsfile
        testcases += testcasesfile
        funcs += funcsfiles
        filenames += filenamesfile
    print(len(dataset), len(labels), len(testcases))

    bin_labels = []
    for label in labels:
        bin_labels.append(multi_labels_to_two(label))
    labels = bin_labels

    batch_size = 1
    test_generator = generator_of_data(dataset, labels, batch_size, maxlen, vector_dim)
    all_test_samples = len(dataset)
    steps_epoch = int(math.ceil(all_test_samples / batch_size))

    t1 = time.time()
    result = model.evaluate_generator(test_generator, steps=steps_epoch)
    t2 = time.time()
    test_time = t2 - t1
    score, TP, FP, FN, precision, recall, f_score= result[0]
    f = open("TP_index_blstm.pkl",'wb')
    pickle.dump(result[1], f)
    f.close()
    os.makedirs("./result_analyze/BGRU", exist_ok=True)
    f_TP = open("./result_analyze/BGRU/TP_filenames.txt","ab+")
    for i in range(len(result[1])):
        TP_index = result[1][i]
        f_TP.write(str(filenames[TP_index])+'\n')
		
    f_FP = open("./result_analyze/BGRU/FP_filenames.txt","ab+")
    for j in range(len(result[2])):
        FP_index = result[2][j]
        f_FP.write(str(filenames[FP_index])+'\n')

    f_FN = open("./result_analyze/BGRU/FN_filenames.txt","a+")
    for k in range(len(result[3])):
        FN_index = result[3][k]
        f_FN.write(str(filenames[FN_index])+'\n')

    TN = all_test_samples - TP - FP - FN
    fwrite = open(resultpath, 'a')
    fwrite.write('cdg_ddg: ' + ' ' + str(all_test_samples) + '\n')
    fwrite.write("TP:" + str(TP) + ' FP:' + str(FP) + ' FN:' + str(FN) + ' TN:' + str(TN) +'\n')
    FPR = float(FP) / (FP + TN)
    fwrite.write('FPR: ' + str(FPR) + '\n')
    FNR = float(FN) / (TP + FN)
    fwrite.write('FNR: ' + str(FNR) + '\n')
    Accuracy = float(TP + TN) / (all_test_samples)
    fwrite.write('Accuracy: ' + str(Accuracy) + '\n')
    precision = float(TP) / (TP + FP)
    fwrite.write('precision: ' + str(precision) + '\n')
    recall = float(TP) / (TP + FN)
    fwrite.write('recall: ' + str(recall) + '\n')
    f_score = (2 * precision * recall) / (precision + recall)
    fwrite.write('fbeta_score: ' + str(f_score) + '\n')
    fwrite.write('train_time:' + str(train_time) +'  ' + 'test_time:' + str(test_time) + '\n')
    fwrite.write('--------------------\n')
    fwrite.close()

    dict_testcase2func = {}
    for i in testcases:
        if not i in dict_testcase2func:
            dict_testcase2func[i] = {}
    TP_indexs = result[1]
    for i in TP_indexs:
        if funcs[i] == []:
            continue
        for func in funcs[i]:
            if func in dict_testcase2func[testcases[i]].keys():
                dict_testcase2func[testcases[i]][func].append("TP")
            else:
                dict_testcase2func[testcases[i]][func] = ["TP"]
    FP_indexs = result[1]
    for i in FP_indexs:
        if funcs[i] == []:
            continue
        for func in funcs[i]:
            if func in dict_testcase2func[testcases[i]].keys():
                dict_testcase2func[testcases[i]][func].append("FP")
            else:
                dict_testcase2func[testcases[i]][func] = ["FP"]
    f = open(resultpath+"_dict_testcase2func.pkl",'wb')
    pickle.dump(dict_testcase2func, f)
    f.close()

def testrealdata(realtestpath, weightpath, batch_size, maxlen, vector_dim, layers, dropout):
    model = build_model(maxlen, vector_dim, layers, dropout)
    model.load_weights(weightpath)
    
    print("Loading data...")
    for filename in os.listdir(realtestpath):
        print(filename)
        f = open(realtestpath+filename, "rb")
        realdata = pickle.load(f,encoding="latin1")
        f.close()
    
        labels = model.predict(x = realdata[0],batch_size = 1)
        for i in range(len(labels)):
            if labels[i][0] >= 0.5:
                print(realdata[1][i])


if __name__ == "__main__":
    batchSize = 16
    vectorDim = 30
    maxLen = 100
    layers = 2
    dropout = 0.2
    traindataSetPath = "../data/dl_input_shuffle/train/"
    testdataSetPath = "../data/dl_input_shuffle/test/"
    realtestdataSetPath = "data/"
    weightPath = './model/BRGU'
    resultPath = "./result/BGRU/BGRU"
    os.makedirs("./model",exist_ok=True)
    os.makedirs("./result",exist_ok=True)
    main(traindataSetPath, testdataSetPath, realtestdataSetPath, weightPath, resultPath, batchSize, maxLen, vectorDim, layers, dropout)
    #testrealdata(realtestdataSetPath, weightPath, batchSize, maxLen, vectorDim, layers, dropout)