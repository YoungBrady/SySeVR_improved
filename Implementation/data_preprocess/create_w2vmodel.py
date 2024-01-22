## coding: utf-8
'''
This python file is used to tranfer the words in corpus to vector, and save the word2vec model under the path 'w2v_model'.
'''

from gensim.models.word2vec import Word2Vec
import pickle
import os
import gc
from tqdm import tqdm

'''
DirofCorpus class
-----------------------------
This class is used to make a generator to produce sentence for word2vec training

# Arguments
    dirname: The src of corpus files 
    
'''

class DirofCorpus(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.len=0
        for d in self.dirname:
            for fn in os.listdir(d):
                self.len+=1
        self.pabr=tqdm(total=self.len)
        
    
    def __iter__(self):
        for d in self.dirname:
            for fn in os.listdir(d):
                # print(fn)
                self.pabr.set_description(fn)
                self.pabr.update(1)
                for filename in os.listdir(os.path.join(d, fn)):
                    with open(os.path.join(d, fn, filename), 'rb')as f:
                        samples = pickle.load(f)[0]
                    for sample in samples:
                        yield sample
                    del samples
                    gc.collect()

'''
generate_w2vmodel function
-----------------------------
This function is used to learning vectors from corpus, and save the model

# Arguments
    decTokenFlawPath: String type, the src of corpus file 
    w2vModelPath: String type, the src of model file 
    
'''

def generate_w2vModel(decTokenFlawPath, w2vModelPath):
    print("training...")
    model = Word2Vec(size=30, alpha=0.01, window=5, min_count=0, max_vocab_size=None, sample=0.001, seed=1, workers=50, min_alpha=0.0001, sg=1, hs=0, negative=10, iter=5)
    init=True
    for d in decTokenFlawPath:
        for fn in tqdm(os.listdir(d)):
            sentences_corpus=[]
            for filename in os.listdir(os.path.join(d, fn)):
                with open(os.path.join(d, fn, filename), 'rb') as f:
                    samples = pickle.load(f)[0]
                    sentences_corpus += samples
            if init:
                model.build_vocab(sentences_corpus)
                init=False
            else:
                model.build_vocab(sentences_corpus, update=True)
            model.train(sentences_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(w2vModelPath)

def evaluate_w2vModel(w2vModelPath):
    print("\nevaluating...")
    model = Word2Vec.load(w2vModelPath)
    for sign in ['(', '+', '-', '*', 'main']:
        print(sign, ":")
        print(model.wv.most_similar_cosmul(positive=[sign], topn=10))
    
def main():
    dec_tokenFlaw_path = ['../corpus/']
    w2v_model_path = "../w2v_model/wordmodel-2.6.0" 
    os.makedirs("../w2v_model",exist_ok=True)
    generate_w2vModel(dec_tokenFlaw_path, w2v_model_path)
    evaluate_w2vModel(w2v_model_path)
    print("success!")
 
if __name__ == "__main__":
    main()


