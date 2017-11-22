# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:48:18 2017

@author: tam
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.externals import joblib
import re

def text_to_wordlist(text):
    txt = re.sub("[^a-zA-Z]"," ", text)
    words = txt.lower().split()
    return(words)


def getCleanReviews(texts):
    clean_texts = []
    for text in texts:
        clean_texts.append( text_to_wordlist(text))
    return clean_texts

    
def load_w2v_model(model_file='../models/glove.6B.50d.txt'):
    with open(model_file) as lines:
        w2v = {line.split()[0]: np.array(list(map(float, line.split()[1:])))
           for line in lines}
    return w2v
    
def get_mean_vectors(words, word2vec, dim):
    nwords = 0.
    featureVec = np.zeros((dim,),dtype="float32")
        
    for word in words:
        if word in word2vec:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, word2vec[word])
    
    featureVec = np.divide(featureVec,nwords)
    return featureVec        

def get_sum_vectors(words, word2vec, dim):
    nwords = 0.
    featureVec = np.zeros((dim,),dtype="float32")
        
    for word in words:
        if word in word2vec:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, word2vec[word])
    
    return featureVec

def get_concat_vectors(words, word2vec, dim, max_words=60):
    nwords = 0
    featureVec = np.zeros((dim*max_words,),dtype="float32")
    for word in words:
        nwords = nwords + 1.
        if word in word2vec:
            featureVec[int((nwords-1)*dim): int(nwords*dim)] = word2vec[word]
        if nwords >= max_words:
            break
    return featureVec

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, embed_size=50):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())
        self.embed_size = embed_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = np.zeros((len(X), self.embed_size))
        for i, text in enumerate(X):
            words = text_to_wordlist(text)
            res[i] = get_mean_vectors(words, self.word2vec, self.embed_size)
        return res
    
class ConcatEmbeddingVectorizer(object):
    def __init__(self, word2vec, embed_size=50):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())
        self.embed_size = embed_size
        self.max_words = 70

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = np.zeros((len(X), self.embed_size*self.max_words))
        for i, text in enumerate(X):
            words = text_to_wordlist(text)
            res[i] = get_concat_vectors(words, self.word2vec, self.embed_size, self.max_words)
        return res
    
            
class SumEmbeddingVectorizer(object):
    def __init__(self, word2vec, embed_size=50):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())
        self.embed_size = embed_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = np.zeros((len(X), self.embed_size))
        for i, text in enumerate(X):
            words = text_to_wordlist(text)
            res[i] = get_sum_vectors(words, self.word2vec, self.embed_size)
        return res
 
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

def extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.50d.txt'):
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    embed_size = int(pre_trained_model.split('.')[-2].replace('d',''))
    print('loading model')
    w2v = load_w2v_model(pre_trained_model)
    vect = MeanEmbeddingVectorizer(w2v, embed_size)
    
    print('transforming')
    X_title_tr = vect.transform(df["title"].tolist())
    print('saving %s' % feat_train_file)
    joblib.dump(X_title_tr, feat_train_file)
    del X_title_tr
    
    X_title_val = vect.transform(df1["title"].tolist())
    print('saving %s' % feat_test_file)
    joblib.dump(X_title_val, feat_test_file)
    print(X_title_val.shape)


def extract_w2v_concat_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.50d.txt'):
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    embed_size = int(pre_trained_model.split('.')[-2].replace('d',''))
    print('loading model')
    w2v = load_w2v_model(pre_trained_model)
    vect = ConcatEmbeddingVectorizer(w2v, embed_size)
    
    print('transforming')
    X_title_tr = vect.transform(df["title"].tolist())
    print('saving %s' % feat_train_file)
    joblib.dump(X_title_tr, feat_train_file)
    del X_title_tr
    
    X_title_val = vect.transform(df1["title"].tolist())
    print('saving %s' % feat_test_file)
    joblib.dump(X_title_val, feat_test_file)
    print(X_title_val.shape)
    

def extract_w2v_tfidf_features(train_file, test_file, feat_train_file, feat_test_file, 
                                analyzer='char', ngram_range=(1, 1), lowercase=True):
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
   
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=lowercase)
    vect.fit(df["title"].tolist())
    
    X_title = vect.transform(df["title"])
    X_title_tr = X_title[:n_trains,:]
    X_title_val = X_title[n_trains:,:]
    
    print('saving %s' % feat_train_file)
    joblib.dump(X_title_tr, feat_train_file)
    
    print('saving %s' % feat_test_file)
    joblib.dump(X_title_val, feat_test_file)


if __name__ == "__main__":
    
    train_file = '../data/data_train.csv'
    test_file = '../data/data_valid.csv'
    
    feat_train_file = '../features/title.w2v_mean.trn'
    feat_test_file = '../features/title.w2v_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file)
    
    feat_train_file = '../features/title.glove.6B.50d_mean.trn'
    feat_test_file = '../features/title.glove.6B.50d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.50d.txt')
    
            
    feat_train_file = '../features/title.glove.6B.100d_mean.trn'
    feat_test_file = '../features/title.glove.6B.100d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.100d.txt')
    
    feat_train_file = '../features/title.glove.6B.200d_mean.trn'
    feat_test_file = '../features/title.glove.6B.200d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.200d.txt')
    
    feat_train_file = '../features/title.glove.6B.300d_mean.trn'
    feat_test_file = '../features/title.glove.6B.300d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.300d.txt')
    
    
    feat_train_file = '../features/title.glove.twitter.27B.25d_mean.trn'
    feat_test_file = '../features/title.glove.twitter.27B.25d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.twitter.27B.25d.txt')
    
    feat_train_file = '../features/title.glove.twitter.27B.50d_mean.trn'
    feat_test_file = '../features/title.glove.twitter.27B.50d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.twitter.27B.50d.txt')
    
    feat_train_file = '../features/title.glove.twitter.27B.100d_mean.trn'
    feat_test_file = '../features/title.glove.twitter.27B.100d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.twitter.27B.100d.txt')
    
    feat_train_file = '../features/title.glove.twitter.27B.200d_mean.trn'
    feat_test_file = '../features/title.glove.twitter.27B.200d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.twitter.27B.200d.txt')
    
    feat_train_file = '../features/title.glove.42B.300d_mean.trn'
    feat_test_file = '../features/title.glove.42B.300d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.42B.300d.txt')
   
    feat_train_file = '../features/title.glove.480B.300d_mean.trn'
    feat_test_file = '../features/title.glove.480B.300d_mean.tst'
    extract_w2v_mean_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.480B.300d.txt')
    
    
    train_file = '../data/data_train.csv'
    test_file = '../data/data_valid.csv'
    
    feat_train_file = '../features/title.glove.6B.50d_concat.trn'
    feat_test_file = '../features/title.glove.6B.50d_concat.tst'
    extract_w2v_concat_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.50d.txt')
            
    feat_train_file = '../features/title.glove.6B.100d_concat.trn'
    feat_test_file = '../features/title.glove.6B.100d_concat.tst'
    extract_w2v_concat_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.100d.txt')
    
    feat_train_file = '../features/title.glove.6B.200d_concat.trn'
    feat_test_file = '../features/title.glove.6B.200d_concat.tst'
    extract_w2v_concat_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.200d.txt')
    
    feat_train_file = '../features/title.glove.6B.300d_concat.trn'
    feat_test_file = '../features/title.glove.6B.300d_concat.tst'
    extract_w2v_concat_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.6B.300d.txt')
    
    
    feat_train_file = '../features/title.glove.twitter.27B.25d_concat.trn'
    feat_test_file = '../features/title.glove.twitter.27B.25d_concat.tst'
    extract_w2v_concat_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.twitter.27B.25d.txt')
    
    feat_train_file = '../features/title.glove.twitter.27B.50d_concat.trn'
    feat_test_file = '../features/title.glove.twitter.27B.50d_concat.tst'
    extract_w2v_concat_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.twitter.27B.50d.txt')
    
    feat_train_file = '../features/title.glove.twitter.27B.100d_concat.trn'
    feat_test_file = '../features/title.glove.twitter.27B.100d_concat.tst'
    extract_w2v_concat_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.twitter.27B.100d.txt')
    
    feat_train_file = '../features/title.glove.twitter.27B.200d_concat.trn'
    feat_test_file = '../features/title.glove.twitter.27B.200d_concat.tst'
    extract_w2v_concat_features(train_file, test_file, feat_train_file, feat_test_file, pre_trained_model='../models/glove.twitter.27B.200d.txt')
   
    
    