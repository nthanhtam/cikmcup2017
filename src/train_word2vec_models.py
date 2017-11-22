#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 21:47:45 2017

@author: tam
"""

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import logging
from sklearn.feature_extraction.text import CountVectorizer
import re
import sys


def get_model(embedding_matrix, num_words, MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=100):
        embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    
        # train a 1D convnet with global maxpooling
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(2, activation='softmax')(x)
        
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      #optimizer='rmsprop',
                      optimizer='adam',
                      metrics=['acc','mse'])
        
        return model

    
def get_rare_tokens(corpus, min_f=10):
    cv = CountVectorizer(lowercase=True)
    corpus = [re.sub("[^a-zA-Z]", " ", text) for text in corpus]
    word_count = cv.fit_transform(corpus)
    rare_word_idx = np.where(word_count.sum(axis=0) < min_f)
    return set([w for w, i in cv.vocabulary_.items() if i in rare_word_idx[1]])

def text_to_wordlist(text, filter=set()):
    txt = re.sub("[^a-zA-Z]"," ", text)
    words = txt.lower().split()
    return list(set(words)-filter)
    
def train_and_predict(model_name='w2v', label_name='conciseness', pre_trained_model='../models/glove.6B.100d.txt', MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=30000, EMBEDDING_DIM=100, fold_set=0):
    
    logger = logging.getLogger('cikmcup2017')
    hdlr = logging.FileHandler('../log/%s.txt' % model_name)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    logger.info('starting')
    
    print('Indexing word vectors.')
    
    embeddings_index = {}
    f = open(pre_trained_model, encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    # second, prepare text samples and their labels
    print('Processing text dataset')
    
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df['conciseness'] = np.loadtxt("../data/%s_train.labels" % label_name, dtype=int)
    
    y_split = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    
    
    n_trains = df.shape[0]
    labels  = df["conciseness"].tolist()
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = pd.concat((df,df1))
    
    texts = df["title"].tolist()
    
    #texts = list(map(lambda x: text_to_wordlist(x, filter=get_rare_tokens(texts)), texts))

    print('Found %s texts.' % len(texts))
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    
    X = data[:n_trains]
    X_test = data[n_trains:]
    
    y = labels
    
    del data, labels
    
    print('Preparing embedding matrix.')
    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    n_folds = 10
    scores = []
    preds = np.zeros(X.shape[0])
    kF = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2017+fold_set*777)
    for k, (trnId, valId) in enumerate(kF.split(X, y_split)):
        X_train = X[trnId]
        y_train = y[trnId]
        X_val = X[valId]
        y_val = y[valId]
    
        # Training...
        model = get_model(embedding_matrix, num_words, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM)
        model.fit(X_train, y_train,
                  batch_size=128,
                  epochs=4,
                  validation_data=(X_val, y_val))
        
        pred = model.predict(X_val)[:,1]
        preds[valId] = pred
        score = mean_squared_error(pred, y_val[:,1])**0.5
        scores.append(score)
        print("Fold #%d RMSE: %f" %(k, score))
        pred_test = model.predict(X_test)[:,1]
        np.savetxt('../pred/Set%d/%s.%s.tst.fold%d.txt' % (fold_set, label_name, model_name, k), pred_test)
        
        logger.info("Fold #%d RMSE: %f" %(k, score))

    np.savetxt('../pred/Set%d/%s.%s.val.txt' % (fold_set, label_name, model_name), preds)
        
    scores = np.asarray(scores)
    print("%d-Fold: %f(%f)" % (n_folds, np.mean(scores), np.std(scores)))    
    logger.info("%d-Fold: %f(%f)" % (n_folds, np.mean(scores), np.std(scores)))
    logger.info("Done!!!")
    
if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        fold_set = int(sys.argv[1])
    else:
        fold_set = 0 
    
    
    train_and_predict(model_name='glove.6B.50d', label_name='clarity', pre_trained_model='../models/glove.6B.50d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=50, fold_set=fold_set)
    
    train_and_predict(model_name='glove.6B.100d', label_name='clarity',pre_trained_model='../models/glove.6B.100d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=100, fold_set=fold_set)    
    
    train_and_predict(model_name='glove.6B.200d', label_name='clarity', pre_trained_model='../models/glove.6B.200d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=200, fold_set=fold_set)
    
    train_and_predict(model_name='glove.6B.300d', label_name='clarity', pre_trained_model='../models/glove.6B.300d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=300, fold_set=fold_set)

    train_and_predict(model_name='glove.twitter.27B.25d', label_name='clarity', pre_trained_model='../models/glove.twitter.27B.25d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=25, fold_set=fold_set)
    
    train_and_predict(model_name='glove.twitter.27B.50d', label_name='clarity', pre_trained_model='../models/glove.twitter.27B.50d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=50, fold_set=fold_set)
    
    train_and_predict(model_name='glove.twitter.27B.100d', label_name='clarity', pre_trained_model='../models/glove.twitter.27B.100d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=100, fold_set=fold_set)
    
    train_and_predict(model_name='glove.twitter.27B.200d', label_name='clarity', pre_trained_model='../models/glove.twitter.27B.200d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=200, fold_set=fold_set)
    
    train_and_predict(model_name='glove.6B.50d', label_name='conciseness', pre_trained_model='../models/glove.6B.50d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=50, fold_set=fold_set)
    
    train_and_predict(model_name='glove.6B.100d', label_name='conciseness',pre_trained_model='../models/glove.6B.100d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=100, fold_set=fold_set)    
    
    train_and_predict(model_name='glove.6B.200d', label_name='conciseness', pre_trained_model='../models/glove.6B.200d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=200, fold_set=fold_set)
    
    train_and_predict(model_name='glove.6B.300d', label_name='conciseness', pre_trained_model='../models/glove.6B.300d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=300, fold_set=fold_set)

    train_and_predict(model_name='glove.twitter.27B.25d', label_name='conciseness', pre_trained_model='../models/glove.twitter.27B.25d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=25, fold_set=fold_set)
    
    train_and_predict(model_name='glove.twitter.27B.50d', label_name='conciseness', pre_trained_model='../models/glove.twitter.27B.50d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=50, fold_set=fold_set)
    
    train_and_predict(model_name='glove.twitter.27B.100d', label_name='conciseness', pre_trained_model='../models/glove.twitter.27B.100d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=100, fold_set=fold_set)
    
    train_and_predict(model_name='glove.twitter.27B.200d', label_name='conciseness', pre_trained_model='../models/glove.twitter.27B.200d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=200, fold_set=fold_set)
    
    
    train_and_predict(model_name='glove.6B.50d_v2', label_name='clarity', pre_trained_model='../models/glove.6B.50d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=50, fold_set=fold_set)
    
    train_and_predict(model_name='glove.6B.100d', label_name='clarity',pre_trained_model='../models/glove.6B.100d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=100, fold_set=fold_set)    
    
    train_and_predict(model_name='glove.6B.200d', label_name='clarity', pre_trained_model='../models/glove.6B.200d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=200, fold_set=fold_set)
    
    train_and_predict(model_name='glove.6B.300d', label_name='clarity', pre_trained_model='../models/glove.6B.300d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=300, fold_set=fold_set)

    train_and_predict(model_name='glove.twitter.27B.25d', label_name='clarity', pre_trained_model='../models/glove.twitter.27B.25d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=25, fold_set=fold_set)
    
    train_and_predict(model_name='glove.twitter.27B.50d', label_name='clarity', pre_trained_model='../models/glove.twitter.27B.50d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=50, fold_set=fold_set)
    
    train_and_predict(model_name='glove.twitter.27B.100d', label_name='clarity', pre_trained_model='../models/glove.twitter.27B.100d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=100, fold_set=fold_set)
    
    train_and_predict(model_name='glove.twitter.27B.200d', label_name='clarity', pre_trained_model='../models/glove.twitter.27B.200d.txt', 
                                        MAX_SEQUENCE_LENGTH=1000, MAX_NB_WORDS=20000, EMBEDDING_DIM=200, fold_set=fold_set)