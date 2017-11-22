# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:12:30 2016

@author: nguyentt
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib

from nltk.stem.snowball import EnglishStemmer

def stemmed_words(doc):
    stemmer = EnglishStemmer()
    analyzer = TfidfVectorizer().build_analyzer()
    return ' '.join((stemmer.stem(w) for w in analyzer(doc)))


def extract_title_text_features(train_file, test_file, 
                                analyzer='char', ngram_range=(1, 1), lowercase=True, stem=False, stop_words=None):
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    if stem:
        df['title'] = df['title'].map(lambda x: stemmed_words(x))
    
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=lowercase, stop_words=stop_words)
    vect.fit(df["title"].tolist())
    
    X_title = vect.transform(df["title"])
    X_title_tr = X_title[:n_trains,:]
    X_title_val = X_title[n_trains:,:]
    return X_title_tr, X_title_val, vect.get_feature_names()
    
def select_feature(model, X, y, X_test, feat_names, set_name='5grams'):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
    
    for k, (trainId, valId) in enumerate(cv.split(X, y)):
        print(k)
        feat_train_file = '../features/title.boc.%s.fold%d.trn' % (set_name, k)
        feat_test_file = '../features/title.boc.%s.fold%d.tst' % (set_name, k)
        
        X_train = X[trainId,:]
        y_train = y[trainId]
        
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
        
        indices = np.argsort(importances)[::-1]
            
        data = []
        for f in range(X.shape[1]):
            data.append({'rank': f+1, 'feature': feat_names[indices[f]], 'importance': importances[indices[f]], 'std': std[indices[f]], 'index': indices[f]})
                
        data = pd.DataFrame(data)
        data = data[data['importance'] > 0]
        
        sel = SelectFromModel(model, prefit=True)
        X1 = sel.transform(X)
        X_test1 = sel.transform(X_test)
        
        print('saving %s' % feat_train_file)
        joblib.dump(X1, feat_train_file)
        
        print('saving %s' % feat_test_file)
        joblib.dump(X_test1, feat_test_file)
        print(X1.shape, X_test1.shape)
    
    
def process(model, model_name='etc_v1', feat_name='title.bow', label_name='conciseness', ngram_range=(2, 5), set_name='5grams'): 
    print("loading data...")
    train_file = '../data/data_train.csv'
    test_file = '../data/data_valid.csv'
    
    X, X_test, feat_names = extract_title_text_features(train_file, test_file, analyzer='char', ngram_range=ngram_range)
    
    y = np.loadtxt("../data/%s_train.labels" % label_name, dtype=int)
    select_feature(model, X, y, X_test, feat_names, set_name)
                  
if __name__ == "__main__":    
    model = ExtraTreesClassifier(n_jobs=-1, n_estimators=500, max_depth=6) 
    process(model)
    
    model = ExtraTreesClassifier(n_jobs=-1, n_estimators=500, max_depth=6) 
    process(model, ngram_range=(6, 8), set_name='8grams')
    