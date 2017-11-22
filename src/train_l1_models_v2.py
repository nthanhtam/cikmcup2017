# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:12:30 2016

@author: nguyentt
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from scipy import sparse as sp
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import os.path
from sklearn.model_selection import StratifiedKFold
import sys
from sklearn.metrics import mean_squared_error, roc_auc_score


def rmse_score(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

def validate_predict(model, X, y, X_test, cv, y_split,
                     train_id, test_id, val_file, val_tst_file, 
                     model_type='xgb', verbose=True):
    
    preds = np.zeros(X.shape[0])
    
    if cv is None:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
    
    if y_split is None:
        folds = cv.split(X, y)
    else:
        folds = cv.split(X, y_split)
        
    for k, (trainId, valId) in enumerate(folds):
        X_train = X[trainId,:]
        y_train = y[trainId]
        X_val = X[valId,:]
        y_val = y[valId]
        if model_type in ['xgb','lgb']:
            model.fit(X_train, y_train,
                          eval_set=[(X_train, y_train), (X_val, y_val)],
                          eval_metric='rmse',
                          verbose=verbose)
                                      
        else:
            model.fit(X_train, y_train)
        
        if hasattr(model, 'predict_proba'):
           preds[valId] = model.predict_proba(X_val)[:,1]
           trn_pred = model.predict_proba(X_train)[:,1]
        else:
            preds[valId] = model.predict(X_val)
            trn_pred = model.predict(X_train)
                
        print("Fold {}, Train RMSE: {:.6f}. Val RMSE: {:.6f}. Val AUC: {:.6f}".format(k, rmse_score(y_train, trn_pred), rmse_score(y_val, preds[valId]), roc_auc_score(y_val, preds[valId])))
        
        val_id = train_id.loc[valId,:]
        val_id["pred"] = preds[valId]
        val_id.to_csv(val_file.replace("foldk","fold%d" % k), index=False)

        if hasattr(model, 'predict_proba'):
            test_id["pred"] = model.predict_proba(X_test)[:,1]
        else:
            test_id["pred"] = model.predict(X_test)
                
        test_id.to_csv(val_tst_file.replace("foldk","fold%d" % k), index=False)
        
    print("{}-fold RMSE: {:.6f}".format(k+1, rmse_score(y, preds)))


def process(model, model_name='xgb_v1', feat_name='title.bow', label_name='conciseness', fold_set=1, n_runs=1): 
    print("loading data...")
    train_id = pd.read_csv("../data/id.trn.csv")
    test_id = pd.read_csv("../data/id.tst.csv")
    y = np.loadtxt("../data/%s_train.labels" % label_name, dtype=int)
    y_split = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    
    feat_names = feat_name

    if type(feat_name) == list:
        feat_names = feat_name
        X = []
        X_test = []
        for feat_name in feat_names:
            print('loading feature %s' % feat_name)
            X1 = joblib.load('../features/%s.trn' % feat_name)
            X_test1 = joblib.load('../features/%s.tst' % feat_name)
            if feat_name.find('.glove.') > 0:
                X1 = np.nan_to_num(X1)
                X_test1 = np.nan_to_num(X_test1) 
            elif model_name.startswith('lor'):
                if feat_name=='cat_cnt_feat' or feat_name=='title_cat_feat':
                    X1[np.isnan(X1)] = 0
                    X_test1[np.isnan(X_test1)] = 0
                    
            X.append(X1)
            X_test.append(X_test1)
        
        X = sp.csr_matrix(sp.hstack(X, format='csr'))
        X_test= sp.csr_matrix(sp.hstack(X_test, format='csr'))
        feat_name = '.'.join(feat_names)
    
    else:
        X = joblib.load('../features/%s.trn' % feat_name)
        X_test = joblib.load('../features/%s.tst' % feat_name)
        if feat_name.find('.glove.') > 0 or feat_name.find('w2v') > 0:
            X = np.nan_to_num(X)
            X_test = np.nan_to_num(X_test)
    
    feat_name = feat_name.replace('.title.','.').replace('_feat.','.')
    
    if model_name.startswith('xgb'):
        X = sp.csc_matrix(sp.hstack((X, sp.csr_matrix(np.ones((X.shape[0], 1))))))
        X_test = sp.csc_matrix(sp.hstack((X_test, sp.csr_matrix(np.ones((X_test.shape[0], 1))))))
        
    
    print(X.shape, X_test.shape)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017+fold_set*777)
    seeds = [111,222,333,444,555,666,777,888,999]
    print('starting training...')
    for run in range(n_runs):
        seed= seeds[run]
        if hasattr(model, 'seed'):
            model.seed=seed
            
        if hasattr(model, 'random_state'):
            model.random_state=seed
            
        val_file = "../pred/Set%d/%s.%s.%s.foldk-run%d.csv" % (fold_set,label_name, model_name, feat_name, run)
        val_tst_file = "../pred/Set%d/%s.%s.%s.foldk-run%d-test.csv" % (fold_set,label_name, model_name, feat_name, run)
        tst_file = "../pred/Set%d/%s.%s.%s.run%d.full.csv" % (fold_set, label_name, model_name, feat_name, run)
    
        if os.path.isfile(tst_file):
            print('model exist')
            break
        
        if model_name.startswith('xgb'):
            validate_predict(model, X, y, X_test, cv, y_split, train_id, test_id, val_file, val_tst_file, model_type='xgb', custom_metric=False, verbose=True)
        else:
            validate_predict(model, X, y, X_test, cv, y_split, train_id, test_id, val_file, val_tst_file, model_type='lor', custom_metric=False, verbose=False)
                      
if __name__ == "__main__":
    
    if len(sys.argv) <= 1:
        fold_set = 0
    else:
        fold_set = int(sys.argv[1])
        
    model = LogisticRegression()
    
    process(model, model_name='lor_v1',feat_name='title.bow', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.bow.3grams', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.bow.stem', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.bow.stem.3grams', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.boc.5grams', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name=['title.boc.5grams','xg_feat'], fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name=['title.boc.5grams','sp_feat'], fold_set=fold_set)
    
    process(model, model_name='lor_v1',feat_name='title.bow.stopword', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.boc.stem.5grams', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand'], fold_set=fold_set)

    process(model, model_name='lor_v1',feat_name='title.bow', label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.bow.3grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.bow.stem', label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.bow.stem.3grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.boc.5grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name=['title.boc.5grams','xg_feat'], label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name=['title.boc.5grams','sp_feat'], label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.bow.stopword', label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name='title.boc.stem.5grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand'], label_name='clarity', fold_set=fold_set)
    
    
    process(model, model_name='lor_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'cat_cnt_feat', 'title_cat_feat'], fold_set=fold_set)
    
    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=18, 
                              n_estimators=500,
                              learning_rate=0.04,
                              subsample=0.7,
                              min_child_weight=5,
                              colsample_bylevel=0.7,
                              colsample_bytree=0.7)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand', 'cat_cnt_feat'], fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand', 'cat_cnt_feat', 'price_feat'], fold_set=fold_set)
    
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand', 'cat_cnt_feat', 'title_cat_feat'], fold_set=fold_set)
    
    
    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=18, 
                              n_estimators=500,
                              learning_rate=0.04,
                              subsample=0.7,
                              min_child_weight=5,
                              colsample_bylevel=0.75,
                              colsample_bytree=0.7)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'cat_cnt_feat', 'title_cat_feat'], fold_set=fold_set)

    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'cat_cnt_feat', 'title_cat_feat', 'price_feat'], fold_set=fold_set)

    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'title_len_hist'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand','title.glove.twitter.27B.25d_mean'], fold_set=fold_set)
    
    import lightgbm as lgb
    model = lgb.LGBMClassifier(objective='binary',
                        num_leaves=35,
                        learning_rate=0.04,
                        n_estimators=500)
    
    process(model, model_name='lgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand'], fold_set=fold_set)
    
    # 
    model = lgb.LGBMClassifier(objective='binary',
                        num_leaves=35,
                        learning_rate=0.04,
                        n_estimators=500)
    
    process(model, model_name='lgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt', 'cat_cnt_feat', 'title_cat_feat'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand','title.glove.twitter.27B.25d_mean'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand','title.glove.twitter.27B.25d_mean','title.glove.6B.50d_mean'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand','title.glove.twitter.27B.25d_mean','price_feat'], fold_set=fold_set)
    
    #
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color', 'desc.color','title.brand','desc.brand','title.glove.twitter.27B.25d_mean'], fold_set=fold_set)

    
    model = lgb.LGBMClassifier(objective='binary',
                        num_leaves=35,
                        learning_rate=0.04,
                        n_estimators=128)
    
    process(model, model_name='lgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='lgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='lgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt', 
                                                   'cat_cnt_feat', 'title_cat_feat'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','title.glove.6B.50d_mean'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat'], label_name='clarity', fold_set=fold_set)
    
    
    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=4, 
                              n_estimators=200,
                              learning_rate=0.05,
                              subsample=0.7,
                              min_child_weight=5,
                              colsample_bylevel=0.7,
                              colsample_bytree=0.7)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'cat_cnt_feat', 'title_cat_feat'], label_name='clarity', fold_set=fold_set)

    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'cat_cnt_feat', 'title_cat_feat', 'price_feat'], label_name='clarity', fold_set=fold_set)
    
    

    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=18, 
                              n_estimators=1000,
                              learning_rate=0.04,
                              subsample=0.6,
                              min_child_weight=5,
                              colsample_bylevel=1,
                              colsample_bytree=1)
    
    
    process(model, model_name='xgb_v2',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand', 'cat_cnt_feat', 'title_cat_feat'], fold_set=fold_set)

    process(model, model_name='xgb_v1',feat_name='title.bow', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name='title.bow.3grams', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name='title.bow.stem', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name='title.bow.stem.3grams', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name='title.boc.5grams', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat'], fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name='title.boc.5grams', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand', 'price_feat'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.bow','title.boc.5grams','xg_feat','title.color','title.brand'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.bow','title.boc.5grams','xg_feat','title.color','title.brand','title.w2v_mean'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand','title.w2v_mean'], fold_set=fold_set)
    
    
    
    model = RandomForestClassifier(n_jobs=-1, n_estimators=50)
    process(model, model_name='rfc_v1',feat_name='title.bow', fold_set=fold_set)
    
    process(model, model_name='rfc_v1',feat_name='title.bow.3grams', fold_set=fold_set)
    process(model, model_name='rfc_v1',feat_name='title.bow.stem', fold_set=fold_set)
    process(model, model_name='rfc_v1',feat_name='title.bow.stem.3grams', fold_set=fold_set)
    process(model, model_name='rfc_v1',feat_name='title.boc.5grams', fold_set=fold_set)
    process(model, model_name='rfc_v1',feat_name=['title.boc.5grams','xg_feat'], fold_set=fold_set)
    process(model, model_name='rfc_v1',feat_name=['title.boc.5grams','sp_feat'], fold_set=fold_set)
    
    
    model = ExtraTreesClassifier(n_jobs=-1, n_estimators=1000, max_depth=18, 
                                 criterion ='entropy',
                                 bootstrap=False, random_state=7777)
    
    process(model, model_name='etc_v1',feat_name='title.bow', fold_set=fold_set)
    process(model, model_name='etc_v1',feat_name='title.bow.3grams', fold_set=fold_set)
    process(model, model_name='etc_v1',feat_name='title.bow.stem', fold_set=fold_set)
    process(model, model_name='etc_v1',feat_name='title.bow.stem.3grams', fold_set=fold_set)
    process(model, model_name='etc_v1',feat_name='title.boc.5grams', fold_set=fold_set)
    
    
    model = MultinomialNB()
    process(model, model_name='nbc_v1',feat_name='title.bow', fold_set=fold_set)
    
    process(model, model_name='nbc_v1',feat_name='title.bow.3grams', fold_set=fold_set)
    process(model, model_name='nbc_v1',feat_name='title.bow.stem', fold_set=fold_set)
    process(model, model_name='nbc_v1',feat_name='title.bow.stem.3grams', fold_set=fold_set)
    process(model, model_name='nbc_v1',feat_name='title.boc.5grams', fold_set=fold_set)
    process(model, model_name='nbc_v1',feat_name=['title.boc.5grams','sp_feat'], fold_set=fold_set)
    
    
    model = SGDClassifier(loss='log', penalty='l1', n_iter=100, verbose=0)
    process(model, model_name='sgd_v1',feat_name='title.bow', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='title.bow.3grams', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='title.bow.stem', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='title.bow.stem.3grams', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='title.boc.5grams', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name=['title.boc.5grams','xg_feat'], fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name=['title.boc.5grams','sp_feat'], fold_set=fold_set)
    
    model = SGDClassifier(loss='log', penalty='l1', n_iter=100, verbose=0)
    process(model, model_name='sgd_v1',feat_name='title.bow', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='title.bow.3grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='title.bow.stem', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='title.bow.stem.3grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='title.boc.5grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name=['title.boc.5grams','xg_feat'], label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name=['title.boc.5grams','sp_feat'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='sgd_v1',feat_name='desc.bow', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='desc.bow.3grams', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='desc.bow.stem', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='desc.bow.stem.3grams', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='desc.boc.5grams', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name=['desc.boc.5grams','sp_feat'], fold_set=fold_set)
    
    process(model, model_name='sgd_v1',feat_name='desc.bow', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='desc.bow.3grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='desc.bow.stem', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='desc.bow.stem.3grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name='desc.boc.5grams', label_name='clarity', fold_set=fold_set)
    
    model = MLPClassifier(verbose=True, max_iter=5, solver='lbfgs', hidden_layer_sizes =(128,), random_state=8888)
    process(model, model_name='mlp_v1',feat_name='title.bow', fold_set=fold_set)
    process(model, model_name='mlp_v1',feat_name='title.bow.3grams', fold_set=fold_set)

    
    model = ExtraTreesClassifier(n_jobs=-1, n_estimators=1000, max_depth=18, 
                                 criterion ='entropy',
                                 bootstrap=False, random_state=7777)
    
    process(model, model_name='etc_v1',feat_name=['title.boc.6grams_v2','xg_feat'], fold_set=fold_set)
    process(model, model_name='etc_v1',feat_name='title.boc.6grams_v2', fold_set=fold_set)
    process(model, model_name='etc_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand'], fold_set=fold_set)
    process(model, model_name='etc_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'price_feat'], fold_set=fold_set)
    
    process(model, model_name='etc_v1',feat_name=['title.boc.5grams','xg_feat'], fold_set=fold_set)
    process(model, model_name='etc_v1',feat_name='title.boc.5grams', fold_set=fold_set)
    process(model, model_name='etc_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand'], fold_set=fold_set)
    
    process(model, model_name='etc_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand', 'price_feat'], fold_set=fold_set)
    

    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=18, 
                              n_estimators=1000,
                              learning_rate=0.04,
                              subsample=0.6,
                              min_child_weight=5,
                              colsample_bylevel=1,
                              colsample_bytree=1)
    
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand','title.glove.6B.100d_mean','title.glove.twitter.27B.25d_mean'], fold_set=fold_set)
    
    
    process(model, model_name='xgb_v1',feat_name=['xg_feat','title.glove.6B.100d_mean','title.glove.twitter.27B.25d_mean'], fold_set=fold_set)

    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand','title.glove.6B.100d_concat','title.glove.twitter.27B.25d_concat'], fold_set=fold_set)
    
    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=4, 
                              n_estimators=200,
                              learning_rate=0.05,
                              subsample=0.7,
                              min_child_weight=5,
                              colsample_bylevel=0.7,
                              colsample_bytree=0.7)
    
    process(model, model_name='xgb_v1',feat_name='title.bow', label_name='clarity', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name='title.bow.3grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name='title.bow.stem', label_name='clarity', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name='title.bow.stem.3grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name='title.boc.5grams', label_name='clarity', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat'], label_name='clarity', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','sp_feat'], label_name='clarity', fold_set=fold_set)
    process(model, model_name='xgb_v1',feat_name=['title.boc.5grams','xg_feat','title.color','title.brand'], label_name='clarity')
    
    
    
    
    
    

    
    