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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
import lightgbm as lgb
import os.path
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


def process(model, model_name='xgb_v1', feat_name='title.bow', label_name='conciseness', fold_set=0, n_runs=1): 
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
            if feat_name.find('glove.') >= 0 or feat_name.find('giba') >= 0 or feat_name.find('entropy') >= 0 or feat_name.find('char') >= 0:
            
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
        if feat_name.find('glove.') >= 0 or feat_name.find('w2v') >= 0 or feat_name.find('giba') >= 0 or feat_name.find('entropy') >= 0 or feat_name.find('char') >= 0:
            X = np.nan_to_num(X)
            X_test = np.nan_to_num(X_test)
    
            
    feat_name = feat_name.replace('.title.','.').replace('_feat.','.').replace('_feat','.')
    
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
            validate_predict(model, X, y, X_test, cv, y_split, train_id, test_id, val_file, val_tst_file, model_type='xgb', verbose=True)
        else:
            validate_predict(model, X, y, X_test, cv, y_split, train_id, test_id, val_file, val_tst_file, model_type='lor', verbose=False)
          
import sys            
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        fold_set = 0
    else:
        fold_set = int(sys.argv[1])
        
    print('Training fold set %d' % fold_set)
    model = LogisticRegression()
    process(model, model_name='lor_v1',feat_name='title.boc.6grams_v2', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name=['title.boc.6grams_v2','sp_feat','title.color','title.brand'], fold_set=fold_set)

    process(model, model_name='lor_v1',feat_name='title.boc.6grams_v2', label_name='clarity', fold_set=fold_set)
    process(model, model_name='lor_v1',feat_name=['title.boc.6grams_v2','sp_feat','title.color','title.brand'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='lor_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex','char_shape_feat',
                                                  'desc_char_shape_feat','cat1_pred_feat'], fold_set=fold_set)
    
    model = MultinomialNB()
    process(model, model_name='nbc_v1',feat_name='title.boc.6grams_v2', fold_set=fold_set)
    process(model, model_name='nbc_v1',feat_name=['title.boc.6grams_v2','sp_feat','title.color','title.brand'], fold_set=fold_set)
    
    process(model, model_name='nbc_v1',feat_name='title.boc.6grams_v2', label_name='clarity', fold_set=fold_set)
    process(model, model_name='nbc_v1',feat_name=['title.boc.6grams_v2','sp_feat','title.color','title.brand'], label_name='clarity', fold_set=fold_set)
    
    model = SGDClassifier(loss='log', penalty='l1', n_iter=100, verbose=0)
    process(model, model_name='sgd_v1',feat_name='title.boc.6grams_v2', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name=['title.boc.6grams_v2','sp_feat','title.color','title.brand'], fold_set=fold_set)
    
    process(model, model_name='sgd_v1',feat_name='title.boc.6grams_v2', label_name='clarity', fold_set=fold_set)
    process(model, model_name='sgd_v1',feat_name=['title.boc.6grams_v2','sp_feat','title.color','title.brand'], label_name='clarity', fold_set=fold_set)
    
    
    
    model = Ridge(alpha=1.00, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=69069)
    process(model, model_name='rdg_v1',feat_name='title.boc.6grams_v2', fold_set=fold_set)
    process(model, model_name='rdg_v1',feat_name=['title.boc.6grams_v2','sp_feat','title.color','title.brand'], fold_set=fold_set)
    
    process(model, model_name='rdg_v1',feat_name='title.boc.6grams_v2', label_name='clarity', fold_set=fold_set)
    process(model, model_name='rdg_v1',feat_name=['title.boc.6grams_v2','sp_feat','title.color','title.brand'], label_name='clarity', fold_set=fold_set)
    
    model = lgb.LGBMClassifier(objective='binary',
                        num_leaves=40,
                        learning_rate=0.03,
                        n_estimators=700,
                        subsample=1,
                        max_depth=-1,
                        colsample_bytree=1,
                        subsample_freq=1,
                        min_child_weight=5,
                        max_bin=300)
    
    process(model, model_name='lgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt', 
                                                   'cat_cnt_feat', 'title_cat_feat'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','title.glove.6B.50d_mean'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 'giba_feat'], fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 'giba_feat', 'top_clarity'], fold_set=fold_set)
    
    model = lgb.LGBMClassifier(objective='binary',
                        num_leaves=38,
                        learning_rate=0.03,
                        n_estimators=733,
                        subsample=1,
                        max_depth=-1,
                        colsample_bytree=1,
                        subsample_freq=1,
                        min_child_weight=5,
                        max_bin=300)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 'giba_feat', 'top_clarity'], fold_set=fold_set)
    
    
    # 10-fold RMSE: 0.321659
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat'], fold_set=fold_set)
    
    # 10-fold RMSE: 0.321648
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex'], fold_set=fold_set)
    
    # 10-fold RMSE: 0.320838
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex','char_shape_feat'], fold_set=fold_set)
    
    
    # 
    model = lgb.LGBMClassifier(objective='binary',
                        num_leaves=40,
                        learning_rate=0.03,
                        n_estimators=700,
                        subsample=1,
                        max_depth=-1,
                        colsample_bytree=1,
                        subsample_freq=1,
                        min_child_weight=5,
                        max_bin=300)
    
    # 10-fold RMSE: 0.320214
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex','char_shape_feat',
                                                  'char_shape_feat_ex'], fold_set=fold_set)
    
    # 10-fold RMSE: 0.320164
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex','char_shape_feat',
                                                  'desc_char_shape_feat'], fold_set=fold_set)
    
    # 10-fold RMSE: 0.320178
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex','char_shape_feat',
                                                  'desc_char_shape_feat','cat1_pred_feat', 'char_shape_feat_ex'], fold_set=fold_set)
    
    #
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean',
                                                  'price_new','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex','char_shape_feat',
                                                  'desc_char_shape_feat','cat1_pred_feat', 'char_shape_feat_ex'], fold_set=fold_set)
    
    
    # 10-fold RMSE: 0.319890
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean',
                                                  'price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex','char_shape_feat',
                                                  'desc_char_shape_feat','cat1_pred_feat', 'char_shape_feat_ex',
                                                  'clarity_encode'], fold_set=fold_set)
    
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
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 'giba_feat'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 'giba_feat', 'top_clarity'], 
    label_name='clarity', fold_set=fold_set)
    
    
    process(model, model_name='lgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex','char_shape_feat','char_shape_feat_ex'], 
    label_name='clarity', fold_set=fold_set)
    
    
    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=18, 
                              n_estimators=500,
                              learning_rate=0.04,
                              subsample=0.7,
                              min_child_weight=5,
                              colsample_bylevel=0.75,
                              colsample_bytree=0.7)
    
    process(model, model_name='xgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt', 
                                                   'cat_cnt_feat', 'title_cat_feat'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','title.glove.6B.50d_mean'], fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat'], fold_set=fold_set)
    
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat','entropy_feat', 
                                                  'giba_feat', 'top_clarity','char_set_feat_ex','char_shape_feat',
                                                  'desc_char_shape_feat','cat1_pred_feat'], fold_set=fold_set)
    
    
    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=4, 
                              n_estimators=200,
                              learning_rate=0.05,
                              subsample=0.7,
                              min_child_weight=5,
                              colsample_bylevel=0.7,
                              colsample_bytree=0.7)
    
    process(model, model_name='xgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='xgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='xgb_v1', feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand', 'item_cnt', 
                                                   'cat_cnt_feat', 'title_cat_feat'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','title.glove.6B.50d_mean'], label_name='clarity', fold_set=fold_set)
    
    process(model, model_name='xgb_v1',feat_name=['title.boc.6grams_v2','xg_feat','title.color','title.brand',
                                                  'title.glove.twitter.27B.25d_mean','price_feat'], label_name='clarity', fold_set=fold_set)
    
    
    
    

    
    