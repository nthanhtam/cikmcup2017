# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:12:30 2017

@author: nguyentt
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

def rmse_score(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


def validate_predict(model, X, y, X_test_fold, cv, y_split,
                     train_id, test_id, val_file, val_tst_file, 
                     model_type='xgb', custom_metric=False, verbose=True, batch_size=16, epochs=5):
    
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
            if custom_metric:
                model.fit(X_train, y_train,
                          eval_set=[(X_train, y_train), (X_val, y_val)],
                          eval_metric='rmse',
                          verbose=verbose)
            else:
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
        
        X_test = X_test_fold[k]
        
        if hasattr(model, 'predict_proba'):
           test_id["pred"] = model.predict_proba(X_test)[:,1]
        else:
           test_id["pred"] = model.predict(X_test)
                
        test_id.to_csv(val_tst_file.replace("foldk","fold%d" % k), index=False)
        
    print("{}-fold RMSE: {:.6f}".format(k+1, rmse_score(y, preds)))


def process(model, model_name='l2_xgb_v1', feat_name='L1_v1', label_name='conciseness', using_rank=False, fold_set=0, n_runs=1): 
    print("loading data...")
    train_id = pd.read_csv("../data/id.trn.csv")
    test_id = pd.read_csv("../data/id.tst.csv")

    print('loading feature %s' % feat_name)
    y = np.loadtxt("../data/%s_train.labels" % label_name, dtype=int)
    y_split = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    
    feat_names = feat_name

    if type(feat_name)==list:
        feat_names = feat_name
        train = []
        test_fold = {}
        for fold in range(10):
            test_fold[fold] = []
            
        for i, feat_name in enumerate(feat_names):
            print(feat_name)
            if i==0:
                train.append(pd.read_csv('../features/%s.%s.TRAINSET.csv' % ('conciseness', feat_name)))
                print(train[-1].shape)
                for fold in range(10):
                    test_fold[fold].append(pd.read_csv('../features/%s.%s.TESTSET.fold%d.csv' % ('conciseness', feat_name, fold)))
                    print(test_fold[fold][-1].shape)
            else:
                train.append(pd.read_csv('../features/%s.%s.TRAINSET.csv' % ('conciseness', feat_name)).drop(['sku_id'], axis=1))
                print(train[-1].shape)
                for fold in range(10):
                    test_fold[fold].append(pd.read_csv('../features/%s.%s.TESTSET.fold%d.csv' % ('conciseness', feat_name, fold)).drop(['sku_id'], axis=1))
                    print(test_fold[fold][-1].shape)
    
        train =pd.concat(train, axis=1) 
        for fold in range(10):
            test_fold[fold] = pd.concat(test_fold[fold], axis=1)
            
        feat_name = '.'.join(feat_names)
    else:
        train = pd.read_csv('../features/%s.%s.TRAINSET.csv' % ('conciseness', feat_name))
        for fold in range(10):
            test_fold[fold] = pd.read_csv('../features/%s.%s.TESTSET.fold%d.csv' % ('conciseness', feat_name, fold))
                    
    drop_cols = ['sku_id', 
                 'conciseness'
                 ]
    
    X = train.drop(drop_cols, axis=1).values
    print(X.shape)
    X_test_fold = {}
    for fold in range(10):
        X_test_fold[fold] = test_fold[fold].drop(drop_cols, axis=1).values
        print(X_test_fold[fold].shape)
        
   
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017+fold_set*777)
    seeds = [111,222,333,444,555,666,777,888,999]
    
    for run in range(n_runs):
        seed= seeds[run]
        if hasattr(model, 'seed'):
            model.seed=seed
            
        if hasattr(model, 'random_state'):
            model.random_state=seed
            
        val_file = "../pred/L2_v2/%s.%s.%s.foldk-run%d.csv" % (label_name, model_name, feat_name, run)
        val_tst_file = "../pred/L2_v2/%s.%s.%s.foldk-run%d-test.csv" % (label_name, model_name, feat_name, run)
    
        if model_name.find('xgb') >= 0 or model_name.find('lgb') >= 0:
            validate_predict(model, X, y, X_test_fold, cv, y_split, train_id, test_id, val_file, val_tst_file, model_type='xgb', custom_metric=False, verbose=True)
                 
        else:
            validate_predict(model, X, y, X_test_fold, cv, y_split, train_id, test_id, val_file, val_tst_file, model_type='lor', custom_metric=False, verbose=False)
    
if __name__ == "__main__":
    
    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=7, 
                              n_estimators=1000,
                              learning_rate=0.008,
                              subsample=0.6,
                              min_child_weight=5,
                              colsample_bytree=1.,
                              gamma=0,
                              max_delta_step=0)
    
    # 10-fold RMSE: 0.207749 -- LB: 0.242408
    process(model, feat_name=['L1_FoldSetX','L1_FoldSet0','L1_w2v'], model_name='l2_xgb_v2', label_name='clarity')
    
    
    model = XGBClassifier(objective='binary:logistic', 
                              max_depth=7, 
                              n_estimators=650,
                              learning_rate=0.008,
                              subsample=0.6,
                              min_child_weight=5,
                              colsample_bytree=1.,
                              gamma=0,
                              max_delta_step=0)
    
       
    process(model, feat_name=['L1_FoldSetX','L1_FoldSet0_xx', 'L1_FoldSet1', 'L1_FoldSet2', 
                              'L1_w2v', 'L1_w2v_FoldSet1', 'L1_w2v_FoldSet2','L1_giba_xgb1.FoldSet0'], model_name='l2_xgb_v2')
    
    
    process(model, feat_name=['L1_FoldSetX','L1_FoldSet0_xx', 'L1_FoldSet1', 'L1_FoldSet2', 'L1_FoldSet3', 'L1_FoldSet4',
                              'L1_w2v', 'L1_w2v_FoldSet1', 'L1_w2v_FoldSet2','L1_giba_xgb1.FoldSet0'], model_name='l2_xgb_v2')
        
        
   
     