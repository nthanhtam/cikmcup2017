# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 23:40:59 2017

@author: gilberto_titericz
"""
#https://competitions.codalab.org/competitions/16652

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb


df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
df['clarity'] = np.loadtxt("../data/clarity_train.labels", dtype=int)
#print df.shape
#print df.head(1)


def load_data( datafold=0 ):
    TRAIN1 = pd.read_csv( '../features/conciseness.L1_FoldSet0_x.TRAINSET.csv' )
    TRAIN1.drop( ['sku_id','conciseness'],inplace=True, axis=1 )
    TEST1  = pd.read_csv( '../features/conciseness.L1_FoldSet0_x.TESTSET.fold'+str(datafold)+'.csv' )
    TEST1.drop( ['sku_id','conciseness'],inplace=True, axis=1 )
    
    TRAIN2 = pd.read_csv( '../features/conciseness.L1_w2v.TRAINSET.csv' )
    TRAIN2.drop( ['sku_id','conciseness'],inplace=True, axis=1 )
    TEST2  = pd.read_csv( '../features/conciseness.L1_w2v.TESTSET.fold'+str(datafold)+'.csv' )
    TEST2.drop( ['sku_id','conciseness'],inplace=True, axis=1 )
    
    TRAIN3 = pd.read_csv( '../features/conciseness.L1_FoldSetX.TRAINSET.csv' )
    TRAIN3.drop( ['sku_id','conciseness',
                 'conciseness.etc_v1.title.boc.5grams.fold0',
                 'conciseness.etc_v1.title.boc.5grams.fold1',
                 'conciseness.etc_v1.title.boc.5grams.fold2',
                 'conciseness.etc_v1.title.boc.5grams.fold3',
                 'conciseness.etc_v1.title.boc.5grams.fold4',
                 'conciseness.etc_v1.title.boc.5grams.fold5',
                 'conciseness.etc_v1.title.boc.5grams.fold6',
                 'conciseness.etc_v1.title.boc.5grams.fold7',
                 'conciseness.etc_v1.title.boc.5grams.fold8',
                 'conciseness.etc_v1.title.boc.5grams.fold9',
                 'conciseness.xgb_v1.title.boc.5grams.fold0',
                 'conciseness.xgb_v1.title.boc.5grams.fold1',
                 'conciseness.xgb_v1.title.boc.5grams.fold2',
                 'conciseness.xgb_v1.title.boc.5grams.fold3',
                 'conciseness.xgb_v1.title.boc.5grams.fold4',
                 'conciseness.xgb_v1.title.boc.5grams.fold5',
                 'conciseness.xgb_v1.title.boc.5grams.fold6',
                 'conciseness.xgb_v1.title.boc.5grams.fold7',
                 'conciseness.xgb_v1.title.boc.5grams.fold8',
                 'conciseness.xgb_v1.title.boc.5grams.fold9'
                 ],inplace=True, axis=1 )
    TEST3 = pd.read_csv( '../features/conciseness.L1_FoldSetX.TESTSET.fold'+str(datafold)+'.csv' )
    TEST3.drop( ['sku_id','conciseness',
                 'conciseness.etc_v1.title.boc.5grams.fold0',
                 'conciseness.etc_v1.title.boc.5grams.fold1',
                 'conciseness.etc_v1.title.boc.5grams.fold2',
                 'conciseness.etc_v1.title.boc.5grams.fold3',
                 'conciseness.etc_v1.title.boc.5grams.fold4',
                 'conciseness.etc_v1.title.boc.5grams.fold5',
                 'conciseness.etc_v1.title.boc.5grams.fold6',
                 'conciseness.etc_v1.title.boc.5grams.fold7',
                 'conciseness.etc_v1.title.boc.5grams.fold8',
                 'conciseness.etc_v1.title.boc.5grams.fold9',
                 'conciseness.xgb_v1.title.boc.5grams.fold0',
                 'conciseness.xgb_v1.title.boc.5grams.fold1',
                 'conciseness.xgb_v1.title.boc.5grams.fold2',
                 'conciseness.xgb_v1.title.boc.5grams.fold3',
                 'conciseness.xgb_v1.title.boc.5grams.fold4',
                 'conciseness.xgb_v1.title.boc.5grams.fold5',
                 'conciseness.xgb_v1.title.boc.5grams.fold6',
                 'conciseness.xgb_v1.title.boc.5grams.fold7',
                 'conciseness.xgb_v1.title.boc.5grams.fold8',
                 'conciseness.xgb_v1.title.boc.5grams.fold9'
                 ],inplace=True, axis=1 )
    
    XTRAIN = np.hstack( (TRAIN1.values,TRAIN2.values,TRAIN3.values) )
    XTEST = np.hstack( (TEST1.values,TEST2.values,TEST3.values) )
    
    return XTRAIN, XTEST





YSTRATIFIED = df['conciseness'].values
TRAIN, TEST = load_data( 0 )

#Train a Linear ensemble for conciseness
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
y = df['conciseness'].values
predtrain1 = np.zeros( (TRAIN.shape[0],2) )
predtest1 = []
FOLD = 0
for train_index, test_index in CV.split(TRAIN, YSTRATIFIED ):
    TRAIN, TEST = load_data( FOLD )

    rd = Ridge(alpha=1.90, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.0001, solver='auto', random_state=69069)
    _ = rd.fit( TRAIN[train_index], y[train_index]  )
    predtrain1[test_index,0] = rd.predict( TRAIN[test_index]  )
#    print np.sqrt( mean_squared_error( y[test_index],predtrain1[test_index,0] )  ), roc_auc_score( y[test_index],predtrain1[test_index,0] )
    predtest1.append(  rd.predict( TEST ) )
    FOLD +=  1
#print np.sqrt( mean_squared_error( y,predtrain1[:,0] )  ), roc_auc_score( y,predtrain1[:,0] )



#Train a GBDT ensemble for conciseness using re-weighted samples
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
y = df['conciseness'].values
W = np.zeros( (len(y)) )
W[ y==0 ] = 1.0
W[ y==1 ] = 0.9614
W = W / np.mean(W)
predtrain2 = np.zeros( (TRAIN.shape[0],2) )
predtest2 = []
FOLD = 0
for train_index, test_index in CV.split(TRAIN, YSTRATIFIED ):
    TRAIN, TEST = load_data( FOLD )

    dtrain = xgb.DMatrix( TRAIN[train_index], weight=W[train_index], label=y[train_index])
    dvalid = xgb.DMatrix( TRAIN[test_index ], weight=W[test_index ], label=y[test_index ])
    dteste = xgb.DMatrix( TEST )

    param = {'max_depth':3, 'sub_sample':0.65, 'colsample_bytree':0.45, 'eta':0.00250, 'silent':1, 'objective':'reg:linear', 'nthread':8 }
    param['tree_method'] = 'hist'
    param['eval_metric'] = 'rmse'    
    evallist  = [(dvalid,'eval')]
    rd = xgb.train( param, dtrain, 4000, evallist, verbose_eval=50, early_stopping_rounds=33 )

    predtrain2[test_index,0] = rd.predict( dvalid )
#    print np.sqrt( mean_squared_error( y[test_index],predtrain2[test_index,0] )  ), roc_auc_score( y[test_index],predtrain2[test_index,0] )
    predtest2.append(  rd.predict( dteste ) )
    FOLD +=  1
#print np.sqrt( mean_squared_error( y,predtrain2[:,0] )  ), roc_auc_score( y,predtrain2[:,0] )




#Train a Linear ensemble for clarity
YSTRATIFIED = df['conciseness'].values
TRAIN, TEST = load_data( 0 )

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
y = df['clarity'].values
predtrain4 = np.zeros( (TRAIN.shape[0],2) )
predtest4 = []
FOLD = 0
for train_index, test_index in CV.split(TRAIN, YSTRATIFIED ):
    TRAIN, TEST = load_data( FOLD )

    rd = Ridge(alpha=1.90, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.0001, solver='auto', random_state=69069)
    _ = rd.fit( TRAIN[train_index], y[train_index]  )
    predtrain4[test_index,0] = rd.predict( TRAIN[test_index]  )
#    print np.sqrt( mean_squared_error( y[test_index],predtrain4[test_index,0] )  ), roc_auc_score( y[test_index],predtrain4[test_index,0] )
    predtest4.append(  rd.predict( TEST ) )
    FOLD +=  1
#print np.sqrt( mean_squared_error( y,predtrain4[:,0] )  ), roc_auc_score( y,predtrain4[:,0] )


#Train a GBDT ensemble for clarity using re-weighted samples
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
y = df['clarity'].values
W = np.zeros( (len(y)) )
W[ y==0 ] = 1.0
W[ y==1 ] = 0.923623177/0.943362
W = W / np.mean(W)
predtrain5 = np.zeros( (TRAIN.shape[0],2) )
predtest5 = []
FOLD = 0
for train_index, test_index in CV.split(TRAIN, YSTRATIFIED ):
    TRAIN, TEST = load_data( FOLD )

    dtrain = xgb.DMatrix( TRAIN[train_index], weight=W[train_index], label=y[train_index])
    dvalid = xgb.DMatrix( TRAIN[test_index ], weight=W[test_index ], label=y[test_index ])
    dteste = xgb.DMatrix( TEST )

    param = {'max_depth':3, 'sub_sample':0.40, 'colsample_bytree':0.40, 'eta':0.0020, 'silent':1, 'objective':'reg:linear', 'nthread':8 }
    param['eval_metric'] = 'rmse'    
    evallist  = [(dvalid,'eval')]
    rd = xgb.train( param, dtrain, 4000, evallist, verbose_eval=50, early_stopping_rounds=33 )

    predtrain5[test_index,0] = rd.predict( dvalid )
#    print np.sqrt( mean_squared_error( y[test_index],predtrain5[test_index,0] )  ), roc_auc_score( y[test_index],predtrain5[test_index,0] )
    predtest5.append(  rd.predict( dteste ) )
    FOLD +=  1
#print np.sqrt( mean_squared_error( y,predtrain5[:,0] )  ), roc_auc_score( y,predtrain5[:,0] )




#Write Train Validation Results to File
df                = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
df['clarity']     = np.loadtxt("../data/clarity_train.labels", dtype=int)
df = df[ ['sku_id','conciseness','clarity'] ]
df[ 'giba_conc_l2' ] = 0.35*predtrain1[:,0] + 0.65*predtrain2[:,0] 
df[ 'giba_clar_l2' ] = 0.30*predtrain4[:,0] + 0.70*predtrain5[:,0] 
np.mean(df[ 'giba_conc_l2' ])
np.mean(df[ 'giba_clar_l2' ])
df.head()
df.to_csv( '../pred/giba_l3.TRAINSET.csv', index=False )
#print np.sqrt( mean_squared_error( df['conciseness'],df['giba_conc_l2'] )  )
#print np.sqrt( mean_squared_error( df['clarity'],df['giba_clar_l2'] )  )


#Write Test Results to File
df                = pd.read_csv("../data/data_test.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
df = df[ ['sku_id'] ]
df['conciseness'] = 0
df['clarity'] = 0
df[ 'giba_conc_l2' ] = 0.35*np.mean(predtest1,axis=0) + 0.65*np.mean(predtest2,axis=0) 
df[ 'giba_clar_l2' ] = 0.30*np.mean(predtest4,axis=0) + 0.70*np.mean(predtest5,axis=0) 
np.mean(df[ 'giba_conc_l2' ])
np.mean(df[ 'giba_clar_l2' ])
df.head()
df.to_csv( '../pred/giba_l3.TESTSET.csv', index=False )
