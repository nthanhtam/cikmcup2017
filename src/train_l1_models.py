# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 23:40:59 2017

@author: gilberto_titericz
"""
#https://competitions.codalab.org/competitions/16652

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.externals import joblib
import xgboost as xgb

df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
df['clarity']     = np.loadtxt("../data/clarity_train.labels", dtype=int)

X, X_test = joblib.load('../features/all.dmp')


YSTRATIFIED = df['conciseness'].values

#Train conciseness
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
y = df['conciseness'].values
predtrain1 = np.zeros( (X.shape[0],2) )
predtest1  = []
for train_index, test_index in CV.split(X, YSTRATIFIED ):

    dtrain = xgb.DMatrix( X[train_index], label=y[train_index])
    dvalid = xgb.DMatrix( X[test_index ], label=y[test_index ])
    dtest  = xgb.DMatrix( X_test )

    param = {'max_depth':0, 'subsample':0.80, 'colsample_bytree':0.80, 'learning_rate':0.015, 'silent':1, 'objective':'reg:linear', 'nthread':8 }
    param['tree_method'] = 'hist'
    param['max_leaves'] = 128
    param['max_bin'] = 128
    param['min_child_weight'] = 2
    param['grow_policy'] = 'lossguide'
    param['eval_metric'] = 'rmse'    
    evallist  = [(dvalid,'eval')]
    bst = xgb.train( param, dtrain, 800, evallist, verbose_eval=50 )

    predtrain1[test_index,0] = bst.predict( dvalid  )
    #print np.sqrt( mean_squared_error( y[test_index],predtrain1[test_index,0] )  ), roc_auc_score( y[test_index],predtrain1[test_index,0] )

    predtest1.append( bst.predict( dtest ) )

#print np.sqrt( mean_squared_error( y,predtrain1[:,0] )  ), roc_auc_score( y,predtrain1[:,0] )


#Train Clarity
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=2017)
y = df['clarity'].values
predtest2  = []
for train_index, test_index in CV.split(X, YSTRATIFIED ):

    dtrain = xgb.DMatrix( X[train_index], label=y[train_index])
    dvalid = xgb.DMatrix( X[test_index ], label=y[test_index ])
    dtest  = xgb.DMatrix( X_test )

    param = {'max_depth':0, 'subsample':0.85, 'colsample_bytree':0.75, 'learning_rate':0.0075, 'silent':1, 'objective':'reg:linear', 'nthread':8 }
    param['tree_method'] = 'hist'
    param['max_leaves'] = 128
    param['max_bin'] = 150
    param['min_child_weight'] = 1
    param['grow_policy'] = 'lossguide'
    param['eval_metric'] = 'rmse'    
    evallist  = [(dvalid,'eval')]
    bst = xgb.train( param, dtrain, 600, evallist, verbose_eval=50 )

    predtrain1[test_index,1] = bst.predict( dvalid  )
#    print np.sqrt( mean_squared_error( y[test_index],predtrain1[test_index,1] )  ), roc_auc_score( y[test_index],predtrain1[test_index,0] )

    predtest2.append( bst.predict( dtest ) )

#print np.sqrt( mean_squared_error( y,predtrain1[:,1] )  ), roc_auc_score( y,predtrain1[:,1] )

y = df[ ['conciseness','clarity'] ].values
#print np.sqrt( mean_squared_error( y,predtrain1 )  )



#Write Train Validation and Test predictions to file
df                = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
df['clarity']     = np.loadtxt("../data/clarity_train.labels", dtype=int)
df = df[ ['sku_id','conciseness','clarity'] ]
predtrain1[ predtrain1>1.0 ] = 1.0
predtrain1[ predtrain1<0.0 ] = 0.0
df[ 'giba_conc_xgb1' ] = predtrain1[:,0]
df[ 'giba_clar_xgb1' ] = predtrain1[:,1]
df.head()
df.to_csv( '../pred/L1_giba_xgb1.FoldSet0.TRAINSET.csv', index=False )
#print np.sqrt( mean_squared_error( y,predtrain1 )  )

fold=0
for fold in range(10):
    pred1 = np.copy(predtest1[fold])
    pred1[ pred1>1.0 ] = 1.0
    pred1[ pred1<0.0 ] = 0.0
    pred2 = np.copy(predtest2[fold])
    pred2[ pred2>1.0 ] = 1.0
    pred2[ pred2<0.0 ] = 0.0
    df                = pd.read_csv("../data/data_test.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df[ ['sku_id'] ]
    df['conciseness'] = 0
    df['clarity'] = 0
    df[ 'giba_conc_xgb1' ] = pred1
    df[ 'giba_clar_xgb1' ] = pred2
    df.head()
    df.to_csv( '../pred/L1_giba_xgb1.FoldSet0.TESTSET.fold'+str(fold)+'.csv', index=False )
