# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:18:22 2017

@author: tam
"""

import pandas as pd
import numpy as np

df = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  

def rescale_conciseness( x ):
    expected_mean = 0.6585
    pred_mean = np.mean(x)
    adjusted_value = expected_mean - pred_mean
    x = x + adjusted_value
    x[x<0.0] = 0.0
    x[x>1.0] = 1.0
    return x

def rescale_clarity( x ):
    expected_mean = 0.923623177
    pred_mean = np.mean(x)
    adjusted_value = expected_mean - pred_mean
    x = x + adjusted_value
    x[x<0.0] = 0.0
    x[x>1.0] = 1.0
    return x


# FINAL SUBMISSION
p = pd.read_csv('../features/giba_l3_TESTSET.csv')

pred = 0
for k in range(10):
    df = pd.read_csv('../pred/L2/conciseness.l2_xgb_v2.L1_FoldSetX.L1_FoldSet0_xx.L1_w2v.L1_w2v_FoldSet1.L1_FoldSet1.L1_FoldSet2.L1_w2v_FoldSet1.L1_giba_xgb1.FoldSet0.fold%d-run0-test.csv' % k)
    pred += df['pred']
    
pred /= 10.0    
df['pred'] = pred*0.6 + 0.4*p['giba_conc_l2'].values
df['pred'] = rescale_conciseness(df['pred'].values)

df[['pred']].to_csv('../submission/conciseness_test.predict', header=None, index=False)


pred = 0
for k in range(10):
    df = pd.read_csv('../pred/L2_v2/clarity.l2_xgb_v2.L1_FoldSetX.L1_FoldSet0.L1_w2v.fold%d-run0-test.csv' % k)
    pred += df['pred']
    
for k in range(10):
    df = pd.read_csv('../pred/L2/clarity.l2_xgb_v2.L1_FoldSetX.L1_FoldSet0.L1_w2v.fold%d-run0-test.csv' % k)
    pred += df['pred']
 
pred /= 20.0 
df['pred'] = pred
df['pred'] = rescale_clarity(df['pred'].values)

df[['pred']].to_csv('../submission/clarity_test.predict', header=None, index=False)
