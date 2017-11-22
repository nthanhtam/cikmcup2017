# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:52:35 2016

@author: nguyentt
"""
import pandas as pd
import numpy as np
import os

def gen_l2_feature(model_name='lor_v1', feat_name='title.bow', label_name='conciseness', pred_folder='', n_runs=1): 
    
    bag_val_pred = []
    cols = []
    for run in range(n_runs):
        f = '%s.%s.%s' % (label_name, model_name, feat_name)
        cols.append(f)
        
        val_pred = []
        for k in range(10): # 10-fold
            fn = "../%s/%s.fold%d-run%d.csv" % (pred_folder, f, k, run)
            val_pred.append(pd.read_csv(fn))
                
        val_pred = pd.concat(val_pred)
        #val_pred = val_pred.sort_values(by="Id").reset_index(drop=True)
            
        print(val_pred.shape)
        bag_val_pred.append(val_pred)
    
    val_pred = bag_val_pred[0]
    val_pred.columns = ["sku_id", "pred_0"] 
    for i, pp in enumerate(bag_val_pred[1:]):
        val_pred["pred_%d" % (i+1)] = pp["pred"]
    
    val_pred.columns = ["sku_id"] + cols
    
    tr = pd.read_csv("../data/id.trn.csv")
    tr['conciseness'] = np.loadtxt("../data/%s_train.labels" % label_name, dtype=int)
    val_pred = tr.merge(val_pred, how="left", on="sku_id")
    val_pred.to_csv("../features/%s.TRAINSET.csv" % f, index=False)        
    
    test_set = []
    for k in range(10): 
        pred = []
        f = '%s.%s.%s' % (label_name, model_name, feat_name)
        for b in range(1):
            fn = "../%s/%s.fold%d-run%d-test.csv" % (pred_folder, f, k, b)
            pred.append(pd.read_csv(fn))
            
        
        p = pred[0] 
        p.columns = ["sku_id", "pred_0"]  
        for i, pp in enumerate(pred[1:]):
            p["pred_%d" % (i+1)] = pp["pred"]
        
        p["conciseness"] = 0
        p.columns = ["sku_id"] + cols + ["conciseness"]
        p[["sku_id", "conciseness"] + cols].to_csv("../features/%s.TESTSET.fold%d.csv" % (f, k), index=False)
        test_set.append("../features/%s.TESTSET.fold%d.csv" % (f, k))
        
    pred = []
    for run in range(n_runs):
        f = '%s.%s.%s' % (label_name, model_name, feat_name)
        fn = "../%s/%s.run%d.full.csv" % (pred_folder, f, run)
        pred.append(pd.read_csv(fn))
        
    p = pred[0] 
    p.columns = ["sku_id", "pred_0"]  
    for k, pp in enumerate(pred[1:]):
        p["pred_%d" % (k+1)] = pp["pred"]
    
    p["conciseness"] = 0
    p.columns = ["sku_id"] + cols + ["conciseness"]
    p[["sku_id", "conciseness"] + cols].to_csv("../features/%s.TESTSET.full.csv" % f, index=False)
    
    return ("../features/%s.TRAINSET.csv" % f, "../features/%s.TESTSET.full.csv" % f, test_set)


if __name__ == '__main__':
    
    fold_set = 0
    list_file = [x for x in os.listdir('../pred/Set%d' % fold_set) if x.endswith('full.csv')]
    
    file_names = []
    for f in list_file:
        label_name = f.split('.')[0]
        model_name = f.split('.')[1]
        feat_name = '.'.join(f.split('.')[2:]).replace('.run0.full.csv','')
        file_names.append(gen_l2_feature(model_name=model_name, feat_name=feat_name, label_name=label_name, pred_folder='pred/Set%d' % fold_set))
    
    
    tr, va, test_set = file_names[0]
    df_tr = pd.read_csv(tr)
    df_va = pd.read_csv(va)
    
    df_te_set = []
    for te in test_set:
        df_te_set.append(pd.read_csv(te))
    

    for tr, va, test_set in file_names[1:]:
        t = pd.read_csv(tr)
        df_tr[t.columns[2]] = t[t.columns[2]]
        t = pd.read_csv(va)
        df_va[t.columns[2]] = t[t.columns[2]]
        for k, te in enumerate(test_set):
            t = pd.read_csv(te)
            df_te_set[k][t.columns[2]] = t[t.columns[2]]
        
    df_tr.to_csv('../features/conciseness.L1_FoldSet%d.TRAINSET.csv' % fold_set, index=False)        
    df_va.to_csv('../features/conciseness.L1_FoldSet%d.TESTSET.full.csv' % fold_set, index=False)        
    for k in range(10):
        df_te = df_te_set[k]
        df_va.to_csv('../features/conciseness.L1_FoldSet%d.TESTSET.fold%d.csv' % (fold_set, k), index=False) 
        
    