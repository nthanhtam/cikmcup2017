# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:52:35 2016

@author: nguyentt
"""
import pandas as pd
import numpy as np
import sys
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        fold_set = 0
    else:
        fold_set = int(sys.argv[1])
        
    model_names = ['glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d',
                   'glove.twitter.27B.25d','glove.twitter.27B.50d','glove.twitter.27B.100d',
                   'glove.twitter.27B.200d']
    
#    fold_set = 0
#    df_tr = pd.read_csv('../features/conciseness.L1_FoldSetX.TRAINSET.csv')        
#    df_va =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.full.csv')  
#    df_va_fold =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.fold0.csv')  
#    
#    df_tr = df_tr[df_tr.columns[:2]]
#    df_va = df_va[df_va.columns[:2]]
#    
#    
#    for m in model_names:
#        df_tr[m] = np.loadtxt('../pred/Set%d/conciseness.%s.val.txt' % (fold_set,m))
#        df_va[m] = np.loadtxt('../pred/Set%d/conciseness.%s.tst.full.txt' % (fold_set,m))
#        
#        df_tr['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.val.txt' % (fold_set,m))
#        df_va['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.tst.full.txt' % (fold_set,m))
#        
#    df_tr.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TRAINSET.csv' % fold_set, index=False)        
#    df_va.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TESTSET.full.csv' % fold_set, index=False)        
#    
#    
#    for fold in range(10):
#        df_va_fold =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.fold%d.csv' % fold)
#        df_va_fold = df_va_fold[df_va_fold.columns[:2]]
#        for m in model_names:
#            df_va_fold[m] = np.loadtxt('../pred/Set%d/conciseness.%s.tst.fold%d.txt' % (fold_set, m, fold))
#            df_va_fold['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.tst.fold%d.txt' % (fold_set, m, fold))
#            
#        df_va_fold.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TESTSET.fold%d.csv' % (fold_set, fold), index=False)        
#        
#        
#
#    fold_set = 2
#    df_tr = pd.read_csv('../features/conciseness.L1_FoldSetX.TRAINSET.csv')        
#    df_va =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.full.csv')  
#    df_va_fold =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.fold0.csv')  
#    
#    df_tr = df_tr[df_tr.columns[:2]]
#    df_va = df_va[df_va.columns[:2]]
#    
#    
#    for m in model_names:
#        df_tr[m] = np.loadtxt('../pred/Set%d/conciseness.%s.val.txt' % (fold_set,m))
#        df_va[m] = np.loadtxt('../pred/Set%d/conciseness.%s.tst.full.txt' % (fold_set,m))
#        
#        df_tr['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.val.txt' % (fold_set,m))
#        df_va['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.tst.full.txt' % (fold_set,m))
#        
#    df_tr.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TRAINSET.csv' % fold_set, index=False)        
#    df_va.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TESTSET.full.csv' % fold_set, index=False)        
#    
#    
#    for fold in range(10):
#        df_va_fold =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.fold%d.csv' % fold)
#        df_va_fold = df_va_fold[df_va_fold.columns[:2]]
#        for m in model_names:
#            df_va_fold[m] = np.loadtxt('../pred/Set%d/conciseness.%s.tst.fold%d.txt' % (fold_set, m, fold))
#            df_va_fold['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.tst.fold%d.txt' % (fold_set, m, fold))
#            
#        df_va_fold.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TESTSET.fold%d.csv' % (fold_set, fold), index=False)        
#                
#        
    fold_set = 3
    df_tr = pd.read_csv('../features/conciseness.L1_FoldSetX.TRAINSET.csv')        
    df_va =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.full.csv')  
    df_va_fold =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.fold0.csv')  
    
    df_tr = df_tr[df_tr.columns[:2]]
    df_va = df_va[df_va.columns[:2]]
    
    
    for m in model_names:
        print(m)
        df_tr[m] = np.loadtxt('../pred/Set%d/conciseness.%s.val.txt' % (fold_set,m))
        df_va[m] = np.loadtxt('../pred/Set%d/conciseness.%s.tst.full.txt' % (fold_set,m))
        
        df_tr['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.val.txt' % (fold_set,m))
        df_va['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.tst.full.txt' % (fold_set,m))
        
    df_tr.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TRAINSET.csv' % fold_set, index=False)        
    df_va.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TESTSET.full.csv' % fold_set, index=False)        
    
    
    for fold in range(10):
        df_va_fold =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.fold%d.csv' % fold)
        df_va_fold = df_va_fold[df_va_fold.columns[:2]]
        for m in model_names:
            df_va_fold[m] = np.loadtxt('../pred/Set%d/conciseness.%s.tst.fold%d.txt' % (fold_set, m, fold))
            df_va_fold['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.tst.fold%d.txt' % (fold_set, m, fold))
            
        df_va_fold.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TESTSET.fold%d.csv' % (fold_set, fold), index=False)        
                
        
    fold_set = 4
    df_tr = pd.read_csv('../features/conciseness.L1_FoldSetX.TRAINSET.csv')        
    df_va =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.full.csv')  
    df_va_fold =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.fold0.csv')  
    
    df_tr = df_tr[df_tr.columns[:2]]
    df_va = df_va[df_va.columns[:2]]
    
    
    for m in model_names:
        print(m)
        df_tr[m] = np.loadtxt('../pred/Set%d/conciseness.%s.val.txt' % (fold_set,m))
        df_va[m] = np.loadtxt('../pred/Set%d/conciseness.%s.tst.full.txt' % (fold_set,m))
        
        df_tr['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.val.txt' % (fold_set,m))
        df_va['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.tst.full.txt' % (fold_set,m))
        
    df_tr.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TRAINSET.csv' % fold_set, index=False)        
    df_va.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TESTSET.full.csv' % fold_set, index=False)        
    
    
    for fold in range(10):
        df_va_fold =  pd.read_csv('../features/conciseness.L1_FoldSetX.TESTSET.fold%d.csv' % fold)
        df_va_fold = df_va_fold[df_va_fold.columns[:2]]
        for m in model_names:
            df_va_fold[m] = np.loadtxt('../pred/Set%d/conciseness.%s.tst.fold%d.txt' % (fold_set, m, fold))
            df_va_fold['cla_%s' % m] = np.loadtxt('../pred/Set%d/clarity.%s.tst.fold%d.txt' % (fold_set, m, fold))
            
        df_va_fold.to_csv('../features/conciseness.L1_w2v_FoldSet%d.TESTSET.fold%d.csv' % (fold_set, fold), index=False)        
                        
    
    