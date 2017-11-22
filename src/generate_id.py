# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:31:13 2017

@author: tam
"""
import pandas as pd
df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])

df[['sku_id']].to_csv('../data/id.trn.csv', index=False)
df1[['sku_id']].to_csv('../data/id.tst.csv', index=False)
