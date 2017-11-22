import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.externals import joblib


def dict_found(x, d):
    x = x.lower()
    xx = [y.replace('(','').replace(')','').replace('[','').replace(']','').strip() for y in x.split(' ')]
    for i in xx:
        if i in d:
            return 1
    return 0

def is_model(x):
    if x[-1].isdigit() and x[-1].isalpha():
        return 1
    else:
        return 0
    
    
def extract_dict_features():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    df = pd.concat((df,df1))
   
    brand = {}
    for b in open('../data/brands_from_lazada_portal.txt'):
        brand[b.strip().lower()] = 0
        
    color = {}
    for c in open('../data/colors.txt'):
        color[c.strip().lower()] = 0
        
    df['title_has_brand'] = df['title'].map(lambda x: dict_found(x, brand))
    df['title_has_color'] = df['title'].map(lambda x: dict_found(x, color))
    df['title_has_model'] = df['title'].map(lambda x: is_model(x))
    feat_names.extend(['title_has_brand','title_has_color','title_has_model'])

    df['title_is_lower'] = df['title'].map(lambda x: x.islower())
    df['title_is_upper'] = df['title'].map(lambda x: x.isupper())
    df['title_is_title'] = df['title'].map(lambda x: x.istitle())
    feat_names.extend(['title_is_lower','title_is_upper','title_is_title'])
    
    df['title_has_year'] = df['title'].map(lambda x: 1 if x.find('2015')>=0 or x.find('2016')>=0 or x.find('2017')>=0 else 0)
    df['title_has_weight'] = df['title'].map(lambda x: 1 if x.lower().find('kg')>=0  else 0)
    df['title_has_ssize'] = df['title'].map(lambda x: 1 if x.lower().find('gb')>=0  else 0)
    df['title_has_len'] = df['title'].map(lambda x: 1 if x.lower().find(' m ')>=0 or x.lower().find('cm') >=0 else 0)
    df['title_has_size'] = df['title'].map(lambda x: 1 if x.lower().find('"')>=0 or x.lower().find('inch') >=0  else 0)
    df['title_has_wh'] = df['title'].map(lambda x: 1 if x.lower().find(' * ')>=0 or x.lower().find(' x ') >=0  else 0)
    df['title_has_pcs'] = df['title'].map(lambda x: 1 if x.lower().find(' pcs ')>=0 else 0)
    
    feat_names.extend(['title_has_year','title_has_weight','title_has_ssize','title_has_len','title_has_size','title_has_wh','title_has_pcs'])

    df['title_has_women'] = df['title'].map(lambda x: 1 if x.lower().find('women')>=0 else 0)
    df['title_has_men'] = df['title'].map(lambda x: 1 if x.lower().find(' men')>=0 else 0)
    df['title_has_fashion'] = df['title'].map(lambda x: 1 if x.lower().find('fashion')>=0 else 0)
    df['title_has_watch'] = df['title'].map(lambda x: 1 if x.lower().find('watch')>=0 else 0)
    df['title_has_bag'] = df['title'].map(lambda x: 1 if x.lower().find('bag')>=0 else 0)
    df['title_has_brand'] = df['title'].map(lambda x: 1 if x.lower().find('brand')>=0 else 0)
    df['title_has_shoe'] = df['title'].map(lambda x: 1 if x.lower().find('shoe')>=0 else 0)

    feat_names.extend(['title_has_women','title_has_men','title_has_fashion','title_has_watch','title_has_brand','title_has_bag','title_has_shoe'])

    X = df[feat_names].astype(float).values
    
    X_dict_train = X[:num_trains,:]
    X_dict_val = X[num_trains:,:]
      
    return X_dict_train, X_dict_val


def extract_entropy_features():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    df = pd.concat((df,df1))
    
    tbl = df.groupby(['category_lvl_1','category_lvl_2'])['sku_id'].count()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_1')['sku_id'].count()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_1')
    tbl2['entropy12'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy12']),'entropy12'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_1')['entropy12'].sum()
    tbl3 = tbl3.to_frame(name='entropy12').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_1')
    feat_names.append('entropy12')
    
    
    tbl = df.groupby(['category_lvl_2','category_lvl_3'])['sku_id'].count()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_2')['sku_id'].count()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_2')
    tbl2['entropy23'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy23']),'entropy23'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_2')['entropy23'].sum()
    tbl3 = tbl3.to_frame(name='entropy23').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_2')
    feat_names.append('entropy23')
    
    
    tbl = df.groupby(['category_lvl_1','category_lvl_3'])['sku_id'].count()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_1')['sku_id'].count()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_1')
    tbl2['entropy13'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy13']),'entropy13'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_1')['entropy13'].sum()
    tbl3 = tbl3.to_frame(name='entropy13').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_1')
    feat_names.append('entropy13')
    
    tbl = df.groupby(['category_lvl_1','category_lvl_2','category_lvl_3'])['sku_id'].count()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_1')['sku_id'].count()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_1')
    tbl2['entropy123'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy123']),'entropy123'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_1')['entropy123'].sum()
    tbl3 = tbl3.to_frame(name='entropy123').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_1')
    feat_names.append('entropy123')
    
    X = df[feat_names].astype(float).values
    
    X_ent_train = X[:num_trains,:]
    X_ent_val = X[num_trains:,:]
     
    return X_ent_train, X_ent_val


def extract_brand_entropy_features():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    df = pd.concat((df,df1))
    
    df['brand'] = df['sku_id'].map(lambda x: x[:5])
    
    tbl = df.groupby(['category_lvl_1','brand'])['sku_id'].count()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_1')['sku_id'].count()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_1')
    tbl2['entropy12'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy12']),'entropy12'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_1')['entropy12'].sum()
    tbl3 = tbl3.to_frame(name='entropy12').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_1')
    feat_names.append('entropy12')
    
    tbl = df.groupby(['category_lvl_2','brand'])['sku_id'].count()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_2')['sku_id'].count()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_2')
    tbl2['entropy22'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy22']),'entropy22'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_2')['entropy22'].sum()
    tbl3 = tbl3.to_frame(name='entropy22').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_2')
    feat_names.append('entropy22')
    
    tbl = df.groupby(['category_lvl_3','brand'])['sku_id'].count()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_3')['sku_id'].count()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_3')
    tbl2['entropy32'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy32']),'entropy32'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_3')['entropy32'].sum()
    tbl3 = tbl3.to_frame(name='entropy32').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_3')
    feat_names.append('entropy32')
    
    
    X = df[feat_names].astype(float).values
    
    X_ent_train = X[:num_trains,:]
    X_ent_val = X[num_trains:,:]
     
    return X_ent_train, X_ent_val

def extract_title_entropy_features():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    df = pd.concat((df,df1))
    
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    
    tbl = df.groupby(['category_lvl_1','category_lvl_2'])['title_len'].sum()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_1')['title_len'].sum()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_1')
    tbl2['entropy12'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy12']),'entropy12'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_1')['entropy12'].sum()
    tbl3 = tbl3.to_frame(name='entropy12').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_1')
    feat_names.append('entropy12')
    
    
    tbl = df.groupby(['category_lvl_2','category_lvl_3'])['title_len'].sum()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_2')['title_len'].sum()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_2')
    tbl2['entropy23'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy23']),'entropy23'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_2')['entropy23'].sum()
    tbl3 = tbl3.to_frame(name='entropy23').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_2')
    feat_names.append('entropy23')
    
    
    tbl = df.groupby(['category_lvl_1','category_lvl_3'])['title_len'].sum()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_1')['title_len'].sum()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_1')
    tbl2['entropy13'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy13']),'entropy13'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_1')['entropy13'].sum()
    tbl3 = tbl3.to_frame(name='entropy13').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_1')
    feat_names.append('entropy13')
    
    tbl = df.groupby(['category_lvl_1','category_lvl_2','category_lvl_3'])['title_len'].sum()
    tbl = tbl.to_frame(name='subgrp_cnt').reset_index()
   
    tbl1 = df.groupby('category_lvl_1')['title_len'].sum()
    tbl1 = tbl1.to_frame(name='grp_cnt').reset_index()
    
    tbl2 = tbl.merge(tbl1, how='left', on='category_lvl_1')
    tbl2['entropy123'] = -np.log(tbl2['subgrp_cnt'] / tbl2['grp_cnt']) * tbl2['subgrp_cnt'] / tbl2['grp_cnt']
    tbl2.ix[pd.isnull(tbl2['entropy123']),'entropy123'] = 0
    
    tbl3 = tbl2.groupby('category_lvl_1')['entropy123'].sum()
    tbl3 = tbl3.to_frame(name='entropy123').reset_index()
    
    df = df.merge(tbl3, how='left', on='category_lvl_1')
    feat_names.append('entropy123')
    
    X = df[feat_names].astype(float).values
    
    X_ent_train = X[:num_trains,:]
    X_ent_val = X[num_trains:,:]
     
    return X_ent_train, X_ent_val


def extract_label_encode_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    df['clarity'] = np.loadtxt("../data/clarity_train.labels", dtype=int)
    
    df = df.fillna('NA')
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1['conciseness'] = -1
    df1['clarity'] = -1
    
    df1 = df1.fillna('NA')
    
    from leave_one_out import LeaveOneOutEncoder    

    loo = LeaveOneOutEncoder(cols=['category_lvl_1','category_lvl_2','category_lvl_3'])
    loo.fit(df, df['clarity'].values)
    
    df = loo.transform(df)
    df1 = loo.transform(df1)
    
    num_trains = df.shape[0]
    df = pd.concat((df, df1))
    
    
    feat_names = ['category_lvl_1','category_lvl_2','category_lvl_3']
    X = df[feat_names].values
    
    X_encode_train = X[:num_trains,:]
    X_encode_val = X[num_trains:,:]
     
    return X_encode_train, X_encode_val

def extract_clarity_encode_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    df['clarity'] = np.loadtxt("../data/clarity_train.labels", dtype=int)
    
    df = df.fillna('NA')
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1['conciseness'] = -1
    df1['clarity'] = -1
    
    df1 = df1.fillna('NA')
    
    from leave_one_out import LeaveOneOutEncoder    
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    df1['title_len'] = df1['title'].map(lambda x: len(x.split(' ')))
    
    df['title_clen'] = df['title'].map(lambda x: len(x))
    df1['title_clen'] = df1['title'].map(lambda x: len(x))
    
    feat_names = ['category_lvl_1','category_lvl_2', 'category_lvl_3','title_len']

    loo = LeaveOneOutEncoder(cols=feat_names,
                             randomized=True, 
                             random_state=8888)
    loo.fit(df, df['clarity'].values)
    
    df = loo.transform(df)
    df1 = loo.transform(df1)
    
    num_trains = df.shape[0]
    df = pd.concat((df, df1))
    
    
    X = df[feat_names].values
    
    X_encode_train = X[:num_trains,:]
    X_encode_val = X[num_trains:,:]
     
    return X_encode_train, X_encode_val


def extract_clarity_encode_feature_ex():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    df['clarity'] = np.loadtxt("../data/clarity_train.labels", dtype=int)
    
    df = df.fillna('NA')
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1['conciseness'] = -1
    df1['clarity'] = -1
    
    df1 = df1.fillna('NA')
    
    from leave_one_out import LeaveOneOutEncoder    
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    df1['title_len'] = df1['title'].map(lambda x: len(x.split(' ')))
    
    df['cat_title_len'] = df['category_lvl_1'] + df['title_len'].map(lambda x: str(x))
    df1['cat_title_len'] = df1['category_lvl_1'] + df1['title_len'].map(lambda x: str(x))
    
    df['title_clen'] = df['title'].map(lambda x: len(x))
    df1['title_clen'] = df1['title'].map(lambda x: len(x))
    
    feat_names = ['category_lvl_1','category_lvl_2', 'category_lvl_3','title_len','cat_title_len']

    loo = LeaveOneOutEncoder(cols=feat_names,
                             randomized=True, 
                             random_state=8888)
    loo.fit(df, df['clarity'].values)
    
    df = loo.transform(df)
    df1 = loo.transform(df1)
    
    num_trains = df.shape[0]
    df = pd.concat((df, df1))
    
    
    X = df[feat_names].values
    
    X_encode_train = X[:num_trains,:]
    X_encode_val = X[num_trains:,:]
     
    return X_encode_train, X_encode_val

def extract_conciseness_encode_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    df['clarity'] = np.loadtxt("../data/clarity_train.labels", dtype=int)
    
    df = df.fillna('NA')
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1['conciseness'] = -1
    df1['clarity'] = -1
    
    df1 = df1.fillna('NA')
    
    from leave_one_out import LeaveOneOutEncoder    
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    df1['title_len'] = df1['title'].map(lambda x: len(x.split(' ')))
    
    df['title_clen'] = df['title'].map(lambda x: len(x))
    df1['title_clen'] = df1['title'].map(lambda x: len(x))
    
    feat_names = ['category_lvl_1','category_lvl_2', 'category_lvl_3','title_len']

    loo = LeaveOneOutEncoder(cols=feat_names,
                             randomized=True, 
                             random_state=8888)
    loo.fit(df, df['conciseness'].values)
    
    df = loo.transform(df)
    df1 = loo.transform(df1)
    
    num_trains = df.shape[0]
    df = pd.concat((df, df1))
    
    
    X = df[feat_names].values
    
    X_encode_train = X[:num_trains,:]
    X_encode_val = X[num_trains:,:]
     
    return X_encode_train, X_encode_val


def extract_top_conciseness_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    df = df.fillna('NA')
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
   
    df1 = df1.fillna('NA')
    
    feat_names = [x[16:].replace('\n','').lstrip() for x in open('../data/Top50_for_Conciseness.txt')]

    num_trains = df.shape[0]
    df = pd.concat((df, df1))
    df['title'] = df['title'].map(lambda x: x.lower())
    for c in feat_names:
        df[c] = df['title'].map(lambda x: 1 if x.find(c) >= 0 else 0)
    
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

def extract_top_clarity_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    df = df.fillna('NA')
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
   
    df1 = df1.fillna('NA')
    
    feat_names = [x[16:].replace('\n','').lstrip() for x in open('../data/Top_50_feature_importances_for__Clarity__according_LinearSVC.txt')]

    num_trains = df.shape[0]
    df = pd.concat((df, df1))
    df['title'] = df['title'].map(lambda x: x.lower())
    for c in feat_names:
        df[c] = df['title'].map(lambda x: 1 if x.find(c) >= 0 else 0)
    
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

def giba_features():
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    num_trains = df.shape[0]
    df = pd.concat((df, df1))
    
    df['country'].value_counts()
    df['sku_id'].value_counts()
    df['sku_id2'] = df['sku_id'].apply( lambda x: x[:2] )
    df['sku_id2'].value_counts()
    df['sku_id_2'] = df['sku_id'].apply( lambda x: x[-2:] )
    df['sku_id_2'].value_counts()
    df['sku_id57'] = df['sku_id'].apply( lambda x: x[5:7] )
    df['sku_id57'].value_counts()
    df['sku_id_53'] = df['sku_id'].apply( lambda x: x[-5:-4] )
    df['sku_id_53'].value_counts()
    df.fillna( 'empty', inplace=True )
    
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    df['title_clen'] = df['title'].map(lambda x: len(x))
    df['desc_len'] = df['short_description'].map(lambda x: len(x.split(' ')))
    df['desc_clen'] = df['short_description'].map(lambda x: len(x))
    
    df['price_avg1'] = df.groupby( ['country','category_lvl_1'] )['price'].transform(np.mean)
    df['price_mdn1'] = df.groupby( ['country','category_lvl_1'] )['price'].transform(np.median)
    df['price_std1'] = df.groupby( ['country','category_lvl_1'] )['price'].transform(np.std)
    df['price_avg2'] = df.groupby( ['country','category_lvl_2'] )['price'].transform(np.mean)
    df['price_mdn2'] = df.groupby( ['country','category_lvl_2'] )['price'].transform(np.median)
    df['price_std2'] = df.groupby( ['country','category_lvl_2'] )['price'].transform(np.std)
    df['price_avg3'] = df.groupby( ['country','category_lvl_3'] )['price'].transform(np.mean)
    df['price_mdn3'] = df.groupby( ['country','category_lvl_3'] )['price'].transform(np.median)
    df['price_std3'] = df.groupby( ['country','category_lvl_3'] )['price'].transform(np.std)
    df['price_avg4'] = df.groupby( ['country','sku_id2'] )['price'].transform(np.mean)
    df['price_mdn4'] = df.groupby( ['country','sku_id2'] )['price'].transform(np.median)
    df['price_std4'] = df.groupby( ['country','sku_id2'] )['price'].transform(np.std)
    
    df['num_special'] = df['title'].map(lambda x: len(x) - len(re.sub('[^a-zA-Z0-9-_*.]', '',x))  )
    df['num_num'] = df['title'].map(lambda x: len(x) - len(re.sub('[^a-zA-Z-_*.]', '',x))  )
    df['diff_num'] = df['num_special'] - df['num_num']
    df['num_special_desc'] = df['short_description'].map(lambda x: len(x) - len(re.sub('[^a-zA-Z0-9-_*.]', '',x))  )
    df['num_num_desc'] = df['short_description'].map(lambda x: len(x) - len(re.sub('[^a-zA-Z-_*.]', '',x))  )
    df['diff_num_desc'] = df['num_special_desc'] - df['num_num_desc']
   
    def count_words( w1, w2 ):
        w1 = w1.lower()
        w2 = w2.lower()
        w1 = w1.split(' ') 
        w2 = w2.split(' ') 
        count = 0
        for a in w1:
            for b in w2:
                if a==b:
                    count+=1
        return count
        
    df['equ1'] = df[ ['title','short_description'] ].apply(lambda x: count_words( x['title'], x['short_description'] ), axis=1 )
    df.head(2)
    
    df['N1'] = df.groupby(['title'])['price'].transform('count')
    df['N2'] = df.groupby(['short_description'])['price'].transform('count')
    
    
    feat_names = ['price_avg1','price_mdn1','price_std1','price_avg2','price_mdn2',
                  'price_std2', 'price_avg3', 'price_mdn3','price_std3','price_avg4','price_mdn4','price_std4',
                  'num_special','num_num','diff_num','num_special_desc','num_num_desc','diff_num_desc','equ1','N1','N2']
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val
    
def extract_char_feat_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    df['title'] = df['title'].map(lambda x: x.lower())

    df['title_len'] = df['title'].map(lambda x: len(x))    
    df['c_set_len'] = df['title'].map(lambda x: len(list(set(x.split()))))
    df['c_alpha_len'] = df['title'].map(lambda x: len(list(set([c for c in x if c.isalpha()]))))
    df['c_digit_len'] = df['title'].map(lambda x: len(list(set([c for c in x if c.isdigit()]))))
    df['c_printable_len'] = df['title'].map(lambda x: len(list(set([c for c in x if c.isprintable()]))))
   
    df['r1'] = df['c_alpha_len'] / df['c_set_len']
    df['r2'] = df['c_digit_len'] / df['c_set_len']
    df['r3'] = df['c_printable_len'] / df['c_set_len']
    
    df['r4'] = df['c_alpha_len'] / df['title_len']
    df['r5'] = df['c_digit_len'] / df['title_len']
    df['r6'] = df['c_printable_len'] / df['title_len']
    
    df['r6'] = df['c_set_len'] / df['title_len']
    
    
    feat_names = ['c_set_len','c_alpha_len','c_digit_len','c_printable_len','r1','r2','r3']
    
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

def extract_char_feat_feature_ex():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    df['title'] = df['title'].map(lambda x: x.lower())

    df['title_len'] = df['title'].map(lambda x: len(x.split()))    
    df['c_set_len'] = df['title'].map(lambda x: len(list(set([c for c in x]))))
    df['c_alpha_len'] = df['title'].map(lambda x: len(list(set([c for c in x if c.isalpha()]))))
    df['c_digit_len'] = df['title'].map(lambda x: len(list(set([c for c in x if c.isdigit()]))))
    df['c_printable_len'] = df['title'].map(lambda x: len(list(set([c for c in x if c.isprintable()]))))
    
  
    
    df['r1'] = df['c_alpha_len'] / df['c_set_len']
    df['r2'] = df['c_digit_len'] / df['c_set_len']
    df['r3'] = df['c_printable_len'] / df['c_set_len']
    
    df['r4'] = df['c_alpha_len'] / df['title_len']
    df['r5'] = df['c_digit_len'] / df['title_len']
    df['r6'] = df['c_printable_len'] / df['title_len']
    
    df['r7'] = df['c_set_len'] / df['title_len']
    
    
    feat_names = ['c_set_len','c_alpha_len','c_digit_len','c_printable_len','r1','r2','r3','r4','r5','r6','r7']
    
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

def extract_char_shape_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    df['upper_cnt'] = df['title'].map(lambda x: len([c for c in x if c.isupper()]))
    df['lower_cnt'] = df['title'].map(lambda x: len([c for c in x if c.islower()]))
    df['space_cnt'] = df['title'].map(lambda x: len([c for c in x if c==' ' or c=='\t']))
    df['startswith_alpha'] = df['title'].map(lambda x: x[0].isalpha()).astype(int)
    df['startswith_space'] = df['title'].map(lambda x: x[0]==' ').astype(int)
    
    df['space_ratio'] = df['space_cnt'] / df['title_len'] 
    
    feat_names = ['startswith_alpha','startswith_space', 'space_ratio', 'space_cnt']
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

def get_shape(x):
    s = ''
    for c in x:
        if c.isalpha():
            if c.isupper():
                s += 'A'
            else:
                s += 'a'
        elif c.isdigit():
            s += 'N'
        else:
            s += c
    return s
        
def extract_char_shape_feature_ex():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    
    df['has_html_and'] = df['title'].map(lambda x: x.find('&amp;')>=0).astype(int)
    df['has_html_space'] = df['title'].map(lambda x: x.find('&nbsp;')>=0).astype(int)
    
    df['title'] = df['title'].map(lambda x: x.replace('&amp;','&'))
    df['title'] = df['title'].map(lambda x: x.replace('&nbsp;',' '))
    
    df['has_duplicates'] = df['title'].map(lambda x: len(x.split(' '))!=len(list(set(x.split(' '))))).astype(int)
    
    df['title'] = df['title'].map(lambda x: get_shape(x))
    
    df['shape_set_len'] = df['title'].map(lambda x: len(set(x.split(' '))))
    
    
    feat_names = ['has_html_and','has_html_space', 'has_duplicates','shape_set_len']
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

from nltk.corpus import wordnet
from nltk.corpus import words

def english_cnt(x):
    cnt = 0
    for w in x.split(' '):
        if not wordnet.synsets(w):
            cnt += 1
    return cnt

def nltk_corpus_cnt(x):
    cnt = 0
    for w in x.split(' '):
        if x in words.words():
            cnt += 1
    return cnt

def extract_char_desc_shape_feature_ex():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    
    
    df['has_html_and'] = df['short_description'].map(lambda x: x.find('&amp;')>=0).astype(int)
    df['has_html_space'] = df['short_description'].map(lambda x: x.find('&nbsp;')>=0).astype(int)
    
    df['has_html_class'] = df['short_description'].map(lambda x: x.find('class=')>=0).astype(int)
    df['has_html_style'] = df['short_description'].map(lambda x: x.find('style=')>=0).astype(int)
    
    df['short_description'] = df['short_description'].map(lambda x: x.replace('&amp;','&'))
    df['short_description'] = df['short_description'].map(lambda x: x.replace('&nbsp;',' '))
    
    df['short_description'] = df['short_description'].fillna(' ').map(lambda x: clean_html(x))
    
    df['has_duplicates'] = df['title'].map(lambda x: len(x.split(' '))!=len(list(set(x.split(' '))))).astype(int)
    
    df['short_description'] = df['short_description'].map(lambda x: get_shape(x))
    
    df['shape_set_len'] = df['short_description'].map(lambda x: len(set(x.split(' '))))
    
    feat_names = ['has_html_and','has_html_space', 'has_duplicates','shape_set_len','has_html_class','has_html_style']
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val


def extract_english_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    df['short_description'] = df['short_description'].fillna(' ').map(lambda x: clean_html(x))
    
    df['title_en_cnt'] = df['title'].map(lambda x: english_cnt(x))
    df['desc_en_cnt'] = df['short_description'].map(lambda x: english_cnt(x))
    
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    df['desc_len'] = df['short_description'].map(lambda x: len(x.split(' ')))
    
    df['title_en_cnt_ratio'] = df['title_en_cnt']/df['title_len']
    df['desc_en_cnt_ratio'] = df['desc_en_cnt']/df['desc_len']
    
    feat_names = ['title_en_cnt','desc_en_cnt', 'title_en_cnt_ratio','desc_en_cnt_ratio']
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

def extract_nltk_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    df['short_description'] = df['short_description'].fillna(' ').map(lambda x: clean_html(x))
    
    df['title_en_cnt'] = df['title'].map(lambda x: nltk_corpus_cnt(x))
    df['desc_en_cnt'] = df['short_description'].map(lambda x: nltk_corpus_cnt(x))
    
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    df['desc_len'] = df['short_description'].map(lambda x: len(x.split(' ')))
    
    df['title_en_cnt_ratio'] = df['title_en_cnt']/df['title_len']
    df['desc_en_cnt_ratio'] = df['desc_en_cnt']/df['desc_len']
    
    feat_names = ['title_en_cnt','desc_en_cnt', 'title_en_cnt_ratio','desc_en_cnt_ratio']
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val


def word_len(x):
    words = x.split(' ')
    word_lens = list(map(lambda x: len(x), words))
    return word_lens

def extract_word_len_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    df['short_description'] = df['short_description'].fillna(' ').map(lambda x: clean_html(x))
    
    df['word_len_min'] = df['title'].map(lambda x: min(word_len(x)))
    df['word_len_max'] = df['title'].map(lambda x: max(word_len(x)))
    df['word_len_mean'] = df['title'].map(lambda x: np.mean(word_len(x)))
    df['word_len_std'] = df['title'].map(lambda x: np.std(word_len(x)))
    
    feat_names = ['word_len_min','word_len_max', 'word_len_mean','word_len_std']
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

from sklearn.ensemble import ExtraTreesClassifier

def extract_cat_pred_feature():
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    
    #vect = CountVectorizer(analyzer='char', ngram_range=(2, 6), min_df=0.005, lowercase=True)
    vect = TfidfVectorizer( ngram_range=(1, 2), max_df=0.90, min_df=2, sublinear_tf=False )
    X = vect.fit_transform(df['title'])
    y = LabelEncoder().fit_transform(df['category_lvl_1'].values)
    #model = LogisticRegression()
    #model = MultinomialNB()
    model = ExtraTreesClassifier(n_estimators=50, random_state=888)
    model.fit(X, y)
    
    X = model.predict_proba(X)
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

def extract_desc_cat_pred_feature():
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    df['short_description'] = df['short_description'].fillna(' ').map(lambda x: clean_html(x))
    
    #vect = CountVectorizer(analyzer='char', ngram_range=(2, 6), min_df=0.005, lowercase=True)
    vect = TfidfVectorizer( ngram_range=(1, 2), max_df=0.90, min_df=2, sublinear_tf=False )
    X = vect.fit_transform(df['short_description'])
    y = LabelEncoder().fit_transform(df['category_lvl_1'].values)
    
    model = ExtraTreesClassifier(n_estimators=50, random_state=888)
    model.fit(X, y)
    
    X = model.predict_proba(X)
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

def extract_cat2_pred_feature():
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    df['short_description'] = df['short_description'].fillna(' ').map(lambda x: clean_html(x))
    
    vect = TfidfVectorizer( ngram_range=(1, 2), max_df=0.90, min_df=2, sublinear_tf=False )
    X = vect.fit_transform(df['title'])
    y = LabelEncoder().fit_transform(df['category_lvl_2'].values)
    #model = LogisticRegression()
    #model = MultinomialNB()
    model = ExtraTreesClassifier(n_estimators=50, random_state=888)
    model.fit(X, y)
    
    X = model.predict_proba(X)
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val


def get_top_words(x, top_k=100):
    tfidf = TfidfVectorizer(stop_words='english')
    response = tfidf.fit_transform(x)
    feature_array = np.array(tfidf.get_feature_names())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    top_words = feature_array[tfidf_sorting][:top_k]
    
    return top_words
    
def gini(p):
   return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))

def entropy(p):
   return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def classification_error(p):
   return 1 - np.max([p, 1 - p])

from nltk.corpus import stopwords

def count_stopwords(x, dict_sw):
    cnt = 0
    for w in x.split(' '):
        if w in dict_sw:
            cnt += 1
    return cnt

def extract_stopword_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    
    df['title'] = df['title'].map(lambda x: x.replace('&amp;', '&').replace('&nbsp;', ' '))
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    
    words = set(stopwords.words('english'))
    
    df['stopwords_count'] = df['title'].map(lambda x: count_stopwords(x, words))
    df['stopwords_ratio'] = df['stopwords_count'] / df['title_len'] 
    
    
    feat_names = ['stopwords_count','stopwords_ratio']
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val



def extract_price_features_new():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    df = df.fillna('NA')
    
    col = 'product_type_price'
    df[col] = df.groupby("product_type")["price"].transform(np.sum)
    feat_names.append(col)
    
    df["category_lvl_12"] = df['category_lvl_1'] + "-" + df['category_lvl_2']
    col = 'category_lvl_12_price'
    df[col] = df.groupby("category_lvl_12")["price"].transform(np.sum)
    feat_names.append(col)
    
    df["category_lvl_23"] = df['category_lvl_2'].map(str) + "-" + df['category_lvl_3']
    col = 'category_lvl_23_price'
    df[col] = df.groupby("category_lvl_23")["price"].transform(np.sum)
    feat_names.append(col)
    
    df["category_lvl_13"] = df['category_lvl_1'].map(str) + "-" + df['category_lvl_3']
    col = 'category_lvl_13_price'
    df[col] = df.groupby("category_lvl_13")["price"].transform(np.sum)
    feat_names.append(col)

    df["category_lvl_123"] = df['category_lvl_1'].map(str) + "-" + df['category_lvl_2'] + "-" + df['category_lvl_3']
    col = 'category_lvl_123_price'
    df[col] = df.groupby("category_lvl_123")["price"].transform(np.sum)
    feat_names.append(col)
    
    df["price"] = df[["country","price"]].apply(lambda x: x[1] if x[0]=='sg' else x[1]*0.32 if x[0]=='my' else x[1]*0.074, axis=1)
    
    col = 'category_lvl_1_price_mean'
    df[col] = df.groupby("category_lvl_1")["price"].transform(np.mean)
    feat_names.append(col)
    
    col = 'category_lvl_1_price_min'
    df[col] = df.groupby("category_lvl_1")["price"].transform(np.min)
    feat_names.append(col)
    
    col = 'category_lvl_1_price_max'
    df[col] = df.groupby("category_lvl_1")["price"].transform(np.max)
    feat_names.append(col)
    
    col = 'category_lvl_1_price_std'
    df[col] = df.groupby("category_lvl_1")["price"].transform(np.std)
    feat_names.append(col)
    
    col = 'category_lvl_2_price_mean'
    df[col] = df.groupby("category_lvl_2")["price"].transform(np.mean)
    feat_names.append(col)
    
    col = 'category_lvl_3_price_mean'
    df[col] = df.groupby("category_lvl_3")["price"].transform(np.mean)
    feat_names.append(col)
    
    ################
    col = 'category_lvl_1c_price_mean'
    df[col] = df.groupby(['country',"category_lvl_1"])["price"].transform(np.mean)
    feat_names.append(col)
    
    col = 'category_lvl_1c_price_std'
    df[col] = df.groupby(['country',"category_lvl_1"])["price"].transform(np.std)
    feat_names.append(col)    
    
    col = 'category_lvl_2c_price_mean'
    df[col] = df.groupby(['country',"category_lvl_2"])["price"].transform(np.mean)
    feat_names.append(col)
    
    col = 'category_lvl_2c_price_std'
    df[col] = df.groupby(['country',"category_lvl_2"])["price"].transform(np.std)
    feat_names.append(col)
    
    col = 'category_lvl_3c_price_mean'
    df[col] = df.groupby(['country',"category_lvl_3"])["price"].transform(np.mean)
    feat_names.append(col)
    
    col = 'category_lvl_3c_price_std'
    df[col] = df.groupby(['country',"category_lvl_3"])["price"].transform(np.std)
    feat_names.append(col)
   
    X = df[feat_names].values
    X_tr = X[:n_trains,:]
    X_val = X[n_trains:,:]
           
    return X_tr, X_val

def extract_char_set_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = df.fillna(' ')
    
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1 = df1.fillna(' ')
    
    df = pd.concat((df,df1))
    df['title'] = df['title'].map(lambda x: x.replace('&amp;', '&').replace('&nbsp;', ' '))
    
    df['short_description'] = df['short_description'].map(lambda x: clean_html(x))
    df['short_description'] = df['short_description'].map(lambda x: x.replace('&amp;', '&').replace('&nbsp;', ' '))

    df['c_set_len'] = df['title'].map(lambda x: len(list(set(x.split()))))
    
    df['c_set_len1'] = df['short_description'].map(lambda x: len(list(set(x.split()))))
   
    feat_names = ['c_set_len','c_set_len1']
    
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val

if __name__ == "__main__":
       
    X_encode_tr, X_encode_val = extract_label_encode_feature()
    
    X_encode_tr, X_encode_val = extract_clarity_encode_feature()
    X_encode_ex_tr, X_encode_ex_val = extract_clarity_encode_feature_ex()
    
    X_encode_tr1, X_encode_val1 = extract_conciseness_encode_feature()
    
    X_top_tr, X_top_val = extract_top_clarity_feature()
    
    X_g_tr, X_g_val = giba_features()

    
    X_ent_train, X_ent_val = extract_entropy_features()
    X_title_ent_train, X_title_ent_val = extract_title_entropy_features()
    X_brand_ent_train, X_brand_ent_val = extract_brand_entropy_features()
    X_c_tr, X_c_val = extract_char_feat_feature()
    
    X_cs_tr, X_cs_val = extract_char_shape_feature()

    X_csd_tr, X_csd_val = extract_char_desc_shape_feature_ex()
    
    X_cs_ex_tr, X_cs_ex_val = extract_char_shape_feature_ex()
    
    X_en_cnt_tr, X_en_cnt_val = extract_nltk_feature()
    
    X_w_cnt_tr, X_w_cnt_val = extract_word_len_feature()
    
    X_cat_pred_tr, X_cat_pred_val = extract_cat_pred_feature()
    X_cat_pred_desc_tr, X_cat_pred_desc_val = extract_desc_cat_pred_feature()

    X_cat2_pred_tr, X_cat2_pred_val = extract_cat2_pred_feature()
    
    X_sw_tr, X_sw_val = extract_stopword_feature()
    
    X_price_new_tr, X_price_new_val = extract_price_features_new()

    X_charset_tr, X_charset_val = extract_char_set_feature()
    
    joblib.dump(X_c_tr, '../features/char_set_feat.trn', protocol=2)
    joblib.dump(X_c_val, '../features/char_set_feat.tst', protocol=2)
    
    joblib.dump(X_ent_train, '../features/entropy_feat.trn', protocol=2)
    joblib.dump(X_ent_val, '../features/entropy_feat.tst', protocol=2)
    
    joblib.dump(X_c_tr, '../features/char_set_feat_ex.trn', protocol=2)
    joblib.dump(X_c_val, '../features/char_set_feat_ex.tst', protocol=2)
    
    joblib.dump(X_cs_tr, '../features/char_shape_feat.trn', protocol=2)
    joblib.dump(X_cs_val, '../features/char_shape_feat.tst', protocol=2)
    
    joblib.dump(X_price_new_tr, '../features/price_new.trn', protocol=2)
    joblib.dump(X_price_new_val, '../features/price_new.tst', protocol=2)
    
    joblib.dump(X_csd_tr, '../features/desc_char_shape_feat.trn', protocol=2)
    joblib.dump(X_csd_val, '../features/desc_char_shape_feat.tst', protocol=2)
    
    joblib.dump(X_cat_pred_tr, '../features/cat1_pred_feat.trn', protocol=2)
    joblib.dump(X_cat_pred_val, '../features/cat1_pred_feat.tst', protocol=2)
    
    joblib.dump(X_cat_pred_desc_tr, '../features/desc_cat1_pred_feat.trn', protocol=2)
    joblib.dump(X_cat_pred_desc_val, '../features/desc_cat1_pred_feat.tst', protocol=2)
    
    joblib.dump(X_cs_ex_tr, '../features/char_shape_feat_ex.trn', protocol=2)
    joblib.dump(X_cs_ex_val, '../features/char_shape_feat_ex.tst', protocol=2)
    
    joblib.dump(X_encode_tr, '../features/clarity_encode.trn', protocol=2)
    joblib.dump(X_encode_val, '../features/clarity_encode.tst', protocol=2)
  
    joblib.dump(X_top_tr, '../features/top_clarity.trn', protocol=2)
    joblib.dump(X_top_val, '../features/top_clarity.tst', protocol=2)

    