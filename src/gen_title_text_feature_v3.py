from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib

import pandas as pd

import numpy as np
from bs4 import BeautifulSoup


def extract_clean_item_ratio(s):  
    if s.find('<li') < 0:
        return 0
    
    soup = BeautifulSoup(s)
    items = soup.find_all('li')
    items = list(map(lambda x: x.get_text().strip(), items))
    clean_items = list(filter(lambda x: x.find(":") > 0, items))
    
    return float(len(clean_items)+0.1) / (len(items)+0.1)

def item_count(s):  
    if s.find('<li') < 0:
        return 0
    
    soup = BeautifulSoup(s)
    items = soup.find_all('li')
    
    return len(items)

def extract_item_len_stats(s):  
    if s.find('<li') < 0:
        return ','.join(['0']*7)
    
    soup = BeautifulSoup(s)
    items = soup.find_all('li')
    items = list(map(lambda x: x.get_text().strip(), items))
    item_lens = np.asarray(list(map(lambda x: len(x.split(' ')), items)))
    
    count_ = np.count_nonzero(item_lens)
    sum_ = np.sum(item_lens)
    mean_ = np.mean(item_lens)
    median_ = np.median(item_lens)
    min_ = np.min(item_lens)
    max_ = np.max(item_lens)
    std_ = np.std(item_lens)
    
    res = [count_, sum_, mean_, median_, min_, max_, std_]
    res = ','.join(list(map(lambda x: str(x), res)))
    return res


def extract_title_count_features(analyzer='char', ngram_range=(1, 1), min_df=1, max_df=1.0, lowercase=True, stop_words=None, binary=False):
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    vect = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df, max_df=max_df, lowercase=lowercase, stop_words=stop_words, binary=binary, decode_error='replace')
    vect.fit(df["title"].tolist())
    
    X_title = vect.transform(df["title"])
    X_title_tr = X_title[:n_trains,:]
    X_title_val = X_title[n_trains:,:]
    return X_title_tr, X_title_val

def extract_desc_item_count():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    df['short_description'] = df['short_description'].fillna('NA')
    
    df['items_cnt'] = df['short_description'].map(lambda x: item_count(x))
    df['items_ratio'] = df['short_description'].map(lambda x: extract_clean_item_ratio(x))
    df['items_len_stats'] = df['short_description'].map(lambda x: extract_item_len_stats(x))
    
    df['item_len_sum'] = df['items_len_stats'].map(lambda x: float(x.split(',')[1]))
    df['item_len_mean'] = df['items_len_stats'].map(lambda x: float(x.split(',')[2]))
    df['item_len_median'] = df['items_len_stats'].map(lambda x: float(x.split(',')[3]))
    df['item_len_min'] = df['items_len_stats'].map(lambda x: float(x.split(',')[4]))
    df['item_len_max'] = df['items_len_stats'].map(lambda x: float(x.split(',')[5]))
    df['item_len_std'] = df['items_len_stats'].map(lambda x: float(x.split(',')[6]))
    
    feat_names.extend(['items_cnt','items_ratio','item_len_sum','item_len_mean','item_len_median','item_len_min','item_len_max','item_len_std'])
    
    X = df[feat_names].astype(float).values
    
    X_item_train = X[:num_trains,:]
    X_item_val = X[num_trains:,:]
      
    return X_item_train, X_item_val


def extract_color_features(analyzer='word', ngram_range=(1, 1), lowercase=True):
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    colors = [x.strip() for x in open("../data/colors.txt").readlines()]
    
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, decode_error='replace')
    vect.fit(colors)
    
    X = vect.transform(df["title"].map(lambda x: x.replace('(', ' (').replace(')', ' )').lower()).values)
    
    X_ct_tr = X[:n_trains,:]
    X_ct_val = X[n_trains:,:]
    
    X = vect.transform(df["short_description"].fillna('NA').map(lambda x: x.replace('(', ' (').replace(')', ' )').lower()).values)
            
    X_cd_tr = X[:n_trains,:]
    X_cd_val = X[n_trains:,:]
    
    return X_ct_tr, X_ct_val, X_cd_tr, X_cd_val
    

def extract_brand_features(analyzer='word', ngram_range=(1, 1), lowercase=True):
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    brands = [x.strip() for x in open("../data/brands_from_lazada_portal.txt").readlines()]
       
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, decode_error='replace')
    vect.fit(brands)
    
    X = vect.transform(df["title"].map(lambda x: x.replace('(', ' (').replace(')', ' )').lower()).values)
    
    X_bt_tr = X[:n_trains,:]
    X_bt_val = X[n_trains:,:]
    
    X = vect.transform(df["short_description"].fillna('NA').map(lambda x: x.replace('(', ' (').replace(')', ' )').lower()).values)
    
    X_bd_tr = X[:n_trains,:]
    X_bd_val = X[n_trains:,:]
    
    return X_bt_tr, X_bt_val, X_bd_tr, X_bd_val
    

if __name__ == "__main__":
    train_file = '../data/data_train.csv'
    test_file = '../data/data_valid.csv'
    
    feat_train_file = '../features/title.boc.6grams_v2.trn'
    feat_test_file = '../features/title.boc.6grams_v2.tst'
    
    X_title_tr, X_title_val = extract_title_count_features(analyzer='char', ngram_range=(2, 6), min_df=0.005, lowercase=True)
    joblib.dump(X_title_tr, feat_train_file)
    joblib.dump(X_title_val, feat_test_file)
    
    X_item_train, X_item_val = extract_desc_item_count()
    
    joblib.dump(X_item_train, '../features/item_cnt.trn', protocol=2)
    joblib.dump(X_item_val, '../features/item_cnt.tst', protocol=2)

    
    X_ct_tr, X_ct_val, X_cd_tr, X_cd_val = extract_color_features()
    X_bt_tr, X_bt_val, X_bd_tr, X_bd_val = extract_brand_features()
    

    joblib.dump(X_cd_tr, '../features/desc.color.trn', protocol=2)
    joblib.dump(X_cd_val, '../features/desc.color.tst', protocol=2)
    
    joblib.dump(X_bd_tr, '../features/desc.brand.trn', protocol=2)
    joblib.dump(X_bd_val, '../features/desc.brand.tst', protocol=2)
    