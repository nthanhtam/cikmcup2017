import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.externals import joblib


def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def contains_number(s):
    regex = re.compile("\d")
    if regex.search(s):
        return 1
    return 0

def number_count(s):
    return sum(1 for c in re.findall('\d',s))

def uppercase_count(s):
    return sum(1 for c in s if c.isupper())
  
def special_char_count(s):
    special_chars = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ '''
    return sum(1 for c in s if c in special_chars)

def contain_special_char(s):
    special_chars = '''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ '''
    return sum(1 for c in s if c in special_chars) >= 1
    
def chinese_count(s):
    return len([1 for c in re.findall(u'[\u4e00-\u9fff]+', s)])

def has_storage_size(s):
    s = s.lower()
    for c in ['gb','tb',' mb']:
        if s.find(c) >= 0:
            return 1
        
    return 0

def has_storage(s):
    s = s.lower()
    for c in ['hdd','ssd',' card', 'mirco','rom','ram', 'rpm']:
        if s.find(c) >= 0:
            return 1
        
    return 0

def has_screen_size(s):
    s = s.lower()
    for c in ['"',"'",'in','']:
        if s.find(c) >= 0:
            return 1
        
    return 0

def has_screen(s):
    s = s.lower()
    for c in ['screen','display','lcd']:
        if s.find(c) >= 0:
            return 1
        
    return 0

def has_cpu(s):
    s = s.lower()
    for c in ['gb','tb',' mb']:
        if s.find(c) >= 0:
            return 1
        
    return 0

def extract_features_xg(train_file, test_file, feat_train_file, feat_test_file):
    feat_names = []
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type']) 
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    
    df = pd.concat((df,df1))
   
    df = df.fillna('NA')
    
    for c in ['category_lvl_1','category_lvl_2','category_lvl_3']:
        df[c] = df[c].map(lambda x: x.replace('&','').replace('/','').replace(',',''))
        df['len_%s' % c]  = df[c].map(lambda x: len(x.split(' ')))
        col = 'len_%s' % c
        feat_names.append(col)
        
    for i in range(df['len_category_lvl_1'].max()):
        col = 'category_lvl_1_%d' % i
        feat_names.append(col)
        df[col]  = df[c].map(lambda x: x.split(' ')[i] if len(x.split(' ')) > i else 'NA')
        df[col] = LabelEncoder().fit_transform(df[col])
        
    for i in range(df['len_category_lvl_2'].max()):
        col = 'category_lvl_2_%d' % i
        feat_names.append(col)
        df[col]  = df[c].map(lambda x: x.split(' ')[i] if len(x.split(' ')) > i else 'NA')
        df[col] = LabelEncoder().fit_transform(df[col])
    
    for i in range(df['len_category_lvl_3'].max()):
        col = 'category_lvl_3_%d' % i
        feat_names.append(col)
        df[col]  = df[c].map(lambda x: x.split(' ')[i] if len(x.split(' ')) > i else 'NA')
        df[col] = LabelEncoder().fit_transform(df[col])
    
    X = df[feat_names].astype(float).values
    
    X_train = X[:num_trains,:]
    X_val = X[num_trains:,:]
                       
    print('saving %s' % feat_train_file)
    joblib.dump(X_train, feat_train_file)
    
    print('saving %s' % feat_test_file)
    joblib.dump(X_val, feat_test_file)

def extract_price_features(train_file, test_file, feat_train_file, feat_test_file):
    feat_names = []
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type']) 
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
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
    
    X = df[feat_names].values
    X_tr = X[:n_trains,:]
    X_val = X[n_trains:,:]
           
    print('saving %s' % feat_train_file)
    joblib.dump(X_tr, feat_train_file)
    
    print('saving %s' % feat_test_file)
    joblib.dump(X_val, feat_test_file)

if __name__ == "__main__":
    # Data loading
    
    train_file = '../data/data_train.csv'
    test_file = '../data/data_valid.csv'
    
    feat_train_file = '../features/price_feat.trn'
    feat_test_file = '../features/price_feat.tst'
    extract_price_features(train_file, test_file, feat_train_file, feat_test_file)
   
    
    
    
    
   

    