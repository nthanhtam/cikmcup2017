import re
import pandas as pd
import numpy as np
from sklearn.externals import joblib

'''
cat brand count
cat word count
cat unique word count
cat sku count
cat attribute count
cat title len statistics
cat prefix count
cat suffix count
cat title detect count
cat title cannot detect count

cat char count
cat char unique count
cat char 

cat title word len statistics
cat word len statistics

subgroup count
subsubgroup count


'''


def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def is_found(x):
    for s in x[1].split():
        if x[0].find(s) > 0:
            return 1
    return 0
        
def extract_features_xg(train_file, test_file, feat_train_file, feat_test_file):
    feat_names = []
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type']) 
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    df['title_clen'] = df['title'].map(lambda x: len(x))
    
    bins = list(range(0,df['title_len'].max()+1, 5))
    group_names = list(range(len(bins)-1))
    df['title_bin'] = pd.cut(df['title_len'], bins, labels=group_names)
    

    bins = list(range(0,df['title_clen'].max()+1, 10))
    group_names = list(range(len(bins)-1))
    df['title_cbin'] = pd.cut(df['title_clen'], bins, labels=group_names)

    feat_names.extend(['title_bin','title_cbin'])
    
    X = df[feat_names].values.astype(float)
    
    X_train = X[:num_trains,:]
    X_val = X[num_trains:,:]
                       
    print('saving %s' % feat_train_file)
    joblib.dump(X_train, feat_train_file)
    
    print('saving %s' % feat_test_file)
    joblib.dump(X_val, feat_test_file)

if __name__ == "__main__":
    # Data loading
    
    train_file = '../data/data_train.csv'
    test_file = '../data/data_valid.csv'
    
    feat_train_file = '../features/title_len_hist.trn'
    feat_test_file = '../features/title_len_hist.tst'
    extract_features_xg(train_file, test_file, feat_train_file, feat_test_file)
   
    
    
    
    
   

    