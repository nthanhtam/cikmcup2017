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

def extract_features_xg(train_file, test_file, feat_train_file, feat_test_file):
    feat_names = []
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type']) 
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    df['short_description'] = df['short_description'].fillna('').map(lambda x: clean_html(x))
    
    df['short_description'] = df['short_description'].map(lambda x: x.replace('(',' ) '))
    df['short_description'] = df['short_description'].map(lambda x: x.replace('/',' / '))
    df['short_description'] = df['short_description'].map(lambda x: x.replace('  ',' '))
    df['short_description'] = df['short_description'].map(lambda x: x.replace('\t',' '))
    
    feat = df.groupby('category_lvl_1')['short_description'].apply(lambda x: ' '.join(x))
    feat = feat.reset_index()
    
    feat['cat1_word_cnt'] = feat['short_description'].map(lambda x: len(x.split(' ')))
    feat['cat1_word_nunique'] = feat['short_description'].map(lambda x: len(set(x.split(' '))))
    
    feat1 = df.groupby('category_lvl_2')['short_description'].apply(lambda x: ' '.join(x))
    feat1 = feat1.reset_index()
    
    feat1['cat2_word_cnt'] = feat1['short_description'].map(lambda x: len(x.split(' ')))
    feat1['cat2_word_nunique'] = feat1['short_description'].map(lambda x: len(set(x.split(' '))))
    
    feat3 = df.groupby('category_lvl_3')['short_description'].apply(lambda x: ' '.join(x))
    feat3 = feat3.reset_index()
    
    feat3['cat3_word_cnt'] = feat3['short_description'].map(lambda x: len(x.split(' ')))
    feat3['cat3_word_nunique'] = feat3['short_description'].map(lambda x: len(set(x.split(' '))))
    
    df = df.merge(feat[['category_lvl_1', 'cat1_word_cnt', 'cat1_word_nunique']], how='left', on='category_lvl_1')
    df = df.merge(feat1[['category_lvl_2', 'cat2_word_cnt', 'cat2_word_nunique']], how='left', on='category_lvl_2')
    df = df.merge(feat3[['category_lvl_3', 'cat3_word_cnt', 'cat3_word_nunique']], how='left', on='category_lvl_3')
    feat_names.extend(['cat1_word_cnt', 'cat1_word_nunique','cat2_word_cnt', 'cat2_word_nunique', 'cat3_word_cnt', 'cat3_word_nunique'])
    
    df['title_len'] = df['short_description'].map(lambda x: len(x))
    df['title_word_cnt'] = df['short_description'].map(lambda x: len(x.split(' ')))
    #df['title_word_nunique'] = df['short_description'].map(lambda x: len(set(x.split(' '))))
    
    feat = df.groupby('category_lvl_1')['title_word_cnt'].agg([np.min, np.max, np.mean, np.median, np.std])
    feat.columns = ['cat1_word_cnt_min','cat1_word_cnt_max','cat1_word_cnt_mean','cat1_word_cnt_median','cat1_word_cnt_std']
    feat = feat.reset_index()
    df = df.merge(feat, how='left', on='category_lvl_1')
    feat_names.extend(['cat1_word_cnt_min','cat1_word_cnt_max','cat1_word_cnt_mean','cat1_word_cnt_median','cat1_word_cnt_std'])
    
    feat = df.groupby('category_lvl_2')['title_word_cnt'].agg([np.min, np.max, np.mean, np.median, np.std])
    feat.columns = ['cat2_word_cnt_min','cat2_word_cnt_max','cat2_word_cnt_mean','cat2_word_cnt_median','cat2_word_cnt_std']
    feat = feat.reset_index()
    df = df.merge(feat, how='left', on='category_lvl_2')
    feat_names.extend(['cat2_word_cnt_min','cat2_word_cnt_max','cat2_word_cnt_mean','cat2_word_cnt_median','cat2_word_cnt_std'])
    
    feat = df.groupby('category_lvl_3')['title_word_cnt'].agg([np.min, np.max, np.mean, np.median, np.std])
    feat.columns = ['cat3_word_cnt_min','cat3_word_cnt_max','cat3_word_cnt_mean','cat3_word_cnt_median','cat3_word_cnt_std']
    feat = feat.reset_index()
    df = df.merge(feat, how='left', on='category_lvl_3')
    feat_names.extend(['cat3_word_cnt_min','cat3_word_cnt_max','cat3_word_cnt_mean','cat3_word_cnt_median','cat3_word_cnt_std'])
    
    
    X = df[feat_names].astype(float).values
    
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
    
    feat_train_file = '../features/cat_desc_cnt_feat.trn'
    feat_test_file = '../features/cat_desc_cnt_feat.tst'
    extract_features_xg(train_file, test_file, feat_train_file, feat_test_file)
   
    
    
    
    
   

    