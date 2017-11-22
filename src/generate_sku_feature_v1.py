import re
import pandas as pd
from sklearn.externals import joblib

def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext


def extract_sku_text_features(train_file, test_file, feat_train_file, feat_test_file, 
                                analyzer='char', ngram_range=(1, 4), lowercase=True, stem=False, stop_words=None):
    feat_names = []
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    df['sku1'] = df['sku_id'].map(lambda x: x[:5])
    df['sku2'] = df['sku_id'].map(lambda x: x[5:9])
    
    feat= pd.crosstab(index=df['category_lvl_1'], columns=df['sku1'])
    feat.columns = ['cat1_%s' % x for x in feat.columns]
    feat = feat.reset_index()
    df = df.merge(feat, how='left', on='category_lvl_1')
    feat_names.extend(feat.columns[1:].tolist())
    
    feat= pd.crosstab(index=df['category_lvl_2'], columns=df['sku1'])
    feat.columns = ['cat2_%s' % x for x in feat.columns]
    feat = feat.reset_index()
    df = df.merge(feat, how='left', on='category_lvl_2')
    feat_names.extend(feat.columns[1:].tolist())
    
    feat= pd.crosstab(index=df['category_lvl_3'], columns=df['sku1'])
    feat.columns = ['cat3_%s' % x for x in feat.columns]
    feat = feat.reset_index()
    df = df.merge(feat, how='left', on='category_lvl_3')
    feat_names.extend(feat.columns[1:].tolist())
    
    X = df[feat_names].astype(float).values
    
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
    
    feat_train_file = '../features/sku_feat.trn'
    feat_test_file = '../features/sku_feat.tst'
    extract_sku_text_features(train_file, test_file, feat_train_file, feat_test_file)
   
    
    
    
    
   

    