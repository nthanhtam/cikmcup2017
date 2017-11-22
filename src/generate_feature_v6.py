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
    df['short_description'] = df['short_description'].fillna('').map(lambda x: clean_html(x))
    
    df['title'] = df['title'].map(lambda x: x.lower())
    df['category_lvl_1'] = df['category_lvl_1'].map(lambda x: x.lower())
    df['category_lvl_2'] = df['category_lvl_2'].map(lambda x: x.lower())
    df['category_lvl_3'] = df['category_lvl_3'].fillna('').map(lambda x: x.lower())
    
    df['category_lvl_1'] = df['category_lvl_1'].map(lambda x: ' '.join(x.split(', ')))
    df['category_lvl_1'] = df['category_lvl_1'].map(lambda x: ' '.join(x.split('/ ')))
    df['category_lvl_1'] = df['category_lvl_1'].map(lambda x: ' '.join(x.split('& ')))
    df['category_lvl_1'] = df['category_lvl_1'].map(lambda x: x.replace('  ',' '))

    df['category_lvl_2'] = df['category_lvl_2'].map(lambda x: ' '.join(x.split(', ')))
    df['category_lvl_2'] = df['category_lvl_2'].map(lambda x: ' '.join(x.split('/ ')))
    df['category_lvl_2'] = df['category_lvl_2'].map(lambda x: ' '.join(x.split('& ')))
    df['category_lvl_2'] = df['category_lvl_2'].map(lambda x: x.replace('  ',' '))
    
    df['category_lvl_3'] = df['category_lvl_3'].map(lambda x: ' '.join(x.split(', ')))
    df['category_lvl_3'] = df['category_lvl_3'].map(lambda x: ' '.join(x.split('/ ')))
    df['category_lvl_3'] = df['category_lvl_3'].map(lambda x: ' '.join(x.split('& ')))
    df['category_lvl_3'] = df['category_lvl_3'].map(lambda x: x.replace('  ',' '))

    df['cat1_in_title'] = df[['title','category_lvl_1']].apply(lambda x: is_found(x), axis=1)
    df['cat2_in_title'] = df[['title','category_lvl_2']].apply(lambda x: is_found(x), axis=1)
    df['cat3_in_title'] = df[['title','category_lvl_3']].apply(lambda x: is_found(x), axis=1)
    
    feat_names.extend(['cat1_in_title','cat2_in_title','cat3_in_title'])
    
    df['title_with_='] =  df['title'].map(lambda x: x.find('=') > 0).astype(int)
    df['title_with_single_quote'] =  df['title'].map(lambda x: x.find("'") > 0).astype(int)
    df['title_with_colon'] =  df['title'].map(lambda x: x.find(":") > 0).astype(int)
    df['title_with_bracket'] =  df['title'].map(lambda x: x.find("(") > 0 or x.find(")") > 0).astype(int)
    
    feat_names.extend(['title_with_=','title_with_single_quote','title_with_colon','title_with_bracket'])
    
    df['title_len'] = df['title'].map(lambda x: len(x.split(' ')))
    df['title_len_small'] = df['title_len'].map(lambda x: x < 5).astype(int)
    df['title_len_medium'] = df['title_len'].map(lambda x: x >= 5 and x < 10).astype(int)
    df['title_len_large'] = df['title_len'].map(lambda x: x >= 10 and x < 13).astype(int)
    df['title_len_xlarge'] = df['title_len'].map(lambda x: x >= 13 and x < 20).astype(int)
    df['title_len_xxlarge'] = df['title_len'].map(lambda x: x >= 20 and x < 30).astype(int)
    df['title_len_outlier'] = df['title_len'].map(lambda x: x >= 30).astype(int)
    
    #Fold 0, Train RMSE: 0.159883. Val RMSE: 0.325740. Val AUC: 0.911841
    feat_names.extend(['title_len_small','title_len_medium','title_len_large','title_len_xlarge','title_len_xxlarge','title_len_outlier'])
    
    df['title_char_len'] = df['title'].map(lambda x: len(x))
    df['title_clen_small'] = df['title_char_len'].map(lambda x: x < 20).astype(int)
    df['title_clen_medium'] = df['title_char_len'].map(lambda x: x >= 20 and x < 50).astype(int)
    df['title_clen_large'] = df['title_char_len'].map(lambda x: x >= 50 and x < 75).astype(int)
    df['title_clen_xlarge'] = df['title_char_len'].map(lambda x: x >= 75 and x < 100).astype(int)
    df['title_clen_xxlarge'] = df['title_char_len'].map(lambda x: x >= 100 and x < 150).astype(int)
    df['title_clen_outlier'] = df['title_char_len'].map(lambda x: x >= 150).astype(int)
    
    feat_names.extend(['title_clen_small','title_clen_medium','title_clen_large','title_clen_xlarge','title_clen_xxlarge','title_clen_outlier'])
    
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
    
    feat_train_file = '../features/title_cat_feat.trn'
    feat_test_file = '../features/title_cat_feat.tst'
    extract_features_xg(train_file, test_file, feat_train_file, feat_test_file)
   
    
    
    
    
   

    