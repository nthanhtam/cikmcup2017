import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd

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
    df['short_description'] = df['short_description'].fillna('').map(lambda x: clean_html(x))
    
    df = df.fillna('')
    
    col = 'title_len'
    df[col] = df['title'].map(len)
    feat_names.append(col)
    
    col = 'title_word_len'
    df[col] = df['title'].map(lambda x: len(x.split()))
    feat_names.append(col)
    
    col = 'contains_number'
    df[col] = df['title'].map(contains_number)
    feat_names.append(col)
    
    col = 'uppercase_count'
    df[col] = df['title'].map(uppercase_count)
    feat_names.append(col)
    
    col = 'special_char_count'
    df[col] = df['title'].map(special_char_count)
    feat_names.append(col)
    
    col = 'chinese_count'
    df[col] = df['title'].map(chinese_count)
    feat_names.append(col)
    
    df["price"] = df[["country","price"]].apply(lambda x: x[1] if x[0]=='sg' else x[1]*0.32 if x[0]=='my' else x[1]*0.074, axis=1)
    feat_names.append("price") 
   
    col = 'contain_special_char'
    df[col] = df['title'].map(contain_special_char)
    feat_names.append(col)
    
    
    col = 'product_type'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)
    
    df["category_lvl_12"] = df['category_lvl_1'] + "-" + df['category_lvl_2']
    col = 'category_lvl_12'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)
    
    df["category_lvl_23"] = df['category_lvl_2'].map(str) + "-" + df['category_lvl_3']
    col = 'category_lvl_23'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)
    
    
    df["category_lvl_13"] = df['category_lvl_1'].map(str) + "-" + df['category_lvl_3']
    col = 'category_lvl_13'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)

    df["category_lvl_123"] = df['category_lvl_1'].map(str) + "-" + df['category_lvl_2'] + "-" + df['category_lvl_3']
    col = 'category_lvl_123'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)
    
    col = 'category_lvl_1'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)
    
    col = 'category_lvl_2'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)
    
    col = 'category_lvl_3'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)
    
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
    
    feat_train_file = '../features/xg_feat.trn'
    feat_test_file = '../features/xg_feat.tst'
    extract_features_xg(train_file, test_file, feat_train_file, feat_test_file)
   
    
    
    
    
   

    