from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

from sklearn.externals import joblib

def extract_features(train_file, test_file, feat_train_file, feat_test_file):
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type']) 
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    df = df.fillna('')
    df = df[['category_lvl_1','category_lvl_2','category_lvl_3','short_description','product_type']]
    for c in df.columns:
        df[c] = LabelEncoder().fit_transform(df[c])
    
    X = OneHotEncoder().fit_transform(df.values)
    
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
    
    feat_train_file = '../features/sp_feat.trn'
    feat_test_file = '../features/sp_feat.tst'
    extract_features(train_file, test_file, feat_train_file, feat_test_file)
   
    
    
    
    
   

    