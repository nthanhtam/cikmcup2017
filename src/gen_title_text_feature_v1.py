from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

import pandas as pd
from nltk.stem.snowball import EnglishStemmer

def stemmed_words(doc):
    stemmer = EnglishStemmer()
    analyzer = TfidfVectorizer().build_analyzer()
    return ' '.join((stemmer.stem(w) for w in analyzer(doc)))

def extract_title_text_features(train_file, test_file, feat_train_file, feat_test_file, 
                                analyzer='char', ngram_range=(1, 1), lowercase=True, stem=False, stop_words=None):
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    if stem:
        df['title'] = df['title'].map(lambda x: stemmed_words(x))
    
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=lowercase, stop_words=stop_words)
    vect.fit(df["title"].tolist())
    
    X_title = vect.transform(df["title"])
    X_title_tr = X_title[:n_trains,:]
    X_title_val = X_title[n_trains:,:]
    
    print('saving %s' % feat_train_file)
    joblib.dump(X_title_tr, feat_train_file)
    
    print('saving %s' % feat_test_file)
    joblib.dump(X_title_val, feat_test_file)

def extract_color_features(train_file, test_file, feat_train_file, feat_test_file,
                           analyzer='word', ngram_range=(1, 1), lowercase=True):
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    colors = [x.strip() for x in open("../data/colors.txt").readlines()]
    c = list(filter(lambda x: len(x.split()) > 1, colors))
    c = list(map(lambda x: x.replace(" ",""), c))
    colors.extend(c)
    
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, decode_error='replace')
    vect.fit(colors)
    
    X = vect.transform(df["title"].map(lambda x: x.replace('(', ' (').replace(')', ' )').lower()).values)
    
    X_ct_tr = X[:n_trains,:]
    X_ct_val = X[n_trains:,:]
    
    print('saving %s' % feat_train_file)
    joblib.dump(X_ct_tr, feat_train_file)
    
    print('saving %s' % feat_test_file)
    joblib.dump(X_ct_val, feat_test_file)
    
    
def extract_brand_features(train_file, test_file, feat_train_file, feat_test_file, 
                           analyzer='word', ngram_range=(1, 1), lowercase=True):
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
    
    print('saving %s' % feat_train_file)
    joblib.dump(X_bt_tr, feat_train_file)
    
    print('saving %s' % feat_test_file)
    joblib.dump(X_bt_val, feat_test_file)

def generate_svd_features(train_file, test_file, feat_train_file, feat_test_file, 
                                analyzer='char', ngram_range=(1, 1), lowercase=True, stem=False, stop_words=None, n_components=15):
    df = pd.read_csv(train_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv(test_file, header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    if stem:
        df['title'] = df['title'].map(lambda x: stemmed_words(x))
    
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=lowercase, stop_words=stop_words)
    vect.fit(df["title"].tolist())
    
    X_title = vect.transform(df["title"])
    
    svd = TruncatedSVD(n_components=n_components)
    X_title = svd.fit_transform(X_title)
    
    X_title_tr = X_title[:n_trains,:]
    X_title_val = X_title[n_trains:,:]
    
    print('saving %s' % feat_train_file)
    joblib.dump(csr_matrix(X_title_tr), feat_train_file)
    
    print('saving %s' % feat_test_file)
    joblib.dump(csr_matrix(X_title_val), feat_test_file)
    

if __name__ == "__main__":
    train_file = '../data/data_train.csv'
    test_file = '../data/data_valid.csv'
    
    feat_train_file = '../features/title.bow.trn'
    feat_test_file = '../features/title.bow.tst'
    extract_title_text_features(train_file, test_file, feat_train_file, feat_test_file, analyzer='word', ngram_range=(1, 1))
    
    feat_train_file = '../features/title.bow.stopword.trn'
    feat_test_file = '../features/title.bow.stopword.tst'
    extract_title_text_features(train_file, test_file, feat_train_file, feat_test_file, analyzer='word', ngram_range=(1, 1), stop_words='english')
    
    
    feat_train_file = '../features/title.bow.3grams.trn'
    feat_test_file = '../features/title.bow.3grams.tst'
    extract_title_text_features(train_file, test_file, feat_train_file, feat_test_file, analyzer='word', ngram_range=(1, 3))
    
    feat_train_file = '../features/title.bow.stem.trn'
    feat_test_file = '../features/title.bow.stem.tst'
    extract_title_text_features(train_file, test_file, feat_train_file, feat_test_file, analyzer='word', ngram_range=(1, 1), stem=True)
    
    feat_train_file = '../features/title.bow.stem.3grams.trn'
    feat_test_file = '../features/title.bow.stem.3grams.tst'
    extract_title_text_features(train_file, test_file, feat_train_file, feat_test_file, analyzer='word', ngram_range=(1, 3), stem=True)
   
    feat_train_file = '../features/title.boc.5grams.trn'
    feat_test_file = '../features/title.boc.5grams.tst'
    extract_title_text_features(train_file, test_file, feat_train_file, feat_test_file, analyzer='char', ngram_range=(2, 5))
    
    feat_train_file = '../features/title.boc.stem.5grams.trn'
    feat_test_file = '../features/title.boc.stem.5grams.tst'
    extract_title_text_features(train_file, test_file, feat_train_file, feat_test_file, analyzer='char', ngram_range=(2, 5), stem=True)
    
    feat_train_file = '../features/title.color.trn'
    feat_test_file = '../features/title.color.tst'
    extract_color_features(train_file, test_file, feat_train_file, feat_test_file)
    
    feat_train_file = '../features/title.brand.trn'
    feat_test_file = '../features/title.brand.tst'
    extract_brand_features(train_file, test_file, feat_train_file, feat_test_file)
    
    feat_train_file = '../features/title.boc.5grams.svd.trn'
    feat_test_file = '../features/title.boc.5grams.svd.tst'
    generate_svd_features(train_file, test_file, feat_train_file, feat_test_file, analyzer='char', ngram_range=(2, 5))
    
    
    