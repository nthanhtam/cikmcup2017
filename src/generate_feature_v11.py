import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, coo_matrix, csr_matrix
import pandas as pd
import itertools
from bs4 import BeautifulSoup
from textstat.textstat import textstat
from nltk.stem.snowball import EnglishStemmer
from textblob import TextBlob
from scipy import sparse
from sklearn.externals import joblib

'''
country : The country where the product is marketed, with three possible values: my for Malaysia, ph for Philippines, sg for Singapore

sku_id : Unique product id, e.g., "NO037FAAA8CLZ2ANMY"

title : Product title, e.g., "RUDY Dress"

category_lvl_1 : General category that the product belongs to, e.g., "Fashion"

category_lvl_2 : Intermediate category that the product belongs to, e.g., "Women"

category_lvl_3 : Specific category that the product belongs to, e.g., "Clothing"

short_description : Short description of the product, which may contain html formatting, e.g., "<ul> <li>Short Sleeve</li> <li>3 Colours 8 Sizes</li> <li>Dress</li> </ul> "

price : Price in the local currency, e.g., "33.0".  When country is my, the price is in Malaysian Ringgit.  When country is sg, the price is in Singapore Dollar.  When country is ph, the price is in Philippine Peso.

product_type : It could have three possible values: local means the product is delivered locally, international means the product is delivered from abroad, NA means not applicable.
'''

# 0.40 - 0.399756(0.003705)
# best LB: 0.320 - 0.212

charsets = {'BasicLatin': u'[\u0020-\u007F]', #english
    'Latin1Supplement': u'[\u00A0-\u00FF]',
    'LatinExtendedA': u'[\u0100-\u017F]',
    'LatinExtendedB': u'[\u0180-\u024F]',
    'IPAExtensions': u'[\u0250-\u02AF]',
    'SpacingModifierLetters': u'[\u02B0-\u02FF]',
    'CombiningDiacriticalMarks': u'[\u0300-\u036F]',
    'GreekandCoptic': u'[\u0370-\u03FF]',
    'Cyrillic': u'[\u0400-\u04FF]',
    'CyrillicSupplementary': u'[\u0500-\u052F]',
    'Armenian': u'[\u0530-\u058F]',
    'Hebrew': u'[\u0590-\u05FF]',
    'Arabic': u'[\u0600-\u06FF]',
    'Syriac': u'[\u0700-\u074F]',
    'Thaana': u'[\u0780-\u07BF]',
    'Devanagari': u'[\u0900-\u097F]',
    'Bengali': u'[\u0980-\u09FF]',
    'Gurmukhi': u'[\u0A00-\u0A7F]',
    'Gujarati': u'[\u0A80-\u0AFF]',
    'Oriya': u'[\u0B00-\u0B7F]',
    'Tamil': u'[\u0B80-\u0BFF]',#Tamil, Badaga, and Sauashtra languages of Tamil Nadu India, Sri Lanka, Singapore, and Malaysia (https://en.wikipedia.org/wiki/Tamil_(Unicode_block)
    'Telugu': u'[\u0C00-\u0C7F]',
    'Kannada': u'[\u0C80-\u0CFF]',
    'Malayalam': u'[\u0D00-\u0D7F]',
    'Sinhala': u'[\u0D80-\u0DFF]',
    'Thai': u'[\u0E00-\u0E7F]',
    'Lao': u'[\u0E80-\u0EFF]',
    'Tibetan': u'[\u0F00-\u0FFF]',
    'Myanmar': u'[\u1000-\u109F]',
    'Georgian': u'[\u10A0-\u10FF]',
    'HangulJamo': u'[\u1100-\u11FF]',
    'Ethiopic': u'[\u1200-\u137F]',
    'Cherokee': u'[\u13A0-\u13FF]',
    'UnifiedCanadianAboriginalSyllabics': u'[\u1400-\u167F]',
    'Ogham': u'[\u1680-\u169F]',
    'Runic': u'[\u16A0-\u16FF]',
    'Tagalog': u'[\u1700-\u171F]',
    'Hanunoo': u'[\u1720-\u173F]',
    'Buhid': u'[\u1740-\u175F]',
    'Tagbanwa': u'[\u1760-\u177F]',
    'Khmer': u'[\u1780-\u17FF]',
    'Mongolian': u'[\u1800-\u18AF]',
    'Limbu': u'[\u1900-\u194F]',
    'TaiLe': u'[\u1950-\u197F]',
    'KhmerSymbols': u'[\u19E0-\u19FF]',
    'PhoneticExtensions': u'[\u1D00-\u1D7F]',
    'LatinExtendedAdditional': u'[\u1E00-\u1EFF]',
    'GreekExtended': u'[\u1F00-\u1FFF]',
    'GeneralPunctuation': u'[\u2000-\u206F]',
    'SuperscriptsandSubscripts': u'[\u2070-\u209F]',
    'CurrencySymbols': u'[\u20A0-\u20CF]',
    'CombiningDiacriticalMarksforSymbols': u'[\u20D0-\u20FF]',
    'LetterlikeSymbols': u'[\u2100-\u214F]',
    'NumberForms': u'[\u2150-\u218F]',
    'Arrows': u'[\u2190-\u21FF]',
    'MathematicalOperators': u'[\u2200-\u22FF]',
    'MiscellaneousTechnical': u'[\u2300-\u23FF]',
    'ControlPictures': u'[\u2400-\u243F]',
    'OpticalCharacterRecognition': u'[\u2440-\u245F]',
    'EnclosedAlphanumerics': u'[\u2460-\u24FF]',
    'BoxDrawing': u'[\u2500-\u257F]',
    'BlockElements': u'[\u2580-\u259F]',
    'GeometricShapes': u'[\u25A0-\u25FF]',
    'MiscellaneousSymbols': u'[\u2600-\u26FF]',
    'Dingbats': u'[\u2700-\u27BF]',
    'MiscellaneousMathematicalSymbolsA': u'[\u27C0-\u27EF]',
    'SupplementalArrowsA': u'[\u27F0-\u27FF]',
    'BraillePatterns': u'[\u2800-\u28FF]',
    'SupplementalArrowsB': u'[\u2900-\u297F]',
    'MiscellaneousMathematicalSymbolsB': u'[\u2980-\u29FF]',
    'SupplementalMathematicalOperators': u'[\u2A00-\u2AFF]',
    'MiscellaneousSymbolsandArrows': u'[\u2B00-\u2BFF]',
    'CJKRadicalsSupplement': u'[\u2E80-\u2EFF]',
    'KangxiRadicals': u'[\u2F00-\u2FDF]',
    'IdeographicDescriptionCharacters': u'[\u2FF0-\u2FFF]',
    'CJKSymbolsandPunctuation': u'[\u3000-\u303F]',
    'Hiragana': u'[\u3040-\u309F]',
    'Katakana': u'[\u30A0-\u30FF]',
    'Bopomofo': u'[\u3100-\u312F]',
    'HangulCompatibilityJamo': u'[\u3130-\u318F]',
    'Kanbun': u'[\u3190-\u319F]',
    'BopomofoExtended': u'[\u31A0-\u31BF]',
    'KatakanaPhoneticExtensions': u'[\u31F0-\u31FF]',
    'EnclosedCJKLettersandMonths': u'[\u3200-\u32FF]',
    'CJKCompatibility': u'[\u3300-\u33FF]',
    'CJKUnifiedIdeographsExtensionA': u'[\u3400-\u4DBF]',
    'YijingHexagramSymbols': u'[\u4DC0-\u4DFF]',
    'CJKUnifiedIdeographs': u'[\u4E00-\u9FFF]', #chinese
    'YiSyllables': u'[\uA000-\uA48F]',
    'YiRadicals': u'[\uA490-\uA4CF]',
    'HangulSyllables': u'[\uAC00-\uD7AF]',
    'HighSurrogates': u'[\uD800-\uDB7F]',
    'HighPrivateUseSurrogates': u'[\uDB80-\uDBFF]',
    'LowSurrogates': u'[\uDC00-\uDFFF]',
    'PrivateUseArea': u'[\uE000-\uF8FF]',
    'CJKCompatibilityIdeographs': u'[\uF900-\uFAFF]',
    'AlphabeticPresentationForms': u'[\uFB00-\uFB4F]',
    'ArabicPresentationFormsA': u'[\uFB50-\uFDFF]',
    'VariationSelectors': u'[\uFE00-\uFE0F]',
    'CombiningHalfMarks': u'[\uFE20-\uFE2F]',
    'CJKCompatibilityForms': u'[\uFE30-\uFE4F]',
    'SmallFormVariants': u'[\uFE50-\uFE6F]',
    'ArabicPresentationFormsB': u'[\uFE70-\uFEFF]',
    'HalfwidthandFullwidthForms': u'[\uFF00-\uFFEF]',
    'Specials': u'[\uFFF0-\uFFFF]'
    }


'''
BoW: Bag-of-words
BoNG: Bag-of-ngrams (up to 3)

Deep BoW/BoNG: including more than one hidden layer
ConvNets: https://arxiv.org/abs/1408.5882, start with single layer of multiple convolutional kernel size (often 2-8) and max-pooling
RNN
Hierarchical model: use CNN on each sentence and use an RNN/CNN to summarize sentenses

BPE: https://github.com/rsennrich/subword-nmt

Character-level embedding of a word: http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12489
    
'''
def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def getCharStatByLanguage(charsets):
    r = {}
    for block in charsets.keys():
        charIdx = [self.chars[c] for c in self.chars.keys() if re.findall(charsets[block], c)]
        count = self._char_count[:, charIdx].sum(axis=1).flatten().tolist()[0]
        freq = self._char_freq[:, charIdx].sum(axis=1).flatten().tolist()[0]
        r[block] = count, freq
    return r
    
def to_text():
    df = pd.read_csv("../data/training/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/validation/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df = pd.concat((df,df1))
    df["title"] = df["title"].map(lambda x: x.replace("("," ( ").replace(")"," ) "))
    df["short_description"] = df["short_description"].fillna("NA").map(lambda x: x.replace("("," ( ").replace(")"," ) "))
    df['short_description'] = df['short_description'].map(lambda x: re.sub('<[^<]+?>', '', x))
    with open('../data/text.txt',"w", encoding='utf8') as f:
        f.write(" ".join(df["title"].tolist() + df['short_description'].tolist()))

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

def extract_features_rec():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    df['clarity'] = np.loadtxt("../data/clarity_train.labels", dtype=int)

    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    df1['conciseness'] = -1
    df1['clarity'] = -1
    
    df = pd.concat((df,df1))
    
    df = df.fillna('')
    
    col = 'has_storage'
    df[col] = df['title'].map(lambda x: has_storage(x))
    feat_names.append(col)
    
    col = 'has_storage_size'
    df[col] = df['title'].map(lambda x: has_storage_size(x))
    feat_names.append(col)
    
    col = 'has_screen'
    df[col] = df['title'].map(lambda x: has_screen(x))
    feat_names.append(col)
    
    col = 'has_screen_size'
    df[col] = df['title'].map(lambda x: has_screen_size(x))
    feat_names.append(col)
    
    train = df[df['conciseness'] != -1]
    val = df[df['conciseness'] == -1]
    
    X_train = train[feat_names].values
    X_val = val[feat_names].values
    
    return X_train, X_val, feat_names
    
def extract_features_xg():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    df['clarity'] = np.loadtxt("../data/clarity_train.labels", dtype=int)

    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1['conciseness'] = -1
    df1['clarity'] = -1
       
    df = pd.concat((df,df1))
    
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
    
    train = df[df['conciseness'] != -1]
    val = df[df['conciseness'] == -1]
    
    X_train = train[feat_names].values
    X_val = val[feat_names].values
               
    
    y1 = train["conciseness"].values
    y2 = train["clarity"].values
           
    return X_train, X_val, y1, y2, feat_names


def extract_title_text_features(analyzer='char', ngram_range=(1, 1), min_df=1, lowercase=True):
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df, lowercase=lowercase)
    vect.fit(df["title"].tolist())
    
    X_title = vect.transform(df["title"])
    X_title_tr = X_title[:n_trains,:]
    X_title_val = X_title[n_trains:,:]
    return X_title_tr, X_title_val


def extract_title_count_features(analyzer='char', ngram_range=(1, 1), min_df=1, max_df=1.0, lowercase=True, stop_words=None, binary=False):
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    vect = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df, max_df=max_df, lowercase=lowercase, stop_words=stop_words, binary=binary)
    vect.fit(df["title"].tolist())
    
    X_title = vect.transform(df["title"])
    X_title_tr = X_title[:n_trains,:]
    X_title_val = X_title[n_trains:,:]
    return X_title_tr, X_title_val



def extract_desc_count_features(analyzer='char', ngram_range=(1, 1), min_df=1, max_df=1.0, lowercase=True, stop_words=None, binary=False):
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    df['short_description'] = df['short_description'].fillna('NA').map(lambda x: re.sub('<[^<]+?>', '', x))
    
    vect = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df, max_df=max_df, lowercase=lowercase, stop_words=stop_words, binary=binary)
    vect.fit(df["short_description"].tolist())
    
    X_title = vect.transform(df["short_description"])
    X_title_tr = X_title[:n_trains,:]
    X_title_val = X_title[n_trains:,:]
    return X_title_tr, X_title_val

def stemmed_words(doc):
    stemmer = EnglishStemmer()
    analyzer = TfidfVectorizer().build_analyzer()
    return (stemmer.stem(w) for w in analyzer(doc))


def extract_title_text_stem_features(ngram_range=(1, 1), lowercase=True):
    df = pd.read_csv("../data/training/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/validation/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))

    vect = TfidfVectorizer(analyzer=stemmed_words, ngram_range=ngram_range, lowercase=lowercase)
    vect.fit(df["title"].tolist())
    
    X_title = vect.transform(df["title"])
    X_title_tr = X_title[:n_trains,:]
    X_title_val = X_title[n_trains:,:]
    return X_title_tr, X_title_val

def extract_desc_text_features(analyzer='char', ngram_range=(1, 1), lowercase=True):
    df = pd.read_csv("../data/training/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/validation/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
   
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    df['short_description'] = df['short_description'].fillna('NA').map(lambda x: re.sub('<[^<]+?>', '', x))
    
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=lowercase)
    vect.fit(df["short_description"].tolist())
    
    X = vect.transform(df["short_description"])
    X_desc_tr = X[:n_trains,:]
    X_desc_val = X[n_trains:,:]
    return X_desc_tr, X_desc_val

def extract_word_features():
    df = pd.read_csv("../data/training/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/validation/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    # mean = 11 ----- max=55 ---- significant=30
    df["title_len"] = df["title"].map(lambda x: len(x.split()))
    for i in range(1,12):
        df["word_%d" % i] = df["title"].map(lambda x: x.split()[i] if i < len(x.split()) else 'NA')
        
    cols = ['title_len'] + ['word_%d' % i for i in range(1,12)]
    feats = df[['sku_id','title_len'] +['word_%d' % i for i in range(1,12)]]
    feats['title_len_cnt'] = feats.groupby('title_len')['sku_id'].transform(len)
    
    for i in range(1,12):
        new_col = 'title_len_word%d_cnt' % i
        tbl = df.groupby(['title_len','word_%d' % i])['sku_id'].count()
        tbl = tbl.reset_index()
        tbl.columns = ['title_len','word_%d' % i] + [new_col]
        feats = feats.merge(tbl, how='left', on=['title_len','word_%d' % i])
        cols.append(new_col)
    
    for i in range(2,10):
        for k, g in enumerate(itertools.combinations(['word_%d' % i for i in range(1,12)], i)):
            new_col = 'title_len_wordlen%d_cnt' % i
            g = ['word_%d' % k for k in range(1,i+1)]
            tbl = df.groupby(g)['sku_id'].count()
            tbl = tbl.reset_index()
            tbl.columns = g + [new_col]
            feats = feats.merge(tbl, how='left', on=g)
            cols.append(new_col)
            
    for col in ['word_%d' % i for i in range(1,12)]:
        feats[col] = LabelEncoder().fit_transform(feats[col].fillna('NA'))
        
    X = feats[cols].values
    
    X_train = X[:n_trains,:]
    X_val = X[n_trains:,:]
    return X_train, X_val
    
def min_word_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y.split()) for y in x]
    return min(xx)

def max_word_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y.split()) for y in x]
    return max(xx)

def sum_word_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y.split()) for y in x]
    return sum(xx)


def std_word_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y.split()) for y in x]
    return np.std(xx)

def median_word_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y.split()) for y in x]
    return np.median(xx)

def min_char_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y) for y in x]
    return min(xx)

def max_char_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y) for y in x]
    return max(xx)

def sum_char_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y) for y in x]
    return sum(xx)

def std_char_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y) for y in x]
    return np.std(xx)

def median_char_cnt(x):
    if type(x)!=list:
        return 0
    xx = [len(y) for y in x]
    return np.median(xx)

def extract_items(s):  
    if s.find('<li') < 0:
        return s
    
    soup = BeautifulSoup(s)
    items = soup.find_all('li')
    items = list(map(lambda x: x.get_text().strip(), items))
    
    return items


def has_condition(s):  
    s = s.lower()
    if s.find('condition') >= 0:
        return 1.0
    elif s.find('brand new') >= 0:
        return 1.0
    elif s.find('warranty') >= 0:
        return 1.0
    else:
        return 0.0
    
def has_size(s):  
    s = s.lower()
    if s.find('size') >= 0:
        return 1.0
    else:
        return 0.0    

def has_color(s):  
    s = s.lower()
    if s.find('color') >= 0:
        return 1.0
    else:
        return 0.0    

def has_material(s):  
    s = s.lower()
    if s.find('material') >= 0:
        return 1.0
    else:
        return 0.0
    
def has_capacity(s):  
    s = s.lower()
    if s.find('capacity') >= 0:
        return 1.0
    else:
        return 0.0
    
def has_style(s):  
    s = s.lower()
    if s.find('style') >= 0:
        return 1.0
    else:
        return 0.0
    
def extract_desc_features():
    df = pd.read_csv("../data/training/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/validation/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    df['items'] = df['short_description'].fillna('NA').map(lambda x: extract_items(x))
    df['num_items'] = df['items'].map(lambda x: len(x) if type(x)==list else 0)
    
    df['min_item_word_cnt'] = df['items'].map(lambda x: min_word_cnt(x))
    df['max_item_word_cnt'] = df['items'].map(lambda x: max_word_cnt(x))
    df['sum_item_word_cnt'] = df['items'].map(lambda x: sum_word_cnt(x))
    df['std_item_word_cnt'] = df['items'].map(lambda x: std_word_cnt(x))
    df['median_item_word_cnt'] = df['items'].map(lambda x: median_word_cnt(x))
    
    df['min_item_char_cnt'] = df['items'].map(lambda x: min_char_cnt(x))
    df['max_item_char_cnt'] = df['items'].map(lambda x: max_char_cnt(x))
    df['sum_item_char_cnt'] = df['items'].map(lambda x: sum_char_cnt(x))
    df['std_item_char_cnt'] = df['items'].map(lambda x: std_char_cnt(x))
    df['median_item_char_cnt'] = df['items'].map(lambda x: median_char_cnt(x))
    
    df['has_condition'] = df['short_description'].fillna('NA').map(has_condition)
    df['has_size'] = df['short_description'].fillna('NA').map(has_size)
    df['has_color'] = df['short_description'].fillna('NA').map(has_color)
    df['has_material'] = df['short_description'].fillna('NA').map(has_material)
    df['has_capacity'] = df['short_description'].fillna('NA').map(has_capacity)
    df['has_style'] = df['short_description'].fillna('NA').map(has_style)
    
    
    cols = ['num_items','min_item_word_cnt','max_item_word_cnt','sum_item_word_cnt','std_item_word_cnt','median_item_word_cnt']
    cols += ['min_item_char_cnt','max_item_char_cnt','sum_item_char_cnt','std_item_char_cnt','median_item_char_cnt']
    
    d = pd.read_csv("../data/attributes.csv", nrows=150)
    attrs = d["att"].values
    for a in attrs:
        df["has_%s" % a] = df['short_description'].fillna('NA').map(lambda x: x.lower().find(a) >= 0).astype(int)
        cols.append("has_%s" % a)
        
    feats = df[cols]
    
    
    X_desc = feats[cols].values
    
    X_desc_train = X_desc[:n_trains,:]
    X_desc_val = X_desc[n_trains:,:]
    return X_desc_train, X_desc_val
    
def extract_price_features():
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
    
    X = df[feat_names].values
    X_tr = X[:n_trains,:]
    X_val = X[n_trains:,:]
           
    return X_tr, X_val, feat_names


def extract_count_features():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    df = df.fillna('NA')
    
    col = 'category_lvl_12_nunique'
    t  = df.groupby("category_lvl_1")["category_lvl_2"].nunique()
    t = t.to_frame(name=col).reset_index()
    df = df.merge(t, how='left', on='category_lvl_1')
    feat_names.append(col)
    
    col = 'category_lvl_13_nunique'
    t  = df.groupby("category_lvl_1")["category_lvl_3"].nunique()
    t = t.to_frame(name=col).reset_index()
    df = df.merge(t, how='left', on='category_lvl_1')
    feat_names.append(col)
    
    col = 'category_lvl_21_nunique'
    t  = df.groupby("category_lvl_2")["category_lvl_1"].nunique()
    t = t.to_frame(name=col).reset_index()
    df = df.merge(t, how='left', on='category_lvl_2')
    feat_names.append(col)
    
    col = 'category_lvl_23_nunique'
    t  = df.groupby("category_lvl_2")["category_lvl_3"].nunique()
    t = t.to_frame(name=col).reset_index()
    df = df.merge(t, how='left', on='category_lvl_2')
    feat_names.append(col)
    
    col = 'category_lvl_31_nunique'
    t  = df.groupby("category_lvl_3")["category_lvl_1"].nunique()
    t = t.to_frame(name=col).reset_index()
    df = df.merge(t, how='left', on='category_lvl_3')
    feat_names.append(col)
    
    col = 'category_lvl_32_nunique'
    t  = df.groupby("category_lvl_3")["category_lvl_2"].nunique()
    t = t.to_frame(name=col).reset_index()
    df = df.merge(t, how='left', on='category_lvl_3')
    feat_names.append(col)
    
    X = df[feat_names].values
    X_tr = X[:n_trains,:]
    X_val = X[n_trains:,:]
    
    return X_tr, X_val, feat_names

def extract_color_features(analyzer='word', ngram_range=(1, 1), lowercase=True):
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    colors = [x.strip() for x in open("../data/colors.txt").readlines()]
    c = list(filter(lambda x: len(x.split()) > 1, colors))
    c = list(map(lambda x: x.replace(" ",""), c))
    colors.extend(c)
    
    
    #vect = CountVectorizer()
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
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
       
    #vect = CountVectorizer()
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    vect.fit(brands)
    
    X = vect.transform(df["title"].map(lambda x: x.replace('(', ' (').replace(')', ' )').lower()).values)
    
    X_bt_tr = X[:n_trains,:]
    X_bt_val = X[n_trains:,:]
    
    X = vect.transform(df["short_description"].fillna('NA').map(lambda x: x.replace('(', ' (').replace(')', ' )').lower()).values)
    
    X_bd_tr = X[:n_trains,:]
    X_bd_val = X[n_trains:,:]
    
    return X_bt_tr, X_bt_val, X_bd_tr, X_bd_val
    
def extract_att_features(analyzer='word', ngram_range=(1, 1), lowercase=True):
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    brands = [x.strip() for x in open("../data/attributes_450.txt").readlines()]
       
    #vect = CountVectorizer()
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    vect.fit(brands)
    
    X = vect.transform(df["title"].map(lambda x: x.replace('(', ' (').replace(')', ' )').lower()).values)
    
    X_bt_tr = X[:n_trains,:]
    X_bt_val = X[n_trains:,:]
    
    X = vect.transform(df["short_description"].fillna('NA').map(lambda x: x.replace('(', ' (').replace(')', ' )').lower()).values)
    
    X_bd_tr = X[:n_trains,:]
    X_bd_val = X[n_trains:,:]
    
    return X_bt_tr, X_bt_val, X_bd_tr, X_bd_val


def extract_html_features():
    feat_names = []
    df = pd.read_csv("../data/training/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/validation/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    
    df = df.fillna('NA')
    df["start_with_ul"] = df["short_description"].map(lambda x: x.strip().startswith("<ul")).astype(int)
    df["start_with_div"] = df["short_description"].map(lambda x: x.strip().startswith("<div")).astype(int)
    df["start_with_p"] = df["short_description"].map(lambda x: x.strip().startswith("<p")).astype(int)
    
    df["end_with_ul"] = df["short_description"].map(lambda x: x.strip().endswith("ul>")).astype(int)
    df["end_with_div"] = df["short_description"].map(lambda x: x.strip().endswith("div>")).astype(int)
    df["end_with_p"] = df["short_description"].map(lambda x: x.strip().endswith("p>")).astype(int)


    feat_names = ['start_with_ul','start_with_div','start_with_p','end_with_ul','end_with_div','end_with_p']
    
    X = df[feat_names].values
    X_h_tr = X[:n_trains,:]
    X_h_val = X[n_trains:,:]
    
    return X_h_tr, X_h_val, feat_names

def process_item(df):
    items = []
    for i in df['items']:
        if type(i)==list:
            items.extend(i)
        else:
            items.append(i)
    # 250921        
    items = list(map(lambda x: x.replace(' : ', ': '), items))
    
    # 76278
    items = list(filter(lambda x: x.find(':') > 0, items))
    att_names = list(map(lambda x: x.split(':')[0].lower(), items))
    
    from collections import Counter

    cnt = Counter(att_names)
    
    d = []
    for k in cnt.keys():
        d.append({"att": k, "count": cnt[k]})
    
    d = pd.DataFrame(d)
    d = d[d["count"] > 1]
    d = d.sort_values(by="count", ascending=False)
    d.to_csv("../data/attributes.csv", index=False)
    

def extract_textstat_features():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    df = pd.concat((df,df1))
    df = df.fillna('NA')
    
    df["flesch_reading_ease"] = df["title"].map(textstat.flesch_reading_ease)
    df["smog_index"] = df["title"].map(textstat.smog_index)
    df["flesch_kincaid_grade"] = df["title"].map(textstat.flesch_kincaid_grade)
    df["automated_readability_index"] = df["title"].map(textstat.automated_readability_index)
    df["dale_chall_readability_score"] = df["title"].map(textstat.dale_chall_readability_score)
    df["difficult_words"] = df["title"].map(textstat.difficult_words)
    df["linsear_write_formula"] = df["title"].map(textstat.linsear_write_formula)
    df["gunning_fog"] = df["title"].map(textstat.gunning_fog)
    df["text_standard"] = df["title"].map(textstat.text_standard)
    
    df["text_standard"] = LabelEncoder().fit_transform(df["text_standard"])
    
    feat_names = ['flesch_reading_ease','smog_index','flesch_kincaid_grade','automated_readability_index','dale_chall_readability_score',
                  'difficult_words','linsear_write_formula','gunning_fog','text_standard']
    
    X = df[feat_names].values
    X_s_tr = X[:n_trains,:]
    X_s_val = X[n_trains:,:]
    
    return X_s_tr, X_s_val, feat_names
    

def np_ratio(s):
    testimonial = TextBlob(s)
    n_cnt = 0
    for k in testimonial.np_counts.keys():
        n_cnt += testimonial.np_counts[k]
    
    n_cnt /= float(len(testimonial.pos_tags))
    return n_cnt

def jj_ratio(s):
    testimonial = TextBlob(s)
    n_cnt = 0
    for w, p in testimonial.pos_tags:
        if p=='JJ':
            n_cnt+=1
    n_cnt /= float(len(testimonial.pos_tags))
    return n_cnt

def extract_pos_features():
    df = pd.read_csv("../data/training/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/validation/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    n_trains = df.shape[0]
    df = pd.concat((df,df1))
    df = df.fillna('NA')
    
    df['np_ratio'] = df['title'].map(lambda x: np_ratio(x))
    df['jj_ratio'] = df['title'].map(lambda x: jj_ratio(x))
    
    feat_names = ['np_ratio','jj_ratio']
    
    X = df[feat_names].values
    X_sent_tr = X[:n_trains,:]
    X_sent_val = X[n_trains:,:]
    
    return X_sent_tr, X_sent_val, feat_names

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def is_found(x):
    for s in x[1].split():
        if x[0].find(s) > 0:
            return 1
    return 0

def extract_desc_count():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    num_trains = df.shape[0]
    
    df = pd.concat((df,df1))
    df['short_description'] = df['short_description'].fillna('').map(lambda x: clean_html(x))
    
    df['short_description'] = df['short_description'].map(lambda x: x.lower())
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

    df['cat1_in_title'] = df[['short_description','category_lvl_1']].apply(lambda x: is_found(x), axis=1)
    df['cat2_in_title'] = df[['short_description','category_lvl_2']].apply(lambda x: is_found(x), axis=1)
    df['cat3_in_title'] = df[['short_description','category_lvl_3']].apply(lambda x: is_found(x), axis=1)
    
    feat_names.extend(['cat1_in_title','cat2_in_title','cat3_in_title'])
    
    df['title_with_='] =  df['short_description'].map(lambda x: x.find('=') > 0).astype(int)
    df['title_with_single_quote'] =  df['short_description'].map(lambda x: x.find("'") > 0).astype(int)
    df['title_with_colon'] =  df['short_description'].map(lambda x: x.find(":") > 0).astype(int)
    df['title_with_bracket'] =  df['short_description'].map(lambda x: x.find("(") > 0 or x.find(")") > 0).astype(int)
    
    feat_names.extend(['title_with_=','title_with_single_quote','title_with_colon','title_with_bracket'])
    
    df['title_len'] = df['short_description'].map(lambda x: len(x.split(' ')))
    df['title_len_small'] = df['title_len'].map(lambda x: x < 5).astype(int)
    df['title_len_medium'] = df['title_len'].map(lambda x: x >= 5 and x < 10).astype(int)
    df['title_len_large'] = df['title_len'].map(lambda x: x >= 10 and x < 13).astype(int)
    df['title_len_xlarge'] = df['title_len'].map(lambda x: x >= 13 and x < 20).astype(int)
    df['title_len_xxlarge'] = df['title_len'].map(lambda x: x >= 20 and x < 30).astype(int)
    df['title_len_outlier'] = df['title_len'].map(lambda x: x >= 30).astype(int)
    
    #Fold 0, Train RMSE: 0.159883. Val RMSE: 0.325740. Val AUC: 0.911841
    feat_names.extend(['title_len_small','title_len_medium','title_len_large','title_len_xlarge','title_len_xxlarge','title_len_outlier'])
    
    df['title_char_len'] = df['short_description'].map(lambda x: len(x))
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
      
    return X_train, X_val


def extract_clean_item_ratio(s):  
    if s.find('<li') < 0:
        return 0
    
    soup = BeautifulSoup(s)
    items = soup.find_all('li')
    items = list(map(lambda x: x.get_text().strip(), items))
    clean_items = list(filter(lambda x: x.find(":") > 0, items))
    
    return float(len(clean_items)) / len(items)

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
    
    #feat_names = ['upper_cnt','lower_cnt','startswith_alpha','startswith_space', 'space_cnt', 'space_ratio']
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

def extract_features_new():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df['conciseness'] = np.loadtxt("../data/conciseness_train.labels", dtype=int)
    df['clarity'] = np.loadtxt("../data/clarity_train.labels", dtype=int)

    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    df1['conciseness'] = -1
    df1['clarity'] = -1
       
    df = pd.concat((df,df1))
    
    df = df.fillna('')
    
    df['cat2'] = df['category_lvl_2']
    df['cat3'] = df['category_lvl_3']
      
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
    
   
    cat3 = df['cat3'].value_counts()
    cat3 = cat3[cat3<70]
    cat3 = cat3.index
    
    df['cat3'] = df['cat3'].map(lambda x: x if not x in cat3 else 'Others')
    
    col = 'cat2'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)
    
    col = 'cat3'
    df[col] = LabelEncoder().fit_transform(df[col])
    feat_names.append(col)
    
    train = df[df['conciseness'] != -1]
    val = df[df['conciseness'] == -1]
    
    X_train = train[feat_names].values.astype(float)
    X_val = val[feat_names].values.astype(float)
               
    
    return X_train, X_val

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


def extract_flag_feature():
    feat_names = []
    df = pd.read_csv("../data/data_train.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])
    
    df['flag'] = df['country'].map(lambda x: 1 if x=='my' else 0)
        
    df['flag1'] = df['flag'].cumsum()
    
    df['flag1'] = df.groupby('flag1')['flag1'].tail(1)
    
        
    num_trains = df.shape[0]
    
    df1 = pd.read_csv("../data/data_valid.csv", header=None, names=['country','sku_id','title','category_lvl_1','category_lvl_2','category_lvl_3','short_description','price','product_type'])  
    
    df1['flag'] = df1['country'].map(lambda x: 1 if x=='my' else 0)
    
    
    df1['flag1'] = df1['flag'].cumsum()
    
        
        
    df = pd.concat((df,df1))
    
    feat_names = ['flag1']
    
    
    X = df[feat_names].values
    
    X_top_train = X[:num_trains,:]
    X_top_val = X[num_trains:,:]
     
    return X_top_train, X_top_val



if __name__ == "__main__":
    # Data loading
    
    X1, X2, y1, y2, feat_names = extract_features_xg()
    X_ct_tr, X_ct_val, X_cd_tr, X_cd_val = extract_color_features()
    X_bt_tr, X_bt_val, X_bd_tr, X_bd_val = extract_brand_features()
    X_price_tr, X_price_val, _ = extract_price_features()

    X_title_tr, X_title_val = extract_title_count_features(analyzer='char', ngram_range=(2, 6), min_df=0.005, lowercase=True)
    
    X_top_tr, X_top_val = extract_top_clarity_feature()
    
    X_g_tr, X_g_val = giba_features()

   
    X_ent_train, X_ent_val = extract_entropy_features()
    X_c_tr, X_c_val = extract_char_feat_feature()
    
    X_cs_tr, X_cs_val = extract_char_shape_feature()

    X_csd_tr, X_csd_val = extract_char_desc_shape_feature_ex()
    
    X_cs_ex_tr, X_cs_ex_val = extract_char_shape_feature_ex()
    
    
 
    X_train = csr_matrix(hstack((coo_matrix(X_title_tr), 
                                 coo_matrix(X_ct_tr), coo_matrix(X_bt_tr),
                                 coo_matrix(X_cd_tr), coo_matrix(X_bd_tr), 
                                 X_price_tr,
                                 X_ent_train,
                                 X_top_tr,
                                 X_g_tr,
                                 X_c_tr,
                                 X_cs_tr,
                                 X_cs_ex_tr,
                                 X_csd_tr,
                                 X1.astype(float),
                                 )))
    
    

    X_val = csr_matrix(hstack((coo_matrix(X_title_val), 
                                 coo_matrix(X_ct_val), coo_matrix(X_bt_val),
                                 coo_matrix(X_cd_val), coo_matrix(X_bd_val), 
                                 X_price_val,
                                 X_ent_val,
                                 X_top_val,
                                 X_g_val,
                                 X_c_val,
                                 X_cs_val,
                                 X_cs_ex_val,
                                 X_csd_tr,                                 
                                 X2.astype(float))))
    
    joblib.dump([X_train, X_val], '../features/all.dmp', protocol=2)
    
    
    
    
    
   

    