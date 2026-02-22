import pandas as pd
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer;
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

MAX_FEATURES = 10000
MAX_VOCABULARY = 10000
MAX_LENGTH = 200

def load_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "Phishing_Email.csv")
    df = pd.read_csv(file_path)
    return df


def explore_dataset(df):
    print("\nHead")
    print(df.head())
    print("\nInfo")
    print(df.info())
    print("\nLabel distribution")
    print(df['Email Type'].value_counts())
    print("\nMissing values")
    print(df.isnull().sum())


def initial_cleaning(df):
    df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna(subset=['Email Text'])
    df = df.reset_index(drop=True)
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)          
    text = re.sub(r'\S+@\S+', ' ', text)                 
    text = re.sub(r'[^a-z\s]', ' ', text)               
    text = re.sub(r'\s+', ' ', text).strip()            
    return text

def preprocess_texts(df):
    df = df.copy()
    df['clean_text'] = df['Email Text'].apply(clean_text)
    df['label'] = (df['Email Type'] == 'Phishing Email').astype(int)
    return df

def encode_labels(df):
    return df['label'].values

def split_data(df, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
    return train.reset_index(drop=True), test.reset_index(drop=True)

def build_tfidf(train_texts, test_texts, max_features = MAX_FEATURES, save_path=None):
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    if save_path:
       os.makedirs(os.path.dirname(save_path), exist_ok=True)
       with open(save_path, 'wb') as f:
         pickle.dump(vectorizer, f)

    return X_train, X_test, vectorizer

def load_vectorizer(path):
    with open (path, 'rb') as f:
        return pickle.load(f)
    
def build_lstm(train_texts, max_vocab=MAX_VOCABULARY, save_path=None):
    tokenizer = Tokenizer(num_words=max_vocab, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)

    if save_path:
       os.makedirs(os.path.dirname(save_path), exist_ok=True)
       with open(save_path, 'wb') as f:
         pickle.dump(tokenizer, f)
    
    return tokenizer

def padding(texts, tokenizer, max_len=MAX_LENGTH):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, max_len, padding='post', truncating='post')
    return padded 

def load_tokenizer(path):
    with open (path, 'rb') as f:
        return pickle.load(f)