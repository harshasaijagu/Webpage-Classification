"""
feature_engineering.py
----------------------
Module to transform cleaned data into features suitable for ML models.
Extended with advanced features for tabular and text data.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import string
import re

# --------------------------------------
# 1. NUMERIC FEATURE ENGINEERING
# --------------------------------------
def scale_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Standardize numeric columns using z-score scaling."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def bin_numeric(df: pd.DataFrame, columns: list, n_bins=5, strategy='quantile') -> pd.DataFrame:
    """Convert numeric columns to categorical bins."""
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    df[columns] = kb.fit_transform(df[columns])
    return df

def interaction_features(df: pd.DataFrame, col_pairs: list) -> pd.DataFrame:
    """Create interaction features for numeric columns (ratios and differences)."""
    for col1, col2 in col_pairs:
        df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
        df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
        df[f"{col1}_times_{col2}"] = df[col1] * df[col2]
        df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-6)
    return df

# --------------------------------------
# 2. CATEGORICAL FEATURE ENCODING
# --------------------------------------
def encode_categorical(df: pd.DataFrame, columns: list, method='label') -> pd.DataFrame:
    """Encode categorical columns. Method can be 'label' or 'onehot'."""
    for col in columns:
        if method == 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        elif method == 'onehot':
            ohe = OneHotEncoder(sparse=False)
            encoded = ohe.fit_transform(df[[col]])
            col_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
            df = pd.concat([df.reset_index(drop=True), pd.DataFrame(encoded, columns=col_names)], axis=1)
            df.drop(columns=[col], inplace=True)
    return df

# --------------------------------------
# 3. TEXT FEATURE ENGINEERING
# --------------------------------------
def tfidf_features(df: pd.DataFrame, column: str, max_features=500, ngram_range=(1,1)) -> pd.DataFrame:
    """Generate TF-IDF features from text column, supports n-grams."""
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_tfidf = tfidf.fit_transform(df[column])
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=[f"{column}_tfidf_{i}" for i in range(X_tfidf.shape[1])])
    df = pd.concat([df.reset_index(drop=True), df_tfidf], axis=1)
    return df

def countvector_features(df: pd.DataFrame, column: str, max_features=500, ngram_range=(1,1)) -> pd.DataFrame:
    """Generate CountVectorizer features for text column."""
    cv = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_cv = cv.fit_transform(df[column])
    df_cv = pd.DataFrame(X_cv.toarray(), columns=[f"{column}_cv_{i}" for i in range(X_cv.shape[1])])
    df = pd.concat([df.reset_index(drop=True), df_cv], axis=1)
    return df

def word2vec_features(df: pd.DataFrame, column: str, vector_size=100, window=5, min_count=1) -> pd.DataFrame:
    """Generate Word2Vec embeddings averaged over each document."""
    tokenized_text = [word_tokenize(str(text)) for text in df[column]]
    w2v_model = Word2Vec(sentences=tokenized_text, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    
    def average_embedding(tokens):
        valid_tokens = [t for t in tokens if t in w2v_model.wv.key_to_index]
        if valid_tokens:
            return np.mean([w2v_model.wv[t] for t in valid_tokens], axis=0)
        else:
            return np.zeros(vector_size)
    
    embeddings = np.vstack([average_embedding(tokens) for tokens in tokenized_text])
    df_embeddings = pd.DataFrame(embeddings, columns=[f"{column}_w2v_{i}" for i in range(vector_size)])
    df = pd.concat([df.reset_index(drop=True), df_embeddings], axis=1)
    return df

def text_statistics(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Add text statistics as features: length, word count, punctuation, stopwords, average word length, sentiment."""
    stop_words = set(pd.read_csv('https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt', header=None)[0])
    
    def count_stopwords(text):
        tokens = word_tokenize(str(text))
        return sum([1 for t in tokens if t.lower() in stop_words])
    
    df[f"{column}_char_count"] = df[column].apply(lambda x: len(str(x)))
    df[f"{column}_word_count"] = df[column].apply(lambda x: len(str(x).split()))
    df[f"{column}_avg_word_len"] = df[column].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split())>0 else 0)
    df[f"{column}_punctuation_count"] = df[column].apply(lambda x: sum([1 for c in str(x) if c in string.punctuation]))
    df[f"{column}_stopword_count"] = df[column].apply(count_stopwords)
    df[f"{column}_sentiment_polarity"] = df[column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df[f"{column}_sentiment_subjectivity"] = df[column].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    return df

# --------------------------------------
# 4. DIMENSIONALITY REDUCTION (OPTIONAL)
# --------------------------------------
def reduce_dimensions(df: pd.DataFrame, columns: list, n_components=50) -> pd.DataFrame:
    """Reduce high-dimensional features using Truncated SVD (for sparse matrices)."""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(df[columns])
    reduced_df = pd.DataFrame(reduced, columns=[f"{col}_svd" for col in range(n_components)])
    df = pd.concat([df.reset_index(drop=True), reduced_df], axis=1)
    df.drop(columns=columns, inplace=True)
    return df

# --------------------------------------
# 5. PIPELINE UTILITIES
# --------------------------------------
def apply_feature_pipeline(df: pd.DataFrame, steps: list) -> pd.DataFrame:
    """
    Apply a sequence of feature engineering steps sequentially.
    Args:
        df: input DataFrame
        steps: list of functions (lambda or normal)
    Returns:
        Feature-enhanced DataFrame
    """
    for step in steps:
        df = step(df)
    return df