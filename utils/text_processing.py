"""
text_processing.py
------------------
Comprehensive module for text preprocessing, NLP feature extraction, and advanced text mining.
Designed for professional ML pipelines and integration with MLflow.
"""

import pandas as pd
import numpy as np
import string
import re
import unicodedata
from typing import List, Optional, Union
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Note: These libraries might require separate installation
from gensim.models import Word2Vec, FastText
from gensim.models.keyedvectors import KeyedVectors
from sentence_transformers import SentenceTransformer
# The textblob and spacy modules were in one of the provided files, but not the others.
# Uncomment the following lines if you need them.
# import spacy
# from textblob import TextBlob
# nlp = spacy.load("en_core_web_sm")


# Download necessary NLTK data
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
print("NLTK data download complete.")

# --------------------------------------
# 1. BASIC TEXT CLEANING
# --------------------------------------

def lowercase_text(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert text column to lowercase."""
    df[column] = df[column].astype(str).str.lower()
    return df

def strip_whitespace(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove leading/trailing whitespace."""
    df[column] = df[column].astype(str).str.strip()
    return df

def remove_punctuation(df: pd.DataFrame, column: str, keep: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove punctuation from text column."""
    if keep is None:
        keep = []
    pattern = f"[{re.escape(''.join(c for c in string.punctuation if c not in keep))}]"
    df[column] = df[column].apply(lambda x: re.sub(pattern, '', str(x)))
    return df

def remove_stopwords(df: pd.DataFrame, column: str, language: str = 'english') -> pd.DataFrame:
    """Remove stopwords from text column."""
    stop_words = set(stopwords.words(language))
    df[column] = df[column].apply(lambda x: ' '.join([w for w in str(x).split() if w not in stop_words]))
    return df

def lemmatize_text(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Lemmatize text column using WordNetLemmatizer."""
    lemmatizer = WordNetLemmatizer()
    df[column] = df[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in str(x).split()]))
    return df

def stem_text(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Stem text column using PorterStemmer."""
    stemmer = PorterStemmer()
    df[column] = df[column].apply(lambda x: ' '.join([stemmer.stem(w) for w in str(x).split()]))
    return df

def normalize_unicode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize unicode characters to ASCII."""
    df[column] = df[column].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii','ignore').decode('utf-8'))
    return df

def remove_html_tags(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove HTML tags from text column."""
    df[column] = df[column].apply(lambda x: re.sub(r'<.*?>', '', str(x)))
    return df

# --------------------------------------
# 2. ADVANCED NLP FEATURES
# --------------------------------------

def extract_entities(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Extract named entities using SpaCy."""
    def ner(text):
        doc = nlp(str(text))
        return [(ent.text, ent.label_) for ent in doc.ents]
    df[column + "_entities"] = df[column].apply(ner)
    return df

def extract_pos_tags(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Extract part-of-speech tags."""
    def pos_tags(text):
        doc = nlp(str(text))
        return [(token.text, token.pos_) for token in doc]
    df[column + "_pos"] = df[column].apply(pos_tags)
    return df

def extract_dependencies(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Extract syntactic dependencies."""
    def deps(text):
        doc = nlp(str(text))
        return [(token.text, token.dep_, token.head.text) for token in doc]
    df[column + "_dependencies"] = df[column].apply(deps)
    return df

def text_statistics(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute basic text statistics."""
    df[column + "_char_count"] = df[column].apply(lambda x: len(str(x)))
    df[column + "_word_count"] = df[column].apply(lambda x: len(str(x).split()))
    df[column + "_unique_words"] = df[column].apply(lambda x: len(set(str(x).split())))
    df[column + "_sentence_count"] = df[column].apply(lambda x: str(x).count('.') + 1)
    return df

def extract_sentiment(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute sentiment polarity and subjectivity."""
    df[column + "_polarity"] = df[column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df[column + "_subjectivity"] = df[column].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    return df


# --------------------------------------
# 3. TOKENIZATION & N-GRAMS
# --------------------------------------

def tokenize_text(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Tokenize text into a list of words."""
    df[column] = df[column].apply(lambda x: word_tokenize(str(x)))
    return df

def generate_ngrams(df: pd.DataFrame, column: str, n: int = 2) -> pd.DataFrame:
    """Generate n-grams from a list of tokens."""
    def ngrams(tokens):
        return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    df[column] = df[column].apply(lambda x: ngrams(x) if isinstance(x, list) else x)
    return df

# --------------------------------------
# 4. VECTORIZATION & EMBEDDINGS
# --------------------------------------

def tfidf_vectorize(corpus: List[str], max_features: int = 5000) -> (np.ndarray, TfidfVectorizer):
    """Compute TF-IDF vectors for a list of documents."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def count_vectorize(corpus: List[str], max_features: int = 5000) -> (np.ndarray, CountVectorizer):
    """Compute Count Vectors for a list of documents."""
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def train_word2vec(tokenized_corpus: List[List[str]], vector_size: int = 100, window: int = 5, min_count: int = 1) -> Word2Vec:
    """Train a Word2Vec model on a tokenized corpus."""
    model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def train_fasttext(tokenized_corpus: List[List[str]], vector_size: int = 100, window: int = 5, min_count: int = 1) -> FastText:
    """Train a FastText model on a tokenized corpus."""
    model = FastText(sentences=tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def sentence_embeddings(corpus: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Generate sentence embeddings using Sentence-BERT."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus, show_progress_bar=True)
    return embeddings

# --------------------------------------
# 5. TOPIC MODELING
# --------------------------------------

def lda_topic_modeling(count_matrix: np.ndarray, n_topics: int = 10) -> LatentDirichletAllocation:
    """Generate topic distribution features using LDA."""
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(count_matrix)
    return lda

# --------------------------------------
# 6. FEATURE EXTRACTION HELPERS
# --------------------------------------

def extract_top_keywords(df: pd.DataFrame, column: str, vectorizer=None, top_n=10) -> pd.DataFrame:
    """Extract top keywords per document using TF-IDF."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df[column].astype(str))
    else:
        X = vectorizer.transform(df[column].astype(str))
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords = []
    for row in X.toarray():
        top_indices = row.argsort()[-top_n:][::-1]
        top_keywords.append(feature_names[top_indices].tolist())
    df[column + "_keywords"] = top_keywords
    return df

def most_common_words(df: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    """Extract and count the most common words in a text column."""
    all_words = ' '.join(df[column].astype(str)).split()
    counter = Counter(all_words)
    most_common = counter.most_common(top_n)
    return pd.DataFrame(most_common, columns=['word', 'count'])

def bag_of_words(df: pd.DataFrame, column: str, max_features: int = 1000) -> (np.ndarray, CountVectorizer):
    """Generate a Bag of Words representation."""
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df[column].astype(str))
    return X, vectorizer

# --------------------------------------
# 7. PIPELINE UTILITY
# --------------------------------------

def apply_text_pipeline(df: pd.DataFrame, column: str, steps: List[callable]) -> pd.DataFrame:
    """Apply a sequence of text processing functions in order."""
    for step in steps:
        df = step(df, column)
    return df