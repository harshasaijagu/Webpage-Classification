"""
data_cleaning.py
----------------
Comprehensive module for cleaning tabular data (text, numeric, categorical).
Designed for pipeline usage and professional projects.
"""

import pandas as pd
import numpy as np
import string
import unicodedata
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------------------
# 1. GENERAL DATA CLEANING
# --------------------------------------
def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates(subset=subset, ignore_index=True)


def drop_missing_rows(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Drop rows with missing values."""
    return df.dropna(subset=subset, ignore_index=True)


def fill_missing(df: pd.DataFrame, column: str, method='mean', value=None) -> pd.DataFrame:
    """
    Fill missing values in a column.
    Supports 'mean', 'median', 'mode', and constant value.
    """
    if method == 'mean':
        df[column] = df[column].fillna(df[column].mean())
    elif method == 'median':
        df[column] = df[column].fillna(df[column].median())
    elif method == 'mode':
        df[column] = df[column].fillna(df[column].mode()[0])
    elif method == 'constant':
        df[column] = df[column].fillna(value)
    return df


# --------------------------------------
# 2. TEXT DATA CLEANING
# --------------------------------------

# Its present in text_processing.py


# --------------------------------------
# 3. NUMERICAL DATA CLEANING
# --------------------------------------
def convert_to_numeric(df: pd.DataFrame, column: str, errors='coerce') -> pd.DataFrame:
    """Convert column to numeric type, invalid parsing set to NaN."""
    df[column] = pd.to_numeric(df[column], errors=errors)
    return df


def clip_outliers(df: pd.DataFrame, column: str, method='iqr') -> pd.DataFrame:
    """Clip outliers using IQR method."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower, upper)
    return df


# --------------------------------------
# 4. CATEGORICAL DATA CLEANING
# --------------------------------------
def standardize_categories(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Strip whitespace and lowercase categorical values."""
    df[column] = df[column].astype(str).str.strip().str.lower()
    return df


def handle_rare_categories(df: pd.DataFrame, column: str, threshold=5) -> pd.DataFrame:
    """Combine rare categories (below threshold) into 'other'."""
    value_counts = df[column].value_counts()
    rare = value_counts[value_counts < threshold].index
    df[column] = df[column].replace(rare, 'other')
    return df


# --------------------------------------
# 5. LOGGING AND PIPELINE UTILITIES
# --------------------------------------
def log_cleaning_step(df_before: pd.DataFrame, df_after: pd.DataFrame, step_name: str):
    """Print info about cleaning step."""
    removed_rows = df_before.shape[0] - df_after.shape[0]
    print(f"[INFO] Step: {step_name} | Removed rows: {removed_rows} | New shape: {df_after.shape}")


def apply_pipeline(df: pd.DataFrame, steps: list) -> pd.DataFrame:
    """
    Apply a sequence of cleaning functions sequentially.
    Args:
        df: input DataFrame
        steps: list of functions (lambda or normal)
    Returns:
        Cleaned DataFrame
    """
    for step in steps:
        df = step(df)
    return df