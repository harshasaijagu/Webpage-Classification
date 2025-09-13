"""
data_io.py
----------
Utility functions for loading and saving datasets in various formats.
Designed for pipeline-ready, professional projects.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os
from typing import Union

# --------------------------------------
# 1. DATA LOADING FUNCTIONS
# --------------------------------------

def load_file(path: str, index_col: Union[str, int, None] = None, sheet_name: Union[str, int] = 0, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load a file (CSV, Excel, or JSON) into a Pandas DataFrame based on its extension.

    Args:
        path (str): Path to the file.
        index_col (str/int, optional): Column to use as the index (for CSV/Excel).
        sheet_name (str/int, optional): Sheet to load (for Excel, default first sheet).
        encoding (str): File encoding (default: 'utf-8', for CSV only).

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found at: {path}")

    ext = path.split('.')[-1].lower()

    if ext == "csv":
        df = pd.read_csv(path, index_col=index_col, encoding=encoding)
        print(f"[INFO] CSV loaded from {path} | Shape: {df.shape}")
    
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(path, sheet_name=sheet_name, index_col=index_col)
        print(f"[INFO] Excel loaded from {path} | Shape: {df.shape}")
    
    elif ext == "json":
        df = pd.read_json(path)
        print(f"[INFO] JSON loaded from {path} | Shape: {df.shape}")
    
    else:
        raise ValueError(f"[ERROR] Unsupported file extension: {ext}. Supported: csv, xls, xlsx, json")

    return df


def load_pickle(path: str):
    """
    Load a Pickle file.
    
    Args:
        path (str): Path to the pickle file.
        
    Returns:
        object: Loaded Python object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Pickle file not found at: {path}")
    
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    
    print(f"[INFO] Pickle loaded from {path}")
    return obj


# --------------------------------------
# 2. DATA SAVING FUNCTIONS
# --------------------------------------

def save_csv(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save DataFrame to CSV.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): File path for saving CSV.
        index (bool): Whether to include DataFrame index (default: False)
    """
    df.to_csv(path, index=index)
    print(f"[INFO] DataFrame saved to CSV at {path} | Shape: {df.shape}")


def save_excel(df: pd.DataFrame, path: str, sheet_name: str = 'Sheet1', index: bool = False):
    """
    Save DataFrame to Excel.
    
    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): File path for saving Excel file.
        sheet_name (str): Excel sheet name (default: 'Sheet1').
        index (bool): Whether to include DataFrame index (default: False)
    """
    df.to_excel(path, sheet_name=sheet_name, index=index)
    print(f"[INFO] DataFrame saved to Excel at {path} | Shape: {df.shape}")


def save_pickle(obj, path: str):
    """
    Save Python object as Pickle.
    
    Args:
        obj: Python object to save.
        path (str): Path for the pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"[INFO] Object saved as Pickle at {path}")

# --------------------------------------
# 3. TRAIN TEST SPLIT FUNCTION
# --------------------------------------

def split_train_test(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
    """
    Split a DataFrame into train and test sets.
    
    Args:
        df: Input DataFrame
        target: Name of target column
        test_size: Fraction for test set
        random_state: Random seed
        stratify: Whether to stratify based on target column (useful for classification)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_col = df[target] if stratify else None
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------
# 3. HELPER FUNCTION
# --------------------------------------

def ensure_folder_exists(path: str):
    """
    Ensure the folder for a given path exists; if not, create it.
    
    Args:
        path (str): Full file path (including filename) or folder path.
    """
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[INFO] Created folder: {folder}")