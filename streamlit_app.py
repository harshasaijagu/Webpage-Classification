# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scipy.sparse import hstack, csr_matrix
import yaml

# ------------------------
# Load Config
# ------------------------
def load_config(config_path="configs/training_config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
use_text_stats = "Text Stats" in config["preprocessing"].get("features", [])

# ------------------------
# Load Artifacts
# ------------------------
@st.cache_data(show_spinner=True)
def load_artifacts(model_path="artifacts/model.joblib", transformers_dir="artifacts/transformers/"):
    """
    Load trained model and all transformers from disk.
    """
    # Load model
    if not Path(model_path).exists():
        st.error(f"Model file not found at {model_path}")
        return None, None
    model = joblib.load(model_path)
    
    # Load transformers
    transformers_dict = {}
    transformers_path = Path(transformers_dir)
    if not transformers_path.exists() or not any(transformers_path.iterdir()):
        st.warning(f"No transformers found in {transformers_dir}")
    else:
        for file in transformers_path.glob("*.joblib"):
            name = file.stem
            transformers_dict[name] = joblib.load(file)
    
    return model, transformers_dict

# ------------------------
# Preprocessing & Feature Engineering
# ------------------------
def preprocess_input(text: str, transformers_dict: dict, use_text_stats: bool = False):
    """
    Apply the saved transformers to a single text input.
    """
    from utils.text_processing import (
        remove_html_tags, normalize_unicode, strip_whitespace, lowercase_text,
        remove_punctuation, remove_stopwords, lemmatize_text,
        text_statistics, sentence_embeddings
    )

    # Wrap user input into DataFrame
    df = pd.DataFrame([text], columns=["text"])

    # 1. Text Cleaning
    cleaning_steps = [
        remove_html_tags, normalize_unicode, strip_whitespace,
        lowercase_text, remove_punctuation, remove_stopwords, lemmatize_text
    ]
    for func in cleaning_steps:
        df = func(df, "text")

    # 2. Feature Engineering
    feature_list = []

    for name, transformer in transformers_dict.items():
        if name == "tfidf":
            feature_list.append(transformer.transform(df["text"].tolist()))
        elif name == "bow":
            feature_list.append(transformer.transform(df["text"].tolist()))
        elif name == "lda_countvec":
            count_vec = transformers_dict.get("lda_countvec")
            X_counts = count_vec.transform(df["text"].tolist())
            lda_model = transformers_dict.get("lda")
            feature_list.append(csr_matrix(lda_model.transform(X_counts)))
        elif name == "sentence_embeddings":
            embeddings = sentence_embeddings(df["text"].tolist(), model_name=transformer)
            feature_list.append(embeddings)
        elif name == "svd":
            # Will apply at the end
            pass

    # Add text stats (if used during training)
    if use_text_stats:
        stats = text_statistics(df, "text")
        stats_features = stats[[col for col in stats.columns if col.startswith("text_")]].values
        feature_list.append(stats_features)

    # Combine features
    dense_features = [f for f in feature_list if isinstance(f, np.ndarray)]
    sparse_features = [f for f in feature_list if not isinstance(f, np.ndarray)]

    if dense_features:
        dense_combined = np.hstack(dense_features)
        X = hstack(sparse_features + [csr_matrix(dense_combined)]) if sparse_features else dense_combined
    else:
        X = hstack(sparse_features) if sparse_features else None

    # Apply SVD if present
    if "svd" in transformers_dict and X is not None:
        X = transformers_dict["svd"].transform(X)

    return X

# ------------------------
# Streamlit App
# ------------------------
st.title("Webpage Classification App")
st.write("Enter a webpage's cleaned text to predict its category.")

# Load model and transformers
model, transformers_dict = load_artifacts()

# User input
user_input = st.text_area("Website Text:")

if st.button("Predict"):
    if not user_input:
        st.warning("Please enter some text to classify.")
    elif model is None or transformers_dict is None:
        st.error("Model or transformers not loaded properly.")
    else:
        X_input = preprocess_input(user_input, transformers_dict, use_text_stats=use_text_stats)
        if X_input is None:
            st.error("Failed to process input text.")
        else:
            prediction = model.predict(X_input)
            st.success(f"Predicted Category: {prediction[0]}")