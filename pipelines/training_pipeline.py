"""
training_pipeline.py
--------------------
Config-driven training pipeline for webpage classification or regression.
Supports flexible preprocessing, feature engineering, model training, hyperparameter tuning, and MLflow logging.
"""

import os
import yaml
import numpy as np
import mlflow
import mlflow.sklearn
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Utils
from utils.data_io import load_file
from utils.data_cleaning import remove_duplicates, drop_missing_rows
from utils.text_processing import (
    remove_html_tags, normalize_unicode, strip_whitespace, lowercase_text,
    remove_punctuation, remove_stopwords, lemmatize_text, text_statistics,
    bag_of_words, tfidf_vectorize, apply_text_pipeline, lda_topic_modeling,
    sentence_embeddings
)
from utils.model_utils import evaluate_classification, evaluate_regression, save_model


# ---------------------------
# Load Config
# ---------------------------
def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Training Pipeline
# ---------------------------
def training_pipeline(file_path: str, text_column: str, target_column: str, config_path: str):
    """
    Training pipeline driven by config file with hyperparameter tuning support.

    Args:
        file_path: Path to CSV
        text_column: Column with text data
        target_column: Column with labels
        config: Dictionary loaded from YAML config
    Returns:
        model, metrics, transformers_dict
    """

    # ------------------------
    # 0. Load config
    # ------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # ------------------------
    # Extract configs
    # ------------------------
    problem_type = config.get("problem_type", "classification")  # classification / regression
    preprocessing_cfg = config.get("preprocessing", {})
    model_cfg = config.get("model", {})

    features = preprocessing_cfg.get("features", ["TF-IDF"])
    max_features = preprocessing_cfg.get("max_features", 5000)
    n_topics = preprocessing_cfg.get("n_topics", 10)
    sentence_embed_model = preprocessing_cfg.get("sentence_embed_model", "all-MiniLM-L6-v2")
    reduce_dim = preprocessing_cfg.get("reduce_dim", False)
    svd_components = preprocessing_cfg.get("svd_components", 300)
    transformers_save_path = preprocessing_cfg.get("save_path", "artifacts/transformers/")  

    model_type = model_cfg.get("type", "Logistic Regression")
    model_params = model_cfg.get("params", {})
    model_save_path = model_cfg.get("save_path", "artifacts/model.pkl")
    tuning_cfg = model_cfg.get("tuning", None)  # New tuning config

    # ------------------------
    # 1. Load + Clean Data
    # ------------------------
    df = load_file(file_path)
    df = remove_duplicates(df)
    df = drop_missing_rows(df, subset=[text_column])
    df = df[[text_column, target_column]]

    # ------------------------
    # 2. Text Cleaning
    # ------------------------
    cleaning_steps = [
        remove_html_tags, normalize_unicode, strip_whitespace, lowercase_text,
        remove_punctuation, remove_stopwords, lemmatize_text
    ]
    df = apply_text_pipeline(df, text_column, cleaning_steps)

    # ------------------------
    # 3. Feature Engineering
    # ------------------------
    transformers_dict = {}
    feature_list = []

    if "Text Stats" in features:
        df = text_statistics(df, text_column)
        stats_features = df[[col for col in df.columns if col.startswith(text_column+"_")]].values
        feature_list.append(stats_features)

    if "TF-IDF" in features:
        X_tfidf, tfidf_vectorizer = tfidf_vectorize(df[text_column].tolist(), max_features=max_features)
        feature_list.append(X_tfidf)
        transformers_dict["tfidf"] = tfidf_vectorizer

    if "Bag-of-Words" in features:
        X_bow, bow_vectorizer = bag_of_words(df, text_column, max_features=max_features)
        feature_list.append(X_bow)
        transformers_dict["bow"] = bow_vectorizer

    if "LDA" in features:
        from sklearn.feature_extraction.text import CountVectorizer
        count_vec = CountVectorizer(max_features=max_features)
        X_counts = count_vec.fit_transform(df[text_column].tolist())
        lda_model = lda_topic_modeling(X_counts, n_topics=n_topics)
        lda_features = lda_model.transform(X_counts)
        feature_list.append(csr_matrix(lda_features))
        transformers_dict["lda"] = lda_model
        transformers_dict["lda_countvec"] = count_vec

    if "Sentence Embeddings" in features:
        embeddings = sentence_embeddings(df[text_column].tolist(), model_name=sentence_embed_model)
        feature_list.append(embeddings)
        transformers_dict["sentence_embeddings"] = sentence_embed_model

    # Combine features (handle sparse + dense)
    dense_features = [f for f in feature_list if isinstance(f, np.ndarray)]
    sparse_features = [f for f in feature_list if not isinstance(f, np.ndarray)]

    if dense_features:
        dense_combined = np.hstack(dense_features)  # stack all dense arrays horizontally
        X = hstack(sparse_features + [csr_matrix(dense_combined)]) if sparse_features else dense_combined
    else:
        X = hstack(sparse_features)

    # Dimensionality reduction
    if reduce_dim:
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        X = svd.fit_transform(X)
        transformers_dict["svd"] = svd

    # Save fitted transformers individually
    os.makedirs(transformers_save_path, exist_ok=True)

    for name, transformer in transformers_dict.items():
        transformer_path = os.path.join(transformers_save_path, f"{name}.joblib")
        save_model(transformer, transformer_path)
        print(f"Saved transformer '{name}' at {transformer_path}")

    # ------------------------
    # 4. Train/Test Split
    # ------------------------
    y = df[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type=="classification" else None
    )

    # ------------------------
    # 5. Model Selection + Hyperparameter Tuning
    # ------------------------
    model_mapping = {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier if problem_type=="classification" else RandomForestRegressor,
        "XGBoost": XGBClassifier if problem_type=="classification" else XGBRegressor,
        "LightGBM": LGBMClassifier if problem_type=="classification" else LGBMRegressor,
        "CatBoost": CatBoostClassifier if problem_type=="classification" else CatBoostRegressor,
        "SVM": SVC if problem_type=="classification" else SVR,
        "Gradient Boosting": GradientBoostingClassifier if problem_type=="classification" else GradientBoostingRegressor,
        "Linear Regression": LinearRegression
    }

    if model_type not in model_mapping:
        raise ValueError(f"Unsupported model: {model_type}")

    ModelClass = model_mapping[model_type]
    model = ModelClass(**model_params)

    # Hyperparameter tuning
    if tuning_cfg.get("enable", False):
        method = tuning_cfg.get("method", "grid")  # grid / random
        param_grid = tuning_cfg.get("param_grid", {})
        cv = tuning_cfg.get("cv", 3)
        scoring = tuning_cfg.get("scoring", None)

        if method == "grid":
            search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
        elif method == "random":
            n_iter = tuning_cfg.get("n_iter", 10)
            search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=n_iter, scoring=scoring)
        else:
            raise ValueError(f"Unsupported tuning method: {method}")

        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model.fit(X_train, y_train)
        best_params = model_params

    # ------------------------
    # 6. Evaluate + Save
    # ------------------------
    if problem_type == "classification":
        metrics = evaluate_classification(model, X_test, y_test)
    else:
        metrics = evaluate_regression(model, X_test, y_test)

    save_model(model, model_save_path)

    # ------------------------
    # 7. Log to MLflow
    # ------------------------
    with mlflow.start_run():
        mlflow.log_params({**preprocessing_cfg, **best_params, "problem_type": problem_type})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "model")

    return model, metrics, transformers_dict