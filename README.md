## Webpage Classification Project

### Overview

This project provides a config-driven, modular pipeline to classify webpages into categories using text data. The pipeline supports:
	•	Flexible preprocessing and feature engineering (TF-IDF, Bag-of-Words, LDA, Sentence Embeddings, Text Statistics)
	•	Dimensionality reduction with TruncatedSVD
	•	Multiple ML models with hyperparameter tuning
	•	Experiment tracking via MLflow
	•	A Streamlit web app for real-time inference on new webpage text

The project emphasizes modularity, reproducibility, and ease of deployment, making it suitable for research or production prototypes.

⸻

### Project Structure

Webpage-Classification/
│
├─ app/                      # Optional scripts for deployment
├─ artifacts/                # Saved models, transformers, and outputs (ignored in git)
├─ configs/                  # YAML config files
│   └─ training_config.yaml
├─ data/                     # Dataset CSVs
│   └─ webpage_data.csv
├─ mlruns/                   # MLflow experiment tracking
├─ pipelines/                # Core training pipeline
│   └─ training_pipeline.py
├─ utils/                    # Helper utilities
│   ├─ data_io.py
│   ├─ data_cleaning.py
│   ├─ text_processing.py
│   └─ model_utils.py
├─ main.py                   # Script to run training pipeline
├─ streamlit_app.py          # Streamlit web app for inference
├─ requirements.txt
└─ README.md

⸻

### Key Features
	1.	Config-driven Training
	-- •	All parameters (preprocessing, model type, hyperparameters, tuning) are defined in configs/training_config.yaml.
	-- •	Adding new features or models does not require changing code.
	2.	Flexible Preprocessing
	-- •	Cleaning: HTML tag removal, whitespace stripping, punctuation removal, stopword removal, lemmatization.
	-- •	Feature engineering:
	-- •	TF-IDF, Bag-of-Words
	-- •	LDA topic modeling
	-- •	Sentence embeddings via Sentence-BERT
	-- •	Text statistics (word count, unique words, average word length)
	-- •	Optional dimensionality reduction using TruncatedSVD.
	3.	Model Training & Hyperparameter Tuning
	-- •	Supports multiple classifiers/regressors:
	-- •	Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, SVM, Linear Regression
	-- •	Hyperparameter tuning via GridSearchCV or RandomizedSearchCV.
	-- •	Evaluation metrics logged using MLflow.
	4.	Artifact Management
	-- •	Fitted transformers are saved individually in artifacts/transformers/.
	-- •	Trained model saved as artifacts/model.joblib.
	-- •	Easy to reload for inference in the Streamlit app.
	5.	Streamlit Web App
	-- •	Real-time webpage text classification.
	-- •	Applies the same preprocessing and feature transformations as the training pipeline.
	-- •	Users can input raw webpage text and get predicted categories instantly.

⸻

## Setup Instructions

1. Clone Repository

git clone <repo_url>
cd Webpage-Classification

2. Create Conda Environment

conda create -n webpage_classify_env python=3.11
conda activate webpage_classify_env

3. Install Dependencies

pip install -r requirements.txt

4. Prepare Dataset

Place your dataset CSV in data/webpage_data.csv. The CSV should have at least:
	•	A text column (e.g., cleaned_website_text)
	•	A target label column (e.g., Category)

5. Train Model

python main.py
	•	Trained model and transformers will be saved in artifacts/.

6. Launch Streamlit App

streamlit run streamlit_app.py
	•	Open the URL displayed in the terminal to interact with the app.

⸻

## Configuration (training_config.yaml)

Example:

preprocessing:
features:
- “TF-IDF”
- “Text Stats”
max_features: 5000
n_topics: 10
sentence_embed_model: “all-MiniLM-L6-v2”
reduce_dim: true
svd_components: 300
save_path: “artifacts/transformers/”

model:
type: “Logistic Regression”
params:
max_iter: 500
save_path: “artifacts/model.joblib”
tuning:
enable: true
method: “grid”
param_grid:
C: [0.1, 1.0, 10]
cv: 3
scoring: “accuracy”
	•	Features: Select preprocessing steps.
	•	Model: Choose algorithm, hyperparameters, and tuning options.
	•	Transformers save path: Where all fitted transformers are persisted.

⸻

Notes
	•	The training pipeline only uses the text column from the dataset to avoid mismatches with TruncatedSVD or other features.
	•	Transformers (TF-IDF, LDA, etc.) are saved individually for modular inference.
	•	use_text_stats can be toggled in the Streamlit app to include/exclude text statistics features.
	•	.gitignore excludes artifacts/, virtual environment, MLflow runs, cache, and VSCode settings.

⸻

Future Improvements
	•	Add multi-class thresholding or confidence scores for predictions.
	•	Support batch predictions in the Streamlit app.
	•	Deploy using Docker or cloud services for production.
	•	Add more feature engineering steps like n-grams, POS tags, or domain-specific embeddings.

⸻

If you want, I can also create a Mermaid flow diagram for this README showing the data flow from dataset → pipeline → transformers → model → Streamlit app.
