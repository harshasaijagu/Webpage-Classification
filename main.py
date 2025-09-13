"""
main.py
--------
Entry point to execute the training pipeline.
Uses the config-driven training_pipeline.py module.
"""

import os
from pathlib import Path
from pipelines.training_pipeline import training_pipeline

def main():
    """
    Run the training pipeline using the config in training_pipeline.py
    """
    
    # Get project root dynamically
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Paths
    config_path = os.path.join(PROJECT_ROOT, "configs", "training_config.yaml")
    file_path = os.path.join(PROJECT_ROOT, "data", "webpages_data.csv")
    
    # Define dataset path and columns
    text_column = "cleaned_website_text"
    target_column = "Category"
    
    # Run training pipeline
    model, metrics, transformers_dict = training_pipeline(
        file_path=file_path,
        text_column=text_column,
        target_column=target_column,
        config_path=config_path  # the pipeline reads the YAML internally
    )
    
    # Print metrics
    print("Training Complete. Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # # Save transformers for inference
    # transformers_save_path = Path("artifacts/transformers")
    # import joblib
    # joblib.dump(transformers_dict, transformers_save_path)
    # print(f"Transformers saved at {transformers_save_path}")

if __name__ == "__main__":
    main()