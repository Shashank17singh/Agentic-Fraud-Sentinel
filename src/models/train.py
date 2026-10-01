import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score

MODEL_DIR = "data/models/"

def train_random_forest_baseline(X_train, Y_train):
    """
    Train a baseline Random Forest classifier for fraud detection.
    
    Args:
        X_train (pd.DataFrame): The training features.
        Y_train (pd.Series/pd.DataFrame): The training labels.
        
    Returns:
        RandomForestClassifier: The trained model.
    """
    print("Training Random Forest baseline:")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, Y_train)
    print("Random Forest baseline trained")

    return model

def save_model(model, name):
    """
    Save a trained scikit-learn model to disk using joblib.
    
    Args:
        model: The trained scikit-learn model.
        name (str): The filename (without extension).
        
    Returns:
        str: The path where the model was saved.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path

def load_model(name):
    """
    Load a saved scikit-learn model from disk.
    
    Args:
        name (str): The filename (without extension).
        
    Returns:
        The loaded scikit-learn model.
    """
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    return joblib.load(path)

def tune_random_forest(X_train, Y_train):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV.
    
    Uses StratifiedKFold cross-validation and optimizes for average_precision (AUC-PR).
    Downsamples the training data if it exceeds 50,000 rows to speed up tuning.
    
    Args:
        X_train (pd.DataFrame): The training features.
        Y_train (pd.Series/pd.DataFrame): The training labels.
        
    Returns:
        tuple: (tuned_model, grid_search_object)
    """
    print("Tuning Random Forest with GridSearchCV...")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    
    base_model = RandomForestClassifier(
        random_state=42, 
        n_jobs=-1, 
        class_weight="balanced"
    )
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='average_precision',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    if len(X_train) > 50000:
        print(f"Downsampling for GridSearchCV from {len(X_train)} to 50000 rows to save time...")
        from sklearn.model_selection import train_test_split
        X_tune, _, Y_tune, _ = train_test_split(
            X_train, Y_train, train_size=50000, stratify=Y_train, random_state=42
        )
        grid_search.fit(X_tune, Y_tune)
    else:
        grid_search.fit(X_train, Y_train)
        
    print(f"\nBest AUC-PR (CV): {grid_search.best_score_:.4f}")
    print("Best params:", grid_search.best_params_)
    
    print("\nTraining final model with best params on full data...")
    tuned_model = RandomForestClassifier(
        **grid_search.best_params_,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    tuned_model.fit(X_train, Y_train)
    
    return tuned_model, grid_search
