import os

import joblib
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

MODEL_DIR = "data/models/"

# The tuned hyperparameters of xgboost_production.pkl, read back from the saved model.
PRODUCTION_XGB_PARAMS = {
    "n_estimators": 705,
    "max_depth": 8,
    "learning_rate": 0.12963216466172517,
    "subsample": 0.7015663799319409,
    "colsample_bytree": 0.9832675663386035,
    "gamma": 0.3162663223081429,
    "min_child_weight": 1,
    "reg_alpha": 5.167682983405994e-08,
    "reg_lambda": 5.7057896030047325e-06,
    "eval_metric": "aucpr",
    "enable_categorical": True,
    "random_state": 42,
    "n_jobs": -1,
}


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
        class_weight="balanced",
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
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
    }

    base_model = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight="balanced"
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )
    if len(X_train) > 50000:
        print(
            f"Downsampling for GridSearchCV from {len(X_train)} to 50000 rows to save time..."
        )
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
        **grid_search.best_params_, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    tuned_model.fit(X_train, Y_train)

    return tuned_model, grid_search


def train_xgboost_baseline(X_train, Y_train):
    """
    Train a baseline XGBoost classifier for fraud detection.

    Args:
        X_train (pd.DataFrame): The training features.
        Y_train (pd.Series/pd.DataFrame): The training labels.

    Returns:
        XGBClassifier: The trained model.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)
    return model


def train_lightgbm_baseline(X_train, Y_train):
    """
    Train a baseline LightGBM classifier for fraud detection.

    Args:
        X_train (pd.DataFrame): The training features.
        Y_train (pd.Series/pd.DataFrame): The training labels.

    Returns:
        LGBMClassifier: The trained model.
    """
    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_train, Y_train)
    return model


def tune_xgboost(X_train, Y_train, n_trials=50):
    """
    Tune XGBoost with Optuna, then refit the best settings on all training data.

    Each trial is scored by average precision on the most recent 20% of the rows,
    which are held out in time order to match how the model is used.

    Args:
        X_train (pd.DataFrame): The training features, in time order.
        Y_train (pd.Series/pd.DataFrame): The training labels.
        n_trials (int): The number of Optuna trials.

    Returns:
        tuple: The refitted XGBClassifier and the Optuna study.
    """
    split = int(len(X_train) * 0.8)
    X_fit, X_val = X_train.iloc[:split], X_train.iloc[split:]
    Y_fit, Y_val = Y_train.iloc[:split], Y_train.iloc[split:]
    fixed = {
        "eval_metric": "aucpr",
        "enable_categorical": True,
        "random_state": 42,
        "n_jobs": -1,
    }

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = XGBClassifier(**params, **fixed)
        model.fit(X_fit, Y_fit)
        return average_precision_score(Y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)

    model = XGBClassifier(**study.best_params, **fixed)
    model.fit(X_train, Y_train)
    return model, study
