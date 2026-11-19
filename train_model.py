"""Train and compare four classifiers for the Diabetes Risk Prediction project."""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "model"
MODELS_PATH = MODEL_DIR / "models.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
ALL_COLUMNS = FEATURES + [TARGET]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42


def load_and_inspect():
    """Read and validate the dataset, and print a short data inspection."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Place diabetes.csv in the dataset folder before training."
        )
    df = pd.read_csv(DATASET_PATH)
    missing_columns = [column for column in ALL_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")
    df = df[ALL_COLUMNS].copy()
    print("Dataset shape:", df.shape)
    print("Missing values before preprocessing:\n", df.isnull().sum())
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Outcome counts:\n", df[TARGET].value_counts().sort_index())
    return df


def preprocess(df):
    """Replace invalid zero measurements and return features and target."""
    cleaned = df.copy()
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, np.nan)
    return cleaned[FEATURES], cleaned[TARGET]


def build_models():
    """Create four pipelines; preprocessing is fitted only on training data."""
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced",
                                               random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                                   class_weight="balanced",
                                                   random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                   class_weight="balanced",
                                                   random_state=RANDOM_STATE,
                                                   n_jobs=-1)),
        ]),
        "Support Vector Machine (SVM)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=1.0, kernel="rbf", probability=True,
                               class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(model, X_test, y_test):
    """Calculate classification metrics on the untouched test partition."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }


def main():
    df = load_and_inspect()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTraining records: {len(X_train)} | Testing records: {len(X_test)}")

    trained_models = {}
    comparison = {}
    for name, model in build_models().items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        comparison[name] = evaluate(model, X_test, y_test)
        print(f"{name}: accuracy={comparison[name]['accuracy']:.4f}, "
              f"recall={comparison[name]['recall']:.4f}, "
              f"F1={comparison[name]['f1_score']:.4f}, "
              f"ROC-AUC={comparison[name]['roc_auc']:.4f}")

    # Rank by held-out accuracy because that is the comparison requested.
    # Recall/F1/ROC-AUC are also shown so accuracy is not the only consideration.
    best_name = max(comparison, key=lambda name: comparison[name]["accuracy"])
    results = {
        "dataset": "Pima Indians Diabetes Dataset",
        "total_records": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "decision_threshold": 0.5,
        "selection_metric": "accuracy",
        "best_algorithm": best_name,
        "features": FEATURES,
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "algorithms": comparison,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models, MODELS_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n--- Algorithm comparison (test set) ---")
    for name, values in sorted(comparison.items(), key=lambda item: item[1]["accuracy"], reverse=True):
        print(f"{name:30} Accuracy: {values['accuracy']:.3f} | "
              f"Precision: {values['precision']:.3f} | Recall: {values['recall']:.3f} | "
              f"F1: {values['f1_score']:.3f} | ROC-AUC: {values['roc_auc']:.3f}")
    print(f"\nBest accuracy in this split: {best_name}")
    print("Saved models:", MODELS_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Training complete. Now run: python app.py")


if __name__ == "__main__":
    main()
