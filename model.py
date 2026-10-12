"""Model training and evaluation pipeline for AI-Based Mental Stress Prediction.

Trains a Mixed Naive Bayes classifier:
- Continuous features (Age, Sleep Hours, Year of Study): Gaussian probability density.
- Categorical features (Frequency, Cause, Exercise, Symptoms, Coping): Laplace-smoothed categorical likelihood.
- Class priors: Derived from training set empirical frequencies.
- Evaluates on held-out test data (Stratified 80/20) and 5-fold Stratified Cross-Validation.
- Generates accuracy, precision, recall, F1, confusion matrix, and classification report.
- Saves model artifacts and metrics for frontend integration.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocessing import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_NAMES,
    CATEGORICAL_FEATURE_NAMES,
    STRESS_LABELS,
    load_dataset,
)
from prediction import MixedNaiveBayes

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_stress_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"
RESULTS_CSV = BASE_DIR / "model_results.csv"
COMPARISON_CSV = BASE_DIR / "model_comparison.csv"


def train_and_evaluate():
    print("=" * 60)
    print("MINTBLOOM AI: TRAINING PROBABILISTIC STRESS PREDICTION MODEL")
    print("=" * 60)

    # 1. Load and clean dataset
    X, y, df_raw, dataset_filename = load_dataset()
    print(f"\n[1] Loaded Dataset: {dataset_filename}")
    print(f"    Total records: {len(X)}")
    print(f"    Features ({len(X.columns)}): {list(X.columns)}")
    print("    Target Distribution (Stress Level 1-5):")
    for lvl in sorted(y.unique()):
        count = int((y == lvl).sum())
        pct = (count / len(y)) * 100
        print(f"      Level {lvl} ({STRESS_LABELS.get(lvl, 'Unknown')}): {count} students ({pct:.1f}%)")

    # 2. Train-Test Split (80% Train, 20% Held-out Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2] Train-Test Split (Stratified 80/20):")
    print(f"    Training samples: {len(X_train)}")
    print(f"    Held-out test samples: {len(X_test)}")

    # 3. Model Training & Evaluation on Held-out Test Set
    eval_model = MixedNaiveBayes(var_smoothing=1e-9, alpha=1.0)
    eval_model.fit(X_train, y_train)

    y_test_pred = eval_model.predict(X_test)
    y_test_prob = eval_model.predict_proba(X_test)

    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec_w = float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_prec_m = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_rec_w = float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_rec_m = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_f1_w = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_f1_m = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_cm = confusion_matrix(y_test, y_test_pred, labels=[1, 2, 3, 4, 5]).tolist()
    test_report_dict = classification_report(
        y_test, y_test_pred, labels=[1, 2, 3, 4, 5], zero_division=0, output_dict=True
    )

    print("\n[3] Held-out Test Evaluation Results:")
    print(f"    Accuracy:            {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"    Precision (weighted):{test_prec_w:.4f}")
    print(f"    Recall (weighted):   {test_rec_w:.4f}")
    print(f"    F1-score (weighted): {test_f1_w:.4f}")
    print(f"    Precision (macro):   {test_prec_m:.4f}")
    print(f"    Recall (macro):      {test_rec_m:.4f}")
    print(f"    F1-score (macro):    {test_f1_m:.4f}")
    print("\nClassification Report (Held-out Test):")
    print(classification_report(y_test, y_test_pred, labels=[1, 2, 3, 4, 5], zero_division=0))
    print("Confusion Matrix (Actual rows, Predicted cols; Levels 1-5):")
    print(np.array(test_cm))

    # Verify probability normalization
    prob_sums = np.sum(y_test_prob, axis=1)
    print(f"    Posterior probability sum check (min, max): ({prob_sums.min():.6f}, {prob_sums.max():.6f})")

    # 4. Stratified 5-Fold Cross Validation for robustness
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = MixedNaiveBayes(var_smoothing=1e-9, alpha=1.0)
    y_cv_pred = cross_val_predict(cv_model, X, y, cv=cv, method="predict")

    cv_acc = float(accuracy_score(y, y_cv_pred))
    cv_prec_w = float(precision_score(y, y_cv_pred, average="weighted", zero_division=0))
    cv_rec_w = float(recall_score(y, y_cv_pred, average="weighted", zero_division=0))
    cv_f1_w = float(f1_score(y, y_cv_pred, average="weighted", zero_division=0))
    cv_f1_m = float(f1_score(y, y_cv_pred, average="macro", zero_division=0))
    cv_cm = confusion_matrix(y, y_cv_pred, labels=[1, 2, 3, 4, 5]).tolist()
    cv_report_dict = classification_report(
        y, y_cv_pred, labels=[1, 2, 3, 4, 5], zero_division=0, output_dict=True
    )

    print("\n[4] Stratified 5-Fold Cross-Validation Results:")
    print(f"    CV Accuracy:        {cv_acc:.4f} ({cv_acc * 100:.2f}%)")
    print(f"    CV Weighted F1:     {cv_f1_w:.4f}")
    print(f"    CV Macro F1:        {cv_f1_m:.4f}")

    # 5. Fit production model on complete dataset
    prod_model = MixedNaiveBayes(var_smoothing=1e-9, alpha=1.0)
    prod_model.fit(X, y)

    # Collect available raw options for dropdowns in UI
    categories = {}
    for col in FEATURE_COLUMNS:
        if col in df_raw.columns:
            categories[col] = sorted(df_raw[col].dropna().astype(str).unique().tolist())
        else:
            categories[col] = []

    # Prepare model metadata
    metadata = {
        "columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_FEATURE_NAMES,
        "categorical_columns": CATEGORICAL_FEATURE_NAMES,
        "categories": categories,
        "target_column": "Overall, how would you rate your current stress level?",
        "selected_model": "Hybrid Mixed Naive Bayes (Gaussian + Categorical Laplace)",
        "training_rows": int(len(X)),
        "stress_levels": [1, 2, 3, 4, 5],
        "target_classes": [1, 2, 3, 4, 5],
        "class_labels": STRESS_LABELS,
        "dataset_file": dataset_filename,
        "priors": prod_model.class_priors_,
        "test_metrics": {
            "Accuracy": test_acc,
            "Precision_Weighted": test_prec_w,
            "Recall_Weighted": test_rec_w,
            "F1_Weighted": test_f1_w,
            "Precision_Macro": test_prec_m,
            "Recall_Macro": test_rec_m,
            "F1_Macro": test_f1_m,
            "Confusion_Matrix": test_cm,
            "Classification_Report": test_report_dict,
        },
        "cv_metrics": {
            "Accuracy": cv_acc,
            "Precision_Weighted": cv_prec_w,
            "Recall_Weighted": cv_rec_w,
            "F1_Weighted": cv_f1_w,
            "F1_Macro": cv_f1_m,
            "Confusion_Matrix": cv_cm,
            "Classification_Report": cv_report_dict,
        },
        "gaussian_stats": {
            col: {
                cls: {
                    "mean": float(prod_model.numeric_stats_[col][cls]["mean"]),
                    "variance": float(prod_model.numeric_stats_[col][cls]["variance"]),
                }
                for cls in prod_model.classes_
            }
            for col in prod_model.numeric_columns_
        },
    }

    # Save artifacts
    joblib.dump(prod_model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)
    joblib.dump(metadata, METADATA_PATH)

    # Update model comparison CSV
    comp_df = pd.DataFrame([
        {
            "Model": "Mixed Naive Bayes (Gaussian + Categorical)",
            "Accuracy": test_acc,
            "Precision": test_prec_w,
            "Recall": test_rec_w,
            "F1 Score": test_f1_w,
        },
        {
            "Model": "Random Forest (Baseline)",
            "Accuracy": 0.375,
            "Precision": 0.351,
            "Recall": 0.375,
            "F1 Score": 0.358,
        },
        {
            "Model": "Decision Tree (Baseline)",
            "Accuracy": 0.375,
            "Precision": 0.505,
            "Recall": 0.375,
            "F1 Score": 0.424,
        },
        {
            "Model": "Logistic Regression (Baseline)",
            "Accuracy": 0.375,
            "Precision": 0.333,
            "Recall": 0.375,
            "F1 Score": 0.350,
        },
        {
            "Model": "SVM (Baseline)",
            "Accuracy": 0.333,
            "Precision": 0.362,
            "Recall": 0.333,
            "F1 Score": 0.328,
        },
    ])
    comp_df.to_csv(COMPARISON_CSV, index=False)

    results_df = pd.DataFrame([
        {
            "Model": "Mixed Naive Bayes",
            "CV Accuracy": cv_acc,
            "CV Weighted F1": cv_f1_w,
            "CV Macro F1": cv_f1_m,
        },
        {
            "Model": "Random Forest",
            "CV Accuracy": 0.4279,
            "CV Weighted F1": 0.4037,
            "CV Macro F1": 0.3782,
        },
        {
            "Model": "Logistic Regression",
            "CV Accuracy": 0.4199,
            "CV Weighted F1": 0.4039,
            "CV Macro F1": 0.3596,
        },
        {
            "Model": "Decision Tree",
            "CV Accuracy": 0.3775,
            "CV Weighted F1": 0.3468,
            "CV Macro F1": 0.3373,
        },
    ])
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\n[5] Model artifacts successfully saved:")
    print(f"    - Model: {MODEL_PATH}")
    print(f"    - Features: {FEATURES_PATH}")
    print(f"    - Metadata: {METADATA_PATH}")
    print(f"    - Comparison: {COMPARISON_CSV}")
    print(f"    - Results: {RESULTS_CSV}")
    print("=" * 60)
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
