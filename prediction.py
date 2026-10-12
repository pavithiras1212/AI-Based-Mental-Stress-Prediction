"""Bayesian Stress Prediction Engine for MindBloom.

Implements a mixed-feature Naive Bayes classifier:
- Continuous features: Gaussian probability density with class-specific mean and variance.
- Categorical features: Class-conditional likelihood with Laplace smoothing.
- Class priors: Derived from training dataset empirical frequencies.
- Inference: Bayes' rule with log-space accumulation and log-sum-exp normalization.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin

from preprocessing import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_NAMES,
    CATEGORICAL_FEATURE_NAMES,
    STRESS_LABELS,
    transform_features,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_stress_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"

ALL_STRESS_LEVELS = [1, 2, 3, 4, 5]


class MixedNaiveBayes(BaseEstimator, ClassifierMixin):
    """Mixed Naive Bayes classifier supporting Gaussian and Categorical likelihoods.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features that is added to
        variances for calculation stability.
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter for categorical features.
    """

    def __init__(self, var_smoothing=1e-9, alpha=1.0):
        self.var_smoothing = var_smoothing
        self.alpha = alpha

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y, dtype=int)

        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) and y ({len(y)}) length mismatch.")

        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        total_samples = float(class_counts.sum())

        # Class Priors: P(C = c) = N_c / N
        self.class_priors_ = {int(c): count / total_samples for c, count in zip(self.classes_, class_counts)}
        self.class_log_prior_ = np.log(class_counts / total_samples)

        # Separate numeric vs categorical columns
        self.numeric_columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_columns_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

        # 1. Gaussian Distribution Statistics for continuous features
        # Computes class-specific mean mu_{c, j} and variance sigma^2_{c, j}
        self.numeric_stats_ = {}
        for col in self.numeric_columns_:
            values = pd.to_numeric(X[col], errors="coerce")
            global_var = float(values.var(skipna=True)) if values.notna().sum() > 1 else 1.0
            if not np.isfinite(global_var) or global_var <= 0:
                global_var = 1.0
            # Epsilon variance floor for numerical stability
            epsilon = max(global_var * self.var_smoothing, 1e-9)
            global_mean = float(values.mean(skipna=True)) if values.notna().any() else 0.0

            stats = {}
            for cls in self.classes_:
                group = values[y == cls].dropna()
                mean = float(group.mean()) if len(group) > 0 else global_mean
                variance = float(group.var(ddof=0)) if len(group) > 1 else global_var
                if not np.isfinite(variance) or variance < epsilon:
                    variance = epsilon
                stats[int(cls)] = {"mean": mean, "variance": variance}
            self.numeric_stats_[col] = stats

        # 2. Categorical Distribution Statistics with Laplace Smoothing
        self.category_values_ = {}
        self.category_counts_ = {}
        for col in self.categorical_columns_:
            series = X[col].astype(str).str.strip().str.lower()
            unique_vals = sorted(series.unique().tolist())
            self.category_values_[col] = unique_vals
            self.category_counts_[col] = {}
            for cls in self.classes_:
                cls_series = series[y == cls]
                self.category_counts_[col][int(cls)] = cls_series.value_counts().to_dict()

        return self

    def _compute_components(self, X):
        """Compute prior, numerical log-likelihoods, and categorical log-likelihoods."""
        X = pd.DataFrame(X).reindex(columns=list(self.feature_names_in_))
        num_rows = len(X)
        num_classes = len(self.classes_)

        prior_log = np.tile(self.class_log_prior_, (num_rows, 1)).astype(float)
        numeric_log = np.zeros((num_rows, num_classes), dtype=float)
        categorical_log = np.zeros((num_rows, num_classes), dtype=float)

        feature_log_likelihoods = {col: np.zeros((num_rows, num_classes), dtype=float) for col in self.feature_names_in_}

        # Continuous numeric features: Gaussian log PDF
        # ln f(x; mu, sigma^2) = -0.5 * ln(2*pi*sigma^2) - 0.5 * (x - mu)^2 / sigma^2
        for col in self.numeric_columns_:
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            for j, cls in enumerate(self.classes_):
                mu = self.numeric_stats_[col][int(cls)]["mean"]
                var = self.numeric_stats_[col][int(cls)]["variance"]
                if valid.any():
                    term = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((values[valid] - mu) ** 2) / var
                    numeric_log[valid, j] += term
                    feature_log_likelihoods[col][valid, j] = term

        # Categorical features: Laplace smoothed log likelihood
        # P(x=v | C=c) = (N_{c,v} + alpha) / (N_c + alpha * (|V| + 1))
        for col in self.categorical_columns_:
            series = X[col].astype(str).str.strip().str.lower().to_numpy()
            known_vals = self.category_values_[col]
            num_buckets = len(known_vals) + 1  # +1 for unseen category smoothing

            for j, cls in enumerate(self.classes_):
                counts = self.category_counts_[col][int(cls)]
                total_counts = sum(counts.values())
                denom = total_counts + self.alpha * num_buckets

                for i, val in enumerate(series):
                    if val in ("nan", "none", "__missing__", ""):
                        continue
                    prob = (counts.get(val, 0) + self.alpha) / denom
                    prob = max(prob, 1e-12)
                    lp = np.log(prob)
                    categorical_log[i, j] += lp
                    feature_log_likelihoods[col][i, j] = lp

        joint = prior_log + numeric_log + categorical_log
        return joint, prior_log, numeric_log, categorical_log, feature_log_likelihoods

    def predict_log_proba(self, X):
        joint, _, _, _, _ = self._compute_components(X)
        # Log-Sum-Exp normalization across classes:
        # ln P(C=c | X) = joint_c - ln(sum_{c'} exp(joint_{c'}))
        return joint - logsumexp(joint, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def load_model_and_features():
    """Load model, features list, and metadata safely."""
    for path in (MODEL_PATH, FEATURES_PATH, METADATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required model artifact not found at {path}. Please run `python model.py` to train and generate artifacts."
            )

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    metadata = joblib.load(METADATA_PATH)

    if isinstance(features, dict) and "columns" in features:
        features = list(features["columns"])
    elif isinstance(features, (list, tuple)):
        features = list(features)

    metadata["columns"] = features
    return model, metadata


def predict_stress_details(student_data):
    """Predict stress level and compute exact Bayesian posterior probabilities.

    Parameters
    ----------
    student_data : dict or pd.DataFrame
        Student responses to the 8 survey questions.

    Returns
    -------
    dict
        Detailed dictionary with predicted_level, label, confidence,
        probabilities for all 5 stress classes, input summary, and
        step-by-step Bayesian inference calculation breakdown.
    """
    model, metadata = load_model_and_features()

    # Preprocess student input
    df_raw = pd.DataFrame([student_data]) if isinstance(student_data, dict) else student_data.copy()
    X_processed = transform_features(df_raw)

    raw_prediction = int(model.predict(X_processed)[0])
    predicted_level = raw_prediction if raw_prediction in ALL_STRESS_LEVELS else 3

    # Posterior probabilities
    posteriors = model.predict_proba(X_processed)[0]
    classes = [int(c) for c in getattr(model, "classes_", ALL_STRESS_LEVELS)]

    probabilities = {str(lvl): 0.0 for lvl in ALL_STRESS_LEVELS}
    for cls, prob in zip(classes, posteriors):
        if cls in ALL_STRESS_LEVELS:
            probabilities[str(cls)] = float(prob)

    # Normalize strictly to sum to 1.0
    total_p = sum(probabilities.values())
    if total_p > 0:
        probabilities = {k: v / total_p for k, v in probabilities.items()}

    confidence = probabilities.get(str(predicted_level), 0.0)

    # Identify any categories unseen during training
    unseen_features = []
    trained_categories = metadata.get("categories", {})
    for col, allowed in trained_categories.items():
        if col in df_raw.columns:
            val = df_raw.iloc[0][col]
            if pd.notna(val) and str(val).strip().casefold() not in {str(a).strip().casefold() for a in allowed}:
                unseen_features.append(col)

    # Compute Bayesian inference breakdown for the single input
    joint, prior_log, numeric_log, categorical_log, feat_log = model._compute_components(X_processed)
    evidence_log = float(logsumexp(joint[0]))

    breakdown = {
        "classes": classes,
        "class_labels": [STRESS_LABELS.get(c, f"Level {c}") for c in classes],
        "log_priors": {str(c): float(prior_log[0, j]) for j, c in enumerate(classes)},
        "priors": {str(c): float(np.exp(prior_log[0, j])) for j, c in enumerate(classes)},
        "numeric_log_likelihood": {str(c): float(numeric_log[0, j]) for j, c in enumerate(classes)},
        "categorical_log_likelihood": {str(c): float(categorical_log[0, j]) for j, c in enumerate(classes)},
        "unnormalized_log_posterior": {str(c): float(joint[0, j]) for j, c in enumerate(classes)},
        "log_evidence": evidence_log,
        "feature_contributions": {
            feat: {str(c): float(feat_log[feat][0, j]) for j, c in enumerate(classes)}
            for feat in model.feature_names_in_
        },
    }

    all_levels_dict = {
        lvl: {
            "label": STRESS_LABELS[lvl],
            "probability": probabilities[str(lvl)] * 100.0,
            "probability_fraction": probabilities[str(lvl)],
        }
        for lvl in ALL_STRESS_LEVELS
    }

    return {
        "predicted_level": predicted_level,
        "prediction": predicted_level,
        "stress_level": predicted_level,
        "label": STRESS_LABELS[predicted_level],
        "stress_label": STRESS_LABELS[predicted_level],
        "confidence": confidence,
        "probabilities": probabilities,
        "all_levels": all_levels_dict,
        "unseen_features": unseen_features,
        "bayesian_breakdown": breakdown,
        "student_input": df_raw.iloc[0].to_dict(),
        "processed_features": X_processed.iloc[0].to_dict(),
    }


def predict_stress(student_data):
    """Simple prediction function returning integer stress level (1-5)."""
    return predict_stress_details(student_data)["predicted_level"]
