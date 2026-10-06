import os
import pandas as pd
import joblib

current_folder = os.path.dirname(os.path.abspath(__file__))
# =========================================================
# LOAD PROCESSED DATASET
# =========================================================

DATA_PATH = os.path.join(
    current_folder,
    "data",
    "processed",
    "cleaned_student_stress_data.xlsx"
)

df = pd.read_excel(DATA_PATH)

print("Dataset loaded successfully!")
print(df.shape)
print(df.columns.tolist())
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace("\n", " ", regex=False)

print("\nCleaned Columns:")
print(df.columns.tolist())
TARGET_COLUMN = "Overall, how would you rate your current stress level?"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column not found: {TARGET_COLUMN}"
    )

print("\nTarget Column:")
print(TARGET_COLUMN)

print("\nTarget Values:")
print(df[TARGET_COLUMN].value_counts())
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])


X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())
X = pd.get_dummies(
    X,
    drop_first=True
)

X = X.fillna(0)

print("\nProcessed Feature Shape:")
print(X.shape)

print("\nProcessed Feature Data Types:")
print(X.dtypes)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    
    "SVM": SVC()
}


results = {}

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))




results_df = pd.DataFrame(results).T

print("\n\n================================")
print("MODEL COMPARISON")
print("================================")

print(results_df)
best_model_name = results_df["F1 Score"].idxmax()

print("\n================================")
print("BEST MODEL")
print("================================")

print("Best Model:", best_model_name)

best_model = models[best_model_name]

best_model.fit(X_train, y_train)


model_path = os.path.join(
    current_folder,
    "best_stress_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)


features_path = os.path.join(
    current_folder,
    "model_features.pkl"
)

joblib.dump(
    list(X.columns),
    features_path
)


results_path = os.path.join(
    current_folder,
    "model_comparison.csv"
)

results_df.to_csv(
    results_path
)


print("\nFiles created successfully:")
print(model_path)
print(features_path)
print(results_path)
