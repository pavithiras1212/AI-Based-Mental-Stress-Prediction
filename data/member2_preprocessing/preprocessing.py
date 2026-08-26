import pandas as pd
import os

# ============================================================
# 1. LOAD DATASET
# ============================================================

# CHANGE THIS ONLY if your Excel filename is different
file_path = "data/processed/cleaned_student_stress_data.xlsx"

df = pd.read_excel(file_path)

print("Original Dataset Shape:")
print(df.shape)

print("\nOriginal Column Names:")
print(df.columns.tolist())


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nCleaned Column Names:")
print(df.columns.tolist())


# ============================================================
# 3. REMOVE DUPLICATE ROWS
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows:")
print(duplicates)

df = df.drop_duplicates()


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

# Numerical columns → median
numeric_columns = df.select_dtypes(include="number").columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

# Text/categorical columns → mode
categorical_columns = df.select_dtypes(include=["object", "string"]).columns

for col in categorical_columns:
    if df[col].isnull().any():
        mode_value = df[col].mode()

        if len(mode_value) > 0:
            df[col] = df[col].fillna(mode_value[0])
        else:
            df[col] = df[col].fillna("Unknown")


print("\nMissing Values After Handling:")
print(df.isnull().sum())


# ============================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================

# Target column
stress_col = "Overall, how would you rate your current stress level?"

# Convert target to numeric
df[stress_col] = pd.to_numeric(df[stress_col], errors="coerce")

# If target has missing values after conversion, remove those rows
df = df.dropna(subset=[stress_col])


# Convert categorical columns using one-hot encoding
categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

# Timestamp is not used as a model feature
if "Timestamp" in categorical_columns:
    categorical_columns.remove("Timestamp")

df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# ============================================================
# 6. REMOVE TIMESTAMP
# ============================================================

if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[stress_col])
y = df[stress_col]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Distribution:")
print(y.value_counts().sort_index())


# ============================================================
# 8. SAVE MODEL-READY DATASET
# ============================================================

output_folder = "../data/processed"

os.makedirs(output_folder, exist_ok=True)

processed_data = pd.concat([X, y], axis=1)

output_file = os.path.join(
    output_folder,
    "model_ready_stress_data.csv"
)

processed_data.to_csv(output_file, index=False)

print("\nProcessed Dataset Shape:")
print(processed_data.shape)

print("\nSaved File:")
print(output_file)

print("\nPreprocessing completed successfully!")