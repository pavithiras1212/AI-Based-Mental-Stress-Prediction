import pandas as pd
import numpy as np
file_path = "data/processed/cleaned_student_stress_data.xlsx"
# Load cleaned dataset

df = pd.read_excel(file_path)




# Basic dataset inspection
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))




# DATA CLEANING & STANDARDIZATION


# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace("\n", " ", regex=False)
df.columns = df.columns.str.replace(r"\s+", " ", regex=True)

print("\nCleaned Column Names:")
print(df.columns.tolist())

# Remove extra spaces from text responses
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

print("\nMissing Values After Standardization:")
print(df.isnull().sum())


# HANDLE MISSING VALUES
# -------------------------------

# Fill categorical missing values with the mode
categorical_columns = [
    "Age",
    "Year of Study",
    "What is the main cause of your stress?",
    "Have you experienced any of the following due to stress?",
    "How do you usually cope with stress?"
]

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill missing stress rating with the median
stress_col = "Overall, how would you rate your current stress level?"
df[stress_col] = df[stress_col].fillna(df[stress_col].median())

print("\nMissing Values After Imputation:")
print(df.isnull().sum())


# -------------------------------
# VERIFY CLEANED DATA
# -------------------------------

print("\nDataset Shape After Cleaning:", df.shape)

print("\nDuplicate Rows After Cleaning:", df.duplicated().sum())

print("\nUnique Values in Categorical Columns:")

for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts())