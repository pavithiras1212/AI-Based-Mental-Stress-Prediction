import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
file_path = "data/processed/cleaned_student_stress_data.xlsx"
df = pd.read_excel(file_path)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Stress level distribution
stress_col = "Overall, how would you rate your current stress level?"

plt.figure()
df[stress_col].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Stress Level")
plt.ylabel("Number of Students")
plt.title("Distribution of Student Stress Levels")
plt.tight_layout()
plt.show()

# Stress level vs year of study
year_col = "Year of Study"

pd.crosstab(df[year_col], df[stress_col]).plot(kind="bar")
plt.xlabel("Year of Study")
plt.ylabel("Number of Students")
plt.title("Stress Level by Year of Study")
plt.tight_layout()
plt.show()


pd.crosstab(df[year_col], df[stress_col]).plot(kind="bar")
plt.xlabel("Year of Study")
plt.ylabel("Number of Students")
plt.title("Stress Level by Year of Study")
plt.tight_layout()
plt.show()

# Stress level vs sleep duration
sleep_col = "How many hours do you sleep on average?"

pd.crosstab(df[sleep_col], df[stress_col]).plot(kind="bar")

plt.xlabel("Average Sleep Duration")
plt.ylabel("Number of Students")
plt.title("Stress Level by Sleep Duration")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Stress level vs exercise frequency
exercise_col = "How often do you exercise or do physical activity?"

pd.crosstab(df[exercise_col], df[stress_col]).plot(kind="bar")

plt.xlabel("Exercise Frequency")
plt.ylabel("Number of Students")
plt.title("Stress Level by Exercise Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation between numeric variables
numeric_df = df.select_dtypes(include="number")

print("\nCorrelation Matrix:")
print(numeric_df.corr())

# Stress level vs main cause of stress
cause_col = "What is the main cause of your stress?"

pd.crosstab(df[cause_col], df[stress_col]).plot(kind="bar")

plt.xlabel("Main Cause of Stress")
plt.ylabel("Number of Students")
plt.title("Stress Level by Main Cause of Stress")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Stress level vs stress-related symptoms
symptom_col = "Have you experienced any of the following due to stress?"

pd.crosstab(df[symptom_col], df[stress_col]).plot(kind="bar")

plt.xlabel("Stress-Related Symptoms")
plt.ylabel("Number of Students")
plt.title("Stress Level by Stress-Related Symptoms")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# Descriptive statistics for stress level
print("\nStress Level Descriptive Statistics:")
print(df[stress_col].describe())

print("\nStress Level Frequency:")
print(df[stress_col].value_counts().sort_index())

# Stress level vs coping method
coping_col = "How do you usually cope with stress?"

pd.crosstab(df[coping_col], df[stress_col]).plot(kind="bar")

plt.xlabel("Coping Method")
plt.ylabel("Number of Students")
plt.title("Stress Level by Coping Method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Average stress level by exercise frequency
mean_stress_exercise = df.groupby(exercise_col)[stress_col].mean().sort_values()

print("\nAverage Stress Level by Exercise Frequency:")
print(mean_stress_exercise)

mean_stress_exercise.plot(kind="bar")

plt.xlabel("Exercise Frequency")
plt.ylabel("Average Stress Level")
plt.title("Average Stress Level by Exercise Frequency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Average stress level by year of study
mean_stress_year = df.groupby(year_col)[stress_col].mean()

print("\nAverage Stress Level by Year of Study:")
print(mean_stress_year)

mean_stress_year.plot(kind="bar")

plt.xlabel("Year of Study")
plt.ylabel("Average Stress Level")
plt.title("Average Stress Level by Year of Study")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Average stress level by sleep duration
mean_stress_sleep = df.groupby(sleep_col)[stress_col].mean()

print("\nAverage Stress Level by Sleep Duration:")
print(mean_stress_sleep)

mean_stress_sleep.plot(kind="bar")

plt.xlabel("Average Sleep Duration")
plt.ylabel("Average Stress Level")
plt.title("Average Stress Level by Sleep Duration")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Average stress level by age group
mean_stress_age = df.groupby("Age")[stress_col].mean()

print("\nAverage Stress Level by Age Group:")
print(mean_stress_age)

mean_stress_age.plot(kind="bar")

plt.xlabel("Age Group")
plt.ylabel("Average Stress Level")
plt.title("Average Stress Level by Age Group")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Stress level percentage distribution
stress_percentage = (
    df[stress_col]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

print("\nStress Level Percentage Distribution:")
print(stress_percentage)

stress_percentage.plot(kind="bar")

plt.xlabel("Stress Level")
plt.ylabel("Percentage of Students")
plt.title("Percentage Distribution of Stress Levels")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# -----------------------------------------
#Average Stress by Main Cause
# -----------------------------------------

mean_stress_cause = df.groupby(cause_col)[stress_col].mean().sort_values()

print("\nAverage Stress Level by Main Cause:")
print(mean_stress_cause)

mean_stress_cause.plot(kind="bar")

plt.xlabel("Main Cause of Stress")
plt.ylabel("Average Stress Level")
plt.title("Average Stress Level by Main Cause")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# -----------------------------------------
# Average Stress by Symptoms
# -----------------------------------------

mean_stress_symptoms = df.groupby(symptom_col)[stress_col].mean().sort_values()

print("\nAverage Stress Level by Stress-Related Symptoms:")
print(mean_stress_symptoms)

mean_stress_symptoms.plot(kind="bar")

plt.xlabel("Stress-Related Symptoms")
plt.ylabel("Average Stress Level")
plt.title("Average Stress Level by Stress-Related Symptoms")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# -----------------------------------------
#  Average Stress by Coping Method
# -----------------------------------------

mean_stress_coping = df.groupby(coping_col)[stress_col].mean().sort_values()

print("\nAverage Stress Level by Coping Method:")
print(mean_stress_coping)

mean_stress_coping.plot(kind="bar")

plt.xlabel("Coping Method")
plt.ylabel("Average Stress Level")
plt.title("Average Stress Level by Coping Method")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ============================================================
# FINAL ANALYSIS SECTION
# ============================================================

# ------------------------------------------------------------
#  Correlation Heatmap
# ------------------------------------------------------------

import numpy as np

# Select numeric variables
numeric_df = df.select_dtypes(include="number")

print("\nNumeric Variables Used for Correlation:")
print(numeric_df.columns.tolist())

# Correlation matrix
correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Heatmap of Numeric Variables")
plt.tight_layout()
plt.show()

#  Stress Level Distribution by Cause

cause_stress_table = pd.crosstab(
    df[cause_col],
    df[stress_col],
    normalize="index"
) * 100

print("\nStress Level Percentage by Main Cause:")
print(cause_stress_table.round(2))

cause_stress_table.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.xlabel("Main Cause of Stress")
plt.ylabel("Percentage of Students")
plt.title("Stress Level Distribution by Main Cause")
plt.legend(title="Stress Level")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# =/=/----------=-
#Stress Level Distribution by Year
# ------------------------------------------------------------

year_stress_table = pd.crosstab(
    df[year_col],
    df[stress_col],
    normalize="index"
) * 100

print("\nStress Level Percentage by Year of Study:")
print(year_stress_table.round(2))

year_stress_table.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 6)
)

plt.xlabel("Year of Study")
plt.ylabel("Percentage of Students")
plt.title("Stress Level Distribution by Year of Study")
plt.legend(title="Stress Level")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# FINAL DATA SUMMARY


print("\n" + "=" * 60)
print("FINAL EDA SUMMARY")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nAverage Stress Level:")
print(round(df[stress_col].mean(), 2))

print("\nMedian Stress Level:")
print(df[stress_col].median())

print("\nMost Common Stress Level:")
print(df[stress_col].mode()[0])

print("\nStress Level Counts:")
print(df[stress_col].value_counts().sort_index())

print("\nEDA analysis completed successfully.")

