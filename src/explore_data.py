
# Customer Churn Prediction
# Exploratory Data Analysis
# Run with: python src/explore_data.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ==========================================
# 2. UNDERSTAND THE DATA
# ==========================================

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("\nMissing Values After Conversion:")
print(df.isnull().sum())


# ==========================================
# 4. CHURN DISTRIBUTION
# ==========================================

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ==========================================
# 5. NUMERICAL FEATURE ANALYSIS
# ==========================================

numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for column in numeric_columns:

    plt.figure(figsize=(7, 4))

    sns.histplot(
        data=df,
        x=column,
        hue="Churn",
        kde=True,
        bins=30
    )

    plt.title(f"{column} Distribution by Churn")
    plt.tight_layout()
    plt.show()


# ==========================================
# 6. CATEGORICAL FEATURE ANALYSIS
# ==========================================

categorical_columns = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport"
]

for column in categorical_columns:

    plt.figure(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=column,
        hue="Churn"
    )

    plt.title(f"{column} vs Customer Churn")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. CORRELATION ANALYSIS
# ==========================================

# Convert Churn into numeric values for correlation
correlation_df = df.copy()

correlation_df["TotalCharges"] = pd.to_numeric(
    correlation_df["TotalCharges"], errors="coerce"
).fillna(0)

correlation_df["Churn"] = (
    correlation_df["Churn"].map({"Yes": 1, "No": 0})
)

correlation_matrix = correlation_df.select_dtypes(
    include=["number"]
).corr()

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


print("\nExploratory Data Analysis Completed!")
