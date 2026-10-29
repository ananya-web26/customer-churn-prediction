import pandas as pd

df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print(df.head())
print("Dataset shape:", df.shape)
print(df.columns.tolist())
print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nChurn counts:")
print(df["Churn"].value_counts())


print("Blank TotalCharges values:")
print((df["TotalCharges"].str.strip() == "").sum())

# Convert TotalCharges from text to numbers
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values again
print("\nMissing TotalCharges values:")
print(df["TotalCharges"].isnull().sum())
print("\nCustomers with missing TotalCharges:")
print(
    df[df["TotalCharges"].isnull()][
        ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

# Fill missing TotalCharges with 0 for zero-tenure customers
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Verify that no TotalCharges values are missing
print("\nMissing TotalCharges after cleaning:")
print(df["TotalCharges"].isnull().sum())

# Separate features and target
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)