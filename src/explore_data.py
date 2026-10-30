
# Customer Churn Prediction
# Run with: python src/train_model.py

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset shape:", df.shape)
print("\nChurn counts:")
print(df["Churn"].value_counts())


# ==========================================
# 2. CLEAN DATA
# ==========================================

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing values after cleaning:", df.isnull().sum().sum())


# ==========================================
# 3. FEATURES (X) AND TARGET (y)
# ==========================================

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 5. PREPROCESSING + PIPELINE HELPERS
# ==========================================

numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(exclude=["number"]).columns


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate(name, y_true, y_pred, show_matrix=False):
    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label="Yes"),
        "Recall": recall_score(y_true, y_pred, pos_label="Yes"),
        "F1": f1_score(y_true, y_pred, pos_label="Yes"),
    }

    print(f"\n========== {name} ==========")

    for metric, value in scores.items():
        if metric != "Model":
            print(f"{metric}: {value:.3f}")

    if show_matrix:
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

    return scores


# ==========================================
# 6. TRAIN THREE MODELS
# ==========================================

logistic_pipeline = build_pipeline(
    LogisticRegression(max_iter=1000, class_weight="balanced")
)

tree_pipeline = build_pipeline(
    DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
)

forest_pipeline = build_pipeline(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
)

results = []

for name, pipeline in [
    ("Logistic Regression", logistic_pipeline),
    ("Decision Tree", tree_pipeline),
    ("Random Forest", forest_pipeline),
]:
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append(
        evaluate(name, y_test, predictions, show_matrix=True)
    )


# ==========================================
# 7. THRESHOLD TUNING (LOGISTIC REGRESSION)
# ==========================================

yes_index = list(logistic_pipeline.classes_).index("Yes")

probabilities = logistic_pipeline.predict_proba(X_test)[:, yes_index]

threshold = 0.3

threshold_predictions = [
    "Yes" if p >= threshold else "No"
    for p in probabilities
]

results.append(
    evaluate(
        f"Logistic Regression (threshold {threshold})",
        y_test,
        threshold_predictions
    )
)


# ==========================================
# 8. MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame(results).set_index("Model").round(3)

print("\n========== MODEL COMPARISON ==========")
print(comparison)


# ==========================================
# 9. TUNE THE RANDOM FOREST (GRID SEARCH)
# ==========================================

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, 15],
    "model__min_samples_split": [2, 5],
}

recall_scorer = make_scorer(
    recall_score,
    pos_label="Yes"
)

rf_pipeline = build_pipeline(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )
)

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring=recall_scorer,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation recall:", round(grid_search.best_score_, 3))

final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test)

evaluate(
    "Tuned Random Forest (final)",
    y_test,
    final_predictions,
    show_matrix=True
)


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

feature_names = (
    final_model.named_steps["preprocessor"].get_feature_names_out()
)

importance = final_model.named_steps["model"].feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 important features:")
print(feature_df.head(10))

top_features = feature_df.head(10)

os.makedirs("models", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Factors Influencing Customer Churn")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("models/feature_importance.png")
plt.show()


# ==========================================
# 11. SAVE THE FINAL MODEL
# ==========================================

joblib.dump(final_model, "models/churn_model.pkl")

print("\nModel saved to models/churn_model.pkl")
print("Feature importance plot saved to models/feature_importance.png")
