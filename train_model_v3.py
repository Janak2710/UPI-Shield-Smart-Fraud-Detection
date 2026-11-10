import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.metrics import precision_recall_curve

# -------------------------------------------------
# 1. Set project paths
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
(MODEL_DIR / "experimental").mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# 2. Load cleaned dataset
# -------------------------------------------------

print("Loading cleaned UPI transaction dataset...")

data_file = DATA_DIR / "transactions_cleaned.csv"
df = pd.read_csv(data_file)

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

df = df.dropna(subset=["timestamp", "is_fraud"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset shape:", df.shape)

# -------------------------------------------------
# 3. Separate features and target
# -------------------------------------------------

target = "is_fraud"

# Columns excluded to prevent data leakage
excluded_columns = [
    "transaction_id",
    "user_id",
    "receiver_id",
    "timestamp",
    "date",
    "status",
    "is_fraud"
]

X = df.drop(columns=excluded_columns, errors="ignore")
print("\nMODEL INPUT FEATURES:")
print(X.columns.tolist())

print("\nTOTAL INPUT FEATURES:", len(X.columns))
y = df[target].astype(int)

# -------------------------------------------------
# DIAGNOSTIC: Investigate suspicious features
# -------------------------------------------------

suspicious_features = [
    "amount_deviation_score",
    "failed_attempts_last_24h",
    "ip_location_mismatch",
    "new_device_flag"
]

print("\nFRAUD RATE BY SUSPICIOUS FEATURE:")

for feature in suspicious_features:
    print(f"\nFeature: {feature}")

    if feature in df.columns:
        print(
            df.groupby(feature, dropna=False)["is_fraud"]
            .agg(["count", "mean"])
            .to_string()
        )
    else:
        print("Feature not found!")

print("\nCATEGORICAL FEATURE VALUES:")

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns

for feature in categorical_features:
    print(f"\n{feature}:")
    print(df[feature].value_counts(dropna=False).to_string())

print("\nTarget distribution:")
print(y.value_counts())

# -------------------------------------------------
# 4. Split data chronologically
# -------------------------------------------------

total_rows = len(df)

train_end = int(total_rows * 0.70)
validation_end = int(total_rows * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_validation))
print("Testing records:", len(X_test))

# -------------------------------------------------
# 5. Prepare numeric and categorical columns
# -------------------------------------------------

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# -------------------------------------------------
# 6. Define machine learning models
# -------------------------------------------------

models = {
    "Logistic Regression (Balanced)": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
        ),

    "Logistic Regression (Unbalanced)": LogisticRegression(
        class_weight=None,
        max_iter=1000,
        random_state=42
        ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
        )
}

# -------------------------------------------------
# 7. Train and compare models
# -------------------------------------------------

results = {}
best_model_name = None
best_model = None
best_validation_score = -1

print("\nStarting model training...")

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    validation_probabilities = pipeline.predict_proba(
        X_validation
    )[:, 1]

    validation_score = average_precision_score(
        y_validation,
        validation_probabilities
    )

    results[name] = {
        "validation_pr_auc": float(validation_score)
    }

    print("Validation PR-AUC:", round(validation_score, 4))

    if validation_score > best_validation_score:
        best_validation_score = validation_score
        best_model_name = name
        best_model = classifier
        best_pipeline = pipeline
        best_validation_probabilities = validation_probabilities

# -------------------------------------------------
# 8. Use the selected validation-trained model
# -------------------------------------------------

print("\nSelected model:", best_model_name)

final_model = best_pipeline

# -------------------------------------------------
# 9. Evaluate on the separate test dataset
# -------------------------------------------------

print("\nEvaluating final model on test data...")

# Select threshold using validation data only

precision_values, recall_values, thresholds = precision_recall_curve(
    y_validation,
    best_validation_probabilities
)

# Find the highest threshold with at least 50% validation recall
valid_thresholds = thresholds[recall_values[:-1] >= 0.50]

if len(valid_thresholds) == 0:
    raise ValueError("No threshold meets the validation recall target.")

best_threshold = float(valid_thresholds.max())

print("\nSelected validation threshold:", round(best_threshold, 4))

# Apply the selected threshold to the test dataset
test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (test_probabilities >= best_threshold).astype(int)

metrics = {
    "selected_model": best_model_name,
    "selected_threshold": float(best_threshold),
    "validation_recall_target": 0.50,
    "test_accuracy": float(accuracy_score(y_test, test_predictions)),
    "test_precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "test_recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "test_f1_score": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "test_roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
    "test_pr_auc": float(
        average_precision_score(y_test, test_probabilities)
    )
}

print("\nFinal Test Results:")

for metric, value in metrics.items():
    print(f"{metric}: {value}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

# -------------------------------------------------
# 10. Save confusion matrix
# -------------------------------------------------

matrix = confusion_matrix(y_test, test_predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Genuine", "Fraud"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("UPI Fraud Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "confusion_matrix_v3.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# 11. Save trained model and results
# -------------------------------------------------

joblib.dump(
    final_model,
    MODEL_DIR / "experimental" / "upi_fraud_model_v3.joblib"
)

# Save the validation-selected threshold separately
joblib.dump(
    best_threshold,
    MODEL_DIR / "experimental" / "upi_fraud_threshold_v3.joblib"
)

output_results = {
    "validation_results": results,
    "final_test_results": metrics,
    "dataset_note": (
        "The dataset is synthetically generated. "
        "Results do not represent real banking performance."
    )
}

with open(
    REPORT_DIR / "model_results_v3.json",
    "w"
) as file:
    json.dump(output_results, file, indent=4)

with open(
    REPORT_DIR / "classification_report_v3.txt",
    "w"
) as file:
    file.write(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

print("\n--------------------------------")
print("MODEL TRAINING COMPLETED!")
print("--------------------------------")
print("Saved model: models/experimental/upi_fraud_model_v3.joblib")
print("Saved threshold: models/experimental/upi_fraud_threshold_v3.joblib")
print("Saved results: reports/model_results_v3.json")
print("Saved graph: reports/confusion_matrix_v3.png")
print("Saved report: reports/classification_report_v3.txt")
