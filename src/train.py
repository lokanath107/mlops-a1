import json
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ---------------- Paths ----------------
DATA_PATH = "data/processed_heart.csv"
MODEL_DIR = "model"
ARTIFACT_DIR = "artifacts"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ============================================================
# 1. Load data (LOCAL, MLOps-friendly)
# ============================================================

columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# Load data from local file
df = pd.read_csv(DATA_PATH, names=columns)

# Basic cleaning
df = df.replace("?", pd.NA).dropna()
df = df.astype(float)

# Binary target
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

X = df.drop(columns="target")
y = df["target"]

# ============================================================
# 2. Feature groups (same as notebook)
# ============================================================

numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# ============================================================
# 3. Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', 'passthrough', categorical_features)
    ]
)
# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- Model Pipeline ----------------
# ============================================================
# 4. Final Random Forest model
# (use best params from GridSearch)
# ============================================================

rf_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    ))
])
rf_pipeline.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = rf_pipeline.predict(X_test)
y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob)
}

# ---------------- Save Model ----------------
model_path = f"{MODEL_DIR}/model.pkl"
joblib.dump(rf_pipeline, model_path)

# ---------------- Save Metrics ----------------
metrics_path = f"{ARTIFACT_DIR}/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("Training complete. Model and metrics saved.")
