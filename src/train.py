import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1. Load data (LOCAL, MLOps-friendly)
# ============================================================

columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# Load data from local file
df = pd.read_csv("data/processed_heart.csv", names=columns)

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

# ============================================================
# 5. Train model
# ============================================================

rf_pipeline.fit(X, y)

# ============================================================
# 6. Save model
# ============================================================

joblib.dump(rf_pipeline, "model/model.pkl")

print("Random Forest model trained and saved to model/model.pkl")
