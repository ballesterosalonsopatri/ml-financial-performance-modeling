import pandas as pd
import joblib
import os
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==============================
# 1. Carga del dataset procesado
# ==============================

print("Cargando dataset procesado...")

DATA_PATH = "data/processed/dataset_model_ready.csv"
df = pd.read_csv(DATA_PATH)

# ==============================
# 2. Definición de variables
# ==============================

TARGET = "Riesgo_Deterioro"

NON_FEATURES = [
    TARGET,
    "Unidad_ID",
    "Departamento",
    "Periodo_ID"
]

X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns])
y = df[TARGET]

FEATURES = X.columns.tolist()

print(f"✔ Número de features utilizadas: {len(FEATURES)}")

# ==============================
# 3. Train / Test split
# ==============================

print("Realizando train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# 4. Guardar datasets train/test
# ==============================

print("Guardando datasets train/test...")

os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

X_train.to_csv("data/train/X_train.csv", index=False)
y_train.to_csv("data/train/y_train.csv", index=False)
X_test.to_csv("data/test/X_test.csv", index=False)
y_test.to_csv("data/test/y_test.csv", index=False)

# ==============================
# 5. Definición de modelos
# ==============================

print("Definiendo modelos...")

baseline_model = LogisticRegression(
    max_iter=3000,
    random_state=42
)

logreg_balanced = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    ))
])

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])

models = {
    "trained_model_1_logreg_baseline.pkl": baseline_model,
    "trained_model_2_logreg_balanced.pkl": logreg_balanced,
    "trained_model_3_decision_tree.pkl": decision_tree,
    "trained_model_4_random_forest.pkl": random_forest,
    "trained_model_5_svm.pkl": svm_pipe
}

# ==============================
# 6. Entrenamiento y guardado
# ==============================

print("Entrenando y guardando modelos...")

os.makedirs("models", exist_ok=True)

for filename, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{filename}")

print("✔ Modelos individuales guardados.")

# ==============================
# 7. Guardar modelo final
# ==============================

FINAL_MODEL_PATH = "models/final_model.pkl"
joblib.dump(svm_pipe, FINAL_MODEL_PATH)

print("Modelo final (SVM) guardado.")

# ==============================
# 8. Guardar configuración
# ==============================

model_config = {
    "model_name": "SVM",
    "kernel": "rbf",
    "class_weight": "balanced",
    "scaler": "StandardScaler",
    "random_state": 42,
    "train_size": len(X_train),
    "test_size": len(X_test),
    "features": FEATURES
}

with open("models/model_config.yaml", "w") as f:
    yaml.dump(model_config, f)

print("model_config.yaml guardado correctamente.")
print("Entrenamiento finalizado sin errores.")
