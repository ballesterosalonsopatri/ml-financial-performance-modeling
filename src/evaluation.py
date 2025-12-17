import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# 1. Cargar datos de test
X_test = pd.read_csv("data/test/X_test.csv")
y_test = pd.read_csv("data/test/y_test.csv").values.ravel()

# 2. Cargar modelo final
svm_pipe = joblib.load("models/final_model.pkl")

# 3. Predicciones
y_pred = svm_pipe.predict(X_test)
y_proba = svm_pipe.predict_proba(X_test)[:, 1]

# 4. Métricas
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"ROC AUC: {roc_auc:.4f}")
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# 5. Guardar métricas
metrics = {
    "ROC_AUC": roc_auc,
    "Precision_riesgo": precision,
    "Recall_riesgo": recall,
    "F1_riesgo": f1
}

pd.DataFrame([metrics]).to_csv(
    "models/final_model_metrics.csv",
    index=False
)

