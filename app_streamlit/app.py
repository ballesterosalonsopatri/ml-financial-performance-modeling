import streamlit as st
import pandas as pd
import joblib
import yaml
from pathlib import Path

# ==============================
# 1. Configuración general
# ==============================

st.set_page_config(
    page_title="Sistema de Alerta Temprana de Rendimiento Financiero",
    layout="centered"
)

st.title("Sistema de Alerta Temprana de Rendimiento Financiero")

st.write(
    """
    Esta herramienta estima la **probabilidad de que una unidad de negocio
    presente señales tempranas de deterioro financiero**, a partir de
    indicadores financieros y operativos.
    """
)

# ==============================
# 2. Carga del modelo y config
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
CONFIG_PATH = BASE_DIR / "models" / "model_config.yaml"

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)

FEATURES = model_config["features"]

# ==============================
# 3. Entrada de datos
# ==============================

st.header("Introduce los indicadores de la unidad de negocio")

def user_input():
    data = {
        "Ingresos": st.number_input("Ingresos", min_value=0.0, value=20000.0),
        "Gastos": st.number_input("Gastos", min_value=0.0, value=15000.0),
        "Activos": st.number_input("Activos", min_value=0.0, value=18000.0),
        "Pasivos": st.number_input("Pasivos", min_value=0.0, value=10000.0),
        "EBIT": st.number_input("EBIT", value=4000.0),

        "ROA": st.number_input("ROA", value=0.05),
        "ROE": st.number_input("ROE", value=0.10),
        "Margen_Explotacion": st.number_input("Margen de Explotación", value=0.20),
        "Endeudamiento": st.number_input("Endeudamiento", value=0.60),
        "Indice_Rentabilidad": st.number_input("Índice de Rentabilidad", value=0.02),

        "Plantilla": st.number_input("Plantilla", min_value=1, value=200),
        "Costes_Fijos": st.number_input("Costes Fijos", value=6000.0),
        "CostesVariables": st.number_input("Costes Variables", value=9000.0),
        "Crecimiento_Ingresos": st.number_input("Crecimiento de Ingresos", value=0.08),

        "Ingresos_por_Empleado": st.number_input("Ingresos por Empleado", value=100.0),
        "Ratio_Gastos_Ingresos": st.number_input("Ratio Gastos / Ingresos", value=0.75),
        "Margen_Operativo": st.number_input("Margen Operativo", value=0.20),
        "Solvencia": st.number_input("Solvencia", value=1.8),
        "Costes_Totales": st.number_input("Costes Totales", value=15000.0),
        "Peso_Costes_Fijos": st.number_input("Peso de Costes Fijos", value=0.4),
    }

    df = pd.DataFrame([data])

    # 🔒 Blindaje de features
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df = df[FEATURES]

    return df


input_df = user_input()

# ==============================
# 4. Predicción
# ==============================

st.markdown("---")

if st.button("📊 Evaluar Riesgo"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ **Riesgo elevado de deterioro financiero**\n\n"
            f"Probabilidad estimada: **{probability:.2%}**"
        )
    else:
        st.success(
            f"✅ **Unidad sin señales relevantes de deterioro**\n\n"
            f"Probabilidad estimada de riesgo: **{probability:.2%}**"
        )

    st.caption(
        "Modelo SVM entrenado como sistema de alerta temprana para la organización."
    )
