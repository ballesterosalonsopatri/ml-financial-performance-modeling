import pandas as pd
import numpy as np


def load_data(path):
    return pd.read_csv(path)


def clean_and_impute(df):
    # Ratios financieros → mediana por departamento
    ratio_cols = [
        "ROA", "ROE", "Margen_Explotacion",
        "Endeudamiento", "Indice_Rentabilidad"
    ]

    df[ratio_cols] = df.groupby("Departamento")[ratio_cols].transform(
        lambda x: x.fillna(x.median())
    )

    # Variables operativas → mediana global
    operational_cols = [
        "Plantilla", "Costes_Fijos",
        "CostesVariables", "Crecimiento_Ingresos"
    ]

    df[operational_cols] = df[operational_cols].fillna(
        df[operational_cols].median()
    )

    return df


def feature_engineering(df):
    df["Ingresos_por_Empleado"] = df["Ingresos"] / df["Plantilla"]
    df["Ratio_Gastos_Ingresos"] = df["Gastos"] / df["Ingresos"]
    df["Margen_Operativo"] = df["EBIT"] / df["Ingresos"]
    df["Solvencia"] = df["Activos"] / df["Pasivos"]

    df["Costes_Totales"] = df["Costes_Fijos"] + df["CostesVariables"]
    df["Peso_Costes_Fijos"] = df["Costes_Fijos"] / df["Costes_Totales"]

    return df


def create_target(df):
    umbral_margen = df["Margen_Explotacion"].quantile(0.25)
    umbral_rent = df["Indice_Rentabilidad"].quantile(0.25)
    umbral_end = df["Endeudamiento"].quantile(0.75)

    df["Riesgo_Deterioro"] = (
        (df["Margen_Explotacion"] <= umbral_margen) &
        (df["Indice_Rentabilidad"] <= umbral_rent) &
        (df["Endeudamiento"] >= umbral_end)
    ).astype(int)

    return df


def prepare_dataset(df):
    cols_to_drop = ["Unidad_ID", "Departamento"]
    df_model = df.drop(columns=cols_to_drop)
    return df_model


def main():
    df = load_data("data/raw/dataset.csv")

    df = clean_and_impute(df)
    df = feature_engineering(df)
    df = create_target(df)

    df_model = prepare_dataset(df)

    df_model.to_csv(
        "data/processed/dataset_model_ready.csv",
        index=False
    )


if __name__ == "__main__":
    main()
