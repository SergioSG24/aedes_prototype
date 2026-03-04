import pandas as pd


def agregar_por_municipio(df):
    return df.groupby("Municipio")["Probabilidad"] \
             .mean().reset_index()


def agregar_por_anio(df):
    return df.groupby("Año")["Probabilidad"] \
             .mean().reset_index()


def convertir_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def crear_poligono_sabana():
    return [
        (4.95, -74.45),
        (4.95, -74.05),
        (4.60, -74.05),
        (4.60, -74.45),
        (4.95, -74.45)
    ]