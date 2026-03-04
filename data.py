# data.py

import pandas as pd
import numpy as np


def generar_datos_sinteticos(n=600):
    np.random.seed(42)

    municipios_coords = {
        "Bojacá": (4.73, -74.34),
        "El Rosal": (4.85, -74.26),
        "Facatativá": (4.81, -74.36),
        "Funza": (4.71, -74.21),
        "Madrid": (4.73, -74.26),
        "Mosquera": (4.71, -74.23),
        "Subachoque": (4.92, -74.17),
        "Zipacón": (4.76, -74.38),
        "Engativá": (4.70, -74.11)
    }

    anios = [2019, 2020, 2021, 2022, 2023]
    data_list = []

    for _ in range(n):
        mun = np.random.choice(list(municipios_coords.keys()))
        lat_base, lon_base = municipios_coords[mun]

        temp = np.random.uniform(18, 32)
        prec = np.random.uniform(0, 300)
        hum = np.random.uniform(50, 95)
        socio = np.random.uniform(1, 6)

        prob_gen = (0.3*(temp/32) + 0.3*(hum/100) +
                    0.2*(prec/300) + 0.2*(socio/6))

        presencia = 1 if prob_gen > 0.55 else 0

        data_list.append([
            np.random.choice(anios), "Cundinamarca", mun,
            lat_base + np.random.uniform(-0.02, 0.02),
            lon_base + np.random.uniform(-0.02, 0.02),
            temp, prec, hum, socio, presencia
        ])

    df = pd.DataFrame(data_list, columns=[
        "Año", "Departamento", "Municipio",
        "Latitud", "Longitud",
        "Temperatura", "Precipitacion",
        "Humedad", "Indice_socioeconomico",
        "Presencia_Aedes"
    ])

    return df


def cargar_csv(file):
    return pd.read_csv(file)