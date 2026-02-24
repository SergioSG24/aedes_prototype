import streamlit as st
import pandas as pd
import numpy as np

st.title("🦟 Prototipo ML - Aedes aegypti")
st.write("Identificación de factores socio-ambientales")

# Datos simulados
np.random.seed(42)
data = pd.DataFrame({
    "Temperatura": np.random.uniform(18, 35, 100),
    "Precipitacion": np.random.uniform(0, 300, 100),
    "Humedad": np.random.uniform(40, 95, 100),
    "Indice_socioeconomico": np.random.uniform(1, 6, 100),
    "Presencia_Aedes": np.random.choice([0,1], 100)
})

st.subheader("Vista previa de datos simulados")
st.dataframe(data.head())

st.subheader("Estadísticas")
st.write(data.describe())