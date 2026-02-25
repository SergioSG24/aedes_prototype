import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title="Prototipo ML - Aedes aegypti",
    layout="wide"
)

st.title("🦟 Prototipo ML - Aedes aegypti")
st.markdown("Identificación de factores socio-ambientales asociados a la presencia del vector")

# =========================
# GENERACIÓN DE DATOS SIMULADOS
# =========================

@st.cache_data
def generar_datos():
    np.random.seed(42)
    df = pd.DataFrame({
        "Temperatura": np.random.uniform(18, 35, 200),
        "Precipitacion": np.random.uniform(0, 300, 200),
        "Humedad": np.random.uniform(40, 95, 200),
        "Indice_socioeconomico": np.random.uniform(1, 6, 200),
        "Presencia_Aedes": np.random.choice([0,1], 200)
    })
    return df

data = generar_datos()

# =========================
# KPIs SUPERIORES
# =========================

col1, col2, col3 = st.columns(3)

col1.metric("Temperatura Promedio", f"{data['Temperatura'].mean():.2f} °C")
col2.metric("Precipitación Promedio", f"{data['Precipitacion'].mean():.2f} mm")
col3.metric("Tasa Presencia Aedes", f"{data['Presencia_Aedes'].mean()*100:.1f}%")

# =========================
# FILTRO
# =========================

st.sidebar.header("Filtros")

presencia = st.sidebar.selectbox(
    "Filtrar por Presencia Aedes",
    ["Todos", "Presente (1)", "Ausente (0)"]
)

df_filtrado = data.copy()

if presencia == "Presente (1)":
    df_filtrado = df_filtrado[df_filtrado["Presencia_Aedes"] == 1]
elif presencia == "Ausente (0)":
    df_filtrado = df_filtrado[df_filtrado["Presencia_Aedes"] == 0]

# =========================
# VISUALIZACIONES
# =========================

st.subheader("Distribución de Temperatura")

fig_temp = px.histogram(
    df_filtrado,
    x="Temperatura",
    nbins=20,
    title="Distribución de Temperatura"
)

st.plotly_chart(fig_temp, use_container_width=True)


st.subheader("Relación Temperatura vs Humedad")

fig_scatter = px.scatter(
    df_filtrado,
    x="Temperatura",
    y="Humedad",
    color="Presencia_Aedes",
    title="Temperatura vs Humedad",
)

st.plotly_chart(fig_scatter, use_container_width=True)


# =========================
# TABLA DE DATOS
# =========================

st.subheader("Vista previa de datos")
st.dataframe(df_filtrado.head())

# =========================
# ESTADÍSTICAS DESCRIPTIVAS
# =========================

st.subheader("Estadísticas descriptivas")
st.write(df_filtrado.describe())