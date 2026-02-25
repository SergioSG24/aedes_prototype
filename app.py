import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# =====================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================
st.set_page_config(layout="wide", page_title="Monitoreo Aedes - Sabana Occidente")

# =====================================================
# 1️⃣ GENERACIÓN DE DATOS (SABANA DE OCCIDENTE)
# =====================================================
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

for _ in range(600):
    mun = np.random.choice(list(municipios_coords.keys()))
    lat_base, lon_base = municipios_coords[mun]
    
    temp = np.random.uniform(18, 32)
    prec = np.random.uniform(0, 300)
    hum = np.random.uniform(50, 95)
    socio = np.random.uniform(1, 6)
    
    prob_gen = (0.3*(temp/32) + 0.3*(hum/100) + 0.2*(prec/300) + 0.2*(socio/6))
    presencia = 1 if prob_gen > 0.55 else 0
    
    data_list.append([
        np.random.choice(anios), "Cundinamarca", mun,
        lat_base + np.random.uniform(-0.02, 0.02),
        lon_base + np.random.uniform(-0.02, 0.02),
        temp, prec, hum, socio, presencia
    ])

df = pd.DataFrame(data_list, columns=[
    "Año", "Departamento", "Municipio", "Latitud", "Longitud",
    "Temperatura", "Precipitacion", "Humedad", "Indice_socioeconomico", "Presencia_Aedes"
])

# =====================================================
# 2️⃣ MODELO DE INTELIGENCIA ARTIFICIAL
# =====================================================
X = df[["Temperatura", "Precipitacion", "Humedad", "Indice_socioeconomico"]]
y = df["Presencia_Aedes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

acc = accuracy_score(y_test, modelo.predict(X_test))
df["Probabilidad"] = modelo.predict_proba(X)[:, 1]

ultima_fecha = datetime.now().strftime("%d/%m/%Y %H:%M")

# =====================================================
# 3️⃣ SIDEBAR
# =====================================================
st.sidebar.title("🛠 Configuración")

tipo_mapa = st.sidebar.radio(
    "Capa base del mapa:",
    ["Estándar", "Relieve", "Satélite"]
)

st.sidebar.divider()

anio_sel = st.sidebar.selectbox(
    "Filtrar por Año",
    ["Todos"] + sorted(df["Año"].unique().tolist())
)

mun_sel = st.sidebar.selectbox(
    "Filtrar por Municipio",
    ["Todos"] + sorted(df["Municipio"].unique().tolist())
)

df_f = df.copy()

if anio_sel != "Todos":
    df_f = df_f[df_f["Año"] == anio_sel]

if mun_sel != "Todos":
    df_f = df_f[df_f["Municipio"] == mun_sel]

# =====================================================
# 4️⃣ VISUALIZACIÓN PRINCIPAL
# =====================================================
st.title("🦟 Sistema Predictivo - Aedes aegypti")
st.markdown(
    f"**Región:** Sabana de Occidente | "
    f"**Precisión:** `{acc*100:.1f}%` | "
    f"**Actualización:** `{ultima_fecha}`"
)

# Polígono Sabana
sabana_poly = [
    (4.95, -74.45), (4.95, -74.05),
    (4.60, -74.05), (4.60, -74.45),
    (4.95, -74.45)
]

p_lats, p_lons = zip(*sabana_poly)

# Crear mapa
fig = px.scatter_mapbox(
    df_f,
    lat="Latitud",
    lon="Longitud",
    color="Probabilidad",
    color_continuous_scale=["green", "yellow", "red"],
    range_color=[0, 1],
    hover_name="Municipio",
    hover_data={
        "Temperatura": True,
        "Humedad": True,
        "Precipitacion": True,
        "Latitud": False,
        "Longitud": False
    },
    zoom=9.2,
    height=600
)

# Polígono azul
fig.add_trace(go.Scattermapbox(
    lat=p_lats,
    lon=p_lons,
    mode="lines",
    line=dict(width=3, color="blue"),
    name="Límite Sabana Occidente",
    hoverinfo="skip"
))

# Configuración mapa base
if tipo_mapa == "Estándar":
    fig.update_layout(mapbox_style="open-street-map")

elif tipo_mapa == "Relieve":
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[{
            "below": "traces",
            "sourcetype": "raster",
            "source": ["https://a.tile.opentopomap.org/{z}/{x}/{y}.png"],
            "sourceattribution": "© OpenTopoMap"
        }]
    )

else:
    fig.update_layout(
        mapbox_style="white-bg",
        mapbox_layers=[{
            "below": "traces",
            "sourcetype": "raster",
            "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
            "sourceattribution": "Tiles © Esri"
        }]
    )

# =====================================================
# 🎨 AJUSTE ESTÉTICO DE BARRA CROMÁTICA
# =====================================================
fig.update_layout(
    margin=dict(r=0, t=40, l=0, b=0),
    mapbox_center=dict(lat=4.75, lon=-74.23),
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.88,
        xanchor="left",
        x=0.01
    ),
    coloraxis_colorbar=dict(
        title=dict(
            text="Probabilidad<br>Presencia Aedes",
            side="top",
            font=dict(size=13)

        ),
        tickmode="array",
        tickvals=[0, 0.25, 0.5, 0.75, 1],
        ticktext=["0", "0.25", "0.5", "0.75", "1"],
        tickfont=dict(size=11),
        len=0.85,          # tamaño proporcional
        thickness=18,
        x=1.0,            # pegada a la derecha
        xanchor="left",
        y=0.5,             # centrada verticalmente
        yanchor="middle"        
        
    )
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 5️⃣ SIMULADOR DE RIESGO
# =====================================================
st.divider()
st.subheader("🔮 Simulador de Riesgo Local")

c1, c2, c3, c4 = st.columns(4)

s_temp = c1.slider("Temp (°C)", 15, 40, 25)
s_hum = c2.slider("Humedad (%)", 30, 100, 70)
s_prec = c3.slider("Lluvia (mm)", 0, 400, 100)
s_ind = c4.slider("Estrato/Indice", 1.0, 6.0, 3.0)

input_user = pd.DataFrame(
    [[s_temp, s_prec, s_hum, s_ind]],
    columns=X.columns
)

res_prob = modelo.predict_proba(input_user)[0][1]

color_res = (
    "green" if res_prob < 0.4
    else "orange" if res_prob < 0.7
    else "red"
)

st.markdown(f"""
<div style="
    background-color: {color_res};
    padding: 20px;
    border-radius: 10px;
    text-align: center;">
    <h2 style="color: white; margin: 0;">
        Riesgo Estimado: {res_prob*100:.1f}%
    </h2>
</div>
""", unsafe_allow_html=True)