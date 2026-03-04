import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==============================
# IMPORTS MODULARES CORRECTOS
# ==============================
from data import generar_datos_sinteticos
from model import entrenar_modelo
from utils import crear_poligono_sabana
from model import entrenar_modelo, calcular_probabilidades

# =====================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================
st.set_page_config(
    layout="wide",
    page_title="Monitoreo Aedes - Sabana Occidente"
)

# =====================================================
# 1️⃣ CARGA DE DATOS
# =====================================================
df = generar_datos_sinteticos()

# =====================================================
# 2️⃣ MODELO IA
# =====================================================
modelo, metricas = entrenar_modelo(df)
df = calcular_probabilidades(modelo, df)
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
# 4️⃣ TABS PRINCIPALES
# =====================================================
tab1, tab2 = st.tabs(["🗺 Mapa Predictivo", "📊 Análisis Exploratorio"])

# =====================================================
# 🗺 TAB 1 — MAPA
# =====================================================
with tab1:

    st.title("🦟 Sistema Predictivo - Aedes aegypti")

    st.markdown(
    f"**Región:** Sabana de Occidente | "
    f"**Accuracy:** `{metricas['accuracy']*100:.1f}%` | "
    f"**Actualización:** `{ultima_fecha}`"
    )

    # Polígono Sabana
    sabana_poly = crear_poligono_sabana()
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

    # Límite Sabana
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

    # Ajuste estético barra cromática
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
            len=0.85,
            thickness=18,
            x=1.0,
            xanchor="left",
            y=0.5,
            yanchor="middle"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 📊 TAB 2 — ANÁLISIS EXPLORATORIO
# =====================================================
with tab2:

    st.subheader("📊 Distribución de Variables")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            px.histogram(df_f, x="Temperatura", nbins=20),
            use_container_width=True
        )

        st.plotly_chart(
            px.histogram(df_f, x="Humedad", nbins=20),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            px.histogram(df_f, x="Precipitacion", nbins=20),
            use_container_width=True
        )

        st.plotly_chart(
            px.histogram(df_f, x="Indice_socioeconomico", nbins=10),
            use_container_width=True
        )

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
    columns=["Temperatura", "Precipitacion", "Humedad", "Indice_socioeconomico"]
)

res_prob = modelo.predict_proba(input_user)[0][1]

if res_prob < 0.4:
    nivel = "BAJA IDONEIDAD ECOLÓGICA"
    descripcion = "Condiciones poco favorables."
    color_lateral = "#2E7D32"
elif res_prob < 0.7:
    nivel = "IDONEIDAD ECOLÓGICA MODERADA"
    descripcion = "Condiciones parcialmente favorables."
    color_lateral = "#F9A825"
else:
    nivel = "ALTA IDONEIDAD ECOLÓGICA"
    descripcion = "Condiciones óptimas."
    color_lateral = "#C62828"

import streamlit.components.v1 as components

components.html(f"""
<div style="
    background-color: #F2F2F2;
    padding: 18px 22px;
    border-radius: 6px;
    border-left: 6px solid {color_lateral};
    margin-top: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    font-family: sans-serif;
">
    <div style="font-size:18px; font-weight:700; color:#333;">
        {nivel}
    </div>

    <div style="font-size:14px; color:#666; margin-top:6px;">
        {descripcion}
    </div>

    <div style="font-size:13px; color:#444; margin-top:10px;">
        Riesgo estimado: <strong>{res_prob*100:.1f}%</strong>
    </div>
</div>
""", height=140)