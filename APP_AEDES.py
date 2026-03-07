import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from datetime import datetime
import io


# ══════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════
st.set_page_config(
    layout="wide",
    page_title="Aptitud Climática — Aedes aegypti Colombia",
    page_icon="🦟"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

:root {
    --risk-high:    #c1121f;
    --risk-med:     #e76f51;
    --risk-low:     #2d6a4f;
    --risk-none:    #adb5bd;
    --accent:       #023e8a;
    --bg:           #f4f6f9;
    --white:        #ffffff;
    --dark:         #12151a;
    --border:       #dee2e6;
    --shadow:       0 1px 12px rgba(0,0,0,0.07);
}

.kpi-card {
    background: var(--white);
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
    border-top: 3px solid var(--border);
    margin-bottom: 8px;
}
.kpi-card.red    { border-color: var(--risk-high); }
.kpi-card.orange { border-color: var(--risk-med); }
.kpi-card.green  { border-color: var(--risk-low); }
.kpi-card.blue   { border-color: var(--accent); }
.kpi-card.gray   { border-color: var(--risk-none); }

.kpi-value { font-size: 28px; font-weight: 700; color: var(--dark); line-height: 1.1; }
.kpi-value.red    { color: var(--risk-high); }
.kpi-value.orange { color: var(--risk-med); }
.kpi-value.green  { color: var(--risk-low); }
.kpi-value.blue   { color: var(--accent); }
.kpi-label { font-size: 10px; font-weight: 600; color: #8a94a6;
             letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

.banner {
    background: linear-gradient(135deg, #012a4a 0%, #023e8a 60%, #0077b6 100%);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    color: white;
}
.banner-title { font-size: 22px; font-weight: 700; line-height: 1.2; }
.banner-sub   { font-size: 13px; opacity: 0.78; margin-top: 6px; line-height: 1.5; }
.banner-meta  { font-size: 11px; opacity: 0.6; margin-top: 12px;
                font-family: 'IBM Plex Mono', monospace; }

.disclaimer {
    background: #fff8f0;
    border: 1px solid #e76f51;
    border-left: 4px solid #e76f51;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
    font-size: 12.5px;
    color: #7f3b1a;
    line-height: 1.5;
}

.section-title {
    font-size: 14px; font-weight: 700;
    color: var(--dark);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 2px solid var(--risk-high);
    padding-bottom: 6px;
    margin: 20px 0 14px 0;
    display: inline-block;
}

.scenario-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-left: 8px;
}
.badge-actual    { background: #e3f0eb; color: #2d6a4f; }
.badge-proyectado{ background: #fde8ea; color: #c1121f; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
#  DATOS SINTÉTICOS — ESCALA NACIONAL COLOMBIA
# ══════════════════════════════════════════════════
np.random.seed(42)

# Municipios representativos de Colombia por región y gradiente altitudinal
municipios_colombia = {
    # Costa Caribe — presencia confirmada, baja altitud
    "Barranquilla":  {"dep": "Atlántico",      "alt": 18,   "lat": 10.99, "lon": -74.80, "presencia_base": 0.95},
    "Cartagena":     {"dep": "Bolívar",         "alt": 2,    "lat": 10.39, "lon": -75.48, "presencia_base": 0.95},
    "Santa Marta":   {"dep": "Magdalena",       "alt": 15,   "lat": 11.24, "lon": -74.19, "presencia_base": 0.92},
    "Montería":      {"dep": "Córdoba",         "alt": 18,   "lat":  8.76, "lon": -75.88, "presencia_base": 0.93},
    # Pacífico — presencia confirmada
    "Cali":          {"dep": "Valle del Cauca", "alt": 995,  "lat":  3.43, "lon": -76.53, "presencia_base": 0.90},
    "Buenaventura":  {"dep": "Valle del Cauca", "alt": 7,    "lat":  3.88, "lon": -77.02, "presencia_base": 0.94},
    # Andina baja — presencia confirmada
    "Medellín":      {"dep": "Antioquia",       "alt": 1495, "lat":  6.25, "lon": -75.56, "presencia_base": 0.88},
    "Ibagué":        {"dep": "Tolima",          "alt": 1285, "lat":  4.44, "lon": -75.23, "presencia_base": 0.87},
    "Neiva":         {"dep": "Huila",           "alt": 442,  "lat":  2.94, "lon": -75.29, "presencia_base": 0.91},
    "Armenia":       {"dep": "Quindío",         "alt": 1483, "lat":  4.53, "lon": -75.68, "presencia_base": 0.86},
    # Andina media — zona de transición
    "Pereira":       {"dep": "Risaralda",       "alt": 1411, "lat":  4.81, "lon": -75.69, "presencia_base": 0.80},
    "Manizales":     {"dep": "Caldas",          "alt": 2153, "lat":  5.07, "lon": -75.52, "presencia_base": 0.60},
    "Tunja":         {"dep": "Boyacá",          "alt": 2820, "lat":  5.54, "lon": -73.36, "presencia_base": 0.20},
    "Duitama":       {"dep": "Boyacá",          "alt": 2590, "lat":  5.83, "lon": -73.03, "presencia_base": 0.25},
    # Andina alta — zona de expansión / riesgo creciente
    "Bogotá":        {"dep": "Cundinamarca",    "alt": 2625, "lat":  4.71, "lon": -74.07, "presencia_base": 0.15},
    "Facatativá":    {"dep": "Cundinamarca",    "alt": 2586, "lat":  4.81, "lon": -74.36, "presencia_base": 0.18},
    "Zipaquirá":     {"dep": "Cundinamarca",    "alt": 2652, "lat":  5.02, "lon": -74.00, "presencia_base": 0.12},
    "Chiquinquirá":  {"dep": "Boyacá",          "alt": 2566, "lat":  5.62, "lon": -73.82, "presencia_base": 0.14},
    # Orinoquía / Amazonía — presencia confirmada
    "Villavicencio": {"dep": "Meta",            "alt": 467,  "lat":  4.14, "lon": -73.64, "presencia_base": 0.92},
    "Florencia":     {"dep": "Caquetá",         "alt": 242,  "lat":  1.61, "lon": -75.60, "presencia_base": 0.93},
}

anios_historicos = [2019, 2020, 2021, 2022, 2023]
rows = []

for _ in range(1200):
    mun   = np.random.choice(list(municipios_colombia.keys()))
    info  = municipios_colombia[mun]
    anio  = np.random.choice(anios_historicos)

    alt   = float(info["alt"] + np.random.uniform(-80, 80))
    # Temperatura decrece ~0.6°C / 100m sobre nivel del mar
    temp_base = 30 - (alt / 100) * 0.55
    temp  = float(np.clip(np.random.normal(temp_base, 1.8), 8, 38))
    prec  = float(np.random.uniform(20, 420))
    hum   = float(np.random.uniform(45, 98))

    # Aptitud climática real para el vector
    apt_temp = 1 / (1 + np.exp(-0.35 * (temp - 18)))   # óptimo >18°C
    apt_hum  = (hum - 40) / 60 if hum > 40 else 0
    apt_prec = min(prec / 300, 1.0)
    apt_alt  = max(0, 1 - (alt - 1800) / 1200) if alt > 1800 else 1.0

    apt_score = 0.35*apt_temp + 0.25*apt_hum + 0.20*apt_prec + 0.20*apt_alt

    # Presencia: función de aptitud + base del municipio
    p_presencia = 0.6 * apt_score + 0.4 * info["presencia_base"]
    presencia   = int(np.random.binomial(1, min(max(p_presencia, 0.02), 0.98)))

    # Casos de arbovirus (validación externa) — correlacionados con aptitud
    casos_base  = max(0, int(np.random.poisson(apt_score * 180 * presencia)))

    rows.append([
        anio, info["dep"], mun,
        round(info["lat"] + np.random.uniform(-0.05, 0.05), 4),
        round(info["lon"] + np.random.uniform(-0.05, 0.05), 4),
        round(temp, 1), round(prec, 1), round(hum, 1),
        round(alt, 0), presencia, casos_base,
        round(apt_score, 4)
    ])

df_hist = pd.DataFrame(rows, columns=[
    "Año", "Departamento", "Municipio", "Latitud", "Longitud",
    "Temperatura", "Precipitacion", "Humedad",
    "Altitud_msnm", "Presencia_Vector", "Casos_Arbovirus",
    "Aptitud_Real"
])

# ── Generar filas proyectadas para años futuros ───
DELTA_ANIO = {2030: 0.8, 2035: 1.2, 2040: 1.8}
proj_rows = []
base_2023 = df_hist[df_hist["Año"] == 2023].copy()

for anio_fut, delta in DELTA_ANIO.items():
    base = base_2023.copy()
    base["Año"]         = anio_fut
    base["Temperatura"] = (base["Temperatura"] + delta).round(1)
    # Presencia_Vector y Casos_Arbovirus se recalcularán tras entrenar el modelo
    proj_rows.append(base)

df_fut = pd.concat(proj_rows, ignore_index=True)
df = pd.concat([df_hist, df_fut], ignore_index=True)

df["Zona"] = pd.cut(
    df["Altitud_msnm"],
    bins=[0, 1000, 1800, 2300, 5000],
    labels=["< 1.000 m", "1.000–1.800 m", "1.800–2.300 m", "> 2.300 m"]
)

FEATURES = ["Temperatura", "Precipitacion", "Humedad", "Altitud_msnm"]
ETIQ = {
    "Temperatura":  "Temperatura (°C)",
    "Precipitacion":"Precipitación (mm)",
    "Humedad":      "Humedad (%)",
    "Altitud_msnm": "Altitud (m.s.n.m.)"
}

X = df[FEATURES].astype(float)
y = df["Presencia_Vector"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

@st.cache_resource
def train_models(Xtr, ytr):
    return {
        "Random Forest":     RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr),
        "Regresión Logística": LogisticRegression(max_iter=1000).fit(Xtr, ytr),
    }

modelos = train_models(X_tr, y_tr)

def metricas(m, Xt, yt):
    yp = m.predict(Xt)
    return yp, accuracy_score(yt,yp), roc_auc_score(yt, m.predict_proba(Xt)[:,1]), confusion_matrix(yt,yp)

# ══════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════
st.sidebar.title("⚙️ Panel de Control")

st.sidebar.markdown("### 🤖 Modelo")
mod_nombre = st.sidebar.selectbox("Algoritmo:", list(modelos.keys()))
modelo = modelos[mod_nombre]
y_pred, acc, auc, cm_v = metricas(modelo, X_te, y_te)

df["Probabilidad"] = modelo.predict_proba(X)[:,1]
df["Nivel_Riesgo"] = pd.cut(
    df["Probabilidad"],
    bins=[0, 0.35, 0.65, 1.0],
    labels=["Bajo", "Moderado", "Alto"]
)

st.sidebar.markdown(f"""
<div style="background:#f0f4ff;border-radius:8px;padding:10px 12px;font-size:12.5px;">
  <b>Exactitud:</b> {acc*100:.1f}% &nbsp;|&nbsp; <b>AUC-ROC:</b> {auc:.3f}
</div>
""", unsafe_allow_html=True)
st.sidebar.divider()

# ── Filtros ──────────────────────────────────────
st.sidebar.markdown("### 🔍 Filtros")
dep_sel  = st.sidebar.selectbox("Departamento", ["Todos"] + sorted(df["Departamento"].unique()))
mun_sel  = st.sidebar.selectbox("Municipio",    ["Todos"] + sorted(df["Municipio"].unique()))

ANIO_LABELS = {
    2019: "2019 (histórico)", 2020: "2020 (histórico)",
    2021: "2021 (histórico)", 2022: "2022 (histórico)",
    2023: "2023 (histórico — línea base)",
    2030: "2030 (proyección +0.8°C)",
    2035: "2035 (proyección +1.2°C)",
    2040: "2040 (proyección +1.8°C)",
}
anios_disponibles = sorted(df["Año"].unique().tolist())
anio_opciones     = ["Todos"] + [ANIO_LABELS[a] for a in anios_disponibles]
anio_sel_label    = st.sidebar.selectbox("Año / Escenario", anio_opciones)
anio_sel_val      = None
if anio_sel_label != "Todos":
    anio_sel_val = anios_disponibles[anio_opciones.index(anio_sel_label) - 1]

is_proj   = anio_sel_val in DELTA_ANIO if anio_sel_val else False
delta_temp = DELTA_ANIO.get(anio_sel_val, 0.0)

riesgo_sel = st.sidebar.multiselect(
    "Nivel de aptitud",
    ["Bajo", "Moderado", "Alto"],
    default=["Bajo", "Moderado", "Alto"]
)

df_viz = df.copy()
if dep_sel       != "Todos": df_viz = df_viz[df_viz["Departamento"] == dep_sel]
if mun_sel       != "Todos": df_viz = df_viz[df_viz["Municipio"]    == mun_sel]
if anio_sel_val  is not None: df_viz = df_viz[df_viz["Año"]         == anio_sel_val]
if riesgo_sel:               df_viz = df_viz[df_viz["Nivel_Riesgo"].isin(riesgo_sel)]

# ══════════════════════════════════════════════════
#  ENCABEZADO
# ══════════════════════════════════════════════════
is_proj = anio_sel_val in DELTA_ANIO if anio_sel_val else False
escenario_label = ANIO_LABELS.get(anio_sel_val, "Todos los años") if anio_sel_val else "Todos los años"
badge_html = (
    f'<span class="scenario-badge badge-proyectado">▲ {escenario_label}</span>'
    if is_proj else
    f'<span class="scenario-badge badge-actual">● {escenario_label}</span>'
)

st.markdown(f"""
<div class="banner">
  <div class="banner-title">🦟 Aptitud Climática para el Establecimiento de <i>Aedes aegypti</i>{badge_html}</div>
  <div class="banner-sub">
    Modelo predictivo a escala municipal · Colombia · Usuario: Instituto Nacional de Salud<br>
    Variables predictoras: temperatura, precipitación, humedad relativa · 
    Validación: casos de arbovirus por municipio
  </div>
  <div class="banner-meta">
    Modelo activo: {mod_nombre} · Registros en vista: {len(df_viz):,} de {len(df):,} ·
    Actualización: {datetime.now().strftime("%d/%m/%Y %H:%M")}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Prototipo con datos sintéticos.</strong> Los registros son generados
  computacionalmente con fines académicos y no representan vigilancia entomológica real.
  Para implementación oficial integrar con fuentes del <strong>SIVIGILA / INS</strong>
  y datos de presencia vectorial del <strong>Grupo de Entomología del INS</strong>.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
#  KPIs
# ══════════════════════════════════════════════════
n_mun_alto = int((df_viz["Nivel_Riesgo"] == "Alto").groupby(df_viz["Municipio"]).any().sum())
n_mun_trans = int(
    df_viz[
        (df_viz["Nivel_Riesgo"] == "Alto") &
        (df_viz["Presencia_Vector"] == 0)
    ]["Municipio"].nunique()
)
prob_media = df_viz["Probabilidad"].mean() * 100 if len(df_viz) else 0
n_mun_vis  = int(df_viz["Municipio"].nunique())

def kpi(valor, label, color):
    return f"""<div class="kpi-card {color}">
        <div class="kpi-value {color}">{valor}</div>
        <div class="kpi-label">{label}</div>
    </div>"""

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(kpi(n_mun_vis,   "Municipios en vista", "blue"),   unsafe_allow_html=True)
with c2: st.markdown(kpi(n_mun_alto,  "Alta aptitud climática", "red"),  unsafe_allow_html=True)
with c3: st.markdown(kpi(n_mun_trans, "En expansión sin presencia", "orange"), unsafe_allow_html=True)
with c4: st.markdown(kpi(f"{prob_media:.1f}%", "Aptitud media", "green"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
#  MAPA PRINCIPAL
# ══════════════════════════════════════════════════
st.markdown('<div class="section-title">Mapa de Aptitud Climática Municipal</div>', unsafe_allow_html=True)

COLOR_SCALE = ["#2d6a4f", "#95d5b2", "#ffd166", "#e76f51", "#c1121f"]

col_map, col_meta = st.columns([3, 1])

with col_map:
    if len(df_viz) == 0:
        st.warning("Sin registros para los filtros seleccionados.")
    else:
        fig_map = px.scatter_mapbox(
            df_viz,
            lat="Latitud", lon="Longitud",
            color="Probabilidad",
            color_continuous_scale=COLOR_SCALE,
            range_color=[0, 1],
            size="Probabilidad",
            size_max=16,
            hover_name="Municipio",
            hover_data={
                "Departamento":    True,
                "Temperatura":     ":.1f",
                "Humedad":         ":.1f",
                "Precipitacion":   ":.1f",
                "Altitud_msnm":    ":.0f",
                "Probabilidad":    ":.2f",
                "Casos_Arbovirus": True,
                "Latitud":  False,
                "Longitud": False
            },
            mapbox_style="open-street-map",
            zoom=4.5,
            center=dict(lat=4.5, lon=-74.0),
            height=560
        )
        fig_map.update_layout(
            margin=dict(r=0, t=0, l=0, b=0),
            coloraxis_colorbar=dict(
                title=dict(text="Aptitud<br>Climática", side="top", font=dict(size=11)),
                tickvals=[0, 0.35, 0.65, 1],
                ticktext=["Baja", "Moderada", "Alta", "Muy alta"],
                len=0.75, thickness=14,
                x=1.0, xanchor="left"
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

with col_meta:
    st.markdown("**Leyenda de niveles**")
    for nivel, color, desc in [
        ("🔴 Alta",     "#fde8ea", "Aptitud ≥ 65% — prioridad máxima de vigilancia"),
        ("🟡 Moderada", "#fff8f0", "Aptitud 35–65% — monitoreo preventivo"),
        ("🟢 Baja",     "#e3f0eb", "Aptitud < 35% — riesgo bajo actual"),
    ]:
        st.markdown(
            f'<div style="background:{color};border-radius:8px;padding:10px 12px;'
            f'margin-bottom:8px;font-size:12.5px;line-height:1.5;">'
            f'<b>{nivel}</b><br>{desc}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**Comparación de escenarios**")
    alto_actual = int((df[df["Año"] == 2023]["Nivel_Riesgo"] == "Alto").sum())
    alto_2040   = int((df[df["Año"] == 2040]["Nivel_Riesgo"] == "Alto").sum())
    delta_pct = ((alto_2040 - alto_actual) / max(alto_actual, 1)) * 100

    st.markdown(
        f'<div style="background:#f4f6f9;border-radius:8px;padding:12px 14px;font-size:12.5px;">'
        f'<b>2023 (base):</b> {alto_actual} registros alta aptitud<br>'
        f'<b>2040 (proyección):</b> {alto_2040} registros<br>'
        f'<b>Variación:</b> '
        f'<span style="color:{"#c1121f" if delta_pct>0 else "#2d6a4f"};font-weight:700;">'
        f'{"▲" if delta_pct>0 else "▼"} {abs(delta_pct):.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════
#  PESTAÑAS
# ══════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Evolución Temporal",
    "🔬 Validación Epidemiológica",
    "🧠 Diagnóstico del Modelo",
    "📋 Tabla Municipal"
])

# ── TAB 1: Evolución temporal ──────────────────
with tab1:
    st.markdown("#### Aptitud climática media por municipio y año")
    mun_disponibles = sorted(df_viz["Municipio"].unique()) if len(df_viz) > 0 else []
    defaults_trend  = [m for m in ["Bogotá", "Manizales", "Tunja", "Medellín", "Barranquilla"] if m in mun_disponibles]
    mun_trend = st.multiselect("Municipios:", mun_disponibles, default=defaults_trend)
    if mun_trend:
        trend = (
            df_viz[df_viz["Municipio"].isin(mun_trend)]
            .groupby(["Año","Municipio"], as_index=False)["Probabilidad"].mean()
        )
        trend["Aptitud (%)"] = (trend["Probabilidad"]*100).round(1)

        fig_t = px.line(
            trend, x="Año", y="Aptitud (%)", color="Municipio",
            markers=True, height=420,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_t.update_traces(line_width=2.5, marker_size=8)
        fig_t.add_hline(y=65, line_dash="dot", line_color="#c1121f",
                        annotation_text="Umbral alto (65%)", annotation_font_size=10)
        fig_t.add_hline(y=35, line_dash="dot", line_color="#e76f51",
                        annotation_text="Umbral moderado (35%)", annotation_font_size=10)
        fig_t.update_layout(
            yaxis_range=[0,100], plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0"), xaxis_showgrid=False,
            legend=dict(orientation="h", y=-0.22),
            xaxis=dict(tickmode="array", tickvals=anios_historicos)
        )
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("Selecciona al menos un municipio.")

# ── TAB 2: Validación epidemiológica ──────────
with tab2:
    st.markdown("#### Correlación entre aptitud climática modelada y casos de arbovirus reportados")
    st.caption(
        "Esta pestaña representa la validación externa del modelo: "
        "los municipios con mayor aptitud climática predicha deben coincidir "
        "con los que históricamente reportan mayor transmisión de arbovirus."
    )

    val_df = (
        df_viz.groupby("Municipio", as_index=False).agg(
            Aptitud_Media=("Probabilidad",    "mean"),
            Casos_Total  =("Casos_Arbovirus", "sum"),
            Altitud      =("Altitud_msnm",    "mean"),
            Presencia_Pct=("Presencia_Vector","mean"),
            Departamento =("Departamento",    "first")
        )
    )
    val_df["Aptitud (%)"]    = (val_df["Aptitud_Media"] * 100).round(1)
    val_df["Presencia (%)"]  = (val_df["Presencia_Pct"] * 100).round(1)

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_v = px.scatter(
            val_df, x="Aptitud (%)", y="Casos_Total",
            color="Departamento", size="Altitud",
            hover_name="Municipio",
            labels={"Casos_Total": "Casos de arbovirus (total histórico)"},
            height=420,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Aptitud climática vs. casos reportados por municipio"
        )
        # Línea de tendencia
        _x = val_df["Aptitud (%)"].values
        _y = val_df["Casos_Total"].values
        _c = np.polyfit(_x, _y, 1)
        _xl = np.linspace(_x.min(), _x.max(), 100)
        fig_v.add_trace(go.Scatter(
            x=_xl, y=np.polyval(_c, _xl), mode="lines",
            name="Tendencia", line=dict(color="#12151a", width=2, dash="dash")
        ))
        fig_v.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            yaxis=dict(gridcolor="#f0f0f0"), xaxis_showgrid=False)
        st.plotly_chart(fig_v, use_container_width=True)

    with col_v2:
        fig_bar = px.bar(
            val_df.sort_values("Casos_Total", ascending=False).head(12),
            x="Municipio", y="Casos_Total", color="Aptitud (%)",
            color_continuous_scale=COLOR_SCALE,
            range_color=[0,100],
            labels={"Casos_Total": "Casos de arbovirus"},
            title="Top 12 municipios por carga de arbovirus",
            height=420
        )
        fig_bar.update_layout(
            xaxis_tickangle=-35, plot_bgcolor="white",
            paper_bgcolor="white", yaxis=dict(gridcolor="#f0f0f0"),
            xaxis_showgrid=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Municipios de alerta: alta aptitud, sin presencia confirmada
    alert_df = val_df[
        (val_df["Aptitud (%)"] >= 50) &
        (val_df["Presencia (%)"] < 30)
    ].sort_values("Aptitud (%)", ascending=False)

    if len(alert_df) > 0:
        st.markdown("##### ⚠️ Municipios prioritarios: alta aptitud climática sin presencia confirmada del vector")
        st.caption("Estos municipios concentran el mayor valor anticipatorio del modelo para el INS.")
        st.dataframe(
            alert_df[["Municipio","Departamento","Aptitud (%)","Presencia (%)","Casos_Total","Altitud"]]
            .rename(columns={
                "Casos_Total": "Casos históricos",
                "Altitud": "Altitud (m)"
            }),
            use_container_width=True, hide_index=True
        )

# ── TAB 3: Diagnóstico del modelo ─────────────
with tab3:
    st.markdown(f"#### Diagnóstico — {mod_nombre}")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**Importancia de variables predictoras**")
        if mod_nombre == "Regresión Logística":
            vals  = modelo.coef_[0]
            label = "Coeficiente"
            cap   = "Coeficientes: valores positivos favorecen la presencia del vector."
        else:
            vals  = modelo.feature_importances_
            label = "Importancia relativa"
            cap   = "Mayor valor = variable más determinante en la predicción."

        imp_df = pd.DataFrame({
            "Variable": [ETIQ[f] for f in FEATURES],
            label: vals
        }).sort_values(label)

        fig_imp = px.bar(
            imp_df, x=label, y="Variable", orientation="h",
            color=label, color_continuous_scale=["#2d6a4f","#ffd166","#c1121f"],
            height=300
        )
        fig_imp.update_layout(
            showlegend=False, coloraxis_showscale=False,
            yaxis_title="", plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"), yaxis_showgrid=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(cap)

    with col_d2:
        st.markdown("**Matriz de confusión**")
        fig_cm = px.imshow(
            cm_v,
            labels=dict(x="Predicho", y="Real", color="Conteo"),
            x=["Ausente", "Presente"], y=["Ausente", "Presente"],
            color_continuous_scale="Blues",
            text_auto=True, height=300
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("**Comparativa de algoritmos**")
    comp = []
    for nm, m in modelos.items():
        _, ac, au, _ = metricas(m, X_te, y_te)
        comp.append({"Algoritmo": nm, "Exactitud (%)": round(ac*100,1),
                     "AUC-ROC": round(au,3), "Activo": "✅" if nm==mod_nombre else ""})
    st.dataframe(pd.DataFrame(comp).sort_values("AUC-ROC", ascending=False),
                 use_container_width=True, hide_index=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Exactitud",   f"{acc*100:.1f}%")
    mc2.metric("AUC-ROC",     f"{auc:.3f}")
    mc3.metric("Variables predictoras", str(len(FEATURES)))

# ── TAB 4: Tabla municipal ─────────────────────
with tab4:
    st.markdown("#### Resumen de aptitud climática por municipio")
    if len(df_viz) == 0:
        st.warning("Sin datos para los filtros seleccionados.")
    else:
        res = df_viz.groupby(["Departamento","Municipio"], as_index=False).agg(
            Aptitud_Media   =("Probabilidad",    "mean"),
            Aptitud_Max     =("Probabilidad",    "max"),
            Altitud_Media   =("Altitud_msnm",    "mean"),
            Temp_Media      =("Temperatura",     "mean"),
            Casos_Total     =("Casos_Arbovirus", "sum"),
            Presencia_Pct   =("Presencia_Vector","mean"),
        )
        res["Aptitud Media (%)"] = (res["Aptitud_Media"]*100).round(1)
        res["Aptitud Máx (%)"]   = (res["Aptitud_Max"]*100).round(1)
        res["Presencia Vector (%)"] = (res["Presencia_Pct"]*100).round(1)
        res["Altitud (m)"]       = res["Altitud_Media"].round(0).astype(int)
        res["Temp. Media (°C)"]  = res["Temp_Media"].round(1)

        def nivel(p):
            if p >= 65:  return "🔴 Alto"
            elif p >= 35: return "🟡 Moderado"
            else:         return "🟢 Bajo"

        res["Nivel"] = res["Aptitud Media (%)"].apply(nivel)
        res = res.sort_values("Aptitud Media (%)", ascending=False)

        st.dataframe(
            res[["Departamento","Municipio","Nivel","Aptitud Media (%)","Aptitud Máx (%)",
                 "Presencia Vector (%)","Casos_Total","Altitud (m)","Temp. Media (°C)"]]
            .rename(columns={"Casos_Total":"Casos Arbovirus"}),
            use_container_width=True, hide_index=True
        )

        csv_buf = io.StringIO()
        res.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Exportar CSV",
            csv_buf.getvalue(),
            f"aptitud_aedes_colombia_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

# ══════════════════════════════════════════════════
#  TABLA DE MUNICIPIOS PRIORIZADOS
# ══════════════════════════════════════════════════
st.divider()
st.markdown(
    '<div class="section-title">Municipios Priorizados para Vigilancia Entomológica</div>',
    unsafe_allow_html=True
)
st.caption(
    "Municipios ordenados por urgencia de intervención según el modelo: "
    "combina aptitud climática actual, incremento proyectado y ausencia de presencia confirmada del vector. "
    "Esta tabla es el producto operativo central para la toma de decisiones del INS."
)

if len(df_viz) == 0:
    st.warning("Sin registros para los filtros seleccionados.")
else:
    # ── Calcular aptitud actual y proyectada ──────────
    prio_actual = df_viz.groupby(["Departamento","Municipio"], as_index=False).agg(
        Aptitud_Actual  =("Probabilidad",    "mean"),
        Presencia_Pct   =("Presencia_Vector","mean"),
        Altitud         =("Altitud_msnm",    "mean"),
        Temp_Media      =("Temperatura",     "mean"),
        Casos_Total     =("Casos_Arbovirus", "sum"),
    )
    X_2040 = df_viz[FEATURES].copy()
    X_2040["Temperatura"] = X_2040["Temperatura"] + 1.8
    df_temp = df_viz.copy()
    df_temp["Prob_2040"] = modelo.predict_proba(X_2040)[:,1]
    prio_2040 = df_temp.groupby("Municipio", as_index=False)["Prob_2040"].mean()
    prio = prio_actual.merge(prio_2040, on="Municipio", how="left")
    prio["Prob_2040"] = prio["Prob_2040"].fillna(prio["Aptitud_Actual"])

    prio["Aptitud Actual (%)"]   = (prio["Aptitud_Actual"] * 100).round(1)
    prio["Aptitud 2040 (%)"]     = (prio["Prob_2040"]      * 100).round(1)
    prio["Incremento (pp)"]      = (prio["Aptitud 2040 (%)"] - prio["Aptitud Actual (%)"]).round(1)
    prio["Presencia Vector (%)"] = (prio["Presencia_Pct"]  * 100).round(1)
    prio["Altitud (m)"]          = prio["Altitud"].round(0).astype(int)
    prio["Temp. Media (°C)"]     = prio["Temp_Media"].round(1)

    def nivel_p(p):
        if p >= 65:   return "🔴 Alto"
        elif p >= 35: return "🟡 Moderado"
        else:         return "🟢 Bajo"

    def urgencia(row):
        score = row["Aptitud Actual (%)"] * 0.5 + row["Incremento (pp)"] * 2.0
        if row["Presencia Vector (%)"] < 30:
            score *= 1.4
        return round(score, 1)

    prio["Nivel Actual"]    = prio["Aptitud Actual (%)"].apply(nivel_p)
    prio["Nivel 2040"]      = prio["Aptitud 2040 (%)"].apply(nivel_p)
    prio["Índice Urgencia"] = prio.apply(urgencia, axis=1)
    prio["Sin presencia"]   = prio["Presencia Vector (%)"].apply(
        lambda x: "⚠️ Sin confirmar" if x < 30 else "✅ Confirmada"
    )
    prio = prio.sort_values("Índice Urgencia", ascending=False)

    # ── Filtros locales ───────────────────────────────
    col_f1, col_f2 = st.columns([2, 2])
    with col_f1:
        umbral_pres = st.checkbox(
            "Mostrar solo municipios sin presencia confirmada del vector",
            value=True,
            help="Filtra municipios donde la presencia aún no ha sido documentada — mayor valor anticipatorio para el INS."
        )
    with col_f2:
        top_n = st.slider("Número de municipios a mostrar:", 5, 20, 10)

    prio_viz = prio[prio["Presencia Vector (%)"] < 30].head(top_n) if umbral_pres else prio.head(top_n)

    if len(prio_viz) == 0:
        st.info("No hay municipios que cumplan los criterios de filtro seleccionados.")
    else:
        # ── Tabla principal ───────────────────────────
        st.dataframe(
            prio_viz[[
                "Departamento", "Municipio", "Sin presencia",
                "Nivel Actual", "Aptitud Actual (%)",
                "Nivel 2040",   "Aptitud 2040 (%)",
                "Incremento (pp)", "Índice Urgencia",
                "Altitud (m)", "Temp. Media (°C)"
            ]],
            use_container_width=True,
            hide_index=True
        )

        # ── Gráfico comparativo actual vs 2040 ────────
        fig_prio = go.Figure()
        fig_prio.add_trace(go.Bar(
            name="Aptitud actual",
            x=prio_viz["Municipio"],
            y=prio_viz["Aptitud Actual (%)"],
            marker_color="#2d6a4f", opacity=0.85
        ))
        fig_prio.add_trace(go.Bar(
            name="Aptitud proyectada (2040)",
            x=prio_viz["Municipio"],
            y=prio_viz["Aptitud 2040 (%)"],
            marker_color="#c1121f", opacity=0.85
        ))
        fig_prio.add_hline(y=65, line_dash="dot", line_color="#c1121f",
                           annotation_text="Umbral alto (65%)", annotation_font_size=10)
        fig_prio.add_hline(y=35, line_dash="dot", line_color="#e76f51",
                           annotation_text="Umbral moderado (35%)", annotation_font_size=10)
        fig_prio.update_layout(
            barmode="group",
            title="Aptitud climática: condiciones actuales vs. proyección 2040",
            yaxis_title="Aptitud climática (%)",
            yaxis_range=[0, 100],
            xaxis_tickangle=-30,
            legend=dict(orientation="h", y=-0.25),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0"),
            xaxis_showgrid=False,
            height=400
        )
        st.plotly_chart(fig_prio, use_container_width=True)

        # ── Exportar ──────────────────────────────────
        csv_prio = io.StringIO()
        prio.to_csv(csv_prio, index=False)
        st.download_button(
            "📥 Exportar tabla completa de priorización (CSV)",
            csv_prio.getvalue(),
            f"municipios_priorizados_aedes_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

# ══════════════════════════════════════════════════
#  PIE
# ══════════════════════════════════════════════════
st.divider()
st.caption(
    f"Prototipo académico — datos sintéticos · "
    f"Variables: temperatura, precipitación, humedad relativa, altitud · "
    f"Validación externa: casos de arbovirus por municipio (SIVIGILA/INS) · "
    f"Modelo: {mod_nombre} · AUC-ROC: {auc:.3f} · "
    f"© {datetime.now().year}"
)
