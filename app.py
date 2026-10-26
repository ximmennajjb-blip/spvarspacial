# -*- coding: utf-8 -*-
"""
SpVAR Colombia – Explorador Espacial (Streamlit)

Si existe el archivo `gdf_eda.geojson` junto a este script, lo usa (resultados
reales del notebook). Si no, calcula un DEMO con datos sintéticos para que la
app funcione igualmente.
"""
import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import Fullscreen
import streamlit as st
from streamlit_folium import st_folium
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SpVAR Colombia – Explorador Espacial",
                   page_icon="🗺️", layout="wide")

GEOJSON_LOCAL = "gdf_eda.geojson"
URLS_GEOMETRIA = [
    "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
    "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json",
]
COLUMNAS_REQUERIDAS = ["DEPARTAMENTO", "COD_DPTO", "log_PIB", "tasa_homicidios",
                       "luces_viirs", "LISA_PIB", "LISA_HOM", "Gi_PIB", "Gi_HOM"]


# ----------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------
def normalizar_columnas(gdf):
    """Detecta las columnas de nombre y código del departamento."""
    col_nombre, col_cod = None, None
    for col in gdf.columns:
        c = col.lower()
        if col_nombre is None and any(k in c for k in ["dpto_cnmbr", "nom_dep", "nombre", "name"]):
            col_nombre = col
        elif col_cod is None and any(k in c for k in ["dpto_ccdgo", "cod_dep", "codigo", "cod"]):
            col_cod = col
    out = gdf.copy()
    out["DEPARTAMENTO"] = out[col_nombre].astype(str) if col_nombre else out.index.astype(str)
    out["COD_DPTO"] = (out[col_cod].astype(str).str.zfill(2) if col_cod
                       else [f"{i:02d}" for i in range(1, len(out) + 1)])
    return out[["COD_DPTO", "DEPARTAMENTO", "geometry"]]


def etiquetas_lisa(lisa):
    etiquetas = []
    for q, p in zip(lisa.q, lisa.p_sim):
        if p > 0.05:
            etiquetas.append("No significativo")
        elif q == 1:
            etiquetas.append("High-High (cluster alto)")
        elif q == 2:
            etiquetas.append("Low-High (outlier)")
        elif q == 3:
            etiquetas.append("Low-Low (cluster bajo)")
        else:
            etiquetas.append("High-Low (outlier)")
    return etiquetas


def calcular_demo(gdf):
    """ESDA con datos SINTÉTICOS (solo para demostración)."""
    rng = np.random.default_rng(42)
    gdf = gdf.copy()
    proj = gdf.to_crs(epsg=3116)
    w = KNN.from_dataframe(proj, k=5)
    w.transform = "r"

    def serie_espacial(escala, media):
        base = rng.normal(0, 1, len(gdf))
        for _ in range(2):  # suaviza con vecinos para crear autocorrelación
            base = 0.5 * base + 0.5 * np.array(
                [np.mean(base[w.neighbors[i]]) for i in range(len(gdf))])
        return media + escala * (base - base.mean()) / base.std()

    gdf["log_PIB"] = serie_espacial(1.0, 15.0)
    gdf["tasa_homicidios"] = np.abs(serie_espacial(12.0, 25.0))
    gdf["luces_viirs"] = np.abs(serie_espacial(1.5, 4.0))

    y_pib, y_hom = gdf["log_PIB"].values, gdf["tasa_homicidios"].values
    lisa_pib, lisa_hom = Moran_Local(y_pib, w, seed=42), Moran_Local(y_hom, w, seed=42)
    gdf["LISA_PIB"], gdf["LISA_HOM"] = etiquetas_lisa(lisa_pib), etiquetas_lisa(lisa_hom)
    gdf["Gi_PIB"] = G_Local(y_pib, w, transform="R").Zs
    gdf["Gi_HOM"] = G_Local(y_hom, w, transform="R").Zs
    info = {
        "W": "KNN (k=5)",
        "moran_pib": Moran(y_pib, w).I,
        "moran_hom": Moran(y_hom, w).I,
    }
    return gdf, info


@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos():
    if os.path.exists(GEOJSON_LOCAL):
        g = gpd.read_file(GEOJSON_LOCAL).to_crs(epsg=4326)
        if all(c in g.columns for c in COLUMNAS_REQUERIDAS):
            w_name = "No disponible"
            if os.path.exists("W_name.txt"):
                with open("W_name.txt", encoding="utf-8") as f:
                    w_name = f.read().strip()
            return g, {"W": w_name}, False

    ultimo_error = None
    for url in URLS_GEOMETRIA:
        try:
            base = gpd.read_file(url).to_crs(epsg=4326)
            base = normalizar_columnas(base)
            g, info = calcular_demo(base)
            return g, info, True
        except Exception as e:  # noqa: BLE001
            ultimo_error = e
    raise RuntimeError(f"No se pudo cargar la geometría: {ultimo_error}")


try:
    gdf_eda, info, es_demo = cargar_datos()
except Exception as e:  # noqa: BLE001
    st.error(str(e))
    st.stop()

W_name = info["W"]

# ----------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------
st.title("🗺️ Explorador Espacial – PIB y Homicidios en Colombia")
st.caption("ESDA + SpVAR · Análisis a nivel departamental")
if es_demo:
    st.warning("Modo DEMO: los datos son sintéticos. Sube `gdf_eda.geojson` "
               "(generado en el notebook) al repositorio para ver resultados reales.")

# ----------------------------------------------------------------------
# Controles
# ----------------------------------------------------------------------
NOMBRES = {"tasa_homicidios": "Tasa de Homicidios", "log_PIB": "log(PIB)",
           "luces_viirs": "Luces Nocturnas VIIRS"}

st.sidebar.header("Controles")
variable = st.sidebar.radio("Variable a explorar", list(NOMBRES), index=0,
                            format_func=lambda x: NOMBRES[x])
deptos = sorted(gdf_eda["DEPARTAMENTO"].dropna().unique().tolist())
departamento = st.sidebar.selectbox("Departamento", deptos, index=0)
st.sidebar.markdown("---")
st.sidebar.info(f"Matriz de pesos espaciales: **{W_name}**")

PALETAS = {"log_PIB": "Spectral_r", "luces_viirs": "magma", "tasa_homicidios": "turbo"}
PALETAS_FOLIUM = {"log_PIB": "Spectral", "luces_viirs": "YlOrRd", "tasa_homicidios": "YlOrRd"}


# ----------------------------------------------------------------------
# Mapas
# ----------------------------------------------------------------------
def mapa_estatico(var):
    fig, ax = plt.subplots(figsize=(7, 8))
    gdf_eda.plot(column=var, cmap=PALETAS[var], legend=True, ax=ax,
                 edgecolor="black", linewidth=0.4,
                 legend_kwds={"shrink": 0.65, "label": NOMBRES[var]})
    ax.set_title(f"{NOMBRES[var]} – Alto Espectro", fontsize=13, fontweight="bold")
    ax.axis("off")
    return fig


def mapa_folium(var):
    cols = ["COD_DPTO", "DEPARTAMENTO", "log_PIB", "tasa_homicidios",
            "luces_viirs", "LISA_PIB", "LISA_HOM", "geometry"]
    g = gdf_eda[cols].copy()
    for c in ["log_PIB", "tasa_homicidios", "luces_viirs"]:
        g[c] = g[c].round(3)
    g["COD_DPTO"] = g["COD_DPTO"].astype(str)

    m = folium.Map(location=[4.6, -74.1], zoom_start=5, tiles="CartoDB positron",
                   control_scale=True)
    Fullscreen().add_to(m)
    folium.Choropleth(
        geo_data=g, data=g, columns=["COD_DPTO", var],
        key_on="feature.properties.COD_DPTO",
        fill_color=PALETAS_FOLIUM[var], fill_opacity=0.8, line_opacity=0.4,
        legend_name=NOMBRES[var], highlight=True,
    ).add_to(m)
    folium.GeoJson(
        g,
        style_function=lambda x: {"fillColor": "transparent", "color": "#222", "weight": 0.6},
        tooltip=folium.GeoJsonTooltip(
            fields=["DEPARTAMENTO", "log_PIB", "tasa_homicidios", "luces_viirs",
                    "LISA_PIB", "LISA_HOM"],
            aliases=["Departamento:", "log(PIB):", "Homicidios:", "Luces:",
                     "LISA PIB:", "LISA Hom:"],
            sticky=True,
        ),
    ).add_to(m)
    return m


# ----------------------------------------------------------------------
# Explicación didáctica
# ----------------------------------------------------------------------
def explicacion(dept, var):
    fila = gdf_eda[gdf_eda["DEPARTAMENTO"] == dept]
    if fila.empty:
        return "Departamento no encontrado en los datos."
    r = fila.iloc[0]
    if var == "log_PIB":
        lisa, gi, val = r["LISA_PIB"], r["Gi_PIB"], r["log_PIB"]
    elif var == "luces_viirs":
        lisa, gi, val = "No aplica (variable de control)", 0.0, r["luces_viirs"]
    else:
        lisa, gi, val = r["LISA_HOM"], r["Gi_HOM"], r["tasa_homicidios"]

    texto = f"""
### Diagnóstico espacial de **{dept}**

| Indicador | Valor |
|---|---|
| Variable | {NOMBRES[var]} |
| Valor observado | **{val:.3f}** |
| Clasificación LISA | **{lisa}** |
| Z-score Gi* | **{gi:.2f}** |

#### Interpretación
"""
    s = str(lisa)
    if "High-High" in s:
        texto += f"**{dept}** forma parte de un **cluster High-High**: él y sus vecinos tienen valores altos."
    elif "Low-Low" in s:
        texto += f"**{dept}** pertenece a un **cluster Low-Low**: valores bajos en él y en su vecindario."
    elif "High-Low" in s:
        texto += f"**{dept}** es un **outlier High-Low**: valor alto rodeado de vecinos bajos."
    elif "Low-High" in s:
        texto += f"**{dept}** es un **outlier Low-High**: valor bajo rodeado de vecinos altos."
    else:
        texto += f"No se detectó un patrón espacial localmente significativo para **{dept}**."
    return texto


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Mapa de Alto Espectro")
    fig = mapa_estatico(variable)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with c2:
    st.subheader("Mapa Interactivo")
    st_folium(mapa_folium(variable), width=700, height=520, returned_objects=[])

st.markdown("---")
st.markdown(explicacion(departamento, variable))

if "moran_pib" in info:
    a, b = st.columns(2)
    a.metric("I de Moran – log(PIB)", f"{info['moran_pib']:.3f}")
    b.metric("I de Moran – Homicidios", f"{info['moran_hom']:.3f}")

st.markdown("""
---
**Notas metodológicas**
- LISA (Moran local) y Gi* provienen del ESDA del pipeline SpVAR.
- El mapa interactivo permite zoom, pan y tooltips.
""")
