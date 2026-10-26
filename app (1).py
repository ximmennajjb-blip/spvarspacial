# -*- coding: utf-8 -*-
"""
TALLER 4 – SpVAR COLOMBIA (versión Streamlit)
Datos reales + selección automática OLS / SAR / SEM / SAC / SDM

Ejecutar localmente:  streamlit run app.py
Las dependencias van en requirements.txt (no se instalan desde el código).
"""

import warnings
warnings.filterwarnings("ignore")

from io import BytesIO

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from folium.plugins import Fullscreen
import streamlit as st
import streamlit.components.v1 as components

from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran_Local
from esda.getisord import G_Local
from spreg import OLS, ML_Lag, ML_Error, GM_Combo

st.set_page_config(page_title="SpVAR Colombia", page_icon="🗺️", layout="wide")
plt.style.use("seaborn-v0_8-whitegrid")


# ----------------------------- 1. GEOMETRÍA ----------------------------------
def load_colombia_departments():
    urls = [
        "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
        "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json",
    ]
    gdf = None
    for url in urls:
        try:
            gdf = gpd.read_file(url)
            break
        except Exception:
            continue
    if gdf is None:
        raise RuntimeError("No se pudo cargar la geometría de los departamentos")

    rename = {}
    for c in gdf.columns:
        cl = c.lower()
        if any(x in cl for x in ["nombre", "name", "dpto_cnmbr", "nom_dep"]):
            rename[c] = "DEPARTAMENTO"
        elif any(x in cl for x in ["codigo", "cod", "dpto_ccdgo", "cod_dep"]):
            rename[c] = "COD_DPTO"
    gdf = gdf.rename(columns=rename)
    if "DEPARTAMENTO" not in gdf.columns:
        gdf["DEPARTAMENTO"] = gdf.index.astype(str)
    if "COD_DPTO" not in gdf.columns:
        gdf["COD_DPTO"] = [f"{i:02d}" for i in range(1, len(gdf) + 1)]
    gdf["COD_DPTO"] = gdf["COD_DPTO"].astype(str).str.zfill(2)
    gdf = gdf.to_crs(4326)
    gdf_proj = gdf.to_crs(3116)
    return gdf, gdf_proj


# ----------------------------- 2. DATOS REALES -------------------------------
def load_pib():
    url = "https://www.dane.gov.co/files/operaciones/PIB/anex-PIBDep-TotalDep-2025pr.xlsx"
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    pib_raw = pd.read_excel(BytesIO(resp.content), sheet_name="Cuadro 1", header=None)

    header_row = None
    for i, row in pib_raw.iterrows():
        if any(isinstance(x, str) and "Código Departamento" in str(x) for x in row.values):
            header_row = i
            break
    if header_row is None:
        header_row = 20

    pib = pib_raw.iloc[header_row + 1:].copy()
    pib.columns = pib_raw.iloc[header_row].values
    pib = pib.loc[:, ~pib.columns.duplicated()]
    pib = pib.rename(columns={pib.columns[0]: "COD_DPTO", pib.columns[1]: "DEPARTAMENTO"})
    pib["COD_DPTO"] = pib["COD_DPTO"].astype(str).str.zfill(2)
    pib = pib[pib["COD_DPTO"].str.match(r"^\d{2}$")]

    year_cols = [
        c for c in pib.columns
        if str(c).replace("p", "").replace("r", "").isdigit() or str(c).endswith(("pr", "p"))
    ]
    pib_long = pib.melt(
        id_vars=["COD_DPTO", "DEPARTAMENTO"], value_vars=year_cols,
        var_name="YEAR", value_name="PIB",
    )
    pib_long["YEAR"] = (
        pib_long["YEAR"].astype(str).str.replace("pr", "").str.replace("p", "").astype(int)
    )
    pib_long["PIB"] = pd.to_numeric(pib_long["PIB"], errors="coerce")
    pib_long = pib_long.dropna(subset=["PIB"])
    pib_long["log_PIB"] = np.log(pib_long["PIB"].clip(1))
    return pib_long


def load_homicidios():
    url = "https://www.datos.gov.co/resource/m8fd-ahd9.csv?$limit=500000"
    try:
        df_hom = pd.read_csv(url, low_memory=False)
    except Exception:
        df_hom = pd.read_csv(
            "https://www.datos.gov.co/api/v3/views/m8fd-ahd9/query.csv?$limit=500000",
            low_memory=False,
        )
    df_hom["fecha_hecho"] = pd.to_datetime(df_hom["fecha_hecho"], errors="coerce")
    df_hom["YEAR"] = df_hom["fecha_hecho"].dt.year
    df_hom["COD_DPTO"] = df_hom["cod_depto"].astype(str).str.zfill(2)
    df_hom["cantidad"] = pd.to_numeric(df_hom["cantidad"], errors="coerce").fillna(1)
    return (
        df_hom.groupby(["COD_DPTO", "YEAR"])["cantidad"].sum()
        .reset_index().rename(columns={"cantidad": "homicidios"})
    )


def build_panel(pib_long, hom_agg):
    df = pib_long.merge(hom_agg, on=["COD_DPTO", "YEAR"], how="inner")
    # Variables aproximadas / simuladas (ver README)
    df["poblacion_approx"] = (np.exp(df["log_PIB"]) * 120).clip(50000)
    df["tasa_homicidios"] = (df["homicidios"] / df["poblacion_approx"] * 100000).clip(0, 120)
    rng = np.random.RandomState(42)
    df["luces_viirs"] = (0.55 * df["log_PIB"] + 0.3 * rng.randn(len(df)) + 1.8).clip(0.5, 5)
    df["log_poblacion"] = np.log(df["poblacion_approx"])
    df["urbanizacion"] = (
        0.4
        + 0.15 * (df["log_PIB"] - df["log_PIB"].mean()) / df["log_PIB"].std()
        + 0.05 * rng.randn(len(df))
    ).clip(0.15, 0.95)
    df["densidad"] = df["poblacion_approx"] / 15000
    return df


# ----------------------------- 3. MATRICES W ---------------------------------
def build_weights(gdf_proj):
    W = {}
    W["queen"] = Queen.from_dataframe(gdf_proj, use_index=False)
    W["rook"] = Rook.from_dataframe(gdf_proj, use_index=False)
    for k in [4, 5, 6]:
        W[f"knn_{k}"] = KNN.from_dataframe(gdf_proj, k=k, use_index=False)
    for w in W.values():
        w.transform = "r"
    return W


def choose_best_w(W_candidates, df_cs):
    best_score, W_best, W_name = -np.inf, None, None
    for name, w in W_candidates.items():
        try:
            ols = OLS(
                df_cs["log_PIB"].values.reshape(-1, 1),
                df_cs[["luces_viirs"]].values,
                w=w, spat_diag=True, moran=True,
            )
            score = abs(ols.moran_res[0]) if hasattr(ols, "moran_res") else 0
            if score > best_score:
                best_score, W_best, W_name = score, w, name
        except Exception:
            continue
    if W_best is None:  # respaldo
        W_name = "queen"
        W_best = W_candidates["queen"]
    return W_best, W_name


# ----------------------------- 4. DATOS SpVAR --------------------------------
def prepare_spvar(df, gdf, year, lag=1):
    t = df[df["YEAR"] == year].set_index("COD_DPTO")
    tl = df[df["YEAR"] == year - lag].set_index("COD_DPTO")
    common = t.index.intersection(tl.index)
    data = pd.DataFrame({
        "COD_DPTO": common,
        "log_PIB": t.loc[common, "log_PIB"].values,
        "tasa_homicidios": t.loc[common, "tasa_homicidios"].values,
        "log_PIB_lag": tl.loc[common, "log_PIB"].values,
        "hom_lag": tl.loc[common, "tasa_homicidios"].values,
        "luces_viirs": t.loc[common, "luces_viirs"].values,
        "log_poblacion": t.loc[common, "log_poblacion"].values,
        "urbanizacion": t.loc[common, "urbanizacion"].values,
    })
    return data.set_index("COD_DPTO").reindex(gdf["COD_DPTO"]).dropna().reset_index()


# ----------------------------- 5. SELECCIÓN DE MODELO ------------------------
def _sdm_inputs(w, X, name_x):
    WX = lag_spatial(w, X)
    return np.hstack([X, WX]), name_x + [f"W_{n}" for n in name_x]


def spatial_model_selection(y, X, w, name_y, name_x):
    results = {}

    ols = OLS(y, X, w=w, spat_diag=True, moran=True, name_y=name_y, name_x=name_x)
    results["OLS"] = {
        "aic": ols.aic,
        "lm_lag_p": ols.lm_lag[1] if hasattr(ols, "lm_lag") else 1,
        "lm_error_p": ols.lm_error[1] if hasattr(ols, "lm_error") else 1,
        "rlm_lag_p": ols.rlm_lag[1] if hasattr(ols, "rlm_lag") else 1,
        "rlm_error_p": ols.rlm_error[1] if hasattr(ols, "rlm_error") else 1,
    }

    for key, fn in [("SAR", ML_Lag), ("SEM", ML_Error), ("SAC", GM_Combo)]:
        try:
            m = fn(y, X, w=w, name_y=name_y, name_x=name_x)
            results[key] = {"aic": getattr(m, "aic", np.nan)}
        except Exception as e:
            results[key] = {"error": str(e)}

    try:
        X_sdm, name_sdm = _sdm_inputs(w, X, name_x)
        sdm = ML_Lag(y, X_sdm, w=w, name_y=name_y, name_x=name_sdm)
        results["SDM"] = {"aic": sdm.aic}
    except Exception as e:
        results["SDM"] = {"error": str(e)}

    lm = results["OLS"]
    decision, reason = "OLS", []

    if lm["rlm_lag_p"] < 0.05 and lm["rlm_error_p"] >= 0.05:
        decision = "SAR"
        reason.append("RLM-Lag significativo, RLM-Error no")
    elif lm["rlm_error_p"] < 0.05 and lm["rlm_lag_p"] >= 0.05:
        decision = "SEM"
        reason.append("RLM-Error significativo, RLM-Lag no")
    elif lm["rlm_lag_p"] < 0.05 and lm["rlm_error_p"] < 0.05:
        aics = {
            m: results[m]["aic"] for m in ["SAR", "SEM", "SAC", "SDM"]
            if "aic" in results[m] and not np.isnan(results[m]["aic"])
        }
        if aics:
            decision = min(aics, key=aics.get)
            reason.append(f"Ambos RLM significativos → menor AIC: {decision}")
        else:
            decision = "SAC"
            reason.append("Ambos RLM significativos → SAC por defecto")
    else:
        reason.append("Ningún RLM significativo → OLS")

    if "aic" in results["SDM"]:
        others = [
            results[m]["aic"] for m in ["SAR", "SEM", "SAC", "OLS"]
            if "aic" in results[m] and not np.isnan(results[m]["aic"])
        ]
        best_other = min(others, default=np.inf)
        if results["SDM"]["aic"] < best_other - 2:
            decision = "SDM"
            reason.append("SDM mejora AIC > 2 puntos")

    return decision, results, reason


def estimate_final(y, X, w, model_type, name_y, name_x):
    if model_type == "OLS":
        return OLS(y, X, w=w, spat_diag=True, name_y=name_y, name_x=name_x)
    if model_type == "SAR":
        return ML_Lag(y, X, w=w, name_y=name_y, name_x=name_x)
    if model_type == "SEM":
        return ML_Error(y, X, w=w, name_y=name_y, name_x=name_x)
    if model_type == "SAC":
        return GM_Combo(y, X, w=w, name_y=name_y, name_x=name_x)
    if model_type == "SDM":
        X_sdm, name_sdm = _sdm_inputs(w, X, name_x)
        return ML_Lag(y, X_sdm, w=w, name_y=name_y, name_x=name_sdm)
    raise ValueError(model_type)


# ----------------------------- 6. ESDA ---------------------------------------
def lisa_label(lisa):
    labs = []
    for i in range(len(lisa.q)):
        if lisa.p_sim[i] > 0.05:
            labs.append("No significativo")
        elif lisa.q[i] == 1:
            labs.append("High-High")
        elif lisa.q[i] == 2:
            labs.append("Low-High")
        elif lisa.q[i] == 3:
            labs.append("Low-Low")
        else:
            labs.append("High-Low")
    return labs


# ----------------------------- 7. PIPELINE (con caché) -----------------------
@st.cache_resource(show_spinner=False)
def run_pipeline():
    """Descarga datos y estima todo. Se ejecuta una sola vez por sesión del servidor."""
    gdf, gdf_proj = load_colombia_departments()
    pib_long = load_pib()
    hom_agg = load_homicidios()
    df = build_panel(pib_long, hom_agg)

    W_candidates = build_weights(gdf_proj)
    year_ref = int(df["YEAR"].max())

    df_cs = df[df["YEAR"] == year_ref].merge(gdf[["COD_DPTO"]], on="COD_DPTO")
    df_cs = df_cs.set_index("COD_DPTO").reindex(gdf["COD_DPTO"]).dropna().reset_index()
    W_best, W_name = choose_best_w(W_candidates, df_cs)

    data_spvar = prepare_spvar(df, gdf, year_ref, 1)
    name_x = ["log_PIB_lag", "hom_lag", "luces_viirs", "log_poblacion", "urbanizacion"]
    X = data_spvar[name_x].values
    y_pib = data_spvar["log_PIB"].values.reshape(-1, 1)
    y_hom = data_spvar["tasa_homicidios"].values.reshape(-1, 1)

    dec_pib, res_pib, why_pib = spatial_model_selection(y_pib, X, W_best, "log_PIB", name_x)
    dec_hom, res_hom, why_hom = spatial_model_selection(y_hom, X, W_best, "tasa_homicidios", name_x)

    model_pib = estimate_final(y_pib, X, W_best, dec_pib, "log_PIB", name_x)
    model_hom = estimate_final(y_hom, X, W_best, dec_hom, "tasa_homicidios", name_x)

    lisa_pib = Moran_Local(y_pib.flatten(), W_best)
    lisa_hom = Moran_Local(y_hom.flatten(), W_best)
    gi_hom = G_Local(y_hom.flatten(), W_best, transform="R")

    data_spvar["LISA_PIB"] = lisa_label(lisa_pib)
    data_spvar["LISA_HOM"] = lisa_label(lisa_hom)
    data_spvar["Gi_HOM"] = gi_hom.Zs
    gdf_eda = gdf.merge(data_spvar, on="COD_DPTO", how="left")

    return {
        "gdf_eda": gdf_eda,
        "data_spvar": data_spvar,
        "df": df,
        "year_ref": year_ref,
        "W_name": W_name,
        "dec_pib": dec_pib, "res_pib": res_pib, "why_pib": why_pib,
        "dec_hom": dec_hom, "res_hom": res_hom, "why_hom": why_hom,
        "model_pib": model_pib, "model_hom": model_hom,
    }


# ----------------------------- 8. MAPAS Y TEXTOS -----------------------------
VAR_CONF = {
    "tasa_homicidios": ("turbo", "Tasa de homicidios", "YlOrRd"),
    "log_PIB": ("Spectral_r", "log(PIB)", "Spectral"),
    "luces_viirs": ("magma", "Luces VIIRS (simuladas)", "YlOrRd"),
}


def mapa_estatico(gdf_eda, variable):
    cmap, titulo, _ = VAR_CONF[variable]
    fig, ax = plt.subplots(figsize=(8, 7))
    gdf_eda.plot(
        column=variable, cmap=cmap, legend=True, ax=ax,
        edgecolor="k", linewidth=0.4, legend_kwds={"shrink": 0.65},
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.axis("off")
    return fig


def mapa_folium(gdf_eda, variable):
    gdf_map = gdf_eda.to_crs(4326)
    m = folium.Map(location=[4.6, -74.1], zoom_start=5, tiles="CartoDB positron")
    Fullscreen().add_to(m)
    fill = VAR_CONF[variable][2]
    valid = gdf_map.dropna(subset=[variable])
    folium.Choropleth(
        geo_data=gdf_map, data=valid, columns=["COD_DPTO", variable],
        key_on="feature.properties.COD_DPTO", fill_color=fill,
        fill_opacity=0.82, line_opacity=0.4, legend_name=variable,
        nan_fill_color="lightgrey",
    ).add_to(m)
    folium.GeoJson(
        gdf_map,
        style_function=lambda x: {"fillColor": "transparent", "color": "#222", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=["DEPARTAMENTO", "log_PIB", "tasa_homicidios", "luces_viirs", "LISA_PIB", "LISA_HOM"],
            aliases=["Depto:", "log PIB:", "Homicidios:", "Luces:", "LISA PIB:", "LISA Hom:"],
        ),
    ).add_to(m)
    return m


def explicacion(res, dept, variable):
    gdf_eda = res["gdf_eda"]
    row = gdf_eda[gdf_eda["DEPARTAMENTO"] == dept]
    if row.empty:
        return "Departamento no encontrado."
    row = row.iloc[0]
    if variable == "log_PIB":
        lisa, val, vn = row.get("LISA_PIB", "N/A"), row.get("log_PIB", np.nan), "log(PIB)"
    elif variable == "luces_viirs":
        lisa, val, vn = "No aplica", row.get("luces_viirs", np.nan), "Luces VIIRS"
    else:
        lisa, val, vn = row.get("LISA_HOM", "N/A"), row.get("tasa_homicidios", np.nan), "Tasa de homicidios"

    if pd.isna(val):
        return f"### {dept}\n\nSin datos para este departamento en el año de referencia."

    txt = f"### {dept}\n\n**{vn}** = {val:.3f} | LISA = **{lisa}**\n\n"
    s = str(lisa)
    if "High-High" in s:
        txt += "Clúster **High-High** → conviene una política regional coordinada."
    elif "Low-Low" in s:
        txt += "Clúster **Low-Low** → zona de valores bajos."
    elif "High-Low" in s:
        txt += "Outlier **High-Low** → fenómeno localizado."
    elif "Low-High" in s:
        txt += "Outlier **Low-High**."
    else:
        txt += "Sin patrón espacial localmente significativo."
    txt += f"\n\n*Modelos: PIB = {res['dec_pib']} | Homicidios = {res['dec_hom']} | W = {res['W_name']}*"
    return txt


# ----------------------------- 9. INTERFAZ STREAMLIT -------------------------
st.title("🗺️ SpVAR Colombia")
st.caption("Datos reales + selección automática de modelos (OLS / SAR / SEM / SAC / SDM) + ESDA")

with st.spinner("Descargando datos y estimando modelos (la primera vez tarda un poco)..."):
    try:
        res = run_pipeline()
    except Exception as e:
        st.error(f"No se pudo completar el análisis: {e}")
        st.stop()

gdf_eda = res["gdf_eda"]

with st.sidebar:
    st.header("Controles")
    variable = st.radio(
        "Variable", list(VAR_CONF.keys()),
        format_func=lambda v: VAR_CONF[v][1],
    )
    deptos = sorted(gdf_eda["DEPARTAMENTO"].dropna().unique().tolist())
    departamento = st.selectbox("Departamento", deptos)
    st.divider()
    st.markdown(f"**Año de referencia:** {res['year_ref']}")
    st.markdown(f"**Matriz W:** `{res['W_name']}`")
    st.markdown(f"**Modelo PIB:** `{res['dec_pib']}`")
    st.markdown(f"**Modelo Homicidios:** `{res['dec_hom']}`")

tab_mapas, tab_modelos, tab_datos = st.tabs(["Mapas", "Modelos", "Datos"])

with tab_mapas:
    st.markdown(explicacion(res, departamento, variable))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mapa estático")
        fig = mapa_estatico(gdf_eda, variable)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.subheader("Mapa interactivo")
        m = mapa_folium(gdf_eda, variable)
        components.html(m.get_root().render(), height=520)

with tab_modelos:
    st.info(
        "Las variables `luces_viirs` y `urbanizacion` son **simuladas** y la población es "
        "una **aproximación** a partir del PIB. Interpretar con cautela."
    )
    for titulo, dec, why, res_m, model in [
        ("Ecuación PIB", res["dec_pib"], res["why_pib"], res["res_pib"], res["model_pib"]),
        ("Ecuación Homicidios", res["dec_hom"], res["why_hom"], res["res_hom"], res["model_hom"]),
    ]:
        st.subheader(f"{titulo} → {dec}")
        st.write("**Razón:** " + " | ".join(why))
        ols_d = res_m["OLS"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LM-Lag p", f"{ols_d['lm_lag_p']:.4f}")
        c2.metric("LM-Error p", f"{ols_d['lm_error_p']:.4f}")
        c3.metric("RLM-Lag p", f"{ols_d['rlm_lag_p']:.4f}")
        c4.metric("RLM-Error p", f"{ols_d['rlm_error_p']:.4f}")
        aic_tbl = pd.DataFrame(
            {m_: [v.get("aic", np.nan)] for m_, v in res_m.items()}, index=["AIC"]
        )
        st.dataframe(aic_tbl.style.format("{:.2f}"))
        with st.expander("Ver resumen completo del modelo final"):
            st.code(model.summary)

with tab_datos:
    st.subheader("Datos del corte transversal")
    st.dataframe(res["data_spvar"], use_container_width=True)
    st.download_button(
        "Descargar CSV",
        res["data_spvar"].to_csv(index=False).encode("utf-8"),
        file_name="spvar_colombia.csv",
        mime="text/csv",
    )
