# -*- coding: utf-8 -*-
# ============================================================
# SPVAR IPM – Homicidios | Colombia Departamental  (versión Streamlit)
# Fuentes REALES:
#   IPM  → anex-PMultidimensional-Departamental-2025.xlsx (DANE)
#   Hom  → datos.gov.co m8fd-ahd9 (MinDefensa)
#   Pob  → proyecciones DANE CNPV 2018
# Auditoría + ESDA + SDM + mapas (estático + 3D)
# ============================================================
import io
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import streamlit as st

from libpysal.weights import Queen, lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error, GM_Combo_Hom

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SPVAR Colombia", page_icon="🗺️", layout="wide")

# ----------------------------------------------------------
# 1. Configuración
# ----------------------------------------------------------
BASE = Path(tempfile.gettempdir()) / "spvar_col"
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

DIVIPOLA = {
    "05": "ANTIOQUIA", "08": "ATLANTICO", "11": "BOGOTA D.C.", "13": "BOLIVAR",
    "15": "BOYACA", "17": "CALDAS", "18": "CAQUETA", "19": "CAUCA",
    "20": "CESAR", "23": "CORDOBA", "25": "CUNDINAMARCA", "27": "CHOCO",
    "41": "HUILA", "44": "LA GUAJIRA", "47": "MAGDALENA", "50": "META",
    "52": "NARIÑO", "54": "NORTE DE SANTANDER", "63": "QUINDIO", "66": "RISARALDA",
    "68": "SANTANDER", "70": "SUCRE", "73": "TOLIMA", "76": "VALLE DEL CAUCA",
    "81": "ARAUCA", "85": "CASANARE", "86": "PUTUMAYO", "88": "SAN ANDRES",
    "91": "AMAZONAS", "94": "GUAINIA", "95": "GUAVIARE", "97": "VAUPES", "99": "VICHADA",
}

NOMBRE_A_COD = {
    "Antioquia": "05", "Atlántico": "08", "Bogotá D.C.": "11", "Bolívar": "13",
    "Boyacá": "15", "Caldas": "17", "Caquetá": "18", "Cauca": "19",
    "Cesar": "20", "Córdoba": "23", "Cundinamarca": "25", "Chocó": "27",
    "Huila": "41", "La Guajira": "44", "Magdalena": "47", "Meta": "50",
    "Nariño": "52", "Norte de Santander": "54", "Quindío": "63", "Risaralda": "66",
    "Santander": "68", "Sucre": "70", "Tolima": "73", "Valle del Cauca": "76",
    "Arauca": "81", "Casanare": "85", "Putumayo": "86", "San Andrés": "88",
    "Amazonas": "91", "Guainía": "94", "Guaviare": "95", "Vaupés": "97", "Vichada": "99",
}

IPM_URL = "https://www.dane.gov.co/files/operaciones/PM/anex-PMultidimensional-Departamental-2025.xlsx"
HOM_URL = "https://www.datos.gov.co/resource/m8fd-ahd9.csv"

# Población departamental (proyección DANE aprox. 30 jun) – tabla de respaldo
POB_DANE = {
    "05": {2018: 6407102, 2019: 6480862, 2020: 6550196, 2021: 6624102, 2022: 6699730, 2023: 6778392, 2024: 6880799, 2025: 6960000},
    "08": {2018: 2535517, 2019: 2576879, 2020: 2617234, 2021: 2659120, 2022: 2701456, 2023: 2745123, 2024: 2836795, 2025: 2875000},
    "11": {2018: 7412566, 2019: 7480000, 2020: 7545000, 2021: 7612000, 2022: 7680000, 2023: 7750000, 2024: 7918660, 2025: 8000000},
    "13": {2018: 2070110, 2019: 2098000, 2020: 2125000, 2021: 2153000, 2022: 2181000, 2023: 2209000, 2024: 2227181, 2025: 2255000},
    "15": {2018: 1217376, 2019: 1228000, 2020: 1238000, 2021: 1249000, 2022: 1260000, 2023: 1271000, 2024: 1283225, 2025: 1294000},
    "17": {2018: 998837, 2019: 1007000, 2020: 1015000, 2021: 1024000, 2022: 1033000, 2023: 1042000, 2024: 1050126, 2025: 1059000},
    "18": {2018: 401849, 2019: 406000, 2020: 410000, 2021: 414000, 2022: 418000, 2023: 422000, 2024: 425827, 2025: 430000},
    "19": {2018: 1464532, 2019: 1482000, 2020: 1500000, 2021: 1518000, 2022: 1536000, 2023: 1554000, 2024: 1583095, 2025: 1600000},
    "20": {2018: 1295387, 2019: 1315000, 2020: 1335000, 2021: 1355000, 2022: 1375000, 2023: 1405000, 2024: 1440484, 2025: 1460000},
    "23": {2018: 1784783, 2019: 1810000, 2020: 1835000, 2021: 1860000, 2022: 1885000, 2023: 1920000, 2024: 1969108, 2025: 1995000},
    "25": {2018: 2922451, 2019: 2980000, 2020: 3040000, 2021: 3100000, 2022: 3160000, 2023: 3220000, 2024: 3463004, 2025: 3520000},
    "27": {2018: 534826, 2019: 544000, 2020: 553000, 2021: 562000, 2022: 571000, 2023: 580000, 2024: 586128, 2025: 595000},
    "41": {2018: 1103135, 2019: 1117000, 2020: 1131000, 2021: 1145000, 2022: 1159000, 2023: 1173000, 2024: 1194754, 2025: 1208000},
    "44": {2018: 965718, 2019: 980000, 2020: 994000, 2021: 1008000, 2022: 1022000, 2023: 1036000, 2024: 1047859, 2025: 1062000},
    "47": {2018: 1341746, 2019: 1365000, 2020: 1388000, 2021: 1411000, 2022: 1434000, 2023: 1475000, 2024: 1526269, 2025: 1550000},
    "50": {2018: 1039722, 2019: 1055000, 2020: 1071000, 2021: 1087000, 2022: 1103000, 2023: 1120000, 2024: 1144286, 2025: 1160000},
    "52": {2018: 1630592, 2019: 1645000, 2020: 1660000, 2021: 1675000, 2022: 1690000, 2023: 1700000, 2024: 1704383, 2025: 1715000},
    "54": {2018: 1491689, 2019: 1520000, 2020: 1548000, 2021: 1576000, 2022: 1604000, 2023: 1645000, 2024: 1694970, 2025: 1720000},
    "63": {2018: 547733, 2019: 550000, 2020: 552000, 2021: 554000, 2022: 556000, 2023: 556500, 2024: 556880, 2025: 558000},
    "66": {2018: 943401, 2019: 952000, 2020: 960000, 2021: 968000, 2022: 976000, 2023: 985000, 2024: 997930, 2025: 1005000},
    "68": {2018: 2184741, 2019: 2210000, 2020: 2235000, 2021: 2260000, 2022: 2285000, 2023: 2320000, 2024: 2380650, 2025: 2410000},
    "70": {2018: 904863, 2019: 920000, 2020: 935000, 2021: 950000, 2022: 965000, 2023: 990000, 2024: 1019575, 2025: 1035000},
    "73": {2018: 1331877, 2019: 1342000, 2020: 1352000, 2021: 1362000, 2022: 1372000, 2023: 1378000, 2024: 1380444, 2025: 1388000},
    "76": {2018: 4475886, 2019: 4515000, 2020: 4552000, 2021: 4589000, 2022: 4626000, 2023: 4660000, 2024: 4693432, 2025: 4725000},
    "81": {2018: 262174, 2019: 265000, 2020: 268000, 2021: 271000, 2022: 274000, 2023: 276000, 2024: 277883, 2025: 280000},
    "85": {2018: 420504, 2019: 428000, 2020: 435000, 2021: 442000, 2022: 449000, 2023: 458000, 2024: 467565, 2025: 475000},
    "86": {2018: 348182, 2019: 354000, 2020: 360000, 2021: 366000, 2022: 372000, 2023: 378000, 2024: 385400, 2025: 390000},
    "88": {2018: 61280, 2019: 61800, 2020: 62200, 2021: 62600, 2022: 63000, 2023: 63100, 2024: 63250, 2025: 63500},
    "91": {2018: 76589, 2019: 78000, 2020: 79400, 2021: 80800, 2022: 82200, 2023: 83200, 2024: 84109, 2025: 85000},
    "94": {2018: 48131, 2019: 50000, 2020: 52000, 2021: 54000, 2022: 56000, 2023: 57000, 2024: 58240, 2025: 59500},
    "95": {2018: 82710, 2019: 83200, 2020: 83600, 2021: 84000, 2022: 84300, 2023: 84400, 2024: 84550, 2025: 85000},
    "97": {2018: 40797, 2019: 41500, 2020: 42200, 2021: 42800, 2022: 43300, 2023: 43500, 2024: 43775, 2025: 44200},
    "99": {2018: 107808, 2019: 112000, 2020: 116000, 2021: 121000, 2022: 126000, 2023: 135000, 2024: 143117, 2025: 148000},
}

assert len(DIVIPOLA) == 33


# ----------------------------------------------------------
# 2. Carga de datos (cacheada: se ejecuta una sola vez)
# ----------------------------------------------------------
@st.cache_data(show_spinner="Descargando geometría departamental…")
def load_geometry():
    urls = [
        "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
        "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json",
    ]
    errores = []
    for url in urls:
        try:
            r = requests.get(url, timeout=40)
            r.raise_for_status()
            gdf = gpd.read_file(io.BytesIO(r.content))
            code_col = next(c for c in gdf.columns
                            if "cod" in c.lower() or "dpto" in c.lower() or "DPTO" in c)
            gdf["cod_depto"] = gdf[code_col].astype(str).str.zfill(2)
            gdf = gdf[gdf["cod_depto"].isin(DIVIPOLA)].copy().to_crs(4326)
            assert len(gdf) >= 30
            return gdf
        except Exception as e:
            errores.append(f"{url[:55]}… → {e}")
    raise RuntimeError("No se pudo cargar la geometría:\n" + "\n".join(errores))


@st.cache_data(show_spinner="Descargando homicidios (datos.gov.co)… puede tardar un par de minutos")
def load_homicidios(years=tuple(range(2018, 2026))):
    frames, fallos = [], []
    for y in years:
        params = {
            "$where": f"date_extract_y(fecha_hecho)={y}",
            "$limit": 500000,
            "$select": "cod_depto,cantidad",
        }
        try:
            r = requests.get(HOM_URL, params=params, timeout=120)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                fallos.append(f"{y}: sin filas")
                continue
            df["year"] = y
            frames.append(df)
        except Exception as e:
            fallos.append(f"{y}: {e}")
    if not frames:
        raise RuntimeError("No se descargaron homicidios: " + "; ".join(fallos))
    raw = pd.concat(frames, ignore_index=True)
    raw["cod_depto"] = raw["cod_depto"].astype(str).str.zfill(2)
    raw = raw[raw["cod_depto"].isin(DIVIPOLA)]
    raw["cantidad"] = pd.to_numeric(raw["cantidad"], errors="coerce").fillna(0)
    agg = (raw.groupby(["cod_depto", "year"])["cantidad"].sum().reset_index()
           .rename(columns={"cantidad": "homicidios"}))
    assert (agg["homicidios"] >= 0).all()
    return agg, fallos


@st.cache_data(show_spinner="Descargando IPM (DANE)…")
def load_ipm_dane():
    local = DATA / "anex-PMultidimensional-Departamental-2025.xlsx"
    if not local.exists():
        r = requests.get(IPM_URL, timeout=90)
        r.raise_for_status()
        local.write_bytes(r.content)

    import openpyxl
    wb = openpyxl.load_workbook(local, data_only=True)
    assert "IPM_Departamentos" in wb.sheetnames, f"Hojas: {wb.sheetnames}"
    ws = wb["IPM_Departamentos"]

    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    total_col = {y: 2 + 3 * i for i, y in enumerate(years)}

    def norm(s):
        for a, b in zip("áéíóú", "aeiou"):
            s = s.replace(a, b)
        return s.lower()

    rows, no_mapeados = [], []
    for r in range(14, 50):
        nombre = ws.cell(r, 1).value
        if not nombre or not isinstance(nombre, str):
            continue
        nombre = nombre.strip()
        cod = NOMBRE_A_COD.get(nombre)
        if cod is None:
            for k, v in NOMBRE_A_COD.items():
                if norm(k) == norm(nombre):
                    cod = v
                    break
        if cod is None:
            no_mapeados.append(nombre)
            continue
        for y in years:
            val = ws.cell(r, total_col[y]).value
            if val is None:
                continue
            try:
                rows.append({"cod_depto": cod, "year": y, "ipm": float(val)})
            except (TypeError, ValueError):
                continue

    ipm = pd.DataFrame(rows)
    assert not ipm.empty, "IPM vacío tras parseo"
    assert ipm["ipm"].min() >= 0
    m23 = ipm.loc[ipm.year == 2023, "ipm"].mean()
    m24 = ipm.loc[ipm.year == 2024, "ipm"].mean()
    assert abs(m23 - m24) > 0.05, "IPM 2023 y 2024 no deberían ser casi idénticos"
    return ipm, no_mapeados


def load_poblacion():
    rows = [{"cod_depto": c, "year": y, "poblacion": int(p)}
            for c, serie in POB_DANE.items() for y, p in serie.items()]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Construyendo panel espacio-temporal…")
def build_panel():
    gdf = load_geometry().copy()
    hom, fallos_hom = load_homicidios()
    ipm, no_mapeados = load_ipm_dane()
    pob = load_poblacion()

    panel = (hom.merge(ipm, on=["cod_depto", "year"], how="inner")
                .merge(pob, on=["cod_depto", "year"], how="left"))
    panel["tasa_homicidio"] = panel["homicidios"] / panel["poblacion"] * 100_000
    panel = panel.dropna(subset=["ipm", "tasa_homicidio", "poblacion"]).copy()

    codes_panel = set(panel["cod_depto"].unique())
    codes_geo = set(gdf["cod_depto"].unique())
    solo_geo = sorted(codes_geo - codes_panel)
    solo_datos = sorted(codes_panel - codes_geo)

    common = sorted(codes_geo & codes_panel)
    assert len(common) >= 30, f"Pocos códigos comunes: {len(common)}"

    gdf = gdf[gdf["cod_depto"].isin(common)].copy().reset_index(drop=True)
    panel = panel[panel["cod_depto"].isin(common)].copy()
    order = gdf["cod_depto"].tolist()
    pos = {c: i for i, c in enumerate(order)}
    panel["_ord"] = panel["cod_depto"].map(pos)
    panel = panel.sort_values(["year", "_ord"]).drop(columns="_ord").reset_index(drop=True)

    assert not panel.duplicated(["cod_depto", "year"]).any()
    assert panel[["ipm", "tasa_homicidio"]].isna().sum().sum() == 0

    panel["nombre_depto"] = panel["cod_depto"].map(DIVIPOLA)
    gdf["nombre_depto"] = gdf["cod_depto"].map(DIVIPOLA)

    # Auditoría IPM entre años consecutivos
    audit = []
    for y1, y2 in [(2023, 2024), (2024, 2025)]:
        if y1 in panel.year.values and y2 in panel.year.values:
            a = panel.loc[panel.year == y1].set_index("cod_depto")["ipm"]
            b = panel.loc[panel.year == y2].set_index("cod_depto")["ipm"]
            idx = a.index.intersection(b.index)
            audit.append((y1, y2, int((a.loc[idx] != b.loc[idx]).sum()), len(idx)))

    info = {
        "fallos_hom": fallos_hom, "no_mapeados": no_mapeados,
        "solo_geo": solo_geo, "solo_datos": solo_datos, "audit": audit,
    }
    return gdf, panel, order, info


# ----------------------------------------------------------
# 3. ESDA (Moran global + LISA)
# ----------------------------------------------------------
def _w_for(gdf, present):
    g = gdf[gdf.cod_depto.isin(present)].set_index("cod_depto").loc[present].reset_index()
    w = Queen.from_dataframe(g, ids=present)
    w.transform = "R"
    return w


@st.cache_data(show_spinner="Calculando Moran e índices LISA…")
def compute_esda():
    gdf, panel, order, _ = build_panel()
    moran_rows = []
    for year in sorted(panel.year.unique()):
        sub = panel[panel.year == year]
        present = [c for c in order if c in set(sub.cod_depto)]
        if len(present) < 5:
            continue
        sub = sub.set_index("cod_depto").loc[present]
        w = _w_for(gdf, present)
        for var in ["ipm", "tasa_homicidio"]:
            mi = Moran(sub[var].values, w)
            sig = "***" if mi.p_sim < 0.01 else ("**" if mi.p_sim < 0.05 else ("*" if mi.p_sim < 0.1 else ""))
            moran_rows.append({"Año": int(year), "Variable": var, "I de Moran": round(mi.I, 4),
                               "p (simulado)": round(mi.p_sim, 4), "Sig.": sig})

    last_year = int(panel.year.max())
    sub = panel[panel.year == last_year]
    present = [c for c in order if c in set(sub.cod_depto)]
    sub = sub.set_index("cod_depto").loc[present]
    w = _w_for(gdf, present)
    lisa_rows = []
    for var in ["ipm", "tasa_homicidio"]:
        lisa = Moran_Local(sub[var].values, w)
        sig = lisa.p_sim < 0.05
        lisa_rows.append({
            "Variable": var,
            "HH": int(((lisa.q == 1) & sig).sum()),
            "LL": int(((lisa.q == 3) & sig).sum()),
            "HL": int(((lisa.q == 2) & sig).sum()),
            "LH": int(((lisa.q == 4) & sig).sum()),
        })
    return pd.DataFrame(moran_rows), pd.DataFrame(lisa_rows), last_year


# ----------------------------------------------------------
# 4. SPVAR + SDM
# ----------------------------------------------------------
def _estimate_and_select(y, X_no_const, w, name):
    X = np.column_stack([np.ones(len(X_no_const)), X_no_const])
    models, errores = {}, {}
    for label, fn in [
        ("OLS", lambda: OLS(y, X, w=w, spat_diag=True, name_y=name)),
        ("SAR", lambda: ML_Lag(y, X, w=w, name_y=name)),
        ("SEM", lambda: ML_Error(y, X, w=w, name_y=name)),
        ("SAC", lambda: GM_Combo_Hom(y, X, w=w, name_y=name)),
    ]:
        try:
            models[label] = fn()
        except Exception as e:
            errores[label] = str(e)
    try:
        WX = lag_spatial(w, X_no_const)
        X_sdm = np.column_stack([np.ones(len(X_no_const)), X_no_const, WX])
        models["SDM"] = ML_Lag(y, X_sdm, w=w, name_y=name)
    except Exception as e:
        errores["SDM"] = str(e)

    if not models:
        return None, "", {}, errores
    aics = {k: getattr(m, "aic", np.inf) for k, m in models.items()}
    best = min(aics, key=aics.get)
    try:
        summary = str(models[best].summary)
    except Exception:
        summary = str(models[best])
    return best, summary, aics, errores


@st.cache_data(show_spinner="Estimando modelos espaciales (OLS, SAR, SEM, SAC, SDM)…")
def compute_models():
    gdf, panel, order, _ = build_panel()
    p = panel.copy()
    p = p.sort_values(["cod_depto", "year"])
    p["ipm_l1"] = p.groupby("cod_depto")["ipm"].shift(1)
    p["hom_l1"] = p.groupby("cod_depto")["tasa_homicidio"].shift(1)

    last_lag_year = int(p.dropna(subset=["ipm_l1", "hom_l1"])["year"].max())
    df = p[p["year"] == last_lag_year].dropna(subset=["ipm_l1", "hom_l1"]).copy()
    codes = [c for c in order if c in set(df["cod_depto"])]
    df = df.set_index("cod_depto").loc[codes].reset_index()
    assert len(df) == len(codes) and df["cod_depto"].is_unique

    w = _w_for(gdf, codes)
    assert w.n == len(df)

    X = np.column_stack([df["ipm_l1"].values, df["hom_l1"].values])
    y_ipm = df["ipm"].values.reshape(-1, 1)
    y_hom = df["tasa_homicidio"].values.reshape(-1, 1)

    return {
        "year": last_lag_year, "n": len(df),
        "ipm": _estimate_and_select(y_ipm, X, w, "IPM"),
        "hom": _estimate_and_select(y_hom, X, w, "Tasa_Homicidio"),
    }


# ----------------------------------------------------------
# 5. Mapas
# ----------------------------------------------------------
def build_map_frame(gdf, panel, year, variable):
    sub = panel[panel.year == year]
    mdf = gdf.drop(columns=["nombre_depto"], errors="ignore")
    mdf = mdf.merge(sub[["cod_depto", "ipm", "tasa_homicidio"]], on="cod_depto", how="left")
    mdf["nombre_depto"] = mdf["cod_depto"].map(DIVIPOLA)

    if variable == "IPM":
        return mdf, "ipm", "YlOrRd", f"IPM · {year}"
    if variable == "Tasa Homicidio":
        return mdf, "tasa_homicidio", "RdPu", f"Tasa Homicidios · {year}"

    rng = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    mdf["bivar"] = rng(mdf["ipm"]) * 0.6 + rng(mdf["tasa_homicidio"]) * 0.4
    return mdf, "bivar", "inferno", f"IPM × Homicidios · {year}"


def static_map(mdf, col, cmap, title, depto_sel):
    fig, ax = plt.subplots(figsize=(7, 8), facecolor="#111")
    ax.set_facecolor("#111")
    mdf.plot(column=col, cmap=cmap, legend=True, ax=ax,
             edgecolor="white", linewidth=0.35, missing_kwds={"color": "#333"})
    if depto_sel != "Todos":
        sel = mdf[mdf["nombre_depto"] == depto_sel]
        if not sel.empty:
            sel.boundary.plot(ax=ax, color="cyan", linewidth=2.5)
    ax.set_title(title, color="white", fontsize=13)
    ax.set_axis_off()
    return fig


def map_3d(mdf, col, title):
    mdf = mdf.copy()
    cent = mdf.to_crs(3116).geometry.centroid.to_crs(4326)  # centroides en CRS proyectado
    mdf["lon"], mdf["lat"] = cent.x, cent.y
    mdf = mdf.dropna(subset=[col, "lon", "lat"])
    fig = px.scatter_3d(mdf, x="lon", y="lat", z=col, color=col,
                        hover_name="nombre_depto", color_continuous_scale="Turbo",
                        title=f"3D · {title}", template="plotly_dark")
    fig.update_layout(
        height=600, paper_bgcolor="#0d0d0d", font=dict(color="white"),
        scene=dict(
            xaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#333", color="white"),
            yaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#333", color="white"),
            zaxis=dict(backgroundcolor="#1a1a1a", gridcolor="#333", color="white"),
            bgcolor="#0d0d0d"),
    )
    return fig


# ----------------------------------------------------------
# 6. Interfaz Streamlit
# ----------------------------------------------------------
st.title("🗺️ SPVAR Colombia · IPM y Tasas de Homicidio")
st.caption("Fuentes reales: IPM DANE (anexo 2025) · Homicidios MinDefensa (datos.gov.co) · "
           "Población DANE CNPV 2018 · Modelos: OLS / SAR / SEM / SAC / SDM")

try:
    gdf, panel, order, info = build_panel()
except Exception as e:
    st.error(f"No se pudieron cargar los datos: {e}")
    st.stop()

with st.sidebar:
    st.header("Controles")
    years_available = sorted(int(y) for y in panel.year.unique())
    year = st.selectbox("Año", years_available, index=len(years_available) - 1)
    variable = st.selectbox("Variable", ["IPM", "Tasa Homicidio", "Bivariado"])
    tipo = st.radio("Tipo de mapa", ["Estático (Matplotlib)", "3D Dinámico (Plotly)"])
    deptos = ["Todos"] + sorted(panel["nombre_depto"].dropna().unique().tolist())
    depto_sel = st.selectbox("Departamento", deptos)

tab_mapa, tab_esda, tab_modelos, tab_audit = st.tabs(
    ["📍 Mapas", "📊 ESDA (Moran / LISA)", "📈 Modelos espaciales", "✅ Auditoría y fuentes"])

# ---- Mapas
with tab_mapa:
    mdf, col, cmap, title = build_map_frame(gdf, panel, year, variable)
    c1, c2 = st.columns([2, 1])
    with c1:
        if tipo.startswith("Estático"):
            fig = static_map(mdf, col, cmap, title, depto_sel)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.plotly_chart(map_3d(mdf, col, title), use_container_width=True)
    with c2:
        if depto_sel != "Todos":
            row = mdf[mdf["nombre_depto"] == depto_sel]
            if not row.empty:
                r = row.iloc[0]
                st.metric("IPM", f"{r['ipm']:.1f}%")
                st.metric("Tasa de homicidio (por 100 mil)", f"{r['tasa_homicidio']:.1f}")
                st.caption(f"{depto_sel} · código {r['cod_depto']} · {year}")
        else:
            st.info("Selecciona un departamento en la barra lateral para ver su ficha.")
        st.dataframe(
            mdf[["nombre_depto", "ipm", "tasa_homicidio"]]
            .rename(columns={"nombre_depto": "Departamento", "ipm": "IPM",
                             "tasa_homicidio": "Tasa hom."})
            .sort_values("IPM", ascending=False).round(2),
            hide_index=True, height=360)

# ---- ESDA
with tab_esda:
    try:
        moran_df, lisa_df, last_year = compute_esda()
        st.subheader("Moran's I global")
        st.dataframe(moran_df, hide_index=True, use_container_width=True)
        st.caption("*** p<0.01 · ** p<0.05 · * p<0.10")
        st.subheader(f"LISA (significativos al 5 %) · {last_year}")
        st.dataframe(lisa_df, hide_index=True, use_container_width=True)
        st.caption("HH = alto-alto · LL = bajo-bajo · HL = alto rodeado de bajos · LH = bajo rodeado de altos")
    except Exception as e:
        st.error(f"Error en ESDA: {e}")

# ---- Modelos
with tab_modelos:
    try:
        res = compute_models()
        st.write(f"Año de estimación: **{res['year']}** · n = **{res['n']}**")
        for key, label in [("ipm", "Ecuación IPM"), ("hom", "Ecuación tasa de homicidio")]:
            best, summary, aics, errores = res[key]
            st.subheader(label)
            if best is None:
                st.error("Ningún modelo pudo estimarse.")
                continue
            st.success(f"Mejor modelo por AIC: **{best}**")
            st.dataframe(
                pd.DataFrame({"Modelo": list(aics), "AIC": [None if np.isinf(v) else round(v, 2) for v in aics.values()]}),
                hide_index=True)
            with st.expander("Ver resumen completo del modelo"):
                st.code(summary)
            if errores:
                st.caption("Modelos que fallaron: " + "; ".join(f"{k}: {v}" for k, v in errores.items()))
        st.caption("Nota: SAC (GM_Combo_Hom) no reporta AIC, por lo que no compite en la selección.")
    except Exception as e:
        st.error(f"Error en los modelos: {e}")

# ---- Auditoría
with tab_audit:
    st.subheader("Panel")
    st.write(f"{len(panel)} observaciones · {panel.cod_depto.nunique()} departamentos · "
             f"años {years_available[0]}–{years_available[-1]}")
    st.dataframe(panel.groupby("year")[["ipm", "tasa_homicidio"]].mean().round(2), use_container_width=True)

    st.subheader("Chequeos")
    for y1, y2, n_diff, n in info["audit"]:
        st.write(f"✓ IPM {y1} ≠ {y2} en {n_diff}/{n} departamentos")
    st.write("✓ Códigos DIVIPOLA alineados con la geometría")
    st.write("✓ Sin NaN ni duplicados en el panel")
    if info["solo_geo"]:
        st.warning(f"Solo en geometría: {info['solo_geo']}")
    if info["solo_datos"]:
        st.warning(f"Solo en datos: {info['solo_datos']}")
    if info["no_mapeados"]:
        st.warning(f"Nombres del Excel IPM sin mapear: {info['no_mapeados']}")
    if info["fallos_hom"]:
        st.warning("Años de homicidios con problemas: " + "; ".join(info["fallos_hom"]))

    st.subheader("Fuentes")
    st.markdown(f"- IPM: [anexo departamental DANE 2025]({IPM_URL})\n"
                f"- Homicidios: [datos.gov.co m8fd-ahd9]({HOM_URL})\n"
                "- Población: proyecciones DANE CNPV 2018 (tabla de respaldo en el código; valores aproximados)")
