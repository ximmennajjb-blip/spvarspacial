# SPVAR IPM – Homicidios | Colombia Departamental

Análisis espacial y econométrico de la relación entre la **pobreza multidimensional (IPM)** y la **tasa de homicidios** en los 33 departamentos (incluye Bogotá D.C.) de Colombia, con visualización interactiva en Gradio.

> Script generado originalmente en Google Colab (`app.ipynb`) y exportado como `app.py`.

---

## 1. Fuentes de datos

| Variable | Fuente | Detalle |
|---|---|---|
| IPM | DANE | `anex-PMultidimensional-Departamental-2025.xlsx`, hoja `IPM_Departamentos`, columna *Total* por año |
| Homicidios | MinDefensa (datos.gov.co) | Dataset `m8fd-ahd9`, consultado vía API SODA |
| Población | DANE | Proyecciones CNPV 2018, tabla de respaldo incluida en el código (`POB_DANE`) |
| Geometría | GitHub (GeoJSON) | `caticoa3/colombia_mapa` (principal) y `santiblanko/colombia.geojson` (respaldo) |

**Periodo:** 2018 – 2025 · **Unidad:** departamento (códigos DIVIPOLA de 2 dígitos).

---

## 2. Requisitos

Python 3.9+ y las siguientes librerías (el script las instala automáticamente si faltan):

```
geopandas  libpysal  esda  spreg  folium  mapclassify
openpyxl   xlrd      requests  tqdm  plotly  kaleido
gradio     splot
```

Además usa `numpy`, `pandas` y `matplotlib`.

Requiere **conexión a internet** para descargar el Excel del DANE, los homicidios y la geometría.

---

## 3. Ejecución

En Google Colab:

```python
!python app.py
```

O localmente:

```bash
python app.py
```

> **Nota:** las rutas de trabajo están fijadas en `/content/spvar_col` (entorno Colab). Para ejecutarlo en local, cambia la variable `BASE` en la Fase 1.

Al finalizar, se lanza la interfaz Gradio con `demo.launch(share=True)`, que genera un enlace público temporal.

---

## 4. Estructura del script (fases)

| Fase | Descripción |
|---|---|
| **0 · Instalación** | Verifica e instala dependencias con `pip`. |
| **1 · Configuración** | Imports, carpetas (`data/`, `output/`) y diccionarios `DIVIPOLA` y `NOMBRE_A_COD`. |
| **2 · Geometría** | Descarga el GeoJSON departamental, estandariza `cod_depto` y filtra a los 33 departamentos. |
| **3 · Homicidios** | Descarga por año (2018–2025) desde la API SODA y agrega por departamento-año. |
| **4 · IPM** | Descarga y lee el Excel oficial del DANE. Cada año ocupa 3 columnas (Total, Cabeceras, Centros poblados); se toma *Total*. Incluye auditoría: IPM 2023 ≠ 2024. |
| **5 · Población** | Carga la tabla `POB_DANE` con proyecciones departamentales 2018–2025. |
| **6 · Panel** | Une homicidios + IPM + población, calcula `tasa_homicidio` (por 100 mil hab.) y ejecuta auditorías (duplicados, NaN, alineación de códigos con la geometría). |
| **7 · W + ESDA** | Matriz de vecindad **Queen** estandarizada por filas; **Moran's I global** por año y variable; **LISA** para el último año (HH, LL, HL, LH). |
| **8 · SPVAR + SDM** | Con rezagos temporales (`ipm_l1`, `hom_l1`) estima OLS, SAR, SEM, SAC y SDM para dos ecuaciones (IPM y tasa de homicidio) y selecciona el mejor por **AIC**. |
| **9 · Mapas** | Función `generar_mapa()`: mapa estático (Matplotlib) o 3D (Plotly `scatter_3d`). |
| **10 · Gradio** | Interfaz web con selectores de año, variable, tipo de mapa y departamento. |
| **11 · Resumen** | Imprime fuentes, tamaño del panel, especificaciones elegidas y checklist de auditoría. |

---

## 5. Modelos espaciales

Para cada ecuación (`y` = IPM o tasa de homicidio) con regresores `ipm_l1` y `hom_l1`:

| Modelo | Descripción | Función (`spreg`) |
|---|---|---|
| OLS | Mínimos cuadrados con diagnósticos espaciales | `OLS` |
| SAR | Rezago espacial de la variable dependiente | `ML_Lag` |
| SEM | Error espacial | `ML_Error` |
| SAC | Rezago + error espacial (combinado) | `GM_Combo_Hom` |
| SDM | Durbin espacial: rezago de `y` + rezago espacial de `X` (`WX`) | `ML_Lag` con `X` ampliada |

El modelo ganador es el de **menor AIC**. La estimación usa un corte transversal del último año con rezago disponible.

---

## 6. Interfaz Gradio

**Controles:**

- **Año:** 2018–2025 (según disponibilidad en el panel).
- **Variable:** `IPM`, `Tasa Homicidio` o `Bivariado` (índice ponderado 60 % IPM normalizado + 40 % homicidios normalizados).
- **Tipo de mapa:** `Estático (Matplotlib)` o `3D Dinámico (Plotly)`.
- **Departamento:** `Todos` o uno específico (se resalta en el mapa estático y se muestra ficha con IPM y tasa).

**Salidas:** imagen del mapa, gráfico 3D, mensaje de estado y ficha del departamento.

---

## 7. Auditorías incluidas

- 33 códigos DIVIPOLA verificados.
- IPM 2023 ≠ 2024 ≠ 2025 (medias y por departamento).
- Sin NaN ni duplicados en el panel.
- Códigos del panel alineados con la geometría.
- `n` de `y` coincide con `n` de la matriz `W`.
- Modelo SDM incluido en la comparación.

---

## 8. Limitaciones y observaciones

1. **Población:** la tabla `POB_DANE` está escrita a mano en el código. La descripción menciona "intento de descarga automática", pero el script solo usa la tabla de respaldo. Muchos valores están redondeados, por lo que las tasas son aproximadas; conviene reemplazarla con las proyecciones oficiales del DANE.
2. **Estructura del Excel DANE:** el parser depende de posiciones fijas (filas 14–49, columnas `2 + 3·i`). Si el DANE cambia el formato del anexo, la lectura fallará.
3. **Geometría:** se usan GeoJSON de repositorios de terceros; no son una fuente oficial.
4. **Estimación:** se hace sobre un solo corte transversal (n ≈ 33), lo que limita la potencia estadística. Pese al nombre "SPVAR", no se estima un VAR espacial de panel completo.
5. **Mapa 3D:** usa centroides en coordenadas geográficas (EPSG:4326), lo que puede generar advertencias de GeoPandas; para mayor precisión conviene proyectar antes de calcular centroides.
6. **Entorno:** rutas fijas de Colab (`/content/...`) y `share=True` en Gradio.

---

## 9. Salidas generadas

```
/content/spvar_col/
├── data/
│   └── anex-PMultidimensional-Departamental-2025.xlsx
└── output/
```

Los mapas estáticos se guardan como PNG temporales (`tempfile`).
