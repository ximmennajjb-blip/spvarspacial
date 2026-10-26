# SpVAR Colombia: PIB y Homicidios (análisis espacial por departamento)

Documentación basada en `app.py`, exportado desde Google Colab (notebook `app.ipynb`).

## 1. Descripción general

El script implementa un flujo completo de econometría espacial para estudiar la relación entre **actividad económica (log PIB)** y **violencia (tasa de homicidios)** en los departamentos de Colombia. Combina:

- Análisis exploratorio de datos espaciales (ESDA).
- Selección automática de la matriz de pesos espaciales **W**.
- Un sistema tipo **SpVAR** (una ecuación por variable) con selección de modelo espacial.
- Impactos directos, indirectos y totales (LeSage & Pace).
- Un componente satelital con **luces nocturnas VIIRS** como proxy de actividad económica.
- Mapas estáticos, interactivos (Folium), vistas 3D (Plotly) y dos interfaces web (Gradio y Streamlit).

## 2. Fuentes de datos

| Dato | Fuente | Notas |
|------|--------|-------|
| Geometría departamental | GeoJSON de GitHub (`caticoa3/colombia_mapa`, respaldo `santiblanko/colombia.geojson`) | Alternativas: DANE COD-AB (HDX), geoBoundaries |
| PIB departamental | DANE, `anex-PIBDep-TotalDep-2025pr.xlsx` | Hojas "Cuadro 1" (precios corrientes) y "Cuadro 2" (volumen, base 2015) |
| Homicidios | MinDefensa / datos.gov.co, dataset `m8fd-ahd9` | CSV vía SODA con `$limit=500000` |
| Luces nocturnas | VIIRS VNL v2.2 (Colorado School of Mines) | En el script se usa una serie sintética; incluye el código de `zonal_stats` para datos reales |

## 3. Estructura del script (por celdas)

| Celda | Contenido |
|-------|-----------|
| 0 | Instalación de librerías (`geopandas`, `libpysal`, `spreg`, `esda`, `folium`, `rasterio`, etc.) |
| 1 | Imports, semilla aleatoria (42) y configuración de visualización |
| 2 | `load_colombia_departments()`: descarga la geometría, normaliza columnas (`DEPARTAMENTO`, `COD_DPTO`) y genera versiones en EPSG:4326 y EPSG:3116 |
| 3 | Descarga del PIB (DANE) y homicidios (datos.gov.co) |
| 4 | `audit_data()`: nulos, duplicados, cobertura espacial, pruebas ADF y VIF |
| 5 | `build_candidate_weights()`: matrices Queen, Rook, k-NN (k=4,5,6) y distancia inversa; estandarización por filas |
| 6 | `select_best_W()`: elige W según |I de Moran| de residuales OLS y AIC de un SAR |
| 7 | `prepare_spvar_data()`: cross-section 2022 con rezago de 1 año |
| 8 | `spatial_model_selection()`: OLS, SAR, SEM, SAC y SDM; decisión con LM robustos y AIC |
| 9 | `estimate_final_model()`: estima el modelo elegido por ecuación |
| 10 | `compute_impacts()`: impactos directo, indirecto y total |
| 11 | `transmission_channels()`: descomposición en cuatro canales de transmisión |
| 12 | `simulate_or_extract_viirs()`: componente satelital y variable `W_luces_viirs` |
| 13-14 | Mapas coropléticos estáticos y mapa interactivo con Folium |
| 15 | `export_results_tables()`: exporta `coeficientes_spvar.csv` e `impactos_espaciales.csv` |
| 16 | Resumen ejecutivo y recomendaciones |
| Extra 1-2 | Instalación de Gradio/Plotly y `run_spatial_eda()` (Moran global, LISA, Getis-Ord Gi*) |
| Extra 3 | `plot_high_spectrum_maps()`: mapas de alto espectro, LISA y hotspots |
| Extra 4 | `create_3d_satellite_view()`: mapas 3D con Plotly |
| Extra 5-ter | Interfaz Gradio |
| Extra Streamlit | Genera `app_spvar.py`, lo lanza en el puerto 8501 y abre un túnel con ngrok |

## 4. Metodología

### 4.1 Matriz de pesos W
Se construyen varias candidatas y se elige la de mayor |I de Moran| de los residuales OLS, penalizando por AIC del modelo SAR (`score = |I| - 0.01 * (AIC - AIC_min)`).

### 4.2 Selección del modelo espacial
Para cada ecuación (log_PIB y tasa_homicidios):

1. Se estima OLS con diagnósticos espaciales (LM-Lag, LM-Error y sus versiones robustas).
2. Si solo RLM-Lag es significativo → **SAR**. Si solo RLM-Error → **SEM**.
3. Si ambos son significativos → el de menor AIC entre SAR, SEM, SAC y SDM.
4. Si ninguno → **OLS**.
5. Si el SDM mejora el AIC en más de 2 puntos, se prefiere **SDM**.

Variables explicativas: `log_PIB_lag`, `hom_lag`, `luces_viirs`, `log_poblacion`, `urbanizacion`.

### 4.3 Impactos espaciales
Con S = (I - ρW)^-1 y la matriz de efectos parciales S(βI + θW):

- **Directo** = promedio de la diagonal.
- **Total** = promedio de las sumas por fila.
- **Indirecto** = Total - Directo.

### 4.4 Canales de transmisión
1. **Autodinámica espacial**: ρ (Wy) e inercia temporal propia.
2. **Cruce local**: efecto rezagado de una variable sobre la otra en el mismo departamento.
3. **Relación espacial directa**: efecto de los vecinos en la misma variable.
4. **Relación espacial cruzada**: efecto de los vecinos en la otra variable.

### 4.5 ESDA
- **I de Moran global** (inferencia por permutación).
- **LISA**: clasifica en High-High, Low-Low, High-Low, Low-High o No significativo (p < 0.05).
- **Getis-Ord Gi\***: hotspots y coldspots (z-score).

## 5. Salidas generadas

- `coeficientes_spvar.csv`: coeficientes, errores estándar, estadístico t/z y p-valor.
- `impactos_espaciales.csv`: impactos directos, indirectos y totales.
- `gdf_eda.geojson` y `W_name.txt`: insumos para la app Streamlit.
- `app_spvar.py`: aplicación Streamlit.

## 6. Interfaces interactivas

Ambas interfaces permiten elegir la variable (tasa de homicidios, log PIB o luces VIIRS) y un departamento, y muestran un mapa estático, un mapa Folium con tooltips y un diagnóstico LISA/Gi* con interpretación didáctica.

- **Gradio**: botón "Actualizar diagnóstico".
- **Streamlit**: sidebar con controles, dos columnas de mapas y notas metodológicas.

## 7. Cómo ejecutarlo

1. Abrir el notebook en Google Colab.
2. Ejecutar las celdas en orden (celda 0 primero; reiniciar el runtime si hay errores de importación).
3. Para Streamlit, configurar un token de ngrok si el túnel automático falla; alternativas: `npx localtunnel --port 8501` o `output.serve_kernel_port_as_window(8501)`.

> `app.py` contiene comandos `!pip install` propios de Colab, por lo que **no se ejecuta como script de Python normal**.

## 8. Puntos a revisar en el código

- **`df` no está definido.** Las celdas 4, 6, 7 y 12 usan un DataFrame panel `df` (columnas `COD_DPTO`, `YEAR`, `log_PIB`, `tasa_homicidios`, `luces_viirs`, `log_poblacion`, `densidad`, `urbanizacion`), pero la celda 3 solo carga `pib_corrientes`, `pib_volumen` y `df_hom_raw` sin limpiarlos ni unirlos. Falta la celda que construye `df` (probablemente la versión con datos sintéticos que la celda 3 reemplazó).
- **Datos sintéticos.** El resumen ejecutivo y la celda 12 indican que las luces VIIRS (y posiblemente otras variables) son sintéticas. Los resultados no son interpretables como evidencia real hasta usar datos oficiales.
- **Tasa de homicidios.** Requiere población departamental (proyecciones DANE); el script solo lo menciona en notas.
- **Cruce de códigos.** Los códigos de departamento de la geometría se generan secuencialmente si no existen (`01`, `02`, ...), lo que puede no coincidir con los códigos DIVIPOLA del DANE.
- **Muestra pequeña.** Con unos 33 departamentos los estimadores ML y los LM tests tienen poca potencia.
- **Funciones duplicadas.** `crear_mapa_estatico`, `crear_mapa_folium` y `generate_explanation` aparecen tanto en la versión Gradio como en la Streamlit (con comportamientos distintos).

## 9. Dependencias principales

`geopandas`, `libpysal`, `spreg`, `esda`, `mapclassify`, `folium`, `rasterstats`, `rasterio`, `statsmodels`, `seaborn`, `matplotlib`, `pandas`, `numpy`, `scipy`, `shapely`, `pyproj`, `contextily`, `requests`, `tqdm`, `gradio`, `plotly`, `streamlit`, `streamlit-folium`, `pyngrok`.
