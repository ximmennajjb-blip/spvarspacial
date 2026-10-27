# SPVAR IPM – Homicidios | Colombia Departamental

Aplicación **Streamlit** que analiza la relación espacial entre la **pobreza multidimensional (IPM)** y la **tasa de homicidios** en los 33 departamentos de Colombia (incluye Bogotá D.C.), 2018–2025. Incluye análisis exploratorio espacial (Moran, LISA), modelos de econometría espacial (OLS, SAR, SEM, SAC, SDM) y mapas estáticos y 3D.

> Originalmente un notebook de Google Colab con interfaz Gradio; migrado a Streamlit para su despliegue en Streamlit Community Cloud.

---

## 1. Fuentes de datos

| Variable | Fuente | Detalle |
|---|---|---|
| IPM | DANE | `anex-PMultidimensional-Departamental-2025.xlsx`, hoja `IPM_Departamentos`, columna *Total* por año |
| Homicidios | MinDefensa (datos.gov.co) | Dataset `m8fd-ahd9`, consultado vía API SODA |
| Población | DANE | Proyecciones CNPV 2018; tabla de respaldo incluida en el código (`POB_DANE`) |
| Geometría | GitHub (GeoJSON) | `caticoa3/colombia_mapa` (principal) y `santiblanko/colombia.geojson` (respaldo) |

**Periodo:** 2018–2025 · **Unidad:** departamento (códigos DIVIPOLA de 2 dígitos).

---

## 2. Estructura del repositorio

```
.
├── app.py             # Aplicación Streamlit
├── requirements.txt   # Dependencias
└── README.md
```

### requirements.txt

```
streamlit
numpy
pandas
geopandas
matplotlib
plotly
requests
openpyxl
libpysal
esda
spreg
```

---

## 3. Ejecución

### En local

```bash
pip install -r requirements.txt
streamlit run app.py
```

### En Streamlit Community Cloud

1. Sube `app.py` y `requirements.txt` a la raíz del repositorio.
2. Crea la app apuntando a `app.py`.
3. En **Advanced settings** elige **Python 3.12** (o 3.11). Versiones muy nuevas (p. ej. 3.14) pueden no tener binarios para las librerías geoespaciales y hacer fallar la instalación.
4. La **primera carga tarda unos minutos** (descarga de ocho años de homicidios, el Excel del DANE y la geometría). Después los resultados quedan en caché.

> Si hay errores, abre **Manage app** (esquina inferior derecha) para ver los logs completos; la interfaz oculta el mensaje original.

---

## 4. Estructura de `app.py`

| Bloque | Descripción |
|---|---|
| **Configuración** | Diccionarios `DIVIPOLA` y `NOMBRE_A_COD`, URLs de datos y tabla de población. Usa una carpeta temporal del sistema para guardar el Excel descargado. |
| **Carga de datos** | `load_geometry()`, `load_homicidios()`, `load_ipm_dane()` y `load_poblacion()`. Las descargas están cacheadas con `@st.cache_data`. |
| **Panel** | `build_panel()` une homicidios + IPM + población, calcula `tasa_homicidio` (por 100 mil hab.) y audita duplicados, NaN y alineación de códigos con la geometría. |
| **ESDA** | `compute_esda()`: matriz de vecindad **Queen** estandarizada por filas, **Moran's I global** por año y variable, y **LISA** del último año (HH, LL, HL, LH). |
| **Modelos** | `compute_models()`: con rezagos temporales (`ipm_l1`, `hom_l1`) estima OLS, SAR, SEM, SAC y SDM para dos ecuaciones (IPM y tasa de homicidio) y elige el de menor **AIC**. |
| **Mapas** | `static_map()` (Matplotlib) y `map_3d()` (Plotly `scatter_3d`, con centroides calculados en CRS proyectado EPSG:3116). |
| **Interfaz** | Barra lateral con controles y cuatro pestañas (ver sección 5). |

---

## 5. Interfaz

**Barra lateral:**

- **Año:** 2018–2025 (según disponibilidad en el panel).
- **Variable:** `IPM`, `Tasa Homicidio` o `Bivariado` (índice ponderado: 60 % IPM normalizado + 40 % tasa de homicidio normalizada).
- **Tipo de mapa:** `Estático (Matplotlib)` o `3D Dinámico (Plotly)`.
- **Departamento:** `Todos` o uno específico (se resalta en el mapa estático y se muestra su ficha con IPM y tasa).

**Pestañas:**

1. **Mapas:** mapa seleccionado, ficha del departamento y tabla ordenada por IPM.
2. **ESDA (Moran / LISA):** tablas de Moran global por año y conteo de clústeres LISA significativos al 5 %.
3. **Modelos espaciales:** mejor modelo por AIC para cada ecuación, tabla de AIC y resumen completo desplegable.
4. **Auditoría y fuentes:** resumen del panel, chequeos de calidad y enlaces a las fuentes.

---

## 6. Modelos espaciales

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

## 7. Auditorías incluidas

- 33 códigos DIVIPOLA verificados.
- IPM 2023 ≠ 2024 (aserción) y conteo de departamentos que cambian entre años consecutivos.
- Sin NaN ni duplicados en el panel.
- Códigos del panel alineados con la geometría.
- Tamaño de `y` igual al tamaño de la matriz `W`.
- Avisos en pantalla si algún año de homicidios falla o si algún nombre del Excel no se puede mapear.

---

## 8. Limitaciones y observaciones

1. **Población:** la tabla `POB_DANE` está escrita a mano y muchos valores están redondeados, por lo que las tasas son aproximadas. Conviene reemplazarla con las proyecciones oficiales del DANE.
2. **Formato del Excel DANE:** el lector depende de posiciones fijas (filas 14–49, columnas `2 + 3·i`). Si el DANE cambia el anexo, la lectura fallará.
3. **Geometría:** proviene de repositorios de terceros, no de una fuente oficial.
4. **Estimación:** se hace sobre un solo corte transversal (n ≈ 33), lo que limita la potencia estadística. Pese al nombre "SPVAR", no se estima un VAR espacial de panel completo.
5. **Selección por AIC:** `GM_Combo_Hom` (SAC) no reporta AIC, por lo que nunca resulta ganador.
6. **Recursos en la nube:** la descarga de homicidios trae todas las filas de cada año; en el plan gratuito de Streamlit Cloud puede acercarse al límite de memoria. Una alternativa es agregar en el servidor de datos.gov.co con `$group` y `sum(cantidad)`.
7. **Disponibilidad de las fuentes:** la app depende de que datos.gov.co, dane.gov.co y GitHub estén en línea al primer arranque o tras limpiar la caché.
