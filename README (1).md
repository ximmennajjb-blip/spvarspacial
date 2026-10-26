# Taller 4 – SpVAR Colombia

Análisis espacial de los departamentos de Colombia que relaciona **PIB** y **homicidios**, con selección automática del modelo de econometría espacial y una aplicación interactiva en **Streamlit**.

---

## 1. ¿Qué hace?

1. Descarga la geometría de los departamentos de Colombia (GeoJSON).
2. Descarga datos reales de **PIB departamental (DANE)** y **homicidios (datos.gov.co)**.
3. Construye un panel departamento–año y varias matrices de pesos espaciales **W**.
4. Elige la mejor matriz **W** según el Moran's I de los residuos OLS.
5. Estima, para cada ecuación (PIB y homicidios), los modelos **OLS, SAR, SEM, SAC y SDM** y elige el más adecuado con pruebas LM/RLM y AIC.
6. Calcula autocorrelación espacial local (**LISA**) y el estadístico **Gi\***.
7. Muestra todo en una app web con mapas estáticos e interactivos, diagnósticos de los modelos y una explicación por departamento.

---

## 2. Requisitos

- **Python 3.11 o 3.12** (con 3.14 algunas librerías geoespaciales aún pueden no instalarse).
- Conexión a internet (se descargan los datos al iniciar).
- Dependencias listadas en `requirements.txt`:

```
streamlit  geopandas  libpysal  spreg  esda  folium  matplotlib
pandas  numpy  scipy  shapely  pyproj  requests  openpyxl
```

---

## 3. Cómo ejecutarlo

### Localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

### En Streamlit Community Cloud

1. Sube `app.py` y `requirements.txt` a un repositorio de GitHub.
2. En [share.streamlit.io](https://share.streamlit.io) crea una app nueva y elige el repositorio.
3. Indica `app.py` como archivo principal.
4. En **Advanced settings**, selecciona **Python 3.11 o 3.12**.
5. Despliega. La primera carga tarda porque descarga los datos y estima los modelos; después queda en caché.

---

## 4. Estructura del código

| Sección | Contenido |
|---|---|
| **Geometría** | `load_colombia_departments()` descarga el GeoJSON, normaliza columnas a `DEPARTAMENTO` y `COD_DPTO`, y devuelve el mapa en EPSG:4326 y en EPSG:3116 (proyectado). |
| **Datos reales** | `load_pib()`, `load_homicidios()` y `build_panel()` construyen el panel final. |
| **Matrices W** | Queen, Rook y KNN (k = 4, 5, 6), estandarizadas por filas. Se escoge la de mayor \|Moran's I\| de los residuos OLS. |
| **Datos SpVAR** | `prepare_spvar()` arma el corte transversal del último año con rezagos de PIB y homicidios. |
| **Selección de modelo** | `spatial_model_selection()` estima los cinco modelos y decide cuál usar. |
| **Estimación final** | `estimate_final()` ajusta el modelo elegido para cada ecuación. |
| **ESDA** | LISA (Moran local) para PIB y homicidios y Gi\* para homicidios. |
| **Pipeline** | `run_pipeline()` ejecuta todo y se guarda en caché con `st.cache_resource`. |
| **Interfaz** | Barra lateral con controles y tres pestañas: Mapas, Modelos y Datos. |

---

## 5. Fuentes de datos

| Dato | Fuente |
|---|---|
| Geometría departamental | GitHub: `caticoa3/colombia_mapa` (respaldo: `santiblanko/colombia.geojson`) |
| PIB departamental | DANE – `anex-PIBDep-TotalDep-2025pr.xlsx`, hoja "Cuadro 1" |
| Homicidios | datos.gov.co – recurso `m8fd-ahd9` |

---

## 6. Variables del modelo

**Dependientes (una ecuación por cada una):** `log_PIB` y `tasa_homicidios`.

**Explicativas (`X`):** `log_PIB_lag`, `hom_lag`, `luces_viirs`, `log_poblacion`, `urbanizacion`.

### ⚠️ Variables aproximadas o simuladas

No todas las variables provienen de fuentes reales. Hay que tenerlo en cuenta al interpretar resultados:

- `poblacion_approx` se calcula como `exp(log_PIB) * 120` (no es población real).
- `tasa_homicidios` usa esa población aproximada.
- `luces_viirs` se **simula** a partir de `log_PIB` más ruido aleatorio.
- `urbanizacion` se **simula** a partir de `log_PIB` más ruido aleatorio.
- `densidad` usa un área fija (15 000) para todos los departamentos.

Para un análisis riguroso, reemplazar estas variables por datos oficiales (proyecciones de población DANE, luces nocturnas VIIRS reales, tasas de urbanización y áreas departamentales).

---

## 7. Lógica de selección de modelo

Se parte de un OLS con diagnósticos espaciales y se decide así (nivel de significancia 0.05):

1. **RLM-Lag significativo y RLM-Error no** → **SAR**
2. **RLM-Error significativo y RLM-Lag no** → **SEM**
3. **Ambos significativos** → el de menor **AIC** entre SAR, SEM, SAC y SDM
4. **Ninguno significativo** → **OLS**
5. Si el **SDM** mejora el AIC en más de 2 puntos frente al mejor resto, se prefiere **SDM**.

---

## 8. Interfaz

**Barra lateral**
- **Variable:** `tasa_homicidios`, `log_PIB` o `luces_viirs`.
- **Departamento:** lista desplegable.
- Resumen del año de referencia, la matriz W y los modelos elegidos.

**Pestañas**
- **Mapas:** explicación del departamento seleccionado (clúster LISA), mapa estático y mapa interactivo (Folium).
- **Modelos:** diagnósticos LM/RLM, tabla de AIC y resumen completo del modelo final de cada ecuación.
- **Datos:** tabla del corte transversal con botón para descargar CSV.

Etiquetas LISA: `High-High`, `Low-Low`, `High-Low`, `Low-High` y `No significativo`.

---

## 9. Limitaciones y posibles mejoras

- Un solo corte transversal (último año) con un rezago, no un SpVAR de panel completo.
- Las URLs externas pueden cambiar o caer; no hay alternativa para el PIB de DANE.
- La elección de W por mayor Moran's I es una heurística, no una prueba formal.
- Varias variables explicativas son simuladas o aproximadas (ver sección 6).
