# Taller 4 – SpVAR Colombia

Análisis espacial de los departamentos de Colombia que relaciona **PIB** y **homicidios**, con selección automática del modelo de econometría espacial y una interfaz interactiva en Gradio.

El script (`app.py`) fue exportado desde Google Colab y está pensado para ejecutarse de principio a fin en una sola celda.

---

## 1. ¿Qué hace?

1. Descarga la geometría de los departamentos de Colombia (GeoJSON).
2. Descarga datos reales de **PIB departamental (DANE)** y **homicidios (datos.gov.co)**.
3. Construye un panel departamento–año y varias matrices de pesos espaciales **W**.
4. Elige la mejor matriz **W** según el Moran's I de los residuos OLS.
5. Estima, para cada ecuación (PIB y homicidios), los modelos **OLS, SAR, SEM, SAC y SDM** y elige el más adecuado con pruebas LM/RLM y AIC.
6. Calcula autocorrelación espacial local (**LISA**) y estadístico **Gi\***.
7. Lanza una interfaz web con mapas estáticos, mapas interactivos (Folium) y una explicación por departamento.

---

## 2. Requisitos

**Python 3.9+** y conexión a internet (descarga datos y paquetes).

Paquetes que el script instala automáticamente con `pip`:

```
geopandas  libpysal  spreg  esda  mapclassify  folium  statsmodels
seaborn  matplotlib  pandas  numpy  scipy  shapely  pyproj
requests  openpyxl  gradio  plotly
```

> Para evitar reinstalar en cada ejecución, conviene instalarlos una vez y eliminar el bucle de instalación (sección 0 del script).

---

## 3. Cómo ejecutarlo

```bash
python app.py
```

En Google Colab: pegar el contenido en una celda y ejecutar. Al final se imprime un enlace público de Gradio (`share=True`).

---

## 4. Estructura del código

| Sección | Contenido |
|---|---|
| **0. Instalación** | Instala las librerías con `pip`. |
| **1. Imports** | Librerías, estilo de gráficos y semilla (`np.random.seed(42)`). |
| **2. Geometría** | `load_colombia_departments()` descarga el GeoJSON, normaliza columnas a `DEPARTAMENTO` y `COD_DPTO`, y devuelve el mapa en EPSG:4326 y en EPSG:3116 (proyectado). |
| **3. Datos reales** | PIB DANE, homicidios y construcción del panel final (`df`). |
| **4. Matrices W** | Queen, Rook y KNN (k = 4, 5, 6), estandarizadas por filas. Se escoge la de mayor \|Moran's I\| de los residuos OLS. |
| **5. Datos SpVAR** | `prepare_spvar()` arma el corte transversal del último año con rezagos de PIB y homicidios. |
| **6. Selección de modelo** | `spatial_model_selection()` estima los cinco modelos y decide cuál usar. |
| **7. Estimación final** | `estimate_final()` ajusta el modelo elegido para cada ecuación. |
| **8. ESDA** | LISA (Moran local) para PIB y homicidios y Gi\* para homicidios. |
| **9. Interfaz Gradio** | Mapas, explicación y controles interactivos. |

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

Se parte de un OLS con diagnósticos espaciales y se decide así:

1. **RLM-Lag significativo y RLM-Error no** → **SAR**
2. **RLM-Error significativo y RLM-Lag no** → **SEM**
3. **Ambos significativos** → el de menor **AIC** entre SAR, SEM, SAC y SDM
4. **Ninguno significativo** → **OLS**
5. Si el **SDM** mejora el AIC en más de 2 puntos frente al mejor resto, se prefiere **SDM**.

Nivel de significancia usado: 0.05.

---

## 8. Interfaz Gradio

- **Variable:** `tasa_homicidios`, `log_PIB` o `luces_viirs`.
- **Departamento:** lista desplegable.
- **Botón "Actualizar diagnóstico":** regenera el mapa estático (PNG), el mapa interactivo (Folium) y el texto explicativo con el tipo de clúster LISA, el modelo usado y la matriz W elegida.

Etiquetas LISA: `High-High`, `Low-Low`, `High-Low`, `Low-High` y `No significativo`.

---

## 9. Limitaciones y posibles mejoras

- Un solo corte transversal (último año) con un rezago, no un SpVAR de panel completo.
- Las URLs externas pueden cambiar o caer; el script no tiene alternativa para el PIB de DANE.
- Los `except` genéricos pueden ocultar errores reales; conviene registrar el mensaje.
- El criterio "mejor W" por mayor Moran's I es una heurística, no una prueba formal.
- `demo.launch(share=True)` expone la app públicamente; usar `share=False` para entornos locales.
