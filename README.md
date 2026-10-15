# TP2 - Reconstrucción 3D (I308 Visión Artificial)

Pipeline de reconstrucción 3D con una cámara estéreo: calibración → rectificación →
disparidad → nube local → pose con un objeto de referencia → fusión en Open3D →
optimización → medición métrica.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Probado con Python 3.11 en macOS (arm64).

Para usar CREStereo (`disparity_method: crestereo`) hace falta el modelo ONNX (~25 MB). El
módulo de la cátedra lo baja solo la primera vez; si falla por certificados SSL (pasa con el
Python de python.org en macOS), bajarlo a mano:

```bash
mkdir -p models
curl -L -o models/crestereo_combined_iter10_720x1280.onnx \
  https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter10_720x1280.onnx
```

## Uso

```bash
python pipeline/run_pipeline.py configs/stereo_budha_charuco.yaml          # corre todo
python pipeline/run_pipeline.py configs/stereo_budha_charuco.yaml --show   # + visores Open3D
python pipeline/run_pipeline.py configs/grupo.yaml                         # dataset propio (CREStereo, ~17 min la 1ra vez)
```

### Notebook / Google Colab

`tp2_reconstruccion.ipynb` recorre el mismo pipeline paso a paso (llama a las funciones de
`pipeline/`) y muestra la figura de cada etapa. Para abrirlo en Colab:
`File → Open notebook → GitHub` y pegar la URL del repo (branch `cami-pipeline`). La primera
celda clona el repo e instala las dependencias. En Colab usa SGBM por defecto (CREStereo en CPU
tarda ~15 min); se puede forzar con `USE_CRESTEREO = True`.

Localmente: `pip install jupyter` en el `.venv` y `jupyter notebook tp2_reconstruccion.ipynb`.

Salidas en `output/<dataset>/`:

| archivo | contenido |
|---|---|
| `stereo_calibration.pkl`, `stereo_maps.pkl` | calibración y mapas de rectificación (mismo formato que `calib-stereo`) |
| `figures/` | figuras para el informe (detección, error por vista, rectificación, disparidad, SGBM vs CREStereo, pose, error PnP por captura, cámaras, etapas de optimización, nube final, malla) |
| `reconstruction.ply` | **nube final optimizada (entregable)**, PLY binario |
| `reconstruction_mesh.ply` | malla por Poisson (opcional del enunciado) |
| `summary.json` | parámetros de calibración, error RMS, poses, puntos por etapa, medición |
| `intermediate/` | nubes de la escena completa y disparidades CREStereo cacheadas (no se entregan) |

## Estructura

```
pipeline/
  dataset.py        lectura de pares left/right y metadatos (checkerboard.txt, markers.yaml)
  calibration.py    4.1  findChessboardCorners, calibrateCamera, stereoCalibrate, stereoRectify, initUndistortRectifyMap
  stereo.py         4.2  SGBM o CREStereo + reprojectImageTo3D con Q
  pose.py           4.2  ChArUco / checkerboard + solvePnP + Rodrigues, transformación local -> mundo
  fusion.py         4.3  caja (OrientedBoundingBox), voxel, plano RANSAC, outliers, DBSCAN, Poisson, medición
  aruco.py          módulo de la cátedra (tutorial 7)
  calib.py          módulo de la cátedra (tutorial 6): board_points, detect_board, draw_checkerboard, np_print
  run_pipeline.py   orquesta todo
configs/
  stereo_budha_charuco.yaml
  grupo.yaml        dataset propio (peluche)
```

## Decisiones del pipeline

- **Calibración**: como el tutorial 6: `calib.detect_board` (findChessboardCorners +
  cornerSubPix) y `stereoCalibrateExtended` (da también el error por vista). Opción
  `mono_init: true` para inicializar con `calibrateCamera` por cámara; con este dataset
  da exactamente el mismo resultado.
- **Rectificación**: `alpha=0` (sin bordes negros).
- **Disparidad**: dos opciones (`disparity_method`):
  - `StereoSGBM` (modo 3WAY). Con f ≈ 600 px y baseline ≈ 60 mm, `d = f·B/Z` da ~145 px a
    25 cm y ~45 px a 80 cm, por eso `numDisparities=192`. Rápido (~0.1 s por par).
  - **CREStereo** (red neuronal, CVPR 2022) con el módulo `disparity` de la cátedra
    (`code_examples/stereodemo_disp_example`), a 1280x720 y 10 iteraciones. ~15 s por par en
    CPU; se cachea en `intermediate/`. Da una disparidad densa y mucho más limpia en zonas
    con poca textura o con el desalineamiento vertical del dataset propio
    (figura `03b_sgbm_vs_crestereo.png`).
- **Pose**: se estima sobre la imagen izquierda **rectificada** con `K = P1[:, :3]` y
  sin distorsión. Así la pose queda en el mismo sistema que la nube de
  `reprojectImageTo3D` y no hace falta componer con `R1`. Correspondencias: esquinas de
  cada marcador ArUco + corners ChArUco interpolados (con el objeto encima del tablero se
  interpolan muy pocos corners ChArUco). `solvePnP` (IPPE, porque el patrón es plano) +
  refinamiento LM.
- **Local → mundo**: `X_world = Rᵀ (X_cam − t)`, y una rotación de 180° sobre X para que
  Z apunte hacia arriba desde el tablero.
- **Optimización**: caja → voxel (1 mm) → plano del tablero por RANSAC (se sacan los puntos
  a < 8 mm) → outliers estadísticos → cluster más grande (DBSCAN).
- **Malla (opcional)**: normales + Poisson (octree depth 8), se descartan los vértices de
  menor densidad (superficie "inventada") y se simplifica a 200k triángulos.
- **Medición**: alto = distancia al plano de apoyo (percentil 99.5); ancho y
  profundidad = extensión en X/Y (percentiles 0.5–99.5).

## Usar un dataset propio

1. Armar la carpeta con la misma estructura que los datasets de la cátedra:
   ```
   mi_dataset/
     calib/     left_0.jpg right_0.jpg ...  + checkerboard.txt
     captures/  left_0.jpg right_0.jpg ...  + markers.yaml
   ```
   (si calibraron con `calib-stereo`, pueden poner sus `.pkl` en `calibration_pkl` / `maps_pkl`
   y no hace falta `calib/`).
2. Copiar `configs/stereo_budha_charuco.yaml` a `configs/mi_dataset.yaml` y cambiar
   `dataset`, `output`, la caja (`box_min`/`box_max`, en mm sobre el tablero) y
   `real_height_mm` si midieron el objeto con regla.
3. `python pipeline/run_pipeline.py configs/mi_dataset.yaml`

## Resultados con `stereo_budha_charuco`

- Calibración: 20/21 pares, RMS estéreo 1.06 px, baseline 59.95 mm.
- Pose: 21/21 capturas, error de reproyección de PnP 0.9–1.8 px.
- Puntos: 11.45 M → caja 5.21 M → voxel 1.34 M → plano 0.72 M → outliers 0.69 M → cluster 0.68 M.
- `reconstruction.ply`: 18.3 MB.
- Medición: alto ≈ 273 mm, ancho X ≈ 200 mm, profundidad Y ≈ 186 mm.

## Resultados con el dataset propio (`tp2_reconstruccion_3d/datasets`, config `configs/grupo.yaml`)

Peluche sobre un ChArUco 9x13 (`DICT_5X5_100`, cuadrado 42 mm, marcador 31.5 mm).

- Calibración: 81/81 pares (checkerboard 9x6, 28 mm), RMS estéreo 1.07 px, baseline 60.70 mm
  (coincide con la calibración de `calib-stereo` en `datasets/stereo_calibration.pkl`).
- Pose: 58/58 capturas, error de reproyección de PnP 1.5–4.3 px (mediana 2.5 px).
- Disparidad: **CREStereo** (con SGBM solo ~50 % de los píxeles tienen disparidad válida y el
  tablero sale muy ruidoso; ver `03b_sgbm_vs_crestereo.png`).
- Puntos: 100.0 M → caja 12.73 M → voxel 2.26 M → plano 0.74 M → outliers 0.71 M → cluster 0.69 M.
- `reconstruction.ply`: 18.5 MB. `reconstruction_mesh.ply` (Poisson, 200k triángulos): 7.7 MB.
- Medición: **alto ≈ 130 mm** sobre el plano del tablero (con SGBM daba 131 mm: la altura es
  estable entre métodos). Ancho X ≈ 217 mm y profundidad Y ≈ 239 mm incluyen la cola extendida y
  un resto del tablero pegado a la base (ver limitación).

### Limitación encontrada

Tras rectificar, los puntos homólogos (SIFT) entre izquierda y derecha tienen un desalineamiento
vertical de ±1–4 px que cambia de signo entre capturas (en el dataset del Buda es ≈0.1 px). Pasa
igual con los mapas de `calib-stereo`, y no es un error fijo de calibración (el ajuste de dy en
función de la posición en la imagen da coeficientes distintos en cada toma). Lo más probable es
movimiento de la cámara en mano durante la captura (blur / falta de sincronía entre sensores).

Consecuencias: error de profundidad de ±5–7 % por vista (la escala global sí es correcta: la
profundidad estéreo / PnP en los corners del tablero da mediana 0.996), el tablero se reconstruye
como una capa de 1–4 cm inclinada ~3°, y la cola del peluche (apoyada) se mezcla con el tablero.
Mejoras posibles: capturar con la cámara quieta/apoyada en cada toma, o alinear cada nube local
al plano conocido del tablero antes de fusionar (o ICP).
