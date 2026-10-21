# TP2 - Reconstrucción 3D (I308 Visión Artificial)

Reconstrucción 3D de un objeto con una cámara estéreo. El pipeline hace: calibración,
rectificación, disparidad, nube local por captura, pose con un objeto de referencia
(ChArUco), fusión de todas las nubes en Open3D, limpieza de la nube y medición del objeto en mm.

## Instalación

```bash
python -m venv vision
# Windows: vision\Scripts\activate    |   Linux/macOS: source vision/bin/activate
pip install -r requirements.txt
```

Versiones usadas (ver `requirements.txt`):

- opencv-python y opencv-contrib-python 4.13.0.92 (la 4.12 rompe el módulo aruco)
- open3d 0.19.0 (con Python 3.13 no existe, hay que usar la 0.20.0)
- numpy 2.4.6, matplotlib 3.11.2, PyYAML 6.0.3
- onnxruntime 1.31.0 (para CREStereo)

Lo probamos con Python 3.11 en macOS y con Python 3.13 en Windows.

Para correr el notebook también hace falta `ipykernel` (o `jupyter`).

Si se usa CREStereo hace falta el modelo ONNX (~25 MB). El notebook lo baja solo la
primera vez. Si falla, se puede bajar a mano:

```bash
mkdir -p models
curl -L -o models/crestereo_combined_iter10_720x1280.onnx \
  https://github.com/nburrus/stereodemo/releases/download/v0.1-crestereo/crestereo_combined_iter10_720x1280.onnx
```

## Uso

Desde la raíz del repo:

```bash
python pipeline/run_pipeline.py configs/grupo.yaml                  # nuestro dataset (CREStereo, ~15 min la primera vez)
python pipeline/run_pipeline.py configs/stereo_budha_charuco.yaml   # dataset del Buda
python pipeline/run_pipeline.py configs/grupo.yaml --show           # además abre los visores de Open3D
```

### Notebook

`tp2_reconstruccion.ipynb` hace lo mismo pero paso a paso, mostrando las figuras de cada
etapa y comentando los resultados. Usa las funciones de `pipeline/`, no repite código.
Hay que abrirlo con el entorno donde se instalaron las dependencias.

En Google Colab: clonar el repo, instalar `requirements.txt` y abrir el notebook. Ahí usa
SGBM por defecto porque CREStereo en CPU es lento (se puede forzar con `USE_CRESTEREO = True`).

### Salidas

Todo se guarda en `output/<dataset>/`:

- `reconstruction.ply`: nube final optimizada (es el entregable), en PLY binario.
- `reconstruction_mesh.ply`: malla con Poisson (opcional).
- `stereo_calibration.pkl` y `stereo_maps.pkl`: calibración y mapas de rectificación, con
  el mismo formato que `calib-stereo`.
- `figures/`: las figuras para el informe.
- `summary.json`: parámetros de calibración, errores, poses, puntos por etapa y medición.
- `intermediate/`: disparidades guardadas y nubes intermedias (no se entregan).

## Estructura

```
pipeline/
  dataset.py        lectura de los pares left/right y de los metadatos (checkerboard.txt, markers.yaml)
  calibration.py    calibración y rectificación estéreo
  stereo.py         disparidad (SGBM o CREStereo) y nube local con reprojectImageTo3D
  pose.py           detección del ChArUco / checkerboard, solvePnP y paso al sistema del mundo
  fusion.py         caja, voxel, plano del tablero, outliers, DBSCAN, Poisson y medición
  aruco.py          módulo de la cátedra (tutorial 7)
  calib.py          módulo de la cátedra (tutorial 6)
  run_pipeline.py   corre todo el pipeline
configs/
  grupo.yaml                 nuestro dataset (peluche)
  stereo_budha_charuco.yaml  dataset del Buda
```

## Decisiones

- **Calibración**: con OpenCV, como en el tutorial 6 (`findChessboardCorners` +
  `cornerSubPix` y `stereoCalibrateExtended`, que además da el error de cada par). Con
  `max_view_error_px` se descartan los pares que calibraron mal y se vuelve a calibrar.
- **Rectificación**: `stereoRectify` con `alpha=0` para que no queden bordes negros.
- **Disparidad**: se puede elegir en la config.
  - SGBM: con f ≈ 600 px y baseline ≈ 60 mm, la disparidad va de ~145 px a 25 cm a ~45 px
    a 80 cm, por eso `numDisparities=192`. Tarda ~0.1 s por par.
  - CREStereo (red neuronal, con el módulo de la cátedra en `code_examples/`): 1280x720 y
    10 iteraciones, ~15 s por par en CPU. Da una disparidad mucho más completa y limpia que
    SGBM en nuestro dataset, así que es la que usamos.
- **Pose**: se estima en la imagen izquierda rectificada, con `K = P1[:, :3]` y sin
  distorsión. Así queda en el mismo sistema que la nube de `reprojectImageTo3D`. Usamos
  `solvePnP` con IPPE (el patrón es plano) y después refinamos con LM.
- **Corrección de la disparidad con el tablero** (`board_disparity_correction`, solo en
  nuestro dataset): en cada captura comparamos la disparidad medida en las esquinas del ChArUco
  con la que corresponde a la profundidad que da PnP, ajustamos `d_medida = a·d_esperada + b`
  y corregimos todo el mapa. Si la cámara mira el tablero casi de frente y la recta queda mal
  definida, se corrige solo el corrimiento `b`. Ver "Problema que encontramos".
- **Local a mundo**: `X_world = Rᵀ (X_cam − t)`, más un giro de 180° sobre X para que Z
  apunte hacia arriba desde el tablero.
- **Limpieza de la nube**: caja alrededor del objeto, voxel de 1 mm, se saca el plano del
  tablero con RANSAC (todo lo que está a menos de 15 mm), outliers estadísticos y nos quedamos
  con el cluster más grande (DBSCAN).
- **Malla (opcional)**: normales, Poisson con profundidad 8, se sacan los vértices con poca
  densidad y se simplifica a 200k triángulos.
- **Medición**: alto = distancia al plano del tablero (percentil 99.5); ancho y profundidad
  = extensión en X e Y (percentiles 0.5 a 99.5).

## Resultados

### Dataset del Buda (`stereo_budha_charuco`)

- Calibración: 20 de 21 pares, RMS estéreo 1.06 px, baseline 59.95 mm.
- Pose: 21 de 21 capturas, error de PnP entre 0.9 y 1.8 px.
- Puntos: 11.45 M, caja 5.21 M, voxel 1.34 M, sin plano 0.72 M, sin outliers 0.69 M, cluster 0.68 M.
- `reconstruction.ply`: 18.3 MB.
- Medición: alto ≈ 273 mm, ancho ≈ 200 mm, profundidad ≈ 186 mm.

### Nuestro dataset (`configs/grupo.yaml`)

Un peluche sobre un ChArUco de 9x13 (`DICT_5X5_100`, cuadrado de 42 mm, marcador de 31.5 mm).

- Calibración: checkerboard 9x6 de 28 mm, detectado en los 81 pares. Con todos, el RMS
  estéreo daba 1.07 px. Descartando los 5 pares con más de 2 px (4, 36, 43, 48 y 51) baja a
  0.71 px. Baseline 60.1 mm.
- Pose: 58 de 58 capturas, error de PnP entre 1.6 y 4.4 px (promedio 2.7 px).
- Disparidad: CREStereo. Con SGBM solo la mitad de los píxeles tiene disparidad válida.
- Corrección con el tablero: se pudo ajustar en las 58 capturas (escala entre 0.91 y 1.09,
  corrimiento entre −9 y +15 px).
- Puntos: 99.7 M, caja 13.1 M, voxel 1.19 M, sin plano 0.42 M, sin outliers 0.39 M, cluster 0.36 M.
- `reconstruction.ply`: 9.8 MB. `reconstruction_mesh.ply`: 7.7 MB.
- Medición: alto ≈ 133 mm sobre el plano del tablero, ancho X ≈ 144 mm y profundidad
  Y ≈ 212 mm (incluye la cola, que está estirada). No medimos el peluche con regla, así que
  todavía no hay error relativo: si se carga `real_height_mm` en la config, se calcula solo.

### Problema que encontramos (y cómo lo corregimos)

Después de rectificar, los puntos que se corresponden entre la imagen izquierda y la derecha
quedan corridos verticalmente entre 1 y 4 px, y el signo cambia de una captura a otra. En el
dataset del Buda ese corrimiento es de ~0.1 px. Pasa lo mismo con los mapas de `calib-stereo`,
así que no parece un error de calibración. Lo más probable es que la cámara se haya movido
mientras sacábamos las fotos (la teníamos en la mano).

Esto mete un error de profundidad distinto en cada vista. Sin corregir, el tablero, que
debería estar en Z = 0, queda entre −25 y +23 mm según la captura. Así el tablero salía como
una capa gruesa, el plano de RANSAC no lo sacaba entero, el peluche quedaba borroso y el ancho
y la profundidad salían inflados (217 y 259 mm).

Para arreglarlo usamos el ChArUco como referencia: en cada captura sabemos, por la pose, a qué
profundidad está cada esquina del tablero, y por lo tanto qué disparidad debería tener.
Comparándola con la medida ajustamos una corrección para esa vista (ver
`stereo.board_disparity_correction`). Con eso el tablero queda entre −1 y +4 mm en todas las
vistas, se saca entero con RANSAC y la nube final queda limpia.

Lo que se podría mejorar:

- Sacar las fotos con la cámara apoyada y quieta, para no tener que corregir cada vista.
- Alinear las vistas con ICP para afinar los bordes, que todavía se ven algo borrosos.
- Medir el peluche con regla para tener el error relativo.
