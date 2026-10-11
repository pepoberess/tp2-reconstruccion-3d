# Paso I: Instalación

## 1. Clonar el repo

```bash
  git clone https://github.com/udesa-vision/i308-calib.git
```

## 2. Crear un virtual environment de python

```bash
  # pararse en el root del proyecto
  cd i308-calib
  
  # crear y activar el virtual environment
  python3 -m venv .venv
  source .venv/bin/activate

```

Nota: Si estamos en Windows para activar el entorno usar `.venv\Scripts\activate.bat`


## 3. instalar el proyecto

```bash

    # instalar dependencias
    pip install -r requirements.txt

    # instalar el proyecto en el entorno
    # el flag -e nos permite tocar los fuentes sin reinstalar en caso de ser necesario
    pip install -e .

    # comprobar que fue instalado correctamente
    # con el comando calib-stereo:
    calib-stereo

```

Si todo sale bien deberíamos ver algo de la forma:

```
    usage: calib-stereo [-h] [-cfg CONFIG] [-v VIDEO] [-r RESOLUTION]
                        [-t THREADED] [-c CHECKERBOARD] [-sq SQUARE_SIZE]
                        [-d DATA]
    calib-stereo: error: Either -v (video) or -cfg (config) must be specified, or both.

```


## 3. Configurar herramienta

Copiamos archivos pre-configurados con diferentes perfiles de uso
con el comando `copy-configs`:

```bash
    
    # esto generará archivos .yaml en la carpeta cfg
    copy-configs
    
```

verifiquemos que los archivos .yaml se copiaron correctamente,
en el directorio `cfg`:

```
    # listamos directorio (dir si estamos en windows) 
    ls cfg
    
    default.yaml	lite.yaml	m1.yaml
    full.yaml	logi.yaml	stereo.yaml
    
```

Si vemos los archivos, pasamos al paso siguiente.


# Paso II: Verificación de captura.


## 1. Conectar cámara estéreo. 
- Idealmente: usar puerto > USB 3.0

## 2. Configuramos captura en la herramienta

Abrimos el archivo de configuración `cfg/stereo.yaml` con algún 
editor de texto (por ejemplo con vscode):

```bash
    # para abrirlo desde la terminal
    code cfg/stereo.yaml
 ```

Deberíamos ver algo de la forma:
```yaml

video: 0
name: Stereo ELP-USB3D1080P02
# capture_mode: 'dshow'
# compression: 'MJPG'

resolution: 3840x1080

# resolutions:
# single > stereo:
#  1920x1080 > 3840x1080
#  1280x720 > 2560x720
#  800x600 > 1600x600
#  640x480 > 1280x480
#  320x240 > 640x240

resolutions:
  - 3840x1080
  - 2560x720
  - 1600x600
  - 1280x480
  - 640x240

```

### Editar los parámetros de captura:

Tenemos que configurar la captura de modo que funcione correctamente en nuestra computadora.

Modificamos parámetro:

- **video**: probamos distintos números 0, 1, 2... hasta que logremos dar con el id asociado a la cámara estéreo.

**Sólo en caso de tener problemas / o que ande muy lento**:
- **resolution**: Probamos reduciendo distintas opciones soportadas (ver listado en el archivo),
- **Otros parámetros**:
  - **compression**: usamos compresion MJPG
  - **capture_mode**: dshow, en Windows a veces hace falta setear este modo.

  
Guardamos el archivo y corremos la herramienta apuntando a este archivo de configuración:

```bash

  calib-stereo -cfg cfg/stereo.yaml      
            
```

Deberíamos ver la captura estéreo en una ventana. 
Evaluamos si esta captura es fluida (tiene que andar rápido).
        
Luego terminamos el proceso: 
- podemos presionar letra 'q' con foco en la GUI, o bien 
- CTRL+C en la terminal


Resultado:
- Si la captura es fluida, estamos OK y pasamos al próximo paso. 
- Si no, iteramos hasta que sea fluida:
  - Volvemos a editar el archivo alivianando requerimientos y volvemos a probar.
  - si ya intentamos varias veces y sigue sin andar bien, podemos probar otro puerto otro cable u otra computadora.


# Paso III: Calibración y Rectificación Estéreo

## 1. Checkerboard
Tener a mano un Checkerboard de calibración de:

- `NxM`: cantidad de corners internos
- `square_size`: tamaño de cuadrado en mm

## 2. Especificamos parámetros del checkerboard
Iniciamos la herramienta pasando como parámetro los parámetros del checkerboard:
Por ejemplo para un checkerboard de 10x7 y tamaño de cuadrado de 20.9 mm:

```bash

    calib-stereo -cfg cfg/stereo.yaml  --checkerboard 10x7 --square-size 20.9

```

## 3. Habilitamos detección del tablero: letra 'd'

- deberíamos ver en pantalla la detección de ambas cámaras
- si se pierde de vista, la app se pone lenta, apagamos deteccion letra 'd' otra vez y centramos el tablero.

## 4. Agregamos capturas al dataset de calibración: letra 'a'

- vamos "pintando" la pantalla con distintas vistas del tablero (variando cercania y rotación) 
- deberíamos cubrir la mayor parte de la lente en especial las esquinas donde hay más distorsión
- entre 30 y 40 imágenes deberían ser suficientes
- es importante que el tablero esté siempre recto y no se pandee.

## 5. Calibración estéreo: letra 'c'

- presionamos la letra 'c', esto invoca la función `cv2.stereoCalibrate` y luego guarda los resultados en el archivo `data/stereo_calibration.pkl`

Por consola se muestran las matrices resultantes.
Verifiquemos si tienen sentido, distancias focales, puntos principales y extrínsecas relativas.


## 6. Rectificación estéreo y mapas: letra 'm'

- presionamos la letra 'm', esto invoca la función `cv2.stereoRectify` y luego `cv2.initUnidistortRectifyMaps`, y luego 
guarda los resultados en el archivo `data/stereo_maps.pkl


Luego de este paso y hemos finalizado la calibración y podemos terminar el proceso de esta herramienta (letra 'q' o CTRL+C).
Verifiquemos los outputs en el directorio data, anotemos las características del checkerboard en algun archivo de metadata
y guardemos estos artefactos resultantes para uso posterior.


# Paso IV: Captura de dataset con objeto a reconstruir


## 1. Preparación de escena

- Colocamos el objeto a reconstruir y un objeto conocido (ejemplo ChArUco board) en la escena.
- Objetos adecuados: material piedra, madera, con textura, sin muchos reflejos ni transparencias. Tamaño ideal, entre 20 y 50 cm.

## 2. Volvemos a correr la herramienta: 

```bash

    calib-stereo -cfg cfg/stereo.yaml

```

## 3. Capturamops el objeto desde distintas vistas: letra 's'

- Importante: no mover los objetos durante la captura, la pose relativa entre el objeto a reconstruir y el objeto conocido debe ser siempre la misma.
- Asegurarse de que el objeto conocido sea visible desde todas las capturas del dataset. 
- Mantener overlap entre capturas consecutivas (ej 70%)
- Crear paralaje entre capturas consecutivas (movimiento hacia el costado, no solo rotacional)
