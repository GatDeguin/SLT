# Contrato de datos LSA-T Multi-Stream

Este documento describe los supuestos y requisitos que debe cumplir el pipeline de datos
para que los componentes de entrenamiento funcionen correctamente.

## Estructura de directorios

Cada split debe exponer una carpeta raíz con las siguientes subcarpetas y archivos:

- `face/`: imágenes RGB (JPEG) con nombre `<video_id>_fXXXXXX.jpg`.
- `hand_l/` y `hand_r/`: recortes cuadrados de las manos con el mismo patrón de nombre.
- `pose/`: archivos comprimidos `.npz` con la clave `pose` que almacena un tensor
  `(num_frames, 3 * landmark_count)` en formato `float32`.
- CSV principal con columnas `video_id` y `texto` (separadas por `;`).
- CSV de índices con la lista de `video_id` pertenecientes al split.

Los índices de frame (`XXXXXX`) deben ser consecutivos desde `000000` sin huecos. El
pipeline de calidad reportará cualquier frame faltante.

## Metadata esperada

El CSV principal puede incluir metadatos adicionales por video. Si están presentes, se
utilizan para controles de calidad:

- `fps`: FPS nominal del video original.
- `duration`: duración del clip (en segundos).
- `frame_count`: cantidad total de frames generados por el extractor.

Cuando `fps` no está disponible pero `duration` y `frame_count` sí lo están, se infiere
`fps = frame_count / duration`.

## Contenido del `SampleItem`

El dataset `LsaTMultiStream` devuelve instancias con los siguientes campos:

- `face`, `hand_l`, `hand_r`: tensores `(T, 3, H, W)` normalizados con ImageNet.
- `pose`: tensor `(T, 3 * landmarks)` con coordenadas normalizadas.
- `pose_conf_mask`: máscara booleana `(T, landmarks)` con `True` cuando la confianza
  supera `min_conf`.
- `pad_mask`: máscara booleana `(T,)` que marca los pasos de tiempo válidos.
- `length`: longitud efectiva (`pad_mask.sum()`) considerando todos los streams.
- `miss_mask_hl` / `miss_mask_hr`: máscaras booleanas que indican frames con detección
  válida de mano izquierda/derecha.
- `quality`: diccionario con `effective_length`, métricas de FPS y frames faltantes.
- `text`: glosa asociada.
- `video_id`: identificador del clip.

## Controles de calidad

El dataset ejecuta validaciones automáticas:

- **Frames faltantes**: detecta saltos en la numeración por stream. Dependiendo de la
  configuración (`quality_strict`), puede emitir `warnings` o detener la carga.
- **FPS**: calcula FPS efectivo a partir de la metadata y lo compara con el FPS esperado.
  Si la diferencia supera `fps_tolerance`, se genera una alerta.
- **Longitud efectiva**: se calcula como el mínimo número de frames disponible entre
  los streams (cara, manos, pose) acotado por `T`.

El campo `quality` permite registrar los resultados de estas validaciones para
post-procesamiento o auditoría.

## Augmentations

El flip horizontal sincronizado entre cara, manos y pose se activa únicamente cuando
`enable_flip=True` y se aplica con probabilidad `flip_prob`. El flip intercambia manos,
refleja el esqueleto y reordena la máscara de confianza.

## Persistencia generada por el extractor

El script `tools/extract_rois_v2.py` genera crops y un archivo `metadata.jsonl` por
sesión (ruta configurable). Cada entrada de metadata contiene:

- nombre del video y ruta original,
- FPS origen, objetivo y límite aplicado,
- `stride` usado para muestreo,
- cantidad de frames escritos y vectores de pose,
- bandera `success` y descripción de error si aplica.

Esta metadata permite reanudar ejecuciones (`--resume`) y auditar errores puntuales.

## Flujo de trabajo recomendado

1. **Instalación**: crea un entorno virtual e instala `requirements-dev.txt` para
   disponer del paquete `slt`, PyTorch y las herramientas de desarrollo
   utilizadas en CI (`pytest`, `ruff`, `black`, `mypy`, `onnx`).
2. **Preparación de datos**: ejecuta `tools/extract_rois_v2.py` con los videos
   brutos para generar `face/`, `hand_l/`, `hand_r/` y `pose/`; construye los
   CSV de metadata y splits siguiendo la estructura anterior.
3. **Entrenamiento**: utiliza `python -m slt` para un *smoke test* rápido o
   `tools/train_slt_multistream_v9.py` para sesiones largas. Ambos scripts
   consumen `LsaTMultiStream` y respetan los campos detallados en este contrato.
4. **Evaluación**: ejecuta `tools/eval_slt_multistream_v9.py` con el checkpoint y
   los splits de validación/prueba para obtener CER, BLEU y pérdida agregada.
5. **Exportación y despliegue**: genera artefactos ONNX/TorchScript con
   `tools/export_onnx_encoder_v9.py` y valida la inferencia en tiempo real usando
   `tools/demo_realtime_multistream.py` o `tools/test_realtime_pipeline.py`.

Los tests automatizados (`tests/data/`, `tests/models/`, `tests/training/` y
`tests/test_cli_main.py`) replican este flujo a escala reducida. Consulta la
tabla de métricas en el `README.md` para conocer los valores esperados que
servirán como referencia durante la verificación del entorno.
