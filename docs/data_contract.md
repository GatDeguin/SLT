# Contrato de datos para `single_signer`

Este documento describe los requisitos que debe cumplir el pipeline de datos
para que las herramientas de entrenamiento, evaluación y exportación funcionen
sin ajustes manuales. Todas las rutas se asumen relativas a la raíz del
repositorio, utilizando `data/single_signer/` y el CSV `meta.csv`.

## Distribución de carpetas

Cada split debe exponerse con la siguiente estructura:

```text
data/
  single_signer/
    processed/
      face/      # JPEG RGB con patrón <video_id>_fXXXXXX.jpg
      hand_l/    # Recortes cuadrados de mano izquierda
      hand_r/    # Recortes cuadrados de mano derecha
      pose/      # Archivos .npz con tensores (T, 3 * landmarks)
      keypoints/ # Archivos .npz/.npy con keypoints MediaPipe
    index/
      train.csv
      val.csv
      test.csv
    annotations/
      gloss.csv  # Opcional: glosas y etiquetas CTC por video
meta.csv          # CSV semicolon con columnas video_id;texto
```

Los índices de frame (`XXXXXX`) deben comenzar en `000000` y avanzar sin huecos.
Los `pose/*.npz` deben contener la clave `pose` con valores `float32`.

### Detalle por carpeta

- `videos/`: clips originales en MP4/MKV; sólo se consumen durante la extracción
  de ROIs.
- `processed/face/`, `processed/hand_l/`, `processed/hand_r/`: recortes en JPEG
  con patrón `<video_id>_fXXXXXX.jpg`, normalizados a RGB.
- `processed/pose/`: archivos `.npz` con un arreglo `pose` de forma
  `(T, 3 * landmarks)` en `float32` y la clave `pose_norm="signing_space_v1"`.
  Las coordenadas `(x, y)` se normalizan al *signing space* (ancho de 6 y alto
  de 7 unidades de cabeza) para quedar en el rango `[0, 1]` centradas en
  `(0.5, 0.5)`. Cuando no hay landmarks válidos se escribe un sentinel con
  `-1` en las coordenadas y visibilidad `0`.
- `processed/keypoints/`: matrices `.npy` o `.npz` con keypoints MediaPipe en
  formato `(T, landmarks, 3)` donde la última dimensión guarda `(x, y, conf)`.
- `index/*.csv`: listas de `video_id` (una columna, sin encabezado) utilizadas
  para los splits de entrenamiento, validación y prueba.
- `metadata.jsonl`: emitido por `tools/extract_rois_v2.py` con métricas por
  video (duración efectiva, FPS medidos, cantidad de frames escritos por stream).
- `annotations/gloss.csv`: archivo opcional con columnas `video_id`, `gloss` y
  `ctc_labels` (separadas por espacios) para alimentar pérdidas CTC.

## Metadata obligatoria en `meta.csv`

- `video_id`: identificador único del video (coincide con los nombres de
  archivo, sin sufijo de frame).
- `texto`: transcripción o glosa asociada.

Campos adicionales admitidos y utilizados por los *quality checks* del dataset:

- `fps`: FPS nominal del video original.
- `duration`: duración en segundos.
- `frame_count`: cantidad total de frames generados.
- `signer` / `subset`: etiquetas libres que pueden emplearse para análisis o
  filtrado manual (se ignoran durante la carga, pero se preservan al unir
  metadata).

Cuando `fps` está ausente pero `duration` y `frame_count` existen, se calcula
`fps = frame_count / duration`.

### CSV de índices (`index/*.csv`)

- Formato: una columna sin encabezado denominada `video_id`.
- Codificación: UTF-8 (se admiten variantes como UTF-8 con BOM).
- Contenido: los `video_id` deben existir en `meta.csv` y contar con streams en
  `processed/`.
- Uso: `tools/train_slt_multistream_v9.py`, `python -m slt` y
  `tools/eval_slt_multistream_v9.py` utilizan estos archivos para crear
  `DataLoader` consistentes entre ejecuciones.

## Elementos devueltos por `LsaTMultiStream`

Cada ejemplo entregado por el dataset incluye:

- `face`, `hand_l`, `hand_r`: tensores `(T, 3, H, W)` normalizados con medias y
  desviaciones de ImageNet.
- `pose`: tensor `(T, 3 * landmarks)` con coordenadas en `[0, 1]` dentro del
  *signing space* y sentinels `-1` cuando no hay detección confiable.
- `pose_conf_mask`: máscara booleana `(T, landmarks)` para filtrar poses
  inestables; ignora automáticamente los sentinels negativos.
- `pad_mask`: máscara booleana `(T,)` con los frames válidos tras truncado.
- `length`: longitud efectiva calculada como `pad_mask.sum()`.
- `miss_mask_hl` / `miss_mask_hr`: marcan frames sin detección confiable de
  manos.
- `keypoints`: tensor `(T, L, 3)` con los keypoints raw normalizados y
  reordenados según MSKA.
- `keypoints_*`: vistas separadas (`body`, `hand_l`, `hand_r`, `face`) con sus
  máscaras por landmark (`*_mask`) y por frame (`*_frame_mask`).
- `keypoints_lengths`: vector con las longitudes efectivas por vista.
- `ctc_labels`, `ctc_mask`, `ctc_lengths`: tensores preparados para pérdidas
  CTC cuando hay metadata de glosas.
- `gloss_sequence` / `gloss_text`: tokens de glosa tokenizados y su texto
  original cuando se provee `gloss.csv`.
- `quality`: diccionario con métricas de FPS, frames perdidos y longitud
  efectiva.
- `text` y `video_id`: valores originales del CSV.

Los campos `quality` y las máscaras se almacenan en CPU para reducir la memoria
requerida al mover los tensores principales a GPU.

## Validaciones automáticas

El dataset ejecuta controles de calidad al cargar cada video:

1. **Frames faltantes:** detecta saltos en la numeración de cada stream y decide
   si continuar (modo laxo) o lanzar una excepción (`quality_strict=True`).
2. **FPS efectivo:** compara el FPS estimado (`frame_count / duration`) con el
   declarado en metadata. Las discrepancias mayores a la tolerancia generan
   advertencias.
3. **Longitud efectiva:** sincroniza cara, manos y pose tomando el mínimo número
   de frames válido y aplicando `T` como límite superior.
4. **Máscaras de confianza:** recalcula `pose_conf_mask` según el umbral
   configurado (`pose_min_conf`).

Los resultados se almacenan en `quality` para auditoría posterior.

## Metadata derivada (`metadata.jsonl`)

`tools/extract_rois_v2.py` genera un archivo `metadata.jsonl` con una entrada por
video. Cada objeto JSON incluye:

- `video_id`: identificador del clip procesado.
- `frames_written`: cantidad de frames escritos por stream (`face`, `hand_l`,
  `hand_r`, `pose`).
- `duration`: duración estimada según timestamps del detector.
- `fps_observed`: tasa derivada de `frames_written / duration`.
- `status`: `"ok"` o detalles del error ocurrido durante la extracción.

Guarda este archivo junto a los recortes para facilitar reintentos con
`--resume` y comparar métricas entre ejecuciones.

## Recomendaciones para extracción de ROIs

- Utiliza `tools/extract_rois_v2.py` con los videos en
  `data/single_signer/videos/` y el CSV `meta.csv`.
- Define `--output data/single_signer/processed` para que los nombres coincidan
  con este contrato.
- Guarda el `metadata.jsonl` generado: permite reintentar videos fallidos y
  verificar métricas como FPS de origen, `stride` aplicado y cantidad de frames
  escritos.

## Validación en CI o entornos limpios

1. Ejecuta `python tools/ci_validate_data_contract.py` para generar un dataset
   sintético que replica la estructura anterior y confirma que los *quality
   checks* permanecen activos.
2. Corre `pytest tests/data/test_dataset_quality.py` para asegurar que los
   avisos y excepciones se comportan como lo documentado.
3. Mantén sincronizados los ejemplos de rutas con el README cuando cambies
   directorios o nombres de archivos.
4. Revisa que los archivos `index/*.csv` no contengan IDs duplicados o filas
   vacías; el dataset emite advertencias cuando detecta inconsistencias.
