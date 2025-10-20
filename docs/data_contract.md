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
    index/
      train.csv
      val.csv
      test.csv
meta.csv          # CSV semicolon con columnas id;texto
```

Los índices de frame (`XXXXXX`) deben comenzar en `000000` y avanzar sin huecos.
Los `pose/*.npz` deben contener la clave `pose` con valores `float32`.

## Metadata obligatoria en `meta.csv`

- `id`: identificador único del video (coincide con los nombres de archivo).
- `texto`: transcripción o glosa asociada.

Campos adicionales admitidos y utilizados por los *quality checks* del dataset:

- `fps`: FPS nominal del video original.
- `duration`: duración en segundos.
- `frame_count`: cantidad total de frames generados.

Cuando `fps` está ausente pero `duration` y `frame_count` existen, se calcula
`fps = frame_count / duration`.

## Elementos devueltos por `LsaTMultiStream`

Cada ejemplo entregado por el dataset incluye:

- `face`, `hand_l`, `hand_r`: tensores `(T, 3, H, W)` normalizados con medias y
  desviaciones de ImageNet.
- `pose`: tensor `(T, 3 * landmarks)` con poses reescaladas a `[-1, 1]`.
- `pose_conf_mask`: máscara booleana `(T, landmarks)` para filtrar poses
  inestables.
- `pad_mask`: máscara booleana `(T,)` con los frames válidos tras truncado.
- `length`: longitud efectiva calculada como `pad_mask.sum()`.
- `miss_mask_hl` / `miss_mask_hr`: marcan frames sin detección confiable de
  manos.
- `quality`: diccionario con métricas de FPS, frames perdidos y longitud
  efectiva.
- `text` y `video_id`: valores originales del CSV.

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
