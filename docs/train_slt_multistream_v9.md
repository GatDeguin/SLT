# Guía de entrenamiento `tools/train_slt_multistream_v9.py`

El script `tools/train_slt_multistream_v9.py` entrena el encoder multi-stream y
un decoder seq2seq compatible con HuggingFace utilizando los datos preparados en
`data/single_signer/processed/` y los índices derivados de `meta.csv`. De forma
predeterminada el modelo se inicializa con el checkpoint validado
`single_signer` cuando el archivo descargado está disponible (en
`data/single_signer/` o vía `SLT_SINGLE_SIGNER_CHECKPOINT`), aunque puedes
deshabilitarlo con `--pretrained none`. Esta guía resume los argumentos más
importantes, patrones de uso avanzados y recomendaciones para reproducir
experimentos. Complementa la referencia rápida incluida en `tools/README.md`.

## Requisitos previos

- Streams generados con `tools/extract_rois_v2.py` siguiendo el contrato de datos.
- CSV `meta.csv` con columnas `video_id;texto` y splits `train.csv`/`val.csv`/`test.csv`.
- Tokenizador HuggingFace compatible con modelos tipo T5/BART.

## Ejecución básica

```bash
python tools/train_slt_multistream_v9.py \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --pose-dir data/single_signer/processed/pose \
  --metadata-csv meta.csv \
  --train-index data/single_signer/index/train.csv \
  --val-index data/single_signer/index/val.csv \
  --work-dir work_dirs/single_signer_experiment \
  --tokenizer hf-internal-testing/tiny-random-T5 \
  --epochs 40 --batch-size 4 --sequence-length 64
```

Artefactos generados en `--work-dir`:

- `last.pt`: checkpoint de la última época.
- `best.pt`: mejor pérdida de validación observada.
- `config.json`: configuración efectiva combinando defaults, archivo y CLI.
- `metrics.jsonl`: historial con pérdidas y métricas por época.

## Argumentos destacados

### Datos

| Bandera | Descripción |
|---------|-------------|
| `--face-dir`, `--hand-left-dir`, `--hand-right-dir` | Directorios de recortes RGB. |
| `--pose-dir` | Directorio con los `.npz` de pose. |
| `--metadata-csv` | Ruta a `meta.csv`. |
| `--keypoints-dir` | Directorio opcional con keypoints MediaPipe (.npy/.npz). |
| `--gloss-csv` | CSV con columnas `video_id;gloss;ctc_labels` para pérdidas MSKA. |
| `--train-index`, `--val-index` | Listas de `video_id` para cada split. |
| `--batch-size`, `--val-batch-size` | Tamaños de lote de entrenamiento y validación. |
| `--num-workers`, `--pin-memory` | Parámetros del `DataLoader`. |
| `--mix-stream STREAM[:P]` | Permuta streams individuales con probabilidad `P`. |
| `--max-target-length` | Longitud máxima de la secuencia tokenizada. |

### Modelo

| Bandera | Descripción |
|---------|-------------|
| `--image-size`, `--sequence-length` | Resolución espacial y temporal del encoder. |
| `--projector-dim`, `--d-model` | Dimensiones internas del encoder. |
| `--pose-landmarks` | Cantidad de puntos de pose (se multiplica por 3 ejes). |
| `--temporal-*` | Configuración del transformer temporal. |
| `--decoder-layers`, `--decoder-heads`, `--decoder-dropout` | Arquitectura del decoder. |
| `--decoder-model`, `--decoder-config` | Modelos/configs HuggingFace precargados. |
| `--decoder-class`, `--decoder-kwargs` | Decoder Python personalizado vía módulo/clase. |
| `--pretrained` | Selecciona `single_signer` (default) o `none` para inicializar pesos. |
| `--pretrained-checkpoint` | Ruta al checkpoint `single_signer` descargado. |
| `--use-mska` | Activa la rama MSKA (requiere keypoints y glosas). |
| `--mska-*` | Hiperparámetros MSKA (`heads`, `ff-multiplier`, `dropout`, `input-dim`, `ctc-vocab`, `detach-teacher`). |
| `--mska-translation-weight`, `--mska-ctc-weight`, `--mska-distillation-weight` | Pesos de la combinación de pérdidas. |
| `--mska-distillation-temperature` | Temperatura aplicada al término de distilación. |

### Optimización y ejecución

| Bandera | Descripción |
|---------|-------------|
| `--optimizer`, `--lr`, `--weight-decay` | Hiperparámetros del optimizador (`AdamW` por defecto). |
| `--scheduler` + parámetros | `none`, `steplr` o `cosine` con banderas asociadas. |
| `--label-smoothing` | Factor aplicado a la pérdida de entropía cruzada. |
| `--precision {amp,float32}` | Controla AMP automático en GPU. |
| `--init-checkpoint` | Inicializa pesos desde un checkpoint previo. |
| `--resume` | Restaura entrenamiento completo (modelo, optimizador, scaler). |
| `--seed`, `--device` | Control determinista y selección de dispositivo. |

## Plantillas de configuración (`--config`)

El script acepta archivos JSON/YAML con secciones `data`, `model`, `optim` y
`training`. Ejemplo mínimo:

```yaml
data:
  face_dir: data/single_signer/processed/face
  hand_left_dir: data/single_signer/processed/hand_l
  hand_right_dir: data/single_signer/processed/hand_r
  pose_dir: data/single_signer/processed/pose
  metadata_csv: meta.csv
  train_index: data/single_signer/index/train.csv
  val_index: data/single_signer/index/val.csv
  work_dir: work_dirs/single_signer_experiment
model:
  sequence_length: 64
  image_size: 224
training:
  epochs: 40
optim:
  lr: 0.0005
```

Sobrescribe campos puntuales usando `--set clave=valor`, por ejemplo:

```bash
python tools/train_slt_multistream_v9.py \
  --config configs/single_signer.yml \
  --set data.batch_size=6 \
  --set optim.scheduler="cosine" \
  --set model.decoder_heads=6
```

Los valores efectivos se persistirán en `config.json` dentro de `work_dir`.

### Gestión avanzada de configuraciones

- Usa notación de puntos para campos anidados (`data.batch_size=6`).
- Valores booleanos aceptan `true`/`false` (insensible a mayúsculas).
- Strings con espacios deben ir entre comillas (`"cosine warmup"`).
- Repite `--set` tantas veces como sea necesario; la última definición prevalece.

Guarda los archivos YAML/JSON utilizados junto al `work_dir` para asegurar
reproducibilidad. Los diffs de configuración se listan en `config.json` bajo la
clave `overrides`.

## Decodificadores personalizados

Para usar un decoder propio, proporciona `--decoder-class` con el formato
`paquete.modulo:Clase` y argumentos opcionales en JSON:

```bash
python tools/train_slt_multistream_v9.py \
  ... \
  --decoder-class my_project.decoders:MyDecoder \
  --decoder-kwargs '{"dropout": 0.0, "share_embeddings": true}'
```

Verifica que la dimensionalidad (`d_model`) del decoder coincida con el encoder
multi-stream. El script valida este punto al inicializar el modelo.

### Mezcla de streams y augmentations

- `--mix-stream stream[:prob]`: intercambia aleatoriamente streams entre clips
  durante el entrenamiento. Omite `:prob` para usar el valor por defecto `1.0`.
  Los nombres válidos son `face`, `hand-left`, `hand-right` y `pose`.
- En archivos `--config` puedes definir `mix_streams` como diccionario:
  ```yaml
  data:
    mix_streams:
      face: 0.5
      hand_left: 0.25
  ```
  Las claves usan guion bajo para coincidencia con los campos del dataset.

`normalise_mix_spec` garantiza que las probabilidades se ajusten al rango `[0,1]`
y emite errores descriptivos cuando la configuración es inválida.

### Optimización avanzada

- `--grad-accum-steps`: acumula gradientes para simular lotes más grandes.
- `--clip-grad-norm`: aplica *gradient clipping* antes de cada actualización.
- `--compile` / `--no-compile`: activa `torch.compile` con el modo indicado en
  `--compile-mode` cuando la versión de PyTorch lo soporta.
- `--precision amp`: habilita *automatic mixed precision* en GPU.

## Reanudaciones y warm start

- `--resume`: requiere que `--work-dir` contenga `last.pt`. Restaura optimizador,
  scaler de AMP y contadores internos.
- `--init-checkpoint`: carga solo los pesos del modelo antes de comenzar la
  primera época. Útil para transferir pesos desde otra sesión.

## Métricas y seguimiento

- El archivo `metrics.jsonl` guarda un objeto JSON por época con pérdidas de
  entrenamiento y validación.
- Usa `tools/ci_validate_metrics.py` para comparar las métricas contra valores de
  referencia (se ejecuta en CI).
- La bandera `--tensorboard` habilita logging directo compatible con
  `tensorboard --logdir work_dirs/...`.
- `work_dir/logs/` almacena los archivos de texto con métricas resumidas por
  época y mensajes de `logging`.

Los reportes incluyen información de tiempo por iteración, tasa de samples y
distribución de pérdidas. Conserva estos artefactos para facilitar auditorías y
reproducibilidad.

## Siguientes pasos

1. Evalúa los checkpoints con `tools/eval_slt_multistream_v9.py` siguiendo el
   flujo descrito en el README.
2. Exporta el encoder con `tools/export_onnx_encoder_v9.py` para integrarlo en
   demos en tiempo real o servicios de inferencia.
3. Si necesitas sustituir backbones, revisa `docs/pretraining.md` para cargar
   pesos provenientes de DINO/iBOT mediante `load_dinov2_backbone`.
