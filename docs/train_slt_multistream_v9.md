# Guía de entrenamiento `tools/train_slt_multistream_v9.py`

El script `tools/train_slt_multistream_v9.py` entrena el encoder multi-stream y
un decoder seq2seq compatible con HuggingFace utilizando los datos preparados en
`data/single_signer/processed/` y los índices derivados de `meta.csv`. Esta guía
resume los argumentos más importantes, patrones de uso avanzados y recomendaciones
para reproducir experimentos.

## Requisitos previos

- Streams generados con `tools/extract_rois_v2.py` siguiendo el contrato de datos.
- CSV `meta.csv` con columnas `id;texto` y splits `train.csv`/`val.csv`/`test.csv`.
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
| `--face-dir`, `--hand-left-dir`, `--hand-right-dir`, `--pose-dir` | Directorios con los streams procesados. |
| `--metadata-csv` | Ruta a `meta.csv`. |
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

## Siguientes pasos

1. Evalúa los checkpoints con `tools/eval_slt_multistream_v9.py` siguiendo el
   flujo descrito en el README.
2. Exporta el encoder con `tools/export_onnx_encoder_v9.py` para integrarlo en
   demos en tiempo real o servicios de inferencia.
3. Si necesitas sustituir backbones, revisa `docs/pretraining.md` para cargar
   pesos provenientes de DINO/iBOT mediante `load_dinov2_backbone`.
