# `tools/train_slt_multistream_v9.py`

El script `tools/train_slt_multistream_v9.py` entrena el stub multi-stream
incluido en el paquete `slt`, conectando el dataset `LsaTMultiStream`, el
codificador `MultiStreamEncoder` y un decoder seq2seq de HuggingFace. Utiliza la
misma estructura que la CLI principal (`python -m slt`) pero expone argumentos
adicionales para afinado fino, reanudación y registro de experimentos.

## Requisitos de entrada

El entrenamiento espera el mismo layout generado durante el flujo con
`single_signer` descrito en el README:

- Cuatro carpetas con los streams sincronizados (`face/`, `hand_l/`, `hand_r/`,
  `pose/`) ubicadas en `data/single_signer/processed/`.
- Un CSV con columnas `id` y `text` (ej. `meta.csv`).
- Archivos CSV con la lista de `video_id` para los splits de entrenamiento y
  validación (por ejemplo `data/single_signer/index/train.csv` y `val.csv`).

## Uso básico

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

Durante el entrenamiento se guardan dos archivos en `--work-dir`:

- `last.pt`: checkpoint de la última época.
- `best.pt`: checkpoint con la mejor pérdida de validación observada.

Además, el script escribe la configuración efectiva en `config.json` para
facilitar la reproducibilidad y crea un registro `metrics.jsonl` con las
métricas por época.

## Argumentos principales

### Datos

| Argumento | Descripción |
|-----------|-------------|
| `--face-dir`, `--hand-left-dir`, `--hand-right-dir`, `--pose-dir` | Directorios con streams. |
| `--metadata-csv` | CSV global (`meta.csv`) con el texto de referencia. |
| `--train-index`, `--val-index` | Listas de `video_id` a usar en cada split. |
| `--batch-size`, `--val-batch-size` | Tamaños de batch para entrenamiento y validación. |
| `--num-workers` | Número de workers de `DataLoader`. |
| `--tensorboard` | Ruta opcional para habilitar logging en TensorBoard. |
| `--mix-stream` | Permite permutar streams concretos con una probabilidad dada (`STREAM[:P]`). |

### Modelo

| Argumento | Descripción |
|-----------|-------------|
| `--image-size` | Resolución de entrada para los backbones. |
| `--sequence-length` | Número de frames muestreados por clip. |
| `--projector-dim`, `--d-model` | Dimensiones internas del encoder. |
| `--pose-landmarks` | Cantidad de puntos de pose (se asume `3 * landmarks` canales). |
| `--temporal-*` | Hiperparámetros del transformer temporal. |
| `--decoder-layers`, `--decoder-heads`, `--decoder-dropout` | Arquitectura del decoder seq2seq. |
| `--decoder-model`, `--decoder-config` | Selección de modelos/configuraciones de HuggingFace. |
| `--decoder-class`, `--decoder-kwargs` | Permiten instanciar un decoder Python personalizado. |
| `--tokenizer` | Identificador o ruta a un tokenizer de HuggingFace. |
| `--max-target-length` | Longitud máxima de las secuencias tokenizadas. |

### Optimización

| Argumento | Descripción |
|-----------|-------------|
| `--optimizer`, `--lr`, `--weight-decay` | Configuración del optimizador (`slt.training.optim`). |
| `--scheduler` | Selector del scheduler (`none`, `steplr`, `cosine`). |
| `--scheduler-step-size`, `--scheduler-gamma`, `--scheduler-tmax` | Parámetros del scheduler. |
| `--label-smoothing` | Factor de *label smoothing* para la pérdida. |
| `--init-checkpoint` | Carga pesos iniciales antes de comenzar el entrenamiento. |
| `--resume` | Restaura entrenamiento, optimizador y scaler desde `last.pt`. |

## Configuración avanzada

### Decodificadores personalizados

Puedes sustituir el decoder por cualquier clase Python compatible (heredando de
`torch.nn.Module`) usando `--decoder-class`. Opcionalmente pasa argumentos
adicionales vía `--decoder-kwargs` (JSON):

```bash
python tools/train_slt_multistream_v9.py \
  ... \
  --decoder-class my_project.decoders:MyDecoder \
  --decoder-kwargs '{"half_precision": true, "tie_embeddings": false}'
```

Si prefieres un modelo de HuggingFace concreto, utiliza `--decoder-model` o
`--decoder-config`; el script validará que la dimensionalidad coincide con el
encoder multi-stream.

> **Importante:** al usar checkpoints preentrenados asegúrate de que el modelo
> sea un decoder T5/BART con la misma dimensión (`d_model`) que el encoder.
> Checkpoints incompatibles fallarán durante la inicialización del decoder.

### Warm start desde checkpoints

Arranca el entrenamiento con pesos preexistentes (sin restaurar el optimizador)
mediante `--init-checkpoint`:

```bash
python tools/train_slt_multistream_v9.py \
  ... \
  --init-checkpoint work_dirs/single_signer_experiment/best.pt
```

Esta opción es independiente de `--resume`, que además recupera el optimizador,
el scaler de AMP y el estado de los RNG.

### Mezcla opcional de streams

`--mix-stream` aplica *mixup* sencillo permutando streams concretos dentro del
batch. Puede declararse varias veces para asignar probabilidades distintas:

```bash
python tools/train_slt_multistream_v9.py \
  ... \
  --mix-stream face:0.5 --mix-stream hand-left:0.3
```

Los nombres admitidos son `face`, `hand-left`, `hand-right` y `pose`.

### Plantillas de configuración

Además de la CLI, el comando acepta `--config config.yml` (JSON o YAML) y
sobre-escrituras puntuales con `--set data.batch_size=8`. Estas plantillas se
persisten en `config.json` para garantizar reproducibilidad.

Un ejemplo mínimo de plantilla es el siguiente:

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

Puedes modificar hiperparámetros concretos al vuelo manteniendo la plantilla
base sin cambios:

```bash
python tools/train_slt_multistream_v9.py \
  --config configs/single_signer.yml \
  --set data.batch_size=6 \
  --set optim.scheduler="cosine"
```

Los valores aplicados se combinan con los defaults del script y se registran en
`config.json` dentro de `work_dir`.

## Evaluación y exportación

Una vez finalizado el entrenamiento, evalúa el checkpoint con
`tools/eval_slt_multistream_v9.py` para generar predicciones y métricas (BLEU,
CHRF, CER) sobre los índices de prueba:

```bash
python tools/eval_slt_multistream_v9.py \
  --checkpoint work_dirs/single_signer_experiment/best.pt \
  --tokenizer hf-internal-testing/tiny-random-T5 \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --pose-dir data/single_signer/processed/pose \
  --metadata-csv meta.csv \
  --eval-index data/single_signer/index/test.csv \
  --output-csv work_dirs/single_signer_experiment/predictions/preds.csv
```

Para desplegar únicamente el encoder, genera artefactos ONNX/TorchScript con
`tools/export_onnx_encoder_v9.py` y prueba los resultados en las demos en tiempo
real:

```bash
python tools/export_onnx_encoder_v9.py \
  --checkpoint work_dirs/single_signer_experiment/best.pt \
  --onnx exports/single_signer_encoder.onnx \
  --torchscript exports/single_signer_encoder.ts \
  --image-size 224 --sequence-length 64 --d-model 512
```

Consulta `docs/pretraining.md` si necesitas sustituir los backbones stub por
pesos auto-supervisados previos.
