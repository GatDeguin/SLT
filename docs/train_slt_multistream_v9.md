# `tools/train_slt_multistream_v9.py`

El script `tools/train_slt_multistream_v9.py` entrena el stub multi-stream
incluido en el paquete `slt`, conectando el dataset `LsaTMultiStream`, el
codificador `MultiStreamEncoder` y un decodificador seq2seq basado en
HuggingFace (`TextSeq2SeqDecoder`). Permite registrar métricas básicas, guardar
checkpoints y activar, de forma opcional, un hook de TensorBoard.

## Requisitos de entrada

El entrenamiento necesita los siguientes recursos:

* Cuatro carpetas con los streams sincronizados (`face`, `hand_l`, `hand_r` y
  `pose`). El stream de pose debe contener archivos `.npz` con la clave `pose`.
* Un CSV principal con columnas `video_id` y `texto`.
* Dos CSV adicionales con las listas de `video_id` para los splits de
  entrenamiento y validación.

## Uso básico

```bash
python tools/train_slt_multistream_v9.py \
  --face-dir data/rois/face \
  --hand-left-dir data/rois/hand_l \
  --hand-right-dir data/rois/hand_r \
  --pose-dir data/rois/pose \
  --metadata-csv data/lsa_t/subs.csv \
  --train-index data/lsa_t/index/train.csv \
  --val-index data/lsa_t/index/val.csv \
  --work-dir work_dirs/multistream_v9 \
  --epochs 40 --batch-size 4
```

Durante el entrenamiento se guardan dos archivos en `--work-dir`:

* `last.pt`: checkpoint de la última época.
* `best.pt`: checkpoint con la mejor pérdida de validación observada.

Además, el script escribe la configuración efectiva en `config.json` para
facilitar la reproducibilidad.

## Argumentos principales

### Datos

| Argumento | Descripción |
|-----------|-------------|
| `--face-dir`, `--hand-left-dir`, `--hand-right-dir`, `--pose-dir` | Rutas a los streams preprocesados. |
| `--metadata-csv` | CSV con `video_id` y texto de referencia. |
| `--train-index`, `--val-index` | Listas de `video_id` a usar en cada split. |
| `--batch-size`, `--val-batch-size` | Tamaños de batch para entrenamiento y validación. |
| `--num-workers` | Número de workers de `DataLoader`. |
| `--tensorboard` | Ruta opcional para habilitar logging en TensorBoard. |
| `--mix-stream` | Permite permutar streams concretos con una probabilidad dada (`STREAM[:P]`). |

### Modelo

| Argumento | Descripción |
|-----------|-------------|
| `--image-size` | Resolución de entrada para los ViT stub. |
| `--sequence-length` | Número de frames muestreados por clip. |
| `--projector-dim`, `--d-model` | Dimensiones internas del encoder. |
| `--pose-landmarks` | Cantidad de puntos de pose (se asume `3 * landmarks` canales). |
| `--temporal-*` | Hiperparámetros del transformer temporal. |
| `--decoder-layers`, `--decoder-heads`, `--decoder-dropout` | Arquitectura del decoder seq2seq. |
| `--decoder-model`, `--decoder-config` | Selección de modelos/configuraciones de HuggingFace para el decoder. |
| `--decoder-class`, `--decoder-kwargs` | Permiten instanciar un decoder Python personalizado con parámetros adicionales. |
| `--tokenizer` | Identificador o ruta a un tokenizer de HuggingFace. |
| `--max-target-length` | Longitud máxima de las secuencias tokenizadas. |

### Optimización

| Argumento | Descripción |
|-----------|-------------|
| `--optimizer`, `--lr`, `--weight-decay` | Configuración del optimizador (vía `slt.training.optim`). |
| `--scheduler` | Selector del scheduler (`none`, `steplr`, `cosine`). |
| `--scheduler-step-size`, `--scheduler-gamma`, `--scheduler-tmax` | Parámetros del scheduler. |
| `--label-smoothing` | Factor de *label smoothing* para la pérdida. |
| `--init-checkpoint` | Carga pesos iniciales antes de comenzar el entrenamiento. |

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

### Warm start desde checkpoints

Arranca el entrenamiento con pesos preexistentes (sin restaurar el optimizador)
gracias a `--init-checkpoint`:

```bash
python tools/train_slt_multistream_v9.py \
  ... \
  --init-checkpoint work_dirs/multistream_v9/best.pt
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

## Notas adicionales

* La opción `--precision amp` habilita *automatic mixed precision* cuando hay
  GPU disponible. Para ejecutar únicamente en CPU utilice `--precision fp32` o
  establezca `--device cpu`.
* Si se solicita TensorBoard y la librería no está instalada se imprimirá un
  aviso, continuando sin logging adicional.
* Los textos del dataset se tokenizan con el tokenizer indicado por
  `--tokenizer`. Las etiquetas utilizan `-100` en posiciones de padding para
  ser ignoradas por la pérdida.

## Evaluación y despliegue

Tras entrenar el modelo, evalúa el checkpoint con `tools/eval_slt_multistream_v9.py`
para obtener métricas de pérdida, CER y BLEU:

```bash
python tools/eval_slt_multistream_v9.py \
  --checkpoint work_dirs/multistream_v9/best.pt \
  --face-dir data/rois/face \
  --metadata-csv data/lsa_t/subs.csv \
  --index data/lsa_t/index/test.csv \
  --tokenizer hf-internal-testing/tiny-random-T5
```

Para desplegar únicamente el encoder, genera artefactos ONNX/TorchScript con
`tools/export_onnx_encoder_v9.py` y valida el resultado en `tools/demo_realtime_multistream.py`
o `tools/test_realtime_pipeline.py`. Estos pasos están automatizados en los tests
(`tests/test_export.py`) y en el flujo de CI descrito en el `README.md`.
