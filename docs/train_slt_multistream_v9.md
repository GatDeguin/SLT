# `tools/train_slt_multistream_v9.py`

El script `tools/train_slt_multistream_v9.py` entrena el stub multi-stream
incluido en el paquete `slt`, conectando el dataset `LsaTMultiStream`, el
codificador `MultiStreamEncoder` y el decodificador ligero `TextDecoderStub`.
Permite registrar métricas básicas, guardar checkpoints y activar, de forma
opcional, un hook de TensorBoard.

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

### Modelo

| Argumento | Descripción |
|-----------|-------------|
| `--image-size` | Resolución de entrada para los ViT stub. |
| `--sequence-length` | Número de frames muestreados por clip. |
| `--projector-dim`, `--d-model` | Dimensiones internas del encoder. |
| `--pose-landmarks` | Cantidad de puntos de pose (se asume `3 * landmarks` canales). |
| `--temporal-*` | Hiperparámetros del transformer temporal. |
| `--vocab-size` | Salida del `TextDecoderStub`. |

### Optimización

| Argumento | Descripción |
|-----------|-------------|
| `--optimizer`, `--lr`, `--weight-decay` | Configuración del optimizador (vía `slt.training.optim`). |
| `--scheduler` | Selector del scheduler (`none`, `steplr`, `cosine`). |
| `--scheduler-step-size`, `--scheduler-gamma`, `--scheduler-tmax` | Parámetros del scheduler. |
| `--label-smoothing` | Factor de *label smoothing* para la pérdida. |

## Notas adicionales

* La opción `--precision amp` habilita *automatic mixed precision* cuando hay
  GPU disponible. Para ejecutar únicamente en CPU utilice `--precision fp32` o
  establezca `--device cpu`.
* Si se solicita TensorBoard y la librería no está instalada se imprimirá un
  aviso, continuando sin logging adicional.
* Los textos del dataset se transforman en etiquetas enteras mediante un hash
  determinista, lo que permite ejecutar el stub sin un tokenizador externo.
