# Preentrenamiento DINO/iBOT

Los scripts `tools/pretrain_dino_face.py` y `tools/pretrain_dino_hands.py`
permiten ejecutar experimentos de auto-supervisión ligeros sobre los recortes de
rostro y manos obtenidos en `data/single_signer/processed/`. Ambos comparten la
misma implementación y aceptan configuraciones declarativas en JSON o TOML para
reproducir los experimentos.

## Ejemplo rápido

```bash
python tools/pretrain_dino_face.py \
  --train-dir data/single_signer/processed/face \
  --output-dir work_dirs/dino_face \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --algorithm dino \
  --export-backbone work_dirs/dino_face/backbone.pt
```

Cambia `--train-dir` por `data/single_signer/processed/hand_l` o
`hand_r` para entrenar proyectores específicos de manos. El stub utiliza
`ViTSmallPatch16`, configurable mediante banderas como `--image-size`,
`--patch-size`, `--vit-depth` o `--vit-embed-dim`. El script admite *warmup* de
`learning rate`, programaciones cosenoidales, actualización EMA del maestro,
*gradient clipping* y máscaras de parches para iBOT.

### Configuración declarativa

Los parámetros de dataset, augmentations, checkpoints y metadatos pueden
centralizarse en un archivo TOML/JSON mediante la bandera `--config`. El
siguiente ejemplo reutiliza los recortes generados en el flujo `single_signer`:

```toml
train_dir = ["data/single_signer/processed/face"]
output_dir = "work_dirs/dino_face"
epochs = 50
algorithm = "ibot"

[dataset]
batch_size = 48
num_workers = 4
pin_memory = true
persistent_workers = true

[augmentation]
brightness = 0.5
gaussian_blur_prob = 0.7
mean = [0.481, 0.457, 0.408]
std = [0.268, 0.261, 0.276]

[checkpointing]
best_name = "best.pt"
last_name = "last.pt"
history_file = "training_metrics.jsonl"

[experiment]
name = "face-ibot-baseline"
tags = ["face", "ibot", "baseline"]
notes = "Entrenamiento inicial con augmentations más agresivos."
```

El archivo actúa como valores por defecto; cualquier argumento de línea de
comandos lo sobrescribe. Las banderas `--dataset-pin-memory`,
`--dataset-persistent-workers`, `--aug-*` y `--checkpoint-*` permiten ajustar los
componentes sin editar código.

## Exportación e integración

Cuando se especifica `--export-backbone`, se escribe un archivo compatible con
`load_dinov2_backbone`. Para cargar los pesos en el encoder multi-stream, apunta
al archivo exportado desde la configuración de entrenamiento:

```bash
python tools/train_slt_multistream_v9.py \
  --config configs/single_signer.yaml \
  --face-dir data/single_signer/processed/face \
  --metadata-csv meta.csv \
  --train-index data/single_signer/index/train.csv \
  --val-index data/single_signer/index/val.csv \
  --model.backbones.face file::work_dirs/dino_face/backbone.pt:slt_vitsmall_patch16
```

Tras el entrenamiento, evalúa y exporta el encoder siguiendo las instrucciones
actualizadas en `docs/train_slt_multistream_v9.md`. Este flujo permite validar
que el backbone preentrenado se integra correctamente en los experimentos
multi-stream.

## Reanudar y checkpoints

Cada época genera `checkpoint_last.pt` y se mantiene automáticamente el mejor
checkpoint (`checkpoint_best.pt`). Ambos incluyen el estado del optimizador, el
*global step*, los proyectores y los pesos del maestro, lo que facilita la
reanudación con `--resume`. Los nombres y la ruta del historial (`metrics.jsonl`)
pueden personalizarse con `--checkpoint-last-name`, `--checkpoint-best-name` y
`--checkpoint-history-file` o a través del archivo de configuración.

## Conjuntos de datos

Los datasets de entrenamiento se esperan como carpetas con imágenes sueltas
(PNG, JPG, JPEG, BMP, TIFF). Las utilidades de `tools/pretrain_utils.py`
proporcionan el `DataLoader` y las transformaciones multi-crop necesarias para
DINO/iBOT sin depender de `torchvision`.

## Seguimiento de experimentos

La carpeta de `output_dir` mantiene tres artefactos clave:

- `params.json`: parámetros del experimento (modelo, augmentations, dataset,
  rutas utilizadas y metadatos declarados).
- `metrics.jsonl`: historial en formato JSONL con las pérdidas por iteración,
  época y los eventos de *best model*.
- `artifacts.json`: registro de checkpoints y pesos exportados con su ruta y
  metadatos asociados.

Estos archivos permiten reproducir y auditar cada corrida sin depender de
servicios externos.

## Mejores prácticas

- Versiona los archivos de configuración junto con los pesos exportados.
- Documenta cambios relevantes en `params.json` usando `--experiment-notes`.
- Reutiliza `--experiment-tag` para etiquetar variantes (por ejemplo `face`,
  `hands`, `ibot`).
- Al usar múltiples workers, activa `--dataset-persistent-workers` para reducir
  el overhead de creación de procesos.
