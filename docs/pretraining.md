# Preentrenamiento DINO/iBOT

Los scripts `tools/pretrain_dino_face.py` y `tools/pretrain_dino_hands.py`
permiten ejecutar experimentos de auto-supervisión ligeros sobre recortes de
rostro o manos. Ambos comparten la misma lógica interna y exponen una CLI con
opciones avanzadas que cubren aspectos habituales del entrenamiento DINO/iBOT.

## Ejemplo rápido

```bash
python tools/pretrain_dino_face.py \
  --train-dir data/rois/face \
  --output-dir work_dirs/dino_face \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --algorithm dino \
  --export-backbone work_dirs/dino_face/backbone.pt
```

El parámetro `--algorithm` puede alternarse entre `dino` e `ibot`. El modelo
subyacente es un `ViTSmallPatch16` configurable mediante banderas como
`--image-size`, `--patch-size`, `--vit-depth` o `--vit-embed-dim`. El script
admite *warmup* de `learning rate`, programaciones cosenoidales, actualización
EMA del maestro, *gradient clipping* y máscaras de parches para iBOT.

### Configuración declarativa

Los parámetros de dataset, augmentations, checkpoints y metadatos del
experimento pueden centralizarse en un archivo TOML/JSON mediante la bandera
`--config`. El siguiente ejemplo define directorios relativos, activa
`persistent_workers` y documenta el experimento:

```toml
train_dir = ["../datasets/face_train"]
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
comandos lo sobrescribe. Las nuevas banderas `--dataset-pin-memory`,
`--dataset-persistent-workers`, `--aug-*` y `--checkpoint-*` permiten ajustar
los componentes sin editar código.

## Exportación de pesos

Cuando se especifica `--export-backbone`, se escribe un archivo compatible con
`load_dinov2_backbone`. Para cargar los pesos en cualquier componente del
paquete basta con indicar la ruta y el modelo stub:

```python
from slt.models import load_dinov2_backbone

backbone = load_dinov2_backbone("file::path/al/backbone.pt:slt_vitsmall_patch16")
```

El propio `MultiStreamEncoder` puede recibir estos backbones a través del
argumento `backbones` al instanciarse.

## Reanudar y checkpoints

Cada época genera `checkpoint_last.pt` y se mantiene automáticamente el mejor
checkpoint (`checkpoint_best.pt`). Ambos incluyen el estado del optimizador, el
*global step*, los proyectores y los pesos del maestro, facilitando la
reanudar con `--resume`. Los nombres y la ruta del historial (`metrics.jsonl`)
pueden personalizarse con `--checkpoint-last-name`, `--checkpoint-best-name` y
`--checkpoint-history-file` o a través del archivo de configuración.

## Conjuntos de datos

Los datasets de entrenamiento se esperan como carpetas con imágenes sueltas
(PNG, JPG, JPEG, BMP, TIFF). Las utilidades de `tools/pretrain_utils.py`
proporcionan el `DataLoader` y las transformaciones multi-crop necesarias para
DINO/iBOT sin depender de `torchvision`.

## Seguimiento de experimentos

La carpeta de `output_dir` mantiene tres artefactos clave:

* `params.json`: parámetros del experimento (modelo, augmentations, dataset,
  rutas utilizadas y metadatos declarados).
* `metrics.jsonl`: historial en formato JSONL con las pérdidas por iteración,
  época y los eventos de *best model*.
* `artifacts.json`: registro de checkpoints y pesos exportados con su ruta y
  metadatos asociados.

Estos archivos permiten reproducir y auditar cada corrida sin depender de
servicios externos.

## Mejores prácticas

* Versioná los archivos de configuración junto con los pesos exportados.
* Documentá cambios relevantes en `params.json` usando `--experiment-notes`.
* Reutilizá `--experiment-tag` para etiquetar variantes (por ejemplo
  `face`, `hands`, `ibot`).
* Al usar múltiples workers, activá `--dataset-persistent-workers` para reducir
  el overhead de creación de procesos.
