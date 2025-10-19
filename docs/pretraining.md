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
reanudación con `--resume`.

## Conjuntos de datos

Los datasets de entrenamiento se esperan como carpetas con imágenes sueltas
(PNG, JPG, JPEG, BMP, TIFF). Las utilidades de `tools/pretrain_utils.py`
proporcionan el `DataLoader` y las transformaciones multi-crop necesarias para
DINO/iBOT sin depender de `torchvision`.
