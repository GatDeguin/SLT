# SLT — Demo de entrenamiento

Este repositorio incluye una implementación modular del stub multi-stream
utilizado durante la experimentación con el corpus LSA-T. El comando
`python -m slt` reproduce el ejemplo del archivo "Proyecto": crea el dataset,
los *DataLoaders* y ejecuta un entrenamiento corto con los modelos de prueba
incluidos en el paquete.

## Requisitos de datos

La demostración espera la misma estructura de carpetas generada por el script
`extract_rois_v2.py`:

- `face/<video_id>_fXXXXXX.jpg`
- `hand_l/<video_id>_fXXXXXX.jpg`
- `hand_r/<video_id>_fXXXXXX.jpg`
- `pose/<video_id>.npz` con la clave `pose`

Además se necesitan:

- Un CSV principal (`video_id;texto`).
- Dos CSV con los identificadores para los splits de entrenamiento y
  validación.

## Ejecución rápida

```bash
python -m slt \
  --face-dir data/rois/face \
  --hand-left-dir data/rois/hand_l \
  --hand-right-dir data/rois/hand_r \
  --pose-dir data/rois/pose \
  --metadata-csv data/lsa_t/subs.csv \
  --train-index data/lsa_t/index/train.csv \
  --val-index data/lsa_t/index/val.csv \
  --work-dir work_dirs/demo \
  --batch-size 2 --epochs 2
```

El script guardará `last.pt` y `best.pt` en `--work-dir` y mostrará en consola
la pérdida de entrenamiento/validación por época. Ajusta los parámetros según
la disponibilidad de hardware (por ejemplo `--device cpu` para forzar la
Ejecución en CPU).

## Sustituir los stubs por modelos reales

Los componentes incluidos en el paquete están pensados para ser reemplazados
por modelos de producción:

1. **Backbones / proyectores**: `slt/models/backbones.py` expone
   `ViTSmallPatch16` como un stub ligero y `slt/models/modules.py` define
   `StreamProjector` y `FuseConcatLinear`. Puedes extenderlos o sobrescribirlos
   para cargar pesos de DINOv2 y proyectores oficiales (método
   `replace_with_dinov2`).
2. **Encoder multi-stream**: `slt/models/multistream.py` centraliza la lógica
   de fusión y temporales. Sustituye sus dependencias por las variantes reales
   y ajusta la máscara de manos si cuentas con detección de frames perdidos.
3. **Decoder textual**: `slt/models/temporal.py` implementa
   `TextSeq2SeqDecoder`, una envoltura sobre modelos seq2seq de HuggingFace.
   Ajusta la configuración (tokenizer, arquitectura, longitud máxima) para
   cargar tu modelo de producción o inicializar pesos pre-entrenados.

Una vez actualizados estos módulos, la demo servirá como punto de partida para
un pipeline de entrenamiento completo con pesos reales.

## Preentrenamiento con DINO/iBOT

Los scripts `tools/pretrain_dino_face.py` y `tools/pretrain_dino_hands.py`
permiten generar pesos auto-supervisados compatibles con
`load_dinov2_backbone`. Consulta la guía en `docs/pretraining.md` para conocer
las opciones disponibles (DINO/iBOT, *multi-crop*, *EMA*, exportación de
backbones) y cómo integrarlos con `MultiStreamEncoder`.
