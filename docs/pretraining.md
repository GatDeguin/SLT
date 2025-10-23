# Preentrenamiento de backbones con DINO/iBOT

Los scripts `tools/pretrain_dino_face.py` y `tools/pretrain_dino_hands.py`
permiten entrenar pesos auto-supervisados sobre los recortes generados en
`data/single_signer/processed/`. Ambos comparten la misma interfaz y producen
backbones compatibles con `load_dinov2_backbone`.

## Flujo recomendado

1. Ejecuta `tools/extract_rois_v2.py` y verifica la estructura documentada en el
   contrato de datos.
2. Selecciona el stream a preentrenar (`face`, `hand_l` o `hand_r`).
3. Define un `work_dir` exclusivo para cada experimento con suficiente espacio
   para checkpoints y métricas (mínimo 5 GB por sesión prolongada).
4. Lanza el entrenamiento con los parámetros deseados. Ejemplo básico:
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
   Activa el regularizador KoLeo cuando busques diversificar los embeddings del
   estudiante:
   ```bash
   python tools/pretrain_dino_face.py \
     --train-dir data/single_signer/processed/face \
     --output-dir work_dirs/dino_face \
     --koleo-weight 0.5 \
     --koleo-epsilon 1e-4
   ```
5. Repite el proceso para manos con `tools/pretrain_dino_hands.py` ajustando
   `--train-dir` a `hand_l` o `hand_r`.
6. Carga los backbones exportados desde `tools/train_slt_multistream_v9.py`
   mediante `--face-backbone`, `--hand-left-backbone` o `--hand-right-backbone`.

Para sesiones distribuidas, lanza el script con `torchrun --nproc_per_node=N` y
las banderas `--distributed` disponibles en el módulo `_pretrain_dino.py`.

## Configuración declarativa

Ambos scripts aceptan archivos TOML/JSON vía `--config`. Ejemplo en TOML:

```toml
train_dir = ["data/single_signer/processed/face"]
output_dir = "work_dirs/dino_face"
algorithm = "ibot"
epochs = 50

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
last_name = "checkpoint_last.pt"
best_name = "checkpoint_best.pt"
history_file = "metrics.jsonl"

[experiment]
name = "face-ibot-baseline"
tags = ["face", "ibot"]
notes = "Primer experimento con augmentations agresivos."
```

Los parámetros definidos en el archivo actúan como defaults y pueden
sobrescribirse desde la CLI.

### Regularización KoLeo

- `--koleo-weight`: peso (por defecto 0) aplicado sobre la suma de pérdidas KoLeo por crop
  global.
- `--koleo-epsilon`: margen numérico para evitar distancias nulas al estimar la entropía.
  Mantiene un valor base de `1e-4`, alineado con la implementación empleada por DINOv2.

## Artefactos generados

- `params.json`: resumen de hiperparámetros y rutas utilizadas.
- `metrics.jsonl`: historial de pérdidas y métricas por iteración/época.
- `artifacts.json`: registro de checkpoints y pesos exportados.
- `checkpoint_last.pt` y `checkpoint_best.pt`: incluyen modelo, optimizador y
  estado EMA (si aplica), listos para reanudar con `--resume`.
- `backbone.pt`: tensor con pesos listos para integrarse en el encoder
  multi-stream mediante `load_dinov2_backbone`.

## Integración con el pipeline principal

Para reutilizar los pesos en el entrenamiento multi-stream:

```bash
python tools/train_slt_multistream_v9.py \
  --config configs/single_signer.yml \
  --set model.face_backbone "file::work_dirs/dino_face/backbone.pt:dinov2_vits14" \
  --set model.freeze_face_backbone true
```

El prefijo `file::` indica que se cargará un checkpoint local. Es posible
combinarlo con backbones distintos para cada mano.

El pipeline unificado combina estos backbones con la rama de keypoints MSKA.
Cuando `--use-mska` está activo, asegúrate de mantener sincronizadas las rutas
de keypoints y glosas para que las pérdidas de traducción, CTC y distilación
puedan evaluarse de forma conjunta durante entrenamiento y validación.

## Buenas prácticas

- Versiona los archivos de configuración junto a los pesos exportados.
- Documenta notas del experimento con `--experiment-notes` y etiquetas usando
  `--experiment-tag`.
- Activa `--dataset-persistent-workers` en sistemas con múltiples núcleos para
  reducir la sobrecarga de `DataLoader`.
- Ajusta `--algorithm` (`dino` o `ibot`) y sus hiperparámetros asociados según el
  objetivo del preentrenamiento.
- Acompaña cada release con un extracto de `metrics.jsonl` o gráficos generados
  a partir de dicho archivo para seguir la convergencia.
- Define nombres específicos en la sección `[checkpointing]` para conservar
  múltiples checkpoints relevantes y facilitar comparaciones entre sesiones.
