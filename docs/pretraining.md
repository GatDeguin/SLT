# Preentrenamiento de backbones con DINO/iBOT

Los scripts `tools/pretrain_dino_face.py`, `tools/pretrain_dino_hands.py` y
`tools/pretrain_dinov2_multistream.py` permiten entrenar pesos auto-supervisados
sobre los recortes generados en `data/single_signer/processed/`. Todos comparten
la misma interfaz básica y producen backbones compatibles con
`load_dinov2_backbone`.

## Preparar crops masivos para preentrenamiento

1. LSA-T utiliza la metadata global `meta.csv` (del repositorio) con columnas
   `video` y rangos temporales (`start`, `end`). Verifica que cualquier fuente
   externa incluya sus videos en dicho CSV antes de lanzar el muestreo.
2. Ejecuta un sondeo inicial para estimar recursos y descubrir rutas con
   `python tools/prepare_lsat_crops.py --lsa-root data/lsa_t/videos --dry-run`.
   El resumen lista hasta 10 videos, su cantidad de clips anotados y la duración
   total cubierta por `meta.csv`.
3. Genera los crops con:
   ```bash
   python tools/prepare_lsat_crops.py \
     --lsa-root data/lsa_t/videos \
     --output-root data/single_signer/processed_lsat \
     --metadata data/single_signer/processed_lsat/lsat_metadata.jsonl \
     --fps 25 --shuffle --target-crops 3_000_000
   ```
   El script aplica las mismas ROI que `extract_rois_v2.py`, respeta `--resume`
   si el archivo JSONL ya contiene entradas exitosas y detiene el recorrido una
   vez alcanzado el número de frames solicitado.
4. Mezcla rutas externas con globs adicionales. Por ejemplo, para integrar clips
   curados en `data/signoteca/*.mp4` y `gs://bucket/proyectos/extra/*.mp4`:
   ```bash
   python tools/prepare_lsat_crops.py \
     --lsa-root data/lsa_t/videos \
     --extra-datasets "data/signoteca/*.mp4" "gs://bucket/proyectos/extra/*.mp4" \
     --output-root data/single_signer/processed_lsat --shuffle
   ```
   Todas las rutas detectadas deben compartir IDs con `meta.csv`; el helper
   aborta si encuentra archivos sin fila asociada para evitar inconsistencias.

### Estimar recursos

- **CPU**: MediaPipe corre en CPU. Un servidor de 16 núcleos logra entre 140 y
  180 FPS agregados (≈6–8 FPS por núcleo) con los tres detectores activos.
- **GPU**: no es obligatoria; sólo se emplea la CPU.
- **Almacenamiento**: cada frame produce tres JPEG de 224×224 (rostro y dos
  manos). Con una compresión media de 28 KB por imagen, 1 millón de frames
  equivale a ~84 GB (`3 × 28 KB × 10^6`). Añade ~1.2 GB por cada millón de
  poses (`.npz`). Reserva al menos un 15 % adicional para márgenes y metadatos.
- **Tiempo**: para 1 millón de frames a 150 FPS efectivos el proceso demora
  ≈1.85 horas (`1e6 / 150 / 3600`). Ajusta `--target-crops` y `--shuffle` para
  escalonar ejecuciones largas en bloques reproducibles.

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
   Añade `--export-checkpoint` junto a `--output-path` para generar un
   `state_dict` del encoder listo para inicializar MSKA.
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
6. Para un entrenamiento conjunto de rostro y manos, utiliza
   `tools/pretrain_dinov2_multistream.py` con las carpetas correspondientes a
   cada stream.
7. Carga los backbones exportados desde `tools/train_slt_multistream_v9.py`
   mediante `--face-backbone`, `--hand-left-backbone` o `--hand-right-backbone`.

### Ejemplo multi-stream

```bash
python tools/pretrain_dinov2_multistream.py \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --output-dir work_dirs/dino_multistream \
  --epochs 80 \
  --batch-size 48 \
  --learning-rate 1e-3 \
  --export-backbone work_dirs/dino_multistream/backbone
```

El parámetro `--export-backbone` actúa como prefijo y genera un archivo por
stream (`*_face.pt`, `*_hand_left.pt`, `*_hand_right.pt`) junto con un manifiesto
JSON (`*_manifest.json`) que detalla la configuración y rutas resultantes. Cada
archivo `.pt` es compatible con `load_dinov2_backbone` usando el prefijo
`file::`.

Para sesiones distribuidas, lanza el script con `torchrun --nproc_per_node=N` y
las banderas `--distributed` disponibles en el módulo `_pretrain_dino.py`.

## Configuración declarativa

Los scripts aceptan archivos TOML/JSON vía `--config`. Ejemplo en TOML:

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

Para multi-stream puedes declarar rutas específicas por stream:

```toml
face_train_dir = ["data/single_signer/processed/face"]
hand_left_train_dir = ["data/single_signer/processed/hand_l"]
hand_right_train_dir = ["data/single_signer/processed/hand_r"]
output_dir = "work_dirs/dino_multistream"
epochs = 60
batch_size = 48
export_backbone = "work_dirs/dino_multistream/backbone"
```

### Regularización KoLeo

- `--koleo-weight`: peso (por defecto 0) aplicado sobre la suma de pérdidas KoLeo por crop
  global.
- `--koleo-epsilon`: margen numérico para evitar distancias nulas al estimar la entropía.
  Mantiene un valor base de `1e-4`, alineado con la implementación empleada por DINOv2.

### Controlar vistas y pseudo-épocas

- `--global-crops`: cantidad de crops globales por imagen. El valor por defecto es 2 y se
  combina con 8 crops locales (`--num-local-crops`) para seguir la receta de DINOv2.
- `--pseudo-epochs`: repite el DataLoader dentro de cada época lógica. El scheduler de *learning
  rate* y el momentum del maestro multiplican automáticamente sus pasos por este valor, útil para
  datasets pequeños o experimentos de *fine-tuning*.

### Normalización Sinkhorn

- `--use-sinkhorn`: activa la proyección Sinkhorn antes de comparar las distribuciones del maestro
  y el estudiante, replicando la estrategia de DINOv2.
- `--sinkhorn-eps`: valor de epsilon (por defecto `0.05`) utilizado para suavizar la matriz antes de
  iterar.
- `--sinkhorn-iters`: número de iteraciones aplicadas sobre la rutina Sinkhorn (3 por defecto).

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
