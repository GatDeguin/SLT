# SLT — Pipeline multi-stream para `single_signer`

Este repositorio centraliza las utilidades para preparar datos, entrenar,
evaluar, exportar y monitorear el modelo multi-stream validado para el flujo
`single_signer`. El pipeline opera sobre el dataset `single_signer`, cuyo CSV de
subtítulos principal es `meta.csv`, y sirve como plantilla para reproducir
experimentos o adaptar el flujo a nuevas variantes.

## Tabla de contenidos

1. [Resumen del pipeline](#resumen-del-pipeline)
2. [Arquitectura del paquete](#arquitectura-del-paquete)
3. [Instalación](#instalación)
4. [Flujo recomendado de extremo a extremo](#flujo-recomendado-de-extremo-a-extremo)
5. [Preparación de datos](#preparacion-de-datos-datasingle_signer)
6. [Entrenamiento y evaluación](#entrenamiento-y-evaluación)
7. [Exportación y demos](#exportación-y-demos-en-tiempo-real)
8. [Preentrenamiento de backbones](#preentrenamiento-de-backbones)
9. [Control de calidad y pruebas](#control-de-calidad-y-pruebas)
10. [Estructura de carpetas](#estructura-de-carpetas)
11. [Soporte y aportes](#soporte-y-aportes)

## Resumen del pipeline

- **Paquete `slt/`**: expone el dataset `LsaTMultiStream`, el encoder
  `MultiStreamEncoder`, un decoder seq2seq y utilidades para datos, métricas y
  entrenamiento.
- **Pipeline ROI + keypoints**: integra la extracción de recortes, el alineado
  de keypoints/glosas y el entrenamiento con pérdidas combinadas MSKA. Los
  detalles paso a paso están en `docs/train_slt_multistream_v9.md`.
- **Herramientas `tools/`**: scripts para extracción de ROIs, entrenamiento
  completo, evaluación, exportación, validación de contratos y demos en tiempo
  real (`docs/operational_checklist.md` lista los pasos sugeridos para releases).
- **Documentación `docs/`**: contratos de datos, guías operativas, manuales de
  preentrenamiento y resúmenes de papers que motivan el enfoque multi-stream.
- **Fine-tuning guiado**: `docs/finetuning.md` explica cómo reaprovechar los
  encoders de rostro/manos exportados y cargarlos dentro de MSKA.
- **Tests `tests/`**: suites de humo que cubren datos sintéticos,
  exportaciones y ejecuciones rápidas del pipeline.
- **Modo offline**: guía paso a paso en `docs/offline_cache.md` para preparar caches
  de tokenizer y decoder cuando la máquina no tiene acceso a Hugging Face.

Consulta `docs/data_contract.md`, `docs/train_slt_multistream_v9.md`,
`docs/pretraining.md` y `docs/finetuning.md` para ampliar cada etapa.
`tools/README.md` describe cada script CLI disponible.

## Arquitectura del paquete

`docs/architecture_overview.md` conecta todos los módulos del repositorio y detalla
cómo fluyen los datos desde los recortes hasta la inferencia en demos tiempo real.
Esta sección resume los componentes principales y dónde encontrarlos en el código
fuente.

- **Dataset multi-stream (`slt/data/lsa_t_multistream.py`)**: normaliza rostro,
  manos, pose y keypoints, emitiendo tensores y máscaras alineadas con el
  contrato de datos.
- **Encoder unificado (`slt/models/multistream.py`)**: proyecta cada stream,
  concatena las representaciones y combina la dinámica temporal junto con MSKA
  opcional.
- **Modelo entrenable (`slt/training/models.py`)**: encapsula el encoder, el
  decoder seq2seq y las cabezas auxiliares de traducción, CTC y distilación.
- **Bucles de entrenamiento (`slt/training/loops.py`)**: implementan
  `train_epoch`/`eval_epoch`, acumulación de gradiente y registro de métricas.
- **Runtime en vivo (`slt/runtime/realtime.py`)**: gestiona ventanas deslizantes y
  máscaras para demos con latencia controlada.
- **Parsers de CLI (`slt/utils/cli.py`)**: convierten rangos y overrides de
  configuración compartidos por las herramientas.

Los presets de decoder y las configuraciones declarativas (`configs/`) conectan
estas piezas con los scripts de `tools/`, facilitando reproducir experimentos o
adaptar el pipeline a otros datasets.

### Pipeline unificado ROI + keypoints

1. **Preprocesar ROIs y keypoints** con `tools/extract_rois_v2.py` y el pipeline
   de MediaPipe documentado en `docs/data_contract.md`. Asegúrate de exportar
   `metadata.jsonl` y los keypoints por video (`.npz`/`.npy`).
2. **Validar alineación y glosas** ejecutando
   `python tools/ci_validate_data_contract.py`, que comprueba máscaras,
   keypoints y etiquetas CTC/gloss en un dataset sintético.
3. **Entrenar con pérdidas combinadas** mediante
   `tools/train_slt_multistream_v9.py --use-mska`, definiendo los pesos de
   traducción, CTC y distilación según el escenario (ver ejemplos en
   `tools/README.md`). El wrapper `tools/train_slt_lsa_mska_v13.py` mantiene
   compatibilidad con scripts heredados.
4. **Evaluar y exportar** con `tools/eval_slt_multistream_v9.py` y
   `tools/export_onnx_encoder_v9.py`, reutilizando las mismas configuraciones de
   MSKA para reproducir métricas y artefactos.

### Pesos pre-entrenados

El repositorio no incluye el checkpoint validado para `single_signer` debido a
restricciones de tamaño. Descarga `single_signer_multistream.pt` desde la
ubicación compartida por el equipo y colócalo en
`data/single_signer/single_signer_multistream.pt` o expón su ruta mediante la
variable de entorno `SLT_SINGLE_SIGNER_CHECKPOINT`. Todas las CLI buscarán el
archivo siguiendo ese orden; puedes desactivarlo con `--pretrained none`. El
encoder puede instanciarse de forma directa con:

```python
from slt.models import MultiStreamEncoder

encoder = MultiStreamEncoder.from_pretrained(
    "single_signer",
    checkpoint_path="data/single_signer/single_signer_multistream.pt",
)
```

### Pesos Phoenix 2014

Acordamos utilizar el checkpoint MSKA entrenado sobre Phoenix 2014 distribuido
como `best.pth`. Coloca el archivo en `data/phoenix_2014/best.pth` o define la
variable `SLT_PHOENIX_CHECKPOINT` apuntando a su ubicación. El helper
`tools/fine_tune_phoenix_lsat.py` asume que `work_dirs/phoenix/best.pth` está
disponible y lanza un ajuste fino de 5 000 pasos con tasas diferenciadas, mientras
que las CLI en `tools/` siguen permitiendo cargarlo pasando
`--pretrained phoenix_2014` junto a
`--pretrained-checkpoint data/phoenix_2014/best.pth`.

## Instalación

Configura un entorno virtual, instala PyTorch compatible con tu hardware y
aplica los requisitos de desarrollo. `requirements-dev.txt` instala el paquete
en modo editable junto con herramientas de linting y pruebas.

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows usa .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Configurar entorno en Windows

Sigue estos pasos para habilitar todas las funcionalidades (extras `media`,
`metrics` y `export`) en Windows:

1. Instala **Python 3.10 (64 bits)** desde python.org y marca *Add python.exe to
   PATH* durante la instalación. MediaPipe publica ruedas hasta 3.12; 3.10 es la
   versión recomendada y totalmente compatible con el resto del pipeline.
2. Instala **Git for Windows** y, si tu entorno no cuenta con compiladores C++,
   añade *Microsoft C++ Build Tools* con la carga de trabajo *Desktop
   development with C++* para cubrir dependencias que requieran extensiones.
3. Instala **PyTorch** siguiendo la guía oficial para Windows, seleccionando la
   versión (CPU o CUDA) que corresponda a tu GPU:
   https://pytorch.org/get-started/locally/
4. Abre PowerShell en la carpeta del repositorio y crea el entorno virtual con
   `py -3.10`. Si la política de ejecución bloquea la activación, ejecuta
   `Set-ExecutionPolicy -Scope Process RemoteSigned -Force` en la misma sesión.

```powershell
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process RemoteSigned -Force  # Solo si es necesario
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

El archivo `requirements-dev.txt` instala el paquete en modo editable junto con
los extras opcionales. Tras la instalación, verifica que `mediapipe` esté
disponible ejecutando `pip show mediapipe` (en Python 3.10 se instala sin
limitaciones) y comprueba PyTorch con
`python -c "import torch; print(torch.__version__)"`.

### Extras opcionales

El paquete define grupos de extras que habilitan funcionalidades específicas:

| Extra | Propósito |
|-------|-----------|
| `media` | Seguimiento facial/manos con MediaPipe para `tools/extract_rois_v2.py`. |
| `metrics` | Cálculo de BLEU, ChrF, CER y WER durante la evaluación. |
| `export` | Exportación ONNX/TorchScript y validación en tiempo real. |

> Nota: el extra `media` depende de MediaPipe, disponible únicamente en Python <=3.12.
> Usa un entorno con esa versión (recomendado 3.10 en Windows) cuando necesites sus
> utilidades.

Instálalos en bloque con `pip install .[media,metrics,export]` en intérpretes
compatibles o agrega cada uno según tus necesidades. Consulta la sección de
[control de calidad](#control-de-calidad-y-pruebas) para conocer las
verificaciones recomendadas tras la instalación.

## Flujo recomendado de extremo a extremo

1. **Preparar datos** siguiendo el [contrato documentado](docs/data_contract.md).
   Ejecuta `tools/extract_rois_v2.py` sobre los videos y construye los splits en
   `data/single_signer/index/`.
2. **Verificar la instalación** corriendo `pytest` y los linters (`ruff`,
   `black`, `mypy`) para confirmar que el entorno está consistente.
3. **Ejecutar un entrenamiento rápido** con `python -m slt` para validar que los
   datos y el tokenizador son correctos. Define el `work_dir` donde se guardarán
   los checkpoints temporales.
4. **Lanzar experimentos completos** con `tools/train_slt_multistream_v9.py`,
   habilitando `--use-mska` cuando el dataset incluya keypoints y glosas. El
   wrapper `tools/train_slt_lsa_mska_v13.py` reenvía los argumentos al flujo
   unificado.
5. **Evaluar resultados** usando `tools/eval_slt_multistream_v9.py` y analiza los
   reportes con `docs/metrics_dashboard_integration.py` o tus dashboards.
6. **Exportar y validar** con `tools/export_onnx_encoder_v9.py` y las demos en
   tiempo real (`tools/demo_realtime_multistream.py`,
   `tools/test_realtime_pipeline.py`).
7. **Documentar y publicar** apoyándote en `docs/operational_checklist.md` antes
   de liberar artefactos o abrir un PR.

## Preparación de datos (`data/single_signer`)

1. Crea la estructura base dentro de `data/single_signer/`:
   ```text
   data/
     single_signer/
       videos/           # Clips fuente en MP4/MKV
       processed/
         face/
         hand_l/
         hand_r/
         pose/
         keypoints/
       annotations/
         gloss.csv
       index/
         train.csv
         val.csv
         test.csv
   meta.csv              # CSV con columnas video_id;texto
   ```
2. Copia los videos originales en `data/single_signer/videos/`.
3. Normaliza `meta.csv` antes de lanzar cualquier pipeline. El comando limpia
   separadores repetidos, descarta filas sin temporización y emite resúmenes de
   outliers junto con `meta_missing.csv` cuando corresponde:
   ```bash
   python tools/prepare_lsat_crops.py \
     --lsa-root data/single_signer/videos \
     --meta-csv meta.csv \
     --dry-run \
     --duration-threshold 20 \
     --delta-threshold 0.5 \
     --fail-on-outliers
   ```
   El `dry-run` evita la ejecución de MediaPipe pero mantiene la limpieza,
   dejando los archivos auxiliares junto al CSV original. Activa
   `--emit-split-json` para exportar `split_segments.jsonl` y reutilizar los
   subtítulos parciales en otros pipelines.
4. Ejecuta la extracción de regiones de interés para rostro, manos y pose:
   ```bash
   python tools/extract_rois_v2.py \
     --videos data/single_signer/videos \
     --output data/single_signer/processed \
     --metadata meta.csv
   ```
   El script genera `face/`, `hand_l/`, `hand_r/` y `pose/` junto a un
   `metadata.jsonl` con métricas por video. Reanuda ejecuciones con `--resume` si
   fuese necesario. Las poses se guardan normalizadas en `[0, 1]` dentro del
   *signing space* (ancho 6, alto 7 unidades de cabeza) y los `.npz` incluyen la
   clave `pose_norm="signing_space_v1"`. Cuando MediaPipe no reporta landmarks
   se replica la pose previa o se rellena con `-1` y visibilidad `0` como
   sentinel.
   Para LSA-T y corpus externos con millones de crops combina fuentes mediante
   `python tools/prepare_lsat_crops.py --lsa-root data/lsa_t/videos --output-root \
   data/single_signer/processed_lsat --extra-datasets "data/externo/**/*.mp4"`.
   El helper reaprovecha la metadata limpia, valida los IDs contra `meta.csv` y
   permite detenerse al alcanzar un número objetivo de frames con
   `--target-crops`. Mantiene los mismos controles de outliers descritos en
   [`docs/data_contract.md`](docs/data_contract.md#control-de-outliers) y puede
   emitir `split_segments.jsonl` al activar `--emit-split-json`.
5. Genera o copia los keypoints multistream en
   `data/single_signer/processed/keypoints/`. Cada archivo debe nombrarse como
   `<video_id>.npy` o `<video_id>.npz` y contener un arreglo `keypoints` en
   formato `(T, landmarks, 3)` con `(x, y, conf)` siguiendo el layout de
   MediaPipe. Si cuentas con anotaciones de glosa, agrégalas al CSV opcional
   `data/single_signer/annotations/gloss.csv` con columnas
   `video_id;gloss;ctc_labels` (índices separados por espacios).
6. Construye los splits desde `meta.csv` según tus criterios. Los CSV deben
   contener un `video_id` por línea sin encabezado. Un ejemplo mínimo:
   ```bash
   python - <<'PY'
   from pathlib import Path
   import pandas as pd

   meta = pd.read_csv('meta.csv', sep=';')
   ids = meta['video_id'].unique()
   out_dir = Path('data/single_signer/index')
   out_dir.mkdir(parents=True, exist_ok=True)
   pd.Series(ids[:80]).to_csv(out_dir / 'train.csv', index=False)
   pd.Series(ids[80:90]).to_csv(out_dir / 'val.csv', index=False)
   pd.Series(ids[90:]).to_csv(out_dir / 'test.csv', index=False)
   PY
   ```
7. Valida la estructura con `python tools/ci_validate_data_contract.py` o
   ejecuta `pytest tests/data/test_dataset_quality.py` para comprobar los
   avisos producidos ante inconsistencias de FPS, frames faltantes, keypoints
   ausentes o splits incompletos.

El contrato de datos completo se detalla en `docs/data_contract.md`, donde se
documentan los campos opcionales de `meta.csv`, las métricas registradas en
`metadata.jsonl` y las convenciones de nomenclatura de archivos.

## Entrenamiento y evaluación

### Entrenamiento rápido (`python -m slt`)

La CLI empaquetada ejecuta un entrenamiento corto para verificar el pipeline.
Por defecto intenta inicializar con los pesos `single_signer` descargados, por
lo que puedes comenzar a afinar el modelo directamente. Si el checkpoint vive en
una ruta distinta utiliza `--pretrained-checkpoint /ruta/al/archivo.pt`. Para
reiniciar desde parámetros aleatorios añade `--pretrained none`.

```bash
python -m slt \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --pose-dir data/single_signer/processed/pose \
  --keypoints-dir data/single_signer/processed/keypoints \
  --metadata-csv meta.csv \
  --train-index data/single_signer/index/train.csv \
  --val-index data/single_signer/index/val.csv \
  --gloss-csv data/single_signer/annotations/gloss.csv \
  --work-dir work_dirs/single_signer_demo \
  --epochs 2 --batch-size 2 --sequence-length 32 \
  --tokenizer hf-internal-testing/tiny-random-T5
```

El comando guarda `last.pt`, `best.pt` y `config.json` dentro de `work_dir`.

### Entrenamiento completo multi-stream

Para sesiones prolongadas utiliza `tools/train_slt_multistream_v9.py`, que
expone reanudación, *mixup* por stream, configuración vía YAML/JSON y
sobrescritura puntual con `--set`. La guía detallada está en
`docs/train_slt_multistream_v9.md`.

```bash
python tools/train_slt_multistream_v9.py \
  --config configs/single_signer.yml \
  --set data.face_dir=data/single_signer/processed/face \
  --set data.work_dir=work_dirs/single_signer_experiment \
  --set training.epochs=40 \
  --set optim.lr=5e-4
```

Los templates aceptan ahora `data.keypoints_dir` y `data.gloss_csv` para
propagar keypoints MediaPipe y secuencias de glosa/CTC al pipeline sin editar el
código base.

`config.json` dentro de `work_dir` refleja la configuración efectiva combinando
defaults, archivo y banderas.

#### Ajuste fino Phoenix→LSA-T limitado a 5 000 pasos

`tools/train_slt_multistream_v9.py` expone los argumentos
`--lr-encoder`, `--lr-decoder`, `--lr-mska`, `--max-train-steps` y
`--subset-size` para controlar tasas diferenciadas y truncar el ciclo de
entrenamiento tras un número fijo de iteraciones. El helper
`tools/fine_tune_phoenix_lsat.py` simplifica el flujo cargando `best.pth` de
Phoenix, aplicando el régimen de 5 000 pasos y reenviando cualquier override
adicional al script principal:

```bash
python tools/fine_tune_phoenix_lsat.py \
  --config configs/phoenix_lsat.yaml \
  --work-dir work_dirs/phoenix_lsat_ft \
  --phoenix-checkpoint work_dirs/phoenix/best.pth \
  --lr-encoder 5e-5 --lr-decoder 1e-4 \
  --subset-size 5000 --max-train-steps 5000 \
  -- --set data.face_dir=data/phoenix_2014/processed/face \
     --set data.hand_left_dir=data/phoenix_2014/processed/hand_l \
     --set data.hand_right_dir=data/phoenix_2014/processed/hand_r \
     --set data.pose_dir=data/phoenix_2014/processed/pose \
     --set data.metadata_csv=data/phoenix_2014/meta.csv \
     --set data.train_index=data/phoenix_2014/index/train.csv \
     --set data.val_index=data/phoenix_2014/index/val.csv
```

El comando crea tres grupos de parámetros: encoder (usa `--lr-encoder`), ramas
MSKA (heredan `--lr-mska` o, si no se define, la tasa del encoder) y decoder
(`--lr-decoder`). Con `--max-train-steps 5000` el bucle de entrenamiento se
detiene tras 5 000 iteraciones aunque `--epochs` sea mayor, y el planificador de
*learning rate* se ajusta automáticamente al límite efectivo.

#### Preset SignMusketeers (T5 v1.1 Base)

El archivo `configs/presets/decoder_signmusketeers_t5.yaml` habilita un flujo listo para
fine-tuning con `google/t5-v1_1-base`, ajustando `projector_dim=192` y `d_model=768`.
La representación concatenada de rostro, manos y pose se envía directamente al decoder.
【F:configs/presets/decoder_signmusketeers_t5.yaml†L1-L31】
Lánzalo directamente desde la CLI con:

```bash
python tools/train_slt_multistream_v9.py \
  --decoder-preset signmusketeers \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --pose-dir data/single_signer/processed/pose \
  --metadata-csv meta.csv \
  --train-index data/single_signer/index/train.csv \
  --val-index data/single_signer/index/val.csv \
  --work-dir work_dirs/signmusketeers_t5 \
  --batch-size 4 --sequence-length 128
```

El tokenizador se resuelve automáticamente al mismo checkpoint T5 y el preset aplica 30
épocas, `lr=5e-4` y `weight_decay=0.01`. Ajusta el tamaño de lote si tu GPU dispone de menos
de 22 GB para evitar *out-of-memory*. 【F:configs/presets/decoder_signmusketeers_t5.yaml†L22-L31】

### Traducción offline con MSKA (`--decoder-preset mska_paper_mbart`)

El preset `mska_paper_mbart` replica los 8 bloques atencionales con 6 cabezas descritos en el paper
MSKA-SLT y habilita un decoder `facebook/mbart-large-cc25` listo para uso sin conexión. El YAML
define variantes para alternar a `google/t5-v1_1-base` desde la CLI. Indica `--decoder-model t5` o
`--decoder-model mbart` y el script actualizará capas, cabezas, dropout, tokenizer y kwargs del
decoder en consecuencia.
【F:configs/presets/mska_paper_mbart.yaml†L1-L55】【F:docs/mska-paper-config.md†L1-L26】

```bash
python tools/train_slt_multistream_v9.py \
  --decoder-preset mska_paper_mbart \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --pose-dir data/single_signer/processed/pose \
  --keypoints-dir data/single_signer/processed/keypoints \
  --gloss-csv data/single_signer/gloss.csv \
  --metadata-csv meta.csv \
  --train-index data/single_signer/index/train.csv \
  --val-index data/single_signer/index/val.csv \
  --work-dir work_dirs/mska_mbart_offline \
  --batch-size 4 --sequence-length 128
```

El preset fija `lr=1e-5`, `weight_decay=1e-3` y `epochs=40`, preservando los pesos MSKA de
distilación (`mska_distillation_weight=1.0`) y CTC (`mska_ctc_weight=1.0`). Las banderas
`--mska-heads`, `--mska-stream-heads` y `--mska-temporal-blocks` vuelven a ajustar `ModelConfig`
incluso cuando el preset está activo, por lo que puedes escalar la atención sin editar el YAML.
Al combinarlo con `--decoder-model t5` la CLI aplica automáticamente la arquitectura de T5 Base y
sincroniza el tokenizer, manteniendo el resto de hiperparámetros MSKA intactos.

Para trabajar offline utiliza `--tokenizer-search-path` y `--decoder-search-path` para apuntar a
las carpetas cacheadas, y añade `--tokenizer-local-files-only --decoder-local-files-only` para
evitar solicitudes al Hub. El documento `docs/offline_cache.md` recopila ejemplos completos y el
flujo recomendado para preparar los artefactos con `huggingface-cli`.

Cuando trabajes con decoders T5 puedes activar *prompt tuning* y scheduled sampling
directamente desde la CLI: `--decoder-prompt-length`, `--decoder-prompt-init`,
`--decoder-prompt-text` y `--teacher-forcing-*` habilitan prompts aprendibles y el
decaimiento del ratio de teacher forcing documentados en
`docs/train_slt_multistream_v9.md`. En nuestras pruebas internos un prompt de 16 tokens
inicializado con texto y un schedule `ratio=1.0 -> 0.4` (`decay=0.92`) redujeron el CER de
validación en ~1 punto tras 20 épocas, manteniendo la estabilidad numérica en GPUs de 24 GB.

### Entrenamiento basado en keypoints

Activa la rama MSKA con `tools/train_slt_multistream_v9.py --use-mska`,
proporcionando `data.keypoints_dir` y `data.gloss_csv` en la configuración.
Esto habilita la combinación de pérdidas (traducción, CTC y distilación) y la
carga de checkpoints individuales para el encoder y las cabezas auxiliares.
El script `tools/train_slt_lsa_mska_v13.py` queda como *wrapper* retrocompatible
que reenvía los parámetros al flujo unificado.

```bash
python tools/train_slt_multistream_v9.py \
  --use-mska \
  --keypoints-dir data/lsa_keypoints \
  --gloss-csv data/lsa_gloss.csv \
  --work-dir work_dirs/lsa_mska \
  --mska-ctc-weight 0.5 --mska-distillation-weight 0.2
```

Ajusta la atención por articulación y los bloques temporales con
`--mska-stream-heads`, `--mska-temporal-blocks`, `--mska-temporal-kernel` y
`--mska-temporal-dilation`, tal
como se describe en la Sección 3.2.2 del paper y en
`docs/train_slt_multistream_v9.md`.

El refinamiento global descrito en la Sección 3.2.4 se activa con
`--mska-use-sgr`. Controla su aportación con `--mska-sgr-mix`, selecciona la
activación mediante `--mska-sgr-activation` y decide si la matriz se comparte
entre streams (`--mska-sgr-shared`) o se aprende por flujo (`--mska-sgr-per-stream`).

Revisa `tools/train_slt_multistream_v9.py --help` y
`docs/train_slt_multistream_v9.md` para detalles de cada argumento.

### Evaluación y reportes

Evalúa uno o varios checkpoints y genera predicciones, métricas y reportes. Para
usar el preset validado sin un checkpoint propio pasa `--checkpoint single_signer`
y, si el archivo no vive en `data/single_signer/`, añade
`--pretrained-checkpoint /ruta/al/archivo.pt`:

```bash
python tools/eval_slt_multistream_v9.py \
  --checkpoint work_dirs/single_signer_demo/best.pt \
  --tokenizer hf-internal-testing/tiny-random-T5 \
  --face-dir data/single_signer/processed/face \
  --hand-left-dir data/single_signer/processed/hand_l \
  --hand-right-dir data/single_signer/processed/hand_r \
  --pose-dir data/single_signer/processed/pose \
  --metadata-csv meta.csv \
  --eval-index data/single_signer/index/test.csv \
  --output-csv work_dirs/single_signer_demo/predictions/preds.csv
```

Cuando el checkpoint activa MSKA puedes ajustar `--ctc-num-beams` (ancho del beam
CTC) y añadir `--report-gloss-wer` para incluir CER/WER de glosas en los reportes.
Los CSV incorporan columnas `gloss_prediction` y `gloss_reference` y las
secciones de métricas agregadas añaden `gloss_cer` y `gloss_wer`.

El script valida el tokenizador con `slt.utils.validate_tokenizer` antes de
procesar los videos, evitando ejecuciones largas con configuraciones inválidas.
Los reportes incluyen métricas BLEU, ChrF, CER y WER; estas últimas se calculan
mediante `slt.utils.character_error_rate` y `slt.utils.word_error_rate` para
facilitar comparaciones con referencias externas.

Los archivos `report.json` y `report.csv` resultantes pueden integrarse en
herramientas analíticas mediante `docs/metrics_dashboard_integration.py`.

## Exportación y demos en tiempo real

Convierte el encoder a ONNX o TorchScript para ejecutarlo fuera de PyTorch. El
ejemplo siguiente exporta el preset `single_signer`; añade
`--pretrained-checkpoint /ruta/al/archivo.pt` si el checkpoint descargado no se
encuentra en `data/single_signer/`:

```bash
python tools/export_onnx_encoder_v9.py \
  --checkpoint single_signer \
  --onnx exports/single_signer_encoder.onnx \
  --torchscript exports/single_signer_encoder.ts \
  --sequence-length 64
```

Valida los artefactos con `tools/demo_realtime_multistream.py` (webcam) o
`tools/test_realtime_pipeline.py` (video en disco). Ambos requieren un modelo
exportado (`--model`) o declarar `--model-format stub`, además de un tokenizador
de HuggingFace (`--tokenizer`).

## Preentrenamiento de backbones

`tools/pretrain_dino_face.py` y `tools/pretrain_dino_hands.py` permiten obtener
pesos auto-supervisados compatibles con `load_dinov2_backbone`. Soportan
entrenamientos declarativos vía `--config`, exportación de backbones (`--export-backbone`)
y registro de métricas en `metrics.jsonl`. Sigue la guía en `docs/pretraining.md`
para elegir hiperparámetros, controlar augmentations y conectar los pesos
resultantes con `tools/train_slt_multistream_v9.py`. La misma guía detalla cómo
activar el regularizador KoLeo mediante `--koleo-weight` y ajustar la estabilidad
numérica con `--koleo-epsilon`.

La configuración por defecto genera dos crops globales y ocho locales por imagen.
Modifícalos con `--global-crops` y `--num-local-crops` según la resolución
disponible. Usa `--pseudo-epochs` para repetir el DataLoader dentro de cada época
—el planificador de *learning rate* se estira automáticamente— y activa la
normalización Sinkhorn de estilo DINOv2 con `--use-sinkhorn`. Ajusta los
parámetros `--sinkhorn-eps` y `--sinkhorn-iters` si necesitas mayor estabilidad.

## Control de calidad y pruebas

Los tests automatizados validan piezas clave del pipeline:

| Escenario | Prueba | Resultado |
|-----------|--------|-----------|
| Contrato datos | `python tools/ci_validate_data_contract.py` | Replica estructura base. |
| E2E sintético | `tests/test_pipeline_end_to_end.py` | Pérdida cae y exporta. |
| CLI demo | `tests/test_cli_main.py` | Entrenamiento corto sin errores. |
| Exportación | `tests/test_export.py` | ONNX y TorchScript válidos. |
| Calidad datos | `tests/data/test_dataset_quality.py` | Detecta frames faltantes y FPS. |
| Métricas regresión | `python tools/ci_validate_metrics.py` | Pérdidas dentro de tolerancias. |

Ejecuta `pytest`, `ruff check .`, `black --check .` y `mypy` antes de subir
cambios. `docs/operational_checklist.md` resume la secuencia recomendada de
verificaciones previas a una release.

## Estructura de carpetas

```text
slt/                 # Paquete instalable con encoder, dataset y utilidades
  data/
  models/
  training/
  runtime/
  utils/
tools/               # Scripts CLI para extracción, entrenamiento, evaluación, demos
docs/                # Guías de datos, operación, preentrenamiento y métricas
tests/               # Suite de smoke tests y fixtures sintéticas
meta.csv             # CSV de subtítulos de `single_signer`
```

## Soporte y aportes

Abre *issues* o PRs describiendo claramente el componente afectado (datos,
entrenamiento, exportación, demos). Acompaña los cambios con actualizaciones en
la documentación cuando modifiques rutas, argumentos o artefactos generados.
